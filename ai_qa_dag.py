"""
AI 智能问答 DAG - 完整版（含 SQL 执行）
功能: 知识库查询 + 智能模板匹配 + SQL 生成 + 实际执行 + 数据分析 + 结果存储
更新: 增加 SQL 执行步骤，实现真实数据分析
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import requests
import json
import sys
import re
from datetime import datetime, date
import pandas as pd  # <--- [修改点 1] 导入 pandas 库

# 添加配置路径
sys.path.insert(0, '/opt/airflow/scripts')
sys.path.insert(0, 'E:/DATA/tools')  # 本地开发路径

from dag_config import KB_API_URL, DEEPSEEK_API_KEY, POSTGRES_CONN_ID, DEFAULT_N_RESULTS
from llm_utils import analyze_with_llm, check_db_cache, format_answer, extract_sql
from prompts import get_template

# DAG 默认参数
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

def query_knowledge_base(**context):
    """查询知识库获取相关信息"""
    print("=" * 60)
    print("步骤 1: 查询知识库")
    print("=" * 60)
    
    # 获取问题参数
    question = context['dag_run'].conf.get('question', context['params'].get('question', ''))
    
    if not question:
        raise ValueError("未提供问题参数！请在触发 DAG 时输入 question")
    
    print(f"问题: {question}\n")
    
    # 先检查数据库缓存
    pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    cached_result = check_db_cache(question, pg_hook)
    
    if cached_result:
        print("✅ 使用缓存结果，跳过 LLM 调用")
        context['ti'].xcom_push(key='use_cache', value=True)
        context['ti'].xcom_push(key='cached_answer', value=cached_result['answer'])
        context['ti'].xcom_push(key='question', value=question)
        return cached_result
    
    try:
        url = f"{KB_API_URL}/query"
        payload = {
            "query": question,
            "n_results": DEFAULT_N_RESULTS
        }
        
        print(f"请求知识库 API: {url}")
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            kb_results = response.json()
            print(f"✅ 找到 {len(kb_results)} 条相关知识\n")
            
            for i, result in enumerate(kb_results, 1):
                print(f"知识 {i}:")
                print(f"  来源: {result['collection']}")
                print(f"  相似度: {result['score']:.4f}")
                print(f"  内容: {result['text'][:150]}...")
                print()
            
            # 推送到 XCom
            context['ti'].xcom_push(key='kb_results', value=kb_results)
            context['ti'].xcom_push(key='question', value=question)
            context['ti'].xcom_push(key='use_cache', value=False)
            
            return kb_results
        else:
            print(f"❌ 知识库查询失败: {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ 查询知识库时出错: {str(e)}")
        raise

def generate_sql_with_ai(**context):
    """使用 AI 生成 SQL 查询"""
    print("=" * 60)
    print("步骤 2: AI 生成 SQL")
    print("=" * 60)
    
    # 获取上游数据
    ti = context['ti']
    use_cache = ti.xcom_pull(task_ids='query_kb', key='use_cache')
    
    # 如果使用缓存，直接跳过
    if use_cache:
        print("⏭️  使用缓存，跳过 SQL 生成")
        return None
    
    question = ti.xcom_pull(task_ids='query_kb', key='question')
    kb_results = ti.xcom_pull(task_ids='query_kb', key='kb_results')
    analysis_type = context['dag_run'].conf.get('analysis_type', context['params'].get('analysis_type'))
    current_date = datetime.now().strftime('%Y-%m-%d')
    context_question = f"请注意，**当前日期是 {current_date}**。所有相对时间（如'本月'、'最近7天'）均基于此日期计算。用户问题：{question}"
    
    # 调用 LLM 生成 SQL
    result = analyze_with_llm(
        question=context_question,
        kb_results=kb_results or [],
        analysis_type=analysis_type
    )
    
    if not result['success']:
        error_msg = f"AI 生成 SQL 失败: {result.get('error', '未知错误')}"
        print(f"❌ {error_msg}")
        raise Exception(error_msg)
    
    sql_code = result.get('sql')
    
    if not sql_code:
        print("⚠️  未生成 SQL，可能是通用查询类型")
        ti.xcom_push(key='sql_code', value=None)
        ti.xcom_push(key='initial_analysis', value=result['answer'])
        ti.xcom_push(key='template_used', value=result['template_used'])
        ti.xcom_push(key='tokens_used', value=result.get('tokens', 0))
        return None
    
    print("✅ SQL 生成成功")
    print(f"使用模板: {result['template_used']}")
    print(f"Token 消耗: {result.get('tokens', 0)}")
    print(f"\n生成的 SQL:\n{sql_code}\n")
    
    # 推送到 XCom
    ti.xcom_push(key='sql_code', value=sql_code)
    ti.xcom_push(key='initial_analysis', value=result['answer'])
    ti.xcom_push(key='template_used', value=result['template_used'])
    ti.xcom_push(key='tokens_used', value=result.get('tokens', 0))
    
    return sql_code

def execute_sql_query(**context):
    """执行生成的 SQL 查询"""
    print("=" * 60)
    print("步骤 3: 执行 SQL 查询")
    print("=" * 60)
    
    ti = context['ti']
    use_cache = ti.xcom_pull(task_ids='query_kb', key='use_cache')
    
    if use_cache:
        print("⏭️  使用缓存，跳过 SQL 执行")
        return None
    
    sql_code = ti.xcom_pull(task_ids='generate_sql', key='sql_code')
    
    if not sql_code:
        print("⚠️  无 SQL 需要执行")
        return None
    
    try:
        # 连接数据库
        pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        
        # 执行 SQL（支持多条语句）
        sql_statements = sql_code.strip().split(';')
        all_results = []
        
        for i, sql in enumerate(sql_statements, 1):
            sql = sql.strip()
            if not sql:
                continue
                
            print(f"\n执行第 {i} 条 SQL:")
            print(f"{sql[:200]}..." if len(sql) > 200 else sql)
            
            # 执行查询
            result = pg_hook.get_pandas_df(sql)
            
            if not result.empty:
                print(f"✅ 返回 {len(result)} 行数据")
                all_results.append({
                    'sql': sql,
                    'data': result.to_dict('records'),
                    'columns': result.columns.tolist(),
                    'row_count': len(result)
                })
            else:
                print("⚠️  查询无结果")
        
        if not all_results:
            print("❌ 所有查询均无结果")
            ti.xcom_push(key='query_results', value=None)
            # ⚠️ 【新增】如果无结果，也推送空列表，防止 extract_comparison_metrics 报错
            ti.xcom_push(key='full_sql_data_list', value=[])
            return None
        
        # ⚠️ 【关键修复 1】: 优化 XCom 传输，将完整的核心数据列表单独推送到一个 Key
        # 假设第一个查询结果是我们需要的核心数据
        core_query_data = all_results[0]['data'] if all_results and all_results[0]['data'] else []
        ti.xcom_push(key='full_sql_data_list', value=core_query_data)
        
        # 保留原有的 key 'query_results'，用于兼容 analyze_query_results 任务
        ti.xcom_push(key='query_results', value=all_results)
        
        print(f"\n✅ 共执行 {len(all_results)} 条有效 SQL。完整 {len(core_query_data)} 行数据已推送至 XCom (full_sql_data_list key)。")
        return all_results
        
    except Exception as e:
        error_msg = f"SQL 执行失败: {str(e)}"
        print(f"❌ {error_msg}")
        ti.xcom_push(key='sql_error', value=error_msg)
        # 不抛出异常，继续流程
        return None


def analyze_query_results(**context):
    print("=" * 60)
    print("步骤 4: 基于真实数据分析")
    print("=" * 60)
    
    ti = context['ti']
    use_cache = ti.xcom_pull(task_ids='query_kb', key='use_cache')  # ⬅️ 移到这里
    
    # 如果使用缓存，直接返回缓存答案
    if use_cache:
        cached_answer = ti.xcom_pull(task_ids='query_kb', key='cached_answer')
        print("✅ 使用缓存答案")
        ti.xcom_push(key='final_answer', value=cached_answer)
        ti.xcom_push(key='from_cache', value=True)
        return cached_answer
    
    question = ti.xcom_pull(task_ids='query_kb', key='question')
    kb_results = ti.xcom_pull(task_ids='query_kb', key='kb_results')
    query_results = ti.xcom_pull(task_ids='execute_sql', key='query_results')
    initial_analysis = ti.xcom_pull(task_ids='generate_sql', key='initial_analysis')
    sql_error = ti.xcom_pull(task_ids='execute_sql', key='sql_error')
    
    # 🔍 添加调试
    print(f"📊 query_results 类型: {type(query_results)}")
    print(f"📊 query_results 长度: {len(query_results) if query_results else 0}")
    if query_results:
        print(f"📊 第一个结果 row_count: {query_results[0].get('row_count')}")
        print(f"📊 第一个结果 data 长度: {len(query_results[0].get('data', []))}")
    
    # 如果没有查询结果
    if not query_results or (isinstance(query_results, list) and len(query_results) == 0):
        return "❌ SQL 查询无结果，无法进行数据分析。请检查：\n1. 时间范围是否有数据\n2. 筛选条件是否过严\n3. 店铺名称是否正确"
    
    # 如果 SQL 执行失败
    if sql_error:
        error_response = f"SQL 执行出错：{sql_error}\n\n初步分析：\n{initial_analysis}"
        ti.xcom_push(key='final_answer', value=error_response)
        ti.xcom_push(key='from_cache', value=False)
        return error_response
    

    
    # 构建包含真实数据的 Prompt
    from datetime import date, datetime

    def json_serial(obj):
        """JSON 序列化日期类型"""
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")
    
    # 构建数据摘要（智能处理大数据集）
    data_summary_parts = []
    for i, r in enumerate(query_results, 1):
        if r['row_count'] > 30:  # 大数据集做预处理
            print("📊 进入大数据集预处理分支")  # 添加这行
            df = pd.DataFrame(r['data'])
            
            # 月度汇总
            monthly_agg = df.groupby(['year', 'month_number']).agg({
                '广告消耗': 'sum',
                '询盘量': 'sum',
                '有效线索量': 'sum',
                '广告点击': 'sum'
            }).round(2).reset_index().to_dict('records')
            # 调试信息（紧跟在上面这行后面）
            print("=" * 60)
            print("🔍 调试信息：月度汇总结果")
            print("=" * 60)
            print(f"DataFrame 总行数: {len(df)}")
            print(f"DataFrame 列名: {df.columns.tolist()}")
            print(f"唯一月份: {sorted(df['month_number'].unique())}")
            print(f"\n月度汇总内容:")
            print(json.dumps(monthly_agg, ensure_ascii=False, indent=2))
            print("=" * 60)            

            
            # 取每月前5条明细样本
            sample_data = []
            for month in df['month_number'].unique():
                sample_data.extend(df[df['month_number'] == month].head(5).to_dict('records'))
            
            summary = f"""**查询 {i}：** 共{r['row_count']}行

    **月度汇总：**
    {json.dumps(monthly_agg, ensure_ascii=False, indent=2)}

    **明细样本（每月前5条）：**
    {json.dumps(sample_data[:15], ensure_ascii=False, indent=2, default=json_serial)}
    """
        else:  # 小数据集直接传
            summary = f"**查询 {i}：** {r['row_count']}行\n{json.dumps(r['data'], ensure_ascii=False, indent=2, default=json_serial)}"
        
        data_summary_parts.append(summary)

    data_summary = "\n\n".join(data_summary_parts)
    
    # <--- [修改点 3b] 注入对比摘要到 Prompt 最前端 --->
    final_prompt = f"""

基于以下真实查询结果，对用户问题进行深度分析。

用户问题：{question}

查询结果：
{data_summary}

严格要求：
1. **所有数值必须来自查询结果** - 禁止编造数据、使用占位符(如"XX%"、"约X个")
2. **数据缺失必须明确说明** - 如"11月无数据，仅分析10月和12月"
3. **结论必须有数据支撑** - 
   - ✅ "询盘成本从18.5元降至12.3元，下降33.5%"
   - ❌ "询盘成本表现良好"、"转化率有所提升"
4. **洞察必须可执行** - 
   - ✅ "淘宝/天猫人群询盘成本18.52元，高于平均成本15元，建议降低该人群预算30%"
   - ✅ "L2人群询盘成本4.75元，是L0人群(58.46元)的1/12，建议将L2预算增加至当前2倍"
   - ❌ "建议优化人群定向"、"持续关注数据"、"降低溢价至XX%"（无对比依据）
5. **允许使用分析框架** - 但必须用真实数据填充(如成本结构分析、漏斗转化分析)
6. **输出格式自适应问题类型** - 不强制表格，可用文字/列表/对比，以最清晰表达为准

【不输出】：SQL代码、中间计算过程
【必须输出】：关键数值、变化幅度、异常原因、具体行动方案

输出格式：

【自主选择最佳输出格式】：
- 趋势分析 → 时间序列描述 + 拐点标注
- 人群对比 → 矩阵排序 + TOP3 洞察
- 异常诊断 → 问题定位 + 根因推断
- 决策支持 → 方案对比表 + 风险评估

【必须包含】：
1. 数据支撑（具体数值）
2. 业务解读（为什么）
3. 行动建议（怎么做 + 优先级）

【禁止】：
- 空洞结论（"表现良好"需量化标准）
- 无效建议（"持续关注"不是建议）
"""
    
    # 调用 LLM 进行最终分析
    from openai import OpenAI
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个专业的数据分析师，擅长从数据中提炼洞察并给出建议"},
                {"role": "user", "content": final_prompt}
            ],
            max_tokens=3000,
            temperature=0.3
        )
        
        final_answer = response.choices[0].message.content
        final_tokens = response.usage.total_tokens
        
        print("✅ 最终分析完成")
        print(f"Token 消耗: {final_tokens}")
        print("=" * 60)
        print("最终分析结果:")
        print("=" * 60)
        print(final_answer[:500] + "..." if len(final_answer) > 500 else final_answer)
        
        # 推送结果
        ti.xcom_push(key='final_answer', value=final_answer)
        ti.xcom_push(key='final_tokens', value=final_tokens)
        ti.xcom_push(key='from_cache', value=False)
        
        return final_answer
        
    except Exception as e:
        error_msg = f"最终分析失败: {str(e)}"
        print(f"❌ {error_msg}")
        # 降级：返回初始分析
        ti.xcom_push(key='final_answer', value=initial_analysis)
        return initial_analysis

def save_to_database(**context):
    """保存分析结果到数据库"""
    print("=" * 60)
    print("步骤 5: 保存结果到数据库")
    print("=" * 60)
    
    # 获取所有数据
    ti = context['ti']
    question = ti.xcom_pull(task_ids='query_kb', key='question')
    kb_results = ti.xcom_pull(task_ids='query_kb', key='kb_results')
    sql_code = ti.xcom_pull(task_ids='generate_sql', key='sql_code')
    final_answer = ti.xcom_pull(task_ids='analyze_results', key='final_answer')
    template_used = ti.xcom_pull(task_ids='generate_sql', key='template_used')
    tokens_used = ti.xcom_pull(task_ids='generate_sql', key='tokens_used')
    final_tokens = ti.xcom_pull(task_ids='analyze_results', key='final_tokens')
    from_cache = ti.xcom_pull(task_ids='analyze_results', key='from_cache')
    
    # 如果是缓存结果，不重复保存
    if from_cache:
        print("⏭️  缓存结果，跳过保存")
        return
    
    try:
        # 连接数据库
        pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        
        # 准备数据
        kb_results_json = json.dumps(kb_results, ensure_ascii=False) if kb_results else None
        total_tokens = (tokens_used or 0) + (final_tokens or 0)
        
        # 插入 SQL（添加 schema 前缀）
        insert_sql = """
        INSERT INTO data_1688.ai_analysis_results 
        (question, kb_results, ai_answer, sql_code, template_used, tokens_used, model_name)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        
        result = pg_hook.get_first(
            insert_sql,
            parameters=(
                question,
                kb_results_json,
                final_answer,
                sql_code,
                template_used,
                total_tokens,
                'deepseek-chat'
            )
        )
        
        result_id = result[0] if result else None
        
        print(f"✅ 结果已保存到数据库")
        print(f"记录 ID: {result_id}")
        print(f"问题: {question}")
        print(f"模板: {template_used}")
        print(f"总 Token 消耗: {total_tokens}")
        
        return result_id
        
    except Exception as e:
        print(f"❌ 保存到数据库失败: {str(e)}")
        raise

def task_summary(**context):
    """任务总结"""
    print("\n" + "=" * 60)
    print("任务执行总结")
    print("=" * 60)
    
    ti = context['ti']
    question = ti.xcom_pull(task_ids='query_kb', key='question')
    kb_results = ti.xcom_pull(task_ids='query_kb', key='kb_results')
    sql_code = ti.xcom_pull(task_ids='generate_sql', key='sql_code')
    query_results = ti.xcom_pull(task_ids='execute_sql', key='query_results')
    template_used = ti.xcom_pull(task_ids='generate_sql', key='template_used')
    tokens_used = ti.xcom_pull(task_ids='generate_sql', key='tokens_used')
    final_tokens = ti.xcom_pull(task_ids='analyze_results', key='final_tokens')
    from_cache = ti.xcom_pull(task_ids='analyze_results', key='from_cache')
    
    # ⚠️ 从新的 key 中检查实际传输的数据行数，而非 summary
    actual_rows = len(ti.xcom_pull(task_ids='execute_sql', key='full_sql_data_list') or [])
    
    print(f"✅ 问题: {question}")
    print(f"✅ 使用模板: {template_used}")
    print(f"✅ 知识库匹配: {len(kb_results) if kb_results else 0} 条")
    print(f"✅ SQL 生成: {'是' if sql_code else '否'}")
    print(f"✅ SQL 执行: {'成功' if query_results else '跳过/失败'}")
    print(f"✅ 查询数据量: {actual_rows} 行 (来自 full_sql_data_list key)") # 引用新的数据行数
    print(f"✅ Token 消耗: {(tokens_used or 0) + (final_tokens or 0)}")
    print(f"✅ 结果来源: {'缓存' if from_cache else '实时分析'}")
    
    print("\n" + "=" * 60)
    print("🎉 AI 智能问答流程执行完毕")
    print("=" * 60)
    
    print("\n查看完整结果:")
    print("1. Airflow 日志: 查看本任务日志")
    print("2. 数据库: SELECT * FROM data_1688.ai_analysis_results ORDER BY id DESC LIMIT 1;")
    print("3. 知识库 UI: http://localhost:8501 → AI 分析历史")

# 定义 DAG
with DAG(
    dag_id='ai_qa_with_knowledge_base_v4.1',
    default_args=default_args,
    description='AI智能问答 v4: SQL生成 + 实际执行 + 真实数据分析',
    doc_md="""
    ## AI 智能问答系统 v3 🆕
    
    ### 核心升级:
    - ✅ **真实数据分析**：不再空想，基于实际查询结果分析
    - ✅ **SQL 自动执行**：生成 SQL 后立即查询数据库
    - ✅ **二次深度分析**：基于查询结果进行精准洞察
    - ✅ **完整数据追踪**：记录 SQL、结果、Token 消耗
    
    ### 工作流程:
    1. 查询知识库 → 获取表结构
    2. AI 生成 SQL → 基于业务场景
    3. **执行 SQL** → 获取真实数据
    4. **AI 深度分析** → 基于实际结果
    5. 保存完整记录 → 包含数据和洞察
    
    ### 使用方法:
    ```json
    {
      "question": "分析 YG 店铺本月销售情况",
      "analysis_type": "sales_analysis"
    }
    ```
    
    ### 分析类型:
    - `sales_analysis` - 销售数据分析
    - `product_comparison` - 商品对比分析
    - `trend_forecast` - 趋势预测分析
    - `competitor_analysis` - 竞品对标分析
    - `ad_performance` - 广告效果分析
    - `customer_profile` - 客户画像分析
    """,
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['ai', 'v3', 'sql执行', '真实数据', '深度分析'],
    params={
        "question": "分析 YG 店铺本月销售情况",
        "analysis_type": None
    }
) as dag:
    
    # 任务 1: 查询知识库
    task_query_kb = PythonOperator(
        task_id='query_kb',
        python_callable=query_knowledge_base,
    )
    
    # 任务 2: 生成 SQL
    task_generate_sql = PythonOperator(
        task_id='generate_sql',
        python_callable=generate_sql_with_ai,
    )
    
    # 任务 3: 执行 SQL
    task_execute_sql = PythonOperator(
        task_id='execute_sql',
        python_callable=execute_sql_query,
    )
    

    
    # 任务 4: 基于结果分析
    task_analyze = PythonOperator(
        task_id='analyze_results',
        python_callable=analyze_query_results,
    )
    
    # 任务 5: 保存到数据库
    task_save = PythonOperator(
        task_id='save_to_db',
        python_callable=save_to_database,
    )
    
    # 任务 6: 总结
    task_summary_task = PythonOperator(
        task_id='summary',
        python_callable=task_summary,
    )
    
    # 定义任务依赖

    task_query_kb >> task_generate_sql >> task_execute_sql >> task_analyze >> task_save >> task_summary_task