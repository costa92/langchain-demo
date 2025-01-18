import os
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import re

# 设置 OpenAI API Key
os.environ["OPENAI_API_KEY"] = 'your_actual_openai_api_key'

# 连接到数据库
db = SQLDatabase.from_uri("sqlite:///FlowerShop.db")

# 初始化语言模型
model_name = "deepseek-chat"
API_KEY = os.getenv("deepseek_api_key")
base_url = "https://api.deepseek.com/beta"

# 创建 LLM 实例
llm = ChatOpenAI(api_key=API_KEY, base_url=base_url, model_name=model_name, temperature=0)

# 创建自定义提示模板
sql_template = """Given an input question, create a syntactically correct SQLite query to run.
Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"

Only return the SQL query and nothing else. Do not include any markdown syntax like ```sql or ```.

Question: {question}
SQLQuery:"""

sql_prompt = PromptTemplate(
    input_variables=["question"],
    template=sql_template
)

# 清理 SQL 查询中的 Markdown 标记
def clean_sql_query(query):
    # 移除多余的 Markdown 标记
    query = re.sub(r'```sql', '', query)
    query = re.sub(r'```', '', query)
    return query.strip()

# 创建 SQL 查询链
def get_schema(_):
    return db.get_table_info()

def run_query(query):
    # 清理查询语句
    cleaned_query = clean_sql_query(query)
    return db.run(cleaned_query)

# 构建查询链
def construct_query_chain():
    return (
        RunnablePassthrough.assign(
            query=sql_prompt | llm | StrOutputParser()
        ) 
        | (lambda x: x['query']) 
        | run_query
    )

# 构建完整的查询链
query_chain = construct_query_chain()

# 定义查询问题
queries = [
    "有多少种不同的鲜花？",
    "哪种鲜花的存货数量最少？",
    "平均销售价格是多少？",
    "从法国进口的鲜花有多少种？",
    "哪种鲜花的销售量最高？"
]

# 执行查询
for query in queries:
    print(f"\n查询: {query}")
    try:
        # 执行查询并获取结果
        result = query_chain.invoke({"question": query})
        print(f"结果: {result}")
    except Exception as e:
        print(f"查询出错: {e}")