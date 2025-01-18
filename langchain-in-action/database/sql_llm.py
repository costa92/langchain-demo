from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
import os
import re

# 初始化语言模型
model_name = "deepseek-chat"
API_KEY = os.getenv("deepseek_api_key")
base_url = "https://api.deepseek.com/beta"

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=base_url,
    model_name=model_name,
    temperature=0.5,
)

# 创建数据库连接
db = SQLDatabase.from_uri("sqlite:///FlowerShop.db")

# 创建SQL查询链
db_chain = create_sql_query_chain(llm, db)

def extract_sql_query(query_text):
    """
    从生成的文本中提取 SQL 查询
    使用正则表达式提取从 SELECT 开始到查询结束的内容
    """
    sql_pattern = r'SELECT.*?;'
    match = re.search(sql_pattern, query_text, re.DOTALL | re.IGNORECASE)
    
    if not match:
        sql_pattern = r'SELECT.*'  # 匹配没有分号的查询
        match = re.search(sql_pattern, query_text, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(0).strip()
    
    print("无法提取 SQL 查询。原始内容:")
    print(query_text)
    return None


def execute_query(question):
    """
    执行 SQL 查询并获取结果，处理查询链和错误
    """
    # 生成 SQL 查询
    query = db_chain.invoke({"question": question})
    
    # 打印原始查询内容（调试用）
    print(f"原始查询内容 (对于问题 '{question}'):", query)
    
    # 提取 SQL 查询
    sql_query = extract_sql_query(query)
    
    if not sql_query:
        print(f"无法为问题 '{question}' 生成有效的 SQL 查询")
        return None
    
    print("提取的 SQL 查询:", sql_query)
    
    # 执行查询并获取结果
    try:
        result = db.run(sql_query)
        return result
    except Exception as e:
        print(f"查询执行错误: {e}")
        print(f"出错的查询: {sql_query}")
        return None


# 调用示例
if __name__ == "__main__":
    # 获取不同鲜花的数量
    print("不同鲜花的数量:")
    result = execute_query("有多少种不同的鲜花？")
    if result:
        print(result)

    # 存货数量最少的鲜花
    print("存货数量最少的鲜花:")
    result = execute_query("哪种鲜花的存货数量最少？")
    if result:
        print(result)

    # 销售数量最多的鲜花
    print("销售数量最多的鲜花:")
    result = execute_query("哪种鲜花的销售数量最多？，输出结果为：鲜花名称，销售数量")
    if result:
        print(result)
