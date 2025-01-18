import os


from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType

# 连接到FlowerShop数据库
db = SQLDatabase.from_uri("sqlite:///FlowerShop.db")
# 初始化语言模型
model_name = "deepseek-chat"
API_KEY = os.getenv("deepseek_api_key")
base_url = "https://api.deepseek.com/beta"

# 创建 LLM 实例
llm = ChatOpenAI(api_key=API_KEY, base_url=base_url, model_name=model_name, temperature=0)
# 创建SQL Agent
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    # verbose=True,
    verbose=False,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

# 使用Agent执行SQL查询
questions = [
    "哪种鲜花的存货数量最少？",
    "平均销售价格是多少？",
]

for question in questions:
    response = agent_executor.invoke({"input": question})
    print(response)