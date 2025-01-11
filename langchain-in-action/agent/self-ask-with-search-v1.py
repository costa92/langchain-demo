import os
from langchain_openai import OpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import Tool
from langchain import hub
from langchain.agents import create_self_ask_with_search_agent,AgentExecutor
# 初始化语言模型
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 OPENAI_API_KEY")

model_name = "Qwen/Qwen2.5-7B-Instruct"
base_url = "https://api.siliconflow.cn/v1"

# model_name="deepseek-chat"
# API_KEY = os.getenv("deepseek_api_key")
# # base_url = os.getenv("deepseek_api_url")
# base_url="https://api.deepseek.com/beta"


llm = OpenAI(
    api_key=API_KEY,
    base_url=base_url,
    model_name=model_name,
)



search = SerpAPIWrapper()

tools = [
    Tool(
        name="Intermediate Answer", 
        func=search.run,
        description="useful for when you need to ask with search", 
    )
]


# 获取使用提示 可以修改此提示
prompt = hub.pull("hwchase17/self-ask-with-search")

# 使用搜索代理构建自助询问
agent = create_self_ask_with_search_agent(llm, tools, prompt)


agent_executor = AgentExecutor(
  agent=agent, 
  tools=tools, 
  handle_parsing_errors=True,
  verbose=True,
  allow_dangerous_code=True,
  return_intermediate_steps=True,
  early_stopping_method="force",  # 保持为 "force"
  max_iterations=5,  # 增加迭代次数
  max_execution_time=30,  # 增加每个工具的执行时间
  execution_timeout=120,  # 增加总的最大时间限制 
)

res = agent_executor.invoke({"input":"使用玫瑰作为国花的国家的首都是哪里?"})

print(res["output"])

