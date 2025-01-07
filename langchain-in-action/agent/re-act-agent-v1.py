
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
import os

# 初始化语言模型
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 OPENAI_API_KEY")

model_name = "Qwen/Qwen2.5-7B-Instruct"
base_url = "https://api.siliconflow.cn/v1"

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=base_url,
    model_name=model_name,
)

# 设置工具
tools = load_tools(["serpapi", "llm-math"], llm=llm)

prompt = hub.pull("llm-react/react")
# 初始化Agent:使用工具、语言模型和代理类型来初始化代理
# create_react_agent 参数： 
# llm:语言模型
# tools:工具
# prompt:代理的提示
# 参数不能写错
agent = create_react_agent(
    llm, 
  tools, 
  prompt,
  
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False,handle_parsing_errors=True)


# 跑起来
res = agent_executor.invoke({"input": "目前市场上玫瑰花的平均价格是多少？如果我在此基础上加价15%卖出，应该如何定价？"})

print(res)