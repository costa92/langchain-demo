

# 配置日志输出
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import os
from langchain_openai import ChatOpenAI
from langchain import hub
import langchain
# langchain.debug = True
langchain.verbose = True
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

# 加载所需的库
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import AgentExecutor, create_react_agent

# 设置工具
tools = load_tools(["serpapi", "llm-math"], llm=llm)


prompt = hub.pull("llm-react/react")
# 初始化Agent
react_agent = create_react_agent(
    llm,
    tools,
    prompt,
)

# 初始化AgentExecutor
agent_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True)

# 跑起来
res=agent_executor.invoke({"input":"目前市场上玫瑰花的平均价格是多少？如果我在此基础上加价15%卖出，应该如何定价？"})
print(res)