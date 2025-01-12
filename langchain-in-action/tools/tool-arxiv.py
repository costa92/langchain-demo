import os
from langchain_community.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

# 初始化语言模型
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 OPENAI_API_KEY")

model_name = "Qwen/Qwen2.5-7B-Instruct"
base_url = "https://api.siliconflow.cn/v1"

# 创建语言模型实例
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=base_url,
    model_name=model_name,
)

# 加载arxiv工具
tools = load_tools(["arxiv"])

# 从Hub中拉取提示模板
prompt = hub.pull("hwchase17/react")

# 创建反应式代理
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

# 创建代理执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    prompt=prompt,
)

# 执行代理，查询特定论文的创新点
res = agent_executor.invoke({"input": "介绍一下2005.14165这篇论文的创新点?"})
print(res["output"])
