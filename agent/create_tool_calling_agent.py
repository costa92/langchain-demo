import re
from langchain import hub 
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain.agents import AgentExecutor, create_openai_tools_agent,create_tool_calling_agent
from langchain_openai import ChatOpenAI
import os


# 使用 TavilySearchResults 工具
tools = [TavilySearchResults(max_results=1)]

# 获取使用的提示
prompt = hub.pull("hwchase17/openai-functions-agent")


# 初始化大模型
# model_name="qwen2.5:7b"
model_name="llama3.2-vision:11b"
api_key="ollama"
base_url="http://localhost:11434/v1/"

# model_name="Qwen/Qwen2.5-7B-Instruct"
# api_key =  os.getenv("OPENAI_API_KEY")
# base_url="https://api.siliconflow.cn/v1/"


llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url)

# 创建 OpenAI 函数代理
# agent = create_openai_tools_agent(llm, tools, prompt)

#  create_openai_tools_agent 使用 create_tool_calling_agent 替换
agent = create_tool_calling_agent(llm, tools, prompt)

# 通过传入代理和工具创建代理执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# 运行代理

response = agent_executor.invoke({
    "input": "目前市场上黄金的平均售价是多少？" 
  })

print(response["output"])