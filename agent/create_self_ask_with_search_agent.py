from langchain import hub
from langchain_community.tools.tavily_search import TavilyAnswer
from langchain.agents import create_self_ask_with_search_agent,AgentExecutor
from langchain_openai import ChatOpenAI

# 将初始化工具，让它提供答案而不是文档

tools = [TavilyAnswer(max_results=1,name="Intermediate Answer",description="查询答案")]

# 初始化大模型
# model_name="qwen2.5:7b"
model_name="llama3.2-vision:11b"
api_key="ollama"
base_url="http://localhost:11434/v1/"

llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url)

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

# res = agent_executor.invoke({"input": "我想知道关于python的一件事情"})
agent_executor.invoke({"input": "中国有哪些省份呢?"})

# print(f"回答：{res.get('output')}\n")