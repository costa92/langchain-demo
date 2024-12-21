from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.tools.tavily_search import TavilySearchResults


# 查询 Tavily 搜索 API 并返回 json 的工具
search = TavilySearchResults()

# 创建将在下游使用的工具列表
tools = [search]

from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI

# 初始化大模型
model_name="llama3.2-vision:11b"
api_key="ollama"
base_url="http://localhost:11434/v1/"


llm = ChatOpenAI(
  model=model_name, 
  api_key=api_key, 
  base_url=base_url,
  temperature=0
) 
# 获取使用的提示
prompt = hub.pull("hwchase17/openai-functions-agent")



# 创建 OpenAI 函数代理
agent = create_openai_functions_agent(llm, tools, prompt)

# 通过传入代理和工具创建代理执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

message_history = ChatMessageHistory()

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

agent_with_chat_history.invoke(
    {"input": "hi! I'm bob"},
    config={"configurable": {"session_id": "<foo>"}},
)

res = agent_with_chat_history.invoke(
    {"input": "what's my name?"},
    config={"configurable": {"session_id": "<foo>"}},
)

print(f"回答：{res.get('output')}\n")

