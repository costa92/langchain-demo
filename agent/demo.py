from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# 初始化工具
tools = [TavilySearchResults(max_results=1)]

# 获取使用的提示
prompt = hub.pull("hwchase17/openai-functions-agent")

model_name = "llama3"
# 选择将驱动代理的LLM
llm = ChatOpenAI(
  model=model_name, 
  api_key="ollama", 
  base_url="http://localhost:11434/v1/",
  )
  

# 构建OpenAI函数代理
agent = create_openai_functions_agent(llm, tools, prompt)

# 通过传入代理和工具创建代理执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 执行
agent_executor.invoke(
    {
        "input":"我叫什么名字?",
        "chat_history":[
            HumanMessage(content="你好！我叫鲍勃"),
            AIMessage(content="你好鲍勃！我今天能帮你什么?"),],
    }
)



