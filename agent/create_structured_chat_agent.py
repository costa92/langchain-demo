# 使用 Tavily Search 测试代理
from langchain_community.tools.tavily_search import TavilySearchResults
# 创建代理Agent
from langchain.agents import AgentExecutor, create_structured_chat_agent
# 初始化大模型
from langchain_openai import ChatOpenAI

from langchain import hub


model_name="qwen2.5:7b"
# model_name="llama3.2-vision:11b"
api_key="ollama"
base_url="http://localhost:11434/v1/"

# import os
# model_name="Qwen/Qwen2.5-7B-Instruct"
# api_key =  os.getenv("OPENAI_API_KEY")
# base_url="https://api.siliconflow.cn/v1/"


# llm = ChatOpenAI(
#   model=model_name, 
#   api_key=api_key,
#   base_url=base_url,
#   temperature=0.5
#   )

from langchain_ollama import ChatOllama
llm = ChatOllama(model=model_name, temperature = 0.8,num_predict = 256,)



tools = [TavilySearchResults(max_results=1)]

# 获取使用提示，也可以自己写
prompt = hub.pull("hwchase17/structured-chat-agent")

# 初始化Agent
agent = create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# 输入代理和工具，创建代理执行器
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=False, 
)

# 运行代理Agent
# response = agent_executor.invoke({"input": "网站https://www.runoob.com中有哪些教程?"})
# print(f"回答：{response.get('output')}\n")


from langchain_core.messages import AIMessage, HumanMessage
response =   agent_executor.invoke(
    {
        "input": "what's my name?",
        "chat_history": [
            HumanMessage(content="hi! my name is bob"),
            AIMessage(content="Hello Bob! How can I assist you today?"),
        ],
    }
)

print(f"回答：{response.get('output')}\n")