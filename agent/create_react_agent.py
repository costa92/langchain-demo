# 访问 SerpApi ，注册账号，选择相应的订阅计划(Free)，然后获取API Key，利用这个API为大模型提供Google搜索工具。
# https://serpapi.com/
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# 开启DEBUG 显示具体的日志信息
# langchain.debug = True
# langchain.verbose = True



# 初始化大模型:语言模型控制代理
# 初始化大模型
# model_name="qwen2.5:7b"
# model_name="llama3.2-vision:11b"
model_name = "llama3"
api_key="ollama"
base_url="http://localhost:11434/v1/"

llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url)
# # 重新build LLMMathChain
# from langchain.chains.llm_math import LLMMathChain
# LLMMathChain.model_rebuild()

# 设置工具:加载使用的工具，serpapi:调用Google搜索引擎 llm-math:通过LLM进行数学计算的工具
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# 获取使用提示
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


# 让代理来回答提出的问题
# res = agent_executor.invoke({"input": "目前市场上苹果手机15的平均售价是多少？如果我在此基础上加价5%卖出，应该如何定价？"})
res = agent_executor.invoke(
    {
        "input": "what's my name?",
        "chat_history":  [
              HumanMessage(content="My name is Bob", extra={"role": "user"}),
              AIMessage(content="Hello Bob!", extra={"role": "assistant"}),
          ]
    }
)

print(f"回答：{res.get('output')}\n")