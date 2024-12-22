# 访问 SerpApi ，注册账号，选择相应的订阅计划(Free)，然后获取API Key，利用这个API为大模型提供Google搜索工具。
# https://serpapi.com/
from langchain_openai import ChatOpenAI
from langchain import hub
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
import uuid
from langchain_core.tools import tool


# 开启DEBUG 显示具体的日志信息
# langchain.debug = True
# langchain.verbose = True



# 初始化大模型:语言模型控制代理
# 初始化大模型
model_name="qwen2.5:7b"
# model_name="llama3.2-vision:11b"
# model_name = "llama3"
api_key="ollama"
base_url="http://localhost:11434/v1/"

llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url)
# # 重新build LLMMathChain
# from langchain.chains.llm_math import LLMMathChain
# LLMMathChain.model_rebuild()


@tool
def get_user_age(name: str) -> str:
    """Use this tool to find the user's age."""
    # This is a placeholder for the actual implementation
    if "bob" in name.lower():
        return "42 years old"
    return "41 years old"


memory = MemorySaver()

# 初始化Agent:使用工具、语言模型和代理类型来初始化代理
# create_react_agent 参数： 
app = create_react_agent(
  model=llm, 
  tools=[get_user_age], 
  checkpointer=memory,
)

# The thread id is a unique key that identifies
# this particular conversation.
# We'll just generate a random uuid here.
# This enables a single application to manage conversations among multiple users.
# zh 线程ID是一个唯一的键，
# 用于标识这个特定的对话。
# 我们只是在这里生成一个随机的uuid。
# 这使得单个应用程序可以管理多个用户之间的对话。
thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}



# Tell the AI that our name is Bob, and ask it to use a tool to confirm
# that it's capable of working like an agent.
# 告诉AI我们的名字是Bob，并要求它使用一个工具来确认
# 它是否能够像代理一样工作。
input_message = HumanMessage(content="hi! I'm bob. What is my age?")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()


# Confirm that the chat bot has access to previous conversation
# and can respond to the user saying that the user's name is Bob.
# 确认聊天机器人可以访问以前的对话
# 并回应用户说用户的名字是Bob。
input_message = HumanMessage(content="do you remember my name?")

for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()
