
# 设置OpenAI和SERPAPI的API密钥
# https://juejin.cn/post/7402475006737252390?searchId=202501072234150D8FC75B554768350229
import os


from langchain_openai import ChatOpenAI,OpenAI
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import Tool
from langchain.chains import LLMMathChain

# 初始化语言模型
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 OPENAI_API_KEY")

model_name = "Qwen/Qwen2.5-7B-Instruct"
base_url = "https://api.siliconflow.cn/v1"

llm = OpenAI(
    api_key=API_KEY,
    base_url=base_url,
    model_name=model_name,
)

search = SerpAPIWrapper()
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    ),
]
model = ChatOpenAI(
    api_key=API_KEY,
    base_url=base_url,
    model_name=model_name,
)
planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

res = agent.invoke("在纽约，100美元能买几束玫瑰?")
print(res)