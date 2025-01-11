
import os 
from langchain_openai import OpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

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

tools = [
    Tool(
        name="Intermediate Answer", 
        func=search.run,
        description="useful for when you need to ask with search", 
    )
]

self_ask_with_search = initialize_agent(
    tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True,handle_parsing_errors=True
)
print(self_ask_with_search.agent.llm_chain.prompt)

self_ask_with_search.invoke(
    "使用玫瑰作为国花的国家的首都是哪里?"  
)