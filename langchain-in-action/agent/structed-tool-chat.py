from langchain_community.agent_toolkits.playwright.toolkit import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_structured_chat_agent
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import render_text_description
from langchain_core.messages import SystemMessage, HumanMessage
from langchain import hub
# 创建浏览器工具
async_browser = create_async_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
tools = toolkit.get_tools()

# 初始化语言模型
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 OPENAI_API_KEY")

model_name = "Qwen/Qwen2.5-7B-Instruct"
base_url = "https://api.siliconflow.cn/v1"

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=base_url,
    model_name=model_name,
)

prompt = hub.pull("hwchase17/structured-chat-agent")

# 创建代理
agent = create_structured_chat_agent(llm, tools, prompt)
# 创建代理执行器
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True
)

async def main():
    try:
        response = await agent_executor.ainvoke({
            "input": "What are the headers on python.langchain.com?",
            "chat_history": [],  # 空的聊天历史
            "agent_scratchpad": []  # 确保提供一个空的消息列表
        })
        print(response)
    except Exception as e:
        print(f"An error occurred: {e}")
        # 打印完整的错误追踪信息
        import traceback
        traceback.print_exc()

import asyncio
loop = asyncio.get_event_loop()
loop.run_until_complete(main())
