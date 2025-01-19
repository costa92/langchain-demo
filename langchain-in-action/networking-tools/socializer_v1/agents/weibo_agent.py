from tools.search_tool import get_UID
from langchain.prompts import PromptTemplate
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
import os

def lookup_V(flower_type: str):
    # 初始化大模型
    model_name = "deepseek-chat"
    API_KEY = os.getenv("deepseek_api_key")
    base_url = "https://api.deepseek.com/beta"
    llm = ChatOpenAI(api_key=API_KEY, base_url=base_url, model=model_name, temperature=0.2)
    
    # 寻找UID的模板
    # template = """Given the flower type '{flower}', find the related 微博 UID.
    #               {tools}\n\n"
    #               Your answer should contain only the UID.
    #               The URL always starts with https://weibo.com/u/
    #               For example, if https://weibo.com/u/1669879400 is the 微博, then 1669879400 is the UID.
    #               Question: Given the flower type '{flower}', find the related 微博 UID.
    #               Thought: I need to use the tools available to find the UID.
    #               Action: {tool_names}
    #               Action Input: The input to the action
    #               Observation: The result of the action
    #               Thought: I now know the final answer.
    #               Action: Final Answer: {agent_scratchpad}"""
    template = """Given the flower type '{flower}', find the related 微博 UID.
    {tools}

    Question: Given the flower type '{flower}', find the related 微博 UID.
    Thought: I need to use the tools available to find the UID.
    Action: {tool_names}
    Action Input: {flower}
    Observation: {agent_scratchpad}
    Thought: I now know the final answer.
    Action: Final Answer
    Action Input: {agent_scratchpad}"""
    # 完整的提示模板
    prompt_template = PromptTemplate(
        input_variables=["flower", "agent_scratchpad", "tool_names"],  # 移除 tools，因为模板中未使用
        template=template
    )

    # 代理的工具
    tools = [
        Tool(
            name="Crawl Google for 微博 page",
            func=get_UID,
            description="useful for when you need get the 微博 UID",
        )
    ]

    # 初始化代理
    agent = create_react_agent(
       tools=tools, 
       llm=llm, 
       prompt=prompt_template
    )

    # 初始化代理执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,  # 处理解析错误
        verbose=True  # 启用详细日志
    )

    # 返回找到的UID
    try:
        result = agent_executor.invoke({
            "flower": flower_type,
            "agent_scratchpad": "",
            "tool_names": [tool.name for tool in tools]  # 传入 tool_names
        })
        # 从返回的字典中提取 UID
        if result and "output" in result:
            return result["output"]
        else:
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None