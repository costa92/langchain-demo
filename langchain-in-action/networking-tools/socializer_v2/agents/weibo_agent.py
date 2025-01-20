from tools.search_tool import get_UID
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, create_react_agent, AgentExecutor
import os

def lookup_V(flower_type: str) -> str:
    """
    Look up the 微博 UID for a given flower type using a LangChain agent.

    Args:
        flower_type (str): The type of flower to look up.

    Returns:
        str: The UID associated with the flower type.
    """
    API_KEY = os.getenv("OPENAI_API_KEY")
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    base_url = "https://api.siliconflow.cn/v1"
    
    if not API_KEY:
        raise ValueError("API key for OpenAI is not set. Please set the OPENAI_API_KEY environment variable.")

    llm = ChatOpenAI(api_key=API_KEY, model_name=model_name, base_url=base_url)

    # 寻找UID的模板
    template = """
        Given the flower type '{flower}', find the related 微博 UID.
        {tools}
        Your answer should contain only the UID.
        The URL always starts with https://weibo.com/u/
        For example, if https://weibo.com/u/1669879400 is the 微博, then 1669879400 is the UID.
        Question: Given the flower type '{flower}', find the related 微博 UID.
        Thought: I need to use the tools available to find the UID.
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: The result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
        
        Begin!

        Question: {flower}
        Thought:{agent_scratchpad}
    """


    # Include all required variables in input_variables
    prompt_template = PromptTemplate(
        input_variables=["flower", "tool_names", "agent_scratchpad", "tools"],
        template=template
    )

    tools = [
        Tool(
            name="Crawl Google for 微博 page",
            func=get_UID,
            description="Useful for retrieving 微博 UID."
        )
    ]

    # Create agent with the fixed prompt
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt_template)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True)


    # Execute the agent
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
