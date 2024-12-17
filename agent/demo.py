from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate,MessagesPlaceholder
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import AIMessage, HumanMessage

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
import os
import numexpr as ne

TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")
# 定义模型
model_name  = "llama3"
temperature = 0
llm = OllamaLLM(model=model_name, temperature=temperature)


# # 定义工具 
# @tool
# def get_weather(city: Literal["nyc", "sf"]):
#     """Use this to get weather information."""
#     if city == "nyc":
#         return "It might be cloudy in nyc"
#     elif city == "sf":
#         return "It's always sunny in sf"
#     else:
#         raise AssertionError("Unknown city")

# 计算工具
@tool
def calculate_expression(expression: str) -> str:
  """
  计算给定的数学表达式并返回结果。
  
  参数:
  expression (str): 要计算的数学表达式。
  
  返回:
  str: 计算结果的字符串表示。
  """
  try:
    expression_str = str(expression)
    result = ne.evaluate(expression_str).item()  # Use numexpr to evaluate the expression
    return str(result)  # Return result as string for agent's output
  except Exception as e:
    print(f"Error calculating expression: {str(e)}")
    return f"Error calculating expression: {str(e)}"

tools = [TavilySearchResults(max_results=1),calculate_expression]

# 定义prompt
prompt = hub.pull("hwchase17/react")

#初始化记忆类型
memory = MemorySaver()

#  agent 三要输
# math_tools=[get_weather]
agent = create_react_agent(
    tools=tools,
    llm=llm,
    prompt=prompt,
)

agent_executor = AgentExecutor(
  agent=agent, 
  tools=tools, 
  verbose=True,
  handle_parsing_errors=False,
  allow_dangerous_code=True,
  early_stopping_method="force",  # 保持为 "force"
  max_iterations=5,  # 增加迭代次数
  max_execution_time=30,  # 增加每个工具的执行时间
  max_time=120,  # 增加总的最大时间限制 
)

# agent_executor.invoke({"input": "总共有三个鸡蛋，被小明吃了一个鸡蛋，请问还有几个鸡蛋？"})

agent_executor.invoke(
    {
        "input": "我的名字是什么？",
        "chat_history": [
            HumanMessage(content="嗨！我的名字是Bob"),
            AIMessage(content="你好Bob！有什么我可以帮助你的吗？"),
        ],
    }
)