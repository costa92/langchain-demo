
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic

from typing import Annotated,Literal
from typing_extensions import TypedDict
from langgraph.graph import MessagesState
from langgraph.types import Command

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent


tavily_tool = TavilySearchResults(max_results=5)
repl = PythonREPL()



### Create tools
@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
  """Use this to execute python code and do math. If you want to see the output of a value,you should print it out with `print(...)`. This is visible to the user."""
  try:
    return repl.run(code)
  except BaseException as e:
    return f"Failed to execute. Error: {repr(e)}"
  # result_str = f"Successfully executed code: \n ```python\n {code}\n```\nStdout: \n{result}"
  result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
  return result_str


### Create Agent Supervisor
members = ["researcher","coder"]

options = members + ["FINISH"]

system_prompt = (
  "You are a supervisor tasked with managing a conversation between the"
  f" following workers: {members}. Given the following user request,"
  " respond with the worker to act next. Each worker will perform a"
  " task and respond with their results and status. When finished,"
  " respond with FINISH."
)

class Router(TypedDict):
  """Worker to router to next. If no worker needed, route to FINISH."""
  # next: Literal[*options]
  next: Literal["researcher","coder","FINISH"]


# llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
from langchain_openai import ChatOpenAI
import os
# model_name="deepseek-chat"
# api_key = os.getenv("deepseek_api_key")
# base_url = os.getenv("deepseek_api_url")


model_name = "qwen2.5:7b"
api_key="ollama"
base_url="http://localhost:11434/v1/"


# model_name="Qwen/Qwen2.5-7B-Instruct"
# import os
# api_key =  os.getenv("OPENAI_API_KEY")
# base_url="https://api.siliconflow.cn/v1/"


llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_url,
)

# https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/multi_agent/agent_supervisor.ipynb
# INSERT_YOUR_REWRITE_HERE
def supervisor_node(state: MessagesState) -> Command[Literal["researcher", "coder", "__end__"]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    try:
        response = llm.with_structured_output(Router).invoke(messages)
        goto = "__end__"
        if response!= None:
            goto = response["next"]
    except (KeyError, TypeError) as e:
        print(f"Error processing response: {e}")
        goto = "__end__"

    if goto == "FINISH":
        goto = "__end__"

    return Command(goto=goto)

### Construct Graph

research_agent = create_react_agent(
    llm, tools=[tavily_tool], state_modifier="You are a researcher. DO NOT do any math."
)


def research_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = research_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="researcher")
            ]
        },
        goto="supervisor",
    )


# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION, WHICH CAN BE UNSAFE WHEN NOT SANDBOXED
code_agent = create_react_agent(llm, tools=[python_repl_tool])
def code_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = code_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="coder")
            ]
        },
        goto="supervisor",
    )


builder = StateGraph(MessagesState)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", research_node)
builder.add_node("coder", code_node)
graph = builder.compile()

# 在 Jupyter notebook 实现:
# from IPython.display import display, Image
# display(Image(graph.get_graph().draw_mermaid_png()))

# INSERT_YOUR_REWRITE_HERE
from IPython.display import display
from PIL import Image as PILImage
import io
image_data = graph.get_graph().draw_mermaid_png()
image = PILImage.open(io.BytesIO(image_data))
image.save("output.png")



#  循环处理
for s in graph.stream(
    {"messages": [("user", "What's the square root of 42?")]}, subgraphs=True
):
  print(s)
  print("----")