
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic

from typing import Annotated,Literal
from typing_extensions import TypedDict
from langgraph.graph import MessagesState
from langgraph.types import Command

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

class Route(TypedDict):
  """Worker to router to next. If no worker needed, route to FINISH."""
  next: Literal[*options]


llm = ChatAnthropic(model="")

# https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/multi_agent/agent_supervisor.ipynb
def supervisor_node(state: MessagesState) -> Command[Literal[*members, "__end__"]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = END

    return Command(goto=goto)