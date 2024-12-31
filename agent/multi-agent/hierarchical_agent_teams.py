import os
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from typing import List,Annotated,Optional,Dict,Literal
from langchain_community.document_loaders import WebBaseLoader
from tempfile import TemporaryDirectory
from pathlib import Path
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent

from langchain_experimental.utilities import PythonREPL
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from typing_extensions import TypedDict
from langgraph.types import Command

_TEMP_DIRECTORY = TemporaryDirectory()
WORKING_DIRECTORY = Path(_TEMP_DIRECTORY.name)


#  Hierarchical Agent Teams 
#  :zh  
# https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/multi_agent/hierarchical_agent_teams.ipynb

tavily_tool = TavilySearchResults(max_results=5)

@tool
def scrape_webpages(urls: List[str]) -> str:
  """Use requests and bs4 to scrape the provided web pages for detailed information."""
  loader = WebBaseLoader(urls)
  docs = loader.load()
  return "\n\n".join(
        [
            f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )


@tool
def create_outline(
    points: Annotated[List[str], "List of main points or sections."],
    file_name: Annotated[str,"File path to save the outline"],
) -> Annotated[str, "Path to the saved outline file."]:
    """Create an outline of the document."""
    with(WORKING_DIRECTORY/file_name).open("w") as f:
       for i,point in enumerate(points):
          f.write(f"{i+1}. {point}\n")
    return f"Outline saved to {file_name}"


@tool
def read_document(
   file_name: Annotated[str, "File path to the document to read."],
   start: Annotated[Optional[int], "The start line. Default is 0"]=None,
   end: Annotated[Optional[int], "The end line. Default is None"]=None,
) -> str:
    """Read the specified document."""
    with (WORKING_DIRECTORY / file_name).open("r") as f:
      lines = f.readlines()
    if start is None:
       start = 0
    return "\n".join(lines[start:end])


@tool
def write_document(
   content: Annotated[str, "The content to write to the document."],
  file_name: Annotated[str, "File path to save the document."],
) -> Annotated[str, "Path to the saved document file."]:
    """Create and save a text document file."""
    with (WORKING_DIRECTORY / file_name).open("w") as f:
      f.write(content)
    return f"Document saved to {file_name}"


@tool
def edit_document(
   file_name: Annotated[str,"Path of the document to be edited"],
   inserts: Annotated[
      Dict[int, str],
      "Dictionary where key is the line number (1-indexed) and value is the text to be inserted at that line.",
   ],
) -> Annotated[str, "Path of the edited document file."]:
    """Edit a document by inserting text at specific line numbers."""
    with (WORKING_DIRECTORY / file_name).open("r") as f:
       lines = f.readlines()
    # sorted_inserts = sorted(inserts.items(), key=lambda x: x[0])
    sorted_inserts = sorted(inserts.items())

    for line_number, text in sorted_inserts:
        if 1 <= line_number <= len(lines) + 1:
           lines.insert(line_number - 1, text + "\n")
        else:
           return f"Error: line number {line_number} is out of range."
        

    with (WORKING_DIRECTORY / file_name).open("w") as f:
       f.writelines(lines)
    return "Document edited and saved to {file_name}."


repl = PythonREPL()
@tool
def python_repl_tool(
  code: Annotated[str, "The python code to execute to generate your chart."], 
):
   """Use this to execute python code. if you want to see the output of a value,
   you should print it out with `print(...)`. This is visible to the user."""
   try:
      return repl.run(code)
   except BaseException as e:
      return f"Failed to execute. Error: {repr(e)}"
   result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
   return 



def make_supervisor_node(llm: BaseChatModel, members:List[str]) -> str:
   options = ["FINISH"] + members

   system_prompt = (
       "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
   )

   class Router(TypedDict):
       """Worker to route to next. If no workers needed, route to FINISH."""
       next: Literal["search","web_scraper","FINISH"]

   def supervisor_node(state: MessagesState) ->  Command[Literal["search", "web_scraper", "__end__"]]:
       """An LLM-based router."""
       messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
       response = llm.with_structured_output(Router).invoke(messages)
       goto = response["next"]
       if goto == "FINISH":
          goto = END
       return Command(goto=goto)
   return supervisor_node

## Define Agent Teams


# llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
from langchain_openai import ChatOpenAI
import os
# model_name="deepseek-chat"
# api_key = os.getenv("deepseek_api_key")
# base_url = os.getenv("deepseek_api_url")


# model_name = "qwen2.5:7b"
# api_key="ollama"
# base_url="http://localhost:11434/v1/"


model_name="Qwen/Qwen2.5-7B-Instruct"
# model_name="Qwen/Qwen2-VL-72B-Instruct"
import os
api_key =  os.getenv("OPENAI_API_KEY")
base_url="https://api.siliconflow.cn/v1/"


llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_url,
)

search_agent = create_react_agent(llm, tools=[tavily_tool])

def search_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = search_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="search")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )

web_scraper_agent = create_react_agent(llm, tools=[scrape_webpages])


def web_scraper_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = web_scraper_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="web_scraper")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )

research_supervisor_node = make_supervisor_node(llm, ["search", "web_scraper"])

research_builder = StateGraph(MessagesState)
research_builder.add_node("supervisor", research_supervisor_node)
research_builder.add_node("search", search_node)
research_builder.add_node("web_scraper", web_scraper_node)

research_builder.add_edge(START, "supervisor")
research_graph = research_builder.compile()


# INSERT_YOUR_REWRITE_HERE
from IPython.display import display
from PIL import Image as PILImage
import io
image_data = research_graph.get_graph().draw_mermaid_png()
image = PILImage.open(io.BytesIO(image_data))
image.save("hierarchical_agent_teams.png")


for s in research_graph.stream(
    {"messages": [("user", "中国传统文化在世界上的影响,2025年春节是否有不同影响,请提供参考链接")]},
    {"recursion_limit": 100},
):
    print(s)
    print("---")