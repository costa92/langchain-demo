import uuid
from langgraph.types import interrupt, Command
from typing import Literal,TypedDict
from langgraph.graph import StateGraph
from langgraph.constants import START
from langgraph.checkpoint.memory import MemorySaver



class State(TypedDict):
   """The graph state."""
   some_text: str


def human_node(state: State):
   value = interrupt(
       # Any JSON serializable value to surface to the human.
        # For example, a question or a piece of text or a set of keys in the state
       {
          "text_to_revise": state["some_text"]
       }
   )
   return {"some_text": value}


# Build the graph
graph_builder = StateGraph(State)
# Add the human-node to the graph
graph_builder.add_node("human_node", human_node)
graph_builder.add_edge(START, "human_node")

# Compile the graph

# A checkpointer is required for `interrupt` to work.
checkpointer = MemorySaver()
graph = graph_builder.compile(
   checkpointer=checkpointer
)


# Pass a thread ID to the graph to run it.
thread_config = {"configurable": {"thread_id": uuid.uuid4()}}


# Using stream() to directly surface the `__interrupt__` information.
for chunk in graph.stream({"some_text": "原文"}, config=thread_config):
   print(chunk)

# Resume using Command
for chunk in graph.stream(Command(resume="编辑文本"), config=thread_config):
   print(chunk)

# 定义一个节点，用于人工审核
def human_approval(state: State) -> Command[Literal["some_node", "another_node"]]:
   is_approved = interrupt(
      {
         "question": "这是正确的吗？",
          # Surface the output that should be
          # reviewed and approved by the human.
          "llm_output": state["llm_output"]
      }
   )
   if is_approved:
      return interrupt("some_node")
   else:
      return interrupt("another_node")
   



# # Add the node to the graph in an appropriate location
# # and connect it to the relevant nodes.
graph_builder.add_node("human_approval", human_approval)
graph = graph_builder.compile(checkpointer=checkpointer)


thread_config = {"configurable": {"thread_id": "some_id"}}
graph.invoke(Command(resume=True), config=thread_config)