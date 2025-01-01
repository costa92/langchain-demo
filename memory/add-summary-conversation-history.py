
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, RemoveMessage
from langgraph.graph import MessagesState,StateGraph,START
from typing import Literal
from langchain_core.messages import HumanMessage


memory = MemorySaver()

# We will add a `summary` attribute (in addition to `messages` key,
# which MessagesState already has)
class State(MessagesState):
    summary: str 


model_name="Qwen/Qwen2.5-7B-Instruct"
# model_name="Qwen/Qwen2-VL-72B-Instruct"
import os
api_key =  os.getenv("OPENAI_API_KEY")
base_url="https://api.siliconflow.cn/v1/"

# We will use this model for both the conversation and the summarization
# :zh: 我们将使用这个模型来进行对话和总结
model = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_url,
)

# Define the logic to call the model
def call_model(state: State):
    # If a summary exists, we add this in as a system message
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
        # system_message = SystemMessage(content=summary)
        # messages = [system_message] + state["messages"]
    else:
        messages = state["messages"]

    # Call the model
    response = model.invoke(messages)

    return {"messages": [response]}

def should_continue(state: State) -> Literal["summarize_conversation", "__end__"]:
    """Return the next node to execute."""
    messages = state["messages"]
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"
    return "__end__"


def summarize_conversation(state: State):
    # First, we summarize the conversion
    summary = state.get("summary","")
    if summary:
        # If a summary already exists, we use a different system prompt
        # to summarize it than if one didn't
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]

    response = model.invoke(messages)
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary":response.content,"messages":delete_messages}


# Define a new graph
workflow = StateGraph(State)

# Define the conversation node and the summarize node
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)
# Set the entrypoint as conversation
workflow.add_edge(START, "conversation")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `conversation`.
    # This means these are the edges taken after the `conversation` node is called.
    "conversation",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `summarize_conversation` to END.
# This means that after `summarize_conversation` is called, we end.
workflow.add_edge("summarize_conversation", "__end__")

# Finally, we compile it!
app = workflow.compile(checkpointer=memory)


def print_update(update):
    for k, v in update.items():
        for m in v["messages"]:
            m.pretty_print()
        if "summary" in v:
            print(v["summary"])




config = {"configurable": {"thread_id": "4"}}
input_message = HumanMessage(content="hi! I'm bob")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

input_message = HumanMessage(content="what's my name?")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

input_message = HumanMessage(content="i like the celtics!")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)