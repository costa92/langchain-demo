# LangChain 还提供了一种使用 LangGraph 的持久性构建具有内存的应用程序的方法。您可以在编译图时提供 LangGraph 应用程序中的持久性。

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

workflow = StateGraph(state_schema=MessagesState)

model_name = "llama3"
llm = ChatOpenAI(
    model=model_name,
    api_key="ollama",
    base_url="http://localhost:11434/v1/",
)

# Define the function that calls the model
def call_model(state: MessagesState):
    system_prompt = (
        "You are a helpful assistant. "
        "Answer all questions to the best of your ability."
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": response}


# Define the node and edge
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# Add simple in-memory checkpointer : zh 一个简单的内存检查点
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


# 我们将在这里将最新的输入传递给对话，并让 LangGraph 使用检查点跟踪对话历史记录：
app.invoke(
    {"messages": [HumanMessage(content="Translate to French: I love programming.")]},
    config={"configurable": {"thread_id": "1"}},
)

app.invoke(
    {"messages": [HumanMessage(content="What did I just ask you?")]},
    config={"configurable": {"thread_id": "1"}},
)


# 修改聊天历史记录
demo_ephemeral_chat_history = [
    HumanMessage(content="Hey there! I'm Nemo."),
    AIMessage(content="Hello!"),
    HumanMessage(content="How are you today?"),
    AIMessage(content="Fine thanks!"),
]

res = app.invoke(
    {
        "messages": demo_ephemeral_chat_history
        + [HumanMessage(content="What's my name?")]
    },
    config={"configurable": {"thread_id": "2"}},
)

print(res["messages"][-1].content)

# 但是，假设我们的上下文窗口非常小，并且我们想要将传递给模型的消息数量修剪为仅最近的 2 条。我们可以使用内置的trim_messages实用程序根据消息到达提示之前的标记计数来修剪消息。在这种情况下，我们将每条消息计为 1 个“标记”，并仅保留最后两条消息：

# from langchain_core.messages import trim_messages

# # Define trimmer
# # count each message as 1 "token" (token_counter=len) and keep only the last two messages
# trimmer = trim_messages(strategy="last", max_tokens=2, token_counter=len)

