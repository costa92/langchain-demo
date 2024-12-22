# LangChain 还提供了一种使用 LangGraph 的持久性构建具有内存的应用程序的方法。您可以在编译图时提供 LangGraph 应用程序中的持久性。

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.messages import trim_messages

# count each message as 1 "token" (token_counter=len) and keep only the last two messages
# 以每条消息计为 1 个“标记”（token_counter=len）并仅保留最后两条消息
trimmer = trim_messages(strategy="last", max_tokens=2, token_counter=len)

workflow = StateGraph(state_schema=MessagesState)

model_name = "llama3"
llm = ChatOpenAI(
    model=model_name,
    api_key="ollama",
    base_url="http://localhost:11434/v1/",
)

# Define the function that calls the model
def call_model(state: MessagesState):
    # add    trimmed_messages 
    trimmed_messages = trimmer.invoke(state["messages"])
    system_prompt = (
        "You are a helpful assistant. "
        "Answer all questions to the best of your ability."
    )
    messages = [SystemMessage(content=system_prompt)] + trimmed_messages
    response = llm.invoke(messages)
    return {"messages": response}


# Define the node and edge
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# Add simple in-memory checkpointer : zh 一个简单的内存检查点
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


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
    config={"configurable": {"thread_id": "3"}},
)

print(res["messages"][-1].content)

# 输出 ：I'm afraid I don't have that information! As our conversation just started, we didn't establish any personal details about you yet. If you'd like to share your name, I'd be happy to know it!

# 从上面的输出可以看出，我们的模型无法回答问题，因为我们的对话刚刚开始，我们还没有建立关于用户的任何个人信息。这是因为我们在调用模型之前使用了trimmer.invoke(state["messages"])来修剪消息。这将确保我们只传递最近的两条消息给模型，而不是整个聊天历史记录。
