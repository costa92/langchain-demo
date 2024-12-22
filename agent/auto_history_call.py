# LangChain 还提供了一种使用 LangGraph 的持久性构建具有内存的应用程序的方法。您可以在编译图时提供 LangGraph 应用程序中的持久性。

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage,RemoveMessage
from langchain_openai import ChatOpenAI


workflow = StateGraph(state_schema=MessagesState)

# model_name = "qwen2.5:7b"
model_name = "llama3.2-vision:11b"
llm = ChatOpenAI(
    model=model_name,
    api_key="ollama",
    base_url="http://localhost:11434/v1/",
)

# Define the function that calls the model
def call_model(state: MessagesState):
    # add    trimmed_messages 
    system_prompt = (
        "You are a helpful assistant. "
        "Answer all questions to the best of your ability."
        "The provided chat history includes a summary of the earlier conversation."
    )
    system_message = SystemMessage(content=system_prompt)
    message_history = state["messages"][:-1]  # exclude the most recent user input  : zh 排除最近的用户输入
    # Summarize the messages if the chat history reaches a certain size
    # 如果聊天历史记录达到一定大小，则对消息进行总结
    if len(message_history) >= 10:
         last_human_message = state["messages"][-1]
         # Invoke the model to generate conversation summary
         # 调用模型生成对话摘要
         summary_prompt = (
            "Distill the above chat messages into a single summary message. "
            "Include as many specific details as you can."
         )
         summary_message = llm.invoke(
            message_history + [HumanMessage(content=summary_prompt)]
         )
         # Delete messages that we no longer want to show up
         # 删除我们不希望显示的消息
         delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]
         # Re-add user message
         # 重新添加用户消息
         human_message = HumanMessage(content=last_human_message.content)
         # Call the model with summary & response
         # 使用摘要和响应调用模型
         response = llm.invoke([system_message, summary_message, human_message])
         message_updates = [summary_message, human_message, response] + delete_messages
    else:
        message_updates = llm.invoke([system_message] + state["messages"])
    return {"messages": message_updates}


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
        + [HumanMessage("What did I say my name was?")]
    },
    config={"configurable": {"thread_id": "6"}},
)

print(res["messages"][-1].content)

# 输出 ：I'm afraid I don't have that information! As our conversation just started, we didn't establish any personal details about you yet. If you'd like to share your name, I'd be happy to know it!

# 从上面的输出可以看出，我们的模型无法回答问题，因为我们的对话刚刚开始，我们还没有建立关于用户的任何个人信息。这是因为我们在调用模型之前使用了trimmer.invoke(state["messages"])来修剪消息。这将确保我们只传递最近的两条消息给模型，而不是整个聊天历史记录。
