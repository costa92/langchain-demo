
# 最简单的记忆形式就是将聊天历史消息传递到链中。以下是示例：

from langchain_openai import ChatOpenAI


# Choose the model and set up the API connection
model_name = "llama3.2-vision:11b"
llm = ChatOpenAI(
    model=model_name,
    api_key="ollama",
    base_url="http://localhost:11434/v1/",
)


from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="你是一位乐于助人的助手。尽你所能回答所有问题。"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | llm

ai_msg = chain.invoke(
    {
        "messages": [
            HumanMessage(
                content="从英语翻译成中文：我喜欢编程."
            ),
            AIMessage(content="I love programming"),
            HumanMessage(content="你刚才说什么？"),
        ],
    }
)
print(ai_msg.content)