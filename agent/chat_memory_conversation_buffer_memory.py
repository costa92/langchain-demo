from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


# ConversationBufferMemory: 最简单的缓冲记忆，将所有对话信息全部存储作为记忆。
# ConversationBufferWindowMemory:  用于跟踪对话的最后k轮。
# ConversationTokenBufferMemory:  过设置最大标记数量（max_token_limits）来决定何时清除交互信息，当对话信息超过 max_token_limits 时，抛弃旧对话信息。
# ConversationStringBufferMemory:  等同于缓冲记忆，固定返回字符串（这是早期 LangChain 封装的记忆组件）。

# 摘要记忆组件的类型
# ConversationSummaryMemory：将传递的历史对话记录总结成摘要进行保存，使用时填充的记忆为摘要，而非对话数据。
# ConversationSummaryBufferMemory：在不超过 max_token_limit 的限制下保存对话历史数据，对于超过的部分进行信息的提取与总结。

# 实体记忆组件

# ConversationEntityMemory: 用于存储对话中的实体信息，例如对话中提到的人名、地名、时间等。




# Set up ConversationBufferMemory to store interactions
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Adding some initial interactions to memory
memory.save_context({"input": "Hello, how are you?"}, {"output": "I'm doing great, thank you!"})
memory.save_context({"input": "What's the weather like?"}, {"output": "It's sunny today."})

# Create a prompt template for the model
prompt_template = PromptTemplate(input_variables=["input", "memory"], template="{memory} {input}")

# Choose the model and set up the API connection
model_name = "llama3"
llm = ChatOpenAI(
    model=model_name,
    api_key="ollama",
    base_url="http://localhost:11434/v1/",
)

# Combine the prompt template and the model to create the runnable sequence
runnable = prompt_template | llm | StrOutputParser()

# Query the memory and invoke the model with the current input
current_input = "What day is today? What's the weather like today?"
res = runnable.invoke({"input": current_input, "memory": memory.load_memory_variables({})["chat_history"]})

# Output the model's response
print(res)
