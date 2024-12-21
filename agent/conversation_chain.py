
# 导入所需的库
from langchain_openai import OpenAI
from langchain.chains.conversation.base import ConversationChain

model_name = "llama3"
# 初始化大语言模型
llm = OpenAI(
  temperature=0.5,
  model=model_name, 
  api_key="ollama", 
  base_url="http://localhost:11434/v1/",
)

# # 初始化对话链
# conv_chain = ConversationChain(llm=llm)

# # 打印ConversationChain中的内置提示模板，RunnableWithMessageHistory 
# # 注意： ConversationChain 已经在 LangChain 0.2.7 中被弃用，但你仍然可以使用 来实现类似的功能，同时要打印内置的提示模板，必须访问 PromptTemplate。不过，直接打印 ConversationChain 的提示模板可能会存在一些挑战，因为它是一个高度封装的对象。
# print(conv_chain.prompt.template)


from langchain_core.runnables.history import RunnableWithMessageHistory
from collections import deque
from langchain.prompts import PromptTemplate

# 定义一个简单的历史记录管理函数
message_history = deque(maxlen=10)  # 保持最近的10条消息

def get_session_history():
    return list(message_history)  # 返回当前会话历史

# 使用RunnableWithMessageHistory
conv_chain = RunnableWithMessageHistory(runnable=llm, get_session_history=get_session_history)

# 假设我们使用的是一个简单的提示模板
prompt_template = PromptTemplate.from_template("User: {input}\nBot:")

# 打印提示模板
print(prompt_template.template)