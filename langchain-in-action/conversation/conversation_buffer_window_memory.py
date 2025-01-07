# 设置OpenAI API密钥
import os
# 导入所需的库
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

# 初始化语言模型
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 OPENAI_API_KEY")
model_name = "Qwen/Qwen2.5-7B-Instruct"
base_url = "https://api.siliconflow.cn/v1"

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=base_url,
    model_name=model_name,
    temperature=0.5,
)


# 初始化对话链
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferWindowMemory(k=1)
)

# 第一天的对话
# 回合1
result = conversation("我姐姐明天要过生日，我需要一束生日花束。")
print(result)
# 回合2
result = conversation("她喜欢粉色玫瑰，颜色是粉色的。")
# print("\n第二次对话后的记忆:\n", conversation.memory.buffer)
print(result)

# 第二天的对话
# 回合3
result = conversation("我又来了，还记得我昨天为什么要来买花吗？")
print(result)