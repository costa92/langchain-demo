from langchain_openai import ChatOpenAI
import os

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
from langchain.chains import ConversationChain
conversation = ConversationChain(llm=llm)
# 打印对话的模板
# print(conversation.prompt.template)
# 输出: 

# The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not kn
# ow the answer to a question, it truthfully says it does not know.
# 
# Current conversation:
# {history}
# Human: {input}
# AI:



# 第一天的对话
# 回合1
conversation("我姐姐明天要过生日，我需要一束生日花束。")
print("第一次对话后的记忆:", conversation.memory.buffer)

# 回合2
conversation("她喜欢粉红色的玫瑰，颜色是粉色。")
print("第二次对话后的记忆:", conversation.memory.buffer)


# 回合3 （第二天）
conversation("我又来了，还记得我昨天为什么需要一束花束吗？")

print("/n第三次对话后时提示:/n",conversation.prompt.template)
print("/n第三次对话后的记忆:/n", conversation.memory.buffer)