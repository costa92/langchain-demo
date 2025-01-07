# 使用 LLMChain 和 ConversationBufferMemory 可以替代废弃的 ConversationChain。

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
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



# 定义对话模板
template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI:"""

from langchain.prompts import PromptTemplate

# 创建 PromptTemplate
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template
)


# 初始化记忆模块
memory = ConversationBufferMemory(memory_key="history")

from langchain.chains import LLMChain

# 初始化 LLMChain
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True  # 打印详细日志
)

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