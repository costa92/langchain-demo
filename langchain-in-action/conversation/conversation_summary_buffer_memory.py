from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
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

# 创建 PromptTemplate
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template
)

# 自定义 token 计数方法
def custom_token_counter(text: str) -> int:
    """
    自定义 token 计数方法。
    这里简单地按空格分割文本并计算单词数量。
    你可以根据实际需求实现更精确的 token 计数逻辑。
    """
    return len(text.split())

# 初始化 ConversationSummaryBufferMemory
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=300,  # 设置总结的最大 token 数
    token_counter=custom_token_counter  # 使用自定义的 token 计数方法
)

# 定义 RunnableSequence
def get_history(memory):
    return memory.load_memory_variables({})["history"]

conversation = RunnableSequence(
    {
        "history": RunnablePassthrough.assign(history=lambda _: get_history(memory)),
        "input": RunnablePassthrough(),
    }
    | prompt
    | llm
)

# 第一天的对话
# 回合1
response1 = conversation.invoke({"input": "我姐姐明天要过生日，我需要一束生日花束。"})
print("AI 回复:", response1.content)
memory.save_context({"input": "我姐姐明天要过生日，我需要一束生日花束。"}, {"output": response1.content})

# 回合2
response2 = conversation.invoke({"input": "她喜欢粉色玫瑰，颜色是粉色的。"})
print("AI 回复:", response2.content)
memory.save_context({"input": "她喜欢粉色玫瑰，颜色是粉色的。"}, {"output": response2.content})

# 第二天的对话
# 回合3
response3 = conversation.invoke({"input": "我又来了，还记得我昨天为什么要来买花吗？"})
print("AI 回复:", response3.content)