from langchain_openai import ChatOpenAI
import os

# 初始化语言模型
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 OPENAI_API_KEY")

model_name = "Qwen/Qwen2.5-7B-Instruct"
base_url = "https://api.siliconflow.cn/v1"

#初始化大语言模型
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=base_url,
    model_name=model_name,
    # n=3  # 生成 3 个候选结果
    temperature=0.5,
)

from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 初始化 ConversationalRetrievalChain
conv_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever="",  # 传入 retriever
    memory=memory
)

# 打印对话的模板（如果有的话）
# 注意：ConversationalRetrievalChain 可能没有直接的 prompt.template 属性
# 你可以通过 conv_chain.combine_docs_chain.llm_chain.prompt.template 来访问模板
print(conv_chain.combine_docs_chain.llm_chain.prompt.template)