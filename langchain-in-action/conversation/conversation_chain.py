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

# 初始化对话链 
# 注意 ConversationChain 在 LangChain 0.2.7 已经废弃
from langchain.chains import ConversationChain
conv_chain = ConversationChain(llm=llm)

# 打印对话的模板
print(conv_chain.prompt.template)



