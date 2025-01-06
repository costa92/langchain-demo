from langchain_openai import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)

import os
API_KEY = os.getenv("OPENAI_API_KEY")
model_name="Qwen/Qwen2.5-7B-Instruct"
base_url="https://api.siliconflow.cn/v1"  

llm = ChatOpenAI(api_key=API_KEY, base_url=base_url, model=model_name, temperature=0.2)

messages = [
    SystemMessage(content="你是一个很棒的智能助手"),  # 系统消息  
    HumanMessage(content="请给我的花店起个名")  # 用户消息
]

response = llm.invoke(messages)
print(response)