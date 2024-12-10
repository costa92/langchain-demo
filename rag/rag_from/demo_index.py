import os
from langchain_openai import ChatOpenAI

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())


api_key =  os.getenv("OPENAI_API_KEY")
base_url="https://api.siliconflow.cn/v1/"

#设置langSmith的环境变量,是统计
LANGCHAIN_TRACING_V2=os.environ['LANGCHAIN_TRACING_V2']
LANGCHAIN_ENDPOINT=os.environ['LANGCHAIN_ENDPOINT']
LANGCHAIN_API_KEY=os.environ['LANGCHAIN_API_KEY']

print(LANGCHAIN_API_KEY)
print(LANGCHAIN_ENDPOINT)
# Initialize the ChatOpenAI instance
llm = ChatOpenAI(
  api_key=api_key,
  base_url=base_url, 
  model="Qwen/Qwen2.5-7B-Instruct",
)

# Invoke the model with a message
# response = llm.invoke("Hello, world!")
# print(response)

print("第一次对话:",llm.invoke("你是一只小狗,只会汪汪叫"),"\n\n第二次对话:",llm.invoke("你是一只小狗嘛"))
