from langchain_core.prompts import PromptTemplate
# defined 
# 原始字符串模板
template = "{flower}的话语是？"

# 创建PromptTemplate对象
prompt_temp = PromptTemplate.from_template(template)
# 第二步 调用模型
import os

API_KEY = os.getenv("OPENAI_API_KEY")
model_name = "Qwen/Qwen2.5-7B-Instruct"
# model_name ="deepseek-ai/DeepSeek-V2.5"
base_url = "https://api.siliconflow.cn/v1"

# model_name="deepseek-chat"
# API_KEY = os.getenv("deepseek_api_key")
# # base_url = os.getenv("deepseek_api_url")
# base_url="https://api.deepseek.com/beta"


# 创建模型实例
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
  temperature=0.2,
  api_key=API_KEY,
  model_name=model_name,
  base_url=base_url,
)

llm_chain = prompt_temp | llm

res = llm_chain.invoke("玫瑰")

print(res.content)

