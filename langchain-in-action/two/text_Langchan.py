

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAI

API_KEY = os.getenv("OPENAI_API_KEY")
model_name="Qwen/Qwen2.5-7B-Instruct"
base_url="https://api.siliconflow.cn/v1"

llm = OpenAI(api_key=API_KEY, base_url=base_url, model=model_name,temperature=0.2)

response = llm.invoke("请给我的花店起个名,直接输出花店名字")
print(response)