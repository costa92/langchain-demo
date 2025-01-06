import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import OpenAI

API_KEY = os.getenv("OPENAI_API_KEY")
model_name="Qwen/Qwen2.5-7B-Instruct"
base_url="https://api.siliconflow.cn/v1"
llm = OpenAI(api_key=API_KEY, base_url=base_url, model=model_name)

res = llm.invoke("请给我写一句情人节红玫瑰的中文宣传语")
print(res)