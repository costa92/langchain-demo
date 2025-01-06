from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量


from openai import OpenAI
import os
API_KEY = os.getenv("OPENAI_API_KEY")
model_name="Qwen/Qwen2.5-7B-Instruct"
base_url="https://api.siliconflow.cn/v1"

client = OpenAI(api_key=API_KEY, base_url=base_url)

response = client.chat.completions.create(  
  model=model_name,
  messages=[
        {"role": "system", "content": "You are a creative AI."},
        {"role": "user", "content": "请给我的花店起个名,请直接输出名字"},
    ],
  temperature=0.8,
  max_tokens=60
)

print(response.choices[0].message.content)