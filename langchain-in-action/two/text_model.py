from dotenv import load_dotenv
import os

from openai import OpenAI
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
model_name="Qwen/Qwen2.5-7B-Instruct"
base_url="https://api.siliconflow.cn/v1"

client = OpenAI(api_key=API_KEY, base_url=base_url)

response = client.completions.create(
  model=model_name,
  temperature=0.5,
  max_tokens=100,
  prompt="请给我的花店起个名")

print(response.choices[0])
print("--------------------------------")
print(response.choices[0].text.strip())