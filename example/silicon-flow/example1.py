from openai import OpenAI
import os

api_key =  os.getenv("OPENAI_API_KEY")

def main():   
  client = OpenAI(api_key=api_key,base_url="https://api.siliconflow.cn/v1/")
  response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct", 
    messages=[
      {
        "role": "user",
        "content": "You are a helpful assistant."
      }
      ],
      stream=True
   )
  
  for chunk in response:
    print(chunk.choices[0].delta.content, end="")


if __name__ == "__main__":
  main()