import os
from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()

from langchain.prompts import PromptTemplate

# 创建原始模板
# template = """你是一位专业的鲜花店文案撰写员。 \n
# 对于售价为 {price} 元的 {flower} 花束，请撰写一段文案，描述其特点和卖点。 \n
# """

template = """您是一位专业的鲜花店文案撰写员。\n
对于售价为 {price} 元的 {flower} ，您能提供一个吸引人的简短描述吗？
"""

# 创建一个PromptTemplate对象
prompt = PromptTemplate.from_template(template)

# # 使用模板生成提示
# result = prompt.format(price=50, flower="玫瑰")


from langchain_openai import ChatOpenAI

API_KEY = os.getenv("OPENAI_API_KEY")
model_name="Qwen/Qwen2.5-7B-Instruct"
base_url="https://api.siliconflow.cn/v1"


llm = ChatOpenAI(api_key=API_KEY, base_url=base_url, model=model_name, temperature=0)

input = prompt.format(price=50, flower="玫瑰")
# print(input)


response = llm.invoke(input)
print(response.content)
