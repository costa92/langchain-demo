
from langchain_core.prompts import PromptTemplate
# 第一步 创建提示

# 原始字符串模板
template = "{flower}的话语是？"
# 创建PromptTemplate对象
prompt_temp = PromptTemplate.from_template(template)

# 打印生成的提示
prompt = prompt_temp.format(flower='玫瑰')


# 第二步 调用模型
import os

API_KEY = os.getenv("OPENAI_API_KEY")
model_name = "Qwen/Qwen2.5-7B-Instruct"
base_url = "https://api.siliconflow.cn/v1"

# 创建模型实例
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(
#     api_key=API_KEY,
#     base_url=base_url,
#     model_name=model_name,
    
# )

from langchain_openai import OpenAI

llm = OpenAI(
    api_key=API_KEY,
    base_url=base_url,
    model_name=model_name,
)

res = llm.invoke(prompt)
print(res)