from dotenv import load_dotenv
load_dotenv()

import os

API_KEY = os.getenv("OPENAI_API_KEY")
model_name="Qwen/Qwen2.5-7B-Instruct"
base_url="https://api.siliconflow.cn/v1"

# load langchain_prompts 

# create model
from langchain.prompts import PromptTemplate

# 创建提示模板
prompt_template = """您是一位专业的鲜花店文案撰写员。
对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？
{format_instructions}"""

prompt = PromptTemplate(template=prompt_template)

# 通过LangChain调用模型
from langchain_openai import OpenAI

llm = OpenAI(api_key=API_KEY, base_url=base_url, model=model_name, temperature=0)


# 导入结构化输出解析器和ResponseSchema
from langchain.output_parsers import ResponseSchema,StructuredOutputParser

# 创建ResponseSchema
response_schema = [
  ResponseSchema(name="description", description="鲜花的描述文案"),
  ResponseSchema(name="reason", description="问什么要这样写这个文案")
]

# 创建StructuredOutputParser
output_parser = StructuredOutputParser.from_response_schemas(response_schema)

# 创建格式化指令
format_instructions = output_parser.get_format_instructions()


# 根据模板创建提示，同时在提示中加入输出解析器的说明
prompt = PromptTemplate.from_template(prompt_template, 
                partial_variables={"format_instructions": format_instructions}) 


# 数据准备
flowers = ["玫瑰", "百合", "康乃馨"]
prices = ["50", "30", "20"]

import pandas as pd
df = pd.DataFrame(columns=["flower", "price", "description", "reason"]) # 先声明列名

for flower, price in zip(flowers, prices):
    # 根据提示准备模型的输入
    input = prompt.format(flower_name=flower, price=price)

    # 获取模型的输出
    output = llm.invoke(input)

    # 解析模型的输出（这是一个字典结构）
    parsed_output = output_parser.parse(output)

    # 在解析后的输出中添加“flower”和“price”
    parsed_output['flower'] = flower
    parsed_output['price'] = price

    # 将解析后的输出添加到DataFrame中
    df.loc[len(df)] = parsed_output  

# 打印字典
print(df.to_dict(orient='records'))
# 保存DataFrame到CSV文件
df.to_csv("flowers_with_descriptions.csv", index=False)