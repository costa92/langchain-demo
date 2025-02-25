
# ------Part 1
import json
from dotenv import load_dotenv
load_dotenv()

import os

API_KEY = os.getenv("OPENAI_API_KEY")
model_name = "Qwen/Qwen2.5-7B-Instruct"
base_url = "https://api.siliconflow.cn/v1"

# 创建模型实例
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=base_url,
    model_name=model_name,
)

# ------Part 2
# 创建一个空的DataFrame用于存储结果
import pandas as pd

df = pd.DataFrame(columns=["flower_type", "price", "description", "reason"])

# 数据准备
flowers = ["玫瑰", "百合", "康乃馨"]
prices = ["50", "30", "20"]

# 定义我们想要接收的数据格式
from pydantic import BaseModel, Field, ValidationError

class FlowerDescription(BaseModel):
    flower_type: str = Field(description="鲜花的种类")
    price: int = Field(description="鲜花的价格")
    description: str = Field(description="鲜花的描述文案")
    reason: str = Field(description="为什么要这样写这个文案")

# ------Part 3
from langchain.output_parsers import PydanticOutputParser
output_parser = PydanticOutputParser(pydantic_object=FlowerDescription)

# 获取输出格式指示
format_instructions = output_parser.get_format_instructions()
# 打印提示
print("输出格式：", format_instructions)

# ------Part 4
# 创建提示模板
from langchain_core.prompts import PromptTemplate

prompt_template = """您是一位专业的鲜花店文案撰写员。
对于售价为 {price} 元的 {flower} ，您能提供一个吸引人的简短中文描述吗？
{format_instructions}"""

# 根据模板创建提示，同时在提示中加入输出解析器的说明
prompt = PromptTemplate.from_template(prompt_template,
                                      partial_variables={"format_instructions": format_instructions})

# 打印提示
print("提示：", prompt)

# ------Part 5
for flower, price in zip(flowers, prices):
    # 根据提示准备模型的输入
    input = prompt.format(flower=flower, price=price)
    # 打印提示
    print("提示：", input)

    # 获取模型的输出
    output = llm.invoke(input)
    # print("输出：", output.content)

    try:
        # 解析模型的输出
        parsed_output = output_parser.parse(output.content)
        parsed_output_dict = parsed_output.model_dump()  # 将Pydantic格式转换为字典

        # 将解析后的输出添加到DataFrame中
        df.loc[len(df)] = parsed_output_dict
    except ValidationError as e:
        print(f"Validation error: {e}")

# 打印字典
# print("输出的数据：", df.to_dict(orient='records'))

# 打印JSON
print(json.dumps(df.to_dict(orient='records'), ensure_ascii=False))
# [{"flower_type": "玫瑰", "price": 50, "description": "精选优质玫瑰，每一朵都承载着最真挚的情感。花瓣娇艳欲滴，香气袭人，是表达爱意与祝福的不二之选。", "reason": "这样的描述突出了玫瑰的品质和情感价值，旨在打动顾客的心，提高购买欲望。"}, {"flower_type": "百合", "price": 30, "description": "精选优质百合，花姿优雅，纯洁高雅，是表达纯洁爱情和友谊的佳选。", "reason": "通过突出百合的品质和象征意义，吸引注重品质和情感表达的顾客购买。"}, {"flower_type": "康乃馨", "price": 20, "description": "精选优质康乃馨，花束散发自然清香，每一朵都蕴含着温柔的情感。赠予挚爱，传递心底的温暖。", "reason": "通过强调康乃馨的品质和情感价值，吸引顾客购买，同时简洁明了地传达了价格信息。"}]