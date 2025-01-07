from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser


class Flower(BaseModel):
  name: str = Field(description="花的名称")
  color: list[str] = Field(description="花的颜色")

# 定义一个用于获取某种花的颜色列表的查询
flower_query = "Generate the charaters for a random flower."

# 定义一个格式不正确的输出
misformatted = "{'name': '康乃馨', 'colors': ['粉红色','白色','红色','紫色','黄色']}"


# 创建一个用于解析输出的Pydantic解析器，此处希望解析为Flower格式
flower_parser = PydanticOutputParser(pydantic_object=Flower)

# 使用Pydantic解析器解析不正确的输出
# parser.parse(misformatted) # 这行代码会出错


from langchain.output_parsers import OutputFixingParser


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

# 使用OutputFixingParser创建一个新的解析器，该解析器能够纠正格式不正确的输出
new_parser = OutputFixingParser.from_llm(parser=flower_parser, llm=llm)

# 使用新的解析器解析不正确的输出
result = new_parser.parse(misformatted) # 错误被自动修正
print(result) # 打印解析后的输出结果