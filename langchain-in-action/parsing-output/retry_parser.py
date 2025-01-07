# 定义一个模板字符串，这个模板将用于生成提问
template = """Based on the user question,provide an Action and Action Input for what step should be taken.
{format_instructions}
Question: {question}
Response:"""


# 定义一个 Pydantic的数据格式，这个格式描述一个“行动”类及其属性
from pydantic import BaseModel, Field

class Action(BaseModel):
    action: str = Field(..., description="action to take")
    action_input: str = Field(..., description="input to the action")


# 使用 Pydantic 格式Action来初始化一个输出解析器
from langchain.output_parsers import PydanticOutputParser
parser = PydanticOutputParser(pydantic_object=Action)


# 定义一个提示模板，这个模板将用于生成提问
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="Answer the user query. \n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}    
)

prompt_value = prompt.format_prompt(query="what are the colors of Orchid?")

# 定义一个错误格式的字符串
bad_response = '{"action":"search"}'
# parser.parse(bad_response) # 如果直接解析，它会引发一个错误



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


# 尝试用OutputFixingParser来解决这个问题
from langchain.output_parsers import OutputFixingParser

fix_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
parser_result = fix_parser.parse(bad_response)

print("OutputFixingParser的结果:",parser_result)


# 初始化RetryWithErrorOutputParser，它会尝试再次提问来得到一个正确的输出
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain_openai import OpenAI

retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=parser,
    llm=OpenAI(temperature=0,api_key=API_KEY,base_url=base_url,model_name=model_name),
    max_retries=3,
)

parse_result = retry_parser.parse_with_prompt(bad_response, prompt_value)
print('RetryWithErrorOutputParser的parse结果:',parse_result)
