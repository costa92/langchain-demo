from typing import List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


model_name="Qwen/Qwen2.5-7B-Instruct"
import os
api_key =  os.getenv("OPENAI_API_KEY")
base_url="https://api.siliconflow.cn/v1/"
llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_url,
)

# 定义一个Pydantic模型
class Person(BaseModel):
    """Information about a person."""

    name: str = Field(..., description="The name of the person")
    height_in_meters: float = Field(
        ..., description="The height of the person expressed in meters."
    )


class People(BaseModel):
    """Identifying information about all people in a text."""

    people: List[Person]


# Set up a parser
parser = PydanticOutputParser(pydantic_object=People)

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user query. Wrap the output in `json` tags\n{format_instructions}",
        ),
        ("human", "{query}"),
    ]
).partial(format_instructions=parser.get_format_instructions())



query = "Anna is 23 years old and she is 6 feet tall"

# print(prompt.invoke({"query": query}).to_string())

chain = prompt | llm | parser


res = chain.invoke({"query": query})
print(res)

