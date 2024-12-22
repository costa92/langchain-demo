from typing import Union,Optional
from langchain_openai import ChatOpenAI
from pydantic import BaseModel,Field




# model_name = "qwen2.5:7b"
# api_key="ollama"
# base_url="http://localhost:11434/v1/"
model_name="Qwen/Qwen2.5-7B-Instruct"
import os
api_key =  os.getenv("OPENAI_API_KEY")
base_url="https://api.siliconflow.cn/v1/"
llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_url,
)


class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )


class ConversationalResponse(BaseModel):
    """Respond in a conversational manner. Be kind and helpful."""

    response: str = Field(description="A conversational response to the user's query")


class FinalResponse(BaseModel):
    final_output: Union[Joke, ConversationalResponse]
 

structured_llm = llm.with_structured_output(FinalResponse)

# res = structured_llm.invoke("Tell me a joke about cats")
# print(res)

raw_response = structured_llm.invoke("How are you today?")
print("Raw response:", raw_response)
