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

from typing_extensions import Annotated, TypedDict


# TypedDict
class Joke(TypedDict):
    """Joke to tell user."""

    setup: Annotated[str, ..., "The setup of the joke"]
    punchline: Annotated[str, ..., "The punchline of the joke"]
    rating: Annotated[Optional[int], None, "How funny the joke is, from 1 to 10"]

structured_llm = llm.with_structured_output(Joke, include_raw=True)


res = structured_llm.invoke("Tell me a joke about cats")



print(res)

# {'setup': 'Why did the cat join the yoga class in the park instead of the usual litter box spot nearby? ', 'punchline': 'To stretch its claws and learn to be more flexible, of course. '}