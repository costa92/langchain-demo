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


structured_llm = llm.with_structured_output(None, method="json_mode")

res = structured_llm.invoke(
    "Tell me a joke about cats, respond in JSON with `setup` and `punchline` keys"
)

print(res)

# {'setup': 'Why did the cat join the yoga class in the park instead of the usual litter box spot nearby? ', 'punchline': 'To stretch its claws and learn to be more flexible, of course. '}