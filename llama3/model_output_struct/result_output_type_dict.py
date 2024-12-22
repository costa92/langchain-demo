from langchain_openai import ChatOpenAI
from typing import Optional
from typing_extensions import TypedDict,Annotated
#  类型扩展：强烈建议导入Annotated和TypedDict导出typing_extensions而不是typing确保跨 Python 版本的行为一致。


# 针对提供用于结构化输出的本机 API 的模型with_structured_output()（如工具/函数调用或 JSON 模式）实现，并在后台利用这些功能。

# Choose the model and set up the API connection
# model_name = "llama3.2-vision:11b"
model_name="qwen2.5:7b"
llm = ChatOpenAI(
    model=model_name,
    api_key="ollama",
    base_url="http://localhost:11434/v1/",
)

# TypedDict model output
class Joke(TypedDict):
    """Joke to tell user."""
    setup: Annotated[str, ..., "The setup of the joke"]
    punchline: Annotated[str, ..., "The punchline of the joke"]
    rating: Annotated[Optional[int], None, "How funny the joke is, from 1 to 10"]

structured_llm = llm.with_structured_output(Joke)

# Invoke the model
res = structured_llm.invoke("Tell me a joke about cats")


print(res.get("setup"))
print(res.get("punchline"))
print(res.get("rating"))