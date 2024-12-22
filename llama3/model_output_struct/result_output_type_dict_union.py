from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from typing import Optional,Union
from typing_extensions import TypedDict,Annotated
#  类型扩展：强烈建议导入Annotated和TypedDict导出typing_extensions而不是typing确保跨 Python 版本的行为一致。


# 针对提供用于结构化输出的本机 API 的模型with_structured_output()（如工具/函数调用或 JSON 模式）实现，并在后台利用这些功能。

# Choose the model and set up the API connection
# model_name = "llama3.2-vision:11b"
model_name="qwen2.5:7b"
# llm = ChatOpenAI(
#     model=model_name,
#     api_key="ollama",
#     base_url="http://localhost:11434/v1/",
# )

llm = ChatOllama(
    model=model_name,
    temperature=0,
    # other params...
)

# TypedDict model output
class Joke(TypedDict):
    """Joke to tell user."""
    setup: Annotated[str, ..., "The setup of the joke"]
    punchline: Annotated[str, ..., "The punchline of the joke"]
    rating: Annotated[Optional[int], None, "How funny the joke is, from 1 to 10"]



class ConversationalResponse(TypedDict):
    """Respond in a conversational manner. Be kind and helpful."""

    response: Annotated[str, ..., "A conversational response to the user's query"]


class FinalResponse(TypedDict):
    final_output: Union[Joke, ConversationalResponse]


structured_llm = llm.with_structured_output(FinalResponse)

# Invoke the model
res = structured_llm.invoke("Tell me a joke about cats")


print(res)