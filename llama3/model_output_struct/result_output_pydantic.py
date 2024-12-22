from langchain_openai import ChatOpenAI
from typing import Optional
from pydantic import BaseModel,Field

# 针对提供用于结构化输出的本机 API 的模型with_structured_output()（如工具/函数调用或 JSON 模式）实现，并在后台利用这些功能。

# Choose the model and set up the API connection
# model_name = "llama3.2-vision:11b"
model_name="qwen2.5:7b"
llm = ChatOpenAI(
    model=model_name,
    api_key="ollama",
    base_url="http://localhost:11434/v1/",
)

# Pydamtic model output
class Joke(BaseModel):
    """Joke to tell user."""
    setup: str = Field(description="The setup of the joke.")
    punchline: str = Field(description="The punchline of the joke.")
    rating: Optional[int] = Field(
        default=None,
        description="The rating of the joke, from 1 to 10.",
    )

structured_llm = llm.with_structured_output(Joke)

# Invoke the model
res = structured_llm.invoke("Tell me a joke about cats")


print(res.setup)
print(res.punchline)
print(res.rating)
