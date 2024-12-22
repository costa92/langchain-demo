from typing_extensions import Annotated, TypedDict
from typing import Optional
from langchain_openai import ChatOpenAI


model_name="qwen2.5:7b"
llm = ChatOpenAI(
    model=model_name,
    api_key="ollama",
    base_url="http://localhost:11434/v1/",
)

# TypedDict
class Joke(TypedDict):
    """Joke to tell user."""

    setup: Annotated[str, ..., "The setup of the joke"]
    punchline: Annotated[str, ..., "The punchline of the joke"]
    rating: Annotated[Optional[int], None, "How funny the joke is, from 1 to 10"]


structured_llm = llm.with_structured_output(Joke)

for chunk in structured_llm.stream("Tell me a joke about cats"):
    print(chunk)