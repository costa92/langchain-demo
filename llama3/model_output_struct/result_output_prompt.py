from typing import Optional
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict,Annotated
from langchain_core.prompts import ChatPromptTemplate
# 定义提示词
# Few-shot  structured output
# doc: https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html

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


system = """You are a hilarious comedian. Your specialty is knock-knock jokes. \
Return a joke which has the setup (the response to "Who's there?") and the final punchline (the response to "<setup> who?").

Here are some examples of jokes:

example_user: Tell me a joke about planes
example_assistant: {{"setup": "Why don't planes ever get tired?", "punchline": "Because they have rest wings!", "rating": 2}}

example_user: Tell me another joke about planes
example_assistant: {{"setup": "Cargo", "punchline": "Cargo 'vroom vroom', but planes go 'zoom zoom'!", "rating": 10}}

example_user: Now about caterpillars
example_assistant: {{"setup": "Caterpillar", "punchline": "Caterpillar really slow, but watch me turn into a butterfly and steal the show!", "rating": 5}}"""

prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{input}")])

few_shot_structured_llm = prompt | structured_llm
res = few_shot_structured_llm.invoke("what's something funny about woodpeckers")

print(res)