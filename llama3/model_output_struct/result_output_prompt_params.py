from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from typing import Optional
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict, Annotated
from langchain_core.prompts import ChatPromptTemplate
import os
# model_name = "qwen2.5:7b"
# api_key="ollama"
# base_url="http://localhost:11434/v1/"
model_name="Qwen/Qwen2.5-7B-Instruct"
api_key =  os.getenv("OPENAI_API_KEY")
base_url="https://api.siliconflow.cn/v1/"
llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_url,
)

# TypedDict
class Joke(TypedDict):
    """Joke to tell user."""
    setup: Annotated[str, ..., "The setup of the joke"]
    punchline: Annotated[str, ..., "The punchline of the joke"]
    rating: Annotated[Optional[int], None, "How funny the joke is, from 1 to 10"]

structured_llm = llm.with_structured_output(Joke)

examples = [
    HumanMessage("Tell me a joke about planes", name="example_user"),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[
            {
                "name": "joke",
                "args": {
                    "setup": "Why don't planes ever get tired?",
                    "punchline": "Because they have rest wings!",
                    "rating": 2,
                },
                "id": "1",
            }
        ],
    ),
    ToolMessage("", tool_call_id="1"),
    HumanMessage("Tell me another joke about planes", name="example_user"),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[
            {
                "name": "joke",
                "args": {
                    "setup": "Cargo",
                    "punchline": "Cargo 'vroom vroom', but planes go 'zoom zoom'!",
                    "rating": 10,
                },
                "id": "2",
            }
        ],
    ),
    ToolMessage("", tool_call_id="2"),
    HumanMessage("Now about caterpillars", name="example_user"),
    AIMessage(
        "",
        tool_calls=[
            {
                "name": "joke",
                "args": {
                    "setup": "Caterpillar",
                    "punchline": "Caterpillar really slow, but watch me turn into a butterfly and steal the show!",
                    "rating": 5,
                },
                "id": "3",
            }
        ],
    ),
    ToolMessage("", tool_call_id="3"),
]

system = """You are a hilarious comedian. Your specialty is knock-knock jokes. \
Return a joke which has the setup (the response to "Who's there?") \
and the final punchline (the response to "<setup> who?")."""

prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("placeholder", "{examples}"), ("human", "{input}")]
)
few_shot_structured_llm = prompt | structured_llm
res = few_shot_structured_llm.invoke({"input": "crocodiles", "examples": examples})
print(res)
