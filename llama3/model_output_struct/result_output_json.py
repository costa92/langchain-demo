# 我们可以传入JSON Schema字典。这不需要导入或类，并且可以非常清楚地说明每个参数的记录方式，但代价是有点冗长。
from langchain_openai import ChatOpenAI

# model_name = "llama3.2-vision:11b"
model_name="qwen2.5:7b"
llm = ChatOpenAI(
    model=model_name,
    api_key="ollama",
    base_url="http://localhost:11434/v1/",
)

json_schema = {
    "title": "joke",
    "description": "Joke to tell user.",
    "type": "object",
    "properties": {
        "setup": {
            "type": "string",
            "description": "The setup of the joke",
        },
        "punchline": {
            "type": "string",
            "description": "The punchline to the joke",
        },
        "rating": {
            "type": "integer",
            "description": "How funny the joke is, from 1 to 10",
            "default": None,
        },
    },
    "required": ["setup", "punchline"],
}
structured_llm = llm.with_structured_output(json_schema)

res = structured_llm.invoke("Tell me a joke about cats")

print(res.get("setup"))
print(res.get("punchline"))
print(res.get("rating"))