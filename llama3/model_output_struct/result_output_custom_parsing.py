#  自定义解析
import json
import re
from typing import List

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


import json
import re
from typing import List

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# 定义一个Pydantic模型
class Person(BaseModel):
    """Information about a person."""

    name: str = Field(..., description="The name of the person")
    height_in_meters: float = Field(
        ..., description="The height of the person expressed in meters."
    )

# 定义一个Pydantic模型
class People(BaseModel):
    """Identifying information about all people in a text."""

    people: List[Person]



# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user query. Output your answer as JSON that  "
            "matches the given schema: \`\`\`json\n{schema}\n\`\`\`. "
            "Make sure to wrap the answer in \`\`\`json and \`\`\` tags",
        ),
        ("human", "{query}"),
    ]
).partial(schema=People.model_json_schema())


# Custom parser
def extract_json(message: AIMessage) -> List[dict]:
    """Extracts JSON content from a string where JSON is embedded between \`\`\`json and \`\`\` tags.

    Parameters:
        text (str): The text containing the JSON content.

    Returns:
        list: A list of extracted JSON strings.
    """
    text = message.content
    # Define the regular expression pattern to match JSON blocks
    pattern = r"\`\`\`json(.*?)\`\`\`"

    # Find all non-overlapping matches of the pattern in the string
    matches = re.findall(pattern, text, re.DOTALL)

    # Return the list of matched JSON strings, stripping any leading or trailing whitespace
    try:
        return [json.loads(match.strip()) for match in matches]
    except Exception:
        raise ValueError(f"Failed to parse: {message}")
    
query = "Anna is 23 years old and she is 6 feet tall"

#  打印提示
print(prompt.format_prompt(query=query).to_string())


# model_name = "llama3.2-vision:11b"
model_name="qwen2.5:7b"
llm = ChatOpenAI(
    model=model_name,
    api_key="ollama",
    base_url="http://localhost:11434/v1/",
)

chain = prompt | llm | extract_json

res = chain.invoke({"query": query})
print(res)