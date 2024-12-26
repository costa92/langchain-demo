# configurable_alternatives is a method that defines alternatives for a Runnable to be set in runtime. For example, we can switch between different LLM providers:
# zh: configurable_alternatives是一个方法，用于定义Runnable的备选方案，以在运行时设置。例如，我们可以在不同的LLM提供商之间切换：
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import (
   ConfigurableField
)

from langchain_core.output_parsers import StrOutputParser



model_name = "llama3.2-vision:11b"
api_key = "ollama"
base_url = "http://localhost:11434/v1/"

llm = ChatOpenAI(  api_key=api_key,
  base_url=base_url,
  model=model_name,
  temperature=0).configurable_alternatives(
    # We can then use this id to configure this field in the runnable
    # zh: 然后我们可以使用此id在runnable中配置此字段
    ConfigurableField(id="llm"),
    # This sets a default_key.
    # zh: 这将设置一个默认键。
    default_key="openai",
    # Adding an alternative
    # zh: 添加一个备选方案
    anthropic=ChatAnthropic(model="gpt-3.5"),
    # You can add more configuration options here
  ) | StrOutputParser()

res = llm.with_config(llm="openai").invoke("Who is Lech Walesa? Answer in 10 words.")

print(res)
# 输出：Lech Walesa is a Polish politician and human rights activist.
# zh: Lech Walesa是一位波兰政治家和人权活动家。
