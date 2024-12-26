# Bind method allows invoking a Runnable with constant arguments that are not part of the output of any preceding Runnable. One everyday use case for this method is attaching OpenAI tools to the chain:

# zh: Bind方法允许使用不是任何前面Runnable输出的常量参数调用Runnable。此方法的一个日常用例是将OpenAI工具附加到链上：
from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_openai import ChatOpenAI

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Compute the arithmetic equation",  # zh: 计算算术方程
            "parameters": {
                "type": "object",
                "properties": {
                    "equation": {
                        "type": "string",
                        "description": "Arithmetic equation expressed in natural language", # zh: 用自然语言表达的算术方程
                    },
                    "solution": {
                         "type": "string",
                         "description": "The solution to the equation", # zh: 方程的解
                    },
                },

                "required": ["equation", "solution"],
            },
        },
    }
]


openai_parser = JsonOutputToolsParser()

model_name = "qwen2.5:7b"
api_key = "ollama"
base_url = "http://localhost:11434/v1/"

model_with_tools = ChatOpenAI(
  model=model_name, 
  api_key=api_key,
  base_url=base_url,
  temperature=0).bind(tools=tools) 


calculator_chain = model_with_tools | openai_parser


# res = calculator_chain.invoke("1 plus 8 and then take the sum to power 2") 
#  1 plus 8 and then take the sum to power 2  :zh 1加8，然后将总和提高到2次方
# print(res)
# 输出：{'equation': '1 plus 8 and then take the sum to power 2', 'solution': '81'}

res = calculator_chain.invoke("计算10加10的和，然后将总和提高到2次方") 
#  计算10加10的和，然后将总和提高到2次方 :zh 10 plus 10 and then take the sum to power 2
print(res)