from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("讲一个关于 {topic} 的笑话")
from langchain_openai import ChatOpenAI


# model_name = "qwen2.5:7b"
model_name = "llama3.2-vision:11b"
llm = ChatOpenAI(
    model=model_name,
    api_key="ollama",
    base_url="http://localhost:11434/v1/",
)


output_parser = StrOutputParser()

chain = prompt | llm | output_parser

res = chain.invoke({"topic": "冰淇淋"})
print(res)
#  pip install grandalf
# chain.get_graph().print_ascii()

#      +-------------+       
#      | PromptInput |       
#      +-------------+       
#             *              
#             *              
#             *              
#   +--------------------+   
#   | ChatPromptTemplate |   
#   +--------------------+   
#             *              
#             *              
#             *              
#       +------------+       
#       | ChatOpenAI |       
#       +------------+       
#             *              
#             *              
#             *              
#    +-----------------+     
#    | StrOutputParser |     
#    +-----------------+     
#             *              
#             *              
#             *              
# +-----------------------+  
# | StrOutputParserOutput |  
# +-----------------------+  

# chain.output_schema().schema()
print(chain.get_output_jsonschema())
print(chain.get_output_schema())
print(chain.get_input_jsonschema()) 