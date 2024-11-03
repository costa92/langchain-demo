from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# template 起到一个规定回答问题的格式模板的作用
template = """Question: {question}
Answer: 请用中文回答我的问题."""
prompt = ChatPromptTemplate.from_template(template)
model = OllamaLLM(model="llama3")
chain = prompt | model

while True:
    question = input("请输入您的问题：")
    if question == "结束":
        print("-"*30+"感谢您的使用"+"-"*30)
        break
    answer = chain.invoke({"question": question})
    print(answer)
    print("-"*30)
