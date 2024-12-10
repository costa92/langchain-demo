
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnableSequence

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

LANGCHAIN_ENDPOINT=os.environ['LANGCHAIN_ENDPOINT']
LANGCHAIN_API_KEY=os.environ['LANGCHAIN_API_KEY']

# 初始化 ChatOpenAI 模型
# chat = ChatOpenAI(model="llama3.2-vision:11b", api_key="ollama", base_url="http://localhost:11434/v1/")
chat = ChatOpenAI(model="llama3:8b", api_key="ollama", base_url="http://localhost:11434/v1/")

# ===================================================== 串联提示词 ==============================
template='''
        你的名字是喵喵,当回答问题的时候使用中文,你都会在开头加上'喵喵~',然后再回答{question}
    '''

#这个question就是用户输入的内容,这行代码不可缺少
prompt=PromptTemplate(
        template=template,
        input_variables=["question"]
)

# #使用链将它们串联起来
# chain = LLMChain(
#         llm=chat,
#         prompt=prompt
#         )

# 使用 RunnableSequence 替代 LLMChain  
chain = prompt | chat

question='你是谁'

res=chain.invoke({"question": question})#运行
    
print(res.content)#打印结果 

# ===================================================== 串联提示词 ============================== 