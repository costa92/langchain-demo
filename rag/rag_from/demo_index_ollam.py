
import os
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

LANGCHAIN_ENDPOINT=os.environ['LANGCHAIN_ENDPOINT']
LANGCHAIN_API_KEY=os.environ['LANGCHAIN_API_KEY']

# 初始化 ChatOpenAI 模型
# chat = ChatOpenAI(model="llama3.2-vision:11b", api_key="ollama", base_url="http://localhost:11434/v1/")
chat = ChatOpenAI(model="llama3:8b", api_key="ollama", base_url="http://localhost:11434/v1/")



template='''
        你的名字是喵喵,当回答问题的时候,你都会在开头加上'喵喵~',然后再回答{question}
    '''
prompt=PromptTemplate(
        template=template,
        input_variables=["question"]#这个question就是用户输入的内容,这行代码不可缺少
)
chain = LLMChain(#将llm与prompt联系起来
        llm=chat,
        prompt=prompt
        )
question='你是谁'

res=chain.invoke(question)#运行
    
print(res['text'])#打印结果
