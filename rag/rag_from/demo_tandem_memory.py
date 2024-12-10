# 参考文档：https://python.langchain.com.cn/docs/modules/memory/



import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnableSequence
from langchain.memory import (ConversationBufferMemory,ConversationBufferWindowMemory,ConversationSummaryMemory)


from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

LANGCHAIN_ENDPOINT=os.environ['LANGCHAIN_ENDPOINT']
LANGCHAIN_API_KEY=os.environ['LANGCHAIN_API_KEY']

# 初始化 ChatOpenAI 模型
# chat = ChatOpenAI(model="llama3.2-vision:11b", api_key="ollama", base_url="http://localhost:11434/v1/")
llm = ChatOpenAI(model="llama3:8b", api_key="ollama", base_url="http://localhost:11434/v1/")

template='''你是一个成熟的大姐姐，回答内容使用中文，你的任务是用温柔的语气回答人类的问题。
        {chat_memory}
       human:{question}
    '''
prompt=PromptTemplate(
        template=template,
        input_variables=["question"]
)
#ConversationBufferMemory
# memory = ConversationBufferMemory(memory_key="chat_memory",return_messages=False)

#  ConversationBufferWindowMemory会话缓冲窗口
# memory = ConversationBufferWindowMemory(memory_key="chat_memory",return_messages=False,k=2)


# ConversationSummaryMemory 会话摘要记忆
memory = ConversationSummaryMemory(llm=llm, memory_key="chat_memory",return_messages=False)

chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=True
        )

chain.invoke("我喜欢美食,我最喜欢的美食是清蒸鲈鱼")
chain.invoke("你是谁?")
chain.invoke("今天的天气真好啊")
res = chain.invoke("我最开始跟你聊的什么呢？")
print(res['text'])

# https://juejin.cn/post/7376484708547149835
# https://python.langchain.com/docs/versions/migrating_memory/conversation_buffer_memory/