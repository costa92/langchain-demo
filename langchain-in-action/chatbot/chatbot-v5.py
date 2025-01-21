# 导入所需的库
import os
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader

# 设置OpenAI API密钥
os.environ["OPENAI_API_KEY"] = 'Your Key'  

# ChatBot类的实现
class ChatbotWithRetrieval:

    def __init__(self, dir):

        # 加载Documents
        base_dir = dir # 文档的存放目录
        documents = []
        for file in os.listdir(base_dir): 
            file_path = os.path.join(base_dir, file)
            if file.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith('.docx') or file.endswith('.doc'):
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith('.txt'):
                loader = TextLoader(file_path)
                documents.extend(loader.load())
        
        # 文本的分割
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        all_splits = text_splitter.split_documents(documents)

              # 定义嵌入模型
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        model_name = "sentence-transformers/all-mpnet-base-v2"

        embedding = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        
        # 向量数据库
        self.vectorstore = Qdrant.from_documents(
            documents=all_splits, # 以分块的文档
            embedding=embedding, # 用OpenAI的Embedding Model做嵌入
            location=":memory:",  # in-memory 存储
            collection_name="my_documents",) # 指定collection_name
        
        # 初始化LLM
        # 初始化 llM
        API_KEY = os.getenv("OPENAI_API_KEY")
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        base_url = "https://api.siliconflow.cn/v1"
        
        # 修复 LLM 初始化
        self.llm = ChatOpenAI(
            openai_api_key=API_KEY, 
            model=model_name, 
            openai_api_base=base_url
        )
        
        # 初始化Memory
        self.memory = ConversationSummaryMemory(
            llm=self.llm, 
            memory_key="chat_history", 
            return_messages=True
            )
        
        # 设置Retrieval Chain
        retriever = self.vectorstore.as_retriever()
        self.qa = ConversationalRetrievalChain.from_llm(
            self.llm, 
            retriever=retriever, 
            memory=self.memory
            )

    def chat_loop(self):
        print("Chatbot 已启动! 输入'exit'来退出程序。")
        while True:
            user_input = input("你: ")
            if user_input.lower() == 'exit':
                print("再见!")
                break
            # 调用 Retrieval Chain  
            response = self.qa(user_input)
            print(f"Chatbot: {response['answer']}")

# Streamlit界面的创建
def main():
    st.title("易速鲜花聊天客服")
    folder = "../three-qa/OneFlower"  # 替换为你的文档目录路径
    # Check if the 'bot' attribute exists in the session state
    if "bot" not in st.session_state:
        st.session_state.bot = ChatbotWithRetrieval(folder)

    user_input = st.text_input("请输入你的问题：")
    
    if user_input:
        response = st.session_state.bot.qa(user_input)
        st.write(f"Chatbot: {response['answer']}")

if __name__ == "__main__":
    main()
