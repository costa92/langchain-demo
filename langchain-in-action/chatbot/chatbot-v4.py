# ChatBot类的实现-带检索功能
import os
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

class ChatbotWithRetrieval:
    def __init__(self,dir):
        # 加载 Documents
        base_dir = dir # 文档的存放目录
        documents = []
        for file in os.listdir(base_dir):
            file_path = os.path.join(base_dir, file) 
            if file.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith('.txt'):
                loader = TextLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
            else:
                print(f"Unsupported file type: {file}")

          
        # 文本的分割
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        all_splits = text_splitter.split_documents(documents)


        # 定义嵌入模型
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        model_name = "sentence-transformers/all-mpnet-base-v2"

        self.embedding = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        # 创建向量存储
        self.vectorstore = Qdrant.from_documents(
            documents=all_splits,
            embedding=self.embedding,
            location=":memory:",  # in-memory 存储
            collection_name="documents"
        )

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

        # 初始化 Memory
        self.memory = ConversationSummaryMemory(
            memory_key="chat_history",
            llm=self.llm,
            return_messages=True
        )
      
        # 初始化 检索链
        retriever = self.vectorstore.as_retriever()

        # 初始化 对话链
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


if __name__ == "__main__":
    # 启动Chatbot
    folder = "../three-qa/OneFlower"
    bot = ChatbotWithRetrieval(folder)
    bot.chat_loop()