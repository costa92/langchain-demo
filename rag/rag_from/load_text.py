import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import weaviate
from langchain_community.vectorstores import Weaviate
from weaviate.embedded import EmbeddedOptions
from langchain_ollama import OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama  # Updated import to avoid deprecation warning
from langchain.schema.runnable import RunnablePassthrough


file_path = "data/唐诗三百首.txt"
loader = TextLoader(file_path)
docs = loader.load()

# Split documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  # Reduced overlap for faster processing
chunks = text_splitter.split_documents(docs)

# Initialize vector database and embed target documents
client = weaviate.Client(
    embedded_options=EmbeddedOptions()
)



### =============== OllamaEmbeddings start ==============================
# embedding = OllamaEmbeddings(model="m3e:large-f16")

### =============== OllamaEmbeddings end ==============================

# ### =============== HuggingFaceEmbeddings start ==============================
# # # 设置模型参数，指定设备为 CPU
# from langchain_huggingface import HuggingFaceEmbeddings  # Updated import

# model_kwargs = {'device': 'cpu'}
# # 设置编码参数，禁用嵌入归一化
# encode_kwargs = {
#     'normalize_embeddings': False,
#     'clean_up_tokenization_spaces': False  # Explicitly set to avoid FutureWarning
# }
# model_name = "sentence-transformers/all-mpnet-base-v2"

# # 创建 HuggingFaceEmbeddings 实例，使用指定的模型和参数
# embedding = HuggingFaceEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs,
# )
# ======================= HuggingFaceEmbeddings end ==========================


#====================== embeddings start ==============================
from langchain_community.embeddings import ModelScopeEmbeddings  
# 生成向量（embedding）
model_id = "damo/nlp_corom_sentence-embedding_chinese-base"
embedding = ModelScopeEmbeddings(model_id=model_id, model_revision="v1.1.0")  
#=============================  embeddings end ==========================

# Vector database
vectorstore = Weaviate.from_documents(
    client=client,
    documents=chunks,
    embedding=embedding,
    by_text=False
)

# Retrieve
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})  # Increased number of results for better context
# retriever = vectorstore.as_retriever()  # Increased number of results for better context

# Query
# LLM prompt template
template = """You are an assistant for question-answering tasks. 
   Use the following pieces of retrieved context to answer the question. 
   If you don't know the answer, just say that you don't know. 
   Use Chinese to answer the question.
   Question: {question} 
   Context: {context} 
   Answer:
   """
prompt = ChatPromptTemplate.from_template(template)
llm = ChatOllama(
  model="llama3:8b", 
  temperature=0,
  verbose=True
)  # Reduced temperature for more deterministic responses

rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

# Start query & generate
query = "浮云终日行 请把这首补全"

response = rag_chain.invoke(query)
if response:
    print(response)