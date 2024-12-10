import os
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load documents from the specified web path
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# Split documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Set up embedding model
model_name = "sentence-transformers/all-mpnet-base-v2"

base_url = "http://localhost:11434/v1/"
embedding_api_key = "ollama"

# Ensure the API key is set
if not embedding_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")



# 设置模型参数，指定设备为 CPU
model_kwargs = {'device': 'cpu'}
# 设置编码参数，禁用嵌入归一化
encode_kwargs = {'normalize_embeddings': False}
# 创建 HuggingFaceEmbeddings 实例，使用指定的模型和参数
embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
# Store documents and embeddings in local vector store
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding, persist_directory="db")
retriever = vectorstore.as_retriever()

#### RETRIEVAL and GENERATION ####

# Load prompt from hub
prompt = hub.pull("rlm/rag-prompt")

# Initialize the language model
llm = ChatOpenAI(model="llama3:8b", api_key=embedding_api_key, base_url=base_url)

# Post-processing function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create the retrieval-augmented generation chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Invoke the chain with a question
response = rag_chain.invoke("What is Task Decomposition?，使用中文回答")
if response:
    print(response)



    