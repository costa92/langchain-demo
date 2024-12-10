import os
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser # 输出解析器
from langchain_core.runnables import RunnablePassthrough
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
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
encode_kwargs = {
    'normalize_embeddings': False,
    'clean_up_tokenization_spaces': False  # Explicitly set to avoid FutureWarning
}
# 创建 HuggingFaceEmbeddings 实例，使用指定的模型和参数
embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)
# Store documents and embeddings in local vector store
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding, persist_directory="db")

# 检索器
# retriever = vectorstore.as_retriever() 
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})


#### RETRIEVAL and GENERATION ####

# Load prompt from hub
prompt = hub.pull("rlm/rag-prompt")

#  prompt 可以自定义提示词 使用 from langchain.prompts import ChatPromptTemplate
#  prompt = ChatPromptTemplate.from_template(template)


model_name = "llama3"
# Initialize the language model
llm = ChatOpenAI(model=model_name, api_key=embedding_api_key, base_url=base_url,temperature=0 )

# Post-processing function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create the retrieval-augmented generation chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough(),}  # 上下文信息
    | prompt
    | llm
    | StrOutputParser()
)

# Invoke the chain with a question
response = rag_chain.invoke("What is Task Decomposition?，使用中文回答")
if response:
    print(response)