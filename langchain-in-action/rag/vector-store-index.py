# 导入文档加载器模块，并使用TextLoader来加载文本文件
import os
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma # 使用 Chroma 作为向量存储


# Load the text file
loader = TextLoader('../three-qa/OneFlower/花语大全.txt', encoding='utf8')



embedd_model = "mxbai-embed-large"
embeddings = OllamaEmbeddings(model=embedd_model)


# # ### =============== HuggingFaceEmbeddings start ==============================
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
# embeddings = HuggingFaceEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs,
# )
# # ======================= HuggingFaceEmbeddings end ==========================


# Create language model instance
# 初始化语言模型
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 OPENAI_API_KEY")

model_name = "Qwen/Qwen2.5-7B-Instruct"
base_url = "https://api.siliconflow.cn/v1"

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=base_url,
    model_name=model_name,
)

# 配置文本分割器
# 调整 chunk_size 和 chunk_overlap，确保生成足够多的文档块
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)  # 调整参数

# Create the vector store index
index_creator = VectorstoreIndexCreator(
    embedding=embeddings,
    text_splitter=text_splitter,
    vectorstore_cls=Chroma
)
index = index_creator.from_loaders([loader])

# Define the query string and execute the query
query = "玫瑰花的花语是什么？"
result = index.query(query, llm=llm)

# Print the query result
# print(result)




# 替换成你所需要的工具
# from langchain.text_splitter import CharacterTextSplitter
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# from langchain.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings()
# index_creator = VectorstoreIndexCreator(
#     vectorstore_cls=Chroma,
#     embedding=OpenAIEmbeddings(),
#     text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# )