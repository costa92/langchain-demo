# 1.Load 导入Document Loaders
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 加载 Documents
base_dir = 'OneFlower' # 文档的存放目录
documents = []
for file in os.listdir(base_dir):
    file_path = os.path.join(base_dir, file)
    if file.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        loader = TextLoader(file_path)
        documents.extend(loader.load())

# 2.Split 将Documents切分成块以便后续进行嵌入和向量存储

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunked_documents = text_splitter.split_documents(documents)

# 3.Store 将分割嵌入并存储在矢量数据库Qdrant中

from langchain_community.vectorstores import Qdrant
from langchain_openai import ChatOpenAI, OpenAIEmbeddings



# embedding = OpenAIEmbeddings()

# ### =============== HuggingFaceEmbeddings start ==============================
# # 设置模型参数，指定设备为 CPU
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import

model_kwargs = {'device': 'cpu'}
# 设置编码参数，禁用嵌入归一化
encode_kwargs = {
    'normalize_embeddings': False,
    'clean_up_tokenization_spaces': False  # Explicitly set to avoid FutureWarning
}
model_name = "sentence-transformers/all-mpnet-base-v2"

# 创建 HuggingFaceEmbeddings 实例，使用指定的模型和参数
embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)
# ======================= HuggingFaceEmbeddings end ==========================

vectorstore = Qdrant.from_documents(
    chunked_documents, # 以分块的文档
    embedding=embedding, # 使用OpenAIEmbeddings进行嵌入
    location=":memory:",  # in-memory 存储
    collection_name="my_documents",  # 指定collection_name
)

# 4. Retrieval 准备模型和Retrieval链
import logging # 导入Logging工具


logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)
from langchain.retrievers.multi_query import MultiQueryRetriever # MultiQueryRetriever工具
from langchain.chains import RetrievalQA # RetrievalQA链


# API_KEY = os.getenv("OPENAI_API_KEY")
# model_name="Qwen/Qwen2.5-7B-Instruct"
# base_url="https://api.siliconflow.cn/v1"  

model_name="deepseek-chat"
API_KEY = os.getenv("deepseek_api_key")
# base_url = os.getenv("deepseek_api_url")
base_url="https://api.deepseek.com/beta"


llm = ChatOpenAI(api_key=API_KEY, base_url=base_url, model=model_name, temperature=0)

# 实例化一个MultiQueryRetriever
retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)

# 实例化一个RetrievalQA链
qa_chain = RetrievalQA.from_chain_type(llm,retriever=retriever_from_llm)

# 5. Output 问答系统的UI实现
from flask import Flask, request, render_template
app = Flask(__name__) # Flask APP

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # 接收用户输入作为问题
        question = request.form.get('question')        
        
        # RetrievalQA链 - 读入问题，生成答案
        result = qa_chain.invoke({"query": question})
        
        # 把大模型的回答结果返回网页进行渲染
        return render_template('index.html', result=result)
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True,port=8080)