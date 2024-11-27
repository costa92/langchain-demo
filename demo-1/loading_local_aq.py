from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

# 优化步骤 1: 加载本地 TXT 文件
file_path = "data/完美世界.txt"  # 替换为你的 TXT 文件路径
loader = TextLoader(file_path, encoding="gb18030")  # 确认文件编码
documents = loader.load()

model_name = "llama3:8b"
# 优化步骤 2: 创建嵌入模型
try:
    embeddings_model = OllamaEmbeddings(model=model_name)  # 指定使用 llama3
except Exception as e:
    print(f"加载嵌入模型时出错: {e}")

# 优化步骤 3: 创建向量存储
try:
    vector_store = Chroma.from_documents(documents, embeddings_model)
except Exception as e:
    print(f"创建向量存储时出错: {e}")

# 优化步骤 4: 创建检索问答链

temperature = 0  # 可调整的参数
llm_model = OllamaLLM(model=model_name, temperature=temperature)

qa_chain = RetrievalQA.from_chain_type(
    llm_model, 
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 1}),  # 设置 n_results = 1
)

# 优化步骤 5: 提问
query = "荒天帝的名字是什么?,使用中文回答"  # 替换为你想要询问的问题
response = qa_chain.invoke(query)

# 检查响应并打印
if response:
    print(response)
else:
    print("未能获得有效响应。")
