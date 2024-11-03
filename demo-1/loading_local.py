from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain.chains.summarize import load_summarize_chain
from langchain_unstructured import UnstructuredLoader
# 导入文件
loader = UnstructuredLoader("data/完美世界.txt", encoding="gb18030")

# 将内容转成 Document 对象
document = loader.load()
# 打印文档的长度
print(f'documents:{len(document)}')

text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=500,
  chunk_overlap=0,
)

# 切分文本
split_documents = text_splitter.split_documents(document)
print(f'documents:{len(split_documents)}')

# 加载 llm 模型
llm = OllamaLLM(model="llama3", temperature=0)

# 创建总结链
chain = load_summarize_chain(llm, chain_type="refine", verbose=True)

# 执行总结链，（为了快速演示，只总结前5段）
chain.invoke(split_documents[:5])