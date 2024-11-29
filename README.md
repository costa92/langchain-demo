# langchain

## 环境安装

[conda 安装](/docs/conda%20安装.md)


## 使用 LangServe 提供服务
```sh
pip install "langserve[all]"
```

## 本地运行的加载程序

请按照以下步骤获取 unstructured 和 其依赖项在本地运行。

```sh
pip install "unstructured[local-inference]" 
```

需要安装的扩展：

  libmagic-dev（文件类型检测）
  poppler-utils（图像和 PDF）
  tesseract-ocr（图像和 PDF）
  libreoffice（MS Office 文档）
  pandoc（EPUB）

## LLaMA3  

### RAG 系统

1. 分割技术：GraphRAG与提示词压缩技术
   1. 智能索引构建：GraphRAG
   2. 高效的检索系统设计：提示词压缩技术

  GraphRAG技术：
    算法： 过层次聚类技术（如常用的 Leiden 算法）


2. 总结：
1. 在信息检索和内容生成，不再依赖传统的向量数据库和文档分割技术已不够充分，而应当充分利用 LLaMA 3 的强大功能，以提升索引、检索和内容生成的智能化程度。
2. RAG 的基本流程涵盖几个关键步骤：文档分割、嵌入生成、向量存储、检索和内容生成

```py

# 导入所需的包
import chromadb
from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

# 定义文件目录
file_directory = "path/to/your/documents"  # 更新为你的文件路径

# 嵌入模型
llm = Ollama(model="llama3", request_timeout=300.0)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = llm
Settings.embed_model = embed_model

# 从目标文档中设置上下文
documents = SimpleDirectoryReader(input_files=[file_directory]).load_data()
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("ollama")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context, 
    embed_model=embed_model,
    transformations=[SentenceSplitter(chunk_size=256, chunk_overlap=10)]
)

# 设置通用的 RAG 提示词模板（中文）
qa_template = PromptTemplate(
    "基于提供的上下文：\n"
    "-----------------------------------------\n"
    "{context_str}\n"
    "-----------------------------------------\n"
    "请回答以下问题：\n"
    "问题：{query_str}\n\n"
    "答案："
)

query_engine = index.as_query_engine(text_qa_template=qa_template, similarity_top_k=3)

# 示例查询
query = "文档的主要发现是什么？"
response = query_engine.query(query)
print(response.response)
```



## 参考文档

[爱鼓捣-blog](https://techdiylife.github.io/blog/topic.html?category2=t07&blogid=0043)
[langchain 中文](http://python.langchain.com.cn/)
[LangChain实战 | 3分钟学会SequentialChain传递多个参数](https://blog.csdn.net/sinat_29950703/article/details/139263894)
[langchain学习之chain机制](https://blog.csdn.net/zc1226/article/details/140011057?spm=1001.2101.3001.6650.15&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-15-140011057-blog-139263894.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-15-140011057-blog-139263894.235%5Ev43%5Epc_blog_bottom_relevance_base9&utm_relevant_index=18)

[OceanBase AI 动手实战营](https://gitee.com/oceanbase-devhub/ai-workshop-2024)


## 许可证

本仓库的代码和文档遵循 [MIT License](LICENSE) 许可证。
