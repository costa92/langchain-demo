# After discussing the LCEL primitives and examining the isolated examples, it’s high time to build a simple chain with a retriever in LCEL. First, we get the documents into the vector database: 5 entries into Wikipedia on top companies in the Gen AI space.
# :zh 在讨论了LCEL原语并检查了孤立的示例之后，现在是时候在LCEL中构建一个带有检索器的简单链了。首先，我们将文档放入向量数据库：Gen AI空间中顶级公司的维基百科上的5个条目。

 # Getting example docs into vectordb
import os

os.environ["USER_AGENT"] ="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel,RunnableLambda,RunnablePassthrough
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings

wiki_on_llm_companies = ["https://en.wikipedia.org/wiki/Mistral_AI",
                         "https://en.wikipedia.org/wiki/Anthropic",
                         "https://en.wikipedia.org/wiki/Hugging_Face",
                         "https://en.wikipedia.org/wiki/Cohere",
                         "https://en.wikipedia.org/wiki/OpenAI"]


loader = WebBaseLoader(wiki_on_llm_companies)
docs_orig = loader.load()

# 参数说明：chunk_size=500, chunk_overlap=200
# chunk_size:递归字符文本分割器
# chunk_overlap:重叠
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)

docs = text_splitter.split_documents(docs_orig)

model_name = "qwen2.5:7b"
embedd_model = "mxbai-embed-large:latest"
api_key = "ollama"
base_url = "http://localhost:11434/v1/"


# embedding_function = OpenAIEmbeddings(
#     api_key=api_key,
#     model_name=embedd_model,
#     base_url="http://localhost:11434/v1/"
# )

# OpenAIEmbeddings is deprecated, use OllamaEmbeddings instead
embedding_function = OllamaEmbeddings(model=embedd_model)

db = Chroma.from_documents(docs, embedding_function)

# Defining simple RAG with retriever in LCEL

retriever = db.as_retriever()

template = """Answer the question based only on the following context.
If the context doesn't contain entities present in the question say you 
don't know.

{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model_name = "qwen2.5:7b"

llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_url, 
    temperature=0,
    verbose=True
)

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


question_with_context = RunnableParallel(
                            context=(itemgetter("question")
                                | retriever
                                | RunnableLambda(lambda x: format_docs(x))
                            ),
                            question=RunnablePassthrough()
                        ) 
chain = question_with_context | prompt | llm | StrOutputParser()

res = chain.invoke({"question": "What are investors of Anthropic?"})
print(res)
# Output:
# Amazon, Google, FTX
