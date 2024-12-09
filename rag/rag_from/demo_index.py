import os
# import getpass
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from uuid import uuid4
# unique_id = uuid4().hex[0:8]
os.environ["USER_AGENT"] = "UserAgent"
# os.environ['LANGCHAIN_TRACING_V2'] = 'true'
# os.environ["LANGCHAIN_PROJECT"] = f"Tracing Walkthrough - {unique_id}"
# os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
# # os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("LANGCHAIN_API_KEY")
# # os.environ["LANGCHAIN_API_KEY"] = os.getenv["LANGCHAIN_API_KEY"]

api_key = os.getenv("LANGCHAIN_API_KEY")
print(api_key)
print(os.getenv("LANGCHAIN_TRACING_V2"))
print(os.getenv("USER_AGENT"))
print(os.getenv("LANGCHAIN_ENDPOINT"))

# load Document

# Load Documents
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

documents = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

# Embed
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()