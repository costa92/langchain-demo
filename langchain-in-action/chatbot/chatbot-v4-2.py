import os
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langgraph.graph import StateGraph, START
from langgraph.types import TypedDict

# 解决 Hugging Face tokenizers 并行化问题
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 定义对话状态模型
class ChatState(TypedDict):
    messages: List[Dict[str, str]]  # 消息列表，包含角色和内容
    context: str  # 上下文

class ChatbotWithRetrieval:
    def __init__(self, dir: str):
        # 加载文档
        self.documents = self.load_documents(dir)

        # 文本分割
        self.all_splits = self.split_documents(self.documents)

        # 初始化嵌入模型
        self.embedding = self.initialize_embedding()

        # 创建向量存储
        self.vectorstore = self.create_vectorstore(self.all_splits)

        # 初始化 LLM
        self.llm = self.initialize_llm()

        # 构建工作流
        self.workflow = self.build_workflow()

        self.compiled_workflow = self.workflow.compile()  # Compile the workflow

    def load_documents(self, base_dir: str) -> List:
        """加载文档"""
        documents = []
        for file in os.listdir(base_dir):
            file_path = os.path.join(base_dir, file)
            if file.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith('.txt'):
                loader = TextLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
            else:
                print(f"Unsupported file type: {file}")
        return documents

    def split_documents(self, documents: List) -> List:
        """分割文档"""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        return text_splitter.split_documents(documents)

    def initialize_embedding(self):
        """初始化嵌入模型"""
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        model_name = "sentence-transformers/all-mpnet-base-v2"

        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    def create_vectorstore(self, documents: List) -> Any:
        """创建向量存储"""
        return Qdrant.from_documents(
            documents=documents,
            embedding=self.embedding,
            location=":memory:",  # 内存存储
            collection_name="documents"
        )

    def initialize_llm(self) -> Any:
        """初始化语言模型"""
        API_KEY = os.getenv("OPENAI_API_KEY") or "your_openai_api_key_here"  # 确保 API_KEY 已设置
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        base_url = "https://api.siliconflow.cn/v1"

        return ChatOpenAI(
            openai_api_key=API_KEY,
            model=model_name,
            openai_api_base=base_url
        )

    def retrieve_and_answer(self, state: ChatState) -> ChatState:
        """检索相关文档并生成回答"""
        user_input = state['messages'][-1]['content']
        retriever = self.vectorstore.as_retriever()
        retrieved_docs = retriever.invoke(user_input)

        # 准备上下文
        # context = "\n".join([doc.page_content for doc in retrieved_docs])
        # response = self.llm.invoke(f"Context:\n{context}\n\nQuestion: {user_input}")

        history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state['messages']])
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        response = self.llm.invoke(f"History:\n{history}\n\nContext:\n{context}\n\nQuestion: {user_input}")

        # 提取响应内容
        response_content = response.content if hasattr(response, 'content') else str(response)

        # 返回更新的状态
        return {
            "messages": state['messages'] + [{"role": "assistant", "content": response_content}],
            "context": context
        }

    def build_workflow(self) -> StateGraph:
        """构建对话工作流"""
        workflow = StateGraph(ChatState)

        # Add the node with a name
        workflow.add_node("retrieve_and_answer", self.retrieve_and_answer)  # Register the node with a name

        # Add the edge using the node name
        workflow.add_edge(START, "retrieve_and_answer")  # Use the node name

        # Set the entry point (if needed)
        workflow.set_entry_point("retrieve_and_answer")

        return workflow

    def chat_loop(self):
        """聊天循环"""
        print("Chatbot 已启动! 输入 'exit' 来退出程序。")
        state = {"messages": [], "context": ""}

        while True:
            user_input = input("你: ")
            if user_input.lower() == 'exit':
                print("再见!")
                break

            # 更新状态并调用工作流
            state['messages'].append({"role": "user", "content": user_input})
            state = self.compiled_workflow.invoke(state)
            print(state)
            print(f"Chatbot: {state['messages'][-1]['content']}")

if __name__ == "__main__":
    # 启动 Chatbot
    folder = "../three-qa/OneFlower"  # 替换为你的文档目录路径
    bot = ChatbotWithRetrieval(folder)
    bot.chat_loop()