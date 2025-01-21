import os
import json
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
    context: str  # 上下文（来自文件内容的检索结果）

class ChatbotWithRetrieval:
    def __init__(self, dir: str, history_file: str = "chat_history.json"):
        # 加载历史对话
        self.history_file = history_file
        self.load_history()

        # 加载文档内容
        self.documents = self.load_documents(dir)

        # 文本分割
        self.all_splits = self.split_documents(self.documents)

        # 初始化嵌入模型
        self.embedding = self.initialize_embedding()

        # 创建向量存储（用于文件内容检索）
        self.vectorstore = self.create_vectorstore(self.all_splits)

        # 初始化 LLM
        self.llm = self.initialize_llm()

        # 构建工作流
        self.workflow = self.build_workflow()

        self.compiled_workflow = self.workflow.compile()  # 编译工作流

    def load_documents(self, base_dir: str) -> List:
        """加载文档内容"""
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
        """分割文档内容"""
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
        """创建向量存储（用于文件内容检索）"""
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
        """检索文件内容并生成回答"""
        user_input = state['messages'][-1]['content']
        retriever = self.vectorstore.as_retriever()
        retrieved_docs = retriever.invoke(user_input)

        # 限制历史对话的长度，只保留最近的5次对话
        max_history_length = 5
        recent_history = state['messages'][-max_history_length:]

        # 动态调整上下文：如果用户的问题与之前的对话无关，清除历史对话
        if "new topic" in user_input.lower():  # 假设 "new topic" 是用户表示新话题的关键词
            recent_history = []

        # 准备对话历史记录
        history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])

        # 准备文件内容上下文
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        # 调用 LLM 生成回答
        response = self.llm.invoke(
            f"对话历史记录:\n{history}\n\n文件内容上下文:\n{context}\n\n问题: {user_input}"
        )

        # 提取响应内容
        response_content = response.content if hasattr(response, 'content') else str(response)

        # 返回更新的状态
        return {
            "messages": state['messages'] + [{"role": "assistant", "content": response_content}],
            "context": context  # 更新上下文为本次检索的文件内容
        }

    def build_workflow(self) -> StateGraph:
        """构建对话工作流"""
        workflow = StateGraph(ChatState)

        # 添加节点
        workflow.add_node("retrieve_and_answer", self.retrieve_and_answer)

        # 添加边
        workflow.add_edge(START, "retrieve_and_answer")

        # 设置入口点
        workflow.set_entry_point("retrieve_and_answer")

        return workflow

    def load_history(self):
        """加载历史对话记录"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = {"messages": [], "context": ""}

    def save_history(self):
        """保存历史对话记录"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f)

    def chat_loop(self):
        """聊天循环"""
        print("Chatbot 已启动! 输入 'exit' 来退出程序。")
        state = self.history

        while True:
            user_input = input("你: ")
            if user_input.lower() == 'exit':
                print("再见!")
                self.save_history()  # 退出前保存历史对话记录
                break

            # 更新状态并调用工作流
            state['messages'].append({"role": "user", "content": user_input})
            state = self.compiled_workflow.invoke(state)
            print(f"Chatbot: {state['messages'][-1]['content']}")

            # 更新历史对话记录
            self.history = state

if __name__ == "__main__":
    # 启动 Chatbot
    folder = "../three-qa/OneFlower"  # 替换为你的文档目录路径
    bot = ChatbotWithRetrieval(folder)
    bot.chat_loop()