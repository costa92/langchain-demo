from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory  # 新增

import os

class ChatBotWithMemory:
    def __init__(self):
        API_KEY = os.getenv("OPENAI_API_KEY")
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        base_url = "https://api.siliconflow.cn/v1"
        
        # 修复 LLM 初始化
        self.llm = ChatOpenAI(
            openai_api_key=API_KEY, 
            model=model_name, 
            openai_api_base=base_url
        )
        
        # 初始化 Prompt
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    "你是一个花卉行家。你通常的回答不超过30字。"
                ),
                MessagesPlaceholder(variable_name="chat_history"),  # 对话历史变量
                HumanMessagePromptTemplate.from_template("{input}")  # 用户输入变量
            ]
        )
        
         # 使用 InMemoryChatMessageHistory 替代 ConversationBufferMemory
        self.chat_history = InMemoryChatMessageHistory()
      
        # 创建基础链
        self.chain = self.prompt | self.llm
        
        # 使用 RunnableWithMessageHistory 包装链
        self.conversation = RunnableWithMessageHistory(
            self.chain,
            lambda session_id: self.chat_history,   # 直接返回 chat_history
            input_messages_key="input",  # 用户输入的键
            history_messages_key="chat_history"  # 对话历史的键
        )

    def chat_loop(self):
        print("Chatbot 已启动! 输入'exit'来退出程序。")
        session_id = "user_session_1"  # 会话 ID，用于区分不同用户的对话历史
        while True:
            user_input = input("请输入你的问题：")
            if user_input.lower() == "exit":
                break
            # 调用对话
            response = self.conversation.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}  # 传递会话 ID
            )
            # 手动添加消息到历史 
            self.chat_history.add_messages([
                HumanMessagePromptTemplate.from_template("{input}").format(input=user_input),
                response,
            ])
            print(f"Chatbot: {response.content}")

if __name__ == "__main__":
    chatbot = ChatBotWithMemory()
    chatbot.chat_loop()
