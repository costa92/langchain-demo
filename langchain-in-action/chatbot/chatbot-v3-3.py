from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

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
        
        # Initialize Prompt
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    "你是一个花卉行家。你通常的回答不超过30字。"
                ),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}")
            ]
        )
                # 创建消息历史存储
        self.store = {}

        # 定义获取会话历史的函数
        def get_session_history(session_id: str):
            if session_id not in self.store:
                self.store[session_id] = ChatMessageHistory()
            return self.store[session_id]


        # Create base runnable
        base_runnable = (
            RunnablePassthrough.assign(
                history=lambda x: x.get("history", [])
            ) 
            | self.prompt 
            | self.llm
        )

        # Wrap with message history
        self.conversation = RunnableWithMessageHistory(
            base_runnable,
            # Use ChatMessageHistory as the message store
            get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="history"
        )

    def chat_loop(self):
        print("Chatbot 已启动! 输入'exit'来退出程序。")
        session_id = "current_chat"  # 固定会话ID
        while True:
            user_input = input("请输入你的问题：")
            if user_input.lower() == "exit":
                break
            
            # Invoke with a configuration that includes a session ID
            config = {"configurable": {"session_id": session_id}}
            
            # Call conversation with config
            response = self.conversation.invoke(
                {"input": user_input}, 
                config=config
            )

            print(f"Chatbot: {response.content}")

                 # 手动打印历史记录以验证
            print("\n当前对话历史:")
            for msg in self.store[session_id].messages:
                print(f"{msg.__class__.__name__}: {msg.content}")
            print("\n")

if __name__ == "__main__":
    chatbot = ChatBotWithMemory()
    chatbot.chat_loop()
