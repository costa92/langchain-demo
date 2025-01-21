from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
import os

class ChatBotWithMemory:
    def __init__(self):
        API_KEY = os.getenv("OPENAI_API_KEY")
        model_name = "Qwen/Qwen2.5-7B-Instruct"

        # 修复 LLM 初始化
        self.llm = ChatOpenAI(
            openai_api_key=API_KEY, 
            model=model_name,  # 使用兼容的模型名称
            openai_api_base="https://api.siliconflow.cn/v1"
        )
        
        # 初始化 Prompt
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    "你是一个花卉行家。你通常的回答不超过30字。"
                ),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}")
            ]
        )
        
        # 初始化 ChatMessageHistory 和 ConversationBufferWindowMemory
        self.chat_history = ChatMessageHistory()
        self.memory = ConversationBufferWindowMemory(
            chat_memory=self.chat_history, 
            return_messages=True
        )
        
        # 简化对话链
        self.conversation = (
            RunnablePassthrough.assign(
                history=lambda inputs: self.memory.chat_memory.messages
            ) 
            | self.prompt 
            | self.llm
        )

    def chat_loop(self):
        print("Chatbot 已启动! 输入'exit'来退出程序。")
        while True:
            user_input = input("请输入你的问题：")
            if user_input.lower() == "exit":
                break
            # 调用对话
            response = self.conversation.invoke({"input": user_input})
            # 保存上下文
            self.memory.save_context({"input": user_input}, {"output": response.content})

            print(f"Chatbot: {response.content}")

if __name__ == "__main__":
    chatbot = ChatBotWithMemory()
    chatbot.chat_loop()
