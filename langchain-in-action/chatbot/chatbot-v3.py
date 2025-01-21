from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import os

class ChatBotWithMemory:
    def __init__(self):
        API_KEY = os.getenv("OPENAI_API_KEY")
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        base_url = "https://api.siliconflow.cn/v1"
        
        # 初始化 LLM
        self.llm = ChatOpenAI(api_key=API_KEY, model_name=model_name, base_url=base_url)
        
        # 初始化 Prompt
        self.prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    "你是一个花卉行家。你通常的回答不超过30字。"
                ),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}")
            ]
        )
        
        # 初始化 Memory
        self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        
        # 初始化 ConversationChain
        self.conversation = ConversationChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory
        )

    def chat_loop(self):
        print("Chatbot 已启动! 输入'exit'来退出程序。")
        while True:
            user_input = input("请输入你的问题：")
            if user_input.lower() == "exit":
                break
            response = self.conversation.run(input=user_input)
            print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot = ChatBotWithMemory()
    chatbot.chat_loop()