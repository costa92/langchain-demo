from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import os
# 定义一个命令聊天机器人
class CommandlineChatBot:
    def __init__(self):
        API_KEY = os.getenv("OPENAI_API_KEY")
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        base_url = "https://api.siliconflow.cn/v1"
        self.chat = ChatOpenAI(api_key=API_KEY, model_name=model_name, base_url=base_url)
        self.messages = [SystemMessage(content="你是一个花卉行家。")]

      # 定义一个循环来持续与用户交互
    def chat_loop(self):
        print("Chatbot 已启动! 输入'exit'来退出程序。")
        while True:
            user_input = input("请输入你的问题：")
            if user_input.lower() == "exit":
                print("Chatbot 已退出。")
                break
            self.messages.append(HumanMessage(content=user_input))
            response = self.chat.invoke(self.messages)
            print(response.content)


if __name__ == "__main__":
    bot = CommandlineChatBot()
    bot.chat_loop()