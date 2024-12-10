
import os
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
import base64
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

LANGCHAIN_ENDPOINT=os.environ['LANGCHAIN_ENDPOINT']
LANGCHAIN_API_KEY=os.environ['LANGCHAIN_API_KEY']

# 初始化 ChatOpenAI 模型
# chat = ChatOpenAI(model="llama3.2-vision:11b", api_key="ollama", base_url="http://localhost:11434/v1/")
chat = ChatOpenAI(model="llama3:8b", api_key="ollama", base_url="http://localhost:11434/v1/")

# print("第一次对话:",chat.invoke("你是一只小狗,只会汪汪叫"),"\n\n第二次对话:",chat.invoke("你是一只小狗嘛"))

# # 初始化对话记忆
# if "memory" not in st.session_state:
#     st.session_state.memory = ConversationBufferMemory()

# # 初始化对话链
# conversation = ConversationChain(llm=chat, memory=st.session_state.memory)

# # 用户输入
# user_input = st.text_input("请输入你想让模型分析的内容:", "分析这张图片")

# # 上传图片
# uploaded_file = st.file_uploader("上传图片", type=["png", "jpg", "jpeg"])

# if uploaded_file and user_input:
#     # 读取并编码图像
#     img_base64 = base64.b64encode(uploaded_file.read()).decode()

#     # 使用 ChatOpenAI 模型进行图像分析
#     messages = [
#         HumanMessage(
#             content=user_input,
#             additional_kwargs={"image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
#         )
#     ]

#     msg = chat.invoke(messages)

#     # 显示结果
#     st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
#     st.write("图像分析结果:")
#     st.write(msg.content)  # 直接访问 content 属性

# # 显示对话记录
# if "chat_history" in st.session_state and st.session_state.chat_history:
#     st.write("对话记录:")
#     for msg in st.session_state.chat_history:
#         st.write(f"用户: {msg['input']}")
#         st.write(f"模型: {msg['output']}")

