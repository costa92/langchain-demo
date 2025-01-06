from langchain.prompts import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate

# 模板的构建
# 系统消息模板  
template = """你是业务咨询咨询顾问。负责为专注于{product}的公司起名字。"""
system_message_template = SystemMessagePromptTemplate.from_template(template)

# 用户消息模板
human_template = """公司主打产品是{product_name}。"""
human_message_template = HumanMessagePromptTemplate.from_template(human_template)

# 构建 ChatPromptTemplate 对象 
prompt_template = ChatPromptTemplate.from_messages([system_message_template,human_message_template])

# 下面调用模型，把提示消息传入模型，生成结果
import os

API_KEY = os.getenv("OPENAI_API_KEY")
model_name="Qwen/Qwen2.5-7B-Instruct"
base_url="https://api.siliconflow.cn/v1"

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(api_key=API_KEY,model_name=model_name,base_url=base_url)

chain = prompt_template | llm

res = chain.invoke({"product": "鲜花装饰", "product_name": "创新的鲜花设计"})

print(res.content)

