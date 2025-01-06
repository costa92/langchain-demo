import os
from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

# 导入LangChain中的提示模板
from langchain.prompts import PromptTemplate

# 创建原始模板
template = """You are a flower shop assitiant。\n
For {price} of {flower_name} ，can you write something for me？
"""

# 创建提示模板
prompt = PromptTemplate.from_template(template)

print(prompt)

# 导入LangChain中的HuggingFace模型接口

api_token = os.getenv("HUGGINGFACE_API_KEY")
# 创建HuggingFaceHub模型接口，添加到git凭证
# model = HuggingFaceEndpoint(
#   repo_id="google/flan-t5-large", 
#   parameters={"max_new_tokens": 250},
#   add_to_git_credential=True
#   )
from langchain_community.llms import HuggingFaceHub

# # 创建模型实例
model= HuggingFaceHub(repo_id="google/flan-t5-large")
# from langchain_huggingface import HuggingFaceEndpoint
# repo_id="google/flan-t5-large", 
# model = HuggingFaceEndpoint(
#     repo_id=repo_id,
#     temperature=0.5,
#     huggingfacehub_api_token=api_token,
#     model_kwargs={"return_full_text": False,"max_length":128},
# )
# 输入提示
input = prompt.format(flower_name=["玫瑰"], price='50')

# 得到模型的输出
output = model.invoke(input)
# 打印输出内容
print("output:")
print(output)