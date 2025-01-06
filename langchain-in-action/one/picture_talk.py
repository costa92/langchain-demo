from dotenv import load_dotenv
import requests

load_dotenv()


#---- Part 0 导入所需要的类
import os
from transformers import BlipProcessor,BlipForConditionalGeneration
from langchain.tools import BaseTool

from PIL import Image
from langchain_openai import OpenAI
from langchain.agents import create_react_agent,AgentExecutor # 未来需要改成下面的设计
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain.agents import initialize_agent, AgentType

#---- Part I 初始化图像字幕生成模型
# 指定要使用的工具模型（HuggingFace中的image-caption模型）
hf_model = "Salesforce/blip-image-captioning-large"

# 初始化处理器和工具模型
# 预处理器将准备图像供模型使用
processor = BlipProcessor.from_pretrained(hf_model)

# 然后我们初始化工具模型本身
model = BlipForConditionalGeneration.from_pretrained(hf_model)

#---- Part II 定义图像字幕生成工具类
class ImageCapTool(BaseTool):
    name: str = "Image captioner"
    description: str = "用于生成图像字幕的工具"
    
    def _run(self, url: str) -> str:
        # 将输入的图像转换为PIL图像对象
        image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
        # 使用预处理器处理图像
        inputs = processor(image, return_tensors="pt")
        # 使用模型生成字幕
        out = model.generate(**inputs, max_new_tokens=20)
        # 将输出转换为字符串
        res =  processor.decode(out[0], skip_special_tokens=True)
        return res
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


#---- Part III 使用工具类生成图像字幕
# 设置OpenAI的API密钥并初始化大语言模型（OpenAI的Text模型）
# os.environ["OPENAI_API_KEY"] = '你的OpenAI API Key'
API_KEY = os.getenv("OPENAI_API_KEY")
model_name="Qwen/Qwen2.5-7B-Instruct"
base_url="https://api.siliconflow.cn/v1"

# model_name="deepseek-chat"
# API_KEY = os.getenv("deepseek_api_key")
# # base_url = os.getenv("deepseek_api_url")
# base_url="https://api.deepseek.com/beta"

llm = OpenAI(api_key=API_KEY, base_url=base_url, model=model_name,temperature=0.2)
# 使用工具初始化智能体并运行
tools = [ImageCapTool()]
prompt = hub.pull("hwchase17/react")


# agent = initialize_agent(
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     tools=tools,
#     llm=llm,
#     verbose=True,
# )
from langchain.agents import create_react_agent # 未来需要改成下面的设计
agent = create_react_agent(
  llm=llm, 
  tools=tools, 
  prompt=prompt
)

agent_executor = AgentExecutor(agent=agent, tools=tools)


img_url = 'https://mir-s3-cdn-cf.behance.net/project_modules/hd/eec79e20058499.563190744f903.jpg'
# agent.run(input=f"{img_url}\n请创作合适的中文推广文案")
res =   agent_executor.invoke({"input":f"{img_url}\n请创作合适的中文推广文案"})
print(res["output"])