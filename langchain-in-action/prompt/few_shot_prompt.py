
# 少样本提示
samples = [
  {
    "flower_type": "玫瑰",
    "occasion": "爱情",
    "ad_copy": "玫瑰代表爱情，是表达爱意的最佳选择。"
  },
  {
    "flower_type": "康乃馨",
    "occasion": "母亲节",
    "ad_copy": "康乃馨代表母爱，是表达母爱的最佳选择。"
  },
  {
    "flower_type": "向日葵",
    "occasion": "生日",
    "ad_copy": "向日葵代表阳光，是表达阳光的最好选择。"
  },
  {
    "flower_type": "百合",
    "occasion": "婚礼",
    "ad_copy": "百合代表纯洁，是表达纯洁的最好选择。"
  }
]

# 创建提示模板
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings

prompt_sample = PromptTemplate(
  input_variables=["flower_type","occasion","ad_copy"],
  template="鲜花类型：{flower_type}\n场合：{occasion}\n文案：{ad_copy}"
)

# 输出少样本提示
# print(prompt_sample.format(**samples[0]))

# 创建一个FewShotPromptTemplate
from langchain.prompts import FewShotPromptTemplate

prompt = FewShotPromptTemplate(
  examples=samples,
  example_prompt=prompt_sample,
  prefix="以下是一些鲜花文案示例：",
  suffix="鲜花类型: {flower_type}\n场合: {occasion}",
  input_variables=["flower_type","occasion"]
)

# print(prompt.format(flower_type="野玫瑰", occasion="爱情"))

# 把提示传递给大模型
# 下面调用模型，把提示消息传入模型，生成结果
import os

API_KEY = os.getenv("OPENAI_API_KEY")
model_name="Qwen/Qwen2.5-7B-Instruct"
base_url="https://api.siliconflow.cn/v1"

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(api_key=API_KEY,model_name=model_name,base_url=base_url)

res = llm.invoke(prompt.format(flower_type="野玫瑰", occasion="爱情"))

print(res.content)

# # .使用示例选择器
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma




# # ### =============== HuggingFaceEmbeddings start ==============================
# # # 设置模型参数，指定设备为 CPU
# from langchain_huggingface import HuggingFaceEmbeddings  # Updated import

# model_kwargs = {'device': 'cpu'}
# # 设置编码参数，禁用嵌入归一化
# encode_kwargs = {
#     'normalize_embeddings': False,
#     'clean_up_tokenization_spaces': False  # Explicitly set to avoid FutureWarning
# }
# model_name = "sentence-transformers/all-mpnet-base-v2"

# # 创建 HuggingFaceEmbeddings 实例，使用指定的模型和参数
# embeddings = HuggingFaceEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs,
# )
# # ======================= HuggingFaceEmbeddings end ==========================



# OpenAIEmbeddings is deprecated, use OllamaEmbeddings instead
embedd_model = "mxbai-embed-large"
embeddings = OllamaEmbeddings(model=embedd_model)




# 初始化示例选择器
example_selector = SemanticSimilarityExampleSelector.from_examples(
    samples,
    embeddings,
    Chroma,
    k=2
)

# 创建一个使用示例选择器的FewShotPromptTemplate对象
prompt = FewShotPromptTemplate(
    example_selector=example_selector, 
    example_prompt=prompt_sample, 
    suffix="鲜花类型: {flower_type}\n场合: {occasion}", 
    input_variables=["flower_type", "occasion"]
)
print("===================")
print(prompt.format(flower_type="红玫瑰", occasion="爱情"))

print("===================")
res = llm.invoke(prompt.format(flower_type="红玫瑰", occasion="爱情"))
print(res.content)
