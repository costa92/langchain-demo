# https://github.com/langchain-ai/langchain/blob/acddfc772e78c9b8a13b57aa7d3237a77159e7b3/docs/docs/integrations/llms/huggingface_pipelines.ipynb
# 指定预训练模型的名称
import os


# model = "meta-llama/Llama-2-7b-chat-hf"
model = "Qwen/Qwen2.5-0.5B-Instruct"

# print(os.environ["HUGGINGFACEHUB_API_TOKEN"])

# 从预训练模型中加载词汇器
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
  model, 
)



# 创建一个文本生成的管道
import transformers
import torch
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    max_length = 1000,
    tokenizer=tokenizer
)


# 创建HuggingFacePipeline实例
from langchain_huggingface import HuggingFacePipeline
llm = HuggingFacePipeline(
  pipeline = pipeline, 
  model_kwargs = {'temperature':0, "max_length":180,},
  pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    ),
  )



# 定义输入模板，该模板用于生成花束的描述
template = """
              为以下的花束生成一个详细且吸引人的描述：
              花束的详细信息：
              ```{flower_details}```
           """

# 使用模板创建提示
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(template=template, 
                     input_variables=["flower_details"])


llm_chain =  prompt | llm

# 需要生成描述的花束的详细信息
flower_details = "12支红玫瑰，搭配白色满天星和绿叶，包装在浪漫的红色纸中。"

# 打印生成的花束描述
output = llm_chain.invoke(flower_details)
# 去掉 HTML 样式
import re
clean_output = re.sub(r'<[^>]+>', '', output)
print(clean_output)