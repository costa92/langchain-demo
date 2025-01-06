import os

# os.environ["USE_MODELSCOPE_HUB"] = "true"
# # api_token = os.getenv("HUGGINGFACE_API_KEY")
# # print(api_token)

# # os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token

# # 导入必要的库
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # model_name = "meta-llama/Llama-3.2-1B"
# model_name = "t5-base"
# # 加载预训练模型的分词器
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# # 加载预训练的模型
# # 使用 device_map 参数将模型自动加载到可用的硬件设备上，例如GPU
# model = AutoModelForCausalLM.from_pretrained(
#           model_name, 
#           device_map = 'auto')  

# # 定义一个提示，希望模型基于此提示生成故事
# prompt = "请给我讲个玫瑰的爱情故事?" 

# # 使用分词器将提示转化为模型可以理解的格式，并将其移动到GPU上 cuda
# inputs = tokenizer(prompt, return_tensors="pt")
# # 使用模型生成文本，设置最大生成令牌数为2000
# outputs = model.generate(inputs["input_ids"], max_new_tokens=2000)

# # 将生成的令牌解码成文本，并跳过任何特殊的令牌，例如[CLS], [SEP]等
# response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# # 打印生成的响应
# print(response)


from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name,legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# input_text = "translate English to German: How old are you?"
# input_text = "请给我讲个玫瑰的爱情故事"
input_text = "Can you tell me the capital of russia"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids,max_new_tokens=2000)

# # 将生成的令牌解码成文本，并跳过任何特殊的令牌，例如[CLS], [SEP]等
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# 打印生成的响应
print(response)