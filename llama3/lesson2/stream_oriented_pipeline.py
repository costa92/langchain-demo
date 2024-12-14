from threading import Thread
import torch
from transformers import pipeline, TextIteratorStreamer
import os
os.environ['CURL_CA_BUNDLE'] = ''

device = "cuda" if torch.cuda.is_available() else "cpu"

messages = [
    {"role": "system", "content": "你是一个诗人,请写一首优美的七言绝句。要求:1. 描写秋天的意境 2. 遵循七言绝句格律 3. 意境优美"},
    {"role": "user", "content": "请写一首赞美秋天的七言绝句"}
]

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
pipe = pipeline("text-generation", model=model_name, device=device, max_new_tokens=100)

# 创建streamer实例
streamer = TextIteratorStreamer(pipe.tokenizer, skip_prompt=True, skip_special_tokens=True)

# 准备输入文本
text = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 设置生成参数
generation_kwargs = dict(text_inputs=text, streamer=streamer)

# 在新线程中运行生成
thread = Thread(target=pipe, kwargs=generation_kwargs)
thread.start()

# 收集输出并打印
output = ""
for text in streamer:
    output += text
    print(text, end="", flush=True)

print("\n")  # 最后打印换行