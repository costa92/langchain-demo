import torch

from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

messages = [ {"role": "user", "content": "请写一首赞美秋天的五言绝句"},]

pipe = pipeline("text-generation", model=model_name, device=device, max_new_tokens=100)

result = pipe(messages)

print(result[-1]['generated_text'][-1]['content'])