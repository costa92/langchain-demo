from transformers import AutoTokenizer,AutoModelForCausalLM
import os

#  1. 禁用 tokenizers 的并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "true"

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


messages = [
    {"role": "user", "content": "请写一首赞美春天的诗，要求不包含春字."},
]

text =  tokenizer.apply_chat_template(
  messages, 
  tokenize=False, 
  add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

result = model.generate(**model_inputs, max_new_tokens=100)

generated_ids = [
  output_ids[len(input_ids):]
  for output_ids, input_ids in zip(result, model_inputs.input_ids)
]

generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(generated_text)