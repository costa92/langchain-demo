from transformers import pipeline,AutoTokenizer

def data():
  for i in range(10):
    yield f"My example {i}"

# 首先创建tokenizer并明确设置clean_up_tokenization_spaces
tokenizer = AutoTokenizer.from_pretrained(
    "openai-community/gpt2",
    clean_up_tokenization_spaces=True  # 明确设置参数
)

pipe = pipeline(
   model="openai-community/gpt2", 
   tokenizer=tokenizer,
   device=0,
   pad_token_id=tokenizer.eos_token_id  # 明确设置pad_token_id
)

generated_characters = 0
for out in pipe(data()):
    generated_characters += len(out[0]["generated_text"])

print(generated_characters)
