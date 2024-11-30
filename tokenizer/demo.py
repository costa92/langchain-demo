from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")

# tokenizer = AutoTokenizer.from_pretrained("ollama/llama3")
print("tokenizer:", tokenizer)

# batch_sequences = [
#   "Hello, world!",
#   "But what about second breakfast?",
# ]


batch_sequences = [
    "This sentence is not too long but we are going to split it anyway.",
    "This sentence is shorter but will still get split.",
]


# 编码 填充 截断 返回张量  
# encoded_input = tokenizer(batch_sequences, padding=True, truncation=True, return_tensors="pt")

encoded_input = tokenizer(
  batch_sequences, 
  # padding='max_length',
  truncation=True, # 截断
  max_length=6, # 最大长度  
  stride=2, # 步长
  return_overflowing_tokens=True, # 返回溢出的token
  return_offsets_mapping=True, # 返回偏移映射
  return_token_type_ids=True, # 返回token类型id
  padding=True, # 填充
)


print("encoded_input:", encoded_input)

print("-"*100)
# 打印每个序列的token
for ids in encoded_input["input_ids"]:
  print(tokenizer.decode(ids))

print("-"*100)
print(encoded_input["overflow_to_sample_mapping"])

# tokens() 返回一个列表，每个元素是一个字符串，表示一个token
tokens = encoded_input.tokens()
tokens1 = encoded_input.tokens(1)
print("tokens:", tokens)
print("tokens1:", tokens1)