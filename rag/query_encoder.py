# 查询编码器  并将其转换为一个固定维度的向量。常用的查询编码器是基于BERT或其变种的模型。
from transformers import BertModel, BertTokenizer


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")


query = "What is the capital of France?"
inputs = tokenizer(query, return_tensors="pt", clean_up_tokenization_spaces=True)
outputs = model(**inputs).last_hidden_state.mean(dim=1)
print(outputs)
