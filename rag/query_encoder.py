# 查询编码器  并将其转换为一个固定维度的向量。常用的查询编码器是基于BERT或其变种的模型。
from transformers import BertModel, BertTokenizer


# 加载预训练的BERT模型和分词器，并设置clean_up_tokenization_spaces=False，以避免在分词过程中自动删除多余的空格。
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",clean_up_tokenization_spaces=False)
model = BertModel.from_pretrained("bert-base-uncased")


query = "What is the capital of France?"
inputs = tokenizer(query, return_tensors="pt")
outputs = model(**inputs).last_hidden_state.mean(dim=1)
print(outputs)
