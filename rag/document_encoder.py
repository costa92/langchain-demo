#  文档编码器（Document Encoder）将预定义知识库中的文档逐一编码成向量。为了提高检索效率，这些向量通常会预先计算并存储起来。
from transformers import BertModel, BertTokenizer


# 加载预训练的BERT模型和分词器，并设置clean_up_tokenization_spaces=False，以避免在分词过程中自动删除多余的空格。
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',clean_up_tokenization_spaces=False)
model = BertModel.from_pretrained('bert-base-uncased')


documents = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "London is the capital of England."
]
document_vectors = []

for doc in documents:
    inputs = tokenizer(doc, return_tensors='pt')
    doc_vector = model(**inputs).last_hidden_state.mean(dim=1)
    document_vectors.append(doc_vector)


print(document_vectors)
