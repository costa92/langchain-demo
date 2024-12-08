# 检索模块实现
# 查询编码器将输入查询转换为向量。常见的实现方法是使用预训练的BERT模型进行编码。以下是使用Python和PyTorch的实现示例：

from transformers import BertTokenizer, BertModel


# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',clean_up_tokenization_spaces=False)
model = BertModel.from_pretrained('bert-base-uncased')


# 输入查询
query = "What is the capital of France?"
inputs = tokenizer(query, return_tensors='pt')

# 获取查询的向量表示
query_vector = model(**inputs).last_hidden_state.mean(dim=1)

# 文档编码

# 文档编码器将知识库中的每个文档编码为向量。这一步通常在离线阶段进行，以便在检索时可以快速计算相似度。
documents = ["Paris is the capital of France.", "Berlin is the capital of Germany."]
document_vectors = []

# 编码每个文档
for doc in documents:
    inputs = tokenizer(doc, return_tensors='pt')
    doc_vector = model(**inputs).last_hidden_state.mean(dim=1)
    document_vectors.append(doc_vector)


# 向量检索
# 检索阶段通过计算查询向量与文档向量之间的相似度，找到与查询最相关的文档。常用的相似度度量是余弦相似度。

import torch

# 计算余弦相似度
def cosine_similarity(vec1, vec2):
    return torch.nn.functional.cosine_similarity(vec1, vec2)

# 找到最相关的文档
k = 3  # 假设我们想要找到最相关的3个文档
similarities = [cosine_similarity(query_vector, doc_vec) for doc_vec in document_vectors]
top_k_docs = sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True)[:k]

for doc, sim in top_k_docs:
    print(f"Document: {doc}\nSimilarity: {sim}\n")

