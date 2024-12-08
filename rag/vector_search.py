from transformers import BertModel, BertTokenizer
import torch

# 初始化BERT模型和分词器
# 使用BertTokenizer的from_pretrained方法加载预训练的BERT模型，并设置clean_up_tokenization_spaces=False，以避免在分词过程中自动删除多余的空格。
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',clean_up_tokenization_spaces=False)
model = BertModel.from_pretrained('bert-base-uncased')

# 编码文档和查询向量的函数
def encode_texts(texts):
    inputs = tokenizer(
        texts, 
        padding=True,
        truncation=True, 
        return_tensors='pt',  
        # clean_up_tokenization_spaces=False,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    # 使用每个文档的[CLS] token的输出作为文档的向量表示
    return outputs.last_hidden_state[:, 0, :]


# 示例文档
documents = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.", 
    "London is the capital of England."
]

# 编码文档向量
document_vectors = encode_texts(documents)

# 示例查询
query = "What is the capital of France?"
query_vector = encode_texts([query])

def cosine_similarity(query_vector, document_vectors):
    return torch.nn.functional.cosine_similarity(query_vector, document_vectors)

# 计算相似度并获取top-k结果
k = 2
similarities = cosine_similarity(query_vector, document_vectors)
##  对相似度进行排序，descending=True表示降序排列。argsort返回的是相似度值排序后的索引。
top_k_indices = torch.argsort(similarities, descending=True)[:k].tolist()
top_k_docs = [(documents[i], similarities[i].item()) for i in top_k_indices]

## 打印结果  
for doc, score in top_k_docs:
    print(f"Document: {doc}")
    print(f"Similarity Score: {score:.4f}\n")


# # 计算相似度
# similarities = [cosine_similarity(query_vector, doc_vec) for doc_vec in document_vectors]
# # 对相似度进行排序，descending=True表示降序排列。
# top_k_docs = sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True)[:k]

# # 打印结果  
# for doc, score in top_k_docs:
#     print(f"Document: {doc}")
#     print(f"Similarity Score: {score.item():.4f}\n")