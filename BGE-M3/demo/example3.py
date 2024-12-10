from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3',use_fp16=False,devices='cpu')

passages = [
    "BGE M3 是一个支持稠密检索、词法匹配和多向量交互的嵌入模型。"
]

passage_embeddings = model.encode(
    passages,
    return_sparse=True,
    return_colbert_vecs=True
)

print(passage_embeddings.keys())

print(passage_embeddings['dense_vecs'].shape) # 稠密向量
print(passage_embeddings['lexical_weights']) # 稀疏向量
print(passage_embeddings['colbert_vecs']) # ColBERT向量

