#  使用BGE-M3和K近邻实现语义检索

from FlagEmbedding import BGEM3FlagModel
from sklearn.neighbors import NearestNeighbors

model = BGEM3FlagModel(
    'BAAI/bge-m3',
    use_fp16=False,
    device='cpu',
    normalize_embeddings=True
)

# 注意：单篇文章的字数最大8192个
sentences = [
    "河南大学是一所历史悠久的综合性高校",
    "河南大学软件学院",
    "软件工程是一个专业",
    "我是河南大学的一名学生"
]

# 对文本进行向量编码，使用稠密编码
sentences_vector_dict = model.encode(sentences, return_dense=True)

# 获取稠密编码
dense_vecs = sentences_vector_dict['dense_vecs']


# 初始化最近邻分类器 搜索对象，n_neighbors 为搜索的邻居数 默认为3; algorithm 为搜索算法 默认为auto
# algorithm 可选参数为 ball_tree, kd_tree, brute, auto
nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(dense_vecs)

# 搜索最相似的邻居
distances, indices = nbrs.kneighbors(dense_vecs)

print("indices: ", indices)
print("distances: ", distances)
