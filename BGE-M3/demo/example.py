# 使用 BGE 计算 colbert 的评分

from FlagEmbedding import BGEM3FlagModel
import os
# 设置环境变量以忽略警告信息，避免不必要的控制台输出
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

# 初始化BGEM3FlagModel模型
# 注意：fp16模式仅支持在GPU上运行，若在CPU上运行需将use_fp16参数设为False
model = BGEM3FlagModel(
    'BAAI/bge-m3',  # 模型名称
    use_fp16=False,  # 是否使用fp16模式
    device='cpu',  # 设备类型，'cpu'或'gpu'
    normalize_embeddings=True  # 是否对嵌入进行归一化
)



sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
               "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

output_1 = model.encode(
  sentences_1, 
  return_dense=True, 
  return_sparse=True, 
  return_colbert_vecs=True
)

output_2 = model.encode(
  sentences_2,
  return_dense=True, 
  return_sparse=True, 
  return_colbert_vecs=True
)

# colbert_score中使用了”爱因斯坦求和“（torch.einsum），colbert_score具体算法请参考相关文献
"""
爱因斯坦求和是一种对求和公式简洁高效的记法，其原则是当变量下标重复出现时，即可省略繁琐的求和符号。
例如：
C = einsum('ij,jk->ik', A, B)

注释：->符号就相当于等号，->左边为输入，右边为输出。
"""
print(model.colbert_score(output_1['colbert_vecs'][0], output_2['colbert_vecs'][0]))
print(model.colbert_score(output_1['colbert_vecs'][0], output_2['colbert_vecs'][1]))
