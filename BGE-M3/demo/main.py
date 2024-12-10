import numpy as np
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

# 定义待编码的文本列表，单篇文章的字数最大为8192个
sentences = [
    "文章摘要1",  # 第一个文本
    "文章摘要2"   # 第二个文本
]

# 使用模型对文本进行编码，返回稠密编码、稀疏编码和多向量编码
sentences_vector_dict = model.encode(
    sentences,
    return_dense=True,  # 返回稠密编码
    return_sparse=True,  # 返回稀疏编码（词汇权重）
    return_colbert_vecs=True  # 返回多向量编码（ColBERT）
)

# 打印编码结果，结果为包含三个属性的对象：dense_vecs、lexical_weights、colbert_vecs
print(sentences_vector_dict)

# 提取稠密编码部分
sentences_vector_arr = sentences_vector_dict.get("dense_vecs")

# 打印稠密编码的类型，预期为numpy.ndarray
print(type(sentences_vector_arr))

# 打印稠密编码的维度信息，预期为(2, 1024)
print(np.shape(sentences_vector_arr))

# 打印第一个文本的稠密语义向量
print(sentences_vector_arr[0])


# 使用sentence_transformers比较两个文本的相似度
from sentence_transformers import util
print(util.cos_sim(sentences_vector_arr[0], sentences_vector_arr[1]))