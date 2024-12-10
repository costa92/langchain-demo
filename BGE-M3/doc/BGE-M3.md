# BGE-M3 简介

BGE-M3 是一个多功能、多语言、多粒度的嵌入模型，具有以下特点：

1. 多功能性：
   - 支持稠密检索（Dense Retrieval）
   - 支持词汇匹配（Lexical Matching）
   - 支持多向量交互（Multi-vector Interaction）
   - 可用于文本相似度计算、语义搜索等任务

2. 多语言支持：
   - 支持100+种语言
   - 在多语言基准测试中表现优异
   - 能够处理跨语言的语义匹配任务

3. 多粒度表示：
   - 提供稠密向量表示（Dense Vectors）
   - 提供稀疏向量表示（Sparse Vectors）
   - 支持ColBERT式的多向量表示（Multi-vector Representation）

4. 高性能：
   - 在多个评估基准上达到领先水平
   - 支持高效的批处理和并行计算
   - 可选择性使用FP16加速（仅GPU环境）

5. 易用性：
   - 提供简单直观的API接口
   - 与主流深度学习框架兼容
   - 支持CPU和GPU环境部署

## 检索流程的建议：

1. 数据预处理：
   - 对文档进行分段处理，确保每段长度适中（建议不超过8192个字符）
   - 清理文本中的噪声数据，如特殊字符、冗余空格等
   - 根据需要进行文本规范化处理

2. 向量编码：
   - 使用BGE-M3对文档进行编码，获取稠密向量表示
   - 根据任务需求，选择是否同时获取稀疏向量和ColBERT向量
   - 对于大规模数据，建议使用批处理方式进行编码

3. 索引构建：
   - 选择合适的索引方法（如KNN、Faiss等）
   - 对稠密向量建立索引，优化检索效率
   - 考虑使用混合检索策略，结合稀疏匹配提高准确性

4. 查询处理：
   - 对用户查询进行同样的编码处理
   - 使用索引进行高效的相似度搜索
   - 根据任务特点选择合适的相似度计算方法

5. 结果优化：
   - 对检索结果进行排序和过滤
   - 结合业务规则进行结果重排
   - 考虑添加多样性约束，避免结果过于相似

6. 性能优化建议：
   - 合理设置批处理大小，平衡效率和资源消耗
   - 根据硬件条件选择合适的计算精度（FP16/FP32）
   - 适当使用缓存机制提高响应速度
   - 对于大规模系统，考虑分布式部署方案

## 混合检索与重排序策略

1. 混合检索方案：

   a) 稠密向量检索：
      - 使用BGE-M3的dense_vecs进行初步检索
      - 通过KNN或Faiss等索引快速获取候选集
      - 适合捕捉语义相似性

   b) 稀疏向量匹配：
      - 利用BGE-M3的lexical_weights进行词汇匹配
      - 类似传统的BM25算法，关注关键词匹配
      - 适合处理专有名词和精确匹配需求

   c) 混合策略：
      - 分别获取稠密检索和稀疏匹配的结果
      - 使用加权方式融合两种结果
      - 可动态调整权重以平衡召回和精度

2. 重排序机制：

   a) 初筛重排：
      - 使用轻量级模型对候选集进行初步重排
      - 考虑文本长度、时效性等基础特征
      - 快速过滤明显不相关的结果

   b) 精排策略：
      - 使用ColBERT多向量表示进行精确重排
      - 计算查询与文档的细粒度相似度
      - 考虑上下文信息和语义对齐度

   c) 业务规则整合：
      - 结合业务特定的排序规则
      - 引入时效性、权威性等外部特征
      - 支持自定义的排序策略

3. 实现建议：

   a) 检索流程：
   
      ```python
      # 混合检索示例
      def hybrid_search(query, documents, dense_weight=0.7, sparse_weight=0.3):
          # 1. 编码查询和文档
          query_vectors = model.encode(query, return_dense=True, return_sparse=True)
          doc_vectors = model.encode(documents, return_dense=True, return_sparse=True)
          
          # 2. 稠密检索得分
          dense_scores = compute_dense_scores(query_vectors['dense_vecs'], 
                                           doc_vectors['dense_vecs'])
          
          # 3. 稀疏匹配得分
          sparse_scores = compute_sparse_scores(query_vectors['lexical_weights'],
                                             doc_vectors['lexical_weights'])
          
          # 4. 混合得分
          final_scores = dense_weight * dense_scores + sparse_weight * sparse_scores
          
          return final_scores
      ```

   b) 重排序流程：

      ```python
      def rerank_results(query, candidates, model):
          # 1. 获取ColBERT表示
          query_colbert = model.encode(query, return_colbert_vecs=True)
          candidates_colbert = model.encode(candidates, return_colbert_vecs=True)
          
          # 2. 计算精确相似度
          similarity_scores = []
          for candidate_vec in candidates_colbert['colbert_vecs']:
              score = model.colbert_score(query_colbert['colbert_vecs'][0], 
                                        candidate_vec)
              similarity_scores.append(score)
          
          # 3. 结合其他特征进行最终排序
          final_scores = combine_features(similarity_scores, other_features)
          
          return sorted_results(candidates, final_scores)
      ```

4. 优化建议：

   - 使用异步处理提高并发性能
   - 实现结果缓存机制减少计算开销
   - 采用分层检索策略，逐步细化结果
   - 定期评估和调整混合权重
   - 建立完善的评估指标体系
5. 技术规格：

   | 模型名 | 维度 | 序列长度 | 介绍 |
   |--------|------|----------|------|
   | BAAI/bge-m3 | 1024 | 8192 | 多语言，基于统一的微调（密集、稀疏、ColBERT） |
   | BAAI/bge-m3-unsupervised | 1024 | 8192 | 对比学习训练，来自 bge-m3-retromae |
   | BAAI/bge-large-en-v1.5 | 1024 | 512 | 英文模型 |
   | BAAI/bge-base-en-v1.5 | 768 | 512 | 英文模型 |
   | BAAI/bge-small-en-v1.5 | 384 | 512 | 英文模型 |


6. 生成密集嵌入:

   ```python
   from FlagEmbedding import BGEM3FlagModel
   
   # 初始化模型
   model = BGEM3FlagModel(
       'BAAI/bge-m3',
       use_fp16=False,  # 是否使用半精度
       device='cpu',    # 设备选择
       normalize_embeddings=True  # 是否对嵌入向量进行归一化
   )

   # 准备文本
   texts = [
       "这是第一个示例文本",
       "这是第二个示例文本",
       "这是第三个示例文本"
   ]

   # 生成密集嵌入向量
   embeddings = model.encode(
       texts,
       batch_size=32,     # 批处理大小
       return_dense=True, # 返回密集向量
       max_length=8192    # 最大序列长度
   )

   # 获取密集向量
   dense_vectors = embeddings['dense_vecs']

   # dense_vectors的形状为 [文本数量, 向量维度]
   # 对于bge-m3模型，向量维度为1024
   print(f"生成的向量维度: {dense_vectors.shape}")
   ```

   注意事项：
   - 文本长度不应超过8192个token
   - 可以通过调整batch_size来平衡内存使用和处理速度
   - normalize_embeddings=True可以使向量更适合余弦相似度计算
   - 返回的dense_vectors可以直接用于相似度计算或构建向量索引

7. 生成稀疏嵌入:

   ```python
   from FlagEmbedding import BGEM3FlagModel
   
   # 初始化模型
   model = BGEM3FlagModel(
       'BAAI/bge-m3',
       use_fp16=False,
       device='cpu',
       normalize_embeddings=True
   )

   # 准备文本
   texts = [
       "这是第一个示例文本",
       "这是第二个示例文本",
       "这是第三个示例文本"
   ]

   # 生成稀疏嵌入向量
   embeddings = model.encode(
       texts,
       batch_size=32,
       return_sparse=True,  # 返回稀疏向量
       max_length=8192
   )

   # 获取稀疏向量(词汇权重)
   lexical_weights = embeddings['lexical_weights']

   # lexical_weights是一个稀疏矩阵列表
   # 每个元素代表一个文本的词汇权重
   # 可以使用scipy.sparse将其转换为CSR矩阵进行操作
   from scipy.sparse import csr_matrix
   
   # 转换第一个文本的词汇权重为CSR矩阵
   sparse_matrix = csr_matrix(lexical_weights[0])
   print(f"稀疏矩阵的形状: {sparse_matrix.shape}")
   print(f"非零元素数量: {sparse_matrix.nnz}")
   ```

   注意事项：
   - 稀疏向量表示词汇级别的匹配信息
   - 返回的lexical_weights是一个稀疏矩阵列表
   - 可以与密集向量结合使用，实现混合检索
   - 稀疏向量特别适合处理关键词匹配和专有名词检索


8. 生成多向量嵌入:

   ```python
   from FlagEmbedding import BGEM3FlagModel
   
   # 初始化模型
   model = BGEM3FlagModel(
       'BAAI/bge-m3',
       use_fp16=False,
       device='cpu',
       normalize_embeddings=True
   )

   # 准备文本
   texts = [
       "人工智能是计算机科学的一个重要分支",
       "机器学习是人工智能的核心技术之一" 
   ]

   # 生成多向量嵌入(ColBERT向量)
   embeddings = model.encode(
       texts,
       batch_size=32,
       return_colbert_vecs=True,  # 返回ColBERT向量
       max_length=8192
   )

   # 获取ColBERT向量
   colbert_vecs = embeddings['colbert_vecs']

   # colbert_vecs是一个列表,每个元素是一个文本的多向量表示
   # 每个文本被编码为多个向量,可以用于精细粒度的文本匹配
   print(f"第一个文本的ColBERT向量形状: {colbert_vecs[0].shape}")

   # 计算两个文本的ColBERT相似度得分
   score = model.colbert_score(colbert_vecs[0], colbert_vecs[1])
   print(f"ColBERT相似度得分: {score}")
   ```

   注意事项:
   - ColBERT向量为每个token生成一个向量表示
   - 可以捕获更细粒度的语义信息
   - 适合需要精确匹配的场景
   - 计算相似度时使用特殊的ColBERT评分函数
   - 存储和计算开销较大,需要权衡效率和效果

9. 文本对评分:

   ```python
   from FlagEmbedding import BGEM3FlagModel

   # 初始化模型
   model = BGEM3FlagModel(
       'BAAI/bge-m3',
       use_fp16=False,
       device='cpu',
       normalize_embeddings=True
   )

   # 准备文本对
   texts_1 = ["什么是BGE M3?", "BM25的定义是什么?"]
   texts_2 = [
       "BGE M3是一个支持密集检索、词汇匹配和多向量交互的嵌入模型。",
       "BM25是一个基于词袋模型的检索函数,根据查询词在文档中的出现情况对文档进行排序。"
   ]

   # 生成文本向量表示
   output_1 = model.encode(
       texts_1,
       return_dense=True,    # 返回密集向量
       return_sparse=True,   # 返回稀疏向量
       return_colbert_vecs=True  # 返回ColBERT向量
   )

   output_2 = model.encode(
       texts_2,
       return_dense=True,
       return_sparse=True,
       return_colbert_vecs=True
   )

   # 计算不同类型的相似度得分
   for i in range(len(texts_1)):
       # ColBERT得分
       colbert_score = model.colbert_score(
           output_1['colbert_vecs'][i], 
           output_2['colbert_vecs'][i]
       )
       print(f"\n文本对 {i+1}:")
       print(f"问题: {texts_1[i]}")
       print(f"答案: {texts_2[i]}")
       print(f"ColBERT得分: {colbert_score}")
   ```

   注意事项:
   - 可以同时获取密集向量、稀疏向量和ColBERT向量
   - ColBERT得分通过特殊的评分函数计算
   - 支持批量处理多个文本对
   - 评分结果可用于相似度排序和筛选
   - 不同向量类型的得分可以根据需要组合使用


## 参考文档

[BGE-M3 一个多功能、多语言、多粒度的语言向量模型](https://blog.csdn.net/weixin_41046245/article/details/142215886)
