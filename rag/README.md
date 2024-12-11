# RAG（Retrieval-Augmented Generation）

## 定义与原理

RAG（检索增强生成）是一种结合信息检索与生成模型的技术。其核心原理在于，当生成模型需要输出时，不仅依赖于自身的知识和推理能力，还通过信息检索系统从外部知识库获取相关信息，以增强生成结果。这种方法的优势在于能够利用大量非结构化文本数据，为生成模型提供更丰富、准确的背景信息和参考知识。

## 工作原理

### 1. 检索阶段
在此阶段，系统从大规模知识库或文档集合中检索与输入查询相关的信息，以确保生成模型能够获取到最新和最相关的内容。

### 2. 生成阶段
根据检索到的信息，生成高质量的文本输出，确保生成结果的准确性和相关性。

## RAG基本流程

### 1. 知识文档
知识文档可以是Word文档、TXT文件、CSV数据表、Excel表格，甚至是PDF文件、图片和视频。处理这些知识的技术包括：
1. OCR技术将图片和视频转换为文本。

处理文档的工具：
- [Docling](https://github.com/DS4SD/docling)

### 2. 嵌入模型
嵌入模型的核心任务是将文本转换为向量形式。处理知识使用到的技术包括：
1. 使用预训练的BERT模型将文本转换为向量，选择合适的模型以满足不同的需求：
   - Word2Vec：用于生成静态向量。
   - BERT：用于动态的词义理解。

### 3. 向量数据库
向量数据库用于存储和管理生成的文本向量，以便于快速检索。

### 4. 检索模块
检索模块负责从知识库中提取与输入查询相关的信息。

### 5. 生成模块
生成模块根据检索到的信息生成高质量的文本输出。

### 6. 融合模块
融合模块将检索到的信息与生成模块的输出进行整合，生成最终的文本结果。使用注意力机制（attention）来优化信息的融合过程。

## 技术架构

![技术架构](./images/technical_architecture.png)

1. **检索模块**
   - 负责从知识库中检索与输入查询相关的信息。
   - 使用预训练的双塔模型（dual-encoder）进行高效向量化检索。
   - 输出若干与查询相关的文档或段落，作为生成模块的输入。

2. **生成模块**
   - 根据检索到的信息生成高质量文本输出。
   - 使用预训练语言模型（如GPT-3）进行文本生成。
   - 生成模块通常包括解码器（decoder）和编码器（encoder），编码器处理检索到的信息，解码器生成最终文本输出。

3. **融合模块**
   - 将检索到的信息与生成模块的输出进行融合，生成最终文本输出。
   - 使用注意力机制（attention）将检索信息与生成输出融合。

## 使用 LangChain 传教 RAG 项目

```sh
#安装 cli

pip install -U langchain-cli

```

```sh
# 创建项目
langchain app new my-app --package rag-chroma-private
```

```sh
# 运行项目
langchain serve
```

## 参考文献

- [LangChain 中文文档](http://python.langchain.com.cn/)
- [RAG技术架构与实现原理](https://cloud.tencent.com/developer/article/2436421)
- [检索增强生成(RAG)技术方法流程最佳实践实验探索](https://www.53ai.com/news/RAG/2024072130482.html)
- [RAG框架](https://www.53ai.com/news/RAG/2024062056319.html)
- [RAG流程优化（微调）的4个基本策略](https://cloud.tencent.com/developer/article/2433287)
- [读懂RAG这一篇就够了，万字详述RAG的5步流程和12个优化策略](https://juejin.cn/post/7329732000087572520)
- [一文详看Langchain框架中的RAG多阶段优化策略：从问题转换到查询路由再到生成优化](https://mp.weixin.qq.com/s/pK2BRLrWpEKKIPFhUtGvcg)
- [LangChain框架：Hub和LangSmith入门](https://blog.csdn.net/Wufjsjjx/article/details/140798687)
- [大模型从入门到应用——LangChain：模型（Models）-[文本嵌入模型：Embaas、Fake Embeddings、Google Vertex AI PaLM等]](https://blog.csdn.net/hy592070616/article/details/131927016)