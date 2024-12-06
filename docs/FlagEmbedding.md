# FlagEmbedding 项目


## BGE-M3
  
  Paper : https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/BGE_M3/BGE_M3.pdf
  Code : https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/BGE_M3


BGE-M3 是第一个具有多功能、多语言和多粒度特性的文本检索模型。

多功能:可以同时执行三种检索功能：单向量检索、多向量检索和稀疏检索。
多语言:支持100多种工作语言。
多粒度:它能够处理不同粒度的输入，从短句子到长达8192个词汇的长文档。
在本项目中，为了提高单一检索模式的性能，提出了一种新的自知识蒸馏方法。 我们优化了批处理策略，支持大批处理大小，这可以在对长文本或大型语言模型进行向量微调时简单使用。 我们还构建了一个用于文档检索的数据集，并提出了一个简单的策略来提高长文本的建模能力。 训练代码和微调数据将在不久的将来开源。



[BGE、FlagEmbedding](https://blog.csdn.net/lovechris00/article/details/138379467)