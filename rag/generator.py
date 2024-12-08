# 生成模块（Generator）

# 生成模块使用强大的生成模型（如T5或BART）根据检索到的文档和输入查询生成最终的回答。这些生成模型已经在大规模数据上进行预训练，并且在生成自然语言文本方面表现出色。

# from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import T5ForConditionalGeneration, T5Tokenizer
# from transformers import MarianMTModel, MarianTokenizer

model_name = "t5-small"
# 加载模型和分词器
tokenizer = T5Tokenizer.from_pretrained(model_name,legacy=False,clean_up_tokenization_spaces=False)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 使用MarianMT模型和分词器
# model_name = "Helsinki-NLP/opus-mt-en-zh"
# tokenizer = MarianTokenizer.from_pretrained(model_name)
# model = MarianMTModel.from_pretrained(model_name)

# 输入文本
inputs = tokenizer("translate English to Chinese: The house is wonderful.", return_tensors="pt")

# 生成输出文本
outputs = model.generate(inputs["input_ids"], max_length=40)  # 使用max_length而不是max_new_tokens

# 解码并打印输出
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded_output)
# 生成模块使用强大的生成模型（如T5或BART）根据检索到的文档和输入查询生成最终的回答。这些生成模型已经在大规模数据上进行预训练，并且在生成自然语言文本方面表现出色。

# 生成模块的输入是检索到的文档和用户输入的查询，输出是最终的回答。这个过程通常包括以下几个步骤：

# 1. 文档检索：使用信息检索技术从大量文档中检索出与查询最相关的文档。
# 2. 文档编码：将检索到的文档转换为模型可以理解的向量表示。
# 3. 查询编码：将用户输入的查询转换为模型可以理解的向量表示。
# 4. 生成回答：使用生成模型根据检索到的文档和查询生成最终的回答。

# 多文档融合
# 多文档融合（Multi-Document Fusion）是RAG系统中的一个重要步骤，用于将多个检索到的文档融合成一个更全面和准确的上下文，以便生成更高质量的回答。以下是多文档融合的主要方法：

# 1. 文档摘要：对检索到的文档进行摘要，提取出最重要的信息，减少冗余，提高生成效率。
# 2. 文档融合：将多个文档的内容融合成一个整体，形成一个更全面和准确的上下文。
# 3. 文档排序：根据文档与查询的相关性进行排序，确保生成回答时能够充分利用最相关的信息。
# 4. 文档选择：选择与查询最相关的文档进行融合，避免引入无关信息，提高回答的准确性。 