
# Tokenizer

## 类型：

Tokenizer 有三种类型：

1. Word-based: 基于词的分词器,将文本按照词的边界进行分割,适用于英文等有明确词边界的语言。

2. Character-based: 基于字符的分词器,将文本拆分成单个字符,适用于中文等没有明显词边界的语言。

3. Subword-based: 基于子词的分词器,将词拆分成更小的有意义单位,如 BPE、WordPiece、SentencePiece 等,可以平衡词表大小和表达能力。

##  Word-based 

基于词的分词器(Word-based Tokenizer)存在一个主要问题:会生成非常大的词表(vocabulary)。

以 Transformer-XL 为例,它的词表大小达到了 267,735 个词。这么大的词表会带来两个问题:

1. 模型需要学习一个巨大的词嵌入矩阵(embedding matrix)
2. 增加了模型的空间复杂度和时间复杂度

为了解决这个问题,现代的 transformers 模型通常会控制词表大小,一般不超过 50,000 个词,特别是在只处理单一语言的场景下。

## Character-based Tokenizer

基于字符的分词器会把文本拆分成一个个字符。这种方式有两个优点：

1. 词表非常小，通常只需要几十到几百个字符。比如英文只需要26个字母加上标点符号等。
2. 几乎不会出现未知词(unknown token)，因为任何词都可以用已知的字符组合表示。

但是这种分词方式也有两个主要缺点：

1. 单个字符的语义信息太少。比如英文中单个字母"t"的含义很难理解，但完整的单词"today"就很容易理解。这会导致模型性能下降。
   - 不过这个问题在不同语言中表现不一样。比如中文中单个汉字就包含了比较丰富的语义信息。

2. 会产生过多的token。比如用基于词的分词器，"hello"就是一个token，但用基于字符的分词器，就会变成5个token(h,e,l,l,o)。token数量增加会增加模型的计算负担。

## Subword-based Tokenizer：

Subword-based tokenizer 是一种介于词级 (Word-based) 和字符级 (Character-based) 分词器之间的折中方案。它的工作原理很简单:

- 对于常见的词(如"today"、"hello")，就保持完整不拆分
- 对于不常见的词，就拆分成更小的、有意义的片段

举个例子:

- "football" 这个词可能出现得不多，所以会被拆分成 "foot" 和 "ball" 两部分
- 因为 "foot" 和 "ball" 分别都是常见词，所以它们会保持完整
- 这样既保留了词的语义，又能用较少的词表处理更多的词

这种方法有几个好处:

1. 词表大小可以控制在合理范围内
2. 基本不会出现未知词，因为可以用已知的片段组合表示
3. 即使遇到新词也能处理，比如 "fastfood" 可以拆分成 "fast" + "food"

## Subword Tokenization 算法

有三种常见的 subword tokenization 算法：Byte Pair Encoding: BPE 、WordPiece、Unigram。

### BPE 

Byte Pair Encoding: BPE 来自于论文 《Neural Machine Translation of Rare Words with Subword Units》（2015） 。

BPE 是一种简单的数据压缩技术，它迭代式地替换序列中最频繁的字节对。我们不是合并频繁的字节对，而是合并频繁的字符或字符序列。

首先，我们用 character vocabulary 初始化 symbol vocabulary ，将每个单词表示为一个字符序列，加上一个特殊的单词结束符 </w>，这允许我们在 tokenization 后恢复原始的 tokenization 。

然后，我们迭代地计算所有 symbol pair ，并用新的 symbol 'AB' 替换最频繁的 symbol pair ('A','B') 。每个merge 操作产生一个新的 symbol ，它代表一个 character n-gram 。

同时，每个 merge 代表一个规则。

最终的 symbol vocabulary 大小等于 initial vocabulary 的大小，加上 merge 操作的次数（这是算法唯一的超参数）。

下面的显示了一个最小化的 Python 实现。在实践中，我们通过索引所有 pair 并增量更新数据结构来提高效率：

```py
import collections,re

def get_stats(vocab): # vocab: 储存 word -> freq 的字典
  '''
  计算词表中，字符的 2-gram  及其出现频率
  '''
  pairs = collections.defaultdict(int)
  for word, freq in vocab.items():
    symbols = word.split()  # 将 word 拆分成字符
    for i in range(len(symbols) - 1):
      pairs[symbols[i], symbols[i+1]] += freq # 统计 2-gram 及其出现频率
  return pairs


def merge_vocab(pair, v_in): # pair: 2-gram, v_in: 词表为已有词表
  '''
    利用最高频的 2-gram 来更新已有词表
  '''
  v_out = {}
  bigram = re.escape(' '.join(pair)) # 将 pair 转换成正则表达式
  p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)') # 匹配单词边界
  # \S 表示非空白字符，
  # (?<!\S) 表示非空白字符前缀，当 bigram 出现在单词边界时，匹配成功
  # (?!\S) 表示非空白字符后缀，当 bigram 出现在单词边界时，匹配成功 
  for word in v_in:
      w_out = p.sub(''.join(pair), word) # 将word中已有的pair替换为紧凑版本(移除中间的空格)
      # 注意这里有两个 join(pair), 一个是 ' '.join() 带空格, 另一个是 ''.join() 不带空格
      v_out[w_out] = v_in[word]
  return v_out

if __name__ == '__main__':
  vocab = {'l o w </w>' : 5, 'l o w e r </w>' : 2, # initial vocabulary
            'n e w e s t </w>' : 6, 'w i d e s t </w>' : 3}
  num_merges = 10
  for i in range(num_merges):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(best)

# 输出最终的词表 
print(vocab)
# {'low</w>': 5, 'low e r </w>': 2, 'newest</w>': 6, 'wi d est</w>': 3}
```

注意，初始的 vocab 已经将单词拆分为字符序列，并用 ' ' 分隔。这个步骤被称作 pre-tokenization 。


####  在机器翻译任务上，有两种应用 BPE 的方法：

1. 独立编码方法：
   - 分别为源语言和目标语言学习独立的编码
   - 优点：文本和词表更紧凑，每个 subword 单元在对应语言的训练文本中都有出现

2. 联合编码方法（joint BPE）：
   - 学习源语言和目标语言词表的并集上的编码
   - 优点：提高源语言和目标语言分词的一致性，避免相同词在两种语言中被不同分词的问题，有助于神经模型学习 subword 单元之间的映射

### Byte-level BPE  

### 基础词表大小的问题

传统方法中，如果使用所有 Unicode 字符作为基础词表(base vocabulary)，会导致词表非常大。因为 Unicode 字符有 65536 个(2^16)。

GPT-2 提出了一个巧妙的解决方案:

- 使用字节(byte)作为基础词表,将基础词表大小限制在 256 个(2^8)
- 同时保证能表示所有字符
- GPT-2 最终词表大小为 50257,包含:
  - 256 个基础字节词元(byte tokens)
  - 1 个文本结束符
  - 50000 个通过合并学习到的新词元

对比之下,传统 BPE 方法的 GPT 模型:

- 词表大小为 40478
- 包含 478 个基础字符
- 通过 40000 次合并得到其他词元



## Padding

### 为什么使用 padding 

在使用批量数据训练模型的时候，训练数据的长度并不总是相同，这会导致训练出问题。因为模型输入张量（tensor）需要有一个统一的形状，所以需要通过在较短的句子中添加一个特殊的填充标记（ padding token ） 来确保较短的序列将和批次中最长的序列或模型接受的最大长度相同。

### padding配置策略

padding参数控制文本填充的方式，它可以是布尔值或字符串：

* True或longest：对批次中的所有序列进行填充，使它们的长度与批次中最长的序列一致（如果你只提供了单个序列，则不应用任何填充）。

* max_length：将所有序列填充或截断到指定的max_length长度，或者如果没有提供max_length（即max_length=None），则填充到模型接受的最大长度。即使你只提供了单个序列，依然会应用填充。

* False或 'do_not_pad'（默认） ：不进行任何填充。


设置参数padding='max_length'并且 max_length=2， max_length 比批次中最短序列的长度还小：

```python


encoded_input = tokenizer(batch_sentences, padding='max_length', max_length=2)
print(f'encoded_input的类型为：{type(encoded_input)}')
print('输出encoded_input如下：')
print(encoded_input)

# tokens() 查看这个批次中第batch_index个文本字符串分词之后的原始tokens列表，你可以使用这个tokens方法来获取，这对于理解分词器如何将文本切分成tokens或者对分词器的输出进行调试非常有用。
tokens1 = encoded_input.tokens()  # 默认是0
tokens2 = encoded_input.tokens(1)
tokens3 = encoded_input.tokens(2)
print(f'第一个句子的tokens为：{tokens1}')
print(f'第二个句子的tokens为：{tokens2}')
print(f'第三个句子的tokens为：{tokens3}')



# 输出encoded_input如下：
# {'input_ids': [[3983, 1128, 911, 2086, 17496, 30], [8002, 944, 1744, 566, 8788, 911, 2086, 17496, 11, 77382, 13], [3838, 911, 11964, 724, 550, 30]], 'attention_mask': [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]}
# 第一个句子的tokens为：['But', 'Ġwhat', 'Ġabout', 'Ġsecond', 'Ġbreakfast', '?']
# 第二个句子的tokens为：['Don', "'t", 'Ġthink', 'Ġhe', 'Ġknows', 'Ġabout', 'Ġsecond', 'Ġbreakfast', ',', 'ĠPip', '.']
# 第三个句子的tokens为：['What', 'Ġabout', 'Ġelev', 'ens', 'ies', '?']


```

当设置padding='max_length' 时，对上面max_length 设置不同长度的情形总结：

* 设置 max_length 比批次中最短序列的长度还小时，不会对序列进行填充，和 do_not_pad 策略一致。
* 设置 max_length 比批次中最短序列的长度大，比最长序列小时，只会对小于max_length的序列进行填充。
* 设置 max_length 比最长序列的长度大时，会将批次中所有的序列填充到 max_length 指定的长度。
* 设置 max_length=None 或者不指定 max_length 时，会将批次中所有的序列填充到模型接受的最大长度。

当设置 padding='do_not_pad' 或者 padding=False或者不设置 padding参数，则不会进行填充。


##  Truncation 截断


在使用tokenizer对文本数据进行处理时，截断（truncation）是一个重要的步骤，主要有以下几个原因：

1. 模型输入长度限制
- 大多数基于Transformer的模型(如BERT、GPT等)都有最大输入长度限制
- 例如BERT通常限制为512个token
- 超过长度限制的文本必须截断到模型可接受的长度

2. 计算资源优化
- 处理长序列需要更多计算资源和时间
- 截断可以使文本长度一致或控制在合理范围内
- 有助于优化计算资源使用、加快训练和推理
- 控制内存使用

3. 提升模型表现
- 并非所有文本信息对特定任务都同等重要
- 截断可以帮助模型聚焦于关键信息
- 例如在情感分析中,文本开头和结尾可能更重要
- 在某些情况下可以提高模型性能

4. 数据一致性
- 训练和推理阶段需要保持输入数据的一致性
- 截断可以使所有文本输入长度相同
- 有助于维持模型的稳定性和准确性


## 参考文档



[Tokenizer](https://www.huaxiaozhuan.com/%E5%B7%A5%E5%85%B7/huggingface_transformer/chapters/1_tokenizer.html)
[Tokenizer 使用介绍](https://juejin.cn/post/7365704546100396084)

[huggingface transformers](https://huggingface.co/docs/transformers/main/zh/main_classes/tokenizer)