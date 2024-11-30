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