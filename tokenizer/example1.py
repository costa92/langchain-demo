

def tokenized_text():
  tokenized_text = "I like NLP".split()
  print(tokenized_text)
  #  输出：['I', 'like', 'NLP']
  tokenized_text_2 = "我 喜欢 NLP".split()
  print(tokenized_text_2)
  #  输出：['我', '喜欢', 'NLP']


if __name__ == "__main__":
  tokenized_text()