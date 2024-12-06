import ollama
import re 
from collections import Counter

# model="x/llama3.2-vision:11b"
model="llama3:8b"
# 设定生成文本的函数
def generate_text(prompt):
    # 使用 ollama 进行对话，要求使用中文回答，并明确指示上下文
    messages = [
        {"role": "system", "content": "根据用户给出上下文提供一种可能解决问题的方案"},
        {"role": "user", "content": prompt}
    ]
    response = ollama.chat(model=model, messages=messages)
    return response['message']['content']


# 统计推理结果并寻找常见的答案
# 统计推理结果并寻找最常见的答案
def extract_answer(text):
    # 尝试从文本中提取最后一个数字作为答案
    matches = re.findall(r'\d+', text)
    if matches:
        # 返回最后一个数字作为答案
        return int(matches[-1])
    return None


if __name__ == "__main__":
    # prompt = "用户说：我电脑的鼠标坏了，请帮我解决"
    # 设定一个多步推理问题，明确要求最后返回一个数字
  prompt = (
      "假设你有一个花园，其中有3个花坛。每个花坛都需要浇水。"
      "如果第一个花坛需要10升水，第二个花坛需要20升水，第三个花坛需要30升水，"
      "请计算总共需要多少升水。"
      "在你的回答中可以包括推理过程，但最终的答案应该是一个数字，表示总水量。"
      "请确保在推理结束时提供最后的数字作为答案。"
  )

  # Self-Consistency: 多次生成并汇总结果
  n_trials = 3
  results = []

  for _ in range(n_trials):
    generated_text = generate_text(prompt)
    if generated_text is not None:
        print(f"推理结果: {generated_text}")
        results.append(generated_text)


  # 提取每次推理的结果
  extracted_answers = [extract_answer(text) for text in results]

  # 计算每个答案出现的次数
  answer_counts = Counter(filter(None, extracted_answers))

  # 找到最常见的答案
  if answer_counts:
      most_common_answer = answer_counts.most_common(1)[0][0]
      print(f"最终确定的答案是：{most_common_answer} 升水。")
  else:
      print("没有找到有效的答案。")
