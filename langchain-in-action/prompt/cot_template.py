import os

API_KEY = os.getenv("OPENAI_API_KEY")
model_name="Qwen/Qwen2.5-7B-Instruct"
base_url="https://api.siliconflow.cn/v1"

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(api_key=API_KEY,model_name=model_name,base_url=base_url,temperature=0.1)

# 设定 AI 角色和目标
role_template = """你是一个为花店电商公司AI助手，你的目标是帮助客户根据他们的喜好做出明智的决定"""

# Cot 的关键部分， AI 解释推理过程。并加入先前的对话示例（few shot learning ）
#   
cot_template = """
作为一个为电商公司工作的AI助手，我的目标是帮助客户根据他们的喜好做出明智的决定。

我会按部就班的思考，先理解客户的需求，然后考虑各种鲜花的涵义，最后根据这个需求，给出我的推荐。
同时，我也会向客户解释我这样推荐的原因。

示例 1:
  人类：我想找一种象征爱情的花。
  AI：首先，我理解你正在寻找一种可以象征爱情的花。在许多文化中，红玫瑰被视为爱情的象征，这是因为它们的红色通常与热情和浓烈的感情联系在一起。因此，考虑到这一点，我会推荐红玫瑰。红玫瑰不仅能够象征爱情，同时也可以传达出强烈的感情，这是你在寻找的。

示例 2:
  人类：我想要一些独特和奇特的花。
  AI：从你的需求中，我理解你想要的是独一无二和引人注目的花朵。兰花是一种非常独特并且颜色鲜艳的花，它们在世界上的许多地方都被视为奢侈品和美的象征。因此，我建议你考虑兰花。选择兰花可以满足你对独特和奇特的要求，而且，兰花的美丽和它们所代表的力量和奢侈也可能会吸引你。
"""

from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate

system_message_role = SystemMessagePromptTemplate.from_template(role_template)
system_message_cot = SystemMessagePromptTemplate.from_template(cot_template)


# 用户的询问
human_template = "{human_input}"
human_message = HumanMessagePromptTemplate.from_template(human_template)
# 将以上所有信息结合为一个聊天提示
chat_prompt = ChatPromptTemplate.from_messages([system_message_role,system_message_cot,human_message])
prompt = chat_prompt.format_prompt(human_input="我想为我的女朋友购买一些花。她喜欢粉色和紫色。你有什么建议吗?").to_messages()
# 下面调用模型，把提示消息传入模型，生成结果
res = llm.invoke(prompt)
print(res.content)

