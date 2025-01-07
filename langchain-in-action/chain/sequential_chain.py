import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains.sequential import SequentialChain
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# 获取API密钥和模型配置
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
BASE_URL = "https://api.siliconflow.cn/v1"

# 创建OpenAI实例
llm = OpenAI(temperature=0.7, api_key=API_KEY, base_url=BASE_URL, model_name=MODEL_NAME)

# 定义通用的PromptTemplate创建函数
def create_prompt_template(input_vars, template_str):
    return PromptTemplate(input_variables=input_vars, template=template_str)

# 第一个LLMChain：生成鲜花的介绍
introduction_template = """
你是一个植物学家。给定花的名称和类型，你需要为这种花写一个200字左右的介绍。
花名: {name}
颜色: {color}
植物学家: 这是关于上述花的介绍:"""
introduction_prompt = create_prompt_template(["name", "color"], introduction_template)
introduction_chain = RunnableParallel({
    "introduction": introduction_prompt | llm
})

# 第二个LLMChain：根据鲜花的介绍写出鲜花的评论
review_template = """
你是一位鲜花评论家。给定一种花的介绍，你需要为这种花写一篇200字左右的评论。
鲜花介绍:
{introduction}
花评人对上述花的评论:"""
review_prompt = create_prompt_template(["introduction"], review_template)
review_chain = RunnableParallel({
    "review": review_prompt | llm
})

# 第三个LLMChain：根据鲜花的介绍和评论写出一篇自媒体的文案
social_post_template = """
你是一家花店的社交媒体经理。给定一种花的介绍和评论，你需要为这种花写一篇社交媒体的帖子，300字左右。
鲜花介绍:
{introduction}
花评人对上述花的评论:
{review}
社交媒体帖子:
"""
social_post_prompt = create_prompt_template(["introduction", "review"], social_post_template)
social_post_chain = RunnableParallel({
    "social_post_text": social_post_prompt | llm
})

# 总的链：按顺序运行三个链
overall_chain = (
    RunnablePassthrough.assign(introduction=introduction_chain)  # 第一步：生成介绍
    | RunnablePassthrough.assign(review=review_chain)            # 第二步：生成评论
    | social_post_chain                                          # 第三步：生成社交媒体文案
)

# 运行链并打印结果
result = overall_chain.invoke({
    "name": "玫瑰",
    "color": "黑色"
})
print(result)