from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["flower","season"],
    template="{flower}的{season}的花语是？"
)


# init model
import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


API_KEY = os.getenv("OPENAI_API_KEY")
model_name = "Qwen/Qwen2.5-7B-Instruct"
base_url = "https://api.siliconflow.cn/v1"
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=base_url,
    model_name=model_name,
    # n=3  # 生成 3 个候选结果
)

llm_chain = prompt | llm | StrOutputParser()

# 调用链
response = llm_chain.invoke({"flower":"玫瑰","season":"春天"})

print("===============")
print(response)
print("===============")

from langchain_core.runnables import RunnablePassthrough, RunnableSequence,RunnableParallel

# 创建一个外部链，首先通过RunnablePassthrough，然后通过llm_chain
outer_chain = RunnableSequence(
    RunnablePassthrough(),  # 传递输入数据而不做任何修改
    llm_chain  # 调用之前定义的llm_chain
)

res = outer_chain.invoke({"flower":"玫瑰","season":"春天"})
print("===============")
print(res)
print("===============")


# 使用 batch 方法处理输入列表
input_list = [
    {"flower": "玫瑰", "season": "夏季"},
    {"flower": "百合", "season": "春季"},
    {"flower": "郁金香", "season": "秋季"}
]

# 使用 batch 方法
result = llm_chain.batch(input_list)

for res in result:
    print("=====batch start==========")
    print(res)
    print("=====batch end==========")

# 
# 定义多个链
chain_1 = prompt | llm | StrOutputParser()
chain_2 = prompt | llm | StrOutputParser()

# 创建一个并行链，将两个链并行运行
parallel_chain = RunnableParallel({
    "chain_1": chain_1,
    "chain_2": chain_2
})

# 运行并行链
result = parallel_chain.invoke({"flower": "玫瑰", "season": "春季"})
print(result)