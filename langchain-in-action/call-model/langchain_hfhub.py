import os


# 导入必要的库
from langchain import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
# 初始化HF LLM
llm = HuggingFaceHub(
    repo_id="google/flan-t5-small",
    model_kwargs={"temperature":0, "max_length":180, 'max_new_tokens' : 120, 'top_k' : 10, 'top_p': 0.95, 'repetition_penalty':1.03}

)

# 创建简单的question-answering提示模板
template = """Question: {question}
              Answer: """

# 创建Prompt          
prompt = PromptTemplate(template=template, input_variables=["question"])

# 调用LLM Chain --- 我们以后会详细讲LLM Chain
llm_chain = prompt | llm

# 准备问题
question = "Can you tell me the capital of russia"

# 调用模型并返回结果
print(llm_chain.invoke({"question": question}))