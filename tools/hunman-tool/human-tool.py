

from langchain_openai import ChatOpenAI,OpenAI
from langchain.agents import initialize_agent,AgentType
from langchain_community.agent_toolkits.load_tools import load_tools
model_name="Qwen/Qwen2.5-7B-Instruct"
# model_name="Qwen/Qwen2-VL-72B-Instruct"
import os
api_key =  os.getenv("OPENAI_API_KEY")
base_url="https://api.siliconflow.cn/v1/"

# We will use this model for both the conversation and the summarization
# :zh: 我们将使用这个模型来进行对话和总结
llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_url,
    temperature=0
)

# 匹配模型
math_llm = OpenAI(model=model_name, api_key=api_key,base_url=base_url,temperature=0)
# tools = load_tools(["human"],  llm=math_llm)


# agent_chain = initialize_agent(
#     tools=tools,
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True
# )

# res = agent_chain.invoke({"input":"你是谁,我需要知道你的名字"})
# print(res)


# Configuring the Input Function
# 配置输入函数
def get_input() -> str:
    print("Insert your text. Enter 'q' or press Ctrl+D (or Ctrl+Z and Enter on Windows) to end.")
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == 'q':
            break
        contents.append(line)
    return "\n".join(contents)

# You can modify the tool when loading
# 你可以修改工具

tools = load_tools(["human", "ddg-search"], llm=math_llm, input_func=get_input)

# # Or you can directly instantiate the tool
from langchain_community.tools import HumanInputRun
tool = HumanInputRun(input_func=get_input)


agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

res = agent_chain.invoke({"input":"我需要帮助引用引文"})
print(res)
