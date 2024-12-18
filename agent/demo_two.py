from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
import os

# 定义查询订单状态的函数
def query_order_status(order_id):
    if order_id == "12345":
        return "订单 12345 的状态是：已发货，预计送达时间是 3-5 个工作日。"
    else:
        return f"未找到订单 {order_id} 的信息，请检查订单号是否正确。"


# 定义退款政策说明函数
def refund_policy(keyword):
    print("keyword = ", keyword)
    return "我们的退款政策是：在购买后30天内可以申请全额退款，需提供购买凭证。"


# 初始化工具
tools = [TavilySearchResults(max_results=1),
         Tool(
             name="queryOrderStatus",
             func=query_order_status,
             description="根据订单ID查询订单状态"
         ),
         Tool(
             name="refundPolicy",
             func=refund_policy,
             description="查询退款政策内容"
         ),
         ]

# 获取使用的提示
prompt = hub.pull("hwchase17/openai-functions-agent")

# 选择将驱动代理的LLM
# model_name = "llama3"
# model_name="x/llama3.2-vision:11b"
# model_name="gpt-3.5-turbo-1106"
# api_key = os.getenv("agicto_api_key") 
# base_url="https://api.agicto.cn/v1"
model_name="Qwen/Qwen2.5-7B-Instruct"
api_key =  os.getenv("OPENAI_API_KEY")
base_url="https://api.siliconflow.cn/v1/"

# 选择将驱动代理的LLM
llm = ChatOpenAI(
  model=model_name, 
  api_key=api_key, 
  base_url=base_url,
  )
  

# 构建OpenAI函数代理
agent = create_openai_functions_agent(llm, tools, prompt)

# 通过传入代理和工具创建代理执行器
agent_executor = AgentExecutor(
  agent=agent, 
  tools=tools, 
  verbose=True,
)

# 定义一些测试询问
queries = [
    "请问订单12345的状态是什么？",
    "你们的退款政策是什么？"
]

# 运行代理并输出结果
for input in queries:
    response = agent_executor.invoke({"input": input})
    print(f"客户提问：{input}")
    print(f"代理回答：{response}\n")
