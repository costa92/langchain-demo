from calendar import c
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import Tool
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
import requests


weather_api_url = 'http://apis.juhe.cn/simpleWeather/query'
weather_api_key = os.getenv("weather_api_key")
def get_weather(city):
  # 在这里实现获取天气信息的逻辑
  apiUrl = weather_api_url  # 接口请求URL
  params = {
    'city': city,  # 城市名称
    'key': weather_api_key,  # API Key
  }
  response = requests.get(apiUrl, params=params)
  # 解析响应结果
  if response.status_code == 200:
    responseResult = response.json()
    # 网络请求成功。可依据业务逻辑和接口文档说明自行处理。
    data = responseResult['result']
    return f"{data['city']} 的天气是 {data['realtime']['info']},温度是 {data['realtime']['temperature']}°C,湿度是 {data['realtime']['humidity']}。"
  else:
    # 网络异常等因素，解析结果异常。可依据业务逻辑自行处理。
    return "无法获取天气信息，请检查城市名称。"

    


  # 初始化工具
tools = [
  TavilySearchResults(max_results=1),
  Tool(
    name='get_weather',
    func=get_weather,
    description='查询天气'  
  )
]


  # 获取使用的提示
prompt = hub.pull("hwchase17/openai-tools-agent")


# 初始化大模型
model_name="qwen2.5:7b"
# model_name="llama3.2-vision:11b"
api_key="ollama"
base_url="http://localhost:11434/v1/"


llm = ChatOpenAI(
  model=model_name, 
  api_key=api_key, 
  base_url=base_url,
  temperature=0
)


# 创建 OpenAI 函数代理
agent = create_openai_tools_agent(llm, tools, prompt)


# 定义一些测试询问
queries = [
    "深圳今天天气情况？",
    "目前市场上黄金的平均售价是多少？"
]

# 通过传入代理和工具创建代理执行器
agent_executor = AgentExecutor(
  agent=agent, 
  tools=tools, 
  verbose=False
)



for query in queries:
    result = agent_executor.invoke({"input": query})
    print(f"提问：{query}")
    print(f"回答：{result.get('output')}\n")