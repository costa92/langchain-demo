import os
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import TypedDict, List
import operator

# 定义代理状态
class AgentState(TypedDict):
    tasks: List[str]
    completed_tasks: List[str]
    current_task: str
    result: str
    iteration: int

# 初始化语言模型
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 OPENAI_API_KEY")

model_name = "Qwen/Qwen2.5-7B-Instruct"
base_url = "https://api.siliconflow.cn/v1"

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=base_url,
    model_name=model_name,
)

# 任务分解函数
def decompose_task(state: AgentState):
    task = state['current_task']
    
    # 使用 LLM 分解任务
    decomposition_prompt = f"""
    将以下任务分解为更小、更具体的子任务:
    主任务: {task}
    
    请提供3-5个详细的子步骤。
    """
    
    response = llm.invoke([HumanMessage(content=decomposition_prompt)])
    subtasks = response.content.split('\n')
    
    return {
        'tasks': subtasks,
        'current_task': subtasks[0] if subtasks else None,
        'iteration': state['iteration'] + 1,
        'completed_tasks': state['completed_tasks'],
        'result': state['result']
    }

# 执行任务函数  
def execute_task(state: AgentState):
    current_task = state['tasks'][0]
    
    # 使用 LLM 执行任务
    execution_prompt = f"""
    执行以下任务并给出详细结果:
    任务: {current_task}
    
    请提供具体的执行步骤和结果。
    """
    
    response = llm.invoke([HumanMessage(content=execution_prompt)])
    
    return {
        'result': response.content,
        'completed_tasks': state['completed_tasks'] + [current_task],
        'tasks': state['tasks'][1:],
        'current_task': state['tasks'][1] if len(state['tasks']) > 1 else None,
        'iteration': state['iteration']
    }

# 反思和评估函数
def reflect_and_evaluate(state: AgentState):
    if not state['tasks']:
        return {'result': state['result'], 'completed_tasks': state['completed_tasks'], 'tasks': [], 'current_task': None, 'iteration': state['iteration']}
    
    # 评估任务完成情况
    reflection_prompt = f"""
    评估已完成任务: {state['completed_tasks']}
    当前结果: {state['result']}
    
    是否需要调整策略或继续执行?
    """
    
    response = llm.invoke([HumanMessage(content=reflection_prompt)])
    
    # 根据反思结果决定是否继续
    if "继续" in response.content:
        return state
    else:
        return {'result': state['result'], 'completed_tasks': state['completed_tasks'], 'tasks': [], 'current_task': None, 'iteration': state['iteration']}

# 构建工作流图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("decompose", decompose_task)
workflow.add_node("execute", execute_task)
workflow.add_node("reflect", reflect_and_evaluate)

# 设置边和条件
workflow.set_entry_point("decompose")
workflow.add_edge("decompose", "execute")
workflow.add_edge("execute", "reflect")
workflow.add_conditional_edges(
    "reflect",
    lambda state: "decompose" if state['tasks'] else "END",
    {
        "decompose": "decompose",
        "END": END
    }
)

# 编译图
app = workflow.compile()

OBJECTIVE = "分析一下北京市今天的气候情况，写出鲜花储存策略。"

# 初始化状态
initial_state: AgentState = {
    'tasks': [],
    'completed_tasks': [],
    'current_task': OBJECTIVE,
    'result': "",
    'iteration': 0
}

# 运行 BabyAGI
result = app.invoke(initial_state)
print(result)