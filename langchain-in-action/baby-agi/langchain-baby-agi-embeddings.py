import os
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import TypedDict, List
from langchain_community.vectorstores import FAISS
import operator

# 解决 Hugging Face tokenizers 并行化问题
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

# 定义嵌入模型
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
model_name = "sentence-transformers/all-mpnet-base-v2"

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

vector_db = FAISS.from_texts([""], embeddings)  # 初始化一个空的向量数据库

# 任务分解函数
def decompose_task(state: AgentState):
    task = state['current_task']

    decomposition_prompt = f"""
    将以下任务分解为更小、更具体的子任务:
    主任务: {task}

    请提供3-5个详细的子步骤，每个子步骤占一行。
    """

    response = llm.invoke([HumanMessage(content=decomposition_prompt)])
    print("任务分解结果:")  # 调试信息
    print(response.content)  # 调试信息

    subtasks = response.content.split('\n')
    subtasks = [s.strip() for s in subtasks if s.strip()]  # 去除空行和空白字符

    if not subtasks:  # 如果没有生成子任务，直接返回
        return {
            'tasks': [],
            'current_task': None,
            'iteration': state['iteration'] + 1,
            'completed_tasks': state['completed_tasks'],
            'result': state['result']
        }

    # 将子任务存储到向量数据库
    vector_db.add_texts(subtasks)

    return {
        'tasks': subtasks,
        'current_task': subtasks[0],  # 更新当前任务为第一个子任务
        'iteration': state['iteration'] + 1,
        'completed_tasks': state['completed_tasks'],
        'result': state['result']
    }

# 执行任务函数  
def execute_task(state: AgentState):
    current_task = state['tasks'][0]

    execution_prompt = f"""
    执行以下任务并给出详细结果:
    任务: {current_task}

    请提供具体的执行步骤和结果，确保结果清晰明确。
    """

    response = llm.invoke([HumanMessage(content=execution_prompt)])
    print("任务执行结果:")  # 调试信息
    print(response.content)  # 调试信息

    # 将结果存储到向量数据库
    vector_db.add_texts([response.content])

    return {
        'result': response.content,
        'completed_tasks': state['completed_tasks'] + [current_task],
        'tasks': state['tasks'][1:],
        'current_task': state['tasks'][1] if len(state['tasks']) > 1 else None,
        'iteration': state['iteration']
    }

# 反思和评估函数
def reflect_and_evaluate(state: AgentState):
    if not state['tasks']:  # 如果任务列表为空，直接返回终止状态
        return {
            'result': state['result'],
            'completed_tasks': state['completed_tasks'],
            'tasks': [],
            'current_task': None,
            'iteration': state['iteration']
        }

    # 评估任务完成情况
    reflection_prompt = f"""
    评估已完成任务: {state['completed_tasks']}
    当前结果: {state['result']}

    是否需要调整策略或继续执行?
    """

    response = llm.invoke([HumanMessage(content=reflection_prompt)])
    print("反思结果:")  # 调试信息
    print(response.content)  # 调试信息

    # 根据反思结果决定是否继续
    if "继续" in response.content:
        return state
    else:
        return {
            'result': state['result'],
            'completed_tasks': state['completed_tasks'],
            'tasks': [],
            'current_task': None,
            'iteration': state['iteration']
        }

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
    lambda state: "decompose" if state['tasks'] else END,  # 如果有任务，继续分解；否则结束
    {
        "decompose": "decompose",
        END: END
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
final_state = app.invoke(initial_state)

# 打印最终结果
print("\n最终结果:")
print(final_state['result'])

# 打印向量数据库中的内容
print("\n向量数据库中的内容:")
docs = vector_db.similarity_search("", k=10)  # 检索所有内容
for i, doc in enumerate(docs):
    print(f"文档 {i + 1}: {doc.page_content}")