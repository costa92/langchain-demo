# BabyAGI 任务执行系统

## 概述

该程序实现了一个基于任务分解、执行和反思的 BabyAGI 系统。通过使用 OpenAI 的 LLM 和 Hugging Face 的嵌入模型，系统能够将复杂任务分解为子任务，逐步执行，并将结果存储到向量数据库中。

## 主要功能

1. **任务分解**：将主任务分解为多个子任务。
2. **任务执行**：执行子任务并生成详细结果。
3. **反思与评估**：评估任务完成情况，决定是否继续执行。
4. **向量数据库存储**：将任务分解和执行结果存储到向量数据库中。

## 代码结构

### 1. 代理状态定义

```python
class AgentState(TypedDict):
    tasks: List[str]  # 待完成任务列表
    completed_tasks: List[str]  # 已完成任务列表
    current_task: str  # 当前任务
    result: str  # 任务执行结果
    iteration: int  # 迭代次数
```

### 2. 任务分解函数

```python
def decompose_task(state: AgentState):
    # 使用 LLM 分解任务
    response = llm.invoke([HumanMessage(content=decomposition_prompt)])
    subtasks = response.content.split('\n')
    # 将子任务存储到向量数据库
    vector_db.add_texts(subtasks)
    return {
        'tasks': subtasks,
        'current_task': subtasks[0],
        'iteration': state['iteration'] + 1,
        'completed_tasks': state['completed_tasks'],
        'result': state['result']
    }
```

### 3. 任务执行函数

```python
def execute_task(state: AgentState):
    # 使用 LLM 执行任务
    response = llm.invoke([HumanMessage(content=execution_prompt)])
    # 将结果存储到向量数据库
    vector_db.add_texts([response.content])
    return {
        'result': response.content,
        'completed_tasks': state['completed_tasks'] + [current_task],
        'tasks': state['tasks'][1:],
        'current_task': state['tasks'][1] if len(state['tasks']) > 1 else None,
        'iteration': state['iteration']
    }
```

### 4. 反思与评估函数

```python
def reflect_and_evaluate(state: AgentState):
    # 评估任务完成情况
    response = llm.invoke([HumanMessage(content=reflection_prompt)])
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
```

### 5. 工作流图构建

```python
workflow = StateGraph(AgentState)
workflow.add_node("decompose", decompose_task)
workflow.add_node("execute", execute_task)
workflow.add_node("reflect", reflect_and_evaluate)
workflow.set_entry_point("decompose")
workflow.add_edge("decompose", "execute")
workflow.add_edge("execute", "reflect")
workflow.add_conditional_edges(
    "reflect",
    lambda state: "decompose" if state['tasks'] else END,
    {
        "decompose": "decompose",
        END: END
    }
)
app = workflow.compile()
```

## 运行流程

1. 初始化：设置初始任务和目标。
2. 任务分解：将主任务分解为子任务。
3. 任务执行：执行子任务并生成结果。
4. 反思与评估：评估任务完成情况，决定是否继续。
5. 终止：当所有任务完成或反思结果决定终止时，结束流程。

## 总结

通过 BabyAGI 系统，可以自动化地分解、执行和反思复杂任务。任务分解和执行结果存储到向量数据库中，便于后续检索和分析。工作流图清晰地展示了系统的运行逻辑。
