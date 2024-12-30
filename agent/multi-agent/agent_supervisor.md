# 多代理系统监督者

该代码实现了一个多代理系统的监督者（Supervisor），用于管理多个代理之间的对话和任务分配。以下是代码的主要功能和结构概述：

## 工具定义

- **搜索工具**：使用 `TavilySearchResults` 进行搜索，最多返回5个结果。
- **代码执行工具**：使用 `PythonREPL` 执行Python代码，并处理可能的异常。

## 代理监督者

- **路由类型**：定义了一个 `Router` 类型，用于指定下一个要执行的代理或结束任务。
- **语言模型**：使用 `ChatOpenAI` 模型作为语言模型，通过环境变量获取API密钥和基础URL。
- **监督节点**：定义了 `supervisor_node` 函数，根据用户请求和当前状态，决定下一个要执行的代理。


## 代理节点

- **研究节点 (`research_node`)**: 该函数调用 `research_agent` 来执行研究任务。`research_agent` 是通过 `create_react_agent` 创建的，使用了 `llm` 作为语言模型，并配置了 `tavily_tool` 作为工具，同时设置了状态修改器 `"You are a researcher. DO NOT do any math."`。

- **代码节点 (`code_node`)**: 该函数调用 `code_agent` 来执行代码任务。`code_agent` 也是通过 `create_react_agent` 创建的，使用了 `llm` 作为语言模型，并配置了 `python_repl_tool` 作为工具。

- **代理创建**: 使用 `create_react_agent` 函数创建了两个代理：`research_agent` 和 `code_agent`。每个代理都指定了相应的工具和状态修改器，以确保它们能够正确地执行各自的任务。

## 状态图构建

- **状态图构建**：使用 `StateGraph` 类构建状态图，定义了从开始节点 (`START`) 到监督者节点 (`supervisor`) 的初始连接。随后，定义了监督者节点 (`supervisor`)、研究节点 (`researcher`) 和代码节点 (`coder`)，并设置了它们之间的连接。最终，编译状态图以准备处理用户请求。
- **图像保存**：编译状态图并保存为图像文件 `output.png`。

## 循环处理

- **请求处理**：使用 `graph.stream` 方法处理用户请求，并在每个状态变化时打印输出。

该代码展示了如何使用LangChain库构建一个多代理系统，并通过状态图管理代理之间的交互和任务分配。