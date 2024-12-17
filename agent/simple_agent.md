# Simple Agent 文档说明

## 概述
本文档描述了一个使用 LangChain 和 Ollama 实现的简单数学计算代理。该代理能够通过自然语言接口执行数学计算。

## 组件说明

### 1. 工具
- **calculate_expression**: 使用 `numexpr` 库的数学计算工具
  - 输入：字符串形式的数学表达式
  - 输出：字符串形式的计算结果
  - 包含优雅的错误处理机制，提供适当的错误信息

### 2. LLM 配置
- 模型：llama3
- 温度值：0（确定性输出）
- 实现：使用 langchain_ollama 中的 OllamaLLM

### 3. 提示模板
代理使用 React 风格的提示模板，包含以下组件：
- 问题格式
- 思考过程
- 动作选择
- 动作输入
- 观察结果
- 最终答案格式

### 4. 代理配置
代理执行器配置了以下参数：
- 最大迭代次数：5
- 每个工具最大执行时间：30秒
- 总最大执行时间：120秒
- 提前停止方法："force"
- 启用详细日志

## 使用示例
```python
user_question = "3233 * 4556 的结果是多少？"
response = agent_executor.invoke({
    "input": user_question,
    "tools": tools_description,
    "tool_names": tool_names,
    "agent_scratchpad": ""
})
```

## 依赖项
- langchain_ollama：用于LLM集成
- langchain：提供代理和工具框架
- numexpr：用于高效的数学表达式计算

## 错误处理
- 捕获并处理 LLM 初始化错误
- 捕获数学表达式计算错误并返回错误消息
- 捕获并记录代理执行错误

## 注意事项
- 代理优化为直接数值响应
- 包含详细日志记录，便于调试
- 通过 numexpr 库支持基本数学运算
