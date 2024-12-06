# agent

## agent 是什么

"大模型的出现为AI Agent提供了“聪明的大脑”，并重新定义了AI Agent" 当前，由大模型驱动的AI Agent架构是比较常见的AI Agent落地架构，包含四大要素：规划（Planning）、记忆（Memory）、工具（Tools）、执行（Action）。

![alt text](./images/agent.png)

## agent 框架

![alt text](./images/agent_framework.png)

1. LangChain
   - [GitHub Agent](https://github.com/langchain-ai/langchain/tree/master/libs/langchain/langchain/agents)

2. LangGraph
   - [GitHub Agent](https://github.com/langchain-ai/langgraph)
   - [文档](https://langchain-ai.github.io/langgraph/)

3. CrewAI
   - [GitHub Agent](https://github.com/crewAIInc/crewAI)
   - [文档](https://docs.crewai.com/)

4. Semantic Kernel
   - [GitHub](https://github.com/microsoft/semantic-kernel)
   - [文档](https://learn.microsoft.com/en-us/semantic-kernel/)

5. AutoGen
   - [GitHub](https://github.com/microsoft/autogen)
   - [文档](https://microsoft.github.io/autogen/docs/Getting-Started/)


总结： 这个五个框架的优缺点

1. **LangChain**
   - **优点**：提供了强大的链式推理能力，支持多种语言模型的集成，易于扩展和定制，适合构建复杂的对话系统和推理任务。
   - **缺点**：对于初学者可能有一定的学习曲线，配置和调试可能较为复杂。
   - **适应场景**：适合需要多步骤推理和复杂逻辑的应用，如智能客服和知识问答系统。

2. **LangGraph**
   - **优点**：通过图形化界面简化了与语言模型的交互，支持任务链和代理的创建，易于可视化和理解。
   - **缺点**：功能相对较新，可能在某些高级功能上不如其他成熟框架稳定。
   - **适应场景**：适合需要可视化推理过程和任务管理的应用，如教育和培训工具。

3. **CrewAI**
   - **优点**：专注于团队协作和任务管理，提供了良好的团队协作功能，适合多用户环境。
   - **缺点**：可能在单用户场景下的功能不够强大，依赖于团队的有效协作。
   - **适应场景**：适合需要团队协作的项目管理和任务分配应用。

4. **Semantic Kernel**
   - **优点**：强调语义理解和知识图谱的结合，能够处理复杂的语义任务，支持多种语言模型。
   - **缺点**：实现和配置可能较为复杂，需要对语义理解有一定的背景知识。
   - **适应场景**：适合需要深度语义理解和知识推理的应用，如智能搜索和推荐系统。

5. **AutoGen**
   - **优点**：自动化生成代码和文档，能够提高开发效率，支持多种编程语言。
   - **缺点**：生成的代码可能需要手动调整，自动化程度可能不够高。
   - **适应场景**：适合需要快速原型开发和文档生成的场景，如初创企业和快速迭代的项目。

## 参考文档

[LangChain 中文](http://python.langchain.com.cn/)
