# Chain (链)

Chain（链） 是LangChain中最核心的概念之一（看名字就知道）。简单的说，就是把自然语言输入、关联知识检索、Prompt组装、可用Tools信息、大模型调用、输出格式化等这些LLM 应用中的常见动作，组装成一个可以运行的“链”式过程。链可以直接调用，也可以用来进一步构建更强大的Agent。

在例子 demo.py 文件中把提示（prompt）、大模型（model）、输出解析（output_parser）几个组件使用管道符号“｜”链接在一起，上个组件的输出作为下一个组件的输入，一起形成了一个链。

```python
chain = prompt | llm | output_parser

res = chain.invoke({"topic": "冰淇淋"})

```