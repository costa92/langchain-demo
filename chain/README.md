# Chain (链)

Chain（链） 是LangChain中最核心的概念之一（看名字就知道）。简单的说，就是把自然语言输入、关联知识检索、Prompt组装、可用Tools信息、大模型调用、输出格式化等这些LLM 应用中的常见动作，组装成一个可以运行的“链”式过程。链可以直接调用，也可以用来进一步构建更强大的Agent。

**LCEL**即LangChain Express Language， 即LangChain表达语言。这是LangChain提供的一种简洁的、用于组装上述“链”的声明性方式。




在例子 demo.py 文件中把提示（prompt）、大模型（model）、输出解析（output_parser）几个组件使用管道符号“｜”链接在一起，上个组件的输出作为下一个组件的输入，一起形成了一个链。

```python
chain = prompt | llm | output_parser

res = chain.invoke({"topic": "冰淇淋"})

```

最常见的RAG应用来说，使用LCEL也无非是在此之上增加一个检索相关文档的动作，类似：

```python
chain = setup\_and\_retrieval | prompt | model | output\_parser

```

一个简单的RAG应用处理过程:检索关联文档 => 组装Prompt => 调用大模型 => 输出处理。

总结：LCEL就是LangChain提供用来组装Chain的一种简单表示方式。用这种方式组装链，可以自动获得诸如批量、流输出、并行、异步等一系列能力；而且链可以进一步通过LCEL组装成更复杂的链与Agent。

[visualization](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/visualization.ipynb)
[colab-demos](https://github.com/gox6/colab-demos/blob/main/practical-examples/langchain-jump-into-lcel-and-runnables.ipynb?source=post_page-----39eb3596cca1--------------------------------)