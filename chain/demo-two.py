from langchain_core.runnables import  RunnableSequence,RunnablePassthrough,RunnableParallel, RunnableLambda,ConfigurableField
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# Defining a simple chain with LCEL
prompt = ChatPromptTemplate.from_template(
    "Provide a definition of the term: {term}"
)

output_parser = StrOutputParser()

# model_name = "qwen2.5:7b"
model_name = "llama3.2-vision:11b"
api_key = "ollama"
base_url = "http://localhost:11434/v1/"
llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_url,
)


chain = prompt | llm | output_parser

# res = chain.invoke("evolution")
# print(res)


# LCEL Primitives :zh LCEL原语

#To implement advanced chains or Retrieval Augmented Generation (RAG) systems in LCEL, it is essential to understand well the LCEL primitives:
# zh: 要在LCEL中实现高级链或检索增强生成（RAG）系统，必须充分了解LCEL原语：

#  RunnablePassthrough is the most basic Runnable primitive. It just passes the key-value pairs unchanged:
# zh: RunnablePassthrough是最基本的Runnable原语。它只是无修改地传递键值对：
# runnable_passthrough = RunnablePassthrough()
# res = runnable_passthrough.invoke({"greeting": "Hello World!"})
# print(res)

# RunnableSequence is about putting Runnables in order. It is a substitute for the pipe | operator in LCEL, which can easily be seen:
# zh: RunnableSequence是关于按顺序放置Runnables。它是LCEL中管道|运算符的替代品，可以轻松地看到：

# runnable_sequence = RunnableSequence(RunnablePassthrough(), RunnablePassthrough())
# print(runnable_sequence)
# 输出：first=RunnablePassthrough() middle=[] last=RunnablePassthrough()
# res = runnable_sequence.invoke({'greeting': 'Hello World!'})
# print(res)
# 輸出：{'greeting': 'Hello World!'}

# RunnableParallel is about running or passing Runnables in parallel, as the name suggests. Any realistic chain will use it. It can accept arguments as key-value pairs or as a dictionary:
# zh: RunnableParallel是关于并行运行或传递Runnables，正如其名称所示。任何现实的链都会使用它。它可以接受键值对或字典作为参数：

# runnable_parallel = RunnableParallel(first=RunnablePassthrough(), second=RunnablePassthrough())
# res = runnable_parallel.invoke({'greeting': 'Hello World!'})
# print(res)
# 输出：{'first': {'greeting': 'Hello World!'}, 'second': {'greeting': 'Hello World!'}}


# RunnableLambda allows for executing lambda function in the chain, taking as input some arguments. In some use cases, you have to access the input arguments to lambda by name because the underlying data model can contain many key-value pairs:
# zh: RunnableLambda允许在链中执行lambda函数，以某些参数作为输入。在某些用例中，您必须按名称访问lambda的输入参数，因为底层数据模型可能包含许多键值对：
# runnable_lambda = RunnableLambda(lambda x:x+1)
# res = runnable_lambda.invoke(1)
# print(res)
# 输出：2


# Note: you have to import itemgetter from standard libary operator:
# from operator import itemgetter
# zh: 注意：您必须从标准库运算符中导入itemgetter：
# from operator import itemgetter

# runnable_lambda = itemgetter("b") | RunnableLambda(lambda x:x+1)
# res = runnable_lambda.invoke({"a": 1, "b": -1})
# print(res)
# 输出：0

# assign is a static method of RunnablePassthrough and RunnableParallel that defines a key-value pair in the data model. The method accepts key-valued arguments and expects Callable or Runnable to be input values.
# zh: assign是RunnablePassthrough和RunnableParallel的静态方法，用于在数据模型中定义键值对。该方法接受键值参数，并期望输入值为Callable或Runnable。

runnable_passthrough = RunnablePassthrough.assign(c=lambda x: x["a"] + 2)
res = runnable_passthrough.invoke({"a": 3})
print(res)
# 输出：{'a': 3, 'c': 5}

# configururable_fields is a method that, with the ConfigurationField class, allows configuring Runnable parameters at runtime. For example, the model component has a temperature that we may want to change dynamically:
# zh: configururable_fields是一个方法，它使用ConfigurationField类，允许在运行时配置Runnable参数。例如，模型组件具有我们可能希望动态更改的温度：


model_with_temperature = ChatOpenAI(
  api_key=api_key,
  base_url=base_url,
  model=model_name,
  temperature=0
).configurable_fields(
    temperature=ConfigurableField(
        id="llm_temperature",
        name="LLM Temperature",
        description="The temperature of the LLM",
    )
)
chain_with_temperature = model_with_temperature | output_parser

res = (
    chain_with_temperature
        .with_config(configurable={"llm_temperature": 1})
        .invoke("Choose a radnom word")
)

print(res)