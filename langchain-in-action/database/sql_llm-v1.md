# 文档：`sql_llm-v1.py` 代码总结

## 概述

该脚本使用 LangChain 和 OpenAI 的 ChatGPT 模型来自动生成 SQL 查询，这些查询基于用户提出的自然语言问题。它连接到一个 SQLite 数据库，并执行生成的查询以获取结果。

## 主要组件

1. **环境设置**
   - 设置 OpenAI API 密钥以使用 ChatGPT 模型。
   - 连接到 SQLite 数据库 `FlowerShop.db`。

2. **模型初始化**
   - 使用 DeepSeek 的 Chat API 初始化语言模型。
   - 设置模型的基本参数，如 API 密钥和基础 URL。

3. **提示模板创建**
   - 创建一个自定义的提示模板，用于指导模型生成正确的 SQL 查询。

4. **查询清理函数**
   - 定义一个函数 `clean_sql_query` 来清理模型生成的 SQL 查询，移除不必要的 Markdown 标记。

5. **SQL 查询链**
   - 构建一个查询链，该链通过 LangChain 的 `RunnablePassthrough` 和 `StrOutputParser` 处理模型的输出，生成并执行 SQL 查询。

6. **查询执行**
   - 定义一系列问题，使用查询链对每个问题生成和执行 SQL 查询，并打印结果。

## 代码结构

- **数据库连接**：使用 `SQLDatabase.from_uri` 方法连接到本地 SQLite 数据库。
- **语言模型实例化**：使用 `ChatOpenAI` 类创建一个语言模型实例。
- **提示模板**：使用 `PromptTemplate` 类创建一个用于生成 SQL 查询的提示模板。
- **查询处理**：定义一个函数 `clean_sql_query` 来清理生成的 SQL 查询。
- **查询链构建**：使用 `RunnablePassthrough` 和 `StrOutputParser` 构建一个处理查询的链。
- **查询执行**：循环遍历定义的问题列表，使用查询链生成和执行 SQL 查询，并处理异常。

## 文件引用

- **主要代码文件**：`sql_llm-v1.py`