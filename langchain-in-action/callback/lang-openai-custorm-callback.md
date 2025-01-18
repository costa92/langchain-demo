# @lang-openai-custorm-callback.py 功能说明

## 概述

此脚本集成了 LangChain 和 OpenAI 语言模型，创建了自定义回调处理器，用于在生成新令牌或完成响应时执行特定操作，如打印信息或执行异步任务。

## 环境配置

### 环境变量
需设置 `OPENAI_API_KEY` 环境变量以认证和访问 OpenAI 服务。

## 组件

### 同步回调处理器
- `MyFlowerShopSyncHandler` 继承自 `BaseCallbackHandler`，用于同步回调，功能是在生成新令牌时打印信息。

### 异步回调处理器
- `MyFlowerShopAsyncHandler` 继承自 `BaseCallbackHandler`，用于异步回调，功能是在生成新令牌和响应结束时执行异步操作。
- 方法：
  - `async def on_llm_new_token(self, token: str, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs) -> None`：在生成新令牌时调用。
  - `async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None`：在响应结束时调用。

## 主函数

### 初始化
使用 `ChatOpenAI` 类初始化语言模型，配置包括模型名称、API 基础 URL 和回调处理器。

### 回复生成
通过 `agenerate` 方法生成并打印聊天回复。

## 执行

### 主循环
使用 `asyncio.run(main())` 启动异步主函数。

## 错误处理
在主函数中使用 `try-except` 捕获异常，未设置环境变量时抛出错误。

## 总结
该脚本展示了如何使用 LangChain 和 OpenAI 创建自定义回调处理器，涵盖环境配置、组件实现、主函数逻辑及错误处理，增强了聊天代理的功能。

## 参考

- [How to dispatch custom callback events](https://python.langchain.com/docs/how_to/callbacks_custom_events/#async-callback-handler)