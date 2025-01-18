# @lang-openai-custorm-callback.py 实现功能说明

## 功能概述

该脚本通过 LangChain 与 OpenAI 的语言模型集成，创建了自定义的回调处理器。这些处理器在语言模型生成新令牌或完成响应时执行特定操作，如打印信息或执行异步任务。

## 环境配置

### 环境变量设置
需要设置 `OPENAI_API_KEY` 环境变量，用于认证和访问 OpenAI 的语言模型服务。

## 组件实现

### 同步回调处理器
- `MyFlowerShopSyncHandler` 类继承自 `BaseCallbackHandler`，用于处理同步回调。
- 功能：在生成新令牌时，打印相关信息。

### 异步回调处理器

- `MyFlowerShopAsyncHandler` 类同样继承自 `BaseCallbackHandler`，用于处理异步回调。
- 功能：在生成新令牌时，执行异步操作（如模拟数据获取），并在语言模型响应结束时执行类似的异步操作。
- 方法签名：
  - `async def on_llm_new_token(self, token: str, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs) -> None`
    - 说明：在生成新令牌时调用，执行异步操作。
  - `async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None`
    - 说明：在语言模型响应结束时调用，执行异步操作。

## 主函数逻辑

### 语言模型初始化
使用 `ChatOpenAI` 类创建语言模型实例，配置包括模型名称、API 基础 URL、回调处理器列表等。

### 聊天回复生成
通过 `agenerate` 方法生成聊天回复，传入用户消息，并打印生成的聊天回复。

## 执行逻辑

### 主循环运行
使用 `asyncio.run(main())` 启动异步主函数。

## 错误处理逻辑
在主函数中，通过 `try-except` 块捕获异常，未设置环境变量时抛出错误。

## 总结
该脚本完整实现了使用 LangChain 和 OpenAI 创建自定义回调处理器的流程，包括环境配置、组件实现、主函数逻辑及错误处理。这些回调处理器在语言模型交互过程中执行同步或异步自定义逻辑，增强了聊天代理的功能和灵活性。