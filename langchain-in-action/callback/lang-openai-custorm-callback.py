import os

import asyncio

from langchain_core.callbacks import BaseCallbackHandler
from typing import Any, Optional
from uuid import UUID
from langchain.schema import LLMResult, HumanMessage
from langchain_openai import ChatOpenAI

# 创建同步回调处理器
class MyFlowerShopSyncHandler(BaseCallbackHandler):
     def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"获取花卉数据: token: {token}")

# 创建异步回调处理器
class MyFlowerShopAsyncHandler(BaseCallbackHandler):
    async def on_llm_new_token(
        self, 
        token: str, 
        run_id: UUID, 
        parent_run_id: Optional[UUID] = None, 
        **kwargs
    ) -> None:
        print("正在获取花卉数据...")
        await asyncio.sleep(0.5)  # 模拟异步操作
        print("花卉数据获取完毕。提供建议...")
    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        print("正在获取花卉数据...")
        await asyncio.sleep(0.5)  # 模拟异步操作
        print("花卉数据获取完毕。提供建议...")


# 主要的异步主函数
async def main(): 
    # 初始化语言模型
    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
      raise ValueError("请设置环境变量 OPENAI_API_KEY")

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    base_url = "https://api.siliconflow.cn/v1"
    flower_shop_chat = ChatOpenAI(
        model_name=model_name,
        base_url=base_url,
        callbacks=[MyFlowerShopSyncHandler(), MyFlowerShopAsyncHandler()],
        temperature=0.7,
        max_tokens=100,
        streaming=True,
    )
    # 异步生成聊天回复
    response = await flower_shop_chat.agenerate([[HumanMessage(content="哪种花卉最适合生日？只简单说3种，不超过50字")]])
    print(response.generations[0][0].text)

if __name__ == "__main__":
    asyncio.run(main())
