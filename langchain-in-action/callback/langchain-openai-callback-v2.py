import asyncio
import os
import logging
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化语言模型
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 OPENAI_API_KEY")

model_name = "Qwen/Qwen2.5-7B-Instruct"
base_url = "https://api.siliconflow.cn/v1"

try:
    llm = ChatOpenAI(
        api_key=API_KEY,
        base_url=base_url,
        model=model_name,
        temperature=0.5,
        max_tokens=1000,
    )

   # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的助手，擅长记住对话上下文。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    # 创建链
    chain = prompt | llm

   # 存储会话历史的字典
    store = {}

    # 创建消息历史管理器的工厂函数
    def get_message_history(session_id: str) -> ChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
       # 使用RunnableWithMessageHistory包装链
    conversation = RunnableWithMessageHistory(
        chain,
        get_message_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    # 使用context manager进行token counting
    with get_openai_callback() as cb:
        # 配置会话ID
        config = {"configurable": {"session_id": "test_session"}}

        # 第一天的对话
        # 回合1
        result1 = conversation.invoke(
            {"input": "我姐姐明天要过生日，我需要一束生日花束。"},
            config=config
        )
        logger.info("第一次对话结果: %s", result1.content)
     # 打印当前会话历史
        logger.info("当前会话历史: %s", 
            [msg.content for msg in store["test_session"].messages]
        )


        # 回合2
        result2 = conversation.invoke(
            {"input": "她喜欢粉色玫瑰，颜色是粉色的。"},
            config=config
        )
        logger.info("第二次对话结果: %s", result2.content)

        logger.info("当前会话历史: %s", 
            [msg.content for msg in store["test_session"].messages]
        )

        # 回合3 （第二天的对话）
        result3 = conversation.invoke(
            {"input": "我又来了，还记得我昨天为什么要来买花吗？"},
            config=config
        )
        logger.info("\n第三次对话后的记忆:\n%s", result3.content)

    # 输出使用的tokens
    logger.info("\n总计使用的tokens: %d", cb.total_tokens)

    # 进行更多的异步交互和token计数
    async def additional_interactions():
        with get_openai_callback() as cb:
            tasks = [llm.ainvoke("我姐姐喜欢什么颜色的花？") for _ in range(3)]
            await asyncio.gather(*tasks)
        logger.info("\n另外的交互中使用的tokens: %d", cb.total_tokens)

    # 运行异步函数
    asyncio.run(additional_interactions())

except Exception as e:
    logger.error("发生错误: %s", str(e))
