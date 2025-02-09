"""
LangChain 多语言笑话生成演示 (LCEL 实现)

本模块演示如何使用 LangChain Expression Language (LCEL) 构建包含环境配置验证和类型安全组件组合的链式工作流。

功能特性：
1. 支持主题笑话生成与多语言翻译的链式处理
2. 环境变量配置验证机制
3. 类型安全的组件组合方式
4. 结构化错误处理与输入验证
"""

from operator import itemgetter
from langchain_openai import ChatOpenAI
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ========================
# 配置与环境设置
# ========================
_DEEPSEEK_API_KEY = os.getenv("deepseek_api_key")
_DEEPSEEK_API_BASE = os.getenv("deepseek_api_url")
_MODEL_NAME = "deepseek-chat"

def _validate_environment():
    """环境配置验证
    - 验证必须的API密钥和接口地址
    - 在初始化前确保服务可用性
    """
    if not all([_DEEPSEEK_API_KEY, _DEEPSEEK_API_BASE]):
        raise EnvironmentError("缺少必需的环境变量: deepseek_api_key 和 deepseek_api_url")

# ========================
# 模型初始化
# ========================
model = ChatOpenAI(
    model=_MODEL_NAME,
    api_key=_DEEPSEEK_API_KEY,
    base_url=_DEEPSEEK_API_BASE,
    temperature=0.7  # 控制生成随机性 (0-1, 默认0.7)
)

# ========================
# 处理链构建
# ========================
# 笑话生成链 (Prompt -> Model -> OutputParser)
joke_prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")
joke_chain = joke_prompt | model | StrOutputParser()

# 多语言翻译链 (组合笑话生成与翻译)
translation_prompt = ChatPromptTemplate.from_template(
    "Translate this joke into {language}:\n{joke}"
)
translation_chain = (
    # 输入映射：组合原始笑话和语言参数
    {"joke": joke_chain, "language": itemgetter("language")}
    | translation_prompt
    | model
    | StrOutputParser()
)

# ========================
# 执行演示
# ========================
if __name__ == "__main__":
    _validate_environment()
    
    # 输入格式：同时包含笑话主题和翻译语言
    input_data = {
        "topic": "obama",    # 笑话主题
        "language": "chinese"  # 目标语言
    }
    
    try:
        result = translation_chain.invoke(input_data)
        print(f"Generated Output: {result}")
    except Exception as e:
        print(f"执行错误: {str(e)}")
        # 可扩展：添加重试逻辑或错误日志记录
