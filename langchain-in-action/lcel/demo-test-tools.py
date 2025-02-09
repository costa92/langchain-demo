"""
LangChain 多语言笑话生成演示 (LCEL 实现)

本模块演示如何使用 LangChain Expression Language (LCEL) 构建包含环境配置验证和类型安全组件组合的链式工作流。

功能特性：
1. 支持主题笑话生成与多语言翻译的链式处理
2. 环境变量配置验证机制
3. 类型安全的组件组合方式
4. 结构化错误处理与输入验证
5. 集成网络搜索工具增强上下文关联性
"""

from operator import itemgetter
from langchain_openai import ChatOpenAI
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_core.tools import Tool

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
# 工具构建 (LCEL 工作流集成)
# ========================
# 网络搜索工具：通过 DuckDuckGo 获取实时背景信息
search_tool = DuckDuckGoSearchRun(backend="auto")  # 使用自动选择搜索引擎后端
search_tool_wrapper = Tool(
    name="web_search",
    func=search_tool.run,
    description="用于获取关于特定主题的实时网络信息和上下文"
)


# 主题上下文提取工具：集成到 LCEL 链中的自定义工具
def get_topic_context(topic: str) -> str:
    """通过搜索工具获取主题上下文（含错误处理）"""
    try:
        search_result = search_tool.run(f"背景信息 {topic}")
        return search_result[:500]  # 限制返回长度避免上下文过长
    except Exception as e:
        return f"无法获取背景信息: {str(e)}"

topic_context_tool = Tool(
    name="topic_context",
    func=get_topic_context,
    description="通过网络搜索获取特定主题的实时背景信息"
)


# ========================
# 工具增强处理链构建
# ========================
# 上下文增强型笑话生成链（集成搜索工具）
enhanced_joke_prompt = ChatPromptTemplate.from_template(
    "基于以下背景信息创建一个幽默的笑话:\n"
    "主题: {topic}\n"
    "背景信息: {topic_context}\n"
    "要求: 笑话要简短、有趣且与背景相关"
)

# 工具增强处理链（组合搜索工具与生成模型）
enhanced_joke_chain = (
    {"topic": itemgetter("topic"), 
     "topic_context": lambda x: topic_context_tool.run(x["topic"])}  # 显式调用上下文工具
    | enhanced_joke_prompt 
    | model 
    | StrOutputParser()
)

# 多语言翻译链（组合工具增强生成链与翻译功能）
translation_prompt = ChatPromptTemplate.from_template(
    "Translate this joke into {language}:\n{joke}"
)
translation_chain = (
    {"joke": enhanced_joke_chain, "language": itemgetter("language")}  # 输入映射工具增强链
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
