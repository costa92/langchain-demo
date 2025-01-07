from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE as RounterTemplate
from langchain.chains import LLMChain
from langchain.chains.router import MultiPromptChain
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义提示模板
flower_core_template = """
你是一个经验丰富的园丁，擅长解答关于养花育花的问题。
下面是需要你回答的问题：
{input}
"""

flower_deco_template = """
你是一位网红插花大师，擅长解答关于鲜花装饰的问题。
下面是需要你回答的问题：
{input}
"""

# 构建提示信息
prompt_infos = [
    {
        "key": "flower_core",
        "description": "适合回答关于鲜花护理的问题",
        "prompt": flower_core_template
    },
    {
        "key": "flower_decoration",
        "description": "插花大师",
        "prompt": flower_deco_template
    }
]

# 初始化语言模型
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 OPENAI_API_KEY")

model_name = "Qwen/Qwen2.5-7B-Instruct"
base_url = "https://api.siliconflow.cn/v1"

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=base_url,
    model_name=model_name,
)

# 构建链映射
chain_map = {}
for info in prompt_infos:
    prompt = PromptTemplate(
        template=info["prompt"],
        input_variables=["input"]
    )
    logger.info(f"目标提升：\n{prompt}")
    chain = LLMChain(llm=llm, prompt=prompt)
    chain_map[info["key"]] = chain

# 创建路由链
destinations = [f"{p['key']}: {p['description']}" for p in prompt_infos]
router_template = RounterTemplate.format(destinations="\n".join(destinations))
logger.info(f"路由模板:\n{router_template}")

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
logger.info(f"路由提示:\n{router_prompt}")

router_chain = LLMRouterChain.from_llm(
    llm,
    router_prompt,
    verbose=True
)

# 构建默认链
# default_prompt = PromptTemplate(
#     template="你是一个通用助手，擅长回答各种问题。下面是需要你回答的问题：\n{input}",
#     input_variables=["input"]
# )
# default_chain = LLMChain(llm=llm, prompt=default_prompt)

# 构建默认链
from langchain.chains import ConversationChain

default_chain = ConversationChain(
    llm=llm,
    output_key="text",
    verbose=True
)

# 构建多提示链
chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=chain_map,
    default_chain=default_chain,
    verbose=True
)

# 测试
# 测试
logger.info(chain.invoke({"input": "如何为玫瑰浇水？"}))
logger.info(chain.invoke({"input": "如何为婚礼场地装饰花朵？"}))
logger.info(chain.invoke({"input": "如何区分阿豆和罗豆？"}))