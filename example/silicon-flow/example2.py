#  https://blog.csdn.net/sinat_29950703/article/details/143386213
import os
from openai import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

API_KEY =  os.getenv("OPENAI_API_KEY")


#  自定义硅基流动大模型类
class CustomLLM_Siliconflow:
    def __call__(self, prompt: str) -> str:
        # 初始化OpenAI客户端（base_url是硅基流动网站的地址）
        client = OpenAI(api_key=API_KEY, base_url="https://api.siliconflow.cn/v1")
        
        # 发送请求到模型
        response = client.chat.completions.create(
            model='THUDM/glm-4-9b-chat',
            messages=[
                {'role': 'user', 
                 'content': f"{prompt}"}  # 用户输入的提示
            ],
        )

        # 打印响应结构，以便调试
        # print("Response structure:", response)

        # 收集所有响应内容
        content = ""
        if hasattr(response, 'choices') and response.choices:
            for choice in response.choices:
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    chunk_content = choice.message.content
                    # print(chunk_content, end='')  # 可选：打印内容
                    content += chunk_content  # 将内容累加到总内容中
        else:
            raise ValueError("Unexpected response structure")

        return content  # 返回最终的响应内容
    
  

def run_one():
    # 创建自定义LLM对象
    llm = CustomLLM_Siliconflow()
    
    # 示例查询：将大象装进冰箱分几步？
    print(llm("把大象装进冰箱分几步？"))


def run_two():
    # 创建自定义LLM对象
    llm = CustomLLM_Siliconflow()
    #  基础版
    # 定义国家名称
    country = """中国"""
    
    # 定义任务模板
    country_template = """
    任务: 输入一个国家，输出国家的首都
    语言：中文

    按json格式输出，输出格式如下：
    country_name
    capital_name

    国家: {country_name}
    """
    # 使用模板创建提示
    prompt_template = ChatPromptTemplate.from_template(country_template)
    messages = prompt_template.format_messages(country_name=country)

        # 获取模型响应
    response = llm(messages)
    print(response)  # 打印响应内容


def run_three():
    llm = CustomLLM_Siliconflow()
     # 进阶版
    country_schema = ResponseSchema(name="country_name", description="国家的名称。")
    capital_schema = ResponseSchema(name="capital_name", description="对应国家的首都名称。")
    response_schemas = [country_schema, capital_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)


    # 获取格式化指令
    format_instructions = output_parser.get_format_instructions()

    # 定义模板字符串，用于构建提示词
    country_template = """\
    任务: 输入一个国家，输出国家的首都
    语言：中文

    国家: {country_name}
    {format_instructions}
    """

    prompt_template = ChatPromptTemplate.from_template(country_template)
    messages = prompt_template.format_messages(country_name="中国",
                                            format_instructions=format_instructions)
    # 发送消息并获取响应
    response = llm(messages)
    print(response)     # 里面本来有response.content,现在用自定义模型,就没有这个内容了

    # 使用 `output_parser` 解析响应内容
    output_dict = output_parser.parse(response)
    print(output_dict)


if __name__ == '__main__':
    run_three()