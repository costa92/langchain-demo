from langchain_ollama.llms import OllamaLLM
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.agents import Tool
import numexpr as ne
from langchain.tools import tool

# Define the mathematical calculation tool
@tool
def calculate_expression(expression: str) -> str:
    """
    计算给定的数学表达式并返回结果。
    
    参数:
    expression (str): 要计算的数学表达式。
    
    返回:
    str: 计算结果的字符串表示。
    """
    try:
        expression_str = str(expression)
        result = ne.evaluate(expression_str).item()  # Use numexpr to evaluate the expression
        return str(result)  # Return result as string for agent's output
    except Exception as e:
        print(f"Error calculating expression: {str(e)}")
        return f"Error calculating expression: {str(e)}"

# Define math tools
math_tools = [calculate_expression]

# Initialize LLM (Large Language Model)
try:
    llm = OllamaLLM(model="llama3:8b")
    # Test LLM to ensure it's functioning
    print(f"LLM initialization test: {llm.invoke('Hello!')}")
except Exception as e:
    print(f"LLM initialization error: {e}")
    exit()

from langchain.prompts import PromptTemplate

# 优化后的 PromptTemplate，确保输出仅包含数字结果
# react_prompt = PromptTemplate(
#     input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
#     template=(
#         "You are a helpful assistant that answers questions using the following tools:\n\n"
#         "{tools}\n\n"
#         "Use the following format to reason through the question:\n\n"
#         "Question: {input}\n"  # 用户的输入问题
#         "Thought: First, think about the best way to approach this question using available tools.\n"
#         "Action: Choose the appropriate action from [{tool_names}]\n"  # 可以选择的工具
#         "Action Input: Provide the input for the chosen action\n"
#         "Observation: The result of the action performed\n"  # 工具执行结果
#         "Thought: After the observation, think if you need to perform any further actions or if you can finalize the answer.\n"
#         "Final Answer: {input}.\n"  # 只返回原问题中的最终答案，而不包含额外说明
#         "Begin!\n\n"
#         "Question: {input}\n"  # 重新强调问题
#         "Thought:{agent_scratchpad}\n"  # 之前的思考过程
#     )
# )
# 输出的：“The final answer is 8.”
react_prompt = PromptTemplate(
    input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
    template=(
        "Answer the following questions as best you can. You have access to the following tools:\n\n"
        "{tools}\n\n"
        "Use the following format:\n\n"
        "Question: the input question you must answer\n"
        "Thought: you should always think about what to do\n"
        "Action: the action to take, should be one of [{tool_names}]\n"
        "Action Input: the input to the action\n"
        "Observation: the result of the action\n"
        "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
        "Thought: I now know the final answer\n"
        "Final Answer: {input}\n\n"  # 修改输出格式
        "Begin!\n\n"
        "Question: {input}\n"
        "Thought: {agent_scratchpad} \n"
        "Please return the calculation result directly without additional explanation."
    )
)

# Define agent
agent = create_react_agent(
    tools=math_tools,
    llm=llm,
    prompt=react_prompt
)

## With LangChain's AgentExecutor, you could configure an early_stopping_method to either return a string saying "Agent stopped due to iteration limit or time limit." ("force") or prompt the LLM a final time to respond ("generate").
# Initialize AgentExecutor with optimized settings
agent_executor = AgentExecutor(
    agent=agent,
    tools=math_tools,
    handle_parsing_errors=False,
    allow_dangerous_code=True,
    early_stopping_method="force",  # 保持为 "force"
    max_iterations=5,  # 增加迭代次数
    max_execution_time=30,  # 增加每个工具的执行时间
    max_time=120,  # 增加总的最大时间限制
    verbose=True  # 启用详细日志
)

# User's question
user_question = "What is 2 * 4 ?"
print(f"User Question: {user_question}")

# Construct input data
inputs = {
    "input": user_question,
    "tools": "\n".join([tool.description for tool in math_tools]),
    "tool_names": ", ".join([tool.name for tool in math_tools]),
    "agent_scratchpad": ""  # Initial empty scratchpad
}

# Print the inputs to debug
print(f"Inputs: {inputs}")

# Execute the agent and print the result
try:
    response = agent_executor.invoke(inputs)
    print(f"Response: {response.get('output')}")
except Exception as e:
    print(f"Execution error: {e}")


