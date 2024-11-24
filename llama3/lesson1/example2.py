from langchain_ollama.llms import OllamaLLM
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
import numexpr as ne
from langchain_core.tools import tool
# Define the mathematical calculation tool

@tool
def calculate_expression(expression):
    """
    Calculate the given mathematical expression and return the result.
    """
    try:
        result = ne.evaluate(expression).item()  # Use numexpr to evaluate the expression
        return result
    except Exception as e:
        return f"Error calculating expression: {str(e)}"

math_tools = [
    Tool(
        name="Numexpr Math",
        func=calculate_expression,
        description="A tool for efficient mathematical calculations. Input is the math expression."
    )
]

# Initialize LLM
llm = OllamaLLM(model="llama3")

# Define prompt template for React agent
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "{input}"),
        # Placeholders fill up a **list** of messages
        ("placeholder", "{agent_scratchpad}"),
    ]
)


# Create the agent
agent = create_react_agent(
    tools=math_tools,
    llm=llm,
    prompt=prompt,
)

# Initialize AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=math_tools, max_iterations=20, max_time=120, verbose=True,handle_parsing_errors=True)

# User question
user_question = "What is 37593 * 67?"
print(f"User question: {user_question}")

# Constructing the input data to pass to the agent
inputs = {
    "input": user_question,
    "tools": "\n".join([tool.description for tool in math_tools]),
    "tool_names": ", ".join([tool.name for tool in math_tools]),
    "agent_scratchpad": ""  # Start with an empty scratchpad
}

# Execute and print the response
try:
    # response = agent_executor.invoke({"input": user_question})
    agent_executor.invoke(inputs)
except Exception as e:
    print(f"Execution error: {e}")
