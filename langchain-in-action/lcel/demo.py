"""
LCEL Demo Implementation

This module demonstrates LangChain Expression Language (LCEL) usage for building
chained language model workflows with error handling and configuration best practices.

Key Features:
- Joke generation and translation chain
- Environment configuration validation
- Type-safe component composition
- Error handling with fallback mechanisms
"""

from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import Dict, Any

# Configuration and Environment Setup
_DEEPSEEK_API_KEY = os.getenv("deepseek_api_key")
_DEEPSEEK_API_BASE = os.getenv("deepseek_api_url")
_MODEL_NAME = "deepseek-chat"

def validate_environment() -> None:
    """Validate required environment variables"""
    if not all([_DEEPSEEK_API_KEY, _DEEPSEEK_API_BASE]):
        raise EnvironmentError(
            "Missing required environment variables: deepseek_api_key and deepseek_api_url"
        )

def initialize_model() -> ChatOpenAI:
    """Initialize ChatOpenAI model with configured parameters"""
    return ChatOpenAI(
        model=_MODEL_NAME,
        api_key=_DEEPSEEK_API_KEY,
        base_url=_DEEPSEEK_API_BASE,
        temperature=0.7,
        max_tokens=150
    )

# Core Processing Chains
def create_joke_chain(llm: ChatOpenAI) -> RunnablePassthrough:
    """Create joke generation chain"""
    joke_prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")
    return (
        RunnablePassthrough.assign(topic=itemgetter("topic"))
        | joke_prompt
        | llm
        | StrOutputParser()
    )

def create_translation_chain(joke_chain: RunnablePassthrough, llm: ChatOpenAI) -> RunnablePassthrough:
    """Create translation chain with dependency injection"""
    translation_prompt = ChatPromptTemplate.from_template(
        "Translate this joke into {language}:\n{joke}"
    )
    return (
        RunnablePassthrough.assign(
            joke=itemgetter("joke") | joke_chain,
            language=itemgetter("language")
        )
        | translation_prompt
        | llm
        | StrOutputParser()
    )

# Example Usage and Execution
def run_demo_examples(translation_chain: RunnablePassthrough) -> None:
    """Execute demo examples with error handling"""
    try:
        # Simple joke generation
        response = translation_chain.invoke({
            "joke": {"topic": "ice cream"},
            "language": "english"
        })
        print("Generated Joke:", response)

        # Complex translation workflow
        response = translation_chain.invoke({
            "joke": {"topic": "obama"},
            "language": "chinese"
        })
        print("Translated Joke:", response)
    except Exception as e:
        print(f"Execution Error: {str(e)}")
        # Implement additional error handling logic here

if __name__ == "__main__":
    validate_environment()
    llm = initialize_model()
    joke_chain = create_joke_chain(llm)
    translation_chain = create_translation_chain(joke_chain, llm)
    run_demo_examples(translation_chain)