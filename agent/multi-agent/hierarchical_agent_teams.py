
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from typing import List


# https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/multi_agent/hierarchical_agent_teams.ipynb

tavily_tool = TavilySearchResults(max_results=5)

@tool
def scrape_webpages(urls: List[str]) -> str:
  return ""