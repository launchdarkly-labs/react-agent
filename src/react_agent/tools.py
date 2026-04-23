"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, Dict

from langchain_tavily import TavilySearch
from ldai.models import AIAgentConfig


def make_search(ai_config: AIAgentConfig) -> Callable[..., Any]:
    """Build a search tool that closes over this run's max_search_results.

    Capturing the value at run setup keeps it stable across the turn (a
    mid-run flag flip won't change it between two tool calls) and means
    the tool body never re-evaluates the AI Config (which would emit an
    extra $ld:ai:agent_config event per tool call).
    """
    max_results = ai_config.model.get_custom("max_search_results") or 10

    async def search(query: str) -> dict:
        """Search for general web results.

        This function performs a search using the Tavily search engine, which is designed
        to provide comprehensive, accurate, and trusted results. It's particularly useful
        for answering questions about current events.
        """
        return await TavilySearch(max_results=max_results).ainvoke({"query": query})

    return search


# Registry of tool factories keyed by the LD AI Tool name. Each factory takes
# the per-run AI Config and returns the actual callable. graph.py materializes
# this into {name: callable} on the first call_model tick.
TOOL_FACTORIES: Dict[str, Callable[[AIAgentConfig], Callable[..., Any]]] = {
    "search": make_search,
}
