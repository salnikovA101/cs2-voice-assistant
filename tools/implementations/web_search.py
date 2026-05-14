import logging
from tools.base import tool
from ddgs import DDGS

logger = logging.getLogger(__name__)


@tool
def web_search(query: str) -> str:
    """
    Performs a web search using DuckDuckGo and returns the results (snippets and links).
    Use this when the user asks to search for something online, find facts,
    or look something up on the internet. This tool returns the actual text of the search results
    so you can read them and summarize the answer for the user.
    Example triggers: 'загугли', 'найди в интернете', 'поищи информацию о'.

    Args:
        query (str): The search query to look up.
    """
    try:
        results = DDGS().text(query, max_results=5, backend="auto")

        if not results:
            return f"No results found for query: {query}"

        formatted_results = []
        for i, res in enumerate(results):
            formatted_results.append(
                f"{i + 1}. Title: {res.get('title')}\n   Snippet: {res.get('body')}\n   URL: {res.get('href')}"
            )

        result_str = "\n\n".join(formatted_results)
        logger.info(
            f"Инструмент выполнен: web_search, query='{query}', found {len(results)} results"
        )
        return f"Search Results for '{query}':\n\n{result_str}"
    except Exception as e:
        logger.error(f"Ошибка web_search: {e}")
        return f"Failed to search: {str(e)}"
