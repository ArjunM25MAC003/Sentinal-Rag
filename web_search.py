import os
from typing import Dict, Any, List

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

try:
    from sentinelrag.utils import generate_citation_id
except ImportError:
    # Fallback if utils isn't set up yet
    def generate_citation_id(source_type: str, index: int) -> str:
        prefix = source_type.upper()[:4]
        return f"{prefix}{index}"

def web_search(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Web Search Node: Uses the Tavily API to search the web for information.

    Iterates over the generated sub-questions and searches the web for each.
    Results are deduplicated by URL and truncated to 500 characters.

    Args:
        state (Dict[str, Any]): The current graph state containing 'sub_questions' 
            and 'active_sources'.

    Returns:
        Dict[str, Any]: Updated state containing 'web_results'.
    """
    active_sources = state.get("active_sources", [])
    if "web" not in active_sources:
        print("[Web Search] Web search is inactive, skipping.")
        return {"web_results": []}

    sub_questions = state.get("sub_questions", [state.get("question", "")])
    all_results = []
    seen_urls = set()

    if not TavilyClient:
        print("[Web Search] Error: tavily-python not installed. Skipping web search.")
        return {"web_results": []}

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        print("[Web Search] Warning: TAVILY_API_KEY not set. Skipping web search.")
        return {"web_results": []}

    client = TavilyClient(api_key=api_key)
    result_index = 1

    for query in sub_questions:
        try:
            print(f"[Web Search] Searching for: '{query}'")
            response = client.search(query=query, max_results=3, search_depth="basic")
            
            for item in response.get('results', []):
                url = item.get('url')
                
                # Deduplicate by URL
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                
                content = item.get('content', "")
                # Truncate to 500 characters
                if len(content) > 500:
                    content = content[:497] + "..."
                
                all_results.append({
                    "text": content,
                    "url": url,
                    "title": item.get('title', 'Unknown Title'),
                    "source_type": "web",
                    "citation_id": generate_citation_id("WEB", result_index)
                })
                result_index += 1

        except Exception as e:
            print(f"[Web Search] Error searching for '{query}': {e}")
            # Continue to the next query instead of crashing

    print(f"[Web Search] Found {len(all_results)} unique results.")
    return {"web_results": all_results}
