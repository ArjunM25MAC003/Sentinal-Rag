from typing import Dict, Any, List

try:
    import wikipedia
except ImportError:
    wikipedia = None

try:
    from sentinelrag.utils import generate_citation_id
except ImportError:
    def generate_citation_id(source_type: str, index: int) -> str:
        prefix = source_type.upper()[:4]
        return f"{prefix}{index}"

def wiki_search(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wikipedia Search Node: Searches Wikipedia for relevant articles.

    Uses the wikipedia library to find articles based on sub-questions.
    Extracts the first few paragraphs and truncates to 600 characters.

    Args:
        state (Dict[str, Any]): The current graph state.

    Returns:
        Dict[str, Any]: Updated state containing 'wiki_results'.
    """
    active_sources = state.get("active_sources", [])
    if "wikipedia" not in active_sources:
        print("[Wiki Search] Wikipedia search is inactive, skipping.")
        return {"wiki_results": []}

    if not wikipedia:
        print("[Wiki Search] Error: wikipedia package not installed. Skipping.")
        return {"wiki_results": []}

    sub_questions = state.get("sub_questions", [state.get("question", "")])
    all_results = []
    seen_titles = set()
    result_index = 1

    for query in sub_questions:
        try:
            print(f"[Wiki Search] Searching for: '{query}'")
            search_results = wikipedia.search(query, results=2)
            
            if not search_results:
                continue
                
            # Take the first valid title
            title_to_fetch = search_results[0]
            
            if title_to_fetch in seen_titles:
                continue
                
            try:
                page = wikipedia.page(title_to_fetch, auto_suggest=False)
            except wikipedia.DisambiguationError as e:
                # Handle disambiguation by taking the first suggestion
                print(f"[Wiki Search] Disambiguation for {title_to_fetch}, trying {e.options[0]}")
                if not e.options:
                    continue
                try:
                    page = wikipedia.page(e.options[0], auto_suggest=False)
                except Exception:
                    continue
            except wikipedia.PageError:
                print(f"[Wiki Search] Page {title_to_fetch} not found.")
                continue

            seen_titles.add(page.title)
            
            # Get content, split by double newline, take first 3 paragraphs
            paragraphs = page.content.split('\n\n')
            summary_text = "\n".join(paragraphs[:3])
            
            # Truncate to 600 characters
            if len(summary_text) > 600:
                summary_text = summary_text[:597] + "..."
                
            all_results.append({
                "text": summary_text,
                "url": page.url,
                "title": page.title,
                "source_type": "wikipedia",
                "citation_id": generate_citation_id("WIKI", result_index)
            })
            result_index += 1
            
        except Exception as e:
            print(f"[Wiki Search] Unexpected error for query '{query}': {e}")

    print(f"[Wiki Search] Found {len(all_results)} unique results.")
    return {"wiki_results": all_results}
