from typing import Dict, Any, List

try:
    import arxiv
except ImportError:
    arxiv = None

try:
    from sentinelrag.utils import generate_citation_id
except ImportError:
    def generate_citation_id(source_type: str, index: int) -> str:
        prefix = source_type.upper()[:4]
        return f"{prefix}{index}"

def is_technical_query(query: str) -> bool:
    """Heuristic check to see if a query is technical/academic."""
    keywords = {
        "algorithm", "model", "neural", "machine learning", "research", 
        "paper", "method", "transformer", "deep learning", "training", 
        "dataset", "benchmark", "optimization", "architecture", "network", 
        "learning rate", "gradient", "loss function", "attention", "embedding"
    }
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in keywords)

def arxiv_search(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ArXiv Search Node: Searches arXiv for academic papers.

    Only runs if 'arxiv' is active and the query appears technical.
    Limits to the first 2 sub-questions and max 2 results per query.

    Args:
        state (Dict[str, Any]): The current graph state.

    Returns:
        Dict[str, Any]: Updated state containing 'arxiv_results'.
    """
    active_sources = state.get("active_sources", [])
    if "arxiv" not in active_sources:
        print("[ArXiv Search] ArXiv search is inactive, skipping.")
        return {"arxiv_results": []}

    if not arxiv:
        print("[ArXiv Search] Error: arxiv package not installed. Skipping.")
        return {"arxiv_results": []}

    main_question = state.get("question", "")
    
    # Check if the query is technical
    if not is_technical_query(main_question):
        print(f"[ArXiv Search] Query '{main_question}' does not appear technical. Skipping arXiv.")
        return {"arxiv_results": []}

    sub_questions = state.get("sub_questions", [main_question])
    all_results = []
    seen_titles = set()
    result_index = 1
    
    # Limit to first 2 sub-questions to avoid too many API calls
    queries_to_search = sub_questions[:2]

    # Use arXiv API v2 client
    client = arxiv.Client()

    for query in queries_to_search:
        try:
            print(f"[ArXiv Search] Searching for: '{query}'")
            search = arxiv.Search(
                query=query,
                max_results=2,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            for paper in client.results(search):
                if paper.title in seen_titles:
                    continue
                seen_titles.add(paper.title)
                
                # Truncate summary to 400 chars
                summary = paper.summary.replace('\n', ' ')
                if len(summary) > 400:
                    summary = summary[:397] + "..."
                    
                authors = [author.name for author in paper.authors[:3]]
                
                all_results.append({
                    "text": summary,
                    "url": paper.entry_id,
                    "title": paper.title,
                    "authors": authors,
                    "source_type": "arxiv",
                    "citation_id": generate_citation_id("ARXI", result_index),
                    "published": paper.published.strftime("%Y-%m-%d")
                })
                result_index += 1
                
        except Exception as e:
            print(f"[ArXiv Search] Error searching for '{query}': {e}")

    print(f"[ArXiv Search] Found {len(all_results)} unique results.")
    return {"arxiv_results": all_results}
