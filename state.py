"""
State definition for the SentinelRAG agent.
"""

from typing import TypedDict, List, Dict, Any

class State(TypedDict):
    """
    State representing the ongoing process in the SentinelRAG agent.
    """
    question: str
    """The user's original research question"""
    sub_questions: List[str]
    """Decomposed sub-questions from the planner"""
    web_results: List[Dict[str, Any]]
    """Results from web search via Tavily, each dict has {text, url, title, source_type, citation_id}"""
    wiki_results: List[Dict[str, Any]]
    """Results from Wikipedia API"""
    arxiv_results: List[Dict[str, Any]]
    """Results from arXiv API (for academic/technical questions)"""
    pdf_results: List[Dict[str, Any]]
    """Results from FAISS index over uploaded PDFs"""
    all_sources: List[Dict[str, Any]]
    """Merged sources from all retrievers with unique citation IDs"""
    draft_answer: str
    """Initial synthesized answer with inline citations"""
    claim_verification: List[Dict[str, Any]]
    """Per-claim NLI verification results"""
    conflicts: List[Dict[str, Any]]
    """Detected contradictions between sources"""
    final_answer: str
    """Final verified answer"""
    faithfulness_score: float
    """Fraction of claims that are verified"""
    active_sources: List[str]
    """Which sources are enabled ('web', 'wikipedia', 'arxiv', 'pdf')"""

def create_initial_state(question: str, active_sources: List[str]) -> State:
    """
    Initialize a new State object with empty default values.
    
    Args:
        question (str): The user's original question.
        active_sources (List[str]): List of active sources.
        
    Returns:
        State: The initialized State dictionary.
    """
    return {
        "question": question,
        "sub_questions": [],
        "web_results": [],
        "wiki_results": [],
        "arxiv_results": [],
        "pdf_results": [],
        "all_sources": [],
        "draft_answer": "",
        "claim_verification": [],
        "conflicts": [],
        "final_answer": "",
        "faithfulness_score": 0.0,
        "active_sources": active_sources
    }
