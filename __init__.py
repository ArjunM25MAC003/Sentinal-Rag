"""
SentinelRAG Graph Nodes
=======================

Each module in this package implements one node of the LangGraph
research pipeline. Nodes are pure functions: they receive the
current graph state and return the fields they want to update.

Pipeline order:
  planner → [web_search, wiki_search, arxiv_search, pdf_search] → synthesiser → critic
"""

from .planner import planner
from .web_search import web_search
from .wiki_search import wiki_search
from .arxiv_search import arxiv_search
from .pdf_search import create_pdf_search_node
from .synthesiser import draft_synthesis_node as synthesise
from .critic import critic_node as critic

__all__ = [
    "planner", "web_search", "wiki_search",
    "arxiv_search", "create_pdf_search_node", "synthesise", "critic",
]
