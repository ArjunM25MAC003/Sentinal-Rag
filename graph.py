"""
LangGraph assembly for the SentinelRAG pipeline.
=================================================

This module connects all the individual nodes (planner, web, wiki, arxiv,
pdf, draft synthesiser, critic) into a cohesive, executable directed graph.

The pipeline flows linearly:
  planner → web_search → wiki_search → arxiv_search → pdf_search
  → synthesiser → critic → END

Each node receives the full state and returns only the keys it wants
to update — LangGraph merges those updates automatically.
"""

from typing import List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

from sentinelrag.state import State, create_initial_state
from sentinelrag.nodes.planner import planner
from sentinelrag.nodes.web_search import web_search
from sentinelrag.nodes.wiki_search import wiki_search
from sentinelrag.nodes.arxiv_search import arxiv_search
from sentinelrag.nodes.pdf_search import create_pdf_search_node
from sentinelrag.nodes.synthesiser import draft_synthesis_node
from sentinelrag.nodes.critic import critic_node


def build_graph(pdf_knowledge_base=None):
    """
    Builds and compiles the full SentinelRAG LangGraph pipeline.

    The graph follows a linear topology:
      plan → web → wiki → arxiv → pdf → draft → critic → END

    Args:
        pdf_knowledge_base: An optional PDFKnowledgeBase instance. When provided,
            the PDF node will search through its indexed documents. When None,
            a no-op stub is used instead.

    Returns:
        A compiled LangGraph ready for .invoke() calls.
    """
    # 1. Create a StateGraph anchored to our State TypedDict schema.
    #    LangGraph will validate that node return values match these keys.
    workflow = StateGraph(State)

    # 2. Register every node with a short, readable name.
    workflow.add_node("plan", planner)
    workflow.add_node("web", web_search)
    workflow.add_node("wiki", wiki_search)
    workflow.add_node("arxiv", arxiv_search)

    # 3. The PDF node needs a knowledge base injected via factory.
    #    If no knowledge base is provided, we register a simple stub
    #    that returns empty results.
    if pdf_knowledge_base is not None:
        workflow.add_node("pdf", create_pdf_search_node(pdf_knowledge_base))
    else:
        # Stub node — no PDFs uploaded, nothing to search
        def _no_pdf_node(state: Dict[str, Any]) -> Dict[str, Any]:
            print("[PDF Search] No knowledge base loaded. Skipping.")
            return {"pdf_results": []}
        workflow.add_node("pdf", _no_pdf_node)

    workflow.add_node("draft", draft_synthesis_node)
    workflow.add_node("critic", critic_node)

    # 4. Set the entry point — the first node to execute
    workflow.set_entry_point("plan")

    # 5. Wire up the linear pipeline edges
    workflow.add_edge("plan", "web")
    workflow.add_edge("web", "wiki")
    workflow.add_edge("wiki", "arxiv")
    workflow.add_edge("arxiv", "pdf")
    workflow.add_edge("pdf", "draft")
    workflow.add_edge("draft", "critic")
    workflow.add_edge("critic", END)

    # 6. Compile into a runnable object
    print("[Graph] Pipeline compiled: plan -> web -> wiki -> arxiv -> pdf -> draft -> critic -> END")
    return workflow.compile()


def run_query(
    question: str,
    active_sources: Optional[List[str]] = None,
    pdf_kb=None,
) -> Dict[str, Any]:
    """
    Convenience wrapper that builds the graph, creates initial state,
    runs the pipeline, and returns the final state.

    Args:
        question:       The user's research question.
        active_sources: Which retrievers to enable. Defaults to all four:
                        ['web', 'wikipedia', 'arxiv', 'pdf'].
        pdf_kb:         An optional PDFKnowledgeBase for the PDF retriever.

    Returns:
        The final state dict after the full pipeline has executed.

    Example:
        >>> result = run_query("What is attention in transformers?")
        >>> print(result["final_answer"])
        >>> print(result["faithfulness_score"])
    """
    if active_sources is None:
        active_sources = ["web", "wikipedia", "arxiv", "pdf"]

    print(f"\n{'='*60}")
    print(f"[SentinelRAG] Starting query: '{question}'")
    print(f"[SentinelRAG] Active sources: {active_sources}")
    print(f"{'='*60}\n")

    # Build the graph (inject PDF KB if provided)
    graph = build_graph(pdf_knowledge_base=pdf_kb)

    # Initialise a clean state with sensible defaults
    initial_state = create_initial_state(
        question=question,
        active_sources=active_sources,
    )

    # Run the full pipeline
    final_state = graph.invoke(initial_state)

    print(f"\n{'='*60}")
    print(f"[SentinelRAG] Pipeline complete.")
    print(f"[SentinelRAG] Faithfulness score: {final_state.get('faithfulness_score', 0.0):.3f}")
    print(f"[SentinelRAG] Conflicts found: {len(final_state.get('conflicts', []))}")
    print(f"{'='*60}\n")

    return final_state


# ---------------------------------------------------------------------------
# Quick test when running this file directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_question = "What are the latest advances in transformer architectures?"
    sources = ["web", "wikipedia", "arxiv"]

    print("--- SentinelRAG Sample Run ---")
    try:
        result = run_query(sample_question, sources)
        print("\n--- Final Answer ---")
        print(result.get("final_answer", "(no answer)"))
        print(f"\nFaithfulness Score: {result.get('faithfulness_score', 0.0):.3f}")

        if result.get("conflicts"):
            print("\n--- Conflicts ---")
            for c in result["conflicts"]:
                print(f"  - {c}")
    except Exception as e:
        print(f"\nExecution failed: {e}")
        print("Tip: make sure GROQ_API_KEY and TAVILY_API_KEY are set in your environment.")
