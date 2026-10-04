"""
PDF Search Node for SentinelRAG.

Uses a factory pattern to inject a PDFKnowledgeBase instance into
a LangGraph-compatible node function. The knowledge base must be
created and populated (via add_pdf / add_pdfs) *before* the graph
is compiled.
"""

from typing import Dict, Any, Callable

from sentinelrag.utils import generate_citation_id


def create_pdf_search_node(knowledge_base) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Factory function that returns a LangGraph node wired to a specific
    PDFKnowledgeBase instance.

    Args:
        knowledge_base: A PDFKnowledgeBase instance (from sentinelrag.pdf_processor).
                        Must implement .search(query, top_k) and .is_empty().

    Returns:
        A pdf_search(state) function suitable for StateGraph.add_node().
    """

    def pdf_search(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        PDF Search Node: retrieves relevant chunks from locally uploaded PDFs.

        Iterates over the sub-questions and runs semantic search against the
        FAISS index maintained by the injected PDFKnowledgeBase.

        Returns:
            Dict with 'pdf_results' — a list of source dicts, each containing
            text, url (source filename), title, source_type, citation_id, score.
        """
        active_sources = state.get("active_sources", [])
        if "pdf" not in active_sources:
            print("[PDF Search] PDF search is inactive, skipping.")
            return {"pdf_results": []}

        # Guard against None or empty knowledge base
        if knowledge_base is None or knowledge_base.is_empty():
            print("[PDF Search] No PDFs indexed. Skipping.")
            return {"pdf_results": []}

        sub_questions = state.get("sub_questions", [state.get("question", "")])
        all_results = []
        seen_texts = set()         # dedup by chunk text
        result_index = 1

        for query in sub_questions:
            try:
                print(f"[PDF Search] Searching for: '{query}'")
                # PDFKnowledgeBase.search() returns List[dict] with keys:
                # text, chunk_index, source_file, source_type, score
                hits = knowledge_base.search(query, top_k=3)

                for hit in hits:
                    text = hit.get("text", "")

                    # Skip duplicates
                    text_key = text[:200]  # use first 200 chars as dedup key
                    if text_key in seen_texts:
                        continue
                    seen_texts.add(text_key)

                    source_file = hit.get("source_file", "uploaded_document.pdf")

                    all_results.append({
                        "text": text,
                        "url": source_file,             # filename as URL stand-in
                        "title": source_file,            # filename as title
                        "source_type": "pdf",
                        "citation_id": generate_citation_id("pdf", result_index),
                        "score": hit.get("score", 0.0),
                    })
                    result_index += 1

            except Exception as e:
                print(f"[PDF Search] Error searching for '{query}': {e}")

        print(f"[PDF Search] Found {len(all_results)} unique results.")
        return {"pdf_results": all_results}

    return pdf_search
