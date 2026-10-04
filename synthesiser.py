"""
Synthesiser Node for SentinelRAG
=================================

Takes all retrieved source chunks (web, Wikipedia, arXiv, PDF) and
synthesises a comprehensive draft answer using an LLM, with strict
inline citation requirements.

Key design decisions:
    - Uses Groq's openai/gpt-oss-20b for fast generation.
  - Re-assigns globally unique citation IDs to avoid collisions
    across source types (in case different retrievers produced
    overlapping numbering).
  - The system prompt enforces five citation rules that minimise
    hallucination and encourage explicit disagreement reporting.
"""

import os
from typing import Dict, Any, List

from groq import Groq

from sentinelrag.utils import generate_citation_id, format_source_context


def draft_synthesis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge all source results and draft a cited answer.

    Workflow:
      1. Collect web / wiki / arxiv / pdf results into one list.
      2. Re-assign globally unique citation IDs (WEB1, WIKI1, ARXIV1, PDF1, …).
      3. Build a formatted context block for the LLM prompt.
      4. Call Groq to generate a draft answer with inline citations.

    Args:
        state: Current graph state. Expected keys: web_results, wiki_results,
               arxiv_results, pdf_results, question.

    Returns:
        Dict with 'draft_answer' (str) and 'all_sources' (list of dicts).
    """
    # ── 1. Merge results from all retrievers ───────────────────────
    all_sources: List[Dict[str, Any]] = []

    # Counters per source type so citation IDs are globally unique
    type_counters = {"web": 0, "wikipedia": 0, "arxiv": 0, "pdf": 0}

    for key, source_type in [
        ("web_results", "web"),
        ("wiki_results", "wikipedia"),
        ("arxiv_results", "arxiv"),
        ("pdf_results", "pdf"),
    ]:
        for src in state.get(key, []):
            stype = src.get("source_type", source_type)
            type_counters[stype] = type_counters.get(stype, 0) + 1
            # Re-assign a clean, unique citation ID
            src_copy = dict(src)
            src_copy["citation_id"] = generate_citation_id(stype, type_counters[stype])
            all_sources.append(src_copy)

    # ── 2. Handle empty sources ────────────────────────────────────
    if not all_sources:
        print("[Synthesiser] No sources retrieved. Cannot generate an answer.")
        return {
            "draft_answer": (
                "I could not find any relevant information from the selected "
                "sources to answer your question. Please try broadening your "
                "query or enabling additional source types."
            ),
            "all_sources": [],
        }

    print(f"[Synthesiser] Drafting answer from {len(all_sources)} sources...")

    # ── 3. Build the formatted context for the LLM ─────────────────
    context_lines = []
    for src in all_sources:
        cid = src.get("citation_id", "???")
        title = src.get("title", "Untitled")
        text = src.get("text", "")
        context_lines.append(f"[{cid}] ({title}): {text}")

    context_block = "\n\n".join(context_lines)

    # ── 4. Craft the prompt ────────────────────────────────────────
    system_prompt = (
        "You are an expert research assistant. Your job is to synthesise "
        "a comprehensive, well-structured answer using ONLY the sources "
        "provided below.\n\n"
        "CITATION RULES — follow these strictly:\n"
        "1. Cite EVERY factual claim with its citation ID in square brackets, "
        "   e.g. [WEB1], [WIKI2], [ARXIV1], [PDF1].\n"
        "2. Do NOT invent or infer information that is not in the sources.\n"
        "3. If two sources disagree, explicitly state the contradiction and "
        "   cite both sides.\n"
        "4. Structure your answer with clear paragraphs.\n"
        "5. End with a brief summary of confidence / limitations.\n\n"
        "SOURCES:\n"
        f"{context_block}"
    )

    user_prompt = f"Question: {state.get('question', '')}\n\nPlease provide a well-cited answer."

    # ── 5. Call Groq API ───────────────────────────────────────────
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,   # low temp for factual synthesis
            max_tokens=1500,
        )
        draft_answer = response.choices[0].message.content
        print(f"[Synthesiser] Draft generated ({len(draft_answer)} chars).")

    except Exception as e:
        print(f"[Synthesiser] Error calling Groq API: {e}")
        draft_answer = (
            "An error occurred while generating the draft answer. "
            f"Please check your GROQ_API_KEY. Error: {e}"
        )

    return {
        "draft_answer": draft_answer,
        "all_sources": all_sources,
    }
