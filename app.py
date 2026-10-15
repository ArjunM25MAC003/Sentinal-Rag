"""
 SentinelRAG — Multi-Source Research Agent with Claim Provenance
===================================================================

Main Gradio application with two tabs:
  Tab 1 "Research Agent"  — run full multi-source RAG with claim verification
  Tab 2 "Claim Verifier"  — standalone NLI tool for ad-hoc verification

Launch:
    python app.py
    → opens at http://localhost:7860

Deploy to HuggingFace Spaces:
    gradio deploy
"""

# ──────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────
import os
import traceback

import gradio as gr
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load .env file for API keys (GROQ_API_KEY, TAVILY_API_KEY)
load_dotenv()

# NLTK sentence tokeniser — download required data on first run
import nltk
for _tok in ("punkt_tab", "punkt"):
    try:
        nltk.data.find(f"tokenizers/{_tok}")
    except LookupError:
        nltk.download(_tok, quiet=True)
from nltk.tokenize import sent_tokenize

# Lazy-loaded heavy dependencies
# SentinelRAG internals
from sentinelrag.graph import build_graph, run_query
from sentinelrag.pdf_processor import PDFKnowledgeBase

# ──────────────────────────────────────────────────────────────────
# Cached NLI model (loaded once, reused across requests)
# ──────────────────────────────────────────────────────────────────
_nli_model = None


def _get_nli_model():
    """Lazily load the cross-encoder NLI model for the Claim Verifier tab."""
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder

        _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    return _nli_model


def _softmax(x):
    """Numerically stable softmax."""
    e = np.exp(np.asarray(x, dtype=np.float64) - np.max(x))
    return e / e.sum()


# NLI label indices for cross-encoder/nli-deberta-v3-small
_CONTRA, _ENTAIL, _NEUTRAL = 0, 1, 2


# ──────────────────────────────────────────────────────────────────
# UI helper: source name mapping
# ──────────────────────────────────────────────────────────────────
_SOURCE_MAP = {
    "Web Search": "web",
    "Wikipedia": "wikipedia",
    "arXiv": "arxiv",
    "PDF Documents": "pdf",
}


# ──────────────────────────────────────────────────────────────────
# HTML formatters
# ──────────────────────────────────────────────────────────────────

def _faithfulness_badge(score: float) -> str:
    """Return an HTML badge coloured by faithfulness score."""
    if score >= 0.70:
        colour, emoji = "#22c55e",       # green
    elif score >= 0.50:
        colour, emoji = "#eab308",      # yellow
    else:
        colour, emoji = "#ef4444",       # red

    return f"""
    <div style="display:flex; align-items:center; justify-content:center; padding:12px;">
      <div style="background:{colour}; color:#fff; padding:10px 24px;
                  border-radius:24px; font-weight:700; font-size:1.15em;
                  box-shadow:0 2px 8px {colour}55;">
        {emoji} Faithfulness Score: {score:.2f}
      </div>
    </div>"""


def _verdict_badge(verdict: str) -> str:
    """Map a verdict string to an emoji-prefixed label."""
    return {
        "VERIFIED":           "VERIFIED",
        "VERIFIED_ALTERNATE": "VERIFIED (alt source)",
        "UNCERTAIN":          "UNCERTAIN",
        "CONTRADICTED":       "CONTRADICTED",
        "UNCITED":            "UNCITED",
        "SOURCE_NOT_FOUND":   " SOURCE NOT FOUND",
    }.get(verdict, f" {verdict}")


def _verdict_html(verdict: str, score: float) -> str:
    """Full HTML block for the standalone Claim Verifier tab."""
    colours = {
        "VERIFIED": "#22c55e",
        "UNCERTAIN": "#eab308",
        "CONTRADICTED": "#ef4444",
    }
    c = colours.get(verdict, "#94a3b8")
    pct = max(0, min(100, int(score * 100)))

    return f"""
    <div style="padding:16px; border:2px solid {c}; border-radius:12px;">
      <h3 style="color:{c}; margin:0 0 8px 0;">{_verdict_badge(verdict)}</h3>
      <div style="width:100%; background:#e5e7eb; border-radius:6px; overflow:hidden;">
        <div style="width:{pct}%; background:{c}; height:10px;"></div>
      </div>
      <small style="display:block; margin-top:6px; color:#555;">
        Entailment confidence: {score:.3f}
      </small>
    </div>"""


# ──────────────────────────────────────────────────────────────────
# Tab 1 — Research Agent
# ──────────────────────────────────────────────────────────────────

def research(question: str, pdf_files, sources: list, progress=gr.Progress()):
    """
    Main research pipeline handler.

    1. Map UI checkbox labels → internal source keys.
    2. Process uploaded PDFs (if any).
    3. Run the full LangGraph pipeline via run_query().
    4. Format all outputs for the Gradio UI.
    """
    if not question or not question.strip():
        return (
            "Please enter a research question.",
            "", pd.DataFrame(), "", "",
        )

    try:
        # ── 1. Map sources ─────────────────────────────────────────
        progress(0.05, desc="Preparing sources...")
        active = [_SOURCE_MAP[s] for s in (sources or []) if s in _SOURCE_MAP]
        if not active:
            active = ["web", "wikipedia"]  # sensible default

        # ── 2. Process PDFs ────────────────────────────────────────
        pdf_kb = None
        if pdf_files and "pdf" in active:
            progress(0.10, desc="Indexing uploaded PDFs...")
            pdf_kb = PDFKnowledgeBase()
            paths = [f.name if hasattr(f, "name") else str(f) for f in pdf_files]
            pdf_kb.add_pdfs(paths)
            if pdf_kb.is_empty():
                pdf_kb = None  # no usable text extracted

        # ── 3. Run the full pipeline ───────────────────────────────
        progress(0.20, desc="Planning sub-questions...")
        result = run_query(
            question=question.strip(),
            active_sources=active,
            pdf_kb=pdf_kb,
        )
        progress(0.85, desc="Formatting results...")

        # ── 4. Format the answer ───────────────────────────────────
        answer = result.get("final_answer") or result.get("draft_answer", "No answer generated.")

        # ── 5. Faithfulness badge ──────────────────────────────────
        faith_score = result.get("faithfulness_score", 0.0)
        badge_html = _faithfulness_badge(faith_score)

        # ── 6. Claim verification table ────────────────────────────
        claims = result.get("claim_verification", [])
        if claims:
            rows = []
            for c in claims:
                rows.append({
                    "Claim": c.get("claim", ""),
                    "Verdict": _verdict_badge(c.get("verdict", "")),
                    "Citation": c.get("citation_id", "—"),
                    "Evidence": (c.get("source_evidence") or "—")[:200],
                    "Score": f"{c.get('entailment_score', 0.0):.3f}",
                })
            df = pd.DataFrame(rows)
        else:
            df = pd.DataFrame(columns=["Claim", "Verdict", "Citation", "Evidence", "Score"])

        # ── 7. Conflicts section ───────────────────────────────────
        conflicts = result.get("conflicts", [])
        if conflicts:
            conflict_lines = []
            for i, c in enumerate(conflicts, 1):
                conflict_lines.append(
                    f"**Conflict {i}:** {c.get('claim', 'N/A')}\n"
                    f"-  Contradicted by: `[{c.get('contradicting_source', '?')}]`\n"
                    f"-  Supported by: `[{c.get('supporting_source', '?')}]`\n"
                    f"- {c.get('note', '')}\n"
                )
            conflicts_md = "\n".join(conflict_lines)
        else:
            conflicts_md = " No conflicts detected across sources."

        # ── 8. Source details accordion ────────────────────────────
        all_sources = result.get("all_sources", [])
        if all_sources:
            src_lines = []
            for src in all_sources:
                cid = src.get("citation_id", "?")
                title = src.get("title", "Untitled")
                stype = src.get("source_type", "unknown").upper()
                url = src.get("url", "N/A")
                text_preview = (src.get("text", ""))[:300]
                src_lines.append(
                    f"### [{cid}] {title}\n"
                    f"**Type:** {stype} · **URL:** {url}\n\n"
                    f"> {text_preview}...\n\n---"
                )
            sources_md = "\n\n".join(src_lines)
        else:
            sources_md = "No sources were retrieved."

        progress(1.0, desc="Done!")
        return answer, badge_html, df, conflicts_md, sources_md

    except Exception as e:
        tb = traceback.format_exc()
        error_msg = f"❌ **Error:** {e}\n\n<details><summary>Traceback</summary>\n\n```\n{tb}\n```\n</details>"
        return error_msg, "", pd.DataFrame(), "", ""


# ──────────────────────────────────────────────────────────────────
# Tab 2 — Standalone Claim Verifier
# ──────────────────────────────────────────────────────────────────

def verify_claim(source_text: str, claim: str):
    """
    Verify a single claim against a source passage using NLI.

    Returns coloured verdict HTML and the best matching evidence sentence.
    """
    if not source_text or not claim:
        return "<p style='color:#888'>Please provide both source text and a claim.</p>", ""

    try:
        model = _get_nli_model()
        source_sents = sent_tokenize(source_text)

        if not source_sents:
            return "<p>Could not tokenise source text into sentences.</p>", ""

        best_score = -1.0
        best_sent = ""
        best_verdict = "UNCERTAIN"

        for sent in source_sents:
            raw = model.predict([(sent, claim)])
            logits = np.asarray(raw)
            if logits.ndim == 1:
                logits = logits.reshape(1, -1)
            probs = _softmax(logits[0])

            ent = float(probs[_ENTAIL])
            contra = float(probs[_CONTRA])

            if ent > 0.70:
                v, s = "VERIFIED", ent
            elif contra > 0.70:
                v, s = "CONTRADICTED", contra
            else:
                v, s = "UNCERTAIN", max(ent, float(probs[_NEUTRAL]))

            if s > best_score:
                best_score = s
                best_sent = sent
                best_verdict = v

        html = _verdict_html(best_verdict, best_score)
        evidence_md = f"**Best matching evidence sentence:**\n\n> {best_sent}"
        return html, evidence_md

    except Exception as e:
        return f"<p style='color:red'>Error: {e}</p>", ""


# ──────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────
_CSS = """
.gradio-container { font-family: 'Inter', -apple-system, sans-serif; }
.badge-green  { background: #22c55e; }
.badge-yellow { background: #eab308; }
.badge-red    { background: #ef4444; }
"""


# ──────────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Soft(),
    css=_CSS,
    title="SentinelRAG — Multi-Source Research Agent",
) as demo:

    # ── Header ─────────────────────────────────────────────────────
    gr.Markdown(
        "#  SentinelRAG — Multi-Source Research Agent\n"
        "Queries **Web + Wikipedia + arXiv + PDFs** in parallel, then verifies "
        "every output sentence against its exact source using NLI cross-encoder "
        "verification. Conflicts between sources are automatically detected."
    )

    with gr.Tabs():
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # TAB 1 — Research Agent
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        with gr.Tab(" Research Agent"):
            with gr.Row():
                # ── Left column: inputs ────────────────────────────
                with gr.Column(scale=2):
                    question_input = gr.Textbox(
                        label="Research Question",
                        placeholder=(
                            "e.g., What are the latest advances in "
                            "transformer architectures?"
                        ),
                        lines=2,
                    )

                    with gr.Row():
                        source_toggles = gr.CheckboxGroup(
                            choices=["Web Search", "Wikipedia", "arXiv", "PDF Documents"],
                            value=["Web Search", "Wikipedia", "arXiv", "PDF Documents"],
                            label="Data Sources",
                        )
                        pdf_upload = gr.File(
                            label="Upload Reference PDFs (optional)",
                            file_types=[".pdf"],
                            file_count="multiple",
                        )

                    submit_btn = gr.Button(
                        " Generate Research Report", variant="primary", size="lg"
                    )

                    gr.Examples(
                        examples=[
                            [
                                "What are the key differences between GPT-4 and Llama 3?",
                                ["Web Search", "Wikipedia", "arXiv"],
                            ],
                            [
                                "Explain the mechanism of action of GLP-1 receptor agonists.",
                                ["Web Search", "Wikipedia"],
                            ],
                            [
                                "What is retrieval-augmented generation and how does it work?",
                                ["Web Search", "Wikipedia", "arXiv"],
                            ],
                        ],
                        inputs=[question_input, source_toggles],
                    )

                # ── Right column: outputs ──────────────────────────
                with gr.Column(scale=3):
                    faithfulness_badge = gr.HTML(label="Faithfulness")
                    answer_output = gr.Markdown(label="Verified Answer")

                    with gr.Accordion("📊 Claims Verification Matrix", open=True):
                        verification_table = gr.Dataframe(
                            headers=["Claim", "Verdict", "Citation", "Evidence", "Score"],
                            wrap=True,
                        )

                    with gr.Accordion(" Detected Conflicts", open=False):
                        conflicts_output = gr.Markdown()

                    with gr.Accordion(" Source Details", open=False):
                        sources_output = gr.Markdown()

            submit_btn.click(
                fn=research,
                inputs=[question_input, pdf_upload, source_toggles],
                outputs=[
                    answer_output,
                    faithfulness_badge,
                    verification_table,
                    conflicts_output,
                    sources_output,
                ],
            )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # TAB 2 — Claim Verifier
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        with gr.Tab(" Claim Verifier"):
            gr.Markdown(
                "Paste any source text and a claim — the NLI model will tell you "
                "whether the claim is **supported**, **contradicted**, or **uncertain** "
                "based on the evidence."
            )

            with gr.Row():
                with gr.Column():
                    verify_source = gr.Textbox(
                        label="Source / Evidence Text",
                        lines=5,
                        placeholder="Paste the source document text here...",
                    )
                    verify_claim_input = gr.Textbox(
                        label="Claim to Verify",
                        lines=2,
                        placeholder="State a claim to check against the source...",
                    )
                    verify_btn = gr.Button("🔬 Verify Claim", variant="primary")

                with gr.Column():
                    verify_verdict = gr.HTML(label="Verdict")
                    verify_evidence = gr.Markdown(label="Evidence")

            verify_btn.click(
                fn=verify_claim,
                inputs=[verify_source, verify_claim_input],
                outputs=[verify_verdict, verify_evidence],
            )

    # ── Footer ─────────────────────────────────────────────────────
    gr.Markdown(
        "<center>"
        "<small>Built with LangGraph · Groq · Tavily · Sentence-Transformers · FAISS · Gradio</small>"
        "</center>"
    )


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
