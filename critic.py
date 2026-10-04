"""
Critic Node for SentinelRAG — NLI Claim Verification Engine
============================================================

This is the most critical component of the entire pipeline. It takes the
draft answer produced by the synthesiser and *rigorously* verifies every
single sentence against its cited source using Natural Language Inference.

How it works:
  1. Tokenise the draft answer into individual sentences (claims).
  2. For each sentence, extract inline citations like [WEB1], [WIKI2].
  3. Look up the cited source in all_sources.
  4. Run an NLI cross-encoder to classify the (source, claim) pair as
     entailment / contradiction / neutral.
  5. If the primary source contradicts the claim, check whether any
     *other* source supports it — if so, flag a cross-source conflict.
  6. Locate the exact sentence in the source that best supports the claim.

Outputs:
  - claim_verification : per-claim verdict list
  - conflicts          : cross-source contradiction records
  - faithfulness_score : fraction of claims that are verified
  - final_answer       : the (unchanged) draft answer
"""

import re
from typing import Dict, Any, List, Tuple

import numpy as np

# NLTK sentence tokeniser — we need punkt_tab for modern NLTK
import nltk
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    print("[Critic] Downloading NLTK punkt_tab tokeniser...")
    nltk.download("punkt_tab", quiet=True)

# Also try legacy punkt as a fallback
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


# ---------------------------------------------------------------------------
# NLI model singleton — loaded once, reused across calls
# ---------------------------------------------------------------------------
_nli_model = None


def _get_nli_model():
    """Lazily load the NLI cross-encoder to save startup time."""
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder

        print("[Critic] Loading NLI model: cross-encoder/nli-deberta-v3-small ...")
        _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
        print("[Critic] NLI model loaded.")
    return _nli_model


def _softmax(logits) -> np.ndarray:
    """Numerically stable softmax over a 1-D array of logits."""
    x = np.asarray(logits, dtype=np.float64)
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ---------------------------------------------------------------------------
# Label ordering for cross-encoder/nli-deberta-v3-small
# ---------------------------------------------------------------------------
# The model outputs 3 logits in the order:
#   index 0 = contradiction
#   index 1 = entailment
#   index 2 = neutral
IDX_CONTRADICTION = 0
IDX_ENTAILMENT = 1
IDX_NEUTRAL = 2


# ---------------------------------------------------------------------------
# Helper: find the best supporting sentence inside a source paragraph
# ---------------------------------------------------------------------------

def _best_supporting_sentence(source_text: str, claim: str) -> Tuple[str, float]:
    """
    Split the source text into sentences and return the one whose
    entailment probability with the claim is highest.

    Returns:
        (best_sentence, best_entailment_prob)
    """
    model = _get_nli_model()
    source_sents = sent_tokenize(source_text)

    if not source_sents:
        return source_text[:200], 0.0

    # Build (premise, hypothesis) pairs
    pairs = [(s, claim) for s in source_sents]
    raw_scores = model.predict(pairs)

    # raw_scores shape: (N, 3) or (3,) when N=1
    if raw_scores.ndim == 1:
        raw_scores = raw_scores.reshape(1, -1)

    best_idx = 0
    best_ent = 0.0
    for i, logits in enumerate(raw_scores):
        probs = _softmax(logits)
        ent = float(probs[IDX_ENTAILMENT])
        if ent > best_ent:
            best_ent = ent
            best_idx = i

    return source_sents[best_idx], best_ent


# ---------------------------------------------------------------------------
# Citation extraction regex
# ---------------------------------------------------------------------------
# Matches [WEB1], [WIKI2], [ARXIV3], [PDF4], etc.
_CITE_PATTERN = re.compile(r"\[([A-Z]+\d+)\]")


# ---------------------------------------------------------------------------
# Main critic node
# ---------------------------------------------------------------------------

def critic_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify every sentence in the draft answer against its cited source.

    Args:
        state: Must contain 'draft_answer' and 'all_sources'.

    Returns:
        Dict with keys: claim_verification, conflicts,
        faithfulness_score, final_answer.
    """
    draft = state.get("draft_answer", "")
    all_sources = state.get("all_sources", [])

    # Nothing to verify?
    if not draft or not all_sources:
        print("[Critic] No draft or no sources - skipping verification.")
        return {
            "claim_verification": [],
            "conflicts": [],
            "faithfulness_score": 0.0,
            "final_answer": draft,
        }

    model = _get_nli_model()

    # Build a quick lookup from citation_id → source dict.
    # Citation IDs are stored WITHOUT brackets, e.g. "WEB1"
    source_map: Dict[str, dict] = {}
    for src in all_sources:
        cid = src.get("citation_id", "")
        if cid:
            source_map[cid] = src

    # Tokenise the draft into individual claim-sentences
    claims = sent_tokenize(draft)
    print(f"[Critic] Verifying {len(claims)} claims against {len(source_map)} sources...")

    verified_list: List[Dict[str, Any]] = []
    conflicts: List[Dict[str, Any]] = []
    n_verified = 0

    for claim in claims:
        # ── 1. Extract inline citation IDs ─────────────────────────
        cite_ids = _CITE_PATTERN.findall(claim)  # e.g. ["WEB1", "WIKI2"]

        # Strip citation brackets so NLI sees clean natural language
        clean_claim = _CITE_PATTERN.sub("", claim).strip()
        if not clean_claim:
            continue  # skip empty fragments

        # ── 2. UNCITED claim ───────────────────────────────────────
        if not cite_ids:
            verified_list.append({
                "claim": clean_claim,
                "verdict": "UNCITED",
                "citation_id": None,
                "source_evidence": None,
                "entailment_score": 0.0,
            })
            continue

        # Use the first cited source as the primary reference
        primary_id = cite_ids[0]
        source = source_map.get(primary_id)

        # ── 3. Source not found ────────────────────────────────────
        if source is None:
            verified_list.append({
                "claim": clean_claim,
                "verdict": "SOURCE_NOT_FOUND",
                "citation_id": primary_id,
                "source_evidence": None,
                "entailment_score": 0.0,
            })
            continue

        source_text = source.get("text", "")

        # ── 4. Run NLI on (source_text, claim) ────────────────────
        raw = model.predict([(source_text, clean_claim)])
        # Ensure 2-D
        logits = np.asarray(raw)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        probs = _softmax(logits[0])

        ent_prob = float(probs[IDX_ENTAILMENT])
        contra_prob = float(probs[IDX_CONTRADICTION])

        verdict = "UNCERTAIN"
        best_evidence = ""
        final_score = ent_prob

        # ── 5. Classification logic ───────────────────────────────
        if ent_prob > 0.70:
            # ✅ High entailment → VERIFIED
            verdict = "VERIFIED"
            n_verified += 1
            best_evidence, _ = _best_supporting_sentence(source_text, clean_claim)

        elif contra_prob > 0.50:
            # Primary source contradicts the claim.
            # Check if any *other* source supports it → conflict detection
            alt_found = False
            for alt_id, alt_src in source_map.items():
                if alt_id == primary_id:
                    continue

                alt_raw = model.predict([(alt_src.get("text", ""), clean_claim)])
                alt_logits = np.asarray(alt_raw)
                if alt_logits.ndim == 1:
                    alt_logits = alt_logits.reshape(1, -1)
                alt_probs = _softmax(alt_logits[0])

                if float(alt_probs[IDX_ENTAILMENT]) > 0.70:
                    verdict = "VERIFIED_ALTERNATE"
                    n_verified += 1
                    alt_found = True
                    best_evidence, final_score = _best_supporting_sentence(
                        alt_src.get("text", ""), clean_claim
                    )

                    # Record the cross-source conflict
                    conflicts.append({
                        "claim": clean_claim,
                        "contradicting_source": primary_id,
                        "supporting_source": alt_id,
                        "note": (
                            f"Source [{primary_id}] contradicts this claim, "
                            f"but source [{alt_id}] supports it."
                        ),
                    })
                    break  # one alternate is enough

            if not alt_found:
                # ❌ No alternate support → genuinely contradicted
                verdict = "CONTRADICTED"

        else:
            # ⚠️ Neither strong entailment nor contradiction
            verdict = "UNCERTAIN"

        # ── 6. Store the per-claim result ──────────────────────────
        verified_list.append({
            "claim": clean_claim,
            "verdict": verdict,
            "citation_id": primary_id,
            "source_evidence": best_evidence[:250] if best_evidence else None,
            "entailment_score": round(final_score, 4),
        })

    # ── 7. Compute faithfulness score ──────────────────────────────
    total = max(len(verified_list), 1)
    faithfulness = round(n_verified / total, 3)

    print(f"[Critic] Results: {n_verified}/{len(verified_list)} verified "
          f"(faithfulness={faithfulness}), {len(conflicts)} conflicts detected.")

    return {
        "claim_verification": verified_list,
        "conflicts": conflicts,
        "faithfulness_score": faithfulness,
        "final_answer": draft,
    }
