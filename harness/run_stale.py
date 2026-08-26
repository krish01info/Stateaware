"""
StateGuard – Person C
Evaluation harness: runs the full StateGuard pipeline over the STALE dataset.

Usage
-----
# Ablation 1 – Layer 1 only (no LLM)
python harness/run_stale.py --mode l1

# Ablation 2 – Full pipeline  L1 + L2 (cheap LLM)
python harness/run_stale.py --mode l1l2 --provider gemini

# Baseline – plain LLM sees the raw query with the haystack (no interceptor)
python harness/run_stale.py --mode baseline --provider gemini

Results are written to:
  harness/results/<timestamp>_<mode>_results.csv
  harness/results/<timestamp>_<mode>_summary.json
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset

from stateguard.db    import init_db, upsert_fact
from stateguard.layer1 import check, Status
from stateguard.layer2 import check_l2

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# STALE probing query dimensions
# dim1 = direct state recall (SR)
# dim2 = premise resistance (PR) ← primary target for StateGuard
# dim3 = implicit policy adaptation (IPA)
DIMS = ("dim1_query", "dim2_query", "dim3_query")
DIM_LABELS = {"dim1_query": "SR", "dim2_query": "PR", "dim3_query": "IPA"}


# ---------------------------------------------------------------------------
# Parse M_old / M_new into (attribute, old_value, new_value)
# ---------------------------------------------------------------------------

def _parse_explanation(explanation: str) -> tuple[str, str]:
    """
    Extract (attribute_key, new_value) from STALE's explanation string.

    Examples:
      "Spatiotemporal_Context.location(city) is now Austin."
        → ("location", "Austin")
      "Dietary_Restrictions.diet is now vegan."
        → ("diet", "vegan")
    """
    # Try to extract the part after "is now "
    if " is now " in explanation:
        parts = explanation.split(" is now ")
        new_value = parts[-1].rstrip(". ").strip()
        # Extract attribute name from the left side
        left = parts[0]
        if "(" in left and ")" in left:
            attr = left[left.rfind("(") + 1 : left.rfind(")")].strip()
        elif "." in left:
            attr = left.split(".")[-1].strip()
        else:
            attr = left.strip()
        return attr.lower(), new_value
    return "state", explanation.strip()


def _extract_old_value(m_old: str, attribute: str) -> str:
    """
    Best-effort extraction of the old attribute value from the M_old sentence.
    Falls back to the full sentence if nothing better is found.
    """
    # Try to find a quoted or parenthesised value first
    import re
    quoted = re.findall(r'"([^"]+)"|\(([^)]+)\)', m_old)
    if quoted:
        return quoted[0][0] or quoted[0][1]
    # For location: look for "in <City>" or "based in <City>"
    loc = re.search(r'\b(?:in|at|to)\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)', m_old)
    if loc and attribute in ("location", "city", "state"):
        return loc.group(1)
    # For diet: scan known diet words
    from stateguard.layer1 import DIET_VALUES
    for v in DIET_VALUES:
        if v in m_old.lower():
            return v
    # Final fallback: first capitalized word group after a verb
    cap = re.search(r'\b(?:am|is|are|been|living|based|working)\s+(?:a\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', m_old)
    if cap:
        return cap.group(1)
    return m_old  # full sentence as last resort


def seed_db(conn, row: dict) -> tuple[str, str, str, str]:
    """
    Seed the SQLite DB from one STALE row.
    Returns (entity, attribute, old_value, new_value).
    """
    uid  = row["uid"]
    m_old = row["M_old"]
    m_new = row["M_new"]
    expl  = row["explanation"]

    attribute, new_value = _parse_explanation(expl)
    entity    = f"user_{uid[:8]}"
    old_value = _extract_old_value(m_old, attribute)

    upsert_fact(conn, entity, attribute, old_value,
                reason=f"Old memory: {m_old}")
    upsert_fact(conn, entity, attribute, new_value,
                reason=f"New memory: {m_new}")

    return entity, attribute, old_value, new_value


# ---------------------------------------------------------------------------
# LLM judge: ask a cheap LLM whether the model output is stale
# ---------------------------------------------------------------------------

def _judge_llm_response(response_text: str, current_value: str,
                         provider: str = "mock") -> bool:
    """
    Returns True if the model response correctly references the *current* fact.
    For the harness we use a deterministic heuristic: check if current_value
    appears in the response text (case-insensitive).
    Replace with a proper LLM judge for the final paper results.
    """
    return current_value.lower() in response_text.lower()


# ---------------------------------------------------------------------------
# Main LLM call (used for baseline and as the downstream model)
# ---------------------------------------------------------------------------

def _call_main_llm(query: str, haystack: list, provider: str = "mock") -> tuple[str, int]:
    """
    Simulate calling the main LLM with (query, haystack context).
    Returns (response_text, approx_tokens_used).

    In 'mock' mode: returns a canned response without hitting any API.
    In 'gemini' or 'openai' mode: calls the real API.
    """
    if provider == "mock":
        # Deterministic stub: return the raw query echoed back
        return f"[MOCK RESPONSE] Query received: {query}", len(query.split()) * 2

    # Build a condensed context from the haystack (last 3 sessions only)
    context_turns = []
    for session in haystack[-3:]:
        for turn in session[-2:]:  # last 2 turns per session
            context_turns.append(f"{turn['role'].upper()}: {turn['content'][:200]}")
    context = "\n".join(context_turns)

    prompt = (
        f"You are a helpful assistant. Use the conversation history below to answer.\n\n"
        f"HISTORY:\n{context}\n\n"
        f"USER QUERY: {query}\n\n"
        f"Answer concisely and accurately."
    )
    tokens = len(prompt.split())

    if provider == "gemini":
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            model = genai.GenerativeModel("gemini-2.5-flash-lite")
            resp  = model.generate_content(prompt)
            return resp.text, tokens
        except Exception as e:
            return f"[ERROR] {e}", tokens

    if provider == "openai":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=256,
            )
            return resp.choices[0].message.content, tokens
        except Exception as e:
            return f"[ERROR] {e}", tokens

    return "[UNKNOWN PROVIDER]", 0


# ---------------------------------------------------------------------------
# Run one row through the chosen mode
# ---------------------------------------------------------------------------

def run_row(conn, row: dict, dim: str, mode: str,
            provider: str = "mock", use_ner: bool = False) -> dict:
    """
    Evaluate one (row, dim) pair.
    Returns a result dict with status, correct, latency_ms, tokens_used.

    use_ner : if False (default, preserves prior behaviour), the entity is
              handed to Layer 1 via known_entity=entity, bypassing
              extract_entity_names() entirely. That means accuracy numbers
              never reflect real-world entity resolution — the harness was
              always testing Layer 1 with NER effectively disabled. Pass
              use_ner=True to run Layer 1's actual regex-based entity
              extraction on the raw query, the way it would run in
              production when the caller doesn't already know who the
              conversation is about.
    """
    query = row["probing_queries"][dim]
    haystack = row["haystack_session"]

    t0 = time.perf_counter()

    entity, attribute, old_value, new_value = seed_db(conn, row)
    known_entity = None if use_ner else entity

    # dim1 (SR) and dim3 (IPA) queries do NOT carry a stale premise the
    # pipeline should intercept — "correct" for them means the pipeline
    # left the query alone (no false-positive CONFLICT). dim2 (PR) queries
    # DO carry a stale premise — "correct" means the pipeline caught it.
    # Previously dim1/dim3 were scored as None and silently dropped from
    # summarise(), so over-triggering (flagging everything as stale) was
    # invisible in the reported accuracy. See PR notes.
    def _score(intercepted: bool) -> bool:
        return intercepted if dim == "dim2_query" else (not intercepted)

    # ── Mode: L1 only ────────────────────────────────────────────────────────
    if mode == "l1":
        l1 = check(conn, query, known_entity=known_entity)
        latency_ms  = (time.perf_counter() - t0) * 1000
        tokens_used = 0
        intercepted = l1.status == Status.CONFLICT
        correct = _score(intercepted)
        return {
            "uid":         row["uid"],
            "type":        row["type"],
            "dim":         DIM_LABELS[dim],
            "mode":        mode,
            "status":      l1.status.value,
            "intercepted": intercepted,
            "correct":     correct,
            "latency_ms":  round(latency_ms, 2),
            "tokens_used": tokens_used,
        }

    # ── Mode: L1 + L2 ────────────────────────────────────────────────────────
    if mode == "l1l2":
        l1 = check(conn, query, known_entity=known_entity)

        # Track whether L1 resolved this query (for reporting L1 hit-rate).
        l1_resolved = l1.status in (Status.CONFLICT, Status.UNCERTAIN)

        if l1.status in (Status.CONFLICT, Status.UNCERTAIN):
            l2 = check_l2(conn, query, l1, provider=provider)
            final_status  = l2.status
            corrected     = l2.corrected_query
            tokens_used   = l2.tokens_used
            parse_error   = l2.parse_error
        else:
            final_status  = l1.status
            corrected     = None
            tokens_used   = 0
            parse_error   = False

        latency_ms  = (time.perf_counter() - t0) * 1000
        intercepted = final_status == Status.CONFLICT
        correct = _score(intercepted)
        return {
            "uid":             row["uid"],
            "type":            row["type"],
            "dim":             DIM_LABELS[dim],
            "mode":            mode,
            "status":          final_status.value,
            "l1_resolved":     l1_resolved,
            "intercepted":     intercepted,
            "corrected_query": corrected,
            "correct":         correct,
            "parse_error":     parse_error,
            "latency_ms":      round(latency_ms, 2),
            "tokens_used":     tokens_used,
        }

    # ── Mode: baseline (no interceptor, raw LLM) ────────────────────────────
    if mode == "baseline":
        resp, tokens_used = _call_main_llm(query, haystack, provider=provider)
        latency_ms  = (time.perf_counter() - t0) * 1000
        correct = _judge_llm_response(resp, new_value, provider=provider)
        return {
            "uid":          row["uid"],
            "type":         row["type"],
            "dim":          DIM_LABELS[dim],
            "mode":         mode,
            "status":       "LLM_RESPONSE",
            "intercepted":  False,
            "llm_response": resp[:200],
            "correct":      correct,
            "latency_ms":   round(latency_ms, 2),
            "tokens_used":  tokens_used,
        }

    raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Summarise results
# ---------------------------------------------------------------------------

def summarise(results: list[dict]) -> dict:
    """Compute accuracy, avg latency, total tokens by dim and type."""
    from collections import defaultdict
    buckets: dict[str, list] = defaultdict(list)
    for r in results:
        key = f"{r['type']}_{r['dim']}"
        buckets[key].append(r)

    summary = {}
    for key, rows in buckets.items():
        evaluated = [r for r in rows if r.get("correct") is not None]
        if not evaluated:
            continue
        accuracy    = sum(1 for r in evaluated if r["correct"]) / len(evaluated)
        avg_latency = sum(r["latency_ms"] for r in evaluated) / len(evaluated)
        total_tok   = sum(r.get("tokens_used", 0) for r in evaluated)
        summary[key] = {
            "n":                len(evaluated),
            "accuracy":         round(accuracy * 100, 1),
            "avg_latency_ms":   round(avg_latency, 1),
            "total_tokens":     total_tok,
        }
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="StateGuard STALE harness")
    parser.add_argument(
        "--mode", choices=["l1", "l1l2", "baseline"], default="l1",
        help="Ablation mode: l1=Layer1 only, l1l2=full pipeline, baseline=raw LLM"
    )
    parser.add_argument(
        "--provider", choices=["gemini", "openai", "groq", "mock"], default="mock",
        help="LLM backend for l1l2 and baseline modes"
    )
    parser.add_argument(
        "--dims", nargs="+", default=list(DIMS),
        choices=list(DIMS),
        help="Which query dimensions to evaluate (default: all three — "
             "dim2_query alone hides false-positive rate on dim1/dim3)"
    )
    parser.add_argument(
        "--use-ner", action="store_true",
        help="Run Layer 1's real entity extraction (extract_entity_names) "
             "instead of cheating with known_entity=<ground-truth entity>. "
             "Without this flag, entity-resolution accuracy is never tested."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of rows (for quick smoke tests)"
    )
    parser.add_argument(
        "--split", default="train",
        help="Dataset split to use (default: train)"
    )
    args = parser.parse_args()

    print(f"[StateGuard Harness] mode={args.mode}  provider={args.provider}  "
          f"dims={args.dims}  limit={args.limit}  use_ner={args.use_ner}")

    # Load dataset
    print("Loading STALEproj/STALE …")
    dataset = load_dataset("STALEproj/STALE")
    rows    = dataset[args.split]
    if args.limit:
        rows = rows.select(range(min(args.limit, len(rows))))
    print(f"  Loaded {len(rows)} rows.")

    # Use in-memory DB — each row gets a fresh seed
    # (For large runs you could reuse a persistent DB and clear between rows)
    results = []
    total   = len(rows) * len(args.dims)
    done    = 0

    for i, row in enumerate(rows):
        conn = init_db(":memory:")   # fresh DB per row
        for dim in args.dims:
            try:
                r = run_row(conn, row, dim, args.mode, args.provider, use_ner=args.use_ner)
            except Exception as exc:
                r = {
                    "uid":        row["uid"],
                    "type":       row["type"],
                    "dim":        DIM_LABELS.get(dim, dim),
                    "mode":       args.mode,
                    "error":      str(exc),
                    "correct":    None,
                    "latency_ms": 0,
                    "tokens_used": 0,
                }
            results.append(r)
            done += 1
        conn.close()

        # Progress every 10 rows
        if (i + 1) % 10 == 0 or i == len(rows) - 1:
            pct = done / total * 100
            print(f"  [{done}/{total}] {pct:.0f}%  uid={row['uid'][:8]}")

    # ── Save results ──────────────────────────────────────────────────────────
    ts        = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    csv_path  = RESULTS_DIR / f"{ts}_{args.mode}_results.csv"
    json_path = RESULTS_DIR / f"{ts}_{args.mode}_summary.json"

    # CSV – all rows
    fieldnames = sorted({k for r in results for k in r.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[OK] Results saved -> {csv_path}")

    # JSON – summary
    summary = summarise(results)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"mode": args.mode, "provider": args.provider,
                   "n_rows": len(rows), "dims": args.dims,
                   "summary": summary}, f, indent=2)
    print(f"[OK] Summary saved -> {json_path}")

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'Type+Dim':<12} {'N':>5} {'Accuracy':>10} {'Avg Lat ms':>12} {'Tokens':>8}")
    print("-" * 52)
    for key, s in sorted(summary.items()):
        print(f"{key:<12} {s['n']:>5} {s['accuracy']:>9.1f}% {s['avg_latency_ms']:>11.1f} {s['total_tokens']:>8}")


if __name__ == "__main__":
    main()
