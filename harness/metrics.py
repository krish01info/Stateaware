"""
Balanced scorecard for a three-way (PASS / REVIEW / BLOCK) interceptor.

A single "accuracy" number rewards a system that blocks everything on a
stale-heavy benchmark. This module reports the quantities that actually
matter for a deployed interceptor, each computed two ways where REVIEW is
ambiguous:

* strict  – only BLOCK counts as intercepting.
* lenient – BLOCK or REVIEW counts as intercepting.

Each record passed in is a dict with at least:
    expected : "stale" | "neutral"
    decision : "pass" | "review" | "block"
and optionally: group keys (e.g. "domain", "category", "dim"), "latency_ms",
"tokens", "error" (bool), "covered" (bool: an entity was resolved).
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Iterable

DECISIONS = ("pass", "review", "block")


def _div(a: float, b: float):
    return round(a / b, 4) if b else None


def _f1(p, r):
    return round(2 * p * r / (p + r), 4) if p and r else (0.0 if p is not None and r is not None else None)


def confusion(records: Iterable[dict]) -> dict[str, dict[str, int]]:
    cm = {e: {d: 0 for d in DECISIONS} for e in ("stale", "neutral")}
    for r in records:
        cm[r["expected"]][r["decision"]] += 1
    return cm


def scorecard(records: list[dict]) -> dict:
    cm = confusion(records)
    s, n = cm["stale"], cm["neutral"]
    n_stale, n_neutral, total = sum(s.values()), sum(n.values()), len(records)

    out = {"n": total, "n_stale": n_stale, "n_neutral": n_neutral, "confusion": cm}
    for mode, hit in (("strict", ("block",)), ("lenient", ("block", "review"))):
        tp = sum(s[d] for d in hit)
        fp = sum(n[d] for d in hit)
        p = _div(tp, tp + fp)
        r = _div(tp, n_stale)
        out[mode] = {
            "precision": p,
            "recall": r,
            "f1": _f1(p, r),
            "false_positive_rate": _div(fp, n_neutral),
        }
    out["block_precision"] = out["strict"]["precision"]
    out["review_rate"] = _div(s["review"] + n["review"], total)
    out["neutral_pass_rate"] = _div(n["pass"], n_neutral)

    if records and "covered" in records[0]:
        out["coverage"] = _div(sum(bool(r.get("covered")) for r in records), total)
    if records and "error" in records[0]:
        out["error_rate"] = _div(sum(bool(r.get("error")) for r in records), total)
    lat = [r["latency_ms"] for r in records if r.get("latency_ms") is not None]
    if lat:
        lat.sort()
        out["latency_ms"] = {
            "p50": round(statistics.median(lat), 3),
            "p95": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 3),
            "mean": round(statistics.fmean(lat), 3),
        }
    tok = [r.get("tokens", 0) or 0 for r in records]
    out["tokens"] = {"total": sum(tok), "mean": round(statistics.fmean(tok), 2) if tok else 0}
    if records and "write_tokens" in records[0]:
        # Write-time cost (implication generation) is paid per fact change,
        # not per query; report it separately so the two are never mixed.
        out["write"] = {"tokens": sum(r.get("write_tokens", 0) for r in records),
                        "calls": sum(r.get("write_calls", 0) for r in records)}
    return out


def by_group(records: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[str(r.get(key))].append(r)
    return {g: scorecard(rs) for g, rs in sorted(groups.items())}


def format_table(cards: dict[str, dict], title: str = "") -> str:
    """Render group scorecards as a fixed-width text table."""
    def f(x):
        return "  -  " if x is None else f"{x:5.2f}"

    head = (f"{'group':<22}{'n':>4}  {'BLKprec':>7} {'recS':>5} {'recL':>5} "
            f"{'FPR-S':>5} {'FPR-L':>5} {'review':>6} {'p50ms':>6}")
    lines = [title, head, "-" * len(head)] if title else [head, "-" * len(head)]
    for g, c in cards.items():
        lat = c.get("latency_ms", {}).get("p50")
        lines.append(
            f"{g:<22}{c['n']:>4}  {f(c['block_precision']):>7} {f(c['strict']['recall'])} "
            f"{f(c['lenient']['recall'])} {f(c['strict']['false_positive_rate'])} "
            f"{f(c['lenient']['false_positive_rate'])} {f(c['review_rate']):>6} "
            f"{('  -' if lat is None else f'{lat:6.2f}')}")
    return "\n".join(lines)
