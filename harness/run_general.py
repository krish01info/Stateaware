"""
Run StateGuard over the general (non-STALE) state-change fixtures and print a
balanced scorecard: overall, per domain, per query category, plus every miss.

Usage
-----
python harness/run_general.py                       # Layer 1 only, explicit entity IDs
python harness/run_general.py --resolver heuristic  # make StateGuard guess the entity
python harness/run_general.py --provider groq       # add the Layer-2 verifier
python harness/run_general.py --implicit review     # route implicit premises to REVIEW
python harness/run_general.py --implications groq   # write-time implications (cached)

The fixtures were written by the developers of this library. Treat them as
a DEVELOPMENT set: report results on a held-out set written by someone who
has not seen the engine.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stateguard import (CheckRequest, Decision, EvidenceKind, HeuristicEntityResolver,
                        InMemoryFactStore, LLMVerifier, StateGuard)
from harness.metrics import by_group, format_table, manifest, scorecard
from stateguard.implications import CachedImplicationProvider, LLMImplicationProvider

HERE = Path(__file__).parent
DEFAULT_FIXTURES = HERE / "fixtures" / "general_state_cases.jsonl"
DEFAULT_CACHE = HERE / "cache" / "implications.json"


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def make_implications(args):
    """Implication provider for this run (cached, so reruns cost nothing)."""
    if not args.implications:
        return None
    if args.implications == "cache-only":
        inner = LLMImplicationProvider(complete_fn=_cache_miss)
    else:
        inner = LLMImplicationProvider(args.implications)
    return CachedImplicationProvider(inner, args.implications_cache)


def _cache_miss(system, user):
    raise LookupError("not in implications cache (run once with a provider to fill it)")


def run_case(case: dict, args, implications=None) -> dict:
    verifier = LLMVerifier(args.provider) if args.provider else None
    guard = StateGuard(
        InMemoryFactStore(),
        resolver=HeuristicEntityResolver(single_entity_default=False),
        verifier=verifier,
        implicit_without_verifier=Decision(args.implicit),
        implications=implications,
    )
    for attr, value, ts in case["history"]:
        guard.remember(case["entity"], attr, value, valid_from=ts)
    # aliases so the heuristic resolver can find "Maya", "pump 3", "Ravi", ...
    for alias in case.get("aliases", []) or _default_aliases(case["entity"]):
        guard.store.add_alias(alias, case["entity"])

    ids = [case["entity"]] if args.resolver == "explicit" else None
    r = guard.check(CheckRequest(case["query"], entity_ids=ids))
    return {
        "id": case["id"], "domain": case["domain"], "category": case["category"],
        "expected": case["expected"], "decision": r.decision.value,
        "confidence": r.confidence, "reason": r.reason, "suggested": r.suggested_query,
        "latency_ms": r.latency_ms, "tokens": r.tokens_used,
        "error": any(e.kind is EvidenceKind.PROVIDER_ERROR for e in r.evidence),
        "covered": not any(e.kind is EvidenceKind.UNRESOLVED_ENTITY for e in r.evidence),
        "write_tokens": guard.write_tokens, "write_calls": guard.write_calls,
        "write_errors": len(guard.write_errors),
        "query": case["query"],
    }


def _default_aliases(entity: str) -> list[str]:
    head = entity.split("_")[0]
    return [entity.replace("_", " "), head] if head != entity else [entity]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    ap.add_argument("--resolver", choices=["explicit", "heuristic"], default="explicit")
    ap.add_argument("--provider", default=None, help="groq | openai | gemini (Layer 2)")
    ap.add_argument("--implicit", choices=["pass", "review"], default="pass",
                    help="decision for implicit premises when no verifier is set")
    ap.add_argument("--implications", default=None,
                    help="groq | openai | gemini | cache-only: generate write-time implications")
    ap.add_argument("--implications-cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args(argv)

    impl = make_implications(args)
    rows = [run_case(c, args, impl) for c in load(args.fixtures)]
    overall = scorecard(rows)
    print(format_table({"ALL": overall}, "Overall"))
    print()
    print(format_table(by_group(rows, "domain"), "By domain"))
    print()
    print(format_table(by_group(rows, "category"), "By query category"))

    misses = [r for r in rows
              if (r["expected"] == "stale" and r["decision"] != "block")
              or (r["expected"] == "neutral" and r["decision"] != "pass")]
    print(f"\nNot ideal ({len(misses)}/{len(rows)}):")
    for r in misses:
        print(f"  [{r['id']}] expected={r['expected']:<7} got={r['decision']:<6} "
              f"({r['category']}) {r['query']}")

    args.out.mkdir(parents=True, exist_ok=True)
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    mode = (f"general_{args.resolver}_{args.provider or 'l1'}_{args.implicit}"
            f"_impl-{args.implications or 'none'}")
    if impl is not None:
        errs = sum(r["write_errors"] for r in rows)
        print(f"\nWrite-time implications: {impl.misses} generated, {impl.hits} from cache, "
              f"{overall.get('write', {}).get('tokens', 0)} tokens this run, {errs} errors")
    (args.out / f"{tag}_{mode}.json").write_text(json.dumps(
        {"manifest": manifest(vars(args)), "mode": mode, "overall": overall, "by_domain": by_group(rows, "domain"),
         "by_category": by_group(rows, "category"), "rows": rows}, indent=2))
    return overall


if __name__ == "__main__":
    main()
