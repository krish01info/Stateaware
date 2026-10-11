"""
Minimal StateGuard integration: check every user request before it reaches
your main model, and act on the three decisions.

Run:  python examples/basic_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stateguard import CheckRequest, Decision, SQLiteFactStore, StateGuard

guard = StateGuard(store=SQLiteFactStore(":memory:"))

# Your app records facts as it learns them. A new value supersedes the old one,
# and the full history is kept.
guard.remember("account_123", "plan", "Basic", valid_from="2026-01-10")
guard.remember("account_123", "plan", "Premium", valid_from="2026-06-15",
               reason="user upgraded in settings", source="billing")
guard.remember("account_123", "shipping_address", "12 Park Street, Kolkata",
               valid_from="2025-09-01")
guard.remember("account_123", "shipping_address", "4 MG Road, Gurugram",
               valid_from="2026-06-20", reason="moved")


def handle(query: str) -> str:
    result = guard.check(CheckRequest(query, entity_ids=["account_123"]))

    if result.decision is Decision.BLOCK:
        # High-confidence stale premise: correct it before generation.
        return f"BLOCK  -> {result.reason}\n         rewrite: {result.suggested_query!r}"
    if result.decision is Decision.REVIEW:
        # Ambiguous: your choice — ask the user, run a verifier, or just log it.
        return f"REVIEW -> {result.reason}"
    return "PASS   -> send to the model unchanged"


for q in [
    "Does the Basic plan include 4K streaming?",
    "Ship my order to 12 Park Street, Kolkata as usual.",
    "How many screens does Premium allow?",
    "Was I charged twice when I moved from Basic to Premium?",
    "What is your return policy?",
]:
    print(f"{q}\n  {handle(q)}\n")
