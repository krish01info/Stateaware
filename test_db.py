"""
Smoke-test for stateguard.db
Run: python test_db.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from stateguard.db import (
    init_db, upsert_fact, get_current_fact,
    get_supersession_context, list_facts
)

# Use in-memory DB so nothing is written to disk during tests
conn = init_db(":memory:")

# --- Scenario: Alice changes her diet ---
upsert_fact(conn, "Alice", "diet",     "vegetarian", reason="Baseline profile")
upsert_fact(conn, "Alice", "location", "New York",   reason="Baseline profile")

fact = get_current_fact(conn, "Alice", "diet")
assert fact["value"] == "vegetarian", f"Expected vegetarian, got {fact}"
print("[OK]  Baseline fact stored:", fact)

# Supersede: Alice is now vegan
upsert_fact(conn, "Alice", "diet", "vegan", reason="User updated diet on Aug 14 2026")

current = get_current_fact(conn, "Alice", "diet")
assert current["value"] == "vegan", f"Expected vegan, got {current}"
print("[OK]  Superseded fact is current:", current)

ctx = get_supersession_context(conn, "Alice", "diet")
assert ctx["old_value"] == "vegetarian"
assert ctx["new_value"] == "vegan"
print("[OK]  Supersession context for Layer-2 LLM:", ctx)

all_facts = list_facts(conn, "Alice")
assert len(all_facts) == 2  # diet + location (both current)
print("[OK]  All current facts for Alice:", all_facts)

conn.close()
print("\nAll tests passed.")
