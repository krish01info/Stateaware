"""
Smoke-test for stateguard Layer 1.
Run: python test_layer1.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from stateguard.db    import init_db, upsert_fact
from stateguard.layer1 import check, Status

conn = init_db(":memory:")

# Seed the DB with Alice's current facts
upsert_fact(conn, "Alice", "diet",     "vegan",     reason="Baseline")
upsert_fact(conn, "Alice", "location", "New York",  reason="Baseline")

# -- Test 1: CONFLICT -- query assumes Alice is still vegetarian
r = check(conn, "Can you recommend a vegetarian restaurant for Alice?")
assert r.status == Status.CONFLICT, f"Expected CONFLICT, got {r.status}"
assert r.conflicts[0].current_value == "vegan"
assert r.conflicts[0].queried_value == "vegetarian"
print(f"[OK]  CONFLICT detected: {r}")

# -- Test 2: PASS -- query correctly assumes Alice is vegan
r = check(conn, "Alice wants vegan options. What should she order?")
assert r.status == Status.PASS, f"Expected PASS, got {r.status}"
print(f"[OK]  PASS (no stale premise): {r}")

# -- Test 3: UNCERTAIN -- diet attribute mentioned, no parseable value
r = check(conn, "Does Alice have any dietary restrictions I should know about?")
assert r.status == Status.UNCERTAIN, f"Expected UNCERTAIN, got {r.status}"
print(f"[OK]  UNCERTAIN (send to Layer 2): {r}")

# -- Test 4: UNKNOWN -- no entity or attribute recognised
r = check(conn, "What is the capital of France?")
assert r.status == Status.UNKNOWN, f"Expected UNKNOWN, got {r.status}"
print(f"[OK]  UNKNOWN (unrelated query, pass through): {r}")

conn.close()
print("\nAll Layer-1 tests passed.")
