"""
Smoke-test for stateguard Layer 2 (mock backend -- no API key needed).
Run: python test_layer2.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from stateguard.db     import init_db, upsert_fact
from stateguard.layer1 import check, Status
from stateguard.layer2 import check_l2

conn = init_db(":memory:")
upsert_fact(conn, "Alice", "diet", "vegetarian", reason="Baseline")
upsert_fact(conn, "Alice", "diet", "vegan",      reason="User updated Aug 14 2026")

# -- Test 1: UNCERTAIN query -> Layer 2 should confirm stale (mock)
query = "Does Alice have any dietary restrictions? She mentioned being vegetarian."
l1 = check(conn, query)
print(f"Layer 1 result : {l1.status}")

l2 = check_l2(conn, query, l1, provider="mock")
print(f"Layer 2 result : {l2.status}")
print(f"Corrected query: {l2.corrected_query}")
assert l2.status == Status.CONFLICT, f"Expected CONFLICT, got {l2.status}"
print("[OK]  Layer 2 correctly identified stale premise.\n")

# -- Test 2: CONFLICT query -> Layer 2 confirms and rewrites
query2 = "Book a vegetarian meal for Alice."
l1b = check(conn, query2)
l2b = check_l2(conn, query2, l1b, provider="mock")
assert l2b.status == Status.CONFLICT
print(f"[OK]  CONFLICT confirmed, corrected: {l2b.corrected_query}")

# -- Test 3: PASS query -> Layer 2 should not change anything
query3 = "Alice wants vegan options. What should she order?"
l1c = check(conn, query3)
l2c = check_l2(conn, query3, l1c, provider="mock")
assert l2c.status == Status.PASS
print("[OK]  PASS query correctly untouched by Layer 2.")

conn.close()
print("\nAll Layer-2 tests passed.")
