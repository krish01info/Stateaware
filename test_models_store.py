"""Tests for the public data model and the versioned fact stores."""
import pytest

from stateguard.models import CheckRequest, CheckResponse, Decision, Evidence, EvidenceKind
from stateguard.store import FactStore, InMemoryFactStore, SQLiteFactStore
from stateguard.db import init_db, upsert_fact


def test_request_defaults():
    r = CheckRequest("hello")
    assert r.entity_ids is None and r.metadata == {} and r.scope is None


def test_response_serialises():
    r = CheckResponse(Decision.BLOCK, 0.9,
                      [Evidence(EvidenceKind.STALE_VALUE, "a", "diet", "vegan", "vegetarian")])
    d = r.to_dict()
    assert d["decision"] == "block"
    assert d["evidence"][0]["kind"] == "stale_value"
    assert r.is_blocked and not r.needs_review


@pytest.fixture(params=["memory", "sqlite"])
def store(request):
    return InMemoryFactStore() if request.param == "memory" else SQLiteFactStore(":memory:")


def test_protocol(store):
    assert isinstance(store, FactStore)


def test_versioning_keeps_full_history(store):
    store.record("acct", "plan", "basic", valid_from="2026-01-01")
    store.record("acct", "plan", "pro", valid_from="2026-03-01", reason="upgraded")
    store.record("acct", "plan", "team", valid_from="2026-06-01", source="billing")
    hist = store.history("acct", "plan")
    assert [h.value for h in hist] == ["team", "pro", "basic"]
    assert hist[0].is_current and not hist[1].is_current
    assert hist[1].valid_to == "2026-06-01"
    assert store.current("acct", "plan")[0].source == "billing"
    assert hist[1].reason == "upgraded"


def test_same_value_is_not_a_change(store):
    store.record("d", "status", "online")
    store.record("d", "status", "online")
    assert len(store.history("d", "status")) == 1


def test_scopes_do_not_supersede_each_other(store):
    store.record("policy", "refund_window", "30 days", scope="EU")
    store.record("policy", "refund_window", "14 days", scope="US")
    assert store.current("policy", "refund_window", scope="EU")[0].value == "30 days"
    assert store.current("policy", "refund_window", scope="US")[0].value == "14 days"
    assert len(store.current("policy")) == 2


def test_aliases(store):
    store.record("user_42", "diet", "vegan")
    store.add_alias("Priya", "user_42")
    assert store.aliases() == {"priya": "user_42"}


def test_sqlite_store_reads_legacy_rows():
    conn = init_db(":memory:")
    upsert_fact(conn, "Alice", "diet", "vegetarian", reason="baseline")
    upsert_fact(conn, "Alice", "diet", "vegan", reason="changed")
    s = SQLiteFactStore(conn)
    assert [h.value for h in s.history("Alice", "diet")] == ["vegan", "vegetarian"]
    assert s.current("Alice")[0].reason == "changed"
