"""
Tests for StateGuard Layer 2 (mock backend — no API key needed).

Run:  pytest test_layer2.py -v
"""
import sys
import types

import pytest

from stateguard.db import init_db, upsert_fact
from stateguard.layer1 import Status, check
from stateguard.layer2 import check_l2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn():
    """In-memory DB with Alice's diet updated from vegetarian → vegan."""
    c = init_db(":memory:")
    upsert_fact(c, "Alice", "diet", "vegetarian", reason="Baseline")
    upsert_fact(c, "Alice", "diet", "vegan",      reason="User updated Aug 14 2026")
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Layer-2 status tests
# ---------------------------------------------------------------------------

def test_uncertain_confirmed_stale(conn):
    """UNCERTAIN query mentioning 'vegetarian' → Layer 2 confirms CONFLICT."""
    query = "Does Alice have any dietary restrictions? She mentioned being vegetarian."
    l1 = check(conn, query)
    l2 = check_l2(conn, query, l1, provider="mock")
    assert l2.status == Status.CONFLICT


def test_conflict_confirmed_and_rewritten(conn):
    """CONFLICT query → Layer 2 confirms and provides corrected_query."""
    query = "Book a vegetarian meal for Alice."
    l1 = check(conn, query)
    l2 = check_l2(conn, query, l1, provider="mock")
    assert l2.status == Status.CONFLICT
    assert l2.corrected_query is not None


def test_pass_query_unchanged(conn):
    """PASS query → Layer 2 leaves it untouched."""
    query = "Alice wants vegan options. What should she order?"
    l1 = check(conn, query)
    l2 = check_l2(conn, query, l1, provider="mock")
    assert l2.status == Status.PASS


# ---------------------------------------------------------------------------
# Provider validation
# ---------------------------------------------------------------------------

def test_unknown_provider_raises(conn):
    """Unknown provider name must raise ValueError, not silently fallback."""
    query = "Does Alice have any dietary restrictions?"
    l1 = check(conn, query)
    # Force status so check_l2 actually tries the backend
    l1.status = Status.UNCERTAIN
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        check_l2(conn, query, l1, provider="typo_provider")


def test_groq_backend_uses_supported_model(monkeypatch):
    """Groq requests use the replacement model for the retired Llama 3.3 ID."""
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(
                usage=types.SimpleNamespace(total_tokens=7),
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="{}"))],
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    from stateguard.layer2 import _call_groq

    _call_groq("system", "user")
    assert captured["model"] == "openai/gpt-oss-120b"
