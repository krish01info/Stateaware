"""
Tests for StateGuard Layer 1.

Run:  pytest test_layer1.py -v
"""
import pytest

from stateguard.db import init_db, upsert_fact
from stateguard.layer1 import (
    Status, check, extract_attribute, extract_entity_names,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn():
    """In-memory DB seeded with Alice's current facts."""
    c = init_db(":memory:")
    upsert_fact(c, "Alice", "diet",     "vegan",    reason="Baseline")
    upsert_fact(c, "Alice", "location", "New York", reason="Baseline")
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Core Layer-1 status tests
# ---------------------------------------------------------------------------

def test_conflict_vegetarian(conn):
    """Query assumes Alice is vegetarian → CONFLICT (she's vegan)."""
    r = check(conn, "Can you recommend a vegetarian restaurant for Alice?")
    assert r.status == Status.CONFLICT
    assert r.conflicts[0].current_value == "vegan"
    assert r.conflicts[0].queried_value == "vegetarian"


def test_pass_vegan(conn):
    """Query correctly assumes Alice is vegan → PASS."""
    r = check(conn, "Alice wants vegan options. What should she order?")
    assert r.status == Status.PASS


def test_uncertain_no_parseable_value(conn):
    """Diet attribute mentioned but no parseable value → UNCERTAIN."""
    r = check(conn, "Does Alice have any dietary restrictions I should know about?")
    assert r.status == Status.UNCERTAIN


def test_unknown_unrelated_query(conn):
    """No entity or attribute recognised → UNKNOWN."""
    r = check(conn, "What is the capital of France?")
    assert r.status == Status.UNKNOWN


# ---------------------------------------------------------------------------
# Word-boundary regression tests (the substring bug)
# ---------------------------------------------------------------------------

class TestExtractAttributeWordBoundary:
    """Ensure extract_attribute uses word boundaries, not substring matching."""

    @pytest.mark.parametrize("word", [
        "manage", "message", "storage", "damage", "average",
        "package", "voyage", "courage", "advantage", "passage",
    ])
    def test_age_not_triggered_by_substring(self, word):
        """Words containing 'age' as a substring must NOT match the age alias."""
        assert extract_attribute(f"How do I {word} my files?") is None

    @pytest.mark.parametrize("word", [
        "carpet", "competition", "trumpet", "puppet", "muppet",
    ])
    def test_pet_not_triggered_by_substring(self, word):
        """Words containing 'pet' as a substring must NOT match the pet alias."""
        assert extract_attribute(f"I need a new {word}.") is None

    def test_age_actual_match(self):
        """The standalone word 'age' should still match correctly."""
        assert extract_attribute("How old is Alice? What's her age?") == "age"

    def test_pet_actual_match(self):
        """The standalone word 'pet' should still match correctly."""
        assert extract_attribute("Does Alice have a pet?") == "pet"

    def test_multi_word_alias_still_works(self):
        """Multi-word aliases like 'lives in' should still match."""
        assert extract_attribute("Alice lives in Austin.") == "location"

    def test_diet_keyword(self):
        """Simple diet keyword should match."""
        assert extract_attribute("What is Alice's diet?") == "diet"


# ---------------------------------------------------------------------------
# Entity extraction stopword tests
# ---------------------------------------------------------------------------

class TestExtractEntityStopwords:
    """Sentence-initial words must not be mistaken for person names."""

    @pytest.mark.parametrize("word", [
        "Will", "Let", "Some", "But", "Just", "Also",
        "Should", "Would", "Could", "May", "Must",
    ])
    def test_stopword_not_entity(self, word):
        assert word not in extract_entity_names(f"{word} you help me with this?")

    def test_real_name_extracted(self):
        names = extract_entity_names("Alice went to the store.")
        assert "Alice" in names
