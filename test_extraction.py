"""Tests for learning facts from user messages (StateGuard.observe)."""
import json

import pytest

from stateguard import Decision, LLMFactExtractor, SQLiteFactStore, StateGuard
from stateguard.extraction import attribute_name, parse
from stateguard.implications import StaticImplicationProvider


def scripted(replies, tokens=60):
    """complete_fn that returns canned extractions in order and logs prompts."""
    calls = []

    def f(system, user):
        calls.append(user)
        r = replies[len(calls) - 1]
        if isinstance(r, Exception):
            raise r
        return json.dumps(r) if not isinstance(r, str) else r, tokens
    f.calls = calls
    return f


def test_observe_learns_and_supersedes():
    fn = scripted([
        {"facts": [{"attribute": "Diet", "value": "vegetarian", "confidence": 0.95,
                    "evidence": "I'm vegetarian"}]},
        {"facts": [{"attribute": "diet", "value": "vegan", "confidence": 0.9,
                    "evidence": "went fully vegan"}]},
    ])
    g = StateGuard(SQLiteFactStore(":memory:"), fact_extractor=LLMFactExtractor(complete_fn=fn))
    g.observe("me", "I'm vegetarian, love paneer", valid_from="2026-01-01")
    r = g.observe("me", "Update: I went fully vegan last month", valid_from="2026-08-01")
    assert [f.value for f in r.written] == ["vegan"]
    assert "KNOWN ATTRIBUTES: diet" in fn.calls[1]            # attribute name reused
    hist = g.store.history("me", "diet")
    assert [h.value for h in hist] == ["vegan", "vegetarian"]
    assert hist[0].source == "message" and hist[0].reason == "went fully vegan"
    assert g.check("Book me a vegetarian dinner", ["me"]).decision is Decision.BLOCK
    assert g.write_tokens == 120


def test_low_confidence_is_returned_not_written():
    fn = scripted([{"facts": [{"attribute": "location", "value": "Goa", "confidence": 0.3}]}])
    g = StateGuard(fact_extractor=LLMFactExtractor(complete_fn=fn))
    r = g.observe("me", "maybe I'll move to Goa someday?")
    assert not r.written and r.skipped[0].value == "Goa"
    assert g.store.current("me") == []


def test_extraction_errors_are_recorded():
    g = StateGuard(fact_extractor=LLMFactExtractor(complete_fn=scripted([RuntimeError("x")])))
    r = g.observe("me", "I'm vegan")
    assert r.error and g.write_errors and g.store.current("me") == []


def test_observe_triggers_implications():
    fn = scripted([
        {"facts": [{"attribute": "diet", "value": "vegetarian", "confidence": 0.9}]},
        {"facts": [{"attribute": "diet", "value": "vegan", "confidence": 0.9}]},
    ])
    g = StateGuard(fact_extractor=LLMFactExtractor(complete_fn=fn),
                   implications=StaticImplicationProvider(
                       {("diet", "vegan"): {"conflicts_with_current": ["paneer"]}}))
    g.observe("me", "vegetarian here", valid_from="2026-01-01")
    g.observe("me", "now vegan", valid_from="2026-02-01")
    assert g.check("paneer tikka recipe for me", ["me"]).decision is not Decision.PASS


def test_cache_makes_reprocessing_free(tmp_path):
    fn = scripted([{"facts": [{"attribute": "diet", "value": "vegan", "confidence": 0.9}]}])
    path = tmp_path / "x.json"
    LLMFactExtractor(complete_fn=fn, cache_path=path).extract("I'm vegan", [])
    again = LLMFactExtractor(complete_fn=scripted([]), cache_path=path)
    assert again.extract("I'm vegan", []).facts[0].value == "vegan" and again.hits == 1


@pytest.mark.parametrize("raw,ok", [
    ('{"facts": []}', True),
    ('{"facts": [{"attribute": "diet", "value": "vegan", "confidence": 0.9}]}', True),
    ('{"facts": [{"attribute": "diet", "value": "vegan"}]}', False),
    ('{"facts": "vegan"}', False),
    ("no json", False),
])
def test_parse(raw, ok):
    assert (parse(raw).error is None) is ok


def test_sentence_values_are_dropped():
    r = parse('{"facts": [{"attribute": "diet", "value": "I have been fully vegan since '
              'June of last year at least", "confidence": 0.9}]}')
    assert r.error is None and r.facts == []


def test_attribute_name():
    assert attribute_name(" Dietary Preference ") == "dietary_preference"
    assert attribute_name("location(city)") == "location_city"
