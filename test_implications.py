"""Tests for write-time implications (one LLM call per fact change)."""
import json

import pytest

from stateguard import (CheckRequest, Decision, EvidenceKind, InMemoryFactStore,
                        LLMVerifier, SQLiteFactStore, StateGuard)
from stateguard.implications import (CONFLICTS, ONLY_PREVIOUS, CachedImplicationProvider,
                                     Implication, LLMImplicationProvider,
                                     StaticImplicationProvider, clean, parse)

RULES = StaticImplicationProvider({
    ("diet", "vegan"): {CONFLICTS: ["paneer", "ghee", "cheese omelette"]},
    ("plan", "Premium"): {ONLY_PREVIOUS: ["can't watch in 4K", "upgrade to Premium"]},
    ("role", "viewer"): {CONFLICTS: ["delete the project", "reset passwords"]},
})


@pytest.fixture(params=["memory", "sqlite"])
def guard(request):
    store = InMemoryFactStore() if request.param == "memory" else SQLiteFactStore(":memory:")
    g = StateGuard(store, implications=RULES)
    g.remember("maya", "diet", "vegetarian", valid_from="2026-01-01")
    g.remember("maya", "diet", "vegan", valid_from="2026-07-02")
    g.remember("acct", "plan", "Basic", valid_from="2026-01-01")
    g.remember("acct", "plan", "Premium", valid_from="2026-06-15")
    g.remember("ravi", "role", "viewer")            # never changed
    return g


def dec(g, q, e):
    return g.check(CheckRequest(q, entity_ids=[e]))


def test_implicit_premise_now_caught(guard):
    r = dec(guard, "Maya wants a paneer tikka recipe for tonight", "maya")
    assert r.decision is Decision.BLOCK                       # 'recipe' is a diet cue
    ev = r.evidence[0]
    assert ev.kind is EvidenceKind.IMPLIED_CONFLICT and ev.assumed_value == "paneer"
    assert "vegan" in r.reason and r.suggested_query is None  # no blind rewrite


def test_implied_without_cue_is_review(guard):
    assert dec(guard, "Pick up some ghee for Maya", "maya").decision is Decision.REVIEW


def test_only_fits_previous(guard):
    r = dec(guard, "Why can't I watch in 4K?", "acct")
    assert r.decision in (Decision.REVIEW, Decision.BLOCK)
    assert "only_fits_previous" in r.evidence[0].note


def test_contradiction_of_unchanged_fact(guard):
    assert dec(guard, "Ravi will reset passwords for the team", "ravi").decision \
        is not Decision.PASS


@pytest.mark.parametrize("q", [
    "Maya is vegan now, what can replace paneer in palak paneer?",   # names current value
    "Maya doesn't eat paneer any more, ideas?",                       # negated / direct
    "Is paneer vegan?",                                               # asks
])
def test_implications_respect_framing(guard, q):
    assert dec(guard, q, "maya").decision is not Decision.BLOCK


def test_implications_dropped_when_value_changes_again(guard):
    guard.remember("maya", "diet", "vegetarian", valid_from="2026-09-01")
    assert dec(guard, "Maya wants a paneer tikka recipe", "maya").decision is Decision.PASS


def test_unrelated_entity_unaffected(guard):
    assert dec(guard, "Paneer tikka recipe please", "acct").decision is Decision.PASS


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------

def _fn(payload, tokens=120):
    calls = []

    def f(system, user):
        calls.append(user)
        if isinstance(payload, Exception):
            raise payload
        return (payload if isinstance(payload, str) else json.dumps(payload)), tokens
    f.calls = calls
    return f


def test_llm_provider_called_once_per_change_and_cost_tracked():
    fn = _fn({"conflicts_with_current": ["paneer", "Ghee", "vegan", "food", ""],
              "only_fits_previous": ["vegetarian thali"]})
    g = StateGuard(implications=LLMImplicationProvider(complete_fn=fn))
    g.remember("maya", "diet", "vegetarian")
    g.remember("maya", "diet", "vegan", reason="ethics")
    g.remember("maya", "diet", "vegan")          # same value: no new call
    for _ in range(5):
        g.check("paneer recipe for maya", ["maya"])  # queries never call the model
    assert len(fn.calls) == 2                    # first value + the change
    assert g.write_tokens == 240 and g.write_calls == 2
    assert "maya" not in fn.calls[-1].lower()    # entity id is not sent
    assert "ethics" in fn.calls[-1] and "'vegetarian'" in fn.calls[-1]
    phrases = [p for p, _ in g._get_impl("maya", "diet", None)]
    assert "vegan" not in phrases                # current value removed by clean()
    assert "paneer" in phrases


@pytest.mark.parametrize("payload", ["nope", {"conflicts_with_current": "paneer"},
                                     RuntimeError("down")])
def test_provider_failure_is_recorded_not_fatal(payload):
    g = StateGuard(implications=LLMImplicationProvider(complete_fn=_fn(payload)))
    g.remember("maya", "diet", "vegetarian")
    g.remember("maya", "diet", "vegan")
    assert g.write_errors and g._get_impl("maya", "diet", None) == []
    assert g.check("vegetarian dinner for maya", ["maya"]).decision is Decision.BLOCK


def test_clean_filters():
    items = [Implication("vegan", CONFLICTS), Implication("the", CONFLICTS),
             Implication("a very long phrase with far too many words", CONFLICTS),
             Implication("Paneer", CONFLICTS), Implication("paneer", CONFLICTS),
             Implication("x", "bogus_kind")]
    assert [i.phrase for i in clean(items, "vegan", ["vegetarian"])] == ["Paneer"]


def test_parse_strict():
    assert parse('{"conflicts_with_current": ["a"], "only_fits_previous": []}').error is None
    assert parse('{"conflicts_with_current": [1]}').error


def test_cache_reuses_across_entities_and_runs(tmp_path):
    fn = _fn({"conflicts_with_current": ["paneer"], "only_fits_previous": []})
    path = tmp_path / "impl.json"
    cached = CachedImplicationProvider(LLMImplicationProvider(complete_fn=fn), path)
    g = StateGuard(implications=cached)
    for who in ("maya", "lena"):
        g.remember(who, "diet", "vegetarian")
        g.remember(who, "diet", "vegan")
    assert len(fn.calls) == 2 and cached.hits == 2          # second user hit the cache
    again = CachedImplicationProvider(LLMImplicationProvider(complete_fn=fn), path)
    g2 = StateGuard(implications=again)
    g2.remember("x", "diet", "vegetarian")
    g2.remember("x", "diet", "vegan")
    assert len(fn.calls) == 2 and again.hits == 2           # fully reproducible offline


def test_verifier_sees_flagged_phrase():
    seen = []

    def vfn(system, user):
        seen.append(user)
        return json.dumps({"assumes_outdated_state": True, "confidence": 0.9,
                           "reason": "paneer is dairy", "assumed_value": "vegetarian",
                           "corrected_query": "vegan tofu tikka"}), 80

    g = StateGuard(implications=RULES, verifier=LLMVerifier(complete_fn=vfn))
    g.remember("maya", "diet", "vegetarian", valid_from="2026-01-01")
    g.remember("maya", "diet", "vegan", valid_from="2026-07-02")
    r = g.check("Pick up some ghee for Maya", ["maya"])
    assert r.decision is Decision.BLOCK and "'ghee'" in seen[0]
