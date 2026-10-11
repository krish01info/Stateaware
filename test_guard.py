"""Tests for the StateGuard decision engine, resolvers, extractors and verifier."""
import json

import pytest

from stateguard import (CheckRequest, Decision, EvidenceKind, HeuristicEntityResolver,
                        InMemoryFactStore, LLMVerifier, SQLiteFactStore, StateGuard)
from stateguard.normalize import find_mentions, normalize, SynonymTable
from stateguard.resolvers import ExplicitEntityResolver
from stateguard.verifier import parse_result


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------

def test_normalise_variants():
    assert normalize("Gluten-Free") == normalize("gluten free")
    assert normalize("Vegetarians") == normalize("vegetarian")
    assert normalize("$9.99/mo") .startswith("9.99")
    assert normalize("9.99 USD") == "9.99"


def test_longest_value_wins():
    m = find_mentions("ask the senior engineer", ["engineer", "senior engineer"])
    assert [x.value for x in m] == ["senior engineer"]


def test_no_substring_false_positive():
    # "cat" must not match inside "category", "age" not inside "manage"
    assert find_mentions("pick a category to manage", ["cat", "age"]) == []


def test_synonyms():
    syn = SynonymTable({"NYC": "New York"})
    assert find_mentions("flights from NYC", ["New York"], syn)[0].exact


# ---------------------------------------------------------------------------
# engine fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=["memory", "sqlite"])
def guard(request):
    store = InMemoryFactStore() if request.param == "memory" else SQLiteFactStore(":memory:")
    g = StateGuard(store)
    g.remember("alice", "diet", "vegetarian", valid_from="2026-01-01")
    g.remember("alice", "diet", "vegan", valid_from="2026-08-14", reason="told us")
    g.remember("alice", "location", "Seattle", valid_from="2025-01-01")
    g.remember("alice", "location", "Portland", valid_from="2026-01-01")
    g.remember("alice", "location", "Austin", valid_from="2026-06-01")
    g.remember("acct_7", "plan", "Basic", valid_from="2026-01-01")
    g.remember("acct_7", "plan", "Pro", valid_from="2026-05-01")
    g.remember("thermostat_1", "mode", "heating", valid_from="2026-01-01")
    g.remember("thermostat_1", "mode", "cooling", valid_from="2026-07-01")
    g.remember("bob", "job", "teacher")          # never changed
    return g


def d(guard, q, ids):
    return guard.check(CheckRequest(q, entity_ids=ids))


# ---------------------------------------------------------------------------
# BLOCK: explicit stale premises, across domains
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q,ids,fix", [
    ("Book a vegetarian dinner for Alice", ["alice"], "vegan"),
    ("Order Alice a vegetarian thali", ["alice"], None),  # no cue word -> REVIEW ok
    ("Upgrade options from my Basic plan?", ["acct_7"], "Pro"),
    ("Why is the thermostat still heating?", ["thermostat_1"], None),
])
def test_block_or_review_stale(guard, q, ids, fix):
    r = d(guard, q, ids)
    assert r.decision in (Decision.BLOCK, Decision.REVIEW)
    if fix:
        assert r.decision is Decision.BLOCK
        assert fix in r.suggested_query


def test_block_on_value_two_versions_back(guard):
    r = d(guard, "Hiking trails near Seattle?", ["alice"])
    assert r.decision is Decision.BLOCK
    stale = [e for e in r.evidence if e.kind is EvidenceKind.STALE_VALUE][0]
    assert stale.versions_back == 2 and stale.current_value == "Austin"
    assert r.suggested_query == "Hiking trails near Austin?"


# ---------------------------------------------------------------------------
# PASS: false-positive controls (adversarial neutral queries)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q,ids", [
    ("Find vegan restaurants for Alice", ["alice"]),                 # current value
    ("Alice is no longer vegetarian, what should she try?", ["alice"]),  # negated / history
    ("Alice switched to vegan instead of vegetarian, recipes?", ["alice"]),  # contrast
    ("What's the capital of France?", ["alice"]),                    # unrelated
    ("Plan a trip to Paris for Bob", ["bob"]),                       # nothing changed
    ("Bob the teacher needs a laptop", ["bob"]),                     # unchanged fact
    ("Can I keep my Pro plan?", ["acct_7"]),                         # current value
    ("A basic question about billing", ["thermostat_1"]),            # 'basic' belongs to another entity
])
def test_pass_neutral(guard, q, ids):
    assert d(guard, q, ids).decision is Decision.PASS


@pytest.mark.parametrize("q", [
    "Is Alice still vegetarian?",                   # asks, does not assume
    "Alice is visiting Seattle next month, hotels?",  # forward-looking
    "I used to be on the Basic plan, any refund?",    # acknowledges history
])
def test_framed_mentions_are_not_blocked(guard, q):
    ids = ["acct_7"] if "plan" in q else ["alice"]
    assert d(guard, q, ids).decision is not Decision.BLOCK


def test_implicit_premise_without_verifier_passes_with_evidence(guard):
    r = d(guard, "Suggest a restaurant for Alice", ["alice"])
    assert r.decision is Decision.PASS
    assert any(e.kind is EvidenceKind.ATTRIBUTE_CUE for e in r.evidence)


def test_implicit_premise_can_be_routed_to_review(guard):
    guard.implicit_without_verifier = Decision.REVIEW
    assert d(guard, "Suggest a restaurant for Alice", ["alice"]).decision is Decision.REVIEW


def test_scope_isolation():
    g = StateGuard()
    g.remember("policy", "refund_window", "30 days", scope="EU", valid_from="2026-01-01")
    g.remember("policy", "refund_window", "14 days", scope="US", valid_from="2026-01-01")
    g.remember("policy", "refund_window", "7 days", scope="US", valid_from="2026-06-01")
    r = g.check(CheckRequest("Customer wants the 30 days refund window", ["policy"], scope="EU"))
    assert r.decision is Decision.PASS
    r = g.check(CheckRequest("Customer wants the 14 days refund window", ["policy"], scope="US"))
    assert r.decision is Decision.BLOCK


def test_value_reverted_is_not_stale():
    g = StateGuard()
    g.remember("u", "location", "Delhi", valid_from="2025-01-01")
    g.remember("u", "location", "Mumbai", valid_from="2025-06-01")
    g.remember("u", "location", "Delhi", valid_from="2026-01-01")
    assert g.check("Cafes in Delhi", ["u"]).decision is Decision.PASS
    assert g.check("Cafes in Mumbai", ["u"]).decision is Decision.BLOCK


# ---------------------------------------------------------------------------
# entity resolution
# ---------------------------------------------------------------------------

def test_heuristic_resolver_by_name_and_alias():
    g = StateGuard()
    g.remember("user_42", "diet", "vegetarian", valid_from="2026-01-01")
    g.remember("user_42", "diet", "vegan", valid_from="2026-02-01")
    g.remember("user_43", "diet", "keto")
    g.store.add_alias("Priya", "user_42")
    assert g.check("Vegetarian lunch for Priya").decision is Decision.BLOCK


def test_first_person_resolution():
    g = StateGuard(resolver=HeuristicEntityResolver(self_entity="me"))
    g.remember("me", "location", "Pune", valid_from="2026-01-01")
    g.remember("me", "location", "Sonipat", valid_from="2026-07-01")
    g.remember("other", "location", "Goa")
    assert g.check("Find gyms in Pune near my place").decision is Decision.BLOCK


def test_unresolved_entity_with_stale_premise_is_review():
    g = StateGuard(resolver=HeuristicEntityResolver(single_entity_default=False))
    g.remember("u1", "plan", "Starter", valid_from="2026-01-01")
    g.remember("u1", "plan", "Scale", valid_from="2026-02-01")
    g.remember("u2", "plan", "Scale")
    r = g.check("How do I cancel the Starter plan?")
    assert r.decision is Decision.REVIEW
    assert r.evidence[0].kind is EvidenceKind.UNRESOLVED_ENTITY
    assert g.check("How do I cancel?").decision is Decision.PASS


def test_explicit_resolver_ignores_names():
    g = StateGuard(resolver=ExplicitEntityResolver())
    g.remember("alice", "diet", "vegetarian", valid_from="2026-01-01")
    g.remember("alice", "diet", "vegan", valid_from="2026-02-01")
    # no ids given: entity unknown, but the premise matches stored history -> REVIEW
    assert g.check("vegetarian food for alice").decision is Decision.REVIEW
    assert g.check("pizza for alice").decision is Decision.PASS
    assert g.check("vegetarian food for alice", ["alice"]).decision is Decision.BLOCK


# ---------------------------------------------------------------------------
# verifier
# ---------------------------------------------------------------------------

def _fn(payload, tokens=50):
    calls = []

    def f(system, user):
        calls.append(user)
        if isinstance(payload, Exception):
            raise payload
        return (payload if isinstance(payload, str) else json.dumps(payload)), tokens
    f.calls = calls
    return f


def _verified_guard(fn, **kw):
    g = StateGuard(verifier=LLMVerifier(complete_fn=fn), **kw)
    g.remember("alice", "diet", "vegetarian", valid_from="2026-01-01", reason="intro")
    g.remember("alice", "diet", "vegan", valid_from="2026-08-14", reason="told us")
    return g


def test_verifier_only_called_for_ambiguous_cases():
    fn = _fn({"assumes_outdated_state": True, "confidence": 0.9, "reason": "r",
              "assumed_value": "vegetarian", "corrected_query": "x"})
    g = _verified_guard(fn)
    g.check("Book a vegetarian dinner", ["alice"])     # deterministic BLOCK
    g.check("Find vegan food", ["alice"])              # deterministic PASS
    assert fn.calls == []
    r = g.check("Suggest a restaurant for Alice", ["alice"])  # implicit -> verifier
    assert len(fn.calls) == 1 and r.decision is Decision.BLOCK and r.tokens_used == 50
    assert "VERSIONS (oldest -> newest)" in fn.calls[0] and "told us" in fn.calls[0]


def test_verifier_confirms_pass():
    fn = _fn({"assumes_outdated_state": False, "confidence": 0.95, "reason": "neutral",
              "assumed_value": None, "corrected_query": None})
    r = _verified_guard(fn).check("Suggest a restaurant for Alice", ["alice"])
    assert r.decision is Decision.PASS and "verifier" in r.layers_used


@pytest.mark.parametrize("payload", [
    "definitely not json",
    {"assumes_outdated_state": "yes", "confidence": 0.9},
    {"assumes_outdated_state": True},                       # missing confidence
    {"assumes_outdated_state": True, "confidence": 3},
    RuntimeError("rate limited"),
])
def test_verifier_failures_become_review(payload):
    r = _verified_guard(_fn(payload)).check("Suggest a restaurant for Alice", ["alice"])
    assert r.decision is Decision.REVIEW
    assert any(e.kind is EvidenceKind.PROVIDER_ERROR for e in r.evidence)


def test_low_confidence_verifier_is_review():
    fn = _fn({"assumes_outdated_state": True, "confidence": 0.4, "reason": "maybe"})
    assert _verified_guard(fn).check("Suggest a restaurant for Alice", ["alice"]).decision \
        is Decision.REVIEW


def test_parse_result_accepts_fenced_json():
    r = parse_result('```json\n{"assumes_outdated_state": false, "confidence": 0.8}\n```')
    assert r.error is None and r.is_stale is False


def test_unknown_provider():
    with pytest.raises(ValueError):
        LLMVerifier("nope")


def test_generic_uncued_word_is_review_not_block():
    g = StateGuard()
    g.remember("acct", "billing_cycle", "monthly", valid_from="2026-01-01")
    g.remember("acct", "billing_cycle", "annual", valid_from="2026-08-01")
    # "monthly" here describes a report, not the billing cycle
    assert g.check("Show me my monthly usage report", ["acct"]).decision is Decision.REVIEW
    # with an attribute cue it is a clear stale premise
    assert g.check("When is my next monthly bill due?", ["acct"]).decision is Decision.BLOCK


# ---------------------------------------------------------------------------
# Hinglish framing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q,expected", [
    ("Maya ke liye vegetarian khana order karo", Decision.BLOCK),           # assumes old
    ("Maya pehle vegetarian thi, ab vegan hai, kya banau?", Decision.PASS),  # history + current
    ("Maya ab vegetarian nahi hai, dinner ideas?", Decision.PASS),           # negated
    ("Kya Maya abhi bhi vegetarian hai?", Decision.REVIEW),                 # asks
    ("Maya agle hafte Seattle ghoomne jaayegi, hotels?", Decision.REVIEW),  # future
])
def test_hinglish(q, expected):
    g = StateGuard()
    g.remember("maya", "diet", "vegetarian", valid_from="2026-01-01")
    g.remember("maya", "diet", "vegan", valid_from="2026-07-02")
    g.remember("maya", "location", "Seattle", valid_from="2025-01-01")
    g.remember("maya", "location", "Austin", valid_from="2026-06-01")
    assert g.check(q, ["maya"]).decision is expected
