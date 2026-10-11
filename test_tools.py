"""Tests for checking agent tool calls before they run."""
import pytest

from stateguard import Decision, EvidenceKind, SQLiteFactStore, StateGuard
from stateguard.implications import StaticImplicationProvider
from stateguard.tools import flatten


@pytest.fixture
def guard():
    g = StateGuard(SQLiteFactStore(":memory:"), implications=StaticImplicationProvider(
        {("diet", "vegan"): {"conflicts_with_current": ["paneer", "ghee"]}}))
    g.remember("maya", "location", "Bengaluru", valid_from="2025-03-01")
    g.remember("maya", "location", "Pune", valid_from="2026-04-10")
    g.remember("maya", "diet", "vegetarian", valid_from="2026-01-01")
    g.remember("maya", "diet", "vegan", valid_from="2026-07-02")
    g.remember("acct", "plan", "Basic", valid_from="2026-01-01")
    g.remember("acct", "plan", "Premium", valid_from="2026-06-15")
    g.remember("acct", "shipping_address", "12 Park Street, Kolkata", valid_from="2025-09-01")
    g.remember("acct", "shipping_address", "4 MG Road, Gurugram", valid_from="2026-06-20")
    return g


def test_flatten():
    assert flatten({"a": {"b": "x"}, "c": [1, "y"], "d": True, "e": None}) == \
        [("a.b", "x"), ("c[0]", "1"), ("c[1]", "y")]


def test_keyed_argument_blocks_and_is_fixed(guard):
    r = guard.check_tool_call("find_gyms", {"city": "Bengaluru", "open_late": True}, ["maya"])
    assert r.decision is Decision.BLOCK
    assert r.suggested_arguments == {"city": "Pune", "open_late": True}
    assert r.evidence[0].argument == "city" and r.evidence[0].changed_at == "2026-04-10"


def test_arg_map_for_unrelated_names(guard):
    r = guard.check_tool_call("book_flight", {"origin": "Bengaluru", "dest": "Delhi"},
                              ["maya"], arg_map={"origin": "location"})
    assert r.decision is Decision.BLOCK and r.suggested_arguments["origin"] == "Pune"


def test_unkeyed_distinctive_value_still_caught(guard):
    r = guard.check_tool_call("book_flight", {"from_airport": "Bengaluru"}, ["maya"])
    assert r.decision is Decision.BLOCK


def test_nested_and_free_text_arguments(guard):
    call = {"items": [{"sku": "kettle"}], "delivery": {"address": "12 Park Street, Kolkata"},
            "note": "leave at Park Street gate"}
    r = guard.check_tool_call("place_order", call, ["acct"])
    assert r.decision is Decision.BLOCK
    assert r.suggested_arguments["delivery"]["address"] == "4 MG Road, Gurugram"


def test_free_text_mention_rewritten(guard):
    r = guard.check_tool_call("order_food", {"instructions": "a vegetarian thali, no onion",
                                             "diet": "vegetarian"}, ["maya"])
    assert r.decision is Decision.BLOCK
    assert r.suggested_arguments["diet"] == "vegan"
    assert r.suggested_arguments["instructions"] == "a vegan thali, no onion"


def test_implication_in_arguments(guard):
    r = guard.check_tool_call("order_food", {"dish": "paneer butter masala"}, ["maya"])
    assert r.decision is not Decision.PASS
    assert any(e.kind is EvidenceKind.IMPLIED_CONFLICT for e in r.evidence)


@pytest.mark.parametrize("call,ids", [
    (("find_gyms", {"city": "Pune"}), ["maya"]),
    (("upgrade", {"plan": "Premium"}), ["acct"]),
    (("send_email", {"subject": "Hello", "body": "See you soon"}), ["acct"]),
    (("find_gyms", {"city": "Bengaluru"}), ["acct"]),       # other entity's history
])
def test_pass_cases(guard, call, ids):
    assert guard.check_tool_call(call[0], call[1], ids).decision is Decision.PASS


def test_generic_unkeyed_value_is_review(guard):
    g = StateGuard()
    g.remember("acct", "billing_cycle", "monthly", valid_from="2026-01-01")
    g.remember("acct", "billing_cycle", "annual", valid_from="2026-08-01")
    r = g.check_tool_call("export_report", {"granularity": "monthly"}, ["acct"])
    assert r.decision is Decision.REVIEW
    r = g.check_tool_call("charge", {"billing_cycle": "monthly"}, ["acct"])
    assert r.decision is Decision.BLOCK
