"""Harness tests: balanced metrics, the general fixtures, and the STALE runner
on a STALE-shaped row (no dataset download needed)."""
import json
from pathlib import Path

from harness.metrics import scorecard, by_group
from harness import run_general
from harness.run_stale import run_row
from stateguard.db import init_db

FAKE_ROW = {
    "uid": "abcdef1234567890",
    "type": "Type_I",
    "M_old": "I am vegetarian and love paneer.",
    "M_new": "I recently became vegan.",
    "explanation": "Dietary_Restrictions.diet is now vegan.",
    "haystack_session": [],
    "probing_queries": {
        "dim1_query": "What diet do I follow?",
        "dim2_query": "Suggest a vegetarian dinner for me.",
        "dim3_query": "Plan my meals for the week.",
    },
}


def test_scorecard_strict_vs_lenient():
    recs = [
        {"expected": "stale", "decision": "block"},
        {"expected": "stale", "decision": "review"},
        {"expected": "stale", "decision": "pass"},
        {"expected": "neutral", "decision": "pass"},
        {"expected": "neutral", "decision": "review"},
        {"expected": "neutral", "decision": "block"},
    ]
    c = scorecard(recs)
    assert c["strict"]["recall"] == round(1 / 3, 4)
    assert c["lenient"]["recall"] == round(2 / 3, 4)
    assert c["strict"]["false_positive_rate"] == round(1 / 3, 4)
    assert c["lenient"]["false_positive_rate"] == round(2 / 3, 4)
    assert c["block_precision"] == 0.5
    assert c["review_rate"] == round(2 / 6, 4)


def test_block_everything_is_exposed():
    """A system that blocks everything must not look good."""
    recs = [{"expected": "stale", "decision": "block"}] * 5 + \
           [{"expected": "neutral", "decision": "block"}] * 5
    c = scorecard(recs)
    assert c["strict"]["recall"] == 1.0
    assert c["strict"]["false_positive_rate"] == 1.0
    assert c["neutral_pass_rate"] == 0.0


def test_by_group():
    recs = [{"expected": "stale", "decision": "block", "d": "a"},
            {"expected": "neutral", "decision": "pass", "d": "b"}]
    assert set(by_group(recs, "d")) == {"a", "b"}


def test_fixtures_are_well_formed():
    rows = run_general.load(run_general.DEFAULT_FIXTURES)
    assert len(rows) >= 50
    assert {r["expected"] for r in rows} == {"stale", "neutral"}
    assert len({r["domain"] for r in rows}) >= 8
    assert len({r["id"] for r in rows}) == len(rows)


def test_general_runner_guardrails(tmp_path):
    """Regression guard on the dev set: precision and FPR must stay sane."""
    overall = run_general.main(["--out", str(tmp_path)])
    assert overall["block_precision"] >= 0.85
    assert overall["strict"]["false_positive_rate"] <= 0.10
    assert list(tmp_path.glob("*.json"))


def test_stale_runner_guard_mode():
    out = {}
    for dim in ("dim1_query", "dim2_query", "dim3_query"):
        conn = init_db(":memory:")
        out[dim] = run_row(conn, FAKE_ROW, dim, "guard", provider="mock")
        conn.close()
    assert out["dim2_query"]["decision"] == "block"
    assert out["dim2_query"]["expected"] == "stale"
    assert "vegan" in out["dim2_query"]["corrected_query"]
    assert out["dim1_query"]["decision"] == "pass"
    assert out["dim3_query"]["decision"] == "pass"


def test_stale_runner_l1_has_decision():
    conn = init_db(":memory:")
    r = run_row(conn, FAKE_ROW, "dim2_query", "l1")
    assert r["decision"] in ("pass", "review", "block") and r["expected"] == "stale"
