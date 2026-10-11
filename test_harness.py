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
    assert len(rows) >= 70
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


def test_general_runner_with_implications(tmp_path, monkeypatch):
    """Plumbing: implications flow from provider -> cache -> engine -> scorecard."""
    import json as _json
    from stateguard.implications import LLMImplicationProvider

    def fake(system, user):
        payload = {"conflicts_with_current": ["cheese", "milkshake"], "only_fits_previous": []} \
            if "'vegan'" in user.split("CURRENT VALUE:")[1] else \
            {"conflicts_with_current": [], "only_fits_previous": []}
        return _json.dumps(payload), 100

    real_init = LLMImplicationProvider.__init__
    monkeypatch.setattr(LLMImplicationProvider, "__init__",
                        lambda self, provider=None, **kw: real_init(self, complete_fn=fake))
    cache = tmp_path / "impl.json"
    out = run_general.main(["--out", str(tmp_path), "--implications", "fake",
                            "--implications-cache", str(cache)])
    assert out["write"]["calls"] > 0 and out["write"]["tokens"] > 0
    assert cache.exists()
    rows = _json.loads(next(tmp_path.glob("*impl-fake.json")).read_text())["rows"]
    assert next(r for r in rows if r["id"] == "imp-01")["decision"] != "pass"
    # second run is served from the cache: no new write tokens
    out2 = run_general.main(["--out", str(tmp_path), "--implications", "cache-only",
                             "--implications-cache", str(cache)])
    assert out2["write"]["tokens"] == 0


def test_stale_runner_with_implications():
    from stateguard.implications import StaticImplicationProvider
    impl = StaticImplicationProvider({("diet", "vegan"): {"conflicts_with_current": ["paneer"]}})
    row = dict(FAKE_ROW, probing_queries=dict(FAKE_ROW["probing_queries"],
                                              dim3_query="Share a paneer butter masala recipe."))
    conn = init_db(":memory:")
    r = run_row(conn, row, "dim3_query", "guard", implications=impl, ipa_label="stale")
    assert r["decision"] != "pass" and r["expected"] == "stale" and r["correct"]


def test_stale_runner_seeds_from_messages():
    """Real-life path: facts are learned from M_old/M_new, not parsed with regexes."""
    import json as _json
    from stateguard.extraction import LLMFactExtractor

    def fn(system, user):
        v = "vegan" if "vegan" in user.split("MESSAGE:")[1] else "vegetarian"
        return _json.dumps({"facts": [{"attribute": "diet", "value": v,
                                       "confidence": 0.9}]}), 50

    ex = LLMFactExtractor(complete_fn=fn)
    conn = init_db(":memory:")
    r = run_row(conn, FAKE_ROW, "dim2_query", "guard", extractor=ex, use_ner=True)
    assert r["seed"] == "messages" and r["seed_ok"]
    assert r["decision"] == "block" and r["write_tokens"] == 100


def test_manifest_records_constants():
    from harness.metrics import manifest
    m = manifest({"mode": "guard"})
    assert m["constants"]["CAP_NEGATED"] == 0.15
    assert m["defaults"]["block_threshold"] == 0.75 and m["args"] == {"mode": "guard"}
