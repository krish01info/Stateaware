"""Tests for the GuardedChat wrapper and clarification/memory-note text."""
import json

from stateguard import Decision, LLMFactExtractor, StateGuard
from stateguard.chat import GuardedChat, clarification, memory_note


class FakeLLM:
    def __init__(self):
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        return f"answer #{len(self.calls)}"


def make_guard(**kw):
    g = StateGuard(**kw)
    g.remember("me", "diet", "vegetarian", valid_from="2026-01-01")
    g.remember("me", "diet", "vegan", valid_from="2026-07-02")
    return g


def test_pass_goes_straight_to_model():
    llm = FakeLLM()
    t = GuardedChat(make_guard(), llm, "me").send("Find vegan places nearby")
    assert t.text == "answer #1" and not t.asked
    assert all(m["role"] != "system" for m in llm.calls[0])


def test_block_adds_memory_note_without_rewriting_user_text():
    llm = FakeLLM()
    t = GuardedChat(make_guard(), llm, "me").send("Book me a vegetarian dinner")
    assert t.check.decision is Decision.BLOCK and not t.asked
    msgs = llm.calls[0]
    assert msgs[-1] == {"role": "user", "content": "Book me a vegetarian dinner"}  # untouched
    note = msgs[-2]["content"]
    assert msgs[-2]["role"] == "system" and "your diet is now 'vegan' (since 2026-07-02)" in note


def test_review_asks_once_then_answers():
    llm = FakeLLM()
    chat = GuardedChat(make_guard(), llm, "me")
    t1 = chat.send("Is a vegetarian thali still okay for me?")
    assert t1.asked and "vegan" in t1.text and llm.calls == []
    t2 = chat.send("yes, stick with vegan")
    assert not t2.asked and t2.text == "answer #1"


def test_answer_to_question_updates_memory():
    def fn(system, user):
        msg = user.split("MESSAGE:")[1]
        facts = [{"attribute": "diet", "value": "vegetarian", "confidence": 0.9}] \
            if "vegetarian again" in msg else []
        return json.dumps({"facts": facts}), 10

    g = make_guard(fact_extractor=LLMFactExtractor(complete_fn=fn))
    llm = FakeLLM()
    chat = GuardedChat(g, llm, "me", on_block="ask")
    assert chat.send("Book me a vegetarian dinner").asked
    t = chat.send("no, I'm actually vegetarian again")
    assert [f.value for f in t.learned] == ["vegetarian"]
    assert t.check.decision is Decision.PASS              # memory now agrees with the request
    assert all(m["role"] != "system" for m in llm.calls[0])


def test_clarification_wording_for_other_entities():
    g = StateGuard()
    g.remember("acct_9", "plan", "Basic", valid_from="2026-01-01")
    g.remember("acct_9", "plan", "Premium", valid_from="2026-06-15")
    r = g.check("Is the Basic plan still active?", ["acct_9"])
    q = clarification(r)
    assert q.startswith("Quick check: you mentioned Basic") and "acct_9's plan" in q
    assert memory_note(r) and "acct_9's plan is now 'Premium'" in memory_note(r)
