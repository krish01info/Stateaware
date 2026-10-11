"""
Drop-in chat wrapper: what a real assistant does with PASS / REVIEW / BLOCK.

Two principles:

* **Never silently rewrite what the user typed.** On BLOCK the request goes
  to the model unchanged, together with a short *memory note* saying what
  changed and when, so the model can answer for the current state and
  tell the user why.
* **When unsure, ask the user, once.** On REVIEW the wrapper returns a
  one-line clarifying question instead of calling the model. The user is
  the authority on their own state; whatever they answer is learned through
  :meth:`StateGuard.observe` (if a fact extractor is configured) and the
  original request is then answered.

    chat = GuardedChat(guard, llm=my_model, entity_id="me")
    chat.send("Book me a vegetarian dinner for Friday")

``llm`` is any callable taking OpenAI-style ``messages`` and returning text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .models import CheckRequest, CheckResponse, Decision, Evidence, EvidenceKind

LLM = Callable[[list[dict]], str]


def _when(ts: Optional[str]) -> str:
    return f" (since {ts[:10]})" if ts else ""


def _subject(entity_id: str, self_entity: Optional[str]) -> tuple[str, str]:
    """('your', 'you') for the user, ("maya's", 'maya') for anyone else."""
    if self_entity is not None and entity_id == self_entity:
        return "your", "you"
    return f"{entity_id}'s", entity_id


def clarification(resp: CheckResponse, self_entity: Optional[str] = None) -> Optional[str]:
    """A one-line question to ask the user about the top evidence, or None."""
    top = next((e for e in resp.evidence if e.kind in (
        EvidenceKind.STALE_VALUE, EvidenceKind.IMPLIED_CONFLICT,
        EvidenceKind.ATTRIBUTE_CUE, EvidenceKind.UNRESOLVED_ENTITY)), None)
    if top is None:
        return None
    attr = (top.attribute or "").replace("_", " ")
    if top.kind is EvidenceKind.UNRESOLVED_ENTITY:
        return "Quick check: who is this for?"
    pos, _ = _subject(top.entity_id, self_entity)
    if top.kind is EvidenceKind.STALE_VALUE:
        return (f"Quick check: you mentioned {top.assumed_value}, but I have {pos} {attr} as "
                f"{top.current_value}{_when(top.changed_at)}. Should I go with "
                f"{top.current_value}?")
    if top.kind is EvidenceKind.IMPLIED_CONFLICT:
        return (f"Quick check: {top.assumed_value} doesn't usually fit {pos} {attr} "
                f"({top.current_value}). Do you still want it?")
    return (f"Quick check: {pos} {attr} changed to {top.current_value}"
            f"{_when(top.changed_at)}. Should I plan around that?")


def memory_note(resp: CheckResponse, self_entity: Optional[str] = None) -> Optional[str]:
    """System-message text telling the model what changed, for BLOCK/REVIEW."""
    lines = []
    seen = set()
    for e in resp.evidence:
        if e.kind not in (EvidenceKind.STALE_VALUE, EvidenceKind.IMPLIED_CONFLICT) \
                or e.score < 0.35 or (e.entity_id, e.attribute) in seen:
            continue
        seen.add((e.entity_id, e.attribute))
        pos, _ = _subject(e.entity_id, self_entity)
        attr = (e.attribute or "").replace("_", " ")
        what = (f"the request mentions '{e.assumed_value}', which was {pos} {attr} earlier"
                if e.kind is EvidenceKind.STALE_VALUE
                else f"the request mentions '{e.assumed_value}', which does not fit")
        lines.append(f"- {pos} {attr} is now '{e.current_value}'{_when(e.changed_at)}; {what}.")
    if not lines:
        return None
    return ("Memory note (StateGuard): the request may rely on outdated information.\n"
            + "\n".join(lines)
            + "\nAnswer for the CURRENT state. Briefly tell the user what you assumed "
              "and why, and do not repeat the outdated value as if it were true.")


@dataclass
class ChatTurn:
    text: str
    check: Optional[CheckResponse]
    asked: bool = False                    # True if text is a clarifying question
    learned: list = field(default_factory=list)


class GuardedChat:
    """Conversation wrapper that checks each user message before the model sees it.

    on_block  : "note" (default) – answer with a memory note;
                "ask"            – ask the user first, like REVIEW.
    on_review : "ask" (default)  – one clarifying question, then answer;
                "note"           – answer with a memory note;
                "pass"           – answer normally.
    learn     : run ``guard.observe`` on every user message (needs a
                fact extractor), so statements like "I'm vegan now" update
                memory before the request is checked.
    """

    def __init__(self, guard, llm: LLM, entity_id: str, *, system_prompt: Optional[str] = None,
                 on_block: str = "note", on_review: str = "ask", learn: bool = True) -> None:
        self.guard, self.llm, self.entity_id = guard, llm, entity_id
        self.on_block, self.on_review = on_block, on_review
        self.learn = learn and getattr(guard, "fact_extractor", None) is not None
        self.messages: list[dict] = ([{"role": "system", "content": system_prompt}]
                                     if system_prompt else [])
        self._pending: Optional[tuple[str, CheckResponse]] = None

    def send(self, text: str) -> ChatTurn:
        learned = self._observe(text)

        if self._pending is not None:
            # The user just answered our question: learn from the answer (done
            # above), re-check the ORIGINAL request against updated memory, and
            # answer it. Never ask twice in a row.
            original, _ = self._pending
            self._pending = None
            self.messages.append({"role": "user", "content": text})
            resp = self.guard.check(CheckRequest(original, entity_ids=[self.entity_id]))
            return self._answer(resp, learned, extra_user=None)

        resp = self.guard.check(CheckRequest(text, entity_ids=[self.entity_id]))
        action = {Decision.BLOCK: self.on_block, Decision.REVIEW: self.on_review}.get(
            resp.decision, "pass")
        if action == "ask":
            q = clarification(resp, self.entity_id)
            if q:
                self._pending = (text, resp)
                self.messages += [{"role": "user", "content": text},
                                  {"role": "assistant", "content": q}]
                return ChatTurn(q, resp, asked=True, learned=learned)
        self.messages.append({"role": "user", "content": text})
        return self._answer(resp, learned, note=(action != "pass"))

    # ---------------------------------------------------------------- helpers

    def _observe(self, text: str) -> list:
        if not self.learn:
            return []
        return self.guard.observe(self.entity_id, text).written

    def _answer(self, resp: CheckResponse, learned: list, note: bool = True,
                extra_user: Optional[str] = None) -> ChatTurn:
        msgs = list(self.messages)
        n = memory_note(resp, self.entity_id) if note and resp.decision is not Decision.PASS \
            else None
        if n:
            # The note goes just before the latest user message, as a system message.
            msgs.insert(len(msgs) - 1, {"role": "system", "content": n})
        reply = self.llm(msgs)
        self.messages.append({"role": "assistant", "content": reply})
        return ChatTurn(reply, resp, asked=False, learned=learned)
