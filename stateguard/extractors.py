"""
Pluggable premise extraction.

Two independent questions are asked of a query:

1. *Which stored attributes does it touch?*  Answered by
   :class:`AttributeExtractor` implementations. The default
   :class:`KeywordAttributeExtractor` combines (a) the attribute's own name
   ("shipping_address" -> "shipping address") with (b) optional domain packs
   of surface cues. No attribute vocabulary is required: any attribute in the
   store is detectable by name, and stored *values* are matched separately in
   :mod:`stateguard.normalize`.

2. *How does it frame the state it mentions?*  Answered by
   :func:`premise_frame`: does the query acknowledge a change ("used to",
   "no longer"), contrast two states ("instead of", "compared to"), look
   forward ("next month", "going to"), or simply ask what the state is?
   These frames are what separate an assumed stale premise from a query that
   merely talks about history.
"""

from __future__ import annotations

import re

from dataclasses import dataclass, field
from typing import Iterable, Optional, Protocol, runtime_checkable

from .normalize import phrase_present, tokenize


@runtime_checkable
class AttributeExtractor(Protocol):
    def cues(self, query: str, attributes: Iterable[str]) -> dict[str, tuple[int, int]]:
        """Return ``{attribute: span}`` for attributes the query touches.

        *attributes* is the set of attributes that actually exist for the
        resolved entities; extractors should only report those.
        """
        ...


# ---------------------------------------------------------------------------
# Domain packs: surface phrase -> canonical attribute
# ---------------------------------------------------------------------------

PERSON_PROFILE: dict[str, str] = {
    # diet
    "diet": "diet", "dietary": "diet", "eat": "diet", "eating": "diet", "food": "diet",
    "meal": "diet", "recipe": "diet", "restaurant": "diet", "cuisine": "diet",
    "dinner": "diet", "lunch": "diet", "breakfast": "diet", "snack": "diet",
    "menu": "diet", "dish": "diet", "cook": "diet", "order food": "diet",
    # location
    "location": "location", "city": "location", "live in": "location",
    "lives in": "location", "living in": "location", "based in": "location",
    "neighborhood": "location", "neighbourhood": "location", "commute": "location",
    "nearby": "location", "near me": "location", "local": "location",
    "around here": "location", "my area": "location",
    # job
    "job": "job", "work as": "job", "works as": "job", "occupation": "job",
    "role": "job", "profession": "job", "career": "job", "office": "job",
    "employer": "job", "workplace": "job", "coworker": "job", "colleague": "job",
    "boss": "job", "company": "job",
    # misc person attributes
    "age": "age", "years old": "age", "birthday": "age",
    "language": "language", "speak": "language",
    "relationship": "relationship_status", "married": "relationship_status",
    "partner": "relationship_status", "spouse": "relationship_status",
    "girlfriend": "relationship_status", "boyfriend": "relationship_status",
    "hobby": "hobby", "hobbies": "hobby", "free time": "hobby", "weekend": "hobby",
    "pet": "pet", "dog": "pet", "cat": "pet",
    "school": "education", "university": "education", "college": "education",
    "degree": "education", "major": "education", "course": "education",
    "gym": "health", "workout": "health", "exercise": "health", "fitness": "health",
    "schedule": "schedule", "routine": "schedule",
}

ACCOUNT_STATE: dict[str, str] = {
    "plan": "plan", "tier": "plan", "subscription": "plan", "upgrade": "plan",
    "downgrade": "plan", "renew": "plan", "renewal": "plan",
    "billing": "billing_cycle", "billed": "billing_cycle", "invoice": "billing_cycle",
    "bill": "billing_cycle",
    "price": "price", "cost": "price", "charge": "price", "fee": "price",
    "quote": "price", "paid": "price", "pay": "price",
    "address": "shipping_address", "ship to": "shipping_address",
    "deliver to": "shipping_address", "delivery": "shipping_address",
    "permission": "role", "access": "role", "admin": "role",
    "status": "status", "online": "status", "offline": "status", "active": "status",
    "appointment": "appointment", "booking": "appointment", "slot": "appointment",
    "policy": "policy_version", "terms": "policy_version",
    "email": "email", "phone": "phone", "payment method": "payment_method",
    "card": "payment_method",
}


class KeywordAttributeExtractor:
    """Attribute detection from attribute names plus optional cue packs."""

    def __init__(self, *packs: dict[str, str], use_attribute_names: bool = True) -> None:
        self.cue_map: dict[str, str] = {}
        for p in packs:
            self.cue_map.update(p)
        self.use_attribute_names = use_attribute_names

    def add(self, phrase: str, attribute: str) -> None:
        self.cue_map[phrase] = attribute

    def cues(self, query: str, attributes: Iterable[str]) -> dict[str, tuple[int, int]]:
        attrs = set(attributes)
        out: dict[str, tuple[int, int]] = {}
        phrases: list[tuple[str, str]] = []
        if self.use_attribute_names:
            for a in attrs:
                phrases.append((a.replace("_", " "), a))
                # "location(city)" style keys from benchmark adapters
                for part in a.replace("(", " ").replace(")", " ").split():
                    if len(part) > 3:
                        phrases.append((part.replace("_", " "), a))
        phrases += [(p, a) for p, a in self.cue_map.items() if a in attrs]
        for phrase, attr in sorted(phrases, key=lambda x: len(x[0]), reverse=True):
            if attr in out:
                continue
            span = phrase_present(query, phrase)
            if span:
                out[attr] = span
        return out


# ---------------------------------------------------------------------------
# Premise framing
# ---------------------------------------------------------------------------

# Query explicitly says the value is past/changed -> it is not *assuming* it.
_HISTORY_STRONG = ["used to", "previously", "formerly", "no longer", "not anymore",
                   "anymore", "back when", "in the past", "switched from", "moved from",
                   "changed from", "went from", "gave up", "stopped being", "quit"]
# Weaker signals of talking about the past (plain past tense, "my old ...").
_HISTORY_WEAK = ["my old", "my previous", "my former", "old", "before", "lived", "was",
                 "were", "had been", "worked as", "used", "earlier"]
# Query contrasts states -> talks about both deliberately.
_CONTRAST = ["instead of", "rather than", "compared to", "compared with", "versus", "vs",
             "difference between", "switch from", "switching from", "move from",
             "moving from", "transition from", "go from", "now that", "changed between",
             "what changed"]
# Query looks forward -> e.g. revisiting an old city is not a stale premise.
_FUTURE = ["going to", "next week", "next month", "next year", "plan to", "planning to",
           "visiting", "trip to", "back to", "return to", "returning to", "flying to"]
# Query asks what the state is (retrieval), rather than presupposing it.
_DIRECT_Q = re.compile(
    r"^\s*(what(?:'s|\s+is|\s+are|\s+was)|where\s+(?:do|does|is|am)|who\s+(?:is|are)|"
    r"remind\s+me|do\s+you\s+(?:remember|know)|what\s+do\s+you\s+know|tell\s+me\s+about)\b"
    r"|\b(currently|right\s+now|these\s+days|at\s+the\s+moment|any\s*more|still|yet)\b",
    re.IGNORECASE)
# Negation directly attached to the stale mention ("not vegetarian").
_NEGATION = ["not", "no", "never", "isn t", "isnt", "aren t", "arent", "don t", "dont",
             "doesn t", "doesnt", "non"]


@dataclass
class PremiseFrame:
    history: bool = False          # strong: "used to", "no longer"
    past: bool = False             # weak: plain past tense, "my old"
    contrast: bool = False
    future: bool = False
    direct_question: bool = False
    negated_spans: list[tuple[int, int]] = field(default_factory=list)


def _any(query: str, phrases: list[str]) -> bool:
    return any(phrase_present(query, p) for p in phrases)


def premise_frame(query: str) -> PremiseFrame:
    f = PremiseFrame(
        history=_any(query, _HISTORY_STRONG),
        past=_any(query, _HISTORY_WEAK),
        contrast=_any(query, _CONTRAST),
        future=_any(query, _FUTURE),
        direct_question=bool(_DIRECT_Q.search(query)),
    )
    toks = tokenize(query)
    neg = {n for n in _NEGATION if " " not in n}
    for i, t in enumerate(toks[:-1]):
        if t.norm in neg:
            f.negated_spans.append((toks[i + 1].start, toks[min(i + 3, len(toks) - 1)].end))
    return f


def is_negated(frame: PremiseFrame, start: int) -> bool:
    return any(a <= start <= b for a, b in frame.negated_spans)
