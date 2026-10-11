"""
Learning facts from what the user says.

A real agent does not get clean ``(attribute, value)`` pairs. It gets
messages: "I finally went vegan last month", "we moved the office to
Gurugram", "I'm on the Premium plan now". :meth:`StateGuard.observe` turns
such a message into fact updates, so the versioned store (and therefore the
interceptor) stays current without hand-written parsing.

Design choices that matter in practice
--------------------------------------
* **Stable attribute names.** The extractor is shown the attributes already
  on record for this entity and told to reuse them. Otherwise "diet" today
  and "dietary_preference" tomorrow would never supersede each other, and
  no change would ever be detected.
* **Short canonical values.** "vegan", not "I have been fully vegan since
  June". Short values are what the query-time matcher can find again.
* **Only current, first-hand facts.** Questions, hypotheticals, other
  people's facts and things that ended ("I used to ...") are not recorded
  as current state.
* **A confidence floor.** Low-confidence extractions are returned but not
  written.
* **Caching.** Keyed by message + known attributes, so a corpus can be
  re-processed offline and reproducibly.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class ExtractedFact:
    attribute: str
    value: str
    confidence: float
    evidence: Optional[str] = None


@dataclass
class ExtractionResult:
    facts: list[ExtractedFact] = field(default_factory=list)
    tokens_used: int = 0
    error: Optional[str] = None


@runtime_checkable
class FactExtractor(Protocol):
    def extract(self, message: str, known_attributes: list[str],
                subject: str = "the user") -> ExtractionResult: ...


def attribute_name(raw: str) -> str:
    """Canonical attribute key: lower snake_case, ascii-ish, no punctuation."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", raw.strip().lower()).strip("_")
    return re.sub(r"_+", "_", s)


SYSTEM_PROMPT = textwrap.dedent("""
    Extract durable facts that a message states about its subject, so an
    assistant's memory can be kept up to date.

    Record a fact only if the message states it as CURRENTLY true about the
    subject. Skip questions, hypotheticals, plans, jokes, other people's
    facts, and states the message says have ended. For "I used to be X, now
    I'm Y", record only Y.

    Attribute names: snake_case, general (diet, location, job, plan, role,
    relationship_status, pet, ...). If one of the KNOWN ATTRIBUTES covers the
    same thing, you MUST reuse that exact name.

    Values: the shortest canonical form, 1-5 words, as a person would type it
    in a later request ("vegan", "Pune", "product manager", "Premium").

    Reply with one JSON object and nothing else:
    {"facts": [{"attribute": "...", "value": "...", "confidence": <0..1>,
                "evidence": "<few words from the message>"}]}
    Return {"facts": []} if nothing qualifies.
""").strip()


def parse(raw: str) -> ExtractionResult:
    m = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not m:
        return ExtractionResult(error=f"no JSON object in response: {(raw or '')[:120]!r}")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return ExtractionResult(error=f"invalid JSON: {e}")
    facts = obj.get("facts") if isinstance(obj, dict) else None
    if not isinstance(facts, list):
        return ExtractionResult(error="'facts' must be a list")
    out = []
    for f in facts:
        if not isinstance(f, dict):
            return ExtractionResult(error="each fact must be an object")
        a, v, c = f.get("attribute"), f.get("value"), f.get("confidence")
        if not (isinstance(a, str) and isinstance(v, str) and a.strip() and v.strip()):
            return ExtractionResult(error="fact needs string 'attribute' and 'value'")
        if isinstance(c, bool) or not isinstance(c, (int, float)) or not 0 <= c <= 1:
            return ExtractionResult(error="'confidence' must be a number in [0,1]")
        if len(v.split()) > 8:
            continue  # a sentence, not a value: would never match a later request
        ev = f.get("evidence")
        out.append(ExtractedFact(attribute_name(a), v.strip(), float(c),
                                 ev if isinstance(ev, str) else None))
    return ExtractionResult(out)


class LLMFactExtractor:
    """Fact extraction with any chat model (named provider or ``complete_fn``)."""

    def __init__(self, provider: Optional[str] = None, *, model: Optional[str] = None,
                 complete_fn: Optional[Callable[[str, str], "tuple[str, int]"]] = None,
                 cache_path: "Optional[str | Path]" = None) -> None:
        if complete_fn is None:
            from .verifier import _PROVIDERS

            provider = provider or os.environ.get("STATEGUARD_LLM", "groq")
            if provider not in _PROVIDERS:
                raise ValueError(f"Unknown LLM provider {provider!r}")
            fn, default_model = _PROVIDERS[provider]
            complete_fn = lambda s, u: fn(s, u, model or default_model)  # noqa: E731
            self.name = f"{provider}:{model or default_model}"
        else:
            self.name = provider or "custom"
        self._complete = complete_fn
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict = (json.loads(self.cache_path.read_text())
                             if self.cache_path and self.cache_path.exists() else {})
        self.hits = self.misses = 0

    def extract(self, message: str, known_attributes: list[str],
                subject: str = "the user") -> ExtractionResult:
        known = sorted(set(known_attributes))
        key = json.dumps([message, known, subject])
        if key in self._cache:
            self.hits += 1
            return ExtractionResult([ExtractedFact(*f) for f in self._cache[key]])
        self.misses += 1
        user = (f"SUBJECT: {subject}\n"
                f"KNOWN ATTRIBUTES: {', '.join(known) or 'none yet'}\n"
                f"MESSAGE: {message}")
        try:
            raw, tokens = self._complete(SYSTEM_PROMPT, user)
        except Exception as e:
            return ExtractionResult(error=f"{type(e).__name__}: {e}")
        res = parse(raw)
        res.tokens_used = tokens or 0
        if not res.error and self.cache_path is not None:
            self._cache[key] = [[f.attribute, f.value, f.confidence, f.evidence]
                                for f in res.facts]
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache, indent=1, ensure_ascii=False))
        return res
