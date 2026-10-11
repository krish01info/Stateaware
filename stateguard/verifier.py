"""
Layer 2 as a *structured verifier*, not the default decision maker.

The verifier is only consulted for evidence the deterministic layer finds
ambiguous. It receives a compact, structured **version chain** for one
entity/attribute -- every value it has held, when, and why it changed --
rather than raw conversation history. That record is typically 100-300
tokens, which is what makes a small, cheap model adequate for the job.

Failure is never silent: a provider error, a timeout, malformed JSON or a
schema violation returns ``VerifierResult(error=...)``, which the engine
turns into ``REVIEW`` with ``provider_error`` evidence. It is never counted
as a confirmed non-stale query.

Any model can be plugged in through ``complete_fn(system, user) -> (text, tokens)``,
including a local model, so no data has to leave the host if that matters.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, runtime_checkable

from .models import Evidence, FactVersion

CompleteFn = Callable[[str, str], "tuple[str, int]"]


@dataclass
class VerifierResult:
    is_stale: Optional[bool]
    confidence: float = 0.0
    reason: Optional[str] = None
    assumed_value: Optional[str] = None
    corrected_query: Optional[str] = None
    tokens_used: int = 0
    error: Optional[str] = None


@runtime_checkable
class Verifier(Protocol):
    def verify(self, query: str, entity_id: str, attribute: str,
               history: list[FactVersion], evidence: list[Evidence]) -> VerifierResult: ...


SYSTEM_PROMPT = textwrap.dedent("""
    You check whether a user's request silently relies on an OUTDATED fact.

    You get the version history of ONE attribute of ONE entity, oldest to
    newest. The last version is the current truth. Decide whether the request,
    as written, presupposes or only makes sense under an earlier version.

    - Mentioning an old value is NOT enough. Asking about history, comparing
      old and new, planning a visit to an old place, or asking what the state
      is right now are not stale.
    - A request can be stale without naming the old value, if what it asks
      for only fits the old state.
    - If the request is unrelated to this attribute, it is not stale.

    Reply with one JSON object and nothing else:
    {"assumes_outdated_state": true|false,
     "confidence": <number 0..1>,
     "assumed_value": <the outdated value relied on, or null>,
     "reason": <one short sentence>,
     "corrected_query": <the request rewritten for the current value, or null>}
""").strip()


def build_context(query: str, entity_id: str, attribute: str,
                  history: list[FactVersion], evidence: list[Evidence]) -> str:
    chain = sorted(history, key=lambda f: f.valid_from)
    lines = [f"ENTITY: {entity_id}", f"ATTRIBUTE: {attribute}", "VERSIONS (oldest -> newest):"]
    for i, f in enumerate(chain, 1):
        tag = "CURRENT" if f.is_current else "superseded"
        why = f"; changed because: {f.reason}" if f.reason else ""
        lines.append(f"  {i}. {f.value!r} [{tag}] from {f.valid_from}{why}")
    hints = [e for e in evidence if e.assumed_value]
    if hints:
        lines.append("PHRASES IN THE REQUEST FLAGGED BY THE DETERMINISTIC LAYER: "
                     + ", ".join(sorted({repr(e.assumed_value) for e in hints})))
    lines += ["", f"REQUEST: {query}"]
    return "\n".join(lines)


_ALLOWED_KEYS = {"assumes_outdated_state", "confidence", "assumed_value", "reason",
                 "corrected_query"}


def parse_result(raw: str) -> VerifierResult:
    """Strictly validate the model's JSON. Any deviation is an error, not a guess."""
    text = re.sub(r"^\s*```(?:json)?|```\s*$", "", raw.strip()).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return VerifierResult(None, error=f"no JSON object in response: {raw[:120]!r}")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return VerifierResult(None, error=f"invalid JSON: {e}")
    if not isinstance(obj, dict):
        return VerifierResult(None, error="JSON is not an object")
    stale = obj.get("assumes_outdated_state", obj.get("is_stale"))
    if not isinstance(stale, bool):
        return VerifierResult(None, error="'assumes_outdated_state' missing or not boolean")
    conf = obj.get("confidence", None)
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
        return VerifierResult(None, error="'confidence' missing or not a number in [0,1]")

    def _s(k):
        v = obj.get(k)
        return v if isinstance(v, str) and v.strip() else None

    return VerifierResult(stale, float(conf), _s("reason"), _s("assumed_value"),
                          _s("corrected_query") if stale else None)


class LLMVerifier:
    """Verifier backed by any chat model.

    Use a named provider (``"groq"``, ``"openai"``, ``"gemini"``) or pass
    ``complete_fn`` for anything else (local models, other APIs, test stubs).
    """

    def __init__(self, provider: Optional[str] = None, *, model: Optional[str] = None,
                 complete_fn: Optional[CompleteFn] = None) -> None:
        if complete_fn is None:
            provider = provider or os.environ.get("STATEGUARD_LLM", "groq")
            if provider not in _PROVIDERS:
                raise ValueError(f"Unknown LLM provider {provider!r}; "
                                 f"choose from {sorted(_PROVIDERS)} or pass complete_fn")
            name, default_model = provider, _PROVIDERS[provider][1]
            fn = _PROVIDERS[provider][0]
            complete_fn = lambda s, u: fn(s, u, model or default_model)  # noqa: E731
            self.name = f"{name}:{model or default_model}"
        else:
            self.name = provider or "custom"
        self._complete = complete_fn

    def verify(self, query, entity_id, attribute, history, evidence) -> VerifierResult:
        user = build_context(query, entity_id, attribute, history, evidence)
        try:
            raw, tokens = self._complete(SYSTEM_PROMPT, user)
        except Exception as e:  # provider down, auth, timeout, rate limit...
            return VerifierResult(None, error=f"{type(e).__name__}: {e}")
        res = parse_result(raw or "")
        res.tokens_used = tokens or 0
        return res


# ---------------------------------------------------------------------------
# Providers (JSON mode enforced where the API supports it)
# ---------------------------------------------------------------------------

def _openai_compatible(system: str, user: str, model: str, *, key_env: str,
                       base_url: Optional[str] = None) -> tuple[str, int]:
    from openai import OpenAI

    key = os.environ.get(key_env)
    if not key:
        raise EnvironmentError(f"Set {key_env}")
    client = OpenAI(api_key=key, base_url=base_url) if base_url else OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model=model, temperature=0, max_tokens=300,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    tok = resp.usage.total_tokens if getattr(resp, "usage", None) else 0
    return resp.choices[0].message.content or "", tok


def _openai(system, user, model):
    return _openai_compatible(system, user, model, key_env="OPENAI_API_KEY")


def _groq(system, user, model):
    return _openai_compatible(system, user, model, key_env="GROQ_API_KEY",
                              base_url="https://api.groq.com/openai/v1")


def _gemini(system, user, model):
    import google.generativeai as genai

    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise EnvironmentError("Set GOOGLE_API_KEY")
    genai.configure(api_key=key)
    m = genai.GenerativeModel(model_name=model, system_instruction=system,
                              generation_config={"response_mime_type": "application/json",
                                                 "temperature": 0})
    resp = m.generate_content(user)
    usage = getattr(resp, "usage_metadata", None)
    return resp.text, (usage.total_token_count if usage else 0)


_PROVIDERS: dict[str, tuple[Callable, str]] = {
    "openai": (_openai, "gpt-4o-mini"),
    "groq": (_groq, "openai/gpt-oss-120b"),
    "gemini": (_gemini, "gemini-2.5-flash-lite"),
}
