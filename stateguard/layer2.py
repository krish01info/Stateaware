"""
StateGuard – Person B, Layer 2
Cheap-LLM semantic check for queries flagged CONFLICT or UNCERTAIN by Layer 1.

Key innovation: the LLM receives a tiny *structured* supersession record from
the DB (e.g. "old=vegetarian → new=vegan, reason=user stated change Aug 14"),
NOT a raw 150 K-token context haystack.  This is why a small cheap model can
outperform frontier models that must search giant contexts.

Supported backends (set STATEGUARD_LLM env var or pass provider= to check_l2):
  - "gemini"   : Google Gemini (default)
  - "openai"   : OpenAI GPT-4o-mini
  - "groq"     : Groq (llama-3.3-70b-versatile, fastest + cheapest for eval)
  - "mock"     : deterministic stub for unit tests (no API key needed)

Public API
----------
check_l2(conn, query, layer1_result, *, provider="gemini") -> InterceptResult
"""

import json
import os
import re
import sqlite3
import textwrap
from typing import Optional

from .db      import get_supersession_context, get_current_fact
from .layer1  import (
    InterceptResult, Status, Conflict,
    extract_entity_names, extract_attribute,
)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent("""
    You are a stale-premise detector for a conversational AI memory system.

    TASK: Determine whether the user's query assumes an OUTDATED fact about the
    entity. A query is "stale" if it relies on a previous state that has since
    been superseded by a newer fact.

    DECISION RULES:
    1. If the query mentions, assumes, or is only relevant to the OLD value
       (previous state) and NOT the CURRENT value -> is_stale = true
    2. If the query is neutral, generic, or consistent with the CURRENT value
       -> is_stale = false
    3. When in doubt and the query seems tied to the old context, lean toward
       is_stale = true (it is better to flag a potential stale query than miss one)

    EXAMPLES:
    - Old: lived in Seattle, Current: lives in Austin
      Query: "What are the best hiking trails near Seattle?"
      -> is_stale: true (query assumes Seattle location)
    - Old: vegetarian, Current: vegan
      Query: "Can you suggest vegetarian restaurants?"
      -> is_stale: true (query assumes vegetarian, but user is now vegan)
    - Old: works from home, Current: works from office
      Query: "What's a good desk setup for a home office?"
      -> is_stale: true (query assumes WFH)

    Reply with ONLY valid JSON, no markdown fences:
    {
      "is_stale": true | false,
      "reason": "<one sentence explaining why>",
      "corrected_query": "<rewritten query using current fact, or null if not stale>"
    }
""").strip()


def _build_user_message(
    query: str,
    entity: str,
    attribute: str,
    current_value: str,
    supersession: Optional[dict],
) -> str:
    """Compose the grounding context + query for the LLM."""
    lines = [
        f"ENTITY: {entity}",
        f"ATTRIBUTE: {attribute}",
        f"CURRENT VALUE: {current_value}",
    ]
    if supersession:
        old_val = supersession.get('old_value', 'unknown')
        reason  = supersession.get('why_it_changed', 'no reason logged')
        lines += [
            f"PREVIOUS VALUE: {old_val}",
            f"CHANGE REASON: {reason}",
        ]
    lines += [
        "",
        f"USER QUERY: {query}",
        "",
        "Does this query assume the PREVIOUS (outdated) value rather than",
        "the CURRENT value? If the query is connected to, assumes, or would",
        "only make sense given the old state, it is stale.",
        "",
        "Reply with JSON.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

def _call_gemini(system: str, user: str) -> tuple[str, int]:
    """Call Google Gemini. Requires GOOGLE_API_KEY env var."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError(
            "Install the Gemini SDK:  pip install google-generativeai"
        )
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("Set GOOGLE_API_KEY environment variable.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash-lite",
        system_instruction=system,
    )
    resp = model.generate_content(user)
    tokens = getattr(resp, 'usage_metadata', None)
    tok_count = tokens.total_token_count if tokens else len(user.split())
    return resp.text, tok_count


def _call_openai(system: str, user: str) -> tuple[str, int]:
    """Call OpenAI GPT-4o-mini. Requires OPENAI_API_KEY env var."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install the OpenAI SDK:  pip install openai")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENAI_API_KEY environment variable.")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0,
        max_tokens=256,
    )
    tok = resp.usage.total_tokens if resp.usage else len(user.split())
    return resp.choices[0].message.content, tok


def _call_groq(system: str, user: str) -> tuple[str, int]:
    """Call Groq (OpenAI-compatible). Requires GROQ_API_KEY env var."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install the OpenAI SDK:  pip install openai")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("Set GROQ_API_KEY environment variable.")
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    resp = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0,
        max_tokens=256,
    )
    tok = resp.usage.total_tokens if resp.usage else len(user.split())
    return resp.choices[0].message.content, tok


_MOCK_WARNED = False

def _call_mock(system: str, user: str) -> str:
    """
    Deterministic stub for UNIT TESTS ONLY — no API key required.

    ⚠️  This backend only recognizes one hardcoded scenario (vegetarian→vegan).
    It will report is_stale=False for every other attribute/value pair,
    regardless of whether the query is actually stale. Do NOT use --provider
    mock to evaluate real accuracy on the STALE benchmark or any dataset
    beyond diet-only smoke tests — results will look artificially poor (or
    meaningless) on every other dimension since this stub can never say
    'true' for them.
    """
    global _MOCK_WARNED
    if not _MOCK_WARNED:
        import sys
        print(
            "[stateguard.layer2] WARNING: using 'mock' LLM backend — this "
            "stub only detects the vegetarian/vegan scenario and will "
            "under-report is_stale=true for everything else. Use "
            "--provider groq/gemini/openai for real accuracy numbers.",
            file=sys.stderr,
        )
        _MOCK_WARNED = True
    is_stale = "vegetarian" in user.lower() and "vegan" in user.lower()
    corrected = None
    if is_stale:
        # Extract the query line to rewrite it
        match = re.search(r"USER QUERY: (.+)", user, re.IGNORECASE)
        if match:
            corrected = match.group(1).replace("vegetarian", "vegan")
    return json.dumps({
        "is_stale": is_stale,
        "reason": "Mock: found conflicting diet keyword.",
        "corrected_query": corrected,
    }), len(user.split())


_BACKENDS = {
    "gemini": _call_gemini,
    "openai": _call_openai,
    "groq":   _call_groq,
    "mock":   _call_mock,
}


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_response(raw: str) -> dict:
    """
    Parse the LLM JSON response.
    Falls back gracefully if the model returns extra whitespace or markdown.

    On failure, returns a fail-safe default (is_stale=False) AND sets
    "_parse_error": True so callers can distinguish "the model genuinely
    said not-stale" from "we couldn't parse the model's output and are
    guessing". Previously these were indistinguishable, which meant
    malformed LLM output silently inflated the "not stale" / false-negative
    count with no visibility.
    """
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        parsed = json.loads(cleaned)
        parsed.setdefault("_parse_error", False)
        return parsed
    except json.JSONDecodeError:
        import sys
        print(
            f"[stateguard.layer2] WARNING: failed to parse LLM response as "
            f"JSON, defaulting to is_stale=False. Raw response (truncated): "
            f"{raw[:200]!r}",
            file=sys.stderr,
        )
        return {
            "is_stale": False,
            "reason": "parse error",
            "corrected_query": None,
            "_parse_error": True,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_l2(
    conn: sqlite3.Connection,
    query: str,
    layer1_result: InterceptResult,
    *,
    provider: Optional[str] = None,
) -> InterceptResult:
    """
    Run Layer-2 LLM check on a query that Layer 1 flagged as CONFLICT or UNCERTAIN.

    Parameters
    ----------
    conn          : open DB connection
    query         : original user query
    layer1_result : InterceptResult from layer1.check()
    provider      : "gemini" | "openai" | "mock"
                    Defaults to STATEGUARD_LLM env var, then "gemini".

    Returns
    -------
    Updated InterceptResult:
      - corrected_query filled in if LLM confirms stale premise
      - status updated to CONFLICT (confirmed stale) or PASS (false alarm)
    """
    if layer1_result.status not in (Status.CONFLICT, Status.UNCERTAIN):
        # Nothing to do — return as-is
        return layer1_result

    provider = provider or os.environ.get("STATEGUARD_LLM", "gemini")
    if provider not in _BACKENDS:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. "
            f"Choose from: {', '.join(sorted(_BACKENDS))}"
        )
    backend = _BACKENDS[provider]

    # Resolve entity/attribute for UNCERTAIN (Layer 1 flagged but no Conflict obj)
    if layer1_result.conflicts:
        c = layer1_result.conflicts[0]
        entity, attribute = c.entity, c.attribute
        current_value = c.current_value
    else:
        # UNCERTAIN: re-derive entity and attribute from query
        entities  = extract_entity_names(query)
        attribute = extract_attribute(query)
        entity    = entities[0] if entities else "unknown"
        fact      = get_current_fact(conn, entity, attribute) if attribute else None
        current_value = fact["value"] if fact else "unknown"

    supersession = get_supersession_context(conn, entity, attribute) if attribute else None

    user_msg = _build_user_message(
        query=query,
        entity=entity,
        attribute=attribute or "unknown",
        current_value=current_value,
        supersession=supersession,
    )

    raw_response, tokens_used = backend(_SYSTEM_PROMPT, user_msg)
    parsed = _parse_response(raw_response)

    result = InterceptResult(
        status=layer1_result.status,
        conflicts=layer1_result.conflicts,
        raw_query=query,
    )

    if parsed.get("is_stale"):
        result.status          = Status.CONFLICT
        result.corrected_query = parsed.get("corrected_query")
    else:
        result.status          = Status.PASS
        result.corrected_query = None

    result.tokens_used = tokens_used
    result.parse_error = parsed.get("_parse_error", False)
    return result
