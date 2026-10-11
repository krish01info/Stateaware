"""
Write-time implications: spend one cheap LLM call when a fact CHANGES, so the
query-time check can stay deterministic.

The query-time engine can only match words it already knows: the stored
values. A request like "a paneer tikka recipe for Maya" (after Maya went
vegan) names no stored value, so Layer 1 cannot see it.

Instead of asking an LLM about every *query*, StateGuard asks once per
*fact change*: "now that diet is 'vegan' (was 'vegetarian'), which phrases
would only make sense if that change had not happened?" The answer is a
short list of surface phrases stored next to the fact version:

* ``conflicts_with_current`` – things the current value rules out
  (vegan -> "paneer", "ghee"; role=viewer -> "delete the project").
* ``only_fits_previous``     – things that only make sense under an earlier
  value (Basic -> Premium: "can't watch in 4K", "upgrade to Premium").

At query time these phrases are matched exactly like stored values, in
microseconds, with the same premise-framing caps. Facts change far less
often than they are queried, so the LLM cost moves from per-query to
per-update.

Providers
---------
LLMImplicationProvider     – any chat model (named provider or ``complete_fn``).
StaticImplicationProvider  – hand-written rules; offline, for tests or curated domains.
CachedImplicationProvider  – JSON cache around another provider so experiments
                             are reproducible and each change is paid for once.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol, runtime_checkable

from .normalize import STOPWORDS, normalize

CONFLICTS = "conflicts_with_current"
ONLY_PREVIOUS = "only_fits_previous"
KINDS = (CONFLICTS, ONLY_PREVIOUS)


@dataclass(frozen=True)
class Implication:
    phrase: str
    kind: str = CONFLICTS


@dataclass
class ImplicationSet:
    items: list[Implication] = field(default_factory=list)
    tokens_used: int = 0
    error: Optional[str] = None
    source: str = ""


@runtime_checkable
class ImplicationProvider(Protocol):
    def implications(self, entity_id: str, attribute: str, current_value: str,
                     previous_values: list[str], reason: Optional[str] = None
                     ) -> ImplicationSet: ...


# ---------------------------------------------------------------------------
# Cleaning (shared by every provider)
# ---------------------------------------------------------------------------

def clean(items: list[Implication], current_value: str, previous_values: list[str],
          max_words: int = 5, max_items: int = 30) -> list[Implication]:
    """Drop phrases that would cause false positives or duplicate stored values.

    Removed: empty / overlong phrases, phrases made only of function words,
    phrases equal to the current or a previous value (already matched
    directly), phrases contained in the current value ("vegan cheese" when
    the current value is "vegan cheese only" would fire on correct queries),
    and duplicates.
    """
    cur = normalize(current_value)
    prev = {normalize(p) for p in previous_values}
    seen: set[str] = set()
    out: list[Implication] = []
    for it in items:
        n = normalize(it.phrase)
        toks = n.split()
        if not toks or len(toks) > max_words or all(t in STOPWORDS for t in toks):
            continue
        if n == cur or n in prev or (n and f" {n} " in f" {cur} "):
            continue
        if it.kind not in KINDS or n in seen:
            continue
        seen.add(n)
        out.append(Implication(it.phrase.strip(), it.kind))
    return out[:max_items]


# ---------------------------------------------------------------------------
# LLM provider
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    A stored fact about a user, account, device or document has just changed.
    List short phrases (1-4 words) that a future request might contain and
    that would only make sense if this change had NOT happened.

    Two lists:
    - conflicts_with_current: things the CURRENT value rules out.
      e.g. diet now "vegan" -> "paneer", "ghee", "honey", "cheese omelette"
           role now "viewer" -> "delete the project", "reset passwords"
           device now "offline" -> "start it now", "run the pump"
    - only_fits_previous: things that only make sense under an EARLIER value.
      e.g. plan "Basic" -> "Premium": "can't watch in 4K", "upgrade to Premium"
           location "Seattle" -> "Austin": "Pike Place", "Seattle traffic"

    Rules:
    - Use concrete words people actually type. No generic words that also
      appear in unrelated requests ("food", "account", "help", "plan").
    - Do not include the current value or the previous values themselves.
    - At most 12 phrases per list. Fewer, precise phrases beat many vague ones.

    Reply with one JSON object and nothing else:
    {"conflicts_with_current": [...], "only_fits_previous": [...]}
""").strip()


def build_prompt(entity_id: str, attribute: str, current_value: str,
                 previous_values: list[str], reason: Optional[str]) -> str:
    lines = [f"ATTRIBUTE: {attribute}",
             f"PREVIOUS VALUES (newest first): {', '.join(repr(p) for p in previous_values) or 'none'}",
             f"CURRENT VALUE: {current_value!r}"]
    if reason:
        lines.append(f"WHY IT CHANGED: {reason}")
    return "\n".join(lines)


def parse(raw: str) -> ImplicationSet:
    m = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not m:
        return ImplicationSet(error=f"no JSON object in response: {(raw or '')[:120]!r}")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return ImplicationSet(error=f"invalid JSON: {e}")
    if not isinstance(obj, dict):
        return ImplicationSet(error="JSON is not an object")
    items: list[Implication] = []
    for kind in KINDS:
        vals = obj.get(kind, [])
        if not isinstance(vals, list) or not all(isinstance(v, str) for v in vals):
            return ImplicationSet(error=f"'{kind}' must be a list of strings")
        items += [Implication(v, kind) for v in vals]
    return ImplicationSet(items)


class LLMImplicationProvider:
    """Generate implications with a chat model at write time.

    The entity id is deliberately NOT sent: implications depend on the
    values, not on who holds them, which keeps the call cheap, cacheable
    and free of personal identifiers.
    """

    def __init__(self, provider: Optional[str] = None, *, model: Optional[str] = None,
                 complete_fn: Optional[Callable[[str, str], "tuple[str, int]"]] = None,
                 max_items: int = 24) -> None:
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
        self.max_items = max_items

    def implications(self, entity_id, attribute, current_value, previous_values,
                     reason=None) -> ImplicationSet:
        user = build_prompt(entity_id, attribute, current_value, previous_values, reason)
        try:
            raw, tokens = self._complete(SYSTEM_PROMPT, user)
        except Exception as e:
            return ImplicationSet(error=f"{type(e).__name__}: {e}", source=self.name)
        res = parse(raw)
        res.tokens_used, res.source = tokens or 0, self.name
        res.items = clean(res.items, current_value, previous_values, max_items=self.max_items)
        return res


# ---------------------------------------------------------------------------
# Static provider
# ---------------------------------------------------------------------------

class StaticImplicationProvider:
    """Hand-written rules, keyed by ``(attribute, current_value)``.

    ``rules = {("diet", "vegan"): {"conflicts_with_current": ["paneer", "ghee"]}}``
    Use ``"*"`` as the attribute to match any attribute with that value.
    """

    def __init__(self, rules: dict[tuple[str, str], dict[str, list[str]]]) -> None:
        self.rules = {(a, normalize(v)): r for (a, v), r in rules.items()}

    def implications(self, entity_id, attribute, current_value, previous_values,
                     reason=None) -> ImplicationSet:
        rule = self.rules.get((attribute, normalize(current_value))) \
            or self.rules.get(("*", normalize(current_value))) or {}
        items = [Implication(p, k) for k in KINDS for p in rule.get(k, [])]
        return ImplicationSet(clean(items, current_value, previous_values), source="static")


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class CachedImplicationProvider:
    """Persist generated implications to a JSON file.

    Key: attribute + current value + previous values (not the entity), so
    the same change is generated once across users and across runs, and an
    experiment can be re-run byte-for-byte without calling the model again.
    Failed generations are not cached.
    """

    def __init__(self, inner: ImplicationProvider, path: "str | Path") -> None:
        self.inner, self.path = inner, Path(path)
        self._data: dict = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.hits = self.misses = 0

    @staticmethod
    def key(attribute: str, current: str, previous: list[str]) -> str:
        return json.dumps([attribute, normalize(current), [normalize(p) for p in previous]])

    def implications(self, entity_id, attribute, current_value, previous_values,
                     reason=None) -> ImplicationSet:
        k = self.key(attribute, current_value, previous_values)
        if k in self._data:
            self.hits += 1
            return ImplicationSet([Implication(p, kd) for p, kd in self._data[k]],
                                  source=f"cache:{getattr(self.inner, 'name', 'inner')}")
        self.misses += 1
        res = self.inner.implications(entity_id, attribute, current_value, previous_values, reason)
        if not res.error:
            self._data[k] = [[i.phrase, i.kind] for i in res.items]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data, indent=1, ensure_ascii=False))
        return res
