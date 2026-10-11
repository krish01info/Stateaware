"""
Text normalisation and value-mention matching.

Design goals: cheap, deterministic and auditable. Every match reports the
exact character span in the original query, so a BLOCK decision can always
point at the words that triggered it.

What is normalised
------------------
* Unicode (NFKC) and case folding.
* Punctuation, hyphens and slashes become spaces ("gluten-free" == "gluten free").
* Light inflection: a trailing plural "s"/"es" is ignored ("vegetarians").
* Money: "$9.99", "9.99 usd", "USD 9.99" all become the token ``9.99``.
* Caller-supplied synonyms ("nyc" -> "new york", "wfh" -> "work from home").

What is deliberately NOT done: aggressive stemming or embedding similarity.
Those blur exactly the distinctions this library must keep (vegan vs
vegetarian). Semantic cases are routed to the verifier instead.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Optional

_TOKEN_RE = re.compile(r"[^\W_]+(?:[.'][^\W_]+)*", re.UNICODE)
_MONEY_RE = re.compile(
    r"(?:(?:[$€£₹]|\b(?:usd|eur|gbp|inr|rs\.?))\s*(\d+(?:[.,]\d+)?))"
    r"|(?:\b(\d+(?:[.,]\d+)?)\s*(?:usd|eur|gbp|inr|dollars?|rupees?)\b)",
    re.IGNORECASE,
)

# Function words that never count as evidence on their own.
STOPWORDS = frozenset("""
a an the and or but if of to in on at by for from with without into onto over under
is are was were be been being am do does did done have has had having can could will
would shall should may might must i me my mine we us our you your he him his she her
it its they them their this that these those there here what which who whom whose when
where why how not no yes so than then too very just also any some all each every more
most other such only own same few both about after before again against between during
up down out off once while because until as per via
""".split())


@dataclass(frozen=True)
class Token:
    norm: str      # normalised form used for comparison
    start: int     # char offset in the original string
    end: int


def _stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith(("ses", "xes", "ches", "shes")):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def tokenize(text: str) -> list[Token]:
    """Tokenise *text*, keeping offsets into the original string."""
    folded = unicodedata.normalize("NFKC", text)
    # Keep offsets aligned: NFKC rarely changes length for Latin text; if it
    # does, fall back to the original so spans stay valid.
    src = folded if len(folded) == len(text) else text
    src = _MONEY_RE.sub(lambda m: _money_placeholder(m), src)
    out = []
    for m in _TOKEN_RE.finditer(src):
        raw = m.group(0).casefold()
        if raw.endswith("'s"):
            raw = raw[:-2]          # possessive: "alice's" -> "alice"
        raw = raw.replace("'", "")
        out.append(Token(_stem(raw.replace(",", ".")), m.start(), m.end()))
    return out


def _money_placeholder(m: re.Match) -> str:
    amount = (m.group(1) or m.group(2)).replace(",", ".")
    # pad to the original length so later offsets stay aligned
    return amount.ljust(len(m.group(0)))


def normalize(text: str) -> str:
    return " ".join(t.norm for t in tokenize(text))


def content_tokens(text: str) -> list[str]:
    return [t.norm for t in tokenize(text) if t.norm not in STOPWORDS]


@dataclass(frozen=True)
class Mention:
    value: str          # the stored value that was matched
    start: int          # span in the original query
    end: int
    exact: bool         # full token-sequence match (vs. partial overlap)
    coverage: float     # fraction of the value's content tokens found


class SynonymTable:
    """Maps surface variants to a canonical form, e.g. ``{"nyc": "new york"}``.

    Both the query and stored values are expanded through the table, so a
    query saying "NYC" matches a stored value "New York" and vice versa.
    """

    def __init__(self, mapping: Optional[dict[str, str]] = None) -> None:
        self._map: dict[tuple[str, ...], tuple[str, ...]] = {}
        for k, v in (mapping or {}).items():
            self.add(k, v)

    def add(self, variant: str, canonical: str) -> None:
        self._map[tuple(t.norm for t in tokenize(variant))] = tuple(
            t.norm for t in tokenize(canonical))

    def variants(self, value: str) -> list[tuple[str, ...]]:
        """All token sequences that should count as a mention of *value*."""
        base = tuple(t.norm for t in tokenize(value))
        out = {base}
        for var, canon in self._map.items():
            if canon == base:
                out.add(var)
            if var == base:
                out.add(canon)
        return [v for v in out if v]


def find_mentions(
    query: str,
    values: Iterable[str],
    synonyms: Optional[SynonymTable] = None,
    partial_min_tokens: int = 3,
    partial_min_coverage: float = 0.6,
    max_gap: int = 2,
) -> list[Mention]:
    """Find which of *values* the query mentions.

    Longest values are matched first and their spans are reserved, so a
    query naming "senior engineer" is not also counted as naming "engineer".

    A value also matches exactly when all of its content words appear in
    order with at most ``max_gap`` extra tokens in between ("can't I watch
    in 4K" matches "can't watch in 4K"; "Tuesday at 10am" matches
    "Tuesday 10am"). Values need at least two content words for this.

    Long free-text values (>= ``partial_min_tokens`` content tokens) can also
    match partially when enough of their content words appear; such matches
    are flagged ``exact=False`` and treated as weak evidence downstream.
    """
    synonyms = synonyms or SynonymTable()
    q = tokenize(query)
    qn = [t.norm for t in q]
    taken = [False] * len(q)
    found: list[Mention] = []

    vals = sorted({v for v in values if v and v.strip()},
                  key=lambda v: len(tokenize(v)), reverse=True)

    for val in vals:
        hit = None
        for seq in sorted(synonyms.variants(val), key=len, reverse=True):
            n = len(seq)
            for i in range(len(qn) - n + 1):
                if tuple(qn[i:i + n]) == seq and not any(taken[i:i + n]):
                    hit = (i, i + n)
                    break
            if hit:
                break
        if not hit:
            hit = _gapped(qn, taken, content_tokens(val), max_gap)
        if hit:
            i, j = hit
            for k in range(i, j):
                taken[k] = True
            found.append(Mention(val, q[i].start, q[j - 1].end, True, 1.0))
            continue

        ct = [c for c in content_tokens(val)]
        if len(set(ct)) >= partial_min_tokens:
            idx = [k for k, t in enumerate(qn) if t in set(ct) and t not in STOPWORDS and not taken[k]]
            cov = len({qn[k] for k in idx}) / len(set(ct))
            if cov >= partial_min_coverage:
                found.append(Mention(val, q[idx[0]].start, q[idx[-1]].end, False, cov))
    return found


def _gapped(qn: list[str], taken: list[bool], ct: list[str], max_gap: int):
    """In-order match of content tokens *ct* allowing up to *max_gap* fillers."""
    if len(ct) < 2 or max_gap <= 0:
        return None
    for i, tok in enumerate(qn):
        if tok != ct[0] or taken[i]:
            continue
        k, j, gaps = 1, i + 1, 0
        while k < len(ct) and j < len(qn) and gaps <= max_gap:
            if taken[j]:
                break
            if qn[j] == ct[k]:
                k += 1
            elif qn[j] not in STOPWORDS:
                gaps += 1
            j += 1
        if k == len(ct) and gaps <= max_gap:
            return i, j
    return None


def phrase_present(query: str, phrase: str) -> Optional[tuple[int, int]]:
    """Return the span of *phrase* in *query* (token-level), or None."""
    q = tokenize(query)
    p = [t.norm for t in tokenize(phrase)]
    n = len(p)
    if not n:
        return None
    for i in range(len(q) - n + 1):
        if [t.norm for t in q[i:i + n]] == p:
            return q[i].start, q[i + n - 1].end
    return None
