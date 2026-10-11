"""
The StateGuard decision engine.

    guard = StateGuard(store=SQLiteFactStore("facts.db"))
    guard.remember("alice", "diet", "vegetarian")
    guard.remember("alice", "diet", "vegan", reason="told us on Aug 14")
    guard.check(CheckRequest("Book a vegetarian table for Alice", entity_ids=["alice"]))
    # -> decision=BLOCK, suggested_query="Book a vegan table for Alice"

How a decision is reached
-------------------------
For every resolved entity and every attribute of it that has *changed*
(two or more versions), the engine collects evidence:

* STALE_VALUE   – the query names a superseded value (any depth of history).
* CURRENT_VALUE – the query names the current value.
* ATTRIBUTE_CUE – the query touches the changed attribute but names no value;
                  a possible *implicit* premise the deterministic layer
                  cannot judge.

Each piece of evidence gets a score in [0, 1]. The score starts from match
quality and is then capped by how the query *frames* the state
(see :func:`stateguard.extractors.premise_frame`): naming an old value while
contrasting it with the new one, negating it, or asking about history is not
the same as assuming it. Finally the score is scaled by how sure we are about
the entity.

    score >= block_threshold                       -> BLOCK
    review_threshold <= score < block_threshold    -> verifier if configured, else REVIEW
    score < review_threshold                       -> PASS

A verifier that errors or returns low confidence yields REVIEW, never PASS.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Union

from .extractors import (ACCOUNT_STATE, PERSON_PROFILE, AttributeExtractor,
                         KeywordAttributeExtractor, PremiseFrame, is_negated,
                         premise_frame)
from .models import (CheckRequest, CheckResponse, Decision, Evidence, EvidenceKind,
                     FactVersion)
from .normalize import STOPWORDS, Mention, SynonymTable, find_mentions, normalize
from .resolvers import EntityResolver, HeuristicEntityResolver, ResolvedEntity
from .store import FactStore, InMemoryFactStore
from .verifier import Verifier
from .implications import ImplicationProvider
from .extraction import ExtractedFact, FactExtractor
from .store import ImplicationStore

# Score caps applied by premise framing. Kept as named constants so the paper
# can report them and ablations can vary them.
CAP_NEGATED = 0.15        # "she is NOT vegetarian"
CAP_CONTRAST = 0.25       # "vegan instead of vegetarian", "now that she's vegan"
CAP_BOTH_VALUES = 0.3     # names old AND current value: describing the change
CAP_HISTORY = 0.3         # "I used to live in Seattle", "no longer vegetarian"
CAP_PAST = 0.6            # plain past tense / "my old ..." -> ambiguous
UNCUED_GENERIC = 0.65     # single lowercase word, no attribute cue ("monthly")
CAP_FUTURE = 0.45         # "visiting Seattle next month"
CAP_DIRECT_Q = 0.45       # "is Alice still vegetarian?"
IMPLICIT_SCORE = 0.4      # attribute touched, no value named
IMPLIED_SCORE = 0.7       # names a write-time implication of the change ("paneer" for vegan)
IMPLIED_CUED_SCORE = 0.8  # ... and also touches the attribute ("paneer RECIPE")
IMPLICIT_SCORE_DIRECT_Q = 0.1


@dataclass
class ObserveResult:
    written: list[FactVersion] = field(default_factory=list)
    skipped: list[ExtractedFact] = field(default_factory=list)
    tokens_used: int = 0
    error: Optional[str] = None


class StateGuard:
    """Pre-hoc interceptor: checks a query against versioned state before generation.

    Parameters
    ----------
    store              : any :class:`FactStore` (defaults to in-memory).
    resolver           : entity resolver (defaults to heuristic, which still
                         prefers explicit ``entity_ids`` when given).
    extractor          : attribute-cue extractor (defaults to attribute names
                         plus the person-profile and account packs).
    verifier           : optional Layer-2 :class:`Verifier`.
    synonyms           : ``dict`` or :class:`SynonymTable` of value variants.
    block_threshold    : minimum score for BLOCK.
    review_threshold   : minimum score for REVIEW / verification.
    verifier_threshold : verifier confidence needed to act on its verdict.
    implicit_without_verifier : decision for implicit-premise evidence when
                         no verifier is configured. PASS by default: the
                         deterministic layer cannot judge implicit premises,
                         and flagging every one would make REVIEW meaningless.
    max_verifications  : cap on Layer-2 calls per query (cost control).
    fact_extractor     : optional :class:`FactExtractor` used by :meth:`observe`
                         to learn facts from user messages.
    implications       : optional :class:`ImplicationProvider`. When set,
                         :meth:`remember` asks it once per new value which
                         phrases the change rules out, and :meth:`check`
                         matches those phrases deterministically.
    """

    def __init__(
        self,
        store: Optional[FactStore] = None,
        resolver: Optional[EntityResolver] = None,
        extractor: Optional[AttributeExtractor] = None,
        verifier: Optional[Verifier] = None,
        synonyms: Union[None, dict, SynonymTable] = None,
        *,
        block_threshold: float = 0.75,
        review_threshold: float = 0.35,
        verifier_threshold: float = 0.7,
        implicit_without_verifier: Decision = Decision.PASS,
        max_verifications: int = 2,
        implications: Optional[ImplicationProvider] = None,
        fact_extractor: Optional[FactExtractor] = None,
    ) -> None:
        self.store = store if store is not None else InMemoryFactStore()
        self.resolver = resolver or HeuristicEntityResolver()
        self.extractor = extractor or KeywordAttributeExtractor(PERSON_PROFILE, ACCOUNT_STATE)
        self.verifier = verifier
        self.synonyms = synonyms if isinstance(synonyms, SynonymTable) else SynonymTable(synonyms)
        self.block_threshold = block_threshold
        self.review_threshold = review_threshold
        self.verifier_threshold = verifier_threshold
        self.implicit_without_verifier = implicit_without_verifier
        self.max_verifications = max_verifications
        self.implications = implications
        self.fact_extractor = fact_extractor
        # Write-time cost accounting, so papers/dashboards can report it.
        self.write_tokens = 0
        self.write_calls = 0
        self.write_errors: list[str] = []
        # Fallback for stores that do not implement ImplicationStore.
        self._impl_mem: dict[tuple, tuple[str, list[tuple[str, str]]]] = {}

    # ------------------------------------------------------------------ write

    def remember(self, entity_id: str, attribute: str, value: str, **kw) -> FactVersion:
        """Record a fact; a different value supersedes the current one.

        If an implication provider is configured and this is a new value,
        the provider is called once and its phrases are stored with the fact.
        """
        fv = self.store.record(entity_id, attribute, value, **kw)
        if self.implications is not None and not self._get_impl(entity_id, attribute, fv.scope):
            self.refresh_implications(entity_id, attribute, scope=fv.scope, reason=kw.get("reason"))
        return fv

    def check_tool_call(self, name: str, arguments: dict,
                        entity_ids: Optional[list[str]] = None, *,
                        arg_map: Optional[dict[str, str]] = None,
                        scope: Optional[str] = None) -> CheckResponse:
        """Check an agent's tool call before it runs. See :mod:`stateguard.tools`.

        On BLOCK, ``suggested_arguments`` holds the call with superseded
        values replaced by current ones.
        """
        from .tools import check_tool_call
        return check_tool_call(self, name, arguments, entity_ids, arg_map, scope)

    def observe(self, entity_id: str, message: str, *, valid_from: Optional[str] = None,
                min_confidence: float = 0.6, subject: Optional[str] = None) -> ObserveResult:
        """Learn facts from a message the entity (usually the user) wrote.

        Facts at or above ``min_confidence`` are recorded with
        :meth:`remember` (so a changed value supersedes the old one and
        write-time implications are generated); the rest are returned in
        ``skipped`` for the host to log or confirm with the user.
        """
        if self.fact_extractor is None:
            raise ValueError("no fact_extractor configured")
        known = [f.attribute for f in self.store.current(entity_id)]
        res = self.fact_extractor.extract(message, known, subject or "the user")
        self.write_tokens += res.tokens_used
        out = ObserveResult(tokens_used=res.tokens_used, error=res.error)
        if res.error:
            self.write_errors.append(f"observe {entity_id}: {res.error}")
            return out
        for f in res.facts:
            if f.confidence < min_confidence:
                out.skipped.append(f)
                continue
            out.written.append(self.remember(
                entity_id, f.attribute, f.value, valid_from=valid_from, source="message",
                confidence=f.confidence, reason=f.evidence or message[:120]))
        return out

    def refresh_implications(self, entity_id: str, attribute: str,
                             scope: Optional[str] = None, reason: Optional[str] = None) -> int:
        """(Re)generate implications for the current value. Returns phrase count."""
        if self.implications is None:
            raise ValueError("no implication provider configured")
        cur = [f for f in self.store.current(entity_id, attribute) if f.scope == scope]
        if not cur:
            return 0
        cur = cur[0]
        hist = self.store.history(entity_id, attribute, scope)
        previous: list[str] = []
        for h in hist:
            if not h.is_current and not _same(h.value, cur.value) \
                    and not any(_same(h.value, p) for p in previous):
                previous.append(h.value)
        res = self.implications.implications(entity_id, attribute, cur.value, previous,
                                             reason or cur.reason)
        self.write_calls += 1
        self.write_tokens += res.tokens_used
        if res.error:
            self.write_errors.append(f"{entity_id}.{attribute}: {res.error}")
            return 0
        phrases = [(i.phrase, i.kind) for i in res.items]
        if isinstance(self.store, ImplicationStore):
            self.store.set_implications(entity_id, attribute, cur.value, phrases, scope)
        else:
            self._impl_mem[(entity_id, attribute, scope)] = (cur.value, phrases)
        return len(phrases)

    def _get_impl(self, entity_id: str, attribute: str, scope: Optional[str]
                  ) -> list[tuple[str, str]]:
        if isinstance(self.store, ImplicationStore):
            return self.store.get_implications(entity_id, attribute, scope)
        out = []
        for f in self.store.current(entity_id, attribute, scope):
            rec = self._impl_mem.get((entity_id, attribute, f.scope))
            if rec and rec[0] == f.value:
                out += rec[1]
        return out

    # ------------------------------------------------------------------ check

    def check(self, request: Union[CheckRequest, str],
              entity_ids: Optional[list[str]] = None) -> CheckResponse:
        if isinstance(request, str):
            request = CheckRequest(query=request, entity_ids=entity_ids)
        t0 = time.perf_counter()
        resp = self._check(request)
        resp.latency_ms = round((time.perf_counter() - t0) * 1000, 3)
        return resp

    def _check(self, req: CheckRequest) -> CheckResponse:
        query = req.query
        frame = premise_frame(query)
        resolved = self.resolver.resolve(query, self.store, req.entity_ids)

        if not resolved:
            return self._unresolved(req, frame)

        evidence: list[Evidence] = []
        histories: dict[tuple[str, str], list[FactVersion]] = {}
        for ent in resolved:
            for ev, hist in self._entity_evidence(query, ent, req.scope, frame):
                evidence.append(ev)
                histories[(ev.entity_id, ev.attribute)] = hist
        return self._decide(req, evidence, histories)

    # ------------------------------------------------------------ evidence

    def _entity_evidence(self, query: str, ent: ResolvedEntity, scope: Optional[str],
                         frame: PremiseFrame):
        current = self.store.current(ent.entity_id, scope=scope)
        changed: dict[str, list[FactVersion]] = {}
        implied: dict[str, list[tuple[str, str]]] = {}
        for fact in current:
            hist = self.store.history(ent.entity_id, fact.attribute, scope)
            if any(not h.is_current and not _same(h.value, fact.value) for h in hist):
                changed[fact.attribute] = hist
            phrases = self._get_impl(ent.entity_id, fact.attribute, scope)
            if phrases:
                implied[fact.attribute] = phrases
                changed.setdefault(fact.attribute, hist)
        if not changed:
            return
        cues = self.extractor.cues(query, changed.keys())

        for attr, hist in changed.items():
            cur = next(h for h in hist if h.is_current)
            past: list[FactVersion] = []
            for h in hist:
                if not h.is_current and not _same(h.value, cur.value) \
                        and not any(_same(h.value, p.value) for p in past):
                    past.append(h)           # newest-first, deduplicated
            mentions = find_mentions(query, [cur.value] + [p.value for p in past], self.synonyms)
            cur_m = [m for m in mentions if _same(m.value, cur.value)]
            stale_m = [m for m in mentions if not _same(m.value, cur.value)]
            base = dict(entity_id=ent.entity_id, attribute=attr, current_value=cur.value,
                        previous_value=past[0].value if past else None,
                        changed_at=cur.valid_from if past else None)

            for m in stale_m:
                depth = next(i for i, p in enumerate(past, 1) if _same(p.value, m.value))
                score, note = self._score_stale(m, attr in cues, bool(cur_m), frame)
                yield Evidence(kind=EvidenceKind.STALE_VALUE, assumed_value=m.value,
                               versions_back=depth, score=round(score * ent.confidence, 4),
                               note=note, **base), hist

            # Write-time implications: phrases the current value rules out.
            impl_m = []
            if attr in implied:
                used = [(m.start, m.end) for m in mentions]
                kinds = {normalize(p): k for p, k in implied[attr]}
                for m in find_mentions(query, [p for p, _ in implied[attr]], self.synonyms,
                                       partial_min_tokens=99):
                    if any(a < m.end and m.start < b for a, b in used):
                        continue
                    impl_m.append(m)
                    score, notes = (IMPLIED_CUED_SCORE, ["names an implication of the change; "
                                                         "attribute cue present"]) \
                        if attr in cues else (IMPLIED_SCORE, ["names an implication of the change"])
                    score, note = self._apply_caps(score, notes, m, bool(cur_m), frame)
                    kind = kinds.get(normalize(m.value), "conflicts_with_current")
                    yield Evidence(kind=EvidenceKind.IMPLIED_CONFLICT, assumed_value=m.value,
                                   score=round(score * ent.confidence, 4),
                                   note=f"{note} [{kind}]", **base), hist

            if not stale_m and not impl_m and attr in changed and len(hist) > 1:
                if cur_m:
                    yield Evidence(kind=EvidenceKind.CURRENT_VALUE, assumed_value=cur.value,
                                   score=0.0, note="query uses the current value", **base), hist
                elif attr in cues and past:
                    s = IMPLICIT_SCORE_DIRECT_Q if frame.direct_question else IMPLICIT_SCORE
                    yield Evidence(kind=EvidenceKind.ATTRIBUTE_CUE, score=round(s * ent.confidence, 4),
                                   note="touches a changed attribute without naming a value",
                                   **base), hist

    def _score_stale(self, m: Mention, cued: bool, names_current: bool,
                     frame: PremiseFrame) -> tuple[float, str]:
        if m.exact:
            norm = normalize(m.value)
            has_content = any(t not in STOPWORDS for t in norm.split())
            if not has_content or (len(norm) <= 2 and not any(c.isdigit() for c in norm)):
                score, notes = 0.5, ["exact mention of a very short/function-word value"]
            elif cued or _distinctive(m.value):
                score, notes = 0.9, ["exact mention of a superseded value"]
            else:
                score, notes = UNCUED_GENERIC, [
                    "exact mention of a generic one-word value with no attribute cue"]
        else:
            score = 0.25 + 0.35 * m.coverage
            notes = [f"partial overlap ({m.coverage:.0%}) with a superseded value"]
        if cued:
            score = min(score + 0.05, 0.98)
            notes.append("attribute cue present")

        return self._apply_caps(score, notes, m, names_current, frame)

    @staticmethod
    def _apply_caps(score: float, notes: list[str], m: Mention, names_current: bool,
                    frame: PremiseFrame) -> tuple[float, str]:
        """Cap a score by how the query frames the mentioned state."""
        caps = []
        if frame.negated_spans and is_negated(frame, m.start):
            caps.append((CAP_NEGATED, "mention is negated"))
        if frame.contrast:
            caps.append((CAP_CONTRAST, "query contrasts old and new state"))
        elif names_current:
            caps.append((CAP_BOTH_VALUES, "query also names the current value"))
        if frame.history:
            caps.append((CAP_HISTORY, "query says the value is in the past"))
        elif frame.past:
            caps.append((CAP_PAST, "query uses past-tense framing"))
        if frame.future:
            caps.append((CAP_FUTURE, "forward-looking request"))
        if frame.direct_question:
            caps.append((CAP_DIRECT_Q, "asks about the state rather than assuming it"))
        for cap, why in caps:
            if score > cap:
                score = cap
                notes.append(why)
        return score, "; ".join(notes)

    # -------------------------------------------------------------- decide

    def _decide(self, req: CheckRequest, evidence: list[Evidence],
                histories: dict) -> CheckResponse:
        layers = ["layer1"]
        if not evidence:
            return CheckResponse(Decision.PASS, 1.0, [], reason="no changed state is involved",
                                 layers_used=layers)
        evidence.sort(key=lambda e: e.score, reverse=True)
        top = evidence[0]

        if top.score >= self.block_threshold:
            blocking = [e for e in evidence if e.score >= self.block_threshold]
            return CheckResponse(Decision.BLOCK, top.score, evidence,
                                 suggested_query=_rewrite(req.query, blocking, self.synonyms),
                                 reason=_explain(top), layers_used=layers)

        if top.score < self.review_threshold:
            reason = ("query is consistent with the current state"
                      if top.kind is EvidenceKind.CURRENT_VALUE
                      else "changed state is mentioned but not assumed")
            return CheckResponse(Decision.PASS, round(1 - top.score, 4), evidence,
                                 reason=reason, layers_used=layers)

        # Ambiguous zone.
        if self.verifier is None:
            if top.kind is EvidenceKind.ATTRIBUTE_CUE and \
                    self.implicit_without_verifier is Decision.PASS and \
                    all(e.kind is not EvidenceKind.STALE_VALUE or e.score < self.review_threshold
                        for e in evidence):
                return CheckResponse(Decision.PASS, round(1 - top.score, 4), evidence,
                                     reason="possible implicit premise; no verifier configured",
                                     layers_used=layers)
            return CheckResponse(Decision.REVIEW, top.score, evidence, reason=_explain(top),
                                 layers_used=layers)
        return self._verify(req, evidence, histories, layers)

    def _verify(self, req, evidence, histories, layers) -> CheckResponse:
        layers.append("verifier")
        tokens = 0
        seen, todo = set(), []
        for e in evidence:
            key = (e.entity_id, e.attribute)
            if e.score >= self.review_threshold and key not in seen:
                seen.add(key)
                todo.append(e)
        todo = todo[: self.max_verifications]

        confirmed_pass = 0
        for e in todo:
            hist = histories[(e.entity_id, e.attribute)]
            related = [x for x in evidence if (x.entity_id, x.attribute) == (e.entity_id, e.attribute)]
            r = self.verifier.verify(req.query, e.entity_id, e.attribute, hist, related)
            tokens += r.tokens_used
            if r.error:
                evidence.append(Evidence(EvidenceKind.PROVIDER_ERROR, e.entity_id, e.attribute,
                                         current_value=e.current_value, source="verifier",
                                         note=r.error))
                return CheckResponse(Decision.REVIEW, e.score, evidence,
                                     reason=f"verifier failed: {r.error}",
                                     layers_used=layers, tokens_used=tokens)
            evidence.append(Evidence(EvidenceKind.VERIFIER, e.entity_id, e.attribute,
                                     current_value=e.current_value,
                                     assumed_value=r.assumed_value,
                                     previous_value=e.previous_value, source="verifier",
                                     score=r.confidence if r.is_stale else 1 - r.confidence,
                                     note=r.reason))
            if r.confidence < self.verifier_threshold:
                return CheckResponse(Decision.REVIEW, r.confidence, evidence,
                                     reason=f"verifier unsure: {r.reason}",
                                     layers_used=layers, tokens_used=tokens)
            if r.is_stale:
                return CheckResponse(Decision.BLOCK, r.confidence, evidence,
                                     suggested_query=r.corrected_query
                                     or _rewrite(req.query, [e], self.synonyms),
                                     reason=r.reason or _explain(e),
                                     layers_used=layers, tokens_used=tokens)
            confirmed_pass += 1

        if confirmed_pass == len(todo):
            return CheckResponse(Decision.PASS, min(1.0, self.verifier_threshold), evidence,
                                 reason="verifier found no stale premise",
                                 layers_used=layers, tokens_used=tokens)
        return CheckResponse(Decision.REVIEW, evidence[0].score, evidence,
                             layers_used=layers, tokens_used=tokens)

    # ---------------------------------------------------------- unresolved

    def _unresolved(self, req: CheckRequest, frame: PremiseFrame) -> CheckResponse:
        """No entity resolved. REVIEW only if the query carries a premise that
        matches some entity's superseded value; otherwise PASS."""
        hits = []
        for eid in self.store.entities()[:1000]:
            ent = ResolvedEntity(eid, 0.5, None, "unresolved_scan")
            for ev, _ in self._entity_evidence(req.query, ent, req.scope, frame):
                if ev.kind is EvidenceKind.STALE_VALUE and ev.score * 2 >= self.block_threshold:
                    hits.append(ev)
        if hits:
            ev = Evidence(EvidenceKind.UNRESOLVED_ENTITY, None, hits[0].attribute,
                          assumed_value=hits[0].assumed_value, score=0.5,
                          note=f"matches a superseded value of {len({h.entity_id for h in hits})} "
                               f"entit(ies) but the query's entity could not be resolved")
            return CheckResponse(Decision.REVIEW, 0.5, [ev] + hits,
                                 reason="stale-looking premise, entity unknown",
                                 layers_used=["layer1"])
        return CheckResponse(Decision.PASS, 0.5,
                             [Evidence(EvidenceKind.UNRESOLVED_ENTITY, None, None,
                                       note="no entity resolved; nothing checked")],
                             reason="no entity resolved", layers_used=["layer1"])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _distinctive(value: str) -> bool:
    """Proper nouns, numbers/IDs and multi-word values rarely appear by accident."""
    toks = [t for t in normalize(value).split() if t not in STOPWORDS]
    return (any(c.isupper() for c in value) or any(c.isdigit() for c in value)
            or len(toks) >= 2)


def _same(a: str, b: str) -> bool:
    return normalize(a) == normalize(b)


def _explain(e: Evidence) -> str:
    if e.kind is EvidenceKind.STALE_VALUE:
        depth = "" if (e.versions_back or 1) == 1 else f" ({e.versions_back} versions ago)"
        return (f"query assumes {e.attribute}={e.assumed_value!r}{depth} for {e.entity_id}, "
                f"but it is now {e.current_value!r}")
    if e.kind is EvidenceKind.IMPLIED_CONFLICT:
        return (f"query mentions {e.assumed_value!r}, which does not fit {e.entity_id}'s "
                f"current {e.attribute} {e.current_value!r}")
    if e.kind is EvidenceKind.ATTRIBUTE_CUE:
        return (f"query depends on {e.entity_id}'s {e.attribute}, which changed to "
                f"{e.current_value!r}")
    return e.note or e.kind.value


def _rewrite(query: str, evidence: list[Evidence], synonyms: SynonymTable) -> Optional[str]:
    """Deterministic correction: swap each stale span for the current value."""
    spans = []
    for e in evidence:
        if e.kind is not EvidenceKind.STALE_VALUE or not e.assumed_value:
            continue
        for m in find_mentions(query, [e.assumed_value], synonyms):
            if m.exact:
                spans.append((m.start, m.end, e.current_value))
    if not spans:
        return None
    out = query
    for s, t, val in sorted(set(spans), reverse=True):
        out = out[:s] + val + out[t:]
    return out
