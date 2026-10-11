"""
Checking agent ACTIONS, not just questions.

Stale memory does the most damage when an agent acts on it: booking a flight
from the city the user left, ordering groceries for a diet they dropped,
granting access with a role they no longer hold. Tool calls are also easier
to check than free text, because the value sits in a named argument.

    guard.check_tool_call("book_flight", {"origin": "Bengaluru", "date": "2026-11-02"},
                          entity_ids=["maya"])
    # -> BLOCK, suggested_arguments={"origin": "Pune", "date": "2026-11-02"}

How arguments are linked to attributes
--------------------------------------
1. ``arg_map`` from the host, e.g. ``{"origin": "location"}`` (most reliable).
2. Otherwise the argument NAME is matched like a query: "city", "address",
   "plan_tier", "diet" hit the attribute's own name or a cue pack entry.
3. Otherwise the argument is compared against every changed attribute of
   the entity, value-only, with lower scores.

Scoring (before entity confidence):
    argument value == a superseded value, keyed      0.95  -> BLOCK
    argument value == a superseded value, unkeyed    0.85 distinctive / 0.6 generic
    superseded value inside a longer string          0.8 keyed / 0.7 / 0.55
    write-time implication inside the argument       0.8 keyed / 0.65 unkeyed
Framing caps do not apply: an argument is an instruction, not a sentence
that might be talking about history.
"""

from __future__ import annotations

import copy
import re
from typing import TYPE_CHECKING, Any, Optional

from .models import CheckRequest, CheckResponse, Decision, Evidence, EvidenceKind, FactVersion
from .normalize import find_mentions, normalize

if TYPE_CHECKING:  # pragma: no cover
    from .guard import StateGuard

_PATH_TOKEN = re.compile(r"[A-Za-z]+")


def flatten(obj: Any, prefix: str = "") -> list[tuple[str, str]]:
    """``{"a": {"b": "x"}, "c": [1, "y"]}`` -> ``[("a.b","x"), ("c[0]","1"), ("c[1]","y")]``."""
    out: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out += flatten(v, f"{prefix}[{i}]")
    elif isinstance(obj, bool) or obj is None:
        pass
    elif isinstance(obj, (str, int, float)):
        out.append((prefix, str(obj)))
    return out


def _set_path(obj: Any, path: str, value: Any) -> None:
    parts = re.findall(r"[^.\[\]]+|\[\d+\]", path)
    for p in parts[:-1]:
        obj = obj[int(p[1:-1])] if p.startswith("[") else obj[p]
    last = parts[-1]
    if last.startswith("["):
        obj[int(last[1:-1])] = value
    else:
        obj[last] = value


def _key_words(path: str) -> str:
    """'passengers[0].home_city' -> 'home city'."""
    last = path.split(".")[-1]
    return " ".join(_PATH_TOKEN.findall(last.replace("_", " "))).lower()


def check_tool_call(guard: "StateGuard", name: str, arguments: dict,
                    entity_ids: Optional[list[str]] = None,
                    arg_map: Optional[dict[str, str]] = None,
                    scope: Optional[str] = None) -> CheckResponse:
    import time

    from .guard import _distinctive, _same

    t0 = time.perf_counter()
    leaves = flatten(arguments)
    rendered = f"{name}(" + ", ".join(f"{p}={v!r}" for p, v in leaves) + ")"
    req = CheckRequest(rendered, entity_ids=entity_ids, scope=scope)
    resolved = guard.resolver.resolve(" ".join(v for _, v in leaves), guard.store, entity_ids)
    if not resolved:
        resp = CheckResponse(Decision.PASS, 0.5,
                             [Evidence(EvidenceKind.UNRESOLVED_ENTITY, None, None,
                                       note="tool call: no entity resolved; nothing checked")],
                             reason="no entity resolved", layers_used=["layer1"])
        resp.latency_ms = round((time.perf_counter() - t0) * 1000, 3)
        return resp

    arg_map = arg_map or {}
    evidence: list[Evidence] = []
    histories: dict[tuple[str, str], list[FactVersion]] = {}
    fixes: list[tuple[str, str, str, Optional[tuple[int, int]]]] = []   # path, old, new, span

    for ent in resolved:
        attrs: dict[str, tuple[FactVersion, list[FactVersion], list[FactVersion]]] = {}
        for fact in guard.store.current(ent.entity_id, scope=scope):
            hist = guard.store.history(ent.entity_id, fact.attribute, scope)
            past: list[FactVersion] = []
            for h in hist:
                if not h.is_current and not _same(h.value, fact.value) \
                        and not any(_same(h.value, p.value) for p in past):
                    past.append(h)
            attrs[fact.attribute] = (fact, past, hist)

        for path, value in leaves:
            mapped = arg_map.get(path) or arg_map.get(path.split(".")[-1])
            if mapped:
                keyed_attrs = [mapped] if mapped in attrs else []
            else:
                kw = _key_words(path)
                keyed_attrs = list(guard.extractor.cues(kw, attrs.keys())) if kw else []
            keyed = bool(keyed_attrs)
            for attr in (keyed_attrs or list(attrs)):
                cur, past, hist = attrs[attr]
                base = dict(entity_id=ent.entity_id, attribute=attr, current_value=cur.value,
                            previous_value=past[0].value if past else None,
                            changed_at=cur.valid_from if past else None, argument=path)
                ev = None
                if _same(value, cur.value):
                    if past:
                        ev = Evidence(EvidenceKind.CURRENT_VALUE, assumed_value=cur.value,
                                      score=0.0, note="argument uses the current value", **base)
                else:
                    eq = next((i for i, p in enumerate(past, 1) if _same(value, p.value)), None)
                    if eq:
                        old = past[eq - 1].value
                        s = 0.95 if keyed else (0.85 if _distinctive(old) else 0.6)
                        ev = Evidence(EvidenceKind.STALE_VALUE, assumed_value=old,
                                      versions_back=eq, score=round(s * ent.confidence, 4),
                                      note=f"argument {path!r} is a superseded value", **base)
                        fixes.append((path, old, cur.value, None))
                    else:
                        ms = [m for m in find_mentions(value, [p.value for p in past],
                                                       guard.synonyms) if m.exact]
                        if ms:
                            m = ms[0]
                            depth = next(i for i, p in enumerate(past, 1)
                                         if _same(p.value, m.value))
                            s = 0.8 if keyed else (0.7 if _distinctive(m.value) else 0.55)
                            ev = Evidence(EvidenceKind.STALE_VALUE, assumed_value=m.value,
                                          versions_back=depth, score=round(s * ent.confidence, 4),
                                          note=f"argument {path!r} mentions a superseded value",
                                          **base)
                            fixes.append((path, m.value, cur.value, (m.start, m.end)))
                        else:
                            phrases = [p for p, _ in guard._get_impl(ent.entity_id, attr, scope)]
                            im = find_mentions(value, phrases, guard.synonyms,
                                               partial_min_tokens=99) if phrases else []
                            if im:
                                s = 0.8 if keyed else 0.65
                                ev = Evidence(EvidenceKind.IMPLIED_CONFLICT,
                                              assumed_value=im[0].value,
                                              score=round(s * ent.confidence, 4),
                                              note=f"argument {path!r} names an implication "
                                                   f"of the change", **base)
                if ev:
                    evidence.append(ev)
                    histories[(ent.entity_id, attr)] = hist

    resp = guard._decide(req, evidence, histories)
    resp.suggested_query = None
    if resp.decision is Decision.BLOCK and fixes:
        new_args = copy.deepcopy(arguments)
        # Once a superseded value is confirmed stale anywhere in the call,
        # correct every argument that uses it, not only the decisive one.
        blocking = {normalize(e.assumed_value) for e in resp.evidence
                    if e.score >= guard.block_threshold and e.kind is EvidenceKind.STALE_VALUE}
        for path, old, new, span in fixes:
            if normalize(old) not in blocking:
                continue
            if span is None:
                _set_path(new_args, path, new)
            else:
                cur_val = dict(flatten(new_args)).get(path, "")
                if cur_val:
                    m = [x for x in find_mentions(cur_val, [old], guard.synonyms) if x.exact]
                    if m:
                        _set_path(new_args, path, cur_val[:m[0].start] + new + cur_val[m[0].end:])
        resp.suggested_arguments = new_args
    if resp.decision is Decision.BLOCK and not resp.reason.startswith("tool"):
        top = resp.evidence[0]
        resp.reason = (f"tool {name}: argument {top.argument!r}={top.assumed_value!r} "
                       f"assumes {top.entity_id}'s old {top.attribute}; it is now "
                       f"{top.current_value!r}")
    resp.latency_ms = round((time.perf_counter() - t0) * 1000, 3)
    return resp
