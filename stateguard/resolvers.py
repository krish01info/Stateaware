"""
Entity resolution, kept separate from contradiction checking.

A resolver answers "who/what is this query about?". The decision engine then
answers "does it assume an outdated state of that entity?". Keeping the two
apart means a failure to resolve an entity is *visible* (it produces an
``UNRESOLVED_ENTITY`` signal) instead of silently passing the query.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from .normalize import phrase_present
from .store import FactStore


@dataclass(frozen=True)
class ResolvedEntity:
    entity_id: str
    confidence: float          # 1.0 = supplied by the host, lower = guessed
    matched_text: Optional[str] = None
    method: str = "explicit"


@runtime_checkable
class EntityResolver(Protocol):
    def resolve(self, query: str, store: FactStore,
                candidates: Optional[list[str]] = None) -> list[ResolvedEntity]: ...


class ExplicitEntityResolver:
    """Trust the entity IDs the host passes in ``CheckRequest.entity_ids``.

    This is the recommended mode: the host application nearly always knows
    which user, account or device a request concerns.
    """

    def resolve(self, query, store, candidates=None) -> list[ResolvedEntity]:
        return [ResolvedEntity(e, 1.0, None, "explicit") for e in (candidates or [])]


_FIRST_PERSON = re.compile(r"\b(i|i'm|im|i've|me|my|mine|myself)\b", re.IGNORECASE)


class HeuristicEntityResolver:
    """Fallback resolver for when the host does not supply entity IDs.

    Resolution order:
    1. Explicit candidates, if any were given.
    2. Aliases and entity IDs from the store found verbatim in the query
       ("Alice", "account 123", "the living-room thermostat").
    3. First-person language ("my", "I") -> ``self_entity`` if configured.
    4. If the store holds exactly one entity and ``single_entity_default`` is
       on, that entity, at low confidence.

    It never invents entities that are not in the store.
    """

    def __init__(self, self_entity: Optional[str] = None,
                 single_entity_default: bool = True) -> None:
        self.self_entity = self_entity
        self.single_entity_default = single_entity_default

    def resolve(self, query, store, candidates=None) -> list[ResolvedEntity]:
        if candidates:
            return ExplicitEntityResolver().resolve(query, store, candidates)

        found: dict[str, ResolvedEntity] = {}
        names = {e: e for e in store.entities()}
        for alias, eid in store.aliases().items():
            names.setdefault(alias, eid)
        for surface, eid in sorted(names.items(), key=lambda x: len(x[0]), reverse=True):
            # IDs such as "account_123" are also matched as "account 123".
            if phrase_present(query, surface.replace("_", " ")) and eid not in found:
                found[eid] = ResolvedEntity(eid, 0.9, surface, "name")

        if not found and self.self_entity and _FIRST_PERSON.search(query):
            found[self.self_entity] = ResolvedEntity(self.self_entity, 0.85, None, "first_person")

        if not found and self.single_entity_default:
            ents = store.entities()
            if len(ents) == 1:
                found[ents[0]] = ResolvedEntity(ents[0], 0.6, None, "single_entity")

        return list(found.values())
