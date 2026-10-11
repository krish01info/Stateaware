"""
Public, benchmark-independent data model for StateGuard.

Nothing in this module knows about the STALE dataset. A host application
builds a :class:`CheckRequest`, hands it to :class:`stateguard.StateGuard`,
and receives a :class:`CheckResponse` carrying a three-way
:class:`Decision` plus the evidence that produced it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class Decision(str, Enum):
    """Outcome of a pre-hoc state check.

    PASS   – no reliable stale or contradicted assumption; continue normally.
    REVIEW – relevant changed state exists but the premise is ambiguous;
             the host decides (clarify with the user, run a verifier, log).
    BLOCK  – strong evidence the query assumes a superseded value.
    """

    PASS = "pass"
    REVIEW = "review"
    BLOCK = "block"


class EvidenceKind(str, Enum):
    """Why a piece of evidence was produced."""

    STALE_VALUE = "stale_value"          # query names a superseded value
    CURRENT_VALUE = "current_value"      # query names the current value
    ATTRIBUTE_CUE = "attribute_cue"      # query touches a changed attribute, no value named
    IMPLIED_CONFLICT = "implied_conflict"  # query names a write-time implication of the change
    UNRESOLVED_ENTITY = "unresolved_entity"
    VERIFIER = "verifier"                # produced by the Layer-2 verifier
    PROVIDER_ERROR = "provider_error"    # verifier failed / unparseable


@dataclass(frozen=True)
class FactVersion:
    """One version of one attribute of one entity.

    ``valid_to is None`` means this version is current.
    ``scope`` lets a host keep parallel truths (region, tenant, policy
    version) without them superseding each other.
    """

    entity_id: str
    attribute: str
    value: str
    valid_from: str
    valid_to: Optional[str] = None
    source: Optional[str] = None
    confidence: float = 1.0
    scope: Optional[str] = None
    reason: Optional[str] = None

    @property
    def is_current(self) -> bool:
        return self.valid_to is None


@dataclass
class CheckRequest:
    """Input to :meth:`StateGuard.check`.

    query      : the user query, before it reaches the main model.
    entity_ids : entities the host already knows the query is about. When
                 given, no name guessing is done.
    scope      : only consider facts in this scope (plus unscoped facts).
    metadata   : free-form, passed through to evidence/verifier untouched.
    """

    query: str
    entity_ids: Optional[list[str]] = None
    scope: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Evidence:
    """A single, auditable reason behind a decision."""

    kind: EvidenceKind
    entity_id: Optional[str]
    attribute: Optional[str]
    current_value: Optional[str] = None
    assumed_value: Optional[str] = None      # what the query appears to assume
    previous_value: Optional[str] = None     # most recent superseded value
    versions_back: Optional[int] = None      # 1 = immediately previous version
    source: str = "layer1"
    score: float = 0.0                        # strength in [0, 1]
    note: Optional[str] = None
    changed_at: Optional[str] = None          # when the current value took effect
    argument: Optional[str] = None            # tool-call argument path, if any


@dataclass
class CheckResponse:
    """Output of :meth:`StateGuard.check`."""

    decision: Decision
    confidence: float
    evidence: list[Evidence] = field(default_factory=list)
    suggested_query: Optional[str] = None
    reason: Optional[str] = None
    suggested_arguments: Optional[dict[str, Any]] = None   # tool calls only
    layers_used: list[str] = field(default_factory=list)
    tokens_used: int = 0
    latency_ms: float = 0.0

    @property
    def is_blocked(self) -> bool:
        return self.decision is Decision.BLOCK

    @property
    def needs_review(self) -> bool:
        return self.decision is Decision.REVIEW

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["decision"] = self.decision.value
        for ev in d["evidence"]:
            ev["kind"] = ev["kind"].value if isinstance(ev["kind"], Enum) else ev["kind"]
        return d
