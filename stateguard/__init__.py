"""StateGuard – pre-hoc interception of stale-state premises.

Stable public API (v0.3):

    from stateguard import StateGuard, CheckRequest, Decision
    guard = StateGuard()
    guard.remember("alice", "diet", "vegetarian")
    guard.remember("alice", "diet", "vegan", reason="told us on Aug 14")
    guard.check(CheckRequest("Book a vegetarian table", entity_ids=["alice"]))

The ``check``/``check_l2``/``init_db``/``upsert_fact`` functions from v0.1
are kept for the STALE harness and existing code.
"""
from .models import (CheckRequest, CheckResponse, Decision, Evidence, EvidenceKind,
                     FactVersion)
from .store import FactStore, InMemoryFactStore, SQLiteFactStore
from .resolvers import (EntityResolver, ExplicitEntityResolver, HeuristicEntityResolver,
                        ResolvedEntity)
from .extractors import (ACCOUNT_STATE, PERSON_PROFILE, AttributeExtractor,
                         KeywordAttributeExtractor)
from .normalize import SynonymTable
from .verifier import LLMVerifier, Verifier, VerifierResult
from .implications import (CachedImplicationProvider, Implication, ImplicationProvider,
                           ImplicationSet, LLMImplicationProvider, StaticImplicationProvider)
from .store import ImplicationStore
from .guard import StateGuard

# v0.1 API (legacy)
from .db import init_db, upsert_fact, get_current_fact, get_supersession_context, list_facts
from .layer1 import check, Status, InterceptResult, Conflict
from .layer2 import check_l2

__version__ = "0.3.0"

__all__ = [
    "StateGuard", "CheckRequest", "CheckResponse", "Decision", "Evidence", "EvidenceKind",
    "FactVersion", "FactStore", "InMemoryFactStore", "SQLiteFactStore",
    "EntityResolver", "ExplicitEntityResolver", "HeuristicEntityResolver", "ResolvedEntity",
    "AttributeExtractor", "KeywordAttributeExtractor", "PERSON_PROFILE", "ACCOUNT_STATE",
    "SynonymTable", "Verifier", "LLMVerifier", "VerifierResult",
    "ImplicationProvider", "LLMImplicationProvider", "StaticImplicationProvider",
    "CachedImplicationProvider", "Implication", "ImplicationSet", "ImplicationStore",
    # legacy
    "init_db", "upsert_fact", "get_current_fact", "get_supersession_context", "list_facts",
    "check", "Status", "InterceptResult", "Conflict", "check_l2",
]
