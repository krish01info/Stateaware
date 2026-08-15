"""StateGuard – pre-hoc stale-memory interceptor."""
from .db import init_db, upsert_fact, get_current_fact, get_supersession_context, list_facts
from .layer1 import check, Status, InterceptResult, Conflict
from .layer2 import check_l2

__all__ = [
    # db
    "init_db",
    "upsert_fact",
    "get_current_fact",
    "get_supersession_context",
    "list_facts",
    # layer 1
    "check",
    "Status",
    "InterceptResult",
    "Conflict",
    # layer 2
    "check_l2",
]
