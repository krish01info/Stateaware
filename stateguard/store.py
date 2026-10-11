"""
Versioned fact-store interface and two reference adapters.

The decision engine only talks to the :class:`FactStore` protocol, so a host
can back StateGuard with Postgres, Redis, an ORM or an existing memory layer
by implementing six small methods. Two adapters ship with the library:

* :class:`InMemoryFactStore` – dict-backed, for tests and small apps.
* :class:`SQLiteFactStore`   – persistent, shares the schema in :mod:`stateguard.db`,
  so facts written with the legacy ``upsert_fact`` are visible here too.

A fact is versioned: recording a new value for ``(entity, attribute, scope)``
closes the previous version (``valid_to`` set) instead of overwriting it.
The *full* history is kept, so the engine can recognise a query that
assumes a value from two or three updates ago, not only the last one.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Optional, Protocol, runtime_checkable

from .models import FactVersion


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _scope_ok(fact_scope: Optional[str], wanted: Optional[str]) -> bool:
    """A query scope sees its own facts plus unscoped (global) facts."""
    return fact_scope is None or wanted is None or fact_scope == wanted


@runtime_checkable
class FactStore(Protocol):
    """What the decision engine needs from a fact store."""

    def record(
        self,
        entity_id: str,
        attribute: str,
        value: str,
        *,
        reason: Optional[str] = None,
        valid_from: Optional[str] = None,
        source: Optional[str] = None,
        confidence: float = 1.0,
        scope: Optional[str] = None,
    ) -> FactVersion: ...

    def current(
        self, entity_id: str, attribute: Optional[str] = None, scope: Optional[str] = None
    ) -> list[FactVersion]: ...

    def history(
        self, entity_id: str, attribute: str, scope: Optional[str] = None
    ) -> list[FactVersion]:
        """All versions, newest first (index 0 is current if one exists)."""
        ...

    def entities(self) -> list[str]: ...

    def add_alias(self, alias: str, entity_id: str) -> None: ...

    def aliases(self) -> dict[str, str]:
        """Lower-cased alias -> entity_id."""
        ...


@runtime_checkable
class ImplicationStore(Protocol):
    """Optional extension: stores write-time implications per current fact.

    Stores that do not implement it still work; implications are then kept
    only in the StateGuard instance's memory.
    """

    def set_implications(self, entity_id: str, attribute: str, value: str,
                         phrases: list[tuple[str, str]], scope: Optional[str] = None) -> None: ...

    def get_implications(self, entity_id: str, attribute: str,
                         scope: Optional[str] = None) -> list[tuple[str, str]]:
        """``[(phrase, kind)]`` for the CURRENT version only; [] if it changed since."""
        ...


# ---------------------------------------------------------------------------
# In-memory adapter
# ---------------------------------------------------------------------------

class InMemoryFactStore:
    """Dict-backed :class:`FactStore`. Not thread-safe; fine for tests/apps."""

    def __init__(self) -> None:
        # (entity, attribute, scope) -> list[FactVersion] oldest..newest
        self._versions: dict[tuple[str, str, Optional[str]], list[FactVersion]] = {}
        self._aliases: dict[str, str] = {}
        self._entities: dict[str, None] = {}
        self._impl: dict[tuple, tuple[str, str, list[tuple[str, str]]]] = {}

    def record(self, entity_id, attribute, value, *, reason=None, valid_from=None,
               source=None, confidence=1.0, scope=None) -> FactVersion:
        key = (entity_id, attribute, scope)
        ts = valid_from or _now()
        chain = self._versions.setdefault(key, [])
        if chain and chain[-1].is_current:
            prev = chain[-1]
            if prev.value == value:
                return prev  # idempotent: same value is not a change
            chain[-1] = FactVersion(**{**prev.__dict__, "valid_to": ts})
        fv = FactVersion(entity_id=entity_id, attribute=attribute, value=value,
                         valid_from=ts, source=source, confidence=confidence,
                         scope=scope, reason=reason)
        chain.append(fv)
        self._entities.setdefault(entity_id)
        return fv

    def current(self, entity_id, attribute=None, scope=None) -> list[FactVersion]:
        out = []
        for (e, a, s), chain in self._versions.items():
            if e != entity_id or (attribute and a != attribute) or not _scope_ok(s, scope):
                continue
            if chain and chain[-1].is_current:
                out.append(chain[-1])
        return sorted(out, key=lambda f: f.attribute)

    def history(self, entity_id, attribute, scope=None) -> list[FactVersion]:
        out = []
        for (e, a, s), chain in self._versions.items():
            if e == entity_id and a == attribute and _scope_ok(s, scope):
                out.extend(chain)
        return sorted(out, key=lambda f: f.valid_from, reverse=True)

    def entities(self) -> list[str]:
        return list(self._entities)

    def add_alias(self, alias: str, entity_id: str) -> None:
        self._aliases[alias.lower()] = entity_id

    def aliases(self) -> dict[str, str]:
        return dict(self._aliases)

    def set_implications(self, entity_id, attribute, value, phrases, scope=None) -> None:
        cur = [f for f in self.current(entity_id, attribute) if f.scope == scope]
        if cur and cur[0].value == value:
            self._impl[(entity_id, attribute, scope)] = (value, cur[0].valid_from, list(phrases))

    def get_implications(self, entity_id, attribute, scope=None) -> list[tuple[str, str]]:
        out = []
        for f in self.current(entity_id, attribute, scope):
            rec = self._impl.get((entity_id, attribute, f.scope))
            if rec and rec[0] == f.value and rec[1] == f.valid_from:
                out += rec[2]
        return out


# ---------------------------------------------------------------------------
# SQLite adapter
# ---------------------------------------------------------------------------

class SQLiteFactStore:
    """Persistent :class:`FactStore` over the schema in :mod:`stateguard.db`.

    Pass an existing connection (e.g. from ``init_db``) or a path.
    """

    def __init__(self, conn_or_path: "sqlite3.Connection | str" = ":memory:") -> None:
        from .db import init_db

        if isinstance(conn_or_path, sqlite3.Connection):
            self.conn = conn_or_path
            self.conn.row_factory = sqlite3.Row
            from .db import migrate
            migrate(self.conn)
        else:
            self.conn = init_db(conn_or_path)

    # -- helpers -----------------------------------------------------------

    def _entity_pk(self, name: str, create: bool) -> Optional[int]:
        row = self.conn.execute("SELECT id FROM entities WHERE name = ?", (name,)).fetchone()
        if row:
            return row["id"]
        if not create:
            return None
        return self.conn.execute("INSERT INTO entities (name) VALUES (?)", (name,)).lastrowid

    _SELECT = """
        SELECT e.name AS entity_id, f.attribute, f.value, f.valid_from, f.valid_until,
               f.source, f.confidence, f.scope,
               (SELECT s.reason FROM supersessions s WHERE s.new_fact_id = f.id) AS reason
          FROM facts f JOIN entities e ON e.id = f.entity_id
    """

    @staticmethod
    def _row(r: sqlite3.Row) -> FactVersion:
        return FactVersion(
            entity_id=r["entity_id"], attribute=r["attribute"], value=r["value"],
            valid_from=r["valid_from"], valid_to=r["valid_until"], source=r["source"],
            confidence=r["confidence"] if r["confidence"] is not None else 1.0,
            scope=r["scope"], reason=r["reason"],
        )

    # -- protocol ----------------------------------------------------------

    def record(self, entity_id, attribute, value, *, reason=None, valid_from=None,
               source=None, confidence=1.0, scope=None) -> FactVersion:
        ts = valid_from or _now()
        pk = self._entity_pk(entity_id, create=True)
        old = self.conn.execute(
            "SELECT id, value FROM facts WHERE entity_id=? AND attribute=? "
            "AND is_current=1 AND scope IS ?", (pk, attribute, scope)).fetchone()
        if old and old["value"] == value:
            self.conn.commit()
            return self.current(entity_id, attribute, scope)[0]
        new_id = self.conn.execute(
            "INSERT INTO facts (entity_id, attribute, value, valid_from, is_current, "
            "source, confidence, scope) VALUES (?,?,?,?,1,?,?,?)",
            (pk, attribute, value, ts, source, confidence, scope)).lastrowid
        if old:
            self.conn.execute("UPDATE facts SET is_current=0, valid_until=? WHERE id=?",
                              (ts, old["id"]))
            self.conn.execute(
                "INSERT INTO supersessions (old_fact_id, new_fact_id, reason) VALUES (?,?,?)",
                (old["id"], new_id, reason))
        self.conn.commit()
        return FactVersion(entity_id=entity_id, attribute=attribute, value=value,
                           valid_from=ts, source=source, confidence=confidence,
                           scope=scope, reason=reason)

    def current(self, entity_id, attribute=None, scope=None) -> list[FactVersion]:
        sql = self._SELECT + " WHERE e.name = ? AND f.is_current = 1"
        args: list = [entity_id]
        if attribute:
            sql += " AND f.attribute = ?"
            args.append(attribute)
        if scope is not None:
            sql += " AND (f.scope IS NULL OR f.scope = ?)"
            args.append(scope)
        return [self._row(r) for r in self.conn.execute(sql + " ORDER BY f.attribute", args)]

    def history(self, entity_id, attribute, scope=None) -> list[FactVersion]:
        sql = self._SELECT + " WHERE e.name = ? AND f.attribute = ?"
        args: list = [entity_id, attribute]
        if scope is not None:
            sql += " AND (f.scope IS NULL OR f.scope = ?)"
            args.append(scope)
        sql += " ORDER BY f.is_current DESC, f.valid_from DESC, f.id DESC"
        return [self._row(r) for r in self.conn.execute(sql, args)]

    def entities(self) -> list[str]:
        return [r["name"] for r in self.conn.execute("SELECT name FROM entities ORDER BY id")]

    def add_alias(self, alias: str, entity_id: str) -> None:
        self._entity_pk(entity_id, create=True)
        self.conn.execute("INSERT OR REPLACE INTO aliases (alias, entity_name) VALUES (?, ?)",
                          (alias.lower(), entity_id))
        self.conn.commit()

    def aliases(self) -> dict[str, str]:
        return {r["alias"]: r["entity_name"]
                for r in self.conn.execute("SELECT alias, entity_name FROM aliases")}

    def _current_fact_id(self, entity_id, attribute, scope) -> Optional[int]:
        row = self.conn.execute(
            "SELECT f.id, f.value FROM facts f JOIN entities e ON e.id = f.entity_id "
            "WHERE e.name=? AND f.attribute=? AND f.is_current=1 AND f.scope IS ?",
            (entity_id, attribute, scope)).fetchone()
        return row

    def set_implications(self, entity_id, attribute, value, phrases, scope=None) -> None:
        row = self._current_fact_id(entity_id, attribute, scope)
        if not row or row["value"] != value:
            return
        self.conn.execute("DELETE FROM implications WHERE fact_id=?", (row["id"],))
        self.conn.executemany(
            "INSERT INTO implications (fact_id, phrase, kind) VALUES (?,?,?)",
            [(row["id"], p, k) for p, k in phrases])
        self.conn.commit()

    def get_implications(self, entity_id, attribute, scope=None) -> list[tuple[str, str]]:
        sql = ("SELECT i.phrase, i.kind FROM implications i JOIN facts f ON f.id = i.fact_id "
               "JOIN entities e ON e.id = f.entity_id "
               "WHERE e.name=? AND f.attribute=? AND f.is_current=1")
        args: list = [entity_id, attribute]
        if scope is not None:
            sql += " AND (f.scope IS NULL OR f.scope = ?)"
            args.append(scope)
        return [(r["phrase"], r["kind"]) for r in self.conn.execute(sql + " ORDER BY i.id", args)]
