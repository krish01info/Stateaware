"""
StateGuard – Person A
SQLite fact store with supersede-with-reason logging.

Public API
----------
init_db(path)            → initialise schema, return connection
upsert_fact(conn, ...)   → insert or supersede a fact
get_current_fact(conn, ...) → latest value for an entity/attribute
get_supersession_context(conn, ...) → structured record for Layer-2 LLM
list_facts(conn, ...)    → all current facts for an entity
"""

import sqlite3
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS entities (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT    NOT NULL UNIQUE,
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS facts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id   INTEGER NOT NULL REFERENCES entities(id),
    attribute   TEXT    NOT NULL,
    value       TEXT    NOT NULL,
    valid_from  TEXT    NOT NULL,
    valid_until TEXT,                       -- NULL  = still current
    is_current  INTEGER NOT NULL DEFAULT 1, -- 1 = current, 0 = superseded
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS supersessions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    old_fact_id    INTEGER NOT NULL REFERENCES facts(id),
    new_fact_id    INTEGER NOT NULL REFERENCES facts(id),
    reason         TEXT,
    superseded_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Fast lookup indexes
CREATE INDEX IF NOT EXISTS idx_facts_entity_attr_current
    ON facts(entity_id, attribute, is_current);
"""


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_db(path: str = "stateguard.db") -> sqlite3.Connection:
    """
    Open (or create) the SQLite database at *path*, apply the schema,
    and return the connection.

    Usage
    -----
    conn = init_db()                  # default file
    conn = init_db(":memory:")        # in-memory for tests
    conn = init_db("custom/path.db")  # custom location
    """
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    migrate(conn)
    return conn


# Columns added for the general-purpose library (stateguard.store). Older
# databases are upgraded in place; legacy functions ignore the new columns.
_FACT_COLUMNS = {
    "source": "TEXT",
    "confidence": "REAL NOT NULL DEFAULT 1.0",
    "scope": "TEXT",
}


def migrate(conn: sqlite3.Connection) -> None:
    """Idempotently bring an existing database up to the current schema."""
    conn.executescript(_SCHEMA)
    have = {r[1] for r in conn.execute("PRAGMA table_info(facts)")}
    for col, decl in _FACT_COLUMNS.items():
        if col not in have:
            conn.execute(f"ALTER TABLE facts ADD COLUMN {col} {decl}")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS aliases ("
        " alias TEXT PRIMARY KEY, entity_name TEXT NOT NULL)"
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _get_or_create_entity(conn: sqlite3.Connection, name: str) -> int:
    """Return entity id, creating the row if it does not exist."""
    row = conn.execute(
        "SELECT id FROM entities WHERE name = ?", (name,)
    ).fetchone()
    if row:
        return row["id"]
    cur = conn.execute(
        "INSERT INTO entities (name) VALUES (?)", (name,)
    )
    conn.commit()
    return cur.lastrowid


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upsert_fact(
    conn: sqlite3.Connection,
    entity: str,
    attribute: str,
    value: str,
    reason: Optional[str] = None,
    valid_from: Optional[str] = None,
) -> int:
    """
    Insert a new fact for *entity/attribute*.
    If a current fact already exists it is superseded (logged with *reason*).

    Parameters
    ----------
    conn      : open DB connection
    entity    : human-readable entity name, e.g. ``"Alice"``
    attribute : fact key, e.g. ``"diet"``
    value     : new value, e.g. ``"vegan"``
    reason    : why the fact changed, e.g. ``"User updated on Aug 5"``
    valid_from: ISO-8601 string; defaults to now

    Returns
    -------
    id of the newly inserted fact row
    """
    entity_id = _get_or_create_entity(conn, entity)
    valid_from = valid_from or _now()

    # Find existing current fact (if any)
    old = conn.execute(
        """
        SELECT id FROM facts
        WHERE entity_id = ? AND attribute = ? AND is_current = 1
        """,
        (entity_id, attribute),
    ).fetchone()

    # Insert the new fact
    cur = conn.execute(
        """
        INSERT INTO facts (entity_id, attribute, value, valid_from, is_current)
        VALUES (?, ?, ?, ?, 1)
        """,
        (entity_id, attribute, value, valid_from),
    )
    new_fact_id = cur.lastrowid

    if old:
        old_fact_id = old["id"]
        now = _now()
        # Mark old fact as superseded
        conn.execute(
            """
            UPDATE facts
               SET is_current = 0,
                   valid_until = ?
             WHERE id = ?
            """,
            (now, old_fact_id),
        )
        # Log the supersession with reason
        conn.execute(
            """
            INSERT INTO supersessions (old_fact_id, new_fact_id, reason)
            VALUES (?, ?, ?)
            """,
            (old_fact_id, new_fact_id, reason),
        )

    conn.commit()
    return new_fact_id


def get_current_fact(
    conn: sqlite3.Connection,
    entity: str,
    attribute: str,
) -> Optional[dict]:
    """
    Return the current fact value for *entity/attribute*, or ``None``.

    Used by Layer 1 for fast O(1) lookup.

    Returns
    -------
    dict with keys: attribute, value, valid_from  — or None if not found
    """
    row = conn.execute(
        """
        SELECT f.attribute, f.value, f.valid_from
          FROM facts f
          JOIN entities e ON e.id = f.entity_id
         WHERE e.name = ? AND f.attribute = ? AND f.is_current = 1
        """,
        (entity, attribute),
    ).fetchone()
    return dict(row) if row else None


def get_supersession_context(
    conn: sqlite3.Connection,
    entity: str,
    attribute: str,
) -> Optional[dict]:
    """
    Return the full supersession record for *entity/attribute*.

    This is the structured grounding context fed to the Layer-2 cheap LLM
    instead of the raw 150K-token context haystack.

    Returns
    -------
    dict with keys: old_value, new_value, why_it_changed, superseded_at
    — or None if no supersession exists
    """
    row = conn.execute(
        """
        SELECT
            old_f.value          AS old_value,
            new_f.value          AS new_value,
            s.reason             AS why_it_changed,
            s.superseded_at
          FROM supersessions s
          JOIN facts old_f ON old_f.id = s.old_fact_id
          JOIN facts new_f ON new_f.id = s.new_fact_id
          JOIN entities e  ON e.id = new_f.entity_id
         WHERE e.name = ? AND new_f.attribute = ?
         ORDER BY s.superseded_at DESC
         LIMIT 1
        """,
        (entity, attribute),
    ).fetchone()
    return dict(row) if row else None


def list_facts(
    conn: sqlite3.Connection,
    entity: str,
) -> list[dict]:
    """
    Return all *current* facts for *entity* as a list of dicts.

    Useful for debugging and for building the full context snapshot.
    """
    rows = conn.execute(
        """
        SELECT f.attribute, f.value, f.valid_from
          FROM facts f
          JOIN entities e ON e.id = f.entity_id
         WHERE e.name = ? AND f.is_current = 1
         ORDER BY f.attribute
        """,
        (entity,),
    ).fetchall()
    return [dict(r) for r in rows]
