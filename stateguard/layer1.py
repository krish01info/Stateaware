"""
StateGuard – Person B, Layer 1
Fast entity/attribute lookup against the SQLite fact store.
No LLM is used here — pure deterministic matching.

Flow
----
query (str)
    → extract_mentions()   : find (entity, attribute) pairs in the query
    → lookup_conflicts()   : compare against DB current facts
    → InterceptResult      : PASS | CONFLICT | UNCERTAIN

Public API
----------
check(conn, query) -> InterceptResult
"""

import re
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .db import get_current_fact, get_supersession_context

# ---------------------------------------------------------------------------
# Outcome types
# ---------------------------------------------------------------------------

class Status(str, Enum):
    PASS      = "PASS"       # query premise matches DB — let it through
    CONFLICT  = "CONFLICT"   # query premise contradicts a known current fact
    UNCERTAIN = "UNCERTAIN"  # entity/attribute found in DB but value unclear → send to Layer 2
    UNKNOWN   = "UNKNOWN"    # entity/attribute not in DB at all → let through


@dataclass
class Conflict:
    entity:        str
    attribute:     str
    queried_value: str   # what the query assumes
    current_value: str   # what the DB says is true


@dataclass
class InterceptResult:
    status:    Status
    conflicts: list[Conflict] = field(default_factory=list)
    raw_query: str = ""

    # Filled in by Layer 2 when status == UNCERTAIN or CONFLICT
    corrected_query: Optional[str] = None
    tokens_used: int = 0
    # True if Layer 2's LLM response failed to parse as JSON and the result
    # below is a fail-safe default (is_stale=False), NOT a genuine model
    # verdict. Kept separate so callers/harnesses don't silently conflate
    # "confirmed not stale" with "we couldn't tell". See layer2._parse_response.
    parse_error: bool = False

    def __repr__(self):
        return (
            f"InterceptResult(status={self.status.value}, "
            f"conflicts={len(self.conflicts)}, "
            f"corrected_query={self.corrected_query!r})"
        )


# ---------------------------------------------------------------------------
# Attribute keyword map
# Maps surface words in a query → canonical attribute name stored in the DB.
# Extend this dict as your fact store grows.
# ---------------------------------------------------------------------------

ATTRIBUTE_ALIASES: dict[str, str] = {
    # diet / food
    "diet":         "diet",
    "dietary":      "diet",
    "eating":       "diet",
    "food":         "diet",
    "vegan":        "diet",
    "vegetarian":   "diet",
    "keto":         "diet",
    "halal":        "diet",
    "kosher":       "diet",
    "meal":         "diet",
    "recipe":       "diet",
    "restaurant":   "diet",
    "cuisine":      "diet",
    # location
    "location":     "location",
    "city":         "location",
    "lives in":     "location",
    "living in":    "location",
    "based in":     "location",
    "moved to":     "location",
    "neighborhood":  "location",
    "commute":      "location",
    "nearby":       "location",
    "local":        "location",
    "area":         "location",
    # job / role / work
    "job":          "job",
    "works as":     "job",
    "occupation":   "job",
    "role":         "job",
    "profession":   "job",
    "career":       "job",
    "office":       "job",
    "remote":       "job",
    "work from":    "job",
    "workplace":    "job",
    "coworker":     "job",
    # age
    "age":          "age",
    "years old":    "age",
    "birthday":     "age",
    # language
    "language":     "language",
    "speaks":       "language",
    # relationship status
    "relationship": "relationship_status",
    "married":      "relationship_status",
    "single":       "relationship_status",
    "dating":       "relationship_status",
    "partner":      "relationship_status",
    "spouse":       "relationship_status",
    "engaged":      "relationship_status",
    # hobby / interests
    "hobby":        "hobby",
    "hobbies":      "hobby",
    "interest":     "hobby",
    "passion":      "hobby",
    "free time":    "hobby",
    "weekend":      "hobby",
    # pet
    "pet":          "pet",
    "dog":          "pet",
    "cat":          "pet",
    # education
    "school":       "education",
    "university":   "education",
    "college":      "education",
    "degree":       "education",
    "studying":     "education",
    "major":        "education",
    # health / fitness
    "health":       "health",
    "fitness":      "health",
    "exercise":     "health",
    "gym":          "health",
    "workout":      "health",
    # schedule / routine
    "schedule":     "schedule",
    "routine":      "schedule",
    "morning":      "schedule",
    "evening":      "schedule",
    "commuting":    "schedule",
    # travel
    "travel":       "travel",
    "trip":         "travel",
    "vacation":     "travel",
    "flight":       "travel",
}

# Known values for direct matching
DIET_VALUES = {"vegan", "vegetarian", "keto", "halal", "kosher",
               "omnivore", "carnivore", "pescatarian", "gluten-free",
               "paleo", "low-carb", "dairy-free"}

LOCATION_VALUES = {
    "seattle", "austin", "new york", "san francisco", "los angeles",
    "chicago", "boston", "denver", "portland", "miami", "houston",
    "atlanta", "dallas", "phoenix", "philadelphia", "detroit",
    "minneapolis", "nashville", "raleigh", "charlotte",
    "london", "paris", "tokyo", "berlin", "toronto", "sydney",
}

JOB_VALUES = {
    "engineer", "developer", "designer", "teacher", "nurse",
    "doctor", "lawyer", "manager", "analyst", "consultant",
    "freelancer", "remote", "hybrid", "in-office", "wfh",
    "work from home", "office-based",
}

RELATIONSHIP_VALUES = {
    "married", "single", "dating", "engaged", "divorced", "widowed",
}


# ---------------------------------------------------------------------------
# Entity extraction  (simple heuristic — swap in spaCy NER for production)
# ---------------------------------------------------------------------------

# Matches "Alice", "Alice's", "for Alice", "to Bob" etc.
_NAME_RE = re.compile(
    r"\b([A-Z][a-z]{1,20})(?:'s)?\b"
)

# Words that look like names but are almost certainly not person names
_STOPWORDS = {
    "I", "The", "A", "An", "This", "That", "It", "He", "She", "They",
    "We", "You", "My", "His", "Her", "Their", "Our", "Can", "Could",
    "Would", "Should", "What", "When", "Where", "Who", "Why", "How",
    "Is", "Are", "Was", "Were", "Do", "Does", "Did", "Make", "Please",
    "Tell", "Give", "Show", "Book", "Order", "Find", "Help", "Plan",
    "New",   # e.g. "New York"
    "Will", "Let", "Some", "But", "Just", "Also", "Want", "Need",
    "Like", "Think", "Know", "Get", "Got", "Each", "Every",
    "May", "Might", "Must", "Been", "Have", "Has", "Had",
    "Not", "All", "Any", "Most", "More", "Very",
}


def extract_entity_names(query: str) -> list[str]:
    """
    Return candidate person-entity names found in *query*.
    Heuristic: capitalised single words that are not common stop-words.
    """
    return [
        m.group(1)
        for m in _NAME_RE.finditer(query)
        if m.group(1) not in _STOPWORDS
    ]


def extract_attribute(query: str) -> Optional[str]:
    """
    Return the first attribute keyword found in *query*, or None.
    Checks multi-word phrases before single words to avoid partial matches.
    """
    q_lower = query.lower()
    # Sort by length descending so "lives in" matches before "in".
    # Use word-boundary regex to avoid substring collisions
    # (e.g. "age" inside "manage", "pet" inside "carpet").
    for alias in sorted(ATTRIBUTE_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", q_lower):
            return ATTRIBUTE_ALIASES[alias]
    return None


def extract_queried_value(query: str, attribute: str) -> Optional[str]:
    """
    Best-effort extraction of the value the query *assumes* for *attribute*,
    using the small closed vocabularies below as a fast, dependency-free path.

    NOTE: this only covers ~4 attributes (diet/location/job/relationship) with
    a hand-picked set of values. It intentionally returns None for anything
    outside those lists (e.g. any hobby/pet/education/health value, or a city
    not in LOCATION_VALUES) so that `check()` can fall back to
    `match_against_known_values()`, which checks the entity's *actual* stored
    values instead of a fixed vocabulary. Prefer extending that fallback over
    growing these lists further — hardcoded value sets don't generalize.
    """
    q_lower = query.lower()
    if attribute == "diet":
        for v in DIET_VALUES:
            if v in q_lower:
                return v
    elif attribute == "location":
        for v in LOCATION_VALUES:
            if v in q_lower:
                return v
    elif attribute == "job":
        for v in JOB_VALUES:
            if v in q_lower:
                return v
    elif attribute == "relationship_status":
        for v in RELATIONSHIP_VALUES:
            if v in q_lower:
                return v
    return None


def match_against_known_values(
    query: str,
    current_value: Optional[str],
    old_value: Optional[str],
) -> Optional[str]:
    """
    Dynamic fallback for extract_queried_value(): instead of checking the
    query against a global hardcoded vocabulary, check it against the
    *specific* values already on record for this entity/attribute (the
    current DB value and, if it exists, the superseded old value).

    This generalizes to arbitrary attributes/values — custom cities, job
    titles, hobbies, pets, whatever's actually in the DB — without needing
    to hand-enumerate every possible value up front.

    Returns the matched value string (from current_value or old_value),
    or None if neither appears in the query.
    """
    q_lower = query.lower()
    candidates = [v for v in (current_value, old_value) if v]
    # Check the longer value first to avoid a short value matching inside
    # a longer one that also appears (e.g. "engineer" inside "senior engineer").
    for val in sorted(candidates, key=len, reverse=True):
        if val.lower() in q_lower:
            return val
    return None


# ---------------------------------------------------------------------------
# Core check function
# ---------------------------------------------------------------------------

def check(
    conn: sqlite3.Connection,
    query: str,
    known_entity: Optional[str] = None,
) -> InterceptResult:
    """
    Run Layer-1 interception on *query*.

    Parameters
    ----------
    conn         : open DB connection (from db.init_db)
    query        : the raw user query string
    known_entity : if the calling system already knows which entity the
                   conversation is about, pass it here to skip NER.

    Returns
    -------
    InterceptResult with status PASS | CONFLICT | UNCERTAIN | UNKNOWN
    """
    result = InterceptResult(status=Status.UNKNOWN, raw_query=query)

    # 1. Identify entities
    entities = [known_entity] if known_entity else extract_entity_names(query)
    if not entities:
        return result  # No entity found -> UNKNOWN

    # 2. Identify attribute
    attribute = extract_attribute(query)
    if not attribute:
        # Even without a known attribute, try word-overlap detection
        # against all facts for the entity
        for entity in entities:
            _overlap = _word_overlap_check(conn, entity, query)
            if _overlap:
                result.status = Status.UNCERTAIN
                return result
        return result  # No known attribute keyword -> UNKNOWN

    conflicts: list[Conflict] = []
    uncertain = False

    for entity in entities:
        db_fact = get_current_fact(conn, entity, attribute)
        if db_fact is None:
            continue

        current_value = db_fact["value"]
        supersession = get_supersession_context(conn, entity, attribute)
        old_value = supersession.get("old_value") if supersession else None

        # 1. Try matching the query against the entity's own current/old
        #    values first — this is more precise than the generic vocab
        #    below, and avoids a bug where a generic word like "engineer"
        #    matches inside a longer current value ("staff platform
        #    reliability engineer") and gets wrongly compared as unequal.
        queried_value = match_against_known_values(query, current_value, old_value)

        # 2. Fallback: fixed vocabulary (diet/location/job/relationship only)
        if queried_value is None:
            queried_value = extract_queried_value(query, attribute)

        if queried_value is None:
            # Still nothing concrete — last resort: loose word-overlap
            # against the OLD value only (weak signal, routes to Layer 2).
            if supersession:
                if _has_word_overlap(query, old_value or "", current_value):
                    uncertain = True
                    continue
            uncertain = True
            continue

        if queried_value.lower() == current_value.lower():
            result.status = Status.PASS
        else:
            conflicts.append(
                Conflict(
                    entity=entity,
                    attribute=attribute,
                    queried_value=queried_value,
                    current_value=current_value,
                )
            )

    if conflicts:
        result.status = Status.CONFLICT
        result.conflicts = conflicts
    elif uncertain:
        result.status = Status.UNCERTAIN
    elif result.status == Status.UNKNOWN and entities:
        result.status = Status.UNKNOWN

    return result


# ---------------------------------------------------------------------------
# Word-overlap detection helpers
# ---------------------------------------------------------------------------

_SUFFIXES = ("ism", "ing", "ers", "er", "ed", "es", "s")

def _stem(word: str) -> str:
    """
    Very light suffix stripping so near-forms overlap, e.g.
    'vegetarianism' ~ 'vegetarian', 'engineering' ~ 'engineer'.
    Not a real stemmer (no Porter rules) — just enough to catch the
    common cases without extra dependencies. Only strips if the
    remaining stem is still >= 3 chars, to avoid over-stripping short words.
    """
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def _tokenize(text: str) -> set[str]:
    """Simple word tokenizer: lowercase, alpha-only, 3+ chars, light-stemmed."""
    return {_stem(w) for w in re.findall(r'[a-z]{3,}', text.lower())}

_COMMON_WORDS = {
    "the", "and", "that", "this", "with", "from", "for", "are", "was",
    "were", "been", "have", "has", "had", "will", "would", "could",
    "should", "can", "may", "might", "not", "but", "also", "just",
    "about", "into", "over", "some", "any", "all", "your", "their",
    "what", "when", "where", "who", "how", "which", "there", "here",
    "than", "then", "very", "more", "most", "other", "each", "every",
    "like", "know", "think", "make", "want", "need", "get", "got",
}

def _has_word_overlap(query: str, old_value: str, current_value: str) -> bool:
    """
    Returns True if the query shares content words with the OLD value
    but NOT with the current value -> likely stale premise.
    """
    q_words   = _tokenize(query)   - _COMMON_WORDS
    old_words  = _tokenize(old_value) - _COMMON_WORDS
    curr_words = _tokenize(current_value) - _COMMON_WORDS

    overlap_old  = q_words & old_words
    overlap_curr = q_words & curr_words

    # If query shares words with old but not current -> suspicious
    return len(overlap_old) >= 1 and len(overlap_curr) == 0


def _word_overlap_check(conn, entity: str, query: str) -> bool:
    """
    Check all facts for entity to see if query has word overlap
    with any superseded value.
    """
    from .db import list_facts
    facts = list_facts(conn, entity)
    for fact in facts:
        attr = fact["attribute"]
        supersession = get_supersession_context(conn, entity, attr)
        if supersession:
            old_val = supersession.get("old_value", "")
            if _has_word_overlap(query, old_val, fact["value"]):
                return True
    return False
