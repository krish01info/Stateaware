# StateGuard General-Purpose Library Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Turn StateGuard from a STALE-benchmark-specific prototype into a reusable state-consistency library that detects outdated or contradicted assumptions across arbitrary entity types and domains, while minimizing unnecessary intervention.

**Architecture:** Separate reusable library logic from benchmark adapters. The library receives a query, an entity resolver, and a versioned fact store; it emits a calibrated decision (`pass`, `review`, `block`) with evidence. Benchmark-specific parsing, dataset seeding, and scoring remain in `harness/`.

**Tech Stack:** Python 3.10+, SQLite reference adapter, dataclasses/protocols, pytest, optional OpenAI-compatible LLM providers.

---

## Problem to solve

The current harness measures whether a query should be intercepted for the STALE dataset. That is useful, but it is not a sufficient product metric. A general library must handle:

- changing facts: location, role, plan, subscription, preferences, availability;
- arbitrary entities: people, accounts, orders, devices, documents, policies;
- explicit and implicit assumptions in a query;
- uncertain matches without turning every uncertain query into a block;
- a safe downstream action: allow, ask a clarification, rewrite, or block.

Do not optimize raw benchmark accuracy alone. A model that blocks everything can score well on stale-premise examples while being unusable in production.

## Product behavior

Use three decisions instead of treating every detection as a binary stale/not-stale result:

| Decision | Meaning | Typical action |
|---|---|---|
| `PASS` | No reliable contradiction or stale assumption | Continue normally |
| `REVIEW` | Relevant state exists but the premise/entity/value is ambiguous | Ask a clarifying question or run an LLM verifier |
| `BLOCK` | Strong evidence that the query assumes a superseded value | Explain current state and optionally propose a corrected query |

This reduces unwanted IPA-style intervention: neutral or future-oriented queries should normally be `PASS`; uncertainty should be `REVIEW`, not a false `BLOCK`.

## Success metrics

Track these by domain, entity type, and query category:

- Precision of `BLOCK`: how often blocked queries really rely on an outdated fact.
- Recall of stale-premise detection: how many true stale queries are caught.
- False-positive rate on neutral/current/future queries: the key guardrail against over-interception.
- Review rate: fraction of requests that need LLM or user clarification.
- Correction fidelity: whether a proposed correction preserves intent and uses the current fact.
- Coverage: fraction of queries for which the library can resolve an entity and relevant state.
- Latency and cost: Layer 1 time, Layer 2 time, LLM tokens, and parse-error rate.

Report a balanced scorecard, not one aggregate “accuracy” number.

---

### Task 1: Define a public, benchmark-independent API

**Objective:** Make the library usable without knowing anything about the STALE dataset.

**Files:**
- Create: `stateguard/models.py`
- Modify: `stateguard/layer1.py`
- Test: `test_models.py`

**Step 1: Write failing tests**

Test a request with `query`, optional `entity_ids`, and metadata; test an output containing `decision`, `confidence`, `evidence`, `current_facts`, and optional `suggested_query`.

**Step 2: Run the test to verify failure**

Run: `python -m pytest test_models.py -q`

**Step 3: Implement minimal models**

Create dataclasses/enums such as:

```python
class Decision(str, Enum):
    PASS = "pass"
    REVIEW = "review"
    BLOCK = "block"

@dataclass
class CheckRequest:
    query: str
    entity_ids: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class Evidence:
    entity_id: str
    attribute: str
    assumed_value: str | None
    current_value: str
    previous_value: str | None
    source: str

@dataclass
class CheckResponse:
    decision: Decision
    confidence: float
    evidence: list[Evidence]
    suggested_query: str | None = None
    reason: str | None = None
```

**Step 4: Run the test to verify pass**

Run: `python -m pytest test_models.py -q`

---

### Task 2: Introduce a versioned fact-store interface

**Objective:** Decouple the library from the current SQLite schema.

**Files:**
- Create: `stateguard/store.py`
- Modify: `stateguard/db.py`
- Test: `test_store.py`

**Step 1: Write failing tests**

Test that the interceptor works against an in-memory fake store and the SQLite adapter.

**Step 2: Implement protocols**

Define a `FactStore` protocol with operations to retrieve current facts, prior versions, and search facts for entities. Keep `SQLiteFactStore` as the initial adapter.

**Step 3: Add fact metadata**

Each fact version should include `entity_id`, `attribute`, `value`, `valid_from`, `valid_to`, `source`, `confidence`, and optional `scope` (for example region, tenant, policy version).

**Step 4: Verify**

Run: `python -m pytest test_store.py -q`

---

### Task 3: Separate entity resolution from fact contradiction

**Objective:** Improve generality and prevent unknown entity names from silently bypassing Layer 2.

**Files:**
- Create: `stateguard/resolvers.py`
- Modify: `stateguard/layer1.py`
- Test: `test_resolvers.py`, `test_layer1.py`

**Step 1: Add resolver protocol**

```python
class EntityResolver(Protocol):
    def resolve(self, query: str, candidates: list[str] | None = None) -> list[ResolvedEntity]: ...
```

**Step 2: Provide two adapters**

- `ExplicitEntityResolver`: for callers that already know the entity ID.
- `HeuristicEntityResolver`: the existing regex behavior, clearly documented as a fallback.

**Step 3: Define uncertainty correctly**

An unresolved or ambiguous entity should yield `REVIEW` only when the query contains a state-relevant premise. Otherwise use `PASS`; do not silently call it a successful non-stale result.

**Step 4: Verify**

Add tests for person, account, device, and unknown entity examples.

---

### Task 4: Generalize premise extraction

**Objective:** Replace hard-coded diet/location/job vocabularies with extensible extractors.

**Files:**
- Create: `stateguard/extractors.py`
- Modify: `stateguard/layer1.py`
- Test: `test_extractors.py`

**Step 1: Keep deterministic exact matching first**

Match actual current and historical stored values, including normalized variants. This is inexpensive and auditable.

**Step 2: Add pluggable attribute extractors**

Implement a protocol for domain adapters. Start with generic text/value matching plus current specialized aliases for person-profile data.

**Step 3: Normalize before comparing**

Use case folding, whitespace normalization, punctuation normalization, controlled synonym tables, and type-aware normalization (dates, currencies, IDs, locations). Avoid aggressive stemming as the only semantic mechanism.

**Step 4: Verify false-positive controls**

Test that a word appearing in unrelated prose does not create a contradiction.

---

### Task 5: Make Layer 2 a structured verifier, not the default decision maker

**Objective:** Use an LLM only for ambiguous cases and make failures observable.

**Files:**
- Modify: `stateguard/layer2.py`
- Modify: `test_layer2.py`

**Step 1: Write failing tests**

Test JSON-mode requests, malformed output handling, and that parse errors yield `REVIEW` rather than silently becoming `PASS`.

**Step 2: Enforce structured output where provider supports it**

Use the provider’s JSON/schema response mode. Validate with a typed schema before returning a decision.

**Step 3: Change failure behavior**

A parse error, provider failure, or unavailable model must return `REVIEW` with `provider_error` evidence. It must never be counted as a verified non-stale query.

**Step 4: Add calibrated confidence thresholds**

- high deterministic evidence → `BLOCK`;
- weak/ambiguous evidence → Layer 2;
- Layer 2 below threshold or malformed → `REVIEW`;
- no relevant state evidence → `PASS`.

**Step 5: Verify**

Run: `python -m pytest test_layer2.py -q`

---

### Task 6: Redesign evaluation around balanced behavior

**Objective:** Prevent a single STALE benchmark from steering library design.

**Files:**
- Create: `harness/metrics.py`
- Modify: `harness/run_stale.py`
- Create: `harness/fixtures/general_state_cases.jsonl`
- Test: `test_metrics.py`

**Step 1: Add unambiguous metrics**

Compute confusion matrix, precision, recall, F1, false-positive rate, review rate, coverage, parse-error rate, latency, and tokens.

**Step 2: Score every dimension intentionally**

Keep SR/PR/IPA as separate categories. Define expected behavior for each, rather than aggregating them into one number.

**Step 3: Add general fixtures**

Create representative changed-state examples outside STALE: subscription renewal, plan tier, shipping address, account permission, device status, policy revision, product price, and appointment availability.

**Step 4: Add adversarial neutral queries**

Examples that mention an entity but do not assume the old value. These are required false-positive tests.

**Step 5: Verify**

Run: `python -m pytest test_metrics.py -q`

---

### Task 7: Package the library cleanly

**Objective:** Make StateGuard installable and easy to integrate.

**Files:**
- Modify: `pyproject.toml`
- Modify: `stateguard/__init__.py`
- Modify: `README.md`
- Create: `examples/basic_check.py`

**Step 1: Export stable public APIs**

Export `StateGuard`, `CheckRequest`, `CheckResponse`, `Decision`, and adapter protocols from `stateguard`.

**Step 2: Provide a facade**

```python
guard = StateGuard(store=SQLiteFactStore(conn), resolver=ExplicitEntityResolver())
result = guard.check(CheckRequest(
    query="Can I renew the old basic plan?",
    entity_ids=["account_123"],
))
```

**Step 3: Document integration and safety**

Document that `BLOCK` is for high-confidence contradictions, `REVIEW` is for uncertainty, and callers choose whether review becomes a clarification prompt, audit event, or manual approval.

**Step 4: Verify packaging**

Run: `uv build` and `python -m pytest -q`.

---

## Decisions to make before implementation

1. What should the library do on `REVIEW`: ask the user, log only, or call a configured LLM?
2. Which first non-person domain matters most: subscriptions, account settings, e-commerce orders, or policy/document versioning?
3. Is the library intended to be an in-process Python package only, or should it expose a REST API later?
4. Is it acceptable to send query/fact data to a third-party LLM provider, or must there be a local-only mode?

## Recommended first release

Build a Python package with:

- versioned facts;
- explicit entity IDs supplied by the host application;
- deterministic current/previous-value matching;
- `PASS` / `REVIEW` / `BLOCK` decisions;
- optional JSON-enforced LLM verifier;
- transparent evidence and metrics;
- two domains: person profile and subscription/account state.

Do not claim general semantic reasoning or optimize a single “accuracy” number until the balanced evaluation suite exists.
