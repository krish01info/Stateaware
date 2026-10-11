<h1 align="center">🛡️ Stateaware</h1>

<p align="center">
  <b>Pre-hoc interception of stale-state premises in AI agents</b><br>
  <i>Catch a request that assumes an outdated fact <em>before</em> the model answers it.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/version-0.2.0-informational" alt="v0.2.0">
</p>

---

## The problem

Agents remember things: *Maya is vegetarian*, *account 123 is on Basic*,
*pump 3 is online*. Then the world changes: Maya goes vegan, the account
upgrades, the pump goes down for maintenance. A request written against the
old state (*"book Maya a vegetarian thali"*, *"does Basic include 4K?"*,
*"pump 3 is online, schedule it"*) quietly produces a wrong answer or a
wrong action.

Most existing fixes act **after** generation: they repair the draft answer or
reconsolidate memory. StateGuard acts **before** generation. It checks the
request against a versioned record of state and returns one of three
decisions, with the evidence behind it.

| Decision | Meaning | What the host usually does |
|---|---|---|
| `PASS` | No reliable stale or contradicted assumption | Send the request on unchanged |
| `REVIEW` | Relevant state changed, but the premise is ambiguous | Ask the user, run the verifier, or log |
| `BLOCK` | Strong evidence the request assumes a superseded value | Use `suggested_query`, or tell the user what changed |

## Quick start

```bash
pip install -e .            # core has no dependencies
pip install -e .[groq]      # optional Layer-2 verifier (or [openai], [gemini])
```

```python
from stateguard import StateGuard, CheckRequest, Decision, SQLiteFactStore

guard = StateGuard(store=SQLiteFactStore("facts.db"))

guard.remember("account_123", "plan", "Basic")
guard.remember("account_123", "plan", "Premium", reason="upgraded in settings")

r = guard.check(CheckRequest("Does the Basic plan include 4K?", entity_ids=["account_123"]))
r.decision         # Decision.BLOCK
r.suggested_query  # 'Does the Premium plan include 4K?'
r.reason           # "query assumes plan='Basic' for account_123, but it is now 'Premium'"
r.evidence         # auditable list: which value, how many versions back, why it scored
```

See [`examples/basic_check.py`](examples/basic_check.py) for a runnable version.

## How it decides

```
request ─► resolve entity ─► for each attribute that has CHANGED:
                               • find mentions of current / superseded values
                               • find attribute cues ("plan", "address", ...)
                               • read the request's framing (contrast, history,
                                 negation, future, direct question)
                             ─► score each piece of evidence in [0, 1]
                             ─► ≥ 0.75 BLOCK · 0.35–0.75 verifier or REVIEW · < 0.35 PASS
```

**1. Versioned state, not overwritten state.** Recording a new value closes
the old version rather than deleting it. The engine therefore recognises a
request that assumes a value from *any* earlier version (Chennai → Bengaluru →
Pune: a question about Chennai beaches is caught), and does not flag a value
that has come back (Delhi → Mumbai → Delhi). Facts can carry `source`,
`confidence` and `scope`, so parallel truths (an EU and a US refund policy)
never supersede each other.

**2. Mentioning an old value is not the same as assuming it.** The core of the
deterministic layer is *premise framing*. A superseded value starts with a
high score, which is then capped by how the request frames it:

| Framing | Example | Cap |
|---|---|---|
| Negated | "she is **not** vegetarian" | 0.15 → PASS |
| Contrast / names the current value too | "moved from Basic **to Premium**" | 0.25–0.30 → PASS |
| Explicit history | "I **used to** live in Seattle" | 0.30 → PASS |
| Forward-looking | "**visiting** Seattle **next month**" | 0.45 → REVIEW |
| Asks about the state | "is Alice **still** vegetarian?" | 0.45 → REVIEW |
| Plain past tense | "she **lived** in Chennai" | 0.60 → REVIEW |
| Generic one-word value, no attribute cue | "my **monthly** usage report" | 0.65 → REVIEW |

All caps are named constants in [`stateguard/guard.py`](stateguard/guard.py)
so they can be reported and ablated.

**3. Layer 2 is a structured verifier, not the default decision maker.**
Only evidence in the ambiguous band is sent to an LLM. It gets the
attribute's **version chain** (every value, when it held, and why it
changed) instead of the raw conversation, typically a few hundred tokens.
Its JSON output is validated strictly. A provider error, unparseable output
or low confidence becomes `REVIEW` with `provider_error` evidence and is
**never** counted as "verified not stale". Any model can be plugged in with
`LLMVerifier(complete_fn=...)`, including a local one.

**4. Entity resolution is separate and visible.** Hosts should pass
`entity_ids`. Without them, `HeuristicEntityResolver` matches entity IDs and
aliases in the text, maps "my/I" to a configured `self_entity`, and otherwise
reports `UNRESOLVED_ENTITY`. An unresolved request is `REVIEW` only if it
carries a premise matching some entity's superseded value. Otherwise it is
`PASS`, explicitly marked as unchecked.

## Plugging in your own pieces

| Interface | Ship with | Implement your own for |
|---|---|---|
| `FactStore` | `InMemoryFactStore`, `SQLiteFactStore` | Postgres, Redis, an existing memory layer |
| `EntityResolver` | `ExplicitEntityResolver`, `HeuristicEntityResolver` | NER, session-based lookup |
| `AttributeExtractor` | `KeywordAttributeExtractor(PERSON_PROFILE, ACCOUNT_STATE)` | new domains: add a cue pack |
| `Verifier` | `LLMVerifier("groq" \| "openai" \| "gemini" \| complete_fn=...)` | rule engines, local models |

Value variants go in a `SynonymTable` (`{"NYC": "New York", "WFH": "work from home"}`).

## Evaluation

A single accuracy number rewards a system that blocks everything. The
harness reports a balanced scorecard instead ([`harness/metrics.py`](harness/metrics.py)):
BLOCK precision, recall and false-positive rate, each **strict** (only BLOCK
counts) and **lenient** (BLOCK or REVIEW counts), plus review rate, coverage,
verifier error rate, latency and tokens.

### General state-change suite

56 cases over 8 domains (person profile, subscription, shipping address,
account permission, device status, policy revision, product price,
appointment) plus multi-update chains. Neutral cases are adversarial: they
mention the entity or even the old value without assuming it.

```bash
python harness/run_general.py                       # Layer 1, entity IDs supplied
python harness/run_general.py --resolver heuristic  # entity must be guessed
python harness/run_general.py --provider groq       # + Layer-2 verifier
```

Layer 1 only, CPU, no LLM:

| Setting | BLOCK prec. | Recall strict / lenient | FPR strict / lenient | Review rate | p50 latency |
|---|---|---|---|---|---|
| Entity IDs supplied | 0.95 | 0.79 / 0.83 | 0.03 / 0.22 | 0.12 | ~2 ms |
| Entity guessed from text | 1.00 | 0.38 / 0.79 | 0.00 / 0.12 | 0.25 | ~2 ms |
| IDs supplied, implicit → REVIEW | 0.95 | 0.79 / 0.88 | 0.03 / 0.31 | 0.20 | ~2 ms |

What these say. Explicit stale premises are caught almost perfectly (18/18
lenient, 17/18 strict). **Implicit** premises (*"Maya wants a paneer tikka
recipe"* after she went vegan) are 0/4 without a verifier; that gap is exactly
what Layer 2 is for. Guessing the entity costs most of the strict recall, so
hosts should pass IDs.

> ⚠️ **These fixtures are a development set.** They were written alongside
> the engine, and some rules were tuned while looking at them. Do not report
> them as a final result. Have a teammate who has not read `guard.py` write a
> held-out set in the same JSONL format, and report on that.

### STALE benchmark

```bash
pip install -e .[eval,groq]
python harness/run_stale.py --mode l1                       # v0.1 Layer 1 (ablation)
python harness/run_stale.py --mode guard                    # v0.2 engine, Layer 1 only
python harness/run_stale.py --mode guard --provider groq    # v0.2 engine + verifier
python harness/run_stale.py --mode guard --use-ner          # entity resolved from text
python harness/run_stale.py --mode baseline --provider gemini
```

Each run writes per-row CSV plus a summary JSON with the balanced scorecard,
split by STALE type and dimension (PR queries count as stale; SR and IPA
queries count as neutral). Seeding the fact store is excluded from latency.

## Project layout

```
stateguard/
  models.py      CheckRequest, CheckResponse, Decision, Evidence, FactVersion
  store.py       FactStore protocol + in-memory and SQLite adapters (versioned, scoped)
  resolvers.py   entity resolution, separate from contradiction checking
  normalize.py   tokenising, money/synonym/plural normalisation, span-level value matching
  extractors.py  attribute cue packs + premise framing
  verifier.py    Layer-2 structured verifier (version-chain grounding, strict JSON)
  guard.py       StateGuard decision engine
  db.py, layer1.py, layer2.py   v0.1 API, kept for the original ablations
harness/
  metrics.py     balanced scorecard
  run_general.py general fixtures runner
  run_stale.py   STALE runner (modes: l1, l1l2, guard, baseline)
  fixtures/general_state_cases.jsonl
examples/basic_check.py
```

## Tests

```bash
pip install -e .[dev]
pytest -q
```

## Known limitations

- **Implicit premises need the verifier.** The deterministic layer only sees
  values and cue words.
- **Values the store has never held are invisible to Layer 1.** "Book a
  steakhouse" for a vegan is a contradiction the store cannot match
  deterministically.
- **Word collisions.** A stored value that is an ordinary phrase ("30 days")
  can match unrelated text. Attribute cues and the generic-word rule reduce
  this but do not remove it.
- **Framing cues are English-only.** They are small hand-written lists;
  they are listed in `extractors.py` so they can be audited and replaced.

## Citation

```bibtex
@software{stateaware2026,
  title  = {StateGuard: Pre-hoc Interception of Stale-State Premises in AI Agents},
  author = {Sharma, Krish},
  year   = {2026},
  url    = {https://github.com/krish01info/Stateaware}
}
```

## License

MIT. See [LICENSE](LICENSE).
