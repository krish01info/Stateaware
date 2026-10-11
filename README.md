<h1 align="center">🛡️ Stateaware</h1>

<p align="center">
  <b>Pre-hoc interception of stale-state premises in AI agents</b><br>
  <i>Catch a request that assumes an outdated fact <em>before</em> the model answers it.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/version-0.5.0-informational" alt="v0.2.0">
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

**4. Write-time implications: pay the LLM per fact change, not per query.**
Layer 1 can only match words it knows. *"Paneer tikka recipe for Maya"*,
after Maya went vegan, names no stored value, so it slips through. With an
implication provider configured, `remember()` makes **one** cheap LLM call
when a value changes and stores a short list of phrases next to that fact
version:

| List | Meaning | Example |
|---|---|---|
| `conflicts_with_current` | things the current value rules out | vegan → "paneer", "ghee"; role=viewer → "add new users" |
| `only_fits_previous` | things that only make sense under an earlier value | Basic → Premium: "get rid of the ads", "can't watch in 4K" |

At query time those phrases are matched like stored values: no LLM call,
same ~2 ms, same framing caps (*"a vegan substitute for paneer"* passes).
They are tied to the fact version, so they disappear the moment the value
changes again. Facts change far less often than they are queried, so this
moves model cost from the hot path to the write path.

```python
from stateguard import StateGuard
from stateguard.implications import LLMImplicationProvider, CachedImplicationProvider

guard = StateGuard(implications=CachedImplicationProvider(
    LLMImplicationProvider("groq"), "implications.json"))
guard.remember("maya", "diet", "vegetarian")
guard.remember("maya", "diet", "vegan")          # one LLM call here
guard.check("Paneer tikka recipe for Maya", ["maya"])   # no LLM call; BLOCK/REVIEW
guard.write_tokens                                # write-time cost so far
```

The entity id is never sent to the model; only attribute and values are. The
cache is keyed by attribute + values, so the same change is generated once
across all users and every experiment can be re-run offline. Failed
generations are recorded in `guard.write_errors` and simply leave the fact
without implications.

**5. Learning facts from what users say.** Real agents get messages, not
clean `(attribute, value)` pairs. `guard.observe(user, message)` asks a
cheap model to extract current, first-hand facts and records them, so a
changed value supersedes the old one automatically.

```python
from stateguard import StateGuard, LLMFactExtractor

guard = StateGuard(fact_extractor=LLMFactExtractor("groq", cache_path="extractions.json"))
guard.observe("me", "I'm vegetarian, I love paneer")
guard.observe("me", "Update: went fully vegan last month")   # diet: vegetarian -> vegan
guard.check("Book me a vegetarian dinner", ["me"])           # BLOCK
```

The extractor is shown the attributes already on record and must reuse
them, so "diet" never splits into "diet" and "dietary_preference" (which
would hide every change). Values are kept to short canonical forms that a
later request can actually contain. Questions, hypotheticals, other people's
facts and ended states are ignored, and extractions below `min_confidence`
are returned in `skipped` instead of being written.

**6. Checking agent actions.** Stale memory does the most damage when an
agent *acts* on it. `check_tool_call` checks a tool call's arguments before
it runs and returns corrected arguments on BLOCK:

```python
r = guard.check_tool_call("book_flight", {"origin": "Bengaluru", "date": "2026-11-02"},
                          ["me"], arg_map={"origin": "location"})
r.decision             # BLOCK
r.suggested_arguments  # {"origin": "Pune", "date": "2026-11-02"}
```

Arguments are linked to attributes through `arg_map`, through the argument
name ("city", "address", "plan_tier"), or by value alone with lower scores.
Nested dicts, lists and free-text arguments are all checked, and every use
of a confirmed stale value in the call is corrected.

**7. A chat wrapper that asks instead of guessing.** `GuardedChat` wraps
any chat model (a callable taking OpenAI-style messages):

* **BLOCK**: the user's text is sent **unchanged**, with a short system
  *memory note* ("your diet is now 'vegan' (since 2026-07-02) ... answer for
  the current state and say what you assumed"). Nothing is rewritten
  behind the user's back.
* **REVIEW**: the wrapper asks one clarifying question ("Quick check: you
  mentioned vegetarian, but I have your diet as vegan since 2026-07-02.
  Should I go with vegan?") and never asks twice in a row. The answer is
  learned through `observe()`, and the original request is re-checked and
  answered.

See [`examples/agent_and_chat.py`](examples/agent_and_chat.py).

**8. Hinglish.** Framing cues include common Hinglish ("pehle", "ab
nahi", "ki jagah", "agle hafte", "abhi bhi"), and negation placed after the
word ("vegetarian **nahi** hai") is handled.

**9. Entity resolution is separate and visible.** Hosts should pass
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
| `ImplicationProvider` | `LLMImplicationProvider`, `StaticImplicationProvider`, `CachedImplicationProvider` | curated domain rules |

Value variants go in a `SynonymTable` (`{"NYC": "New York", "WFH": "work from home"}`).

## Evaluation

A single accuracy number rewards a system that blocks everything. The
harness reports a balanced scorecard instead ([`harness/metrics.py`](harness/metrics.py)):
BLOCK precision, recall and false-positive rate, each **strict** (only BLOCK
counts) and **lenient** (BLOCK or REVIEW counts), plus review rate, coverage,
verifier error rate, latency and tokens.

### General state-change suite

70 cases over 8 domains (person profile, subscription, shipping address,
account permission, device status, policy revision, product price,
appointment) plus multi-update chains. Neutral cases are adversarial: they
mention the entity, the old value, or words a stale request would use,
without assuming the old state.

```bash
python harness/run_general.py                          # Layer 1, entity IDs supplied
python harness/run_general.py --resolver heuristic     # entity must be guessed
python harness/run_general.py --implications groq      # + write-time implications (cached)
python harness/run_general.py --implications cache-only  # rerun offline from the cache
python harness/run_general.py --provider groq          # + Layer-2 verifier
```

Layer 1 only, CPU, no LLM:

| Setting | BLOCK prec. | Recall strict / lenient | FPR strict / lenient | Review rate | p50 latency |
|---|---|---|---|---|---|
| Entity IDs supplied | 0.90 | 0.59 / 0.62 | 0.05 / 0.21 | 0.10 | ~2 ms |
| Entity guessed from text | 1.00 | 0.28 / 0.59 | 0.00 / 0.13 | 0.21 | ~2 ms |
| IDs supplied, implicit → REVIEW | 0.90 | 0.59 / 0.69 | 0.05 / 0.29 | 0.17 | ~2 ms |

By category (IDs supplied): explicit stale premises 17/18 strict, 18/18
lenient; multi-update chains 2/2; **implicit premises 0/12**. That 0/12 is
the baseline write-time implications are meant to fix. Run
`--implications groq` and report the change in implicit recall *and* in the
`implicit_adversarial` false-positive rate side by side, with write-time
tokens (`write.tokens` in the summary JSON) next to query-time tokens.

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
python harness/run_stale.py --mode guard --implications groq --ipa-label stale
python harness/run_stale.py --mode guard --seed messages --use-ner   # the real-life path
python harness/run_stale.py --mode baseline --provider gemini
```

Each run writes per-row CSV plus a summary JSON with the balanced scorecard,
split by STALE type and dimension. PR queries count as stale and SR queries
as neutral. IPA queries never name the old state but only have a correct
answer under the new one, so whether an interceptor should flag them is a
framing choice: `--ipa-label neutral` (default) or `stale`. Report both.
Seeding the store and generating implications are excluded from query
latency and reported as write-time cost.

`--seed messages` is the setting closest to deployment. Facts are learned
from STALE's raw memory messages with `observe()`, exactly as an agent would
learn them; STALE's answer-bearing `explanation` field is never read. With
`--use-ner`, the user's own "I/my" resolves to them, as it would for any
assistant that knows who it is talking to.

### Evaluation protocol: STALE is a test, not a training signal

To keep results honest and transferable to real use:

1. Every rule, cue list and threshold is fixed **before** STALE is run, and
   is developed only on the general fixtures and real usage.
2. STALE is run once per configuration. If a STALE result prompts a change,
   the change has to be justified by a non-STALE failure and the run is
   reported as a new version, not a replacement.
3. Every summary JSON includes a `manifest`: git commit, whether the tree
   was dirty, library version, every decision constant and the run
   arguments. Report the commit hash with each number in the paper.

## Project layout

```
stateguard/
  models.py      CheckRequest, CheckResponse, Decision, Evidence, FactVersion
  store.py       FactStore protocol + in-memory and SQLite adapters (versioned, scoped)
  resolvers.py   entity resolution, separate from contradiction checking
  normalize.py   tokenising, money/synonym/plural normalisation, span-level value matching
  extractors.py  attribute cue packs + premise framing
  verifier.py    Layer-2 structured verifier (version-chain grounding, strict JSON)
  implications.py write-time implication providers (LLM, static, cached)
  extraction.py  learning facts from user messages (StateGuard.observe)
  tools.py       checking agent tool calls before they run
  chat.py        GuardedChat wrapper, clarifying questions, memory notes
  guard.py       StateGuard decision engine
  db.py, layer1.py, layer2.py   v0.1 API, kept for the original ablations
harness/
  metrics.py     balanced scorecard
  run_general.py general fixtures runner
  run_stale.py   STALE runner (modes: l1, l1l2, guard, baseline)
  fixtures/general_state_cases.jsonl
examples/basic_check.py, examples/agent_and_chat.py
```

## Tests

```bash
pip install -e .[dev]
pytest -q
```

## Known limitations

- **Implicit premises need implications or the verifier.** Without either,
  the deterministic layer only sees stored values and cue words (0/12 on the
  dev set). Implications are only as good as the phrases the model lists,
  and users may phrase things differently.
- **Old values used for someone else.** "Ask an admin to add users for
  Ravi's team" mentions Ravi's old role about a different person and is
  wrongly blocked.
- **Word collisions.** A stored value that is an ordinary phrase ("30 days")
  can match unrelated text. Attribute cues and the generic-word rule reduce
  this but do not remove it.
- **Framing cues cover English and some Hinglish only.** They are small
  hand-written lists in `extractors.py`, so they can be audited and extended.

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
