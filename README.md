<h1 align="center">🛡️ StateGuard</h1>

<p align="center">
  <b>Pre-hoc Interception of Stale-Memory Queries in AI Agents</b><br>
  <i>A two-layer pipeline that catches outdated premises <em>before</em> the LLM generates a wrong answer.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/dataset-STALE-orange" alt="STALE Dataset">
</p>

---

## 🧠 What is StateGuard?

Long-running AI agents accumulate user facts over time (*"Alice is vegetarian"*).
When facts change (*"Alice is now vegan"*), downstream queries that assume the old
state produce incorrect responses — a problem called **stale-memory hallucination**.

StateGuard intercepts queries **before** they reach the LLM, using a two-layer
architecture:

| Layer | Type | Speed | Token Cost |
|-------|------|-------|------------|
| **Layer 1** | Deterministic keyword + DB lookup | < 1ms | 0 |
| **Layer 2** | Cheap LLM (Groq/Gemini/OpenAI) with structured context | ~800ms | ~150–300 |

> **Key insight**: Layer 2 receives a tiny *structured supersession record*
> (e.g. `old=vegetarian → new=vegan`), NOT the full conversation history.
> This is why a small cheap model outperforms frontier models searching 150K-token haystacks.

## 📁 Project Structure

```
StateGuard/
├── stateguard/
│   ├── __init__.py          # Package exports
│   ├── db.py                # SQLite fact store with supersession logging
│   ├── layer1.py            # Deterministic entity/attribute interceptor
│   └── layer2.py            # LLM semantic validator (Groq/Gemini/OpenAI/mock)
├── harness/
│   ├── run_stale.py         # STALE dataset evaluation runner
│   └── evaluate.py          # Ablation table printer
├── test_db.py               # DB smoke tests
├── test_layer1.py           # Layer 1 smoke tests
├── test_layer2.py           # Layer 2 smoke tests (mock backend)
├── requirements.txt
├── LICENSE
└── README.md
```

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run unit tests

```bash
python test_db.py
python test_layer1.py
python test_layer2.py
```

### 3. Run the STALE benchmark

```bash
# Layer 1 only (no API key needed, instant)
python harness/run_stale.py --mode l1 --dims dim2_query

# Full pipeline with Groq (free, fast)
export GROQ_API_KEY="your-key-here"
python harness/run_stale.py --mode l1l2 --provider groq --dims dim2_query

# Limit to N rows for quick testing
python harness/run_stale.py --mode l1l2 --provider groq --dims dim2_query --limit 20
```

### 4. Generate ablation table

```bash
python harness/evaluate.py harness/results/*_l1_summary.json harness/results/*_l1l2_summary.json
```

## ⚙️ Supported LLM Backends

| Provider | Model | Env Variable | Cost |
|----------|-------|-------------|------|
| `groq` | llama-3.3-70b-versatile | `GROQ_API_KEY` | Free tier |
| `gemini` | gemini-2.5-flash-lite | `GOOGLE_API_KEY` | Free tier |
| `openai` | gpt-4o-mini | `OPENAI_API_KEY` | Paid |
| `mock` | Deterministic stub | None | Free |

## 📊 Evaluation Modes

| Mode | Description |
|------|-------------|
| `l1` | Layer 1 only — pure deterministic lookup, no LLM calls |
| `l1l2` | Full pipeline — Layer 1 + Layer 2 LLM validation |
| `baseline` | Raw LLM without StateGuard (for comparison) |

## 🔬 STALE Dataset Dimensions

| Dimension | Label | Description |
|-----------|-------|-------------|
| `dim1_query` | SR | Simple retrieval — straightforward fact lookup |
| `dim2_query` | PR | Premise-dependent retrieval — query contains stale premise |
| `dim3_query` | IPA | Inference-dependent post-state analysis |

## 🏗️ Architecture

```
User Query
    │
    ▼
┌──────────────┐     PASS/UNKNOWN
│   Layer 1    │─────────────────────► LLM generates answer
│ (Keyword DB) │
└──────┬───────┘
       │ CONFLICT / UNCERTAIN
       ▼
┌──────────────┐     PASS (false alarm)
│   Layer 2    │─────────────────────► LLM generates answer
│ (Cheap LLM)  │
└──────┬───────┘
       │ CONFLICT (confirmed stale)
       ▼
  Corrected Query ──────────────────► LLM generates correct answer
```

## 📄 Citation

If you use StateGuard in your research, please cite:

```bibtex
@software{stateguard2026,
  title={StateGuard: Pre-hoc Interception of Stale-Memory Queries in AI Agents},
  year={2026},
  url={https://github.com/your-username/StateGuard}
}
```

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.
