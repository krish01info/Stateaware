"""
StateGuard – Person C
Ablation table printer + evaluation scorer.

Usage
-----
# Print a table from one results file
python harness/evaluate.py harness/results/20240814T*.json

# Compare two modes (L1 vs L1+L2) side by side
python harness/evaluate.py harness/results/*l1_summary.json harness/results/*l1l2_summary.json
"""

import json
import sys
from pathlib import Path


def load_summary(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_table(summaries: list[dict]):
    """Print a GitHub-flavoured markdown ablation table."""
    # Collect all keys
    all_keys = sorted({k for s in summaries for k in s["summary"]})
    dim_order = {"SR": 0, "PR": 1, "IPA": 2}
    all_keys.sort(key=lambda k: (k.split("_")[0], dim_order.get(k.split("_")[1], 9)))

    modes = [s["mode"] for s in summaries]

    # Header
    header = "| Metric (Type_Dim) |" + "".join(f" {m} Acc% | {m} Lat ms |" for m in modes)
    sep    = "| --- |" + "".join(" ---: | ---: |" for _ in modes)
    print(header)
    print(sep)

    for key in all_keys:
        row = f"| {key} |"
        for s in summaries:
            entry = s["summary"].get(key)
            if entry:
                row += f" {entry['accuracy']} | {entry['avg_latency_ms']} |"
            else:
                row += " — | — |"
        print(row)

    print()
    # Token cost comparison
    print("### Token Cost")
    cost_header = "| Mode | Dim | Total Tokens |"
    cost_sep    = "| --- | --- | ---: |"
    print(cost_header)
    print(cost_sep)
    for s in summaries:
        for key, entry in sorted(s["summary"].items()):
            _, dim = key.split("_", 1)
            print(f"| {s['mode']} | {dim} | {entry['total_tokens']:,} |")


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <summary.json> [<summary2.json> ...]")
        sys.exit(1)

    paths     = sys.argv[1:]
    summaries = []
    for p in paths:
        # Support glob-like patterns via pathlib
        matched = list(Path(".").glob(p)) if "*" in p else [Path(p)]
        for m in matched:
            try:
                summaries.append(load_summary(m))
                print(f"Loaded: {m}")
            except Exception as e:
                print(f"Warning: could not load {m}: {e}")

    if not summaries:
        print("No valid summary files found.")
        sys.exit(1)

    print()
    print_table(summaries)


if __name__ == "__main__":
    main()
