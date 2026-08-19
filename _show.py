import json, glob, csv

# Summaries
for f in sorted(glob.glob("harness/results/*_summary.json")):
    d = json.load(open(f))
    mode = d["mode"]
    for k, v in d.get("summary", {}).items():
        print(f"| {mode:<10} | {k:<10} | {v['n']:>3} | {v['accuracy']:>5.1f}% | {v['avg_latency_ms']:>6.1f}ms | {v['total_tokens']:>6} |")

# L1 resolve rate
for f in sorted(glob.glob("harness/results/*_l1l2_results.csv")):
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    tot = len(rows)
    res = sum(1 for r in rows if r.get("l1_resolved") == "True")
    print(f"\nL1 hit-rate: {res}/{tot} = {res/tot*100:.1f}%")
