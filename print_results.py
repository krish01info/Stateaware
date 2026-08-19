"""Print a comparison table from all harness summary JSON files."""
import json, glob, os

files = sorted(glob.glob("harness/results/*_summary.json"))
print(f"{'File':<45} {'Mode':<10} {'Key':<15} {'N':>5} {'Acc%':>7} {'Lat_ms':>8} {'Tokens':>8}")
print("-" * 100)
for f in files:
    data = json.load(open(f))
    mode = data["mode"]
    for key, s in data.get("summary", {}).items():
        print(f"{os.path.basename(f):<45} {mode:<10} {key:<15} {s['n']:>5} {s['accuracy']:>6.1f}% {s['avg_latency_ms']:>8.1f} {s['total_tokens']:>8}")

# Also check l1l2 for l1_resolved stats
import csv
l1l2_csvs = sorted(glob.glob("harness/results/*_l1l2_results.csv"))
for csvf in l1l2_csvs:
    rows = list(csv.DictReader(open(csvf, encoding="utf-8")))
    total = len(rows)
    resolved = sum(1 for r in rows if r.get("l1_resolved") == "True")
    print(f"\nL1+L2 hit-rate ({os.path.basename(csvf)}):")
    print(f"  Total rows: {total}")
    print(f"  L1 resolved: {resolved} ({resolved/total*100:.1f}%)")
    print(f"  Fell to L2:  {total - resolved} ({(total-resolved)/total*100:.1f}%)")
