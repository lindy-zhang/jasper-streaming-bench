#!/usr/bin/env python3
"""From the batch-size sweep, estimate which batch sizes keep up with each of CANDOR's
event rates: a batch keeps up if it finishes before the next one arrives,
i.e. its insert throughput >= the event rate (queries and evaluation excluded)."""
import csv, glob, re
RATES = [10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]
print("% of insert batches that keep up, by batch size (rows) and CANDOR event rate (columns)")
print(f"{'batch':>7} {'median vec/s':>13} " + " ".join(f"{r // 1000:>6}K" for r in RATES))
paths = glob.glob("results/candor/sift_batch_[0-9]*.csv")
for p in sorted(paths, key=lambda s: int(re.findall(r"batch_(\d+)", s)[0])):
    bs = int(re.findall(r"batch_(\d+)", p)[0])
    v = sorted(float(r["append_vps"]) for r in csv.DictReader(open(p)) if r.get("append_vps"))
    med = v[len(v) // 2]
    cells = " ".join(f"{100 * sum(x >= r for x in v) / len(v):>6.0f}%" for r in RATES)
    print(f"{bs:>7,} {med:>13,.0f} {cells}")
