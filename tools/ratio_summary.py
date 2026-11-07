#!/usr/bin/env python3
"""Summarize run_ratio.sh: recall after inserting 10K vectors at once vs in batches of 100."""
import csv, glob, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def final_recall(path, bw):
    rows = [r for r in csv.DictReader(open(path)) if r.get(f"recall@10_bw{bw}")]
    return float(rows[0][f"recall@10_bw{bw}"]), float(rows[-1][f"recall@10_bw{bw}"])

runs = {}
for p in glob.glob("results/ratio/n*_bs*.csv"):
    n, bs = map(int, re.findall(r"n(\d+)_bs(\d+)", os.path.basename(p))[0])
    runs.setdefault(n, {})[bs] = p
sizes = sorted(n for n in runs if len(runs[n]) == 2)
new = max(max(runs[sizes[0]]), 1) if sizes else 0
print(f"Insert {new:,} new vectors into an index of N vectors (Recall@10)")
for bw in (32, 64):
    print(f"\nbeam width {bw}")
    print(f"{'index N':>9} {'batch/index':>12} {'before':>8} {'batches of 100':>15} {'one batch':>10} {'penalty':>8}")
    xs, pens = [], []
    for n in sizes:
        b0, small = final_recall(runs[n][100], bw)
        _, big = final_recall(runs[n][new], bw)
        print(f"{n:>9,} {new / n:>11.0%} {b0:>8.4f} {small:>15.4f} {big:>10.4f} {small - big:>+8.4f}")
        xs.append(new / n); pens.append(small - big)
    if bw == 64 and xs:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot([x * 100 for x in xs], pens, "o-")
        ax.set(xscale="log", xlabel="Batch size as % of index size (log)",
               ylabel="Recall@10 lost vs. batches of 100",
               title=f"Inserting {new:,} vectors at once: recall penalty")
        ax.axhline(0, color="gray", lw=0.8); ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig("results/ratio/ratio_penalty.png", dpi=150)
print("\nsaved results/ratio/ratio_penalty.png")
