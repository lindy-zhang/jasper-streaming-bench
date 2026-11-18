#!/usr/bin/env python3
"""Summarize topology/results/fingerprints.csv: a table sorted by difficulty, scatter plots of the
metrics (real vs synthetic), and a grid of the UMAP/PCA pictures. If topology/results/recall.csv
(columns: dataset,recall) exists, also plots each metric against recall."""
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
R = sys.argv[1] if len(sys.argv) > 1 else "topology/results"
rows = list(csv.DictReader(open(os.path.join(R, "fingerprints.csv"))))
f = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else float("nan")
rows.sort(key=lambda r: f(r, "lid_query"))

cols = [("dataset", "{:<14}"), ("kind", "{:<9}"), ("dim", "{:>4}"), ("lid_query", "{:>6.1f}"),
        ("lid_base", "{:>6.1f}"), ("rc_query", "{:>6.2f}"), ("query_shift", "{:>6.2f}"),
        ("hub_skew", "{:>6.2f}"), ("antihub_frac", "{:>6.3f}"), ("pca90_frac", "{:>6.2f}"),
        ("km_var100", "{:>6.2f}"), ("km_sep100", "{:>6.2f}"), ("drift_locality", "{:>6.2f}"), ("dup_frac", "{:>6.3f}"),
        ("norm_cv", "{:>6.3f}")]
short = {"lid_query": "LIDq", "lid_base": "LIDb", "rc_query": "RC", "query_shift": "qshift",
         "hub_skew": "hub", "antihub_frac": "anti", "pca90_frac": "pca90", "km_var100": "km100",
         "km_sep100": "sep", "drift_locality": "drift", "norm_cv": "normCV", "dup_frac": "dups"}
print(" ".join(f"{short.get(c, c):>6}" if c not in ("dataset", "kind") else fmt.format(c) for c, fmt in cols))
for r in rows:
    out = []
    for c, fmt in cols:
        v = r.get(c, "")
        out.append(fmt.format(v if c in ("dataset", "kind") else (int(float(v)) if c == "dim" else f(r, c))))
    print(" ".join(out))

recall = {}
rp = os.path.join(R, "recall.csv")
if os.path.exists(rp):
    recall = {r["dataset"]: float(r["recall"]) for r in csv.DictReader(open(rp))}

colors = {"real": "#1F4E8C", "synthetic": "#DD8452", "reordered": "#55A868"}
pairs = [("lid_query", "rc_query"), ("lid_query", "hub_skew"), ("km_var100", "drift_locality")]
if recall:
    pairs = [("lid_query", "recall"), ("rc_query", "recall"), ("hub_skew", "recall")] + pairs
fig, axes = plt.subplots(1, len(pairs), figsize=(4.4 * len(pairs), 3.8))
for ax, (xk, yk) in zip(axes, pairs):
    for r in rows:
        y = recall.get(r["dataset"]) if yk == "recall" else f(r, yk)
        if y is None:
            continue
        ax.scatter(f(r, xk), y, color=colors.get(r["kind"], "gray"), s=30)
        ax.annotate(r["dataset"], (f(r, xk), y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set(xlabel=short.get(xk, xk), ylabel=short.get(yk, yk)); ax.grid(alpha=0.3)
handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in colors.items()]
axes[0].legend(handles=handles, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(R, "metrics_scatter.png"), dpi=150)
print(f"saved {R}/metrics_scatter.png")

imgs = sorted(glob.glob(os.path.join(R, "embed_*.png")))
if imgs:
    cols_n = 4
    fig, axes = plt.subplots((len(imgs) + cols_n - 1) // cols_n, cols_n, figsize=(4 * cols_n, 3.6 * ((len(imgs) + cols_n - 1) // cols_n)))
    for ax in getattr(axes, "flat", [axes]):
        ax.axis("off")
    for ax, p in zip(getattr(axes, "flat", [axes]), imgs):
        ax.imshow(plt.imread(p))
    fig.tight_layout(); fig.savefig(os.path.join(R, "embed_grid.png"), dpi=130)
    print(f"saved {R}/embed_grid.png")
