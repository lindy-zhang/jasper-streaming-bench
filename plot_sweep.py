#!/usr/bin/env python3
"""
Summarize and plot a batch-size sweep produced by run_sweep.sh.

  python3 plot_sweep.py results/sift1m_*.csv

Prints a summary table and saves results/sweep_summary.png with three panels:
  1. insert throughput (vectors/s) vs batch size, one line per pattern
  2. GPU utilization during append vs batch size
  3. recall over the stream at the middle beam width, one line per run
"""
import csv
import glob
import os
import re
import sys
import warnings

warnings.simplefilter("ignore", RuntimeWarning)  # blank columns (e.g. no GPU utilization) average to NaN

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    rows = list(csv.DictReader(open(path)))
    m = re.search(r"_(insert_only|sliding_window|random_delete)_bs(\d+)\.csv$", path)
    if not m:
        return None
    return {"pattern": m.group(1), "bs": int(m.group(2)), "rows": rows}


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def main():
    paths = sorted(p for a in sys.argv[1:] for p in glob.glob(a))
    runs = [r for r in (load(p) for p in paths) if r]
    if not runs:
        sys.exit("no matching CSVs (expected *_<pattern>_bs<N>.csv)")

    print(f"{'pattern':15s} {'batch':>7s} {'append vec/s':>13s} {'append util%':>13s} "
          f"{'consol. s':>10s} {'final recall':>13s}")
    summary = []
    for r in runs:
        stream = r["rows"][1:]
        vps = np.nanmedian([num(x.get("append_vps")) for x in stream])
        util = np.nanmean([num(x.get("append_gpu_util_mean")) for x in stream])
        cons = np.nanmean([num(x.get("consolidate_s")) for x in stream])
        rkeys = [k for k in r["rows"][0] if k.startswith("recall")]
        mid = rkeys[len(rkeys) // 2] if rkeys else None
        evals = [(int(x["batch"]), num(x[mid])) for x in r["rows"] if mid and x.get(mid)]
        summary.append((r["pattern"], r["bs"], vps, util, evals, mid))
        print(f"{r['pattern']:15s} {r['bs']:7d} {vps:13,.0f} {util:13.1f} {cons:10.3f} "
              f"{evals[-1][1] if evals else float('nan'):13.4f}")

    patterns = sorted({s[0] for s in summary})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for p in patterns:
        s = sorted([x for x in summary if x[0] == p], key=lambda x: x[1])
        ax[0].plot([x[1] for x in s], [x[2] for x in s], marker="o", label=p)
        ax[1].plot([x[1] for x in s], [x[3] for x in s], marker="o", label=p)
    for x in summary:
        if x[4]:
            ax[2].plot([e[0] for e in x[4]], [e[1] for e in x[4]], marker=".",
                       label=f"{x[0]} bs={x[1]}")
    ax[0].set(xscale="log", xlabel="Batch size", ylabel="Insert throughput (vectors/s)",
              title="Insert throughput vs batch size")
    ax[1].set(xscale="log", xlabel="Batch size", ylabel="GPU utilization during append (%)",
              title="GPU busy time during append", ylim=(0, 105))
    ax[2].set(xlabel="Batch", ylabel=summary[0][5] or "recall", title="Recall over the stream")
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    ax[2].legend(fontsize=6)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(paths[0]) or ".", "sweep_summary.png")
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
