#!/usr/bin/env python3
"""
Overlay several stream_bench.py runs (e.g. CANDOR batch sizes, or deletion levels).

  python3 plot_runs.py results/candor/sift_batch_*.csv
  python3 plot_runs.py results/candor/sift_*delete*.csv --beam-width 64

Prints a one-line summary per run and saves <first file's folder>/runs_<name>.png
with four panels over the stream: insert throughput, GPU utilization during
insert, recall, and GPU memory.
"""
import argparse
import csv
import glob
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.simplefilter("ignore", RuntimeWarning)


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="+")
    p.add_argument("--beam-width", type=int, default=64)
    p.add_argument("--name", default="", help="Suffix for the output image name")
    a = p.parse_args()
    paths = sorted(x for pat in a.csvs for x in glob.glob(pat))
    if not paths:
        raise SystemExit("no CSVs matched")

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    print(f"{'run':32s} {'batches':>7s} {'median insert vec/s':>20s} {'append util%':>13s} "
          f"{'keeps up':>9s} {'final recall':>13s} {'final mem GB':>13s}")
    for path in paths:
        rows = list(csv.DictReader(open(path)))
        label = os.path.splitext(os.path.basename(path))[0]
        ins = [r for r in rows if r.get("append_vps")]
        x = [num(r["inserted_total"]) for r in ins]
        vps = [num(r["append_vps"]) for r in ins]
        util = [num(r.get("append_gpu_util_mean")) for r in ins]
        rkey = next((k for k in rows[0] if k.startswith("recall") and k.endswith(f"bw{a.beam_width}")), None)
        ev = [(num(r["inserted_total"]), num(r[rkey])) for r in rows if rkey and r.get(rkey)]
        mem = [(num(r["inserted_total"]), num(r.get("gpu_mem_used_gb"))) for r in rows]
        keeps = [r.get("keeps_up") for r in ins if r.get("keeps_up")]
        keep_frac = f"{100 * sum(k == 'True' for k in keeps) / len(keeps):.0f}%" if keeps else "-"

        ax[0, 0].plot(x, vps, lw=1, label=label)
        ax[0, 1].plot(x, util, lw=1, label=label)
        if ev:
            ax[1, 0].plot([e[0] for e in ev], [e[1] for e in ev], marker="o", ms=3, label=label)
        ax[1, 1].plot([m[0] for m in mem], [m[1] for m in mem], lw=1, label=label)
        print(f"{label:32s} {len(ins):7d} {np.nanmedian(vps):20,.0f} {np.nanmean(util):13.1f} "
              f"{keep_frac:>9s} {ev[-1][1] if ev else float('nan'):13.4f} {mem[-1][1]:13.2f}")

    ax[0, 0].set(title="Insert throughput", xlabel="Vectors inserted so far", ylabel="vectors/s", yscale="log")
    ax[0, 1].set(title="GPU utilization during insert (NVML)", xlabel="Vectors inserted so far",
                 ylabel="%", ylim=(0, 105))
    ax[1, 0].set(title=f"Recall (beam width {a.beam_width})", xlabel="Vectors inserted so far",
                 ylabel="recall@k")
    ax[1, 1].set(title="GPU memory in use", xlabel="Vectors inserted so far", ylabel="GB")
    for axis in ax.flat:
        axis.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(paths[0]) or ".", f"runs{('_' + a.name) if a.name else ''}.png")
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
