#!/usr/bin/env python3
"""Plot a BigANN/CONDA-style run per timestep (one timestep = one insert or delete op),
like the CONDA paper's figures: Recall@10 over time and update latency over time."""
import csv, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def f(v):
    try: return float(v)
    except (TypeError, ValueError): return None

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
print(f"{'run':28s} {'steps':>6} {'mean recall':>12} {'final recall':>13} {'mean update s':>14} {'max update s':>13}")
for path in sys.argv[1:]:
    rows = list(csv.DictReader(open(path)))
    bw = next((k for k in rows[0] if k.startswith("recall@10_bw100")), None) or \
         next(k for k in rows[0] if k.startswith("recall@10"))
    t, rx, ry, ux, uy = 0, [], [], [], []
    for r in rows:
        upd = [f(r.get(c)) for c in ("append_s", "delete_s", "consolidate_s")]
        if f(r.get("append_s")) is not None or f(r.get("delete_s")) is not None:
            t += 1
            ux.append(t); uy.append(sum(x for x in upd if x))
        if f(r.get(bw)) is not None and t > 0:
            rx.append(t); ry.append(f(r[bw]))
    lab = os.path.splitext(os.path.basename(path))[0]
    ax[0].plot(rx, ry, lw=1.2, label=lab); ax[1].plot(ux, uy, lw=1, label=lab)
    print(f"{lab:28s} {t:>6} {sum(ry)/len(ry):>12.4f} {ry[-1]:>13.4f} {sum(uy)/len(uy):>14.4f} {max(uy):>13.4f}")
ax[0].set(title=f"Recall@10 over timestep ({bw.split('_')[-1]})", xlabel="Timestep (insert/delete op)", ylabel="Recall@10")
ax[1].set(title="Update latency per op (insert or delete + consolidate)", xlabel="Timestep", ylabel="seconds")
for a in ax: a.grid(alpha=0.3); a.legend(fontsize=8)
fig.tight_layout()
out = os.path.join(os.path.dirname(sys.argv[1]) or ".", "timesteps.png")
fig.savefig(out, dpi=150); print("saved", out)
