#!/usr/bin/env python3
"""
3D UMAP of each dataset, to check whether the clusters/islands seen in the 2D pictures are real
or an artifact of squashing to 2D.

  python3 topology/umap3d.py                 # every dataset that is present
  python3 topology/umap3d.py wikipedia sift  # some

For each dataset (same sample as the 2D plots: 20K base points spread over file order + 1K queries):
  topology/results/umap3d/<name>.html     interactive 3D plot (drag to rotate, scroll to zoom)
  topology/results/umap3d/<name>.png      three fixed views of the 3D embedding, side by side
and one summary table, topology/results/umap3d/faithfulness.csv:
  trust_2d / trust_3d   trustworthiness (0..1) of the 2D and 3D embeddings: how many of each point's
                        nearest neighbors in the picture are also true neighbors. Higher = more faithful.
  km_orig / km_2d / km_3d  share of variance explained by 10 k-means clusters in the original
                        space vs in the 2D / 3D pictures. If the pictures show much stronger clusters
                        than the original data, the clusters are (partly) a UMAP artifact.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from fingerprint import mips_to_l2, read_bin  # noqa: E402

DATASETS = [  # name, base, queries, metric
    ("sift", "data/candor/sift/data_1000000_128", "data/candor/sift/queries_10000_128", "l2"),
    ("bigann", "data/bigann/base.1M.u8bin", "data/bigann/query.u8bin", "l2"),
    ("deep", "data/deep/base.1M.fbin", "data/deep/query.fbin", "l2"),
    ("msturing", "data/msturing/base.1M.fbin", "data/msturing/query.fbin", "l2"),
    ("msspacev", "data/msspacev/base.1M.i8bin", "data/msspacev/query.i8bin", "l2"),
    ("text2image", "data/text2image/base.1M.fbin", "data/text2image/query.fbin", "ip"),
    ("wikipedia", "data/wikipedia-1M/base.1M.fbin", "data/wikipedia-1M/query.5K.fbin", "ip"),
]
OUT = "topology/results/umap3d"
N_BASE, N_QUERY, N_TRUST = 20000, 1000, 3000


def km_var(x, k=10, seed=0):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(x)
    total = ((x - x.mean(0)) ** 2).sum()
    return 1.0 - km.inertia_ / total


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import umap
    from sklearn.manifold import trustworthiness

    want = set(sys.argv[1:])
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for name, bpath, qpath, metric in DATASETS:
        if want and name not in want:
            continue
        if not (os.path.exists(bpath) and os.path.exists(qpath)):
            print(f"-- skip {name} (files not found)")
            continue
        base = read_bin(bpath, 1_000_000)
        queries = read_bin(qpath, N_QUERY)
        if metric == "ip":
            base, queries = mips_to_l2(base, queries)
        idx = np.linspace(0, len(base) - 1, N_BASE).astype(int)
        pts = np.vstack([base[idx], queries]).astype(np.float32)
        nb = len(idx)
        print(f"[{name}] UMAP 2D + 3D on {len(pts)} points ...", flush=True)
        e2 = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(pts)
        e3 = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(pts)

        rng = np.random.default_rng(0)
        sub = rng.choice(nb, min(N_TRUST, nb), replace=False)       # data points only
        row = {"dataset": name,
               "trust_2d": trustworthiness(pts[sub], e2[sub], n_neighbors=10),
               "trust_3d": trustworthiness(pts[sub], e3[sub], n_neighbors=10),
               "km_orig": km_var(pts[:nb]), "km_2d": km_var(e2[:nb]), "km_3d": km_var(e3[:nb])}
        rows.append(row)
        print("  " + "  ".join(f"{k} {v:.3f}" for k, v in row.items() if k != "dataset"), flush=True)

        # interactive HTML
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=e3[:nb, 0], y=e3[:nb, 1], z=e3[:nb, 2], mode="markers", name="data",
                                   marker=dict(size=1.5, color=idx, colorscale="Viridis", opacity=0.6,
                                               colorbar=dict(title="file position"))))
        fig.add_trace(go.Scatter3d(x=e3[nb:, 0], y=e3[nb:, 1], z=e3[nb:, 2], mode="markers", name="queries",
                                   marker=dict(size=2.5, color="red")))
        fig.update_layout(title=f"{name}: 3D UMAP (drag to rotate)", margin=dict(l=0, r=0, t=40, b=0),
                          scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))
        fig.write_html(os.path.join(OUT, f"{name}.html"), include_plotlyjs="cdn")

        # three static views
        fig2 = plt.figure(figsize=(13, 4.2))
        for i, (elev, azim) in enumerate([(20, 30), (20, 150), (80, 0)]):
            ax = fig2.add_subplot(1, 3, i + 1, projection="3d")
            ax.scatter(e3[:nb, 0], e3[:nb, 1], e3[:nb, 2], c=idx, cmap="viridis", s=0.5, alpha=0.5, linewidths=0)
            ax.scatter(e3[nb:, 0], e3[nb:, 1], e3[nb:, 2], c="red", s=2, alpha=0.8, linewidths=0)
            ax.view_init(elev=elev, azim=azim)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_title(f"view {i + 1}", fontsize=9)
        fig2.suptitle(f"{name}: 3D UMAP from three angles (red = queries)", fontsize=11)
        fig2.tight_layout(); fig2.savefig(os.path.join(OUT, f"{name}.png"), dpi=130); plt.close(fig2)
        print(f"  saved {OUT}/{name}.html and {name}.png", flush=True)

    if rows:
        path = os.path.join(OUT, "faithfulness.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
            w.writerows([{k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows])
        print(f"\n{'dataset':12s} {'trust2D':>8s} {'trust3D':>8s} {'clusters: orig':>15s} {'2D':>6s} {'3D':>6s}")
        for r in rows:
            print(f"{r['dataset']:12s} {r['trust_2d']:8.3f} {r['trust_3d']:8.3f} {r['km_orig']:15.3f} {r['km_2d']:6.3f} {r['km_3d']:6.3f}")
        print(f"saved {path}")


if __name__ == "__main__":
    main()
