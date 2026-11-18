#!/usr/bin/env python3
"""
Dataset "fingerprints": geometry/topology metrics that predict how hard a dataset is for
graph-based ANN indexes. One row per dataset in topology/results/fingerprints.csv.

  python3 topology/fingerprint.py NAME BASE_FILE QUERY_FILE [--metric l2|ip] [options]

Files are BigANN-style binaries: [n uint32][d uint32] then n*d values; the element type is
taken from the extension (.fbin/.bin = float32, .u8bin = uint8, .i8bin = int8; override
with --dtype). Only the first --n-pool base vectors are read, in file order.

Distances are Euclidean; inner-product datasets are first transformed so that Euclidean
neighbors equal inner-product neighbors (see below).

Metrics
  lid_query / lid_base   Local intrinsic dimensionality, MLE (Levina-Bickel / Amsaleg et al.)
                         with k=32, the k the CONDA paper uses. Higher = harder.
  rc_query               Relative contrast: mean distance to a random base point divided by
                         the distance to the nearest neighbor. Near 1 = neighbors do not
                         stand out from the crowd = hard.
  query_shift            Mean 10-NN distance of queries / mean 10-NN distance of base points.
                         > 1 means queries lie off the data (out of distribution).
  hub_skew               Skewness of the k-occurrence N_10 (how often each point is someone's
                         10-NN) in a pool. High = a few "hub" points dominate graph links.
  antihub_frac           Fraction of points that are nobody's 10-NN (hard to reach in a graph).
  pca90 / pca90_frac     Dimensions needed for 90% of the variance (and as a fraction of d).
  km_var10 / km_var100   Share of variance explained by k-means with 10 / 100 clusters.
                         Random Gaussian data is near 0; clustered data is high.
  km_sep100              Mean distance to own centroid / distance to 2nd-nearest centroid
                         (100 clusters). Low = tight, well-separated clusters.
  drift_locality         For points in file (insertion) order, split into 10 chunks: share of
                         each point's 10-NN that come from its own chunk, divided by 0.1.
                         1 = order carries no geometry (shuffled / static); >1 = points that
                         arrive together are close together (drift, real time order).
  norm_cv                Coefficient of variation of vector norms (matters for inner product).
  dup_frac               Share of queries/base probes whose nearest neighbor is an exact
                         duplicate (distance 0); duplicates are skipped in LID and RC.

Inner-product datasets (--metric ip) are measured after the MIPS-to-L2 transform, i.e. in
the space where Euclidean neighbors equal inner-product neighbors.
"""
import argparse
import csv
import math
import os
import time

import numpy as np
import torch

DTYPES = {".fbin": np.float32, ".bin": np.float32, ".u8bin": np.uint8, ".i8bin": np.int8}


def read_bin(path, n_max=0, dtype=None):
    with open(path, "rb") as f:
        n, d = np.fromfile(f, dtype=np.uint32, count=2)
    n, d = int(n), int(d)
    if dtype is None:
        dtype = DTYPES.get(os.path.splitext(path)[1], np.float32)
    n_avail = (os.path.getsize(path) - 8) // (d * np.dtype(dtype).itemsize)
    n = min(n, n_avail, n_max) if n_max else min(n, n_avail)
    x = np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(n, d))
    return np.ascontiguousarray(x).astype(np.float32)


def sq_norms(x):
    return (x * x).sum(1)


def knn(q, x, k, exclude_self=False, chunk=0):
    """Exact k-NN (Euclidean) of rows of q among rows of x, on the GPU if present.
    Returns (distances, indices), distances sorted ascending, as torch tensors on q.device."""
    xn = sq_norms(x)
    chunk = chunk or max(1, int(2.5e8 // x.shape[0]))   # ~1 GB distance block
    out_d, out_i = [], []
    kk = k + 1 if exclude_self else k
    for s in range(0, q.shape[0], chunk):
        qq = q[s:s + chunk]
        d2 = sq_norms(qq)[:, None] + xn[None, :] - 2.0 * qq @ x.T
        d2.clamp_(min=0)
        if exclude_self:
            rows = torch.arange(qq.shape[0], device=q.device)
            d2[rows, s + rows] = float("inf")
        dv, iv = torch.topk(d2, kk, dim=1, largest=False)
        out_d.append(dv[:, :k].sqrt()); out_i.append(iv[:, :k])
    return torch.cat(out_d), torch.cat(out_i)


def lid_mle(dists):
    """Per-point LID from sorted k-NN distances: -(1/k' * sum log(d_i / d_k))^-1.
    Exact duplicates (distance 0) are skipped, so k' = number of nonzero neighbors; rows with
    fewer than 5 distinct-distance neighbors are dropped (see dup_frac)."""
    dk = dists[:, -1:]
    pos = dists > 1e-3 * dk          # float rounding makes exact duplicates ~1e-3, not 0
    logs = torch.where(pos, torch.log(dists.clamp(min=1e-30) / dk.clamp(min=1e-30)), torch.zeros_like(dists))
    cnt = pos[:, :-1].sum(1)
    s = logs[:, :-1].sum(1) / cnt.clamp(min=1)
    lid = -1.0 / s.clamp(max=-1e-12)
    keep = (cnt >= 5) & (dk[:, 0] > 0)
    lid = lid[keep]
    return lid[torch.isfinite(lid)]


def first_nonzero(dists):
    """Distance to the nearest neighbor that is not an exact duplicate (inf if none)."""
    d = torch.where(dists > 1e-3 * dists[:, -1:], dists, torch.full_like(dists, float("inf")))
    return d.min(1).values


def mips_to_l2(base, queries):
    """Inner-product data -> Euclidean space with the same neighbor ranking (as in the Jasper
    harness): x -> [x, sqrt(M^2-|x|^2)], q -> [q * M/|q|, 0]. Without this, L2 neighbors of the
    raw vectors are unrelated to inner-product neighbors (overlap ~0.001 on Wikipedia)."""
    bn = np.linalg.norm(base, axis=1)
    m = float(bn.max()) * 1.0001
    b2 = np.hstack([base, np.sqrt(np.clip(m * m - bn * bn, 0, None))[:, None]]).astype(np.float32)
    qn = np.linalg.norm(queries, axis=1, keepdims=True)
    q2 = np.hstack([queries * (m / np.maximum(qn, 1e-12)), np.zeros((len(queries), 1))]).astype(np.float32)
    return b2, q2


def kmeans(x, k, iters=25, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    c = x[torch.randperm(x.shape[0], generator=g)[:k].to(x.device)].clone()
    for _ in range(iters):
        _, a = knn(x, c, 1)
        a = a[:, 0]
        cnt = torch.bincount(a, minlength=k).float()
        s = torch.zeros_like(c).index_add_(0, a, x)
        mask = cnt > 0
        c[mask] = s[mask] / cnt[mask, None]
    d, a = knn(x, c, 2)
    return c, a[:, 0], d


def skew(v):
    v = v.float(); m = v.mean(); s = v.std()
    return float(((v - m) ** 3).mean() / (s ** 3 + 1e-12))


def fingerprint(base, queries, args, dev, log):
    rng = np.random.default_rng(0)
    n, d = base.shape
    X = torch.from_numpy(base).to(dev)
    Q = torch.from_numpy(queries[: args.n_queries]).to(dev)
    out = {"n_pool": n, "dim": d, "n_queries": Q.shape[0]}

    # LID + relative contrast + query shift, queries against the whole pool
    t = time.time()
    qd, _ = knn(Q, X, 32)
    lid_q = lid_mle(qd)
    probe = torch.from_numpy(rng.choice(n, min(args.n_probe, n), replace=False)).to(dev)
    bd, _ = knn(X[probe], X, 33, exclude_self=False)
    bd = bd[:, 1:]                     # drop the point itself (distance 0)
    lid_b = lid_mle(bd)
    out["lid_query"] = float(lid_q.mean()); out["lid_query_median"] = float(lid_q.median())
    out["lid_base"] = float(lid_b.mean())
    rand = X[torch.from_numpy(rng.choice(n, 1000, replace=False)).to(dev)]
    rd, _ = knn(Q, rand, 1000)
    nn1 = first_nonzero(qd)
    okq = torch.isfinite(nn1)
    out["rc_query"] = float((rd.mean(1)[okq] / nn1[okq]).mean())
    is_dup = lambda d: (d[:, 0] <= 1e-3 * d[:, -1]).float().mean()
    out["dup_frac"] = float((is_dup(qd) + is_dup(bd)) / 2)
    out["query_shift"] = float(qd[:, :10].mean() / bd[:, :10].mean())
    log(f"  LID query {out['lid_query']:.1f}  base {out['lid_base']:.1f}  RC {out['rc_query']:.3f}  "
        f"query shift {out['query_shift']:.2f}   ({time.time() - t:.1f}s)")

    # hubness + drift locality in a pool of the first n_hub points (file order kept)
    t = time.time()
    m = min(args.n_hub, n)
    _, nn10 = knn(X[:m], X[:m], 10, exclude_self=True)
    occ = torch.bincount(nn10.reshape(-1), minlength=m)
    out["hub_skew"] = skew(occ); out["antihub_frac"] = float((occ == 0).float().mean())
    chunk = torch.arange(m, device=dev) * 10 // m
    same = (chunk[nn10] == chunk[:, None]).float().mean()
    out["drift_locality"] = float(same / 0.1)
    log(f"  hub skew {out['hub_skew']:.2f}  antihubs {out['antihub_frac']:.3f}  "
        f"drift locality {out['drift_locality']:.2f}   ({time.time() - t:.1f}s)")

    # PCA + clustering on a sample
    t = time.time()
    S = X[torch.from_numpy(rng.choice(n, min(args.n_sample, n), replace=False)).to(dev)]
    Sc = S - S.mean(0)
    ev = torch.linalg.eigvalsh((Sc.T @ Sc) / S.shape[0]).flip(0).clamp(min=0)
    cum = torch.cumsum(ev, 0) / ev.sum()
    out["pca90"] = int((cum < 0.9).sum()) + 1; out["pca90_frac"] = out["pca90"] / d
    total = float((Sc * Sc).sum(1).mean())
    for k in (10, 100):
        _, _, dist = kmeans(S, k)
        out[f"km_var{k}"] = 1.0 - float((dist[:, 0] ** 2).mean()) / total
        if k == 100:
            out["km_sep100"] = float((dist[:, 0] / dist[:, 1].clamp(min=1e-12)).mean())
    nrm = X.norm(dim=1)
    out["norm_cv"] = float(nrm.std() / nrm.mean())
    log(f"  PCA90 {out['pca90']}/{d}  kmeans var10 {out['km_var10']:.3f} var100 {out['km_var100']:.3f} "
        f"sep100 {out['km_sep100']:.3f}  norm CV {out['norm_cv']:.3f}   ({time.time() - t:.1f}s)")
    return out, S


def plot_embedding(name, base, queries, out_dir, n_points, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = base.shape[0]
    idx = np.linspace(0, n - 1, min(n_points, n)).astype(int)        # evenly over file order
    pts = np.vstack([base[idx], queries[: min(1000, len(queries))]])
    method = "PCA"
    try:
        import umap
        emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(pts)
        method = "UMAP"
    except Exception as e:  # umap-learn missing or failed: fall back to PCA
        log(f"  (UMAP unavailable: {type(e).__name__}; using PCA)")
        c = pts - pts.mean(0)
        _, _, vt = np.linalg.svd(c[: 20000], full_matrices=False)
        emb = c @ vt[:2].T
    nb = len(idx)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    sc = ax.scatter(emb[:nb, 0], emb[:nb, 1], c=idx, cmap="viridis", s=1.5, alpha=0.6, linewidths=0)
    ax.scatter(emb[nb:, 0], emb[nb:, 1], c="red", s=3, alpha=0.7, linewidths=0, label="queries")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02); cb.set_label("insertion order (file position)")
    ax.set(title=f"{name} ({method})", xticks=[], yticks=[]); ax.legend(loc="lower right", fontsize=7, markerscale=4)
    fig.tight_layout()
    path = os.path.join(out_dir, f"embed_{name}.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    log(f"  saved {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name"); p.add_argument("base"); p.add_argument("queries")
    p.add_argument("--metric", default="l2", choices=["l2", "ip"], help="Dataset's native metric (recorded only)")
    p.add_argument("--kind", default="real", help="Label: real / synthetic / reordered")
    p.add_argument("--dtype", choices=["f32", "u8", "i8"], help="Override element type")
    p.add_argument("--n-pool", type=int, default=1_000_000, help="Base vectors read (file order)")
    p.add_argument("--n-queries", type=int, default=1000)
    p.add_argument("--n-probe", type=int, default=2000, help="Base points used for base LID / query shift")
    p.add_argument("--n-hub", type=int, default=100_000, help="Pool for hubness and drift (first N in file order)")
    p.add_argument("--n-sample", type=int, default=100_000, help="Sample for PCA and k-means")
    p.add_argument("--n-plot", type=int, default=20_000, help="Points in the UMAP/PCA plot (0 = no plot)")
    p.add_argument("--out", default="topology/results")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = lambda s: print(s, flush=True)
    dt = {"f32": np.float32, "u8": np.uint8, "i8": np.int8}.get(a.dtype)
    t0 = time.time()
    base = read_bin(a.base, a.n_pool, dt)
    queries = read_bin(a.queries, a.n_queries, dt)
    log(f"[{a.name}] base {base.shape} queries {queries.shape} ({a.kind}, {a.metric})")
    norm_cv_raw = float(np.std(np.linalg.norm(base, axis=1)) / np.mean(np.linalg.norm(base, axis=1)))
    if a.metric == "ip":
        base, queries = mips_to_l2(base, queries)
        log("  inner product: measured in the equivalent L2 space (MIPS-to-L2 transform)")
    with torch.no_grad():
        row, _ = fingerprint(base, queries, a, a.device, log)
    if a.n_plot:
        plot_embedding(a.name, base, queries, a.out, a.n_plot, log)
    row["norm_cv"] = norm_cv_raw
    row = {"dataset": a.name, "kind": a.kind, "metric": a.metric, **row}
    path = os.path.join(a.out, "fingerprints.csv")
    rows = [r for r in (csv.DictReader(open(path)) if os.path.exists(path) else []) if r["dataset"] != a.name]
    rows.append({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in row.items()})
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    log(f"[{a.name}] done in {time.time() - t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
