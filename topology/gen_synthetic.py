#!/usr/bin/env python3
"""
Synthetic datasets with one controlled property each, written as BigANN .fbin files
(base + 10K in-distribution queries) under data/synthetic/<name>/.

  python3 topology/gen_synthetic.py            # all of them, 1M x 128 each (~0.5 GB each)
  python3 topology/gen_synthetic.py --n 100000 # smaller

  random_gauss       i.i.d. N(0, 1) in 128-d, same recipe as CANDOR's random-m (seed 42):
                     no structure at all
  gmm_s{0.2,0.6,1.2} 100 Gaussian clusters; spread controls how tight/separated they are
  manifold_m{4,16,64} an m-dimensional curved manifold embedded in 128-d: controls
                     intrinsic dimension while the ambient dimension stays 128
  drift_gmm          100 clusters whose mix changes with insertion order: the stream moves
                     from cluster to cluster over time (like real timestamped data)
"""
import argparse
import os

import numpy as np


def write_fbin(path, x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    with open(path, "wb") as f:
        np.array(x.shape, dtype=np.uint32).tofile(f)
        x.tofile(f)


def gmm(rng, n, centers, spread, labels=None):
    if labels is None:
        labels = rng.integers(0, len(centers), n)
    return centers[labels] + spread * rng.standard_normal((n, centers.shape[1]), dtype=np.float32)


def manifold(rng, n, m, d, W1, W2):
    z = rng.standard_normal((n, m), dtype=np.float32)
    x = np.tanh(z @ W1) @ W2
    return x + 0.01 * rng.standard_normal((n, d), dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1_000_000)
    p.add_argument("--nq", type=int, default=10_000)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--only", nargs="*", help="Subset of dataset names")
    a = p.parse_args()
    n, nq, d = a.n, a.nq, a.d
    sets = {}

    def random_gauss():
        np.random.seed(42)                                    # CANDOR's RandomDataset recipe
        b = np.random.randn(n, d).astype(np.float32)
        q = np.random.default_rng(43).standard_normal((nq, d), dtype=np.float32)
        return b, q
    sets["random_gauss"] = random_gauss

    for s in (0.2, 0.6, 1.2):
        def f(s=s):
            rng = np.random.default_rng(1)
            c = rng.standard_normal((100, d), dtype=np.float32)
            return gmm(rng, n, c, s), gmm(rng, nq, c, s)
        sets[f"gmm_s{s}"] = f

    for m in (4, 16, 64):
        def f(m=m):
            rng = np.random.default_rng(2)
            W1 = rng.standard_normal((m, 4 * m), dtype=np.float32) / np.sqrt(m)
            W2 = rng.standard_normal((4 * m, d), dtype=np.float32) / np.sqrt(4 * m)
            return manifold(rng, n, m, d, W1, W2), manifold(rng, nq, m, d, W1, W2)
        sets[f"manifold_m{m}"] = f

    def drift():
        rng = np.random.default_rng(3)
        c = rng.standard_normal((100, d), dtype=np.float32)
        # point i draws its cluster from a window of ~10 clusters centred at i/n * 100
        centre = np.arange(n) / n * 100
        lab = np.clip(np.round(centre + rng.normal(0, 3, n)), 0, 99).astype(int)
        return gmm(rng, n, c, 0.6, lab), gmm(rng, nq, c, 0.6)
    sets["drift_gmm"] = drift

    for name, make in sets.items():
        if a.only and name not in a.only:
            continue
        folder = os.path.join(a.out, name)
        os.makedirs(folder, exist_ok=True)
        b, q = make()
        write_fbin(os.path.join(folder, "base.fbin"), b)
        write_fbin(os.path.join(folder, "query.fbin"), q)
        print(f"wrote {folder}: base {b.shape}, queries {q.shape}", flush=True)


if __name__ == "__main__":
    main()
