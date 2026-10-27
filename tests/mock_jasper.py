"""
CPU stand-in for the `jasper` Python module, used only to test the benchmark's
bookkeeping without a GPU. Search is exact brute force over live vectors, so a
correct harness must report recall = 1.0 at every checkpoint.

Mirrors the real API used by stream_bench.py:
  read_bin, generate_groundtruth, Graph.build / append / mark_deleted /
  consolidate / search, n_live, n_tombstoned.
Stable ids follow Jasper's rule: build gives ids 0..n-1, append continues upward.
"""
import struct

import numpy as np
import torch


def read_bin(path, dtype="f32", max_vectors=0):
    np_dtype = np.float32 if dtype == "f32" else np.uint8
    with open(path, "rb") as f:
        n, dim = struct.unpack("II", f.read(8))
        if max_vectors > 0:
            n = min(n, max_vectors)
        data = np.frombuffer(f.read(n * dim * np.dtype(np_dtype).itemsize), dtype=np_dtype)
    return torch.from_numpy(data.reshape(n, dim).copy()).to(torch.float16)  # no pinning on CPU


def _knn(vectors, queries, k, distance):
    v = vectors.float()
    q = queries.float()
    if distance == "l2":
        d = (q * q).sum(1, keepdim=True) + (v * v).sum(1)[None, :] - 2 * q @ v.T
    else:
        d = -(q @ v.T)
    dist, idx = d.topk(k, dim=1, largest=False)
    return idx, dist


def generate_groundtruth(vectors, queries, k=100, distance="l2", device="cpu", **_):
    idx, dist = _knn(vectors, queries, k, distance)
    return idx.to(torch.int32), dist


class Graph:
    def __init__(self, vectors, distance):
        self._vecs = [vectors.cpu()]
        self._ids = [torch.arange(vectors.shape[0])]
        self._next_id = vectors.shape[0]
        self._deleted = set()
        self._distance = distance

    @classmethod
    def build(cls, vectors, n_neighbors=32, distance="l2", alpha=1.2, workspace_budget="10GB", **_):
        return cls(vectors, distance)

    def append(self, vectors, alpha=1.2):
        n = vectors.shape[0]
        ids = torch.arange(self._next_id, self._next_id + n)
        self._next_id += n
        self._vecs.append(vectors.cpu())
        self._ids.append(ids)
        return ids.to(torch.int32)

    def mark_deleted(self, ids):
        self._deleted.update(int(i) for i in ids.tolist())

    def consolidate(self, alpha=1.2):
        self._deleted_consolidated = True  # tombstones stay hidden; nothing to repair

    @property
    def n_live(self):
        return int(sum(len(i) for i in self._ids)) - len(self._deleted)

    @property
    def n_tombstoned(self):
        return len(self._deleted)

    def search(self, queries, k=10, beam_width=64):
        vecs = torch.cat(self._vecs)
        ids = torch.cat(self._ids)
        keep = torch.tensor([int(i) not in self._deleted for i in ids.tolist()])
        idx, dist = _knn(vecs[keep], queries.cpu(), k, self._distance)
        return ids[keep][idx].to(torch.int32), dist
