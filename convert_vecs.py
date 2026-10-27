#!/usr/bin/env python3
"""
Convert .fvecs / .bvecs files (TEXMEX format, e.g. SIFT1M, GIST1M) into Jasper's
.bin format: [n:uint32][dim:uint32][row-major data].

Big-ANN .fbin / .u8bin files (bigann, deep, msturing, msspacev) already use
Jasper's format and need no conversion.

Usage
  python3 convert_vecs.py sift/sift_base.fvecs   data/sift1m_base.bin
  python3 convert_vecs.py sift/sift_query.fvecs  data/sift1m_query.bin
  python3 convert_vecs.py bigann_base.bvecs      data/bigann_base.bin --max 10000000
Then pass --dtype f32 for .fvecs sources and --dtype u8 for .bvecs sources.
"""
import argparse
import struct

import numpy as np


def read_vecs(path, max_n=0):
    if path.endswith(".fvecs"):
        elem, size = np.float32, 4
    elif path.endswith(".bvecs"):
        elem, size = np.uint8, 1
    else:
        raise ValueError("expected a .fvecs or .bvecs file")
    raw = np.fromfile(path, dtype=np.uint8)
    dim = int(raw[:4].view(np.int32)[0])
    row_bytes = 4 + dim * size              # each row: int32 dim, then the values
    n = raw.size // row_bytes
    if max_n:
        n = min(n, max_n)
    rows = raw[: n * row_bytes].reshape(n, row_bytes)[:, 4:]
    return np.ascontiguousarray(rows).view(elem).reshape(n, dim)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--max", type=int, default=0, help="Keep only the first N vectors")
    a = p.parse_args()
    x = read_vecs(a.src, a.max)
    with open(a.dst, "wb") as f:
        f.write(struct.pack("II", x.shape[0], x.shape[1]))
        f.write(x.tobytes())
    print(f"wrote {a.dst}: {x.shape[0]} vectors, dim {x.shape[1]}, dtype {x.dtype}")


if __name__ == "__main__":
    main()
