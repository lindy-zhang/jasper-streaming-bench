import os, struct, sys
import numpy as np, torch, jasper
D = "data/wikipedia-1M"
for fn in ("base.1M.fbin", "query.5K.fbin"):
    p = os.path.join(D, fn); n, d = struct.unpack("<II", open(p, "rb").read(8))
    print(f"{fn}: header n={n} d={d}  bytes={os.path.getsize(p):,}  expected={8 + n * d * 4:,}")
base = jasper.read_bin(f"{D}/base.1M.fbin", "f32", max_vectors=20000)
qs = jasper.read_bin(f"{D}/query.5K.fbin", "f32")[:200].contiguous()
for name, x in (("base", base), ("queries", qs)):
    xf = x.float(); nrm = xf.norm(dim=1)
    print(f"{name}: shape={tuple(x.shape)} dtype={x.dtype} nan={int(torch.isnan(xf).sum())} "
          f"zero_rows={int((nrm == 0).sum())} norm mean={nrm.mean():.3f} min={nrm.min():.3f} max={nrm.max():.3f}")
print("first base vector, first 5 values:", base[0, :5].tolist())
b, q = base.float(), qs.float()
ip = q @ b.T
true_ip = ip.topk(10, dim=1, largest=True).indices.numpy()
worst_ip = ip.topk(10, dim=1, largest=False).indices.numpy()
l2 = (q * q).sum(1, keepdim=True) + (b * b).sum(1)[None] - 2 * ip
true_l2 = l2.topk(10, dim=1, largest=False).indices.numpy()
def ov(a, c): return np.mean([len(set(x) & set(y)) / 10 for x, y in zip(a, c)])
print(f"overlap true-IP vs true-L2 top10: {ov(true_ip, true_l2):.3f}")
for dist in ("ip", "l2"):
    gt, _ = jasper.generate_groundtruth(base, qs, k=10, distance=dist, device="cuda")
    gt = gt.cpu().numpy()
    print(f"[{dist}] jasper GT vs true-IP {ov(gt, true_ip):.3f} | vs worst-IP {ov(gt, worst_ip):.3f} | vs true-L2 {ov(gt, true_l2):.3f}")
    g = jasper.Graph.build(base.cuda(), n_neighbors=32, distance=dist, alpha=1.2, workspace_budget="6GB")
    idx, dd = g.search(qs.cuda(), k=10, beam_width=100)
    idx = idx.cpu().numpy()
    print(f"[{dist}] jasper search vs jasper GT {ov(idx, gt):.3f} | vs true-IP {ov(idx, true_ip):.3f} "
          f"| vs worst-IP {ov(idx, worst_ip):.3f} | vs true-L2 {ov(idx, true_l2):.3f}")
    print(f"[{dist}] sample ids {idx[0][:5].tolist()} dists {dd[0][:3].tolist()} | true-IP {true_ip[0][:5].tolist()}")
    del g; torch.cuda.empty_cache()
