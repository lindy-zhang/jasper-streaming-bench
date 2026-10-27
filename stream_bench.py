#!/usr/bin/env python3
"""
Streaming benchmark for Jasper (GPU Vamana index with insert/delete support).

Two ways to describe a workload
  1. Built-in pattern (--pattern): build on --init-frac of the data, then stream the
     rest in batches of --batch-size with one of these update patterns:
       insert_only        only inserts; the index grows
       sliding_window     after each batch, delete the oldest vectors so the live
                          count stays at --window (default: initial size)
       random_delete      after each batch, delete --delete-ratio x batch random
                          live vectors
       batch_tail_delete  after each batch, delete the last --delete-ratio of that
                          same batch (CANDOR's batch_insert_delete)
  2. CANDOR runbook (--runbook FILE --runbook-dataset NAME): replays one of
     CANDOR-Bench's YAML runbooks. Supported operations: initial, batch_insert,
     batch_insert_delete, delete, search (startHPC/endHPC/waitPending are ignored;
     enableScenario, stress_test and auto_batch_delete are refused). eventRate is
     not enforced; each batch records whether Jasper's insert speed would keep up
     with it. Rows are taken in file order, exactly as CANDOR does, so a
     dataset stored in time order (e.g. OpenImagesStreaming) streams with its drift.

For every inserted batch it records insert time and throughput, GPU utilization
(NVML) and GPU memory. At evaluation points it recomputes exact ground truth over
the vectors live at that moment and measures search QPS and Recall@k.
Output: one CSV row per step (build, each batch, each bulk delete) + a JSON of settings.

Examples
  python3 stream_bench.py --base data/sift1m_base.bin --queries data/sift1m_query.bin \
      --batch-size 10000 --pattern sliding_window --consolidate-every 5 \
      --out results/sift1m_sliding_bs10000.csv

  python3 stream_bench.py --base candor/sift/data_1000000_128 \
      --queries candor/sift/queries_10000_128 \
      --runbook CANDOR-Bench/runbooks/batch_sizes/batch_10000.yaml --runbook-dataset sift \
      --out results/candor_sift_batch10000.csv
"""
import argparse
import contextlib
import csv
import json
import os
import sys
import threading
import time

import numpy as np
import torch


# ----------------------------------------------------------------------------
# GPU utilization sampler
# ----------------------------------------------------------------------------
class GpuMonitor:
    """Samples GPU utilization in a background thread, tagged by phase.

    NVML 'utilization' = % of time at least one kernel was running over the
    driver's last sample period. It is not how busy the SMs were, and a phase
    much shorter than ~100 ms may get no samples; use Nsight Systems for those.
    If NVML is unavailable (no `nvidia-ml-py`, or no GPU) the columns stay blank.
    """

    def __init__(self, interval_s=0.02, device_index=0):
        self.interval_s = interval_s
        self.phase = None
        self.samples = []
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        try:
            import pynvml  # pip install nvidia-ml-py
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception as e:
            print(f"[monitor] NVML unavailable ({e}); GPU utilization will be blank",
                  file=sys.stderr)
            self._nvml = None

    def start(self):
        if self._nvml is None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join()

    def _loop(self):
        while self._running:
            if self.phase is not None:
                u = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                with self._lock:
                    self.samples.append((u.gpu, u.memory))
            time.sleep(self.interval_s)

    @contextlib.contextmanager
    def measure(self, name, out):
        """Run a block as phase `name`; write mean/max utilization into `out`."""
        with self._lock:
            self.samples = []
        self.phase = name
        try:
            yield
        finally:
            self.phase = None
            with self._lock:
                s = list(self.samples)
            out[f"{name}_util_samples"] = len(s)
            if s:
                gpu = [x[0] for x in s]
                mem = [x[1] for x in s]
                out[f"{name}_gpu_util_mean"] = round(float(np.mean(gpu)), 1)
                out[f"{name}_gpu_util_max"] = int(max(gpu))
                out[f"{name}_mem_util_mean"] = round(float(np.mean(mem)), 1)


# ----------------------------------------------------------------------------
# Workloads: each compiles to a list of steps
#   ("build", rows)              initial graph from these dataset rows
#   ("insert", rows, meta)       append these rows (meta: event_rate, op)
#   ("delete", rows, standalone) mark these rows deleted; standalone = own CSV row
#   ("consolidate",)
#   ("eval", label)
# ----------------------------------------------------------------------------
def compile_pattern(args, n_total, rng):
    order = np.arange(n_total) if args.order == "file" else rng.permutation(n_total)
    n0 = int(n_total * args.init_frac)
    init_rows, stream_rows = order[:n0], order[n0:]
    window = args.window or n0
    bs = args.batch_size
    n_batches = (len(stream_rows) + bs - 1) // bs
    if args.max_batches:
        n_batches = min(n_batches, args.max_batches)
    eval_every = args.eval_every or max(1, n_batches // 10)

    steps = [("build", init_rows), ("eval", "after build")]
    live = np.zeros(n_total, dtype=bool)
    live[init_rows] = True
    insert_order = list(init_rows)
    oldest = 0
    for b in range(1, n_batches + 1):
        rows = stream_rows[(b - 1) * bs: b * bs]
        steps.append(("insert", rows, {"op": args.pattern}))
        live[rows] = True
        insert_order.extend(rows.tolist())
        if args.pattern == "sliding_window":
            excess = int(live.sum()) - window
            dels = []
            while excess > 0 and oldest < len(insert_order):
                r = insert_order[oldest]
                oldest += 1
                if live[r]:
                    dels.append(r)
                    excess -= 1
            dels = np.asarray(dels, dtype=np.int64)
        elif args.pattern == "random_delete":
            n_del = min(int(round(args.delete_ratio * len(rows))), int(live.sum()) - 1)
            dels = rng.choice(np.nonzero(live)[0], size=max(n_del, 0), replace=False)
        elif args.pattern == "batch_tail_delete":
            dels = rows[max(0, len(rows) - int(len(rows) * args.delete_ratio)):]
        else:
            dels = np.empty(0, dtype=np.int64)
        if len(dels):
            steps.append(("delete", np.asarray(dels, dtype=np.int64), False))
            live[dels] = False
        if args.consolidate_every and b % args.consolidate_every == 0:
            steps.append(("consolidate",))
        if b % eval_every == 0 or b == n_batches:
            steps.append(("eval", f"batch {b}"))
    return steps, {"n_initial": n0, "window": window, "n_batches": n_batches}


IGNORED_OPS = {"startHPC", "endHPC", "waitPending"}


def compile_runbook(args, n_total):
    import yaml
    book = yaml.safe_load(open(args.runbook))
    if args.runbook_dataset not in book:
        sys.exit(f"dataset '{args.runbook_dataset}' not in runbook; choose one of: "
                 + ", ".join(k for k in book))
    rb = book[args.runbook_dataset]
    ops = [rb[k] for k in sorted(k for k in rb if isinstance(k, int))]

    # Count insert batches first, to pick an evaluation interval.
    total_batches = 0
    for op in ops:
        if op["operation"] in ("batch_insert", "batch_insert_delete"):
            total_batches += -(-(op["end"] - op["start"]) // op["batchSize"])
    limit = args.max_batches or total_batches
    eval_every = args.eval_every or max(1, min(limit, total_batches) // 10)

    steps, b, built = [], 0, False
    for op in ops:
        name = op["operation"]
        if name in IGNORED_OPS:
            continue
        if name == "initial":
            rows = np.arange(op["start"], op["end"])
            steps += [("build", rows), ("eval", "after build")]
            built = True
        elif name in ("batch_insert", "batch_insert_delete"):
            if not built:
                sys.exit("runbook inserts before an 'initial' build; not supported")
            bs, start, end = op["batchSize"], op["start"], op["end"]
            pct = op.get("deletion_percentage", 0.0) if name == "batch_insert_delete" else 0.0
            for s in range(start, end, bs):
                if b >= limit:
                    break
                b += 1
                rows = np.arange(s, min(s + bs, end))
                steps.append(("insert", rows, {"op": name, "event_rate": op.get("eventRate")}))
                if pct:  # CANDOR deletes the last `pct` of each batch it just inserted
                    dels = rows[max(0, len(rows) - int(bs * pct)):]
                    if len(dels):
                        steps.append(("delete", dels, False))
                if args.consolidate_every and b % args.consolidate_every == 0:
                    steps.append(("consolidate",))
                if b % eval_every == 0:
                    steps.append(("eval", f"batch {b}"))
            steps.append(("eval", f"end of {name}"))
        elif name == "delete":
            steps.append(("delete", np.arange(op["start"], op["end"]), True))
            if args.consolidate_every:
                steps.append(("consolidate",))
        elif name == "search":
            steps.append(("eval", "runbook search"))
        elif name == "enableScenario":
            sys.exit("runbook uses enableScenario (CANDOR's noise/drop injection), which this harness "
                     "does not reproduce; running it would silently become plain inserts")
        else:
            sys.exit(f"runbook operation '{name}' is not supported by this harness")
    need = max(int(s[1].max()) for s in steps if s[0] in ("build", "insert")) + 1
    if need > n_total:
        sys.exit(f"runbook needs {need} vectors but the base file has {n_total}")
    return steps, {"runbook_ops": [o["operation"] for o in ops], "n_batches": b,
                   "n_initial": len(steps[0][1])}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def physical_gpu_index(override):
    """NVML numbers GPUs physically and ignores CUDA_VISIBLE_DEVICES, so map the
    GPU this process uses (the first visible one) back to its physical index."""
    if override >= 0:
        return override
    first = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    return int(first) if first.isdigit() else 0


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def gpu_mem_used_gb(device):
    if device != "cuda":
        return None
    free, total = torch.cuda.mem_get_info()
    return round((total - free) / 1e9, 3)


def recall_at_k(result_ids, gt_ids, k):
    res, gt = result_ids[:, :k], gt_ids[:, :k]
    hits = (res[:, :, None] == gt[:, None, :]).any(axis=2).sum()
    return float(hits) / (gt.shape[0] * k)


def parse_args():
    p = argparse.ArgumentParser(description="Jasper streaming insert/delete benchmark")
    # data
    p.add_argument("--base", required=True, help="Base vectors (.bin/.fbin: [n u32][dim u32][data])")
    p.add_argument("--queries", required=True, help="Query vectors (same format)")
    p.add_argument("--dtype", default="f32", choices=["f32", "u8"], help="On-disk element type")
    p.add_argument("--max-vectors", type=int, default=0, help="Load only the first N base vectors (0 = all)")
    p.add_argument("--n-queries", type=int, default=1000, help="Queries per evaluation")
    p.add_argument("--seed", type=int, default=0)
    # index
    p.add_argument("--n-neighbors", type=int, default=64, choices=[8, 16, 32, 64], help="Max degree R")
    p.add_argument("--distance", default="l2", choices=["l2", "ip"])
    p.add_argument("--alpha", type=float, default=1.2)
    p.add_argument("--workspace-budget", default="10GB")
    # built-in pattern workload
    p.add_argument("--pattern", default="insert_only",
                   choices=["insert_only", "sliding_window", "random_delete", "batch_tail_delete"])
    p.add_argument("--order", default="file", choices=["file", "shuffle"], help="Insertion order")
    p.add_argument("--init-frac", type=float, default=0.5, help="Fraction of vectors in the initial build")
    p.add_argument("--batch-size", type=int, default=10000)
    p.add_argument("--window", type=int, default=0, help="Live size for sliding_window (0 = initial size)")
    p.add_argument("--delete-ratio", type=float, default=1.0,
                   help="random_delete: deletes per insert; batch_tail_delete: fraction of each batch")
    # CANDOR runbook workload (overrides the pattern options above)
    p.add_argument("--runbook", default="", help="CANDOR-Bench runbook YAML to replay")
    p.add_argument("--runbook-dataset", default="", help="Dataset key inside the runbook, e.g. sift")
    # shared workload options
    p.add_argument("--max-batches", type=int, default=0, help="Stop after this many insert batches (0 = all)")
    p.add_argument("--consolidate-every", type=int, default=0, help="Consolidate every C batches (0 = never)")
    # evaluation
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--beam-widths", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--eval-every", type=int, default=0, help="Evaluate every E batches (0 = about 10 evals)")
    # output
    p.add_argument("--out", default="results/run.csv")
    p.add_argument("--device", default="cuda", help="cuda (real runs) or cpu (mock tests only)")
    p.add_argument("--nvml-index", type=int, default=-1,
                   help="Physical GPU index for utilization sampling (-1 = first in CUDA_VISIBLE_DEVICES)")
    p.add_argument("--mock", action="store_true", help="Use tests/mock_jasper.py instead of jasper (CPU test)")
    a = p.parse_args()
    if a.runbook and not a.runbook_dataset:
        p.error("--runbook needs --runbook-dataset")
    return a


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    if args.mock:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
        import mock_jasper as jasper
    else:
        import jasper
    dev = args.device
    rng = np.random.default_rng(args.seed)

    vectors = jasper.read_bin(args.base, args.dtype, max_vectors=args.max_vectors)  # CPU float16
    n_total, dim = vectors.shape
    queries_cpu = jasper.read_bin(args.queries, args.dtype)[: args.n_queries].contiguous()
    queries = queries_cpu.to(dev)
    print(f"[data] {n_total} base vectors, dim {dim}, {queries.shape[0]} queries", flush=True)

    if args.runbook:
        steps, info = compile_runbook(args, n_total)
    else:
        steps, info = compile_pattern(args, n_total, rng)
    print(f"[workload] {info.get('n_batches')} insert batches, {len(steps)} steps", flush=True)

    id_of_row = np.full(n_total, -1, dtype=np.int64)  # dataset row -> Jasper stable id
    live = np.zeros(n_total, dtype=bool)
    inserted = 0
    mon = GpuMonitor(device_index=physical_gpu_index(args.nvml_index))
    mon.start()
    records, rec, g = [], None, None
    batch, dirty, n_evals = 0, True, 0

    def counts(r):
        r.update({"inserted_total": inserted, "n_live": int(live.sum()),
                  "jasper_n_live": g.n_live, "jasper_n_tombstoned": g.n_tombstoned,
                  "gpu_mem_used_gb": gpu_mem_used_gb(dev)})

    def evaluate(r, label):
        live_rows = np.nonzero(live)[0]
        gt_local, _ = jasper.generate_groundtruth(
            vectors[torch.from_numpy(live_rows)], queries_cpu, k=args.k,
            distance=args.distance, device=dev)
        gt_ids = id_of_row[live_rows[gt_local.numpy().astype(np.int64)]]
        r["eval"] = label
        for bw in args.beam_widths:
            st = {}
            sync(dev)
            t = time.perf_counter()
            with mon.measure("search", st):
                idx, _ = g.search(queries, k=args.k, beam_width=bw)
                sync(dev)
            dt = time.perf_counter() - t
            r[f"recall@{args.k}_bw{bw}"] = round(
                recall_at_k(idx.cpu().numpy().astype(np.int64), gt_ids, args.k), 4)
            r[f"qps_bw{bw}"] = round(queries.shape[0] / dt, 1)
            r[f"search_gpu_util_mean_bw{bw}"] = st.get("search_gpu_util_mean")

    for step in steps:
        kind = step[0]

        if kind == "build":
            rows = step[1]
            rec = {"batch": 0, "step": "build", "batch_size": len(rows)}
            records.append(rec)
            init_vecs = vectors[torch.from_numpy(rows)].to(dev)
            sync(dev)
            t = time.perf_counter()
            with mon.measure("build", rec):
                g = jasper.Graph.build(init_vecs, n_neighbors=args.n_neighbors, distance=args.distance,
                                       alpha=args.alpha, workspace_budget=args.workspace_budget)
                sync(dev)
            s = max(time.perf_counter() - t, 1e-9)
            rec["build_s"] = round(s, 3)
            rec["build_vps"] = round(len(rows) / s, 1)
            del init_vecs
            id_of_row[rows] = np.arange(len(rows))  # build assigns identity ids 0..n-1
            live[rows] = True
            inserted += len(rows)
            print(f"[build] {len(rows)} vectors in {rec['build_s']:.2f}s", flush=True)

        elif kind == "insert":
            rows, meta = step[1], step[2]
            batch += 1
            rec = {"batch": batch, "step": meta.get("op", "insert"), "batch_size": len(rows)}
            records.append(rec)
            batch_vecs = vectors[torch.from_numpy(rows)].to(dev)  # host->device copy, not timed
            sync(dev)
            t = time.perf_counter()
            with mon.measure("append", rec):
                ids = g.append(batch_vecs, alpha=args.alpha)
                sync(dev)
            s = max(time.perf_counter() - t, 1e-9)
            del batch_vecs
            id_of_row[rows] = ids.cpu().numpy().astype(np.int64)
            live[rows] = True
            inserted += len(rows)
            rec["append_s"] = round(s, 4)
            rec["append_vps"] = round(len(rows) / s, 1)
            if meta.get("event_rate"):
                rec["event_rate"] = meta["event_rate"]
                rec["keeps_up"] = rec["append_vps"] >= meta["event_rate"]

        elif kind == "delete":
            rows, standalone = step[1], step[2]
            rows = rows[live[rows]]                    # only rows that are live now
            if standalone:
                rec = {"batch": batch, "step": "bulk_delete"}
                records.append(rec)
            if len(rows):
                t = time.perf_counter()
                with mon.measure("delete", rec):
                    g.mark_deleted(torch.from_numpy(id_of_row[rows]).to(torch.int32))
                    sync(dev)
                rec["delete_s"] = round(rec.get("delete_s", 0) + time.perf_counter() - t, 4)
                rec["n_deleted"] = rec.get("n_deleted", 0) + len(rows)
                live[rows] = False

        elif kind == "consolidate":
            t = time.perf_counter()
            with mon.measure("consolidate", rec):
                g.consolidate(alpha=args.alpha)
                sync(dev)
            rec["consolidate_s"] = round(time.perf_counter() - t, 4)

        elif kind == "eval":
            if dirty or n_evals == 0:
                evaluate(rec, step[1])
                n_evals += 1
            dirty = False
            counts(rec)
            print(f"[{step[1]}] live={rec['n_live']} "
                  + " ".join(f"{k}={v}" for k, v in rec.items() if k.startswith("recall")), flush=True)
            continue

        dirty = True
        counts(rec)
        if kind == "insert":
            print(f"[batch {batch}] live={rec['n_live']} append={rec['append_vps']:,.0f} vec/s "
                  f"util={rec.get('append_gpu_util_mean')}", flush=True)

    mon.stop()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fields = []
    for r in records:
        for key in r:
            if key not in fields:
                fields.append(key)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    meta = vars(args).copy()
    meta.update(info)
    meta.update({"n_total": n_total, "dim": dim,
                 "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"})
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[done] wrote {args.out} ({len(records)} rows)")


if __name__ == "__main__":
    main()
