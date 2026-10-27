#!/usr/bin/env bash
# CPU smoke test: runs stream_bench.py against mock_jasper.py (exact brute-force
# search) for every built-in pattern and for scaled-down CANDOR runbooks.
# With exact search every recall must be 1.0, and live counts must match what
# the workload implies; the script fails otherwise.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p tests/tmp

python3 - <<'EOF'
import numpy as np, struct
rng = np.random.default_rng(0)
for name, n in [("base", 20000), ("query", 50)]:
    x = rng.normal(size=(n, 16)).astype(np.float32)
    with open(f"tests/tmp/{name}.bin", "wb") as f:
        f.write(struct.pack("II", n, 16)); f.write(x.tobytes())
EOF

check() {  # check <csv> <label> <expected final n_live>
  python3 - "$1" "$2" "$3" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
recalls = [float(v) for r in rows for k, v in r.items() if k.startswith("recall") and v]
assert recalls and all(abs(r - 1.0) < 1e-9 for r in recalls), f"recall != 1: {recalls}"
assert int(rows[-1]["n_live"]) == int(sys.argv[3]), f"final live {rows[-1]['n_live']} != {sys.argv[3]}"
print(f"  ok {sys.argv[2]:40s} rows={len(rows):4d} final_live={rows[-1]['n_live']:>6s} evals={len(recalls)//3}")
EOF
}

echo "built-in patterns (3000 vectors, half built, 6 batches of 250)"
for order in file shuffle; do
  for spec in "insert_only:3000" "sliding_window:1500" "random_delete:1500" "batch_tail_delete:2250"; do
    pattern=${spec%%:*}; expect=${spec##*:}
    out="tests/tmp/${pattern}_${order}.csv"
    python3 stream_bench.py --mock --device cpu --base tests/tmp/base.bin --queries tests/tmp/query.bin \
      --max-vectors 3000 --n-queries 50 --init-frac 0.5 --batch-size 250 --eval-every 2 \
      --consolidate-every 3 --delete-ratio $([ "$pattern" = batch_tail_delete ] && echo 0.5 || echo 1.0) \
      --pattern "$pattern" --order "$order" --out "$out" > "${out%.csv}.log" 2>&1
    check "$out" "$pattern ($order)" "$expect"
  done
done

echo "CANDOR runbooks (scaled x0.02: 20K vectors)"
for spec in "batch_insert:20000" "batch_insert_delete:18100" "bulk_delete:18000" "concept_drift:20000"; do
  name=${spec%%:*}; expect=${spec##*:}
  out="tests/tmp/runbook_${name}.csv"
  python3 stream_bench.py --mock --device cpu --base tests/tmp/base.bin --queries tests/tmp/query.bin \
    --n-queries 50 --runbook "tests/runbooks/${name}.yaml" --runbook-dataset test \
    --consolidate-every 50 --out "$out" > "${out%.csv}.log" 2>&1
  check "$out" "runbook $name" "$expect"
done
echo "all tests passed"
