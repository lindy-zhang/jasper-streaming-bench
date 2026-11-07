#!/usr/bin/env bash
# Batch-to-index ratio experiment: insert the same 10,000 new vectors into indexes of
# different sizes, either in ONE batch of 10,000 or in 100 batches of 100, and compare recall.
# The recall gap = cost of inserting a big batch at once, as a function of batch/index ratio.
#   bash run_ratio.sh            (extra args, e.g. --mock --device cpu, are passed through)
set -euo pipefail
BASE=${BASE:-data/candor/sift/data_1000000_128}
QUERIES=${QUERIES:-data/candor/sift/queries_10000_128}
NEW=${NEW:-10000}
SIZES=${SIZES:-"10000 20000 50000 100000 200000 500000"}
mkdir -p results/ratio
for n in $SIZES; do
  frac=$(python3 -c "print(($n + 0.5) / ($n + $NEW))")
  for bs in 100 $NEW; do
    out=results/ratio/n${n}_bs${bs}.csv
    echo "=== index $n, insert $NEW in batches of $bs ==="
    python3 stream_bench.py --base "$BASE" --queries "$QUERIES" --max-vectors $((n + NEW)) \
      --pattern insert_only --order file --init-frac "$frac" --batch-size "$bs" \
      --eval-every 100000 --workspace-budget 6GB --out "$out" "$@" > "${out%.csv}.log" 2>&1
    tail -1 "${out%.csv}.log"
  done
done
python3 tools/ratio_summary.py
