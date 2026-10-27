#!/usr/bin/env bash
# Sweep batch sizes x update patterns for one dataset.
# Edit the variables below, then:  bash run_sweep.sh
# Each run writes results/<name>_<pattern>_bs<batch>.csv (+ .json, .log).
set -euo pipefail

NAME=sift1m
BASE=data/sift1m_base.bin
QUERIES=data/sift1m_query.bin
DTYPE=f32                 # f32 for .fvecs/.fbin sources, u8 for .bvecs/.u8bin
R=64                      # max degree: 8, 16, 32 or 64
INIT_FRAC=0.5             # half built up front, half streamed in
STREAM_VECTORS=100000     # cap on streamed vectors per run, so small batches finish
BATCH_SIZES="1000 10000 100000"
PATTERNS="insert_only sliding_window random_delete"

mkdir -p results
for pattern in $PATTERNS; do
  for bs in $BATCH_SIZES; do
    max_batches=$(( STREAM_VECTORS / bs ))
    eval_every=$(( max_batches / 5 )); [ "$eval_every" -lt 1 ] && eval_every=1
    consolidate=$([ "$pattern" = insert_only ] && echo 0 || echo 5)
    out="results/${NAME}_${pattern}_bs${bs}.csv"
    echo "=== $pattern  batch=$bs  batches=$max_batches ==="
    python3 stream_bench.py \
      --base "$BASE" --queries "$QUERIES" --dtype "$DTYPE" \
      --n-neighbors "$R" --init-frac "$INIT_FRAC" \
      --batch-size "$bs" --max-batches "$max_batches" \
      --pattern "$pattern" --consolidate-every "$consolidate" \
      --eval-every "$eval_every" --out "$out" 2>&1 | tee "${out%.csv}.log"
  done
done
echo "done; plot with: python3 plot_sweep.py results/${NAME}_*.csv"
