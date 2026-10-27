#!/usr/bin/env bash
# Run Jasper on CANDOR-Bench workloads: the paper's own runbooks and datasets.
#   bash run_candor.sh            # all experiments below
#   bash run_candor.sh batch      # only experiments whose name contains "batch"
# Results: results/candor/<experiment>.csv (+ .json, .log). Plot with:
#   python3 plot_runs.py results/candor/sift_batch_*.csv
set -euo pipefail

CANDOR=../CANDOR-Bench/runbooks      # path to CANDOR-Bench's runbooks folder
DATA=data/candor                     # where the CANDOR datasets were downloaded
R=64                                 # Jasper max degree: 8, 16, 32 or 64
CONSOLIDATE=10                       # consolidate every N batches when a runbook deletes
FILTER=${1:-}

# dataset key in runbook -> "base file|query file"  (all float32 [n][d] .bin; use --dtype f32)
declare -A FILES=(
  [sift]="$DATA/sift/data_1000000_128|$DATA/sift/queries_10000_128"
  [openimage-streaming]="$DATA/openimages/base.1M.fbin|$DATA/openimages/query.public.10K.fbin"
  [glove]="$DATA/glove/base.fbin|$DATA/glove/query.fbin"
  [msong]="$DATA/msong/base.fbin|$DATA/msong/query.fbin"
  [sun]="$DATA/sun/base.fbin|$DATA/sun/query.fbin"
  [coco]="$DATA/coco/base.fbin|$DATA/coco/query.fbin"
)

# experiment name | runbook (relative to $CANDOR) | dataset key
EXPERIMENTS=(
  "sift_batch_500|batch_sizes/batch_500.yaml|sift"
  "sift_batch_2500|batch_sizes/batch_2500.yaml|sift"
  "sift_batch_10000|batch_sizes/batch_10000.yaml|sift"
  "sift_batch_50000|batch_sizes/batch_50000.yaml|sift"
  "sift_batch_delete_0.1|deletion_patterns/batch_delete_0.1.yaml|sift"
  "sift_batch_delete_0.5|deletion_patterns/batch_delete_0.5.yaml|sift"
  "sift_bulk_delete_0.1|bulk_deletion/bulk_delete_0.1.yaml|sift"
  "sift_bulk_delete_0.5|bulk_deletion/bulk_delete_0.5.yaml|sift"
  "openimages_concept_drift|concept_drift/concept_drift.yaml|openimage-streaming"
  "glove_general|general_experiment/general_experiment.yaml|glove"
  "msong_general|general_experiment/general_experiment.yaml|msong"
)

mkdir -p results/candor
for exp in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r name runbook ds <<< "$exp"
  [[ -n "$FILTER" && "$name" != *"$FILTER"* ]] && continue
  IFS='|' read -r base query <<< "${FILES[$ds]}"
  if [[ ! -f "$base" ]]; then echo "skip $name: $base not found"; continue; fi
  out="results/candor/${name}.csv"
  echo "=== $name  ($runbook, $ds) ==="
  python3 stream_bench.py --base "$base" --queries "$query" --dtype f32 --n-neighbors "$R" \
    --runbook "$CANDOR/$runbook" --runbook-dataset "$ds" --consolidate-every "$CONSOLIDATE" \
    --out "$out" 2>&1 | tee "${out%.csv}.log"
done
