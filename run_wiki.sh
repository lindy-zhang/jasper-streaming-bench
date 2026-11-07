#!/usr/bin/env bash
# Jasper on CONDA's (PVLDB'26) Wikipedia-1M workloads: Sliding Window (SW) and Expiration (EX).
# Data: BigANN wikipedia-cohere (768-d float32, inner product), first 1M vectors + 5K queries.
#   bash run_wiki.sh            # download (once, ~3 GB), then run SW and EX
set -euo pipefail
DATA=data/wikipedia-1M
CONDA_DIR=${CONDA_DIR:-../CONDA}
URL=https://comp21storage.z5.web.core.windows.net/wiki-cohere-35M
R=${R:-32}            # CONDA's paper uses R=32, L=64, alpha=1.2
mkdir -p "$DATA" results/wiki

# 1. Data: first 1M base vectors (byte-range download) + all 5K queries
BYTES=$((8 + 1000000 * 768 * 4))
if [ ! -f "$DATA/base.1M.fbin" ] || [ "$(stat -c%s "$DATA/base.1M.fbin")" -ne "$BYTES" ]; then
  echo "downloading first 1M Wikipedia vectors (3.07 GB)..."
  curl -L --fail -# -H "Range: bytes=0-$((BYTES - 1))" "$URL/wikipedia_base.bin" -o "$DATA/base.1M.fbin"
  python3 -c "import struct; f=open('$DATA/base.1M.fbin','r+b'); f.write(struct.pack('<II', 1000000, 768))"
fi
[ -f "$DATA/query.5K.fbin" ] || curl -L --fail -# "$URL/wikipedia_query.bin" -o "$DATA/query.5K.fbin"

# 2. Runbooks from the CONDA repo
[ -d "$CONDA_DIR" ] || git clone --depth 1 https://github.com/darae-lee/CONDA "$CONDA_DIR"

# 3. Run: one 10K insert op = one Jasper batch; consolidate after every delete op
for wl in sliding_window expiration; do
  out=results/wiki/wiki_${wl}.csv
  echo "=== wikipedia-1M, $wl ==="
  python3 stream_bench.py --base "$DATA/base.1M.fbin" --queries "$DATA/query.5K.fbin" \
    --mips-to-l2 --n-neighbors "$R" --alpha 1.2 --n-queries 5000 --beam-widths 64 100 128 \
    --runbook "$CONDA_DIR/runbooks/${wl}_runbook.yaml" --runbook-dataset wikipedia-1M \
    --consolidate-every 1 --workspace-budget 6GB --out "$out" "$@" 2>&1 | tee "${out%.csv}.log" | grep -E "^\[(search|done)|Error|error" || true
done
python3 tools/plot_timesteps.py results/wiki/wiki_sliding_window.csv results/wiki/wiki_expiration.csv
