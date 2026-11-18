#!/usr/bin/env bash
# Fingerprint every dataset that is present (real + synthetic), then summarize.
#   bash topology/run_topology.sh        # extra args (e.g. --n-plot 0) go to fingerprint.py
set -uo pipefail
FP="python3 topology/fingerprint.py"
run() {  # run NAME BASE QUERY METRIC KIND
  if [ -f "$2" ] && [ -f "$3" ]; then $FP "$1" "$2" "$3" --metric "$4" --kind "$5" "${EXTRA[@]}" || echo "!! $1 failed";
  else echo "-- skip $1 (files not found)"; fi
}
EXTRA=("$@")
# real (already on nezha from earlier runs)
run sift       data/candor/sift/data_1000000_128   data/candor/sift/queries_10000_128 l2 real
run wikipedia  data/wikipedia-1M/base.1M.fbin      data/wikipedia-1M/query.5K.fbin    ip real
run glove      data/candor/glove/base.fbin         data/candor/glove/query.fbin       l2 real
run msong      data/candor/msong/base.fbin         data/candor/msong/query.fbin       l2 real
# real (topology/get_datasets.sh)
run bigann     data/bigann/base.1M.u8bin           data/bigann/query.u8bin            l2 real
run msturing   data/msturing/base.1M.fbin          data/msturing/query.fbin           l2 real
run msspacev   data/msspacev/base.1M.i8bin         data/msspacev/query.i8bin          l2 real
run text2image data/text2image/base.1M.fbin        data/text2image/query.fbin         ip real
run deep       data/deep/base.1M.fbin              data/deep/query.fbin               l2 real
# synthetic (topology/gen_synthetic.py)
for d in data/synthetic/*/; do
  [ -d "$d" ] || continue
  n=$(basename "$d"); run "$n" "$d/base.fbin" "$d/query.fbin" l2 synthetic
done
python3 topology/summarize.py
