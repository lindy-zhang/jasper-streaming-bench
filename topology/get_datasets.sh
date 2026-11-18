#!/usr/bin/env bash
# Download the first 1M base vectors (byte-range request) + queries of the BigANN datasets used
# in the CONDA paper, into data/<name>/. Skips anything already downloaded.
#   bash topology/get_datasets.sh                  # all
#   bash topology/get_datasets.sh msturing deep    # some
set -euo pipefail
N=1000000
get() {  # get NAME BASE_URL QUERY_URL DIM BYTES_PER_VALUE EXT
  local name=$1 burl=$2 qurl=$3 d=$4 b=$5 ext=$6 dir=data/$1
  mkdir -p "$dir"
  local want=$((8 + N * d * b)) f="$dir/base.1M.$ext"
  if [ ! -f "$f" ] || [ "$(stat -c%s "$f")" -ne "$want" ]; then
    echo "== $name: first 1M base vectors ($((want / 1000000)) MB)"
    curl -L --fail -# -r 0-$((want - 1)) "$burl" -o "$f"
    python3 -c "import struct; f=open('$f','r+b'); f.write(struct.pack('<II', $N, $d))"
  fi
  [ -f "$dir/query.$ext" ] || { echo "== $name: queries"; curl -L --fail -# "$qurl" -o "$dir/query.$ext"; }
}
ALL="$*"
sel() { [ -z "$ALL" ] || [[ " $ALL " == *" $1 "* ]]; }
FB=https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks
AZ=https://comp21storage.z5.web.core.windows.net/comp21
YA=https://storage.yandexcloud.net/yandex-research/ann-datasets
sel bigann     && get bigann     $FB/bigann/base.1B.u8bin  $FB/bigann/query.public.10K.u8bin 128 1 u8bin
sel msturing   && get msturing   $AZ/MSFT-TURING-ANNS/base1b.fbin $AZ/MSFT-TURING-ANNS/query100K.fbin 100 4 fbin
sel msspacev   && get msspacev   $AZ/spacev1b/spacev1b_base.i8bin $AZ/spacev1b/query.i8bin 100 1 i8bin
sel text2image && get text2image $YA/T2I/base.1B.fbin $YA/T2I/query.public.100K.fbin 200 4 fbin
sel deep       && get deep       $YA/DEEP/base.1B.fbin $YA/DEEP/query.public.10K.fbin 96 4 fbin
echo "done"
