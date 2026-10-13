#!/bin/bash
# usage: runset.sh <tree> <python> <out.jsonl> <deck-stem>...   (stems like g8hul_x1.nec4)
tree=$1; py=$2; out=$3; shift 3
here=$(cd "$(dirname "$0")" && pwd)
for d in "$@"; do
  dialect=${d##*.}
  for m in served reduced; do
    "$py" "$here/refrun.py" "$tree" "$here/decks/$d.nec" "$dialect" bspline "$m" 2> "$here/last.err" | tail -1 >> "$out"
    echo "$d $m rc=${PIPESTATUS[0]}"
  done
done
