#!/bin/bash
# usage: run_engine.sh <exe> <outroot> <deck-stem>...
# Each deck in a fresh run directory; argv `exe deck printout`.
exe=$1; root=$2; shift 2
here=$(cd "$(dirname "$0")" && pwd)
for d in "$@"; do
  run="$root/$d"
  rm -rf "$run"; mkdir -p "$run"
  cp "$here/decks/$d.nec" "$run/model.nec"
  ( cd "$run" && "$exe" model.nec model.out < /dev/null > stdout.txt 2> stderr.txt; echo "$d rc=$?" )
  grep -A4 "ANTENNA INPUT PARAMETERS" "$run/model.out" | tail -2
done
