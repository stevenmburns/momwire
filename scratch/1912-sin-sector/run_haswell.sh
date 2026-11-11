#!/bin/bash
# momwire#1912 timing + accuracy on one box, route vs dense, same machine.
# Run from this directory in a clone whose .venv has momwire built.
set -u
PY=../../.venv/bin/python
OUT=${OUT:-results.jsonl}
mkdir -p y
for deck in sym4 hex6; do
  for th in 1 8; do
    for mode in sector dense; do
      OMP_NUM_THREADS=$th OPENBLAS_NUM_THREADS=$th PYTHONPATH=../../src \
        $PY timing.py $deck $mode $OUT --save y/$deck-$mode-t$th.npy \
        > log-$deck-$mode-t$th.txt 2>&1
      echo "$deck $mode t$th rc=$?"
    done
  done
  $PY compare.py y/$deck-sector-t8.npy y/$deck-dense-t8.npy
  $PY compare.py y/$deck-sector-t1.npy y/$deck-dense-t1.npy
done
# over a Sommerfeld half-space (the deck sits 1 m up), one frequency
for deck in sym4 hex6; do
  for mode in sector dense; do
    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH=../../src \
      $PY timing.py $deck $mode $OUT --ground sommerfeld --nk 1 \
      --save y/$deck-$mode-somm.npy > log-$deck-$mode-somm.txt 2>&1
    echo "$deck $mode somm rc=$?"
  done
  $PY compare.py y/$deck-sector-somm.npy y/$deck-dense-somm.npy
done
