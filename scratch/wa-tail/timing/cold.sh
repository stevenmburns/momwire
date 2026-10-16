#!/bin/bash
# Cold-call timing, main (base) vs branch (wa), interleaved, 3 reps.
# Guard: before each run wait for load < 0.5 and no other heavy job; record load after.
cd ~/u4/watail
waitq() { until awk "BEGIN{exit !($(cut -d" " -f1 /proc/loadavg) < 0.5)}" && [ "$(ps -eo args | grep -cE "[p]ytest|[n]ec5cl|[g]ate_lm|[m]omwire-nec5")" = 0 ]; do sleep 15; done; }
one() { # tree deck basis variant threads
  waitq; rt=$(mktemp -d); l0=$(cut -d" " -f1 /proc/loadavg)
  r=$(OMP_NUM_THREADS=$5 OPENBLAS_NUM_THREADS=$5 MOMWIRE_FORCE_VARIANT=$4 MOMWIRE_SOMM_CACHE_DIR=$rt MOMWIRE_BELOW_ANCHORS=off \
      $1/.venv/bin/python sk/run_one.py $2 $rt/o.out $3 2>/dev/null)
  l1=$(cut -d" " -f1 /proc/loadavg); rm -rf $rt; echo "$r load $l0/$l1"
}
for cfg in "avx2 1" "sse2 2"; do set -- $cfg; V=$1; T=$2
  for deck in decks/danlike.nec decks/ward.nec; do for basis in razor-2p bspline; do
    for r in 1 2 3; do for t in base wa; do
      echo "$V/$T $(basename $deck) $basis $t $(one $t $deck $basis $V $T)"
    done; done
  done; done
done
echo DONE
