#!/bin/bash
# 41-frequency sweep in ONE resident server per (tree, deck, anchors mode):
# launcher calls against one runtime dir, cold Sommerfeld cache at the start.
# argv: variant threads trees...
V=$1; T=$2; shift 2
cd ~/u4/watail
waitq() { until awk "BEGIN{exit !($(cut -d" " -f1 /proc/loadavg) < 0.5)}" && [ "$(ps -eo args | grep -cE "[p]ytest|[n]ec5cl|[g]ate_lm|[m]omwire-nec5")" = 0 ]; do sleep 15; done; }
for deck in dan ward; do for mode in off auto; do for t in "$@"; do
  waitq
  tag=$t-$deck-$mode-$V$T; rt=/tmp/wa-rt-$tag; c=/tmp/wa-cache-$tag; o=/tmp/wa-out-$tag
  rm -rf $rt $c $o; mkdir -p $rt $c $o
  l0=$(cut -d" " -f1 /proc/loadavg); times=""
  for i in $(seq 0 40); do
    s=$(date +%s.%N)
    MOMWIRE_BELOW_ANCHORS=$mode OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MOMWIRE_FORCE_VARIANT=$V MOMWIRE_SOMM_CACHE_DIR=$c \
      MOMWIRE_PORTAL_RUNTIME_DIR=$rt XDG_RUNTIME_DIR=$rt $t/.venv/bin/momwire-nec5-razor-2p decks/$deck/f$i.nec $o/$i.out > /dev/null 2>&1
    e=$(date +%s.%N); times="$times $(echo "$e - $s" | bc)"
  done
  l1=$(cut -d" " -f1 /proc/loadavg)
  z=$(grep -a -A4 "ANTENNA INPUT" $o/20.out | tail -1 | tr -s " " | cut -d" " -f8-9)
  echo "$tag load $l0/$l1 z20 $z times$times"
  for p in $(pgrep -u $USER -f "momwire"); do grep -q "wa-rt-$tag" /proc/$p/environ 2>/dev/null && echo $p; done > /tmp/wa-pids-$tag
  for p in $(cat /tmp/wa-pids-$tag); do kill $p; done
  rm -rf $rt $c
done; done; done
echo DONE
