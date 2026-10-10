#!/bin/bash
cd ~/stevenmburns/mw1410
out=timings-$(date +%H%M).txt
for side in ${SIDES:-base change}; do
  echo "== $side $(cd $side && git rev-parse --short HEAD)" >> $out
  for c in dipole301 nbs301 fat1000 fat3000-fill; do
    echo "  uptime before: $(uptime)" >> $out; (cd $side && .venv/bin/python ../timings.py $c 3) >> $out 2>&1; echo "  uptime after: $(uptime)" >> $out
  done
  echo "  uptime before: $(uptime)" >> $out; (cd $side && .venv/bin/python ../timings.py fat3000 2) >> $out 2>&1; echo "  uptime after: $(uptime)" >> $out
  for c in fat3000 fat3000-fill; do
    (cd $side && /usr/bin/time -v .venv/bin/python ../timings.py $c 1) > rss.tmp 2>&1
    echo "  uptime: $(uptime)" >> $out; echo "RSS $side $c: $(grep 'Maximum resident' rss.tmp) :: $(grep total rss.tmp)" >> $out
  done
done
echo DONE >> $out
echo $out
