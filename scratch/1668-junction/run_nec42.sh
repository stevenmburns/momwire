#!/bin/bash
# usage: run_nec42.sh <local dir with *.nec>  -> writes <deck>.out beside each
# Runs licensed NEC-4.2 (release build, sha 5fc0d508) on Haswell as a black box.
set -e
d=$(realpath "$1"); tag=$(basename "$d")
H=192.168.1.198; R=/tmp/nec42run-1668/$tag
ssh $H "rm -rf $R && mkdir -p $R"
scp -q "$d"/*.nec $H:$R/
ssh $H "cd $R && sha256sum ~/stevenmburns/bench-nec42/exe/nec42-release | cut -c1-8 > SHA && for f in *.nec; do b=\${f%.nec}; mkdir -p w_\$b; cp \$f w_\$b/m.nec; (cd w_\$b && OMP_NUM_THREADS=1 ~/stevenmburns/bench-nec42/exe/nec42-release m.nec m.out >/dev/null 2>&1; cp m.out ../\$b.out); done; cat SHA"
scp -q "$H:$R/*.out" "$d"/
