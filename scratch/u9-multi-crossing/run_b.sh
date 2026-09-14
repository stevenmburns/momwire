#!/usr/bin/env bash
# U9 (b) runner (PLAN.md Amendment 1): guards first, and nothing else if they miss.
# Launch as a transient service, e.g.
#   systemd-run --user --unit=u9-b-collapse -p MemoryMax=24G \
#       scratch/u9-multi-crossing/run_b.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(cd "$HERE/../.." && pwd)
PY=${PY:-/home/smburns/stevenmburns/antennaknobs/.venv/bin/python}
export PYTHONPATH="$W/src"
cd "$HERE" || exit 1

run() {
    local name=$1
    shift
    echo "== $name start $(date -u +%FT%TZ)"
    "$PY" b_collapse.py "$@" --out "b_$name.json" >"b_$name.log" 2>&1
    local rc=$?
    echo "== $name rc=$rc end $(date -u +%FT%TZ)"
    return $rc
}

run guards guards || {
    echo "== guards missed: stopping before any (b) Z"
    exit 3
}
for sp in A B; do run "${sp}12_cross" run --spelling "$sp" --d 12 --mode cross; done
for sp in A B; do run "${sp}12_same" run --spelling "$sp" --d 12 --mode same; done
for sp in A B; do run "${sp}12_blind" run --spelling "$sp" --d 12 --mode blind; done
for sp in A B; do run "${sp}1_cross" run --spelling "$sp" --d 1 --mode cross; done
echo "== done $(date -u +%FT%TZ)"
