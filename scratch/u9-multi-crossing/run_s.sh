#!/usr/bin/env bash
# U9 route 2 screen runner (PLAN.md Amendment 2): S1 panels, then S2/S3 per lattice.
# Launch as a transient service, e.g.
#   systemd-run --user --unit=u9-s-screen -p MemoryMax=24G \
#       scratch/u9-multi-crossing/run_s.sh
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
    "$PY" s_table_screen.py "$@" --out "s_$name.json" >"s_$name.log" 2>&1
    echo "== $name rc=$? end $(date -u +%FT%TZ)"
}

run panels panels
for lat in L2 L2p L3; do run "interp_$lat" interp --lattice "$lat"; done
echo "== done $(date -u +%FT%TZ)"
