#!/usr/bin/env bash
# U9: any of momwire's Makefile pytest lanes, run locally on the src branch,
# sequentially, into `lanes_<lane>.log` and `run_lanes_<lanes>.out`.
#
#   systemd-run --user --unit=u9-lanes -p MemoryMax=24G \
#       scratch/u9-multi-crossing/run_lanes.sh test integration
#
# A separate file from run_g5.sh on purpose: that script was still executing,
# and bash reads a running script incrementally, so editing it in place could
# have broken the live run.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${SRC:-/home/smburns/stevenmburns/momwire-wt-u9src}
PY_BIN=${PY_BIN:-/home/smburns/stevenmburns/antennaknobs/.venv/bin/python}
LANES=("$@")
[ ${#LANES[@]} -gt 0 ] || LANES=(test integration)
TAG=$(printf "%s_" "${LANES[@]}")
OUT="$HERE/run_lanes_${TAG%_}.out"
cd "$SRC" || exit 1
export PYTHONPATH="$SRC/src"

{
    echo "src=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
    echo "dirty=$(git status --porcelain | wc -l)"
    "$PY_BIN" -c "import momwire; print('momwire', momwire.__file__)" 2>/dev/null
} >"$OUT"

for lane in "${LANES[@]}"; do
    echo "== $lane start $(date -u +%FT%TZ)" >>"$OUT"
    make "$lane" PYTHON="$PY_BIN" PY="$PY_BIN -m" >"$HERE/lanes_$lane.log" 2>&1
    echo "== $lane rc=$? end $(date -u +%FT%TZ)" >>"$OUT"
    grep -E "^(FAILED|ERROR) |[0-9]+ (passed|failed)|OVER HARD" "$HERE/lanes_$lane.log" | tail -40 >>"$OUT"
done
echo "== done $(date -u +%FT%TZ)" >>"$OUT"
