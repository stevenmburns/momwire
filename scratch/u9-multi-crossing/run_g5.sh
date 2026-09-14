#!/usr/bin/env bash
# U9 PR gate G5 (PLAN.md): momwire's slow lane and crossgate, run locally on the
# src branch, sequentially (the Makefile's own rule: every pytest lane already
# saturates the box through addopts' `-n auto`).
#
#   systemd-run --user --unit=u9-g5 -p MemoryMax=24G scratch/u9-multi-crossing/run_g5.sh
#
# A script file rather than an inline `bash -c`: a `systemd-run --unit` command
# line has its $VAR / ${VAR} expanded by systemd before bash sees it.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${SRC:-/home/smburns/stevenmburns/momwire-wt-u9src}
PY_BIN=${PY_BIN:-/home/smburns/stevenmburns/antennaknobs/.venv/bin/python}
OUT="$HERE/run_g5.out"
cd "$SRC" || exit 1
export PYTHONPATH="$SRC/src"

{
    echo "src=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
    echo "dirty=$(git status --porcelain | wc -l)"
    "$PY_BIN" -c "import momwire; print('momwire', momwire.__file__)" 2>/dev/null
} >"$OUT"

for lane in slow crossgate; do
    echo "== $lane start $(date -u +%FT%TZ)" >>"$OUT"
    make "$lane" PYTHON="$PY_BIN" PY="$PY_BIN -m" >"$HERE/g5_$lane.log" 2>&1
    echo "== $lane rc=$? end $(date -u +%FT%TZ)" >>"$OUT"
    grep -E "^(FAILED|ERROR) |[0-9]+ (passed|failed)" "$HERE/g5_$lane.log" | tail -40 >>"$OUT"
done
echo "== done $(date -u +%FT%TZ)" >>"$OUT"
