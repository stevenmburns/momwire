#!/usr/bin/env bash
# momwire#1064 G6 Population A (PLAN.md G6): antennaknobs' standing buried census,
# with main in the "mid" column and the branch in the "main" column. Run after G5.
#
#   systemd-run --user --unit=m1064-g6a -p MemoryMax=24G scratch/1064-fifth-band/run_g6a.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
AKW=${AKW:-/home/smburns/stevenmburns/antennaknobs-wt-1064ak}
export MOMWIRE_MID=${MOMWIRE_MID:-/home/smburns/stevenmburns/momwire-wt-1064-main/src}
export MOMWIRE_MAIN=${MOMWIRE_MAIN:-/home/smburns/stevenmburns/momwire-wt-1064/src}
export NEC5_EXE=${NEC5_EXE:-/home/smburns/nec5-timing/nec5cl-x13-static}
cd "$AKW" || exit 1
{
    echo "start $(date -u +%FT%TZ)"
    echo "antennaknobs $(git -C "$AKW" rev-parse --short HEAD)"
    echo "mid (main) $(git -C "$MOMWIRE_MID" rev-parse --short HEAD)"
    echo "main column (branch) $(git -C "$MOMWIRE_MAIN" rev-parse --short HEAD)"
    echo "nec5 $(sha256sum "$NEC5_EXE" | cut -c1-16)"
} >"$HERE/run_g6a.out"
.venv/bin/python scratch/956-census/popa_run.py --out "$HERE/popa-rows-1064.csv" \
    --raw "$HERE/popa-rows-1064.jsonl" >"$HERE/popa_run_1064.log" 2>&1
echo "rc=$? end $(date -u +%FT%TZ)" >>"$HERE/run_g6a.out"
