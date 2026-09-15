#!/usr/bin/env bash
# momwire#1064 D5 (PLAN.md): per-node cost and a cold-solve profile, main against
# the branch, alternating trees. Run ALONE on the box, after G1a-sse2 and G6.
#
#   systemd-run --user --unit=m1064-d5 -p MemoryMax=24G scratch/1064-fifth-band/run_d5.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/home/smburns/stevenmburns/antennaknobs/.venv/bin/python}
AK=${AK:-/home/smburns/stevenmburns/antennaknobs-wt-1064ak/src}
MAIN=${MAIN:-/home/smburns/stevenmburns/momwire-wt-1064-main/src}
BRANCH=${BRANCH:-/home/smburns/stevenmburns/momwire-wt-1064/src}
cd "$HERE" || exit 1
: >"$HERE/d5_nodecost.jsonl"
: >"$HERE/d5_profile.jsonl"
echo "start $(date -u +%FT%TZ)" >"$HERE/run_d5.out"
for rep in 1 2 3; do
    if [ $((rep % 2)) -eq 1 ]; then order=(main branch); else order=(branch main); fi
    for tree in "${order[@]}"; do
        if [ "$tree" = main ]; then src="$MAIN"; else src="$BRANCH"; fi
        PYTHONPATH="$src" "$PY" d5_nodecost.py --tree "$tree" --rep "$rep" \
            --out "$HERE/d5_nodecost.jsonl" >>"$HERE/d5.log" 2>&1
        echo "nodecost rep=$rep tree=$tree rc=$? $(date -u +%FT%TZ)" >>"$HERE/run_d5.out"
    done
done
for rep in 1 2; do
    if [ $((rep % 2)) -eq 1 ]; then order=(main branch); else order=(branch main); fi
    for tree in "${order[@]}"; do
        if [ "$tree" = main ]; then src="$MAIN"; else src="$BRANCH"; fi
        PYTHONPATH="$src:$AK" "$PY" d5_profile.py --deck dipole1mm --tree "$tree" --rep "$rep" \
            --out "$HERE/d5_profile.jsonl" >>"$HERE/d5.log" 2>&1
        echo "profile rep=$rep tree=$tree rc=$? $(date -u +%FT%TZ)" >>"$HERE/run_d5.out"
    done
done
echo "done $(date -u +%FT%TZ)" >>"$HERE/run_d5.out"
