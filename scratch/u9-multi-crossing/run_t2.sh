#!/usr/bin/env bash
# U9 fill-cost timing runner (PLAN.md Amendment 5). Main and the src branch
# alternate within every (repeat, deck), and the order flips each repeat, so a
# thermal drift cannot land on one tree. Run ALONE on the box (not beside G5).
#
#   systemd-run --user --unit=u9-t2 -p MemoryMax=24G scratch/u9-multi-crossing/run_t2.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/home/smburns/stevenmburns/antennaknobs/.venv/bin/python}
AK=${AK:-/home/smburns/stevenmburns/antennaknobs-wt-pin055/src}
MAIN=${MAIN:-/home/smburns/stevenmburns/momwire-wt-u9/src}
BRANCH=${BRANCH:-/home/smburns/stevenmburns/momwire-wt-u9src/src}
OUT="$HERE/t2_fill_cost.jsonl"
REPS=${REPS:-3}
DECKS=(buried_dipole brv_default ebc_default dipole935 brv_corner brv48)
cd "$HERE" || exit 1
: >"$OUT"
echo "start $(date -u +%FT%TZ)" >"$HERE/run_t2.out"
for rep in $(seq 1 "$REPS"); do
    if [ $((rep % 2)) -eq 1 ]; then order=("$MAIN" "$BRANCH"); else order=("$BRANCH" "$MAIN"); fi
    for deck in "${DECKS[@]}"; do
        for tree in "${order[@]}"; do
            PYTHONPATH="$tree:$AK" "$PY" t2_fill_cost.py --deck "$deck" --rep "$rep" --out "$OUT" \
                >>"$HERE/t2_fill_cost.log" 2>&1
            echo "rep=$rep deck=$deck tree=$tree rc=$? $(date -u +%FT%TZ)" >>"$HERE/run_t2.out"
        done
    done
done
echo "done $(date -u +%FT%TZ)" >>"$HERE/run_t2.out"
