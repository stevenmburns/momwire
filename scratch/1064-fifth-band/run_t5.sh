#!/usr/bin/env bash
# momwire#1064 G5 re-timing (PLAN.md G5). The three trees rotate order every
# repeat, so a thermal drift cannot land on one tree. Run ALONE on the box.
#
#   systemd-run --user --unit=m1064-t5 -p MemoryMax=24G scratch/1064-fifth-band/run_t5.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/home/smburns/stevenmburns/antennaknobs/.venv/bin/python}
AK=${AK:-/home/smburns/stevenmburns/antennaknobs-wt-1064ak/src}
V055=${V055:-/home/smburns/stevenmburns/momwire-wt-1064-v055/src}
MAIN=${MAIN:-/home/smburns/stevenmburns/momwire-wt-1064-main/src}
BRANCH=${BRANCH:-/home/smburns/stevenmburns/momwire-wt-1064/src}
OUT="$HERE/t5_fill_cost.jsonl"
REPS=${REPS:-3}
DECKS=(buried_dipole brv_default ebc_default brv48 dipole935 brv_corner dipole1mm twonode11)
cd "$HERE" || exit 1
: >"$OUT"
echo "start $(date -u +%FT%TZ)" >"$HERE/run_t5.out"
for rep in $(seq 1 "$REPS"); do
    case $((rep % 3)) in
        1) order=(v055 main branch) ;;
        2) order=(main branch v055) ;;
        *) order=(branch v055 main) ;;
    esac
    for deck in "${DECKS[@]}"; do
        for tree in "${order[@]}"; do
            # 0.55.0 refuses both decks under 0.05 deg (step 0 reads that).
            if [ "$tree" = v055 ] && { [ "$deck" = dipole1mm ] || [ "$deck" = twonode11 ]; }; then
                continue
            fi
            case "$tree" in
                v055) src="$V055" ;;
                main) src="$MAIN" ;;
                *) src="$BRANCH" ;;
            esac
            PYTHONPATH="$src:$AK" "$PY" t5_fill_cost.py --deck "$deck" --rep "$rep" \
                --tree "$tree" --out "$OUT" >>"$HERE/t5_fill_cost.log" 2>&1
            echo "rep=$rep deck=$deck tree=$tree rc=$? $(date -u +%FT%TZ)" >>"$HERE/run_t5.out"
        done
    done
done
echo "done $(date -u +%FT%TZ)" >>"$HERE/run_t5.out"
