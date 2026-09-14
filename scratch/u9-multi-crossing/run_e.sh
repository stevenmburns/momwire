#!/usr/bin/env bash
# U9 Amendment 8 runner, one step per invocation, alone on the box. Each step's
# checks are read before the next step launches (PLAN.md Amendment 8, order).
#
#   systemd-run --user --unit=u9-e-nec5 -p MemoryMax=24G scratch/u9-multi-crossing/run_e.sh nec5
#   systemd-run --user --unit=u9-e-mw -p MemoryMax=24G scratch/u9-multi-crossing/run_e.sh momwire
#   systemd-run --user --unit=u9-e-same -p MemoryMax=24G scratch/u9-multi-crossing/run_e.sh same
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/home/smburns/stevenmburns/antennaknobs/.venv/bin/python}
AK=${AK:-/home/smburns/stevenmburns/antennaknobs-wt-pin055/src}
MW=${MW:-/home/smburns/stevenmburns/momwire-wt-u9src/src}
NEC5=${NEC5:-/home/smburns/antennas/NEC5-downloads/nec5-linux/nec5cl}
OUT="$HERE/e1_two_node.jsonl"
SEPS=(3 5 8 11)
cd "$HERE" || exit 1

solve() {
    echo "== $* start $(date -u +%FT%TZ)" >>"$HERE/run_e.out"
    PYTHONPATH="$MW:$AK" "$PY" e1_two_node_solve.py "$@" --nec5-exe "$NEC5" --out "$OUT" \
        >>"$HERE/e1_two_node.log" 2>&1
    echo "== $* rc=$? end $(date -u +%FT%TZ)" >>"$HERE/run_e.out"
}

case "${1:-}" in
    nec5)
        for d in "${SEPS[@]}"; do
            for rung in r1 far3 all3; do solve --d "$d" --rung "$rung" --engine nec5; done
        done
        ;;
    momwire)
        for d in "${SEPS[@]}"; do
            for rung in r1 far3; do solve --d "$d" --rung "$rung" --engine momwire; done
        done
        ;;
    same)
        for d in "${SEPS[@]}"; do solve --d "$d" --rung r1 --engine momwire --corner same; done
        ;;
    *)
        echo "usage: run_e.sh nec5|momwire|same" >&2
        exit 2
        ;;
esac
