#!/usr/bin/env bash
# U9 Amendment 8c runner, one step per invocation, alone on the box. Each step's
# checks are read before the next step launches (PLAN.md Amendment 8c, order):
# check1 is read from e3_check1.log, and the others with `e5_checks.py <step>`.
#
#   systemd-run --user --unit=u9-8c-check1 -p MemoryMax=24G scratch/u9-multi-crossing/run_e8c.sh check1
#   systemd-run --user --unit=u9-8c-nec5 -p MemoryMax=24G scratch/u9-multi-crossing/run_e8c.sh nec5
#   systemd-run --user --unit=u9-8c-mw -p MemoryMax=24G scratch/u9-multi-crossing/run_e8c.sh momwire
#   systemd-run --user --unit=u9-8c-same -p MemoryMax=24G scratch/u9-multi-crossing/run_e8c.sh same
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/home/smburns/stevenmburns/antennaknobs/.venv/bin/python}
AK=${AK:-/home/smburns/stevenmburns/antennaknobs-wt-pin055/src}
MW=${MW:-/home/smburns/stevenmburns/momwire-wt-u9src/src}
NEC5=${NEC5:-/home/smburns/antennas/NEC5-downloads/nec5-linux/nec5cl}
OUT="$HERE/e4_knot.jsonl"
SEPS=(3 5 8 11)
cd "$HERE" || exit 1

solve() {
    echo "== $* start $(date -u +%FT%TZ)" >>"$HERE/run_e8c.out"
    PYTHONPATH="$MW:$AK" "$PY" e4_knot_solve.py "$@" --nec5-exe "$NEC5" --out "$OUT" \
        >>"$HERE/e4_knot.log" 2>&1
    echo "== $* rc=$? end $(date -u +%FT%TZ)" >>"$HERE/run_e8c.out"
}

case "${1:-}" in
    check1)
        echo "== check1 start $(date -u +%FT%TZ)" >>"$HERE/run_e8c.out"
        PYTHONPATH="$MW:$AK" "$PY" e3_knot_decks.py --check1 --out "$HERE/e3_check1.json" \
            >"$HERE/e3_check1.log" 2>&1
        echo "== check1 rc=$? end $(date -u +%FT%TZ)" >>"$HERE/run_e8c.out"
        ;;
    nec5)
        for d in "${SEPS[@]}"; do
            for rung in r1 far3 all3; do solve --stem "two_node_d${d}_${rung}" --engine nec5; done
        done
        for rung in r1 far3; do solve --stem "ctrl_d3_${rung}" --engine nec5; done
        ;;
    momwire)
        for d in "${SEPS[@]}"; do
            for rung in r1 far3; do solve --stem "two_node_d${d}_${rung}" --engine momwire; done
        done
        for rung in r1 far3; do solve --stem "ctrl_d3_${rung}" --engine momwire; done
        ;;
    same)
        for d in "${SEPS[@]}"; do solve --stem "two_node_d${d}_r1" --engine momwire --corner same; done
        ;;
    *)
        echo "usage: run_e8c.sh check1|nec5|momwire|same" >&2
        exit 2
        ;;
esac
