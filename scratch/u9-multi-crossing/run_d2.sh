#!/usr/bin/env bash
# U9 (d) runner (PLAN.md Amendment 3), one step per invocation, alone on the box:
#
#   systemd-run --user --unit=u9-d2-cost -p MemoryMax=24G scratch/u9-multi-crossing/run_d2.sh cost
#   systemd-run --user --unit=u9-d2-rest -p MemoryMax=24G scratch/u9-multi-crossing/run_d2.sh rest
#   systemd-run --user --unit=u9-d2-mwfull -p MemoryMax=24G scratch/u9-multi-crossing/run_d2.sh mw_full_far3
#
# `mw_full_far3` is launched only after the cost step's projection passes
# Amendment 3's rule 1 (projected peak <= 20 GB and time <= 6 h), and that
# reading is recorded in PLAN.md before the launch.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/home/smburns/stevenmburns/antennaknobs/.venv/bin/python}
AK=${AK:-/home/smburns/stevenmburns/antennaknobs-wt-pin055/src}
MW=${MW:-/home/smburns/stevenmburns/momwire-wt-u9src/src}
NEC5=${NEC5:-/home/smburns/antennas/NEC5-downloads/nec5-linux/nec5cl}
OUT="$HERE/d2_lpda_solve.jsonl"
cd "$HERE" || exit 1

solve() {
    local deck=$1 rung=$2 engine=$3
    echo "== $deck $rung $engine start $(date -u +%FT%TZ)" >>"$HERE/run_d2.out"
    PYTHONPATH="$MW:$AK" "$PY" d2_lpda_solve.py --deck "$deck" --rung "$rung" \
        --engine "$engine" --nec5-exe "$NEC5" --out "$OUT" >>"$HERE/d2_lpda_solve.log" 2>&1
    echo "== $deck $rung $engine rc=$? end $(date -u +%FT%TZ)" >>"$HERE/run_d2.out"
}

case "${1:-}" in
    cost)
        solve one r1 momwire
        solve full r1 momwire
        solve one r1 nec5
        solve full r1 nec5
        ;;
    rest)
        solve one far3 nec5
        solve full far3 nec5
        solve one far3 momwire
        ;;
    route_check_one)
        # Amendment 6: the route's reciprocity check on the re-spelled one-node
        # deck. Launched only after the native NEC-5 check passed; momwire's
        # one-node Z is not run until this row is read as passing.
        solve one r1 nec5
        ;;
    mw_one_r1)
        solve one r1 momwire
        ;;
    mw_full_far3)
        solve full far3 momwire
        ;;
    *)
        echo "usage: run_d2.sh cost|rest|route_check_one|mw_one_r1|mw_full_far3" >&2
        exit 2
        ;;
esac
