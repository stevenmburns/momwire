#!/usr/bin/env bash
# momwire#1064 gate runner, one step per invocation, in PLAN.md's order.
#
#   run_gates.sh s0                   step 0, geometry only, all three trees
#   run_gates.sh g4                   G4, the fill counter (branch)
#   run_gates.sh g12 v055|main|branch G1a/G2 capture (branch adds --direct)
#   run_gates.sh g3 avx2|sse2         G3 capture, one accelerator variant
#
# Launch through a transient unit with a script file (systemd-run expands $VAR
# in its own arguments), e.g.
#   systemd-run --user --unit=m1064-g4 -p MemoryMax=24G scratch/1064-fifth-band/run_gates.sh g4
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/home/smburns/stevenmburns/antennaknobs/.venv/bin/python}
V055=${V055:-/home/smburns/stevenmburns/momwire-wt-1064-v055/src}
MAIN=${MAIN:-/home/smburns/stevenmburns/momwire-wt-1064-main/src}
BRANCH=${BRANCH:-/home/smburns/stevenmburns/momwire-wt-1064/src}
cd "$HERE" || exit 1

src_of() {
    case "$1" in
        v055) echo "$V055" ;;
        main) echo "$MAIN" ;;
        branch) echo "$BRANCH" ;;
        *) echo "unknown tree $1" >&2; exit 2 ;;
    esac
}

echo "== $* start $(date -u +%FT%TZ)" >>"$HERE/run_gates.out"
case "${1:-}" in
    s0)
        : >"$HERE/s0_geometry.jsonl"
        for t in v055 main branch; do
            PYTHONPATH="$(src_of "$t")" "$PY" s0_geometry.py --tree "$t" \
                --out "$HERE/s0_geometry.jsonl" >>"$HERE/s0_geometry.log" 2>&1
        done
        ;;
    g4)
        PYTHONPATH="$BRANCH" "$PY" g4_fill_counter.py --out "$HERE/g4_fill_counter.json" \
            >"$HERE/g4_fill_counter.log" 2>&1
        ;;
    g12)
        t="${2:?tree}"
        extra=()
        if [ "$t" = branch ]; then extra=(--direct); fi
        PYTHONPATH="$(src_of "$t")" "$PY" g12_grid.py --tree "$t" --out "$HERE/g12_$t.json" \
            "${extra[@]}" >"$HERE/g12_$t.log" 2>&1
        ;;
    g3)
        v="${2:?variant}"
        MOMWIRE_FORCE_VARIANT="$v" PYTHONPATH="$BRANCH" "$PY" g3_dispatch.py --label "$v" \
            --out "$HERE/g3_$v.json" >"$HERE/g3_$v.log" 2>&1
        ;;
    *)
        echo "usage: run_gates.sh s0 | g4 | g12 <tree> | g3 <variant>" >&2
        exit 2
        ;;
esac
echo "== $* rc=$? end $(date -u +%FT%TZ)" >>"$HERE/run_gates.out"
