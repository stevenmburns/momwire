#!/usr/bin/env bash
# momwire#1064 G6 Population B (PLAN.md G6 addendum and re-pin): the census's
# momwire corpus runner over TODAY's translation of the six decks, once on main
# and once on the branch, identical inputs. Run after G5.
#
#   systemd-run --user --unit=m1064-g6b -p MemoryMax=24G scratch/1064-fifth-band/run_g6b.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
AKW=${AKW:-/home/smburns/stevenmburns/antennaknobs-wt-1064ak}
SRC=${SRC:-/home/smburns/nec5-timing/popb-1064/nec5}
MAIN=${MAIN:-/home/smburns/stevenmburns/momwire-wt-1064-main/src}
BRANCH=${BRANCH:-/home/smburns/stevenmburns/momwire-wt-1064/src}
cd "$AKW" || exit 1
echo "start $(date -u +%FT%TZ) antennaknobs $(git -C "$AKW" rev-parse --short HEAD)" >"$HERE/run_g6b.out"
for tree in main branch; do
    if [ "$tree" = main ]; then src="$MAIN"; else src="$BRANCH"; fi
    # The per-deck children inherit this environment, so PYTHONPATH selects momwire.
    PYTHONPATH="$src:$AKW/src" .venv/bin/python scratch/896-census/census_momwire.py \
        --src "$SRC" --report "$HERE/popb_momwire_$tree.jsonl" \
        >"$HERE/popb_momwire_$tree.log" 2>&1
    echo "$tree ($(git -C "$src" rev-parse --short HEAD)) rc=$? $(date -u +%FT%TZ)" >>"$HERE/run_g6b.out"
done
echo "end $(date -u +%FT%TZ)" >>"$HERE/run_g6b.out"
