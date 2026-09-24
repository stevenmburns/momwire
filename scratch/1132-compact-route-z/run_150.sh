#!/usr/bin/env bash
# momwire#1132: the 150-radial route, before/after, fresh process each.
#
#   BASE=<momwire worktree at origin/main, built>  MINE=<this branch, built> \
#   PY=<python with antennaknobs installed> THREADS=4 \
#   bash scratch/1132-compact-route-z/run_150.sh [radials=150]
#
# Both trees must be built the same way on this box (`make build`). Each arm
# is one fresh `attr_probe.py` process under /usr/bin/time -v: peak RSS
# (VmHWM and time's Maximum resident), the main thread's stack at the peak,
# and Z_in. Arms run one at a time, base -> mine -> base -> mine. The last
# line compares Z_in as hex across every arm: it must be ONE value.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
N=${1:-150}
OUT=${OUT:-$HERE/out_$N}
mkdir -p "$OUT"
: "${BASE:?set BASE}" "${MINE:?set MINE}" "${PY:?set PY}"
T=${THREADS:-4}
for rep in 1 2; do
  for arm in base mine; do
    tree=$BASE; [ "$arm" = mine ] && tree=$MINE
    tag=${arm}_${N}_r$rep
    load=$(cut -d' ' -f1 /proc/loadavg)
    OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T \
      PYTHONPATH="$tree/src" /usr/bin/time -v "$PY" "$HERE/attr_probe.py" \
      --radials "$N" --step 50 >"$OUT/$tag.json" 2>"$OUT/$tag.time"
    rc=$?
    rss=$(awk '/Maximum resident/ {print $NF}' "$OUT/$tag.time")
    echo "$tag exit=$rc load=$load maxrss_kB=$rss"
  done
done
"$PY" - "$OUT" <<'PYEOF'
import glob, json, sys
zs = {}
for f in sorted(glob.glob(sys.argv[1] + "/*.json")):
    d = json.load(open(f))
    zs[f.rsplit("/", 1)[-1]] = (d["z"][0].hex(), d["z"][1].hex())
    print(f.rsplit("/", 1)[-1], "hwm", d["vmhwm_mb"], "MB  s", d["seconds"],
          " peak at", d["peak_stack"][-3:])
print("Z_in values across arms:", len(set(zs.values())), "(must be 1)")
PYEOF
