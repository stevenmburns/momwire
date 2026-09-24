#!/usr/bin/env bash
# momwire#1131: dense vs route at N radials (default 150), fresh process each,
# above-ground N6LF screen (surface convention, Sommerfeld unless GROUND=...).
#
#   TREE=<this worktree, built> PY=<python> THREADS=4 GROUND=sommerfeld \
#   bash scratch/1131-route-above-ground/run_150.sh [radials=150]
#
# Arms run one at a time, dense -> route -> dense -> route, each under
# /usr/bin/time -v. The dense arm at 150 radials holds the (n, n) Z plus the
# chunked fill's 256 MB transient (n ~ 3000): budget ~1.5 GB. The last lines
# print wall / VmHWM per arm and the route-vs-dense |dZ|/|Z| (bar 1e-9).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
N=${1:-150}
OUT=${OUT:-$HERE/out_$N}
mkdir -p "$OUT"
: "${TREE:?set TREE}" "${PY:?set PY}"
T=${THREADS:-4}
G=${GROUND:-sommerfeld}
for rep in 1 2; do
  for arm in dense route; do
    tag=${arm}_${G}_${N}_r$rep
    load=$(cut -d' ' -f1 /proc/loadavg)
    OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T \
      PYTHONPATH="$TREE/src:$HERE" /usr/bin/time -v "$PY" "$HERE/mem_probe.py" \
      --radials "$N" --arm "$arm" --ground "$G" >"$OUT/$tag.json" 2>"$OUT/$tag.time"
    rc=$?
    rss=$(awk '/Maximum resident/ {print $NF}' "$OUT/$tag.time")
    echo "$tag exit=$rc load=$load maxrss_kB=$rss"
  done
done
"$PY" - "$OUT" <<'PYEOF'
import glob, json, sys
zs = {}
for f in sorted(glob.glob(sys.argv[1] + "/*.json")):
    try:
        d = json.load(open(f))
    except Exception as e:
        print(f, "unreadable:", e); continue
    zs.setdefault(d["arm"], []).append(complex(*d["z_float"]))
    print(f.rsplit("/", 1)[-1], "n", d["n_basis"], "s %.1f" % d["seconds"], "hwm %.0f MB" % d["vmhwm_mb"])
if "dense" in zs and "route" in zs:
    zd, zr = zs["dense"][0], zs["route"][0]
    print("route vs dense |dZ|/|Z| =", abs(zr - zd) / abs(zd), "(bar 1e-9)")
PYEOF
