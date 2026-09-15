#!/usr/bin/env bash
# Profile one rung: py-spy as the PARENT over the whole process, phases sliced
# afterwards by frame name.
#
# NOT `--native`. Measured on this box: `--native --rate 100` on the 48-radial
# rung ran 29 minutes for a 28 s solve and printed "810.08s behind in sampling,
# results may be inaccurate" — the native unwinder cannot keep up with this
# workload, so its output would be both slow and untrustworthy. Python-level
# sampling is the right tool for a PHASE table anyway: momwire's phases are Python
# functions, and time inside the C++ accelerators is attributed to the Python
# frame that called them, which is exactly the attribution the table wants.
#
# `--pid` attach is not available here: /proc/sys/kernel/yama/ptrace_scope is 1,
# so py-spy may only profile a process it launched, and sudo wants a password on
# this box. The script therefore puts the cold solve inside `_phase_cold()` and
# the warm one inside `_phase_warm()`, and `slice_profile.py` attributes every
# sample by the frames in its stack — exact, and with no clock alignment between
# two tools.
set -u
N=$1
O=${2:-$HOME/nec5-timing/profile-1029}
AK=/home/smburns/stevenmburns/antennaknobs
MW=$AK/momwire
mkdir -p "$O"
G=$(mktemp -d)
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

systemd-run --user --scope -q -p MemoryMax=40G \
  /usr/bin/time -v uvx py-spy record --rate 100 --format speedscope \
    -o "$O/whole-$N.speedscope.json" -- \
    "$AK/.venv/bin/python" "$MW/scratch/1029-profile/profile_rung.py" \
      --radials "$N" --gate-dir "$G" --rss-log "$O/rss-$N.csv" --out "$O/result-$N.json" \
  > "$O/stdout-$N.txt" 2> "$O/time-$N.txt"
rc=$?
rm -rf "$G"
echo "N=$N rc=$rc"
grep -E "Maximum resident set size|Elapsed \(wall" "$O/time-$N.txt" | sed 's/^/  /'
grep -E "Samples|Wrote speedscope" "$O/time-$N.txt" | sed 's/^/  /'
cat "$O/result-$N.json" 2>/dev/null
