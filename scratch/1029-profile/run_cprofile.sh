#!/usr/bin/env bash
# One cProfile'd warm solve per rung. cProfile, not a sampler — see the note in
# profile_rung.py: py-spy costs 5.6x here without --native and is unusable with
# it, while cProfile costs 2.2 % (measured at 48 radials: warm 13.65 s against
# 13.35 s unprofiled).
set -u
O=${O:-$HOME/nec5-timing/profile-1029}
AK=/home/smburns/stevenmburns/antennaknobs
MW=$AK/momwire
mkdir -p "$O"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
for N in "$@"; do
  G=$(mktemp -d)
  echo "=== N=$N ($(date +%T)) ==="
  systemd-run --user --scope -q -p MemoryMax=40G /usr/bin/time -v \
    "$AK/.venv/bin/python" "$MW/scratch/1029-profile/profile_rung.py" \
      --radials "$N" --gate-dir "$G" --rss-log "$O/rss-$N.csv" \
      --out "$O/result-$N.json" --cprofile "$O/warm-$N.prof" 2> "$O/time-$N.txt"
  grep -E "Elapsed \(wall|Maximum resident" "$O/time-$N.txt" | sed 's/^/  /'
  rm -rf "$G"
done
echo CPROFILEDONE
