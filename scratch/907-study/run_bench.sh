set -e
cd /home/smburns/stevenmburns/momwire-813
for T in 8 1; do
  # settle before each arm so a previous run's heat is not measured
  while [ "$(awk '{print ($1<0.8)?1:0}' /proc/loadavg)" != "1" ]; do sleep 10; done
  echo "######## OMP_NUM_THREADS=$T ########"
  OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T BENCH_REPS=7 \
    .venv/bin/python scratch/907-study/bench_907.py
done
