#!/bin/bash
# Seeded-bug red controls for #1410, run in the change clone; each reverted.
cd ~/stevenmburns/mw1410/change2
F=src/momwire/_accel_exact_kernel.cpp
run() {
  name="$1"; from="$2"; to="$3"
  python3 - "$F" "$from" "$to" <<'PY'
import sys
p, a, b = sys.argv[1:]
s = open(p).read()
assert s.count(a) == 1, (a, s.count(a))
open(p, "w").write(s.replace(a, b))
PY
  PATH=$HOME/.local/bin:$PATH make build > ../red2-build.log 2>&1 || { echo "$name: BUILD FAILED"; git checkout -q -- $F; return; }
  .venv/bin/python -m pytest tests/test_exact_kernel_accel_1410.py tests/test_exact_kernel_rows_1411.py tests/test_exact_kernel_1408.py -m "not memgate" -q -n 4 -p no:warnings > ../red2-$name.log 2>&1
  echo "$name: exit $? :: $(tail -1 ../red2-$name.log)"
  git checkout -q -- $F
}
run transpose 'X[p * MAXND + q] = o.T ? V[q * nd + p] : V[p * nd + q];' 'X[p * MAXND + q] = V[p * nd + q];'
run cache_slot 'slot[ii * n + j] = cache_.find(o.key);' 'slot[ii * n + j] = std::min<int64_t>(cache_.find(o.key) + 1, static_cast<int64_t>(cache_.size()) - 1);'
run window_r0 'const int64_t i = r0 + ii;
            Canon oc;' 'const int64_t i = ii;
            Canon oc;'
run far_power 'y / std::pow(r, 2 * n + 1);' 'y / std::pow(r, 2 * n);'
run reversal_sign '((r & 1) ? -1.0 : 1.0)' '1.0'
PATH=$HOME/.local/bin:$PATH make build > ../red2-build.log 2>&1; echo "restored build $?"; git status --short
