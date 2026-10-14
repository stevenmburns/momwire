#!/usr/bin/env bash
# Link the standalone razor chain driver against the SAME core objects the
# extension links (explore/core-binding-split). No Python, no pybind11.
#
#   make build                                   # compiles the core TUs with the extension flags
#   bash scripts/core_split/build_driver.sh      # -> build/core_split/razor_chain_driver
#   python scripts/core_split/capture_chain.py /tmp/cap 108
#   build/core_split/razor_chain_driver /tmp/cap 200
#   python scripts/core_split/capture_chain.py /tmp/cap --compare
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
variant=${VARIANT:-_accelerators_avx2}
tmp=$(ls -d "$here"/build/temp.*/"$variant"/src/momwire)
out="$here/build/core_split"
mkdir -p "$out"
rm -f "$out/libmomwire_core.a"
ar rcs "$out/libmomwire_core.a" "$tmp/_razor_core.o" "$tmp/_razor_t2_core.o" "$tmp/_accel_razor_cplx.o"
g++ -O2 -std=gnu++17 -fopenmp -I "$here/src/momwire" \
    "$here/scripts/core_split/razor_chain_driver.cpp" "$out/libmomwire_core.a" \
    -fopenmp -lmvec -o "$out/razor_chain_driver"
echo "$out/razor_chain_driver"
