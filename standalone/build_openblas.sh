#!/bin/bash
# The static OpenBLAS momwire-razor2p links (Linux):
#
#     standalone/build_openblas.sh WORKDIR      -> WORKDIR/libmwxp.a
#
# Pinned to the build SciPy's wheels bundle (`scipy.libs/libscipy_openblas-*`,
# which reports "OpenBLAS 0.3.31.dev DYNAMIC_ARCH NO_AFFINITY"): the commit
# scipy-openblas32 0.3.31.22.x builds (v0.3.31-22-g5ffbf38b4), LP64, pthreads
# (USE_OPENMP=0: an OpenMP build at 16 threads factors differently from SciPy's
# library, measured in the 10-14 spike), and that package's make flags. So the
# executable's LU is SciPy's to the bit, and the C++ route and the embedded
# Python both run on it (`sys._mw_host_lapack`).
#
# ONE DEPARTURE from SciPy's flags: DYNAMIC_LIST leaves out SKYLAKEX (and so,
# by OpenBLAS's own fallbacks in driver/others/dynamic.c, COOPERLAKE and
# SAPPHIRERAPIDS too): on an AVX-512 CPU the library runs its HASWELL kernels,
# the ones every AVX2 CPU without AVX-512 runs. AVX-512 bought no LU speed
# here (the 10-12 microbench), and its kernels round differently: on
# Rocketlake they moved 21 of the 108 golden printouts, every one a network
# deck. Without SKYLAKEX in the list the library cannot select those kernels
# by detection or by OPENBLAS_CORETYPE, so the executable's answer does not
# depend on whether the CPU has AVX-512. On a CPU without AVX-512 nothing
# changes: the kernels it detects are all still in the list. SciPy's own
# library (the launcher's) is held to the same kernels by the
# OPENBLAS_CORETYPE=Haswell `momwire_serve_client.pin_cpu_env` exports on an
# AVX-512 CPU.
#
# Every global symbol the archive defines is then renamed `mwx_<name>`
# (`objcopy --redefine-syms`, references inside the archive included), so the
# executable's copy can never bind to, or be bound by, the OpenBLAS builds
# numpy and scipy load into the same process.
set -euo pipefail
W=$(realpath -m "${1:?usage: build_openblas.sh WORKDIR}")
COMMIT=5ffbf38b41a1fe494d038efcb09cf22f4d527c22
mkdir -p "$W"
cd "$W"
[ -d src ] || git clone -q https://github.com/OpenMathLib/OpenBLAS src
cd src
git checkout -q "$COMMIT"
git clean -qfdx
CFLAGS="-fvisibility=protected -Wno-uninitialized" make -j"$(nproc)" \
    BUFFERSIZE=20 DYNAMIC_ARCH=1 QUIET_MAKE=1 USE_OPENMP=0 NUM_THREADS=64 \
    DYNAMIC_LIST="PRESCOTT NEHALEM SANDYBRIDGE HASWELL" BINARY=64 \
    TARGET=PRESCOTT NOFORTRAN=1 NO_SHARED=1 > "$W/build.log" 2>&1
ARCHIVE=$(ls libopenblas*.a | head -1)
nm -g --defined-only "$ARCHIVE" 2>/dev/null |
    awk 'NF == 3 { print $3 " mwx_" $3 }' | sort -u > "$W/syms.txt"
objcopy --redefine-syms="$W/syms.txt" "$ARCHIVE" "$W/libmwxp.a"
nm -g --defined-only "$W/libmwxp.a" 2>/dev/null > "$W/defined.txt"
for s in mwx_zgetrf_ mwx_zgetrs_ mwx_dgetrf_ mwx_dgetrs_ mwx_zgemm_ mwx_zgemv_ \
         mwx_openblas_set_num_threads; do
    grep -q " T $s\$" "$W/defined.txt" || { echo "missing $s in $W/libmwxp.a" >&2; exit 1; }
done
echo "$W/libmwxp.a ($(wc -l < "$W/syms.txt") symbols renamed)"
