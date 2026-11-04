#!/bin/bash
# The two static OpenBLAS archives the Linux executables link:
#
#     standalone/build_openblas.sh WORKDIR  -> WORKDIR/libmwxp.a   SciPy's build, mwx_
#                                           -> WORKDIR/libmwnp.a   numpy's build, mwn_
#
# WHY TWO (momwire#1751). The launcher the executables replay runs on two
# OpenBLAS builds, and they are different releases: numpy's wheel bundles
# scipy-openblas64 (the products: matmul, dot, numpy.linalg.solve) and SciPy's
# bundles scipy-openblas32 (the LAPACK: lu_factor/lu_solve, the symmetric
# factor, the pivoted QR, the expert solve). One library cannot answer as
# both once OpenBLAS threads a call: zgemm at 2+ threads, and SciPy's zsytrf
# (which blocks through zgemm) at 2+ threads from n ~ 200, round differently
# between the two releases (measured on Rocketlake, momwire#1751). Before
# #1751 the one archive was SciPy's, so an executable's threaded products
# were not numpy's: 82 of the 324 NEC-4.2 crossing decks differed at 8
# threads (sg42). So each library the launcher calls is here at its own
# commit, and blas.h sends each call to the library the launcher's call goes
# to: MWX() SciPy's, MWN() numpy's.
#
#   numpy 2.5.4: scipy-openblas64 0.3.34.106.0 (numpy's
#     requirements/ci_requirements.txt), reports "OpenBLAS 0.3.34.106.0
#     USE64BITINT DYNAMIC_ARCH NO_AFFINITY"; built by MacPython/openblas-libs
#     at tag v0.3.34.106.0 from OpenBLAS v0.3.34-106-g446c436e.
#   SciPy 1.18.1: scipy-openblas32 0.3.31.22.0 (SciPy's
#     requirements/openblas.txt), reports "OpenBLAS 0.3.31.dev DYNAMIC_ARCH
#     NO_AFFINITY"; built at tag v0.3.31.22.0 from v0.3.31-22-g5ffbf38b4.
#
# tests/test_standalone_openblas_pin.py fails when the installed numpy or
# SciPy bundles a release other than the one recorded here, so a numpy or
# SciPy bump moves this file with it, or fails loudly. standalone.yml's
# Windows jobs read NUMPY_OPENBLAS_VERSION from here too.
#
# Both are built with the make flags of those tags' tools/build_steps.sh
# (identical in the two): pthreads (USE_OPENMP=0: an OpenMP build at 16
# threads factors differently, measured in the 10-14 spike), BUFFERSIZE=20,
# NUM_THREADS=64, DYNAMIC_ARCH. Both LP64 (blas.h's interface): numpy's
# library is ILP64 (INTERFACE64=1), which widens the integers of the entry
# points and changes no kernel; the LP64 build of numpy's tag is
# scipy-openblas32 0.3.34.106.0, which numpy pins beside the 64-bit one.
#
# NOFORTRAN=1 (OpenBLAS's C translation of the reference LAPACK, where the
# wheels compile the Fortran) and NO_SHARED=1 are build-environment choices
# this script has always made.
#
# ONE DEPARTURE from the wheels' flags: DYNAMIC_LIST leaves out SKYLAKEX (and so,
# by OpenBLAS's own fallbacks in driver/others/dynamic.c, COOPERLAKE and
# SAPPHIRERAPIDS too): on an AVX-512 CPU the library runs its HASWELL kernels,
# the ones every AVX2 CPU without AVX-512 runs. AVX-512 bought no LU speed
# here (the 10-12 microbench), and its kernels round differently: on
# Rocketlake they moved 21 of the 108 golden printouts, every one a network
# deck. Without SKYLAKEX in the list the library cannot select those kernels
# by detection or by OPENBLAS_CORETYPE, so the executable's answer does not
# depend on whether the CPU has AVX-512. On a CPU without AVX-512 nothing
# changes: the kernels it detects are all still in the list. numpy's and
# SciPy's own libraries (the launcher's) are held to the same kernels by the
# OPENBLAS_CORETYPE=Haswell `momwire_serve_client.pin_cpu_env` exports on an
# AVX-512 CPU.
#
# Every global symbol an archive defines is then renamed `<prefix><name>`
# (`objcopy --redefine-syms`, references inside the archive included), so
# the two archives never meet each other, and neither can bind to, or be
# bound by, the OpenBLAS builds numpy and scipy load into the same process.
set -euo pipefail
W=$(realpath -m "${1:?usage: build_openblas.sh WORKDIR}")

# numpy's: the products (MWN)
NUMPY_OPENBLAS_VERSION=0.3.34.106.0
NUMPY_OPENBLAS_REPORTS=0.3.34.106.0
NUMPY_OPENBLAS_COMMIT=446c436e10450c348169808f1a6b3fae0925c9f7
# SciPy's: the LAPACK (MWX). Its wheel stamps no version: it reports 0.3.31.dev.
SCIPY_OPENBLAS_VERSION=0.3.31.22.0
SCIPY_OPENBLAS_REPORTS=0.3.31.dev
SCIPY_OPENBLAS_COMMIT=5ffbf38b41a1fe494d038efcb09cf22f4d527c22

# build COMMIT REPORTS PREFIX OUT REQUIRED...
build() {
    local commit=$1 reports=$2 prefix=$3 out=$4
    shift 4
    cd "$W/src"
    git checkout -q -f "$commit"
    git clean -qfdx
    # the version string the wheel reports (openblas-libs stamps it on newer
    # tags, its build-openblas.sh; SciPy's tag leaves the source's own)
    sed -e "s/^VERSION = .*/VERSION = ${reports}/" -i Makefile.rule
    CFLAGS="-fvisibility=protected -Wno-uninitialized" make -j"$(nproc)" \
        BUFFERSIZE=20 DYNAMIC_ARCH=1 QUIET_MAKE=1 USE_OPENMP=0 NUM_THREADS=64 \
        DYNAMIC_LIST="PRESCOTT NEHALEM SANDYBRIDGE HASWELL" BINARY=64 \
        TARGET=PRESCOTT NOFORTRAN=1 NO_SHARED=1 > "$W/build-$prefix.log" 2>&1
    local archive
    archive=$(ls libopenblas*.a | head -1)
    nm -g --defined-only "$archive" 2>/dev/null |
        awk -v p="$prefix" 'NF == 3 { print $3 " " p $3 }' | sort -u > "$W/syms-$prefix.txt"
    objcopy --redefine-syms="$W/syms-$prefix.txt" "$archive" "$W/$out"
    nm -g --defined-only "$W/$out" 2>/dev/null > "$W/defined-$prefix.txt"
    for s in "$@" openblas_set_num_threads blas_thread_shutdown_; do
        grep -q " T $prefix$s\$" "$W/defined-$prefix.txt" ||
            { echo "missing $prefix$s in $W/$out" >&2; exit 1; }
    done
    echo "$W/$out: OpenBLAS $reports ($commit), $(wc -l < "$W/syms-$prefix.txt") symbols renamed $prefix"
}

mkdir -p "$W"
[ -d "$W/src" ] || git clone -q https://github.com/OpenMathLib/OpenBLAS "$W/src"
build "$SCIPY_OPENBLAS_COMMIT" "$SCIPY_OPENBLAS_REPORTS" mwx_ libmwxp.a \
    zgetrf_ zgetrs_ dgetrf_ dgetrs_ zgeqp3_ zgesvx_ zsytrf_ zsytrs_
build "$NUMPY_OPENBLAS_COMMIT" "$NUMPY_OPENBLAS_REPORTS" mwn_ libmwnp.a \
    zgemm_ zgemv_ zgesv_ cblas_zgemm cblas_zgemv cblas_dgemm cblas_dgemv cblas_dsyrk \
    cblas_ddot cblas_zdotu_sub cblas_zdotc_sub
