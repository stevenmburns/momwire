#!/bin/bash
# The Linux release artifact of the no-Python executables (momwire#1512,
# momwire#1517): run INSIDE the manylinux_2_28 image, the one the wheels are
# built in, so the floor is glibc 2.28 and the compiler is the wheels'.
#
#     docker run --rm -v "$PWD:/io" -w /io quay.io/pypa/manylinux_2_28_x86_64 \
#         standalone/build_release_linux.sh OPENBLAS_DIR OUT_DIR
#
# -> OUT_DIR/momwire-nec5-linux-x86_64.tar.gz, which unpacks to one
# directory holding six engines, each a command and its two builds:
#
#   momwire-nec5-razor2p-linux        razor-2p: picks the build this CPU runs
#   momwire-nec5-razor2p-linux-avx2   the C++ core compiled -mavx2 -mfma
#   momwire-nec5-razor2p-linux-sse2   the x86-64 baseline build
#   momwire-nec5-bs2-linux            bs2 (`momwire-nec5-bspline`), the same
#   momwire-nec5-bs2-linux-avx2       picking and the same two builds
#   momwire-nec5-bs2-linux-sse2
#   momwire-nec42-sinusoidal-linux       PREVIEW (momwire#1607): the NEC-4.2
#   momwire-nec42-sinusoidal-linux-avx2  slot's `momwire-nec42-sinusoidal`,
#   momwire-nec42-sinusoidal-linux-sse2  the same picking and the same builds
#   momwire-nec42-sinusoidal-galerkin-linux[-avx2|-sse2]  PREVIEW (momwire#1718):
#   momwire-nec5-sinusoidal-galerkin-linux[-avx2|-sse2]   the Sinusoidal-Galerkin
#                                        basis for the NEC-4.2 slot and NEC-5
#   momwire-nec42-linux[-avx2|-sse2]     PREVIEW (momwire#1607 Phase C): the
#                                        NEC-4.2 slot's default `momwire-nec42`
#                                        (bs2)
#   README.txt
#
# Each build is `momwire-razor2p-standalone` / `momwire-bs2-standalone` /
# `momwire-nec42-sinusoidal-standalone` / the two
# `momwire-nec{42,5}-sinusoidal-galerkin-standalone` / `momwire-nec42-standalone`
# (MW_EXE_NO_PYTHON), renamed: `nec5`
# in the name is what makes SimNEC write it NEC-5 decks (momwire#1239), and
# `nec42` NEC-4.2 decks (momwire#1443). OpenBLAS, libstdc++, libgcc and (where the
# toolchain has an archive of it) libgomp are linked in; glibc stays shared,
# at the 2.28 floor, as the wheels have it.
set -euo pipefail
OB=$(realpath -m "${1:?usage: build_release_linux.sh OPENBLAS_DIR OUT_DIR}")
OUT=$(realpath -m "${2:?usage: build_release_linux.sh OPENBLAS_DIR OUT_DIR}")
ROOT=$(cd "$(dirname "$0")/.." && pwd)
NAME=momwire-nec5-razor2p-linux
PKG=momwire-nec5-linux-x86_64

# cmake and ninja from the image's CPython when the image has none
if ! command -v cmake > /dev/null || ! command -v ninja > /dev/null; then
    PY=/opt/python/cp312-cp312/bin/python
    "$PY" -m pip install -q cmake ninja
    export PATH="$(dirname "$PY"):$PATH"
fi

[ -f "$OB/libmwxp.a" ] || "$ROOT/standalone/build_openblas.sh" "$OB"

GOMP=$(gcc -print-file-name=libgomp.a)
STATIC_GOMP=OFF
[ -f "$GOMP" ] && STATIC_GOMP=ON

STAGE=$(mktemp -d)/$PKG
mkdir -p "$STAGE" "$OUT"
# engine: command name, CMake basis, the target it builds
for engine in "$NAME razor2p momwire-razor2p-standalone" \
              "momwire-nec5-bs2-linux bspline_nec5 momwire-bs2-standalone" \
              "momwire-nec42-sinusoidal-linux sin42 momwire-nec42-sinusoidal-standalone" \
              "momwire-nec42-sinusoidal-galerkin-linux sg42 momwire-nec42-sinusoidal-galerkin-standalone" \
              "momwire-nec5-sinusoidal-galerkin-linux sg5 momwire-nec5-sinusoidal-galerkin-standalone" \
              "momwire-nec42-linux bspline_nec42 momwire-nec42-standalone"; do
    set -- $engine
    cmd=$1 basis=$2 target=$3
    for v in avx2 sse2; do
        B=$(mktemp -d)
        cmake -S "$ROOT/standalone" -B "$B" -G Ninja -DMW_EXE_NO_PYTHON=ON \
            -DMW_EXE_BASIS=$basis -DMW_EXE_VARIANT=$v -DMW_OPENBLAS_LIB="$OB/libmwxp.a" \
            -DMW_EXE_STATIC_GOMP=$STATIC_GOMP
        cmake --build "$B"
        cp "$B/$target" "$STAGE/$cmd-$v"
        strip "$STAGE/$cmd-$v"
    done
    cp "$ROOT/standalone/release/$cmd" "$STAGE/$cmd"
    chmod 755 "$STAGE/$cmd" "$STAGE/$cmd-avx2" "$STAGE/$cmd-sse2"

    # What each build needs from the system, and the newest glibc symbol it
    # asks for; MW_GLIBC_FLOOR (the workflow's 2.28) fails the build above it.
    for v in avx2 sse2; do
        echo "== $cmd-$v (static libgomp: $STATIC_GOMP)"
        ldd "$STAGE/$cmd-$v"
        newest=$(objdump -T "$STAGE/$cmd-$v" | grep -o 'GLIBC_[0-9.]*' | sed 's/GLIBC_//' | sort -uV | tail -1)
        echo "newest glibc symbol: $newest"
        if [ -n "${MW_GLIBC_FLOOR:-}" ] &&
            [ "$(printf '%s\n%s\n' "$newest" "$MW_GLIBC_FLOOR" | sort -V | tail -1)" != "$MW_GLIBC_FLOOR" ]; then
            echo "$cmd-$v asks for glibc $newest, above the $MW_GLIBC_FLOOR floor" >&2
            exit 1
        fi
    done
done
cp "$ROOT/standalone/release/README.txt" "$STAGE/README.txt"

tar -C "$(dirname "$STAGE")" --owner=0 --group=0 -czf "$OUT/$PKG.tar.gz" "$PKG"
echo "$OUT/$PKG.tar.gz"
