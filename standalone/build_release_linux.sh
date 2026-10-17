#!/bin/bash
# The Linux release artifact of the no-Python razor-2p executable
# (momwire#1512): run INSIDE the manylinux_2_28 image, the one the wheels are
# built in, so the floor is glibc 2.28 and the compiler is the wheels'.
#
#     docker run --rm -v "$PWD:/io" -w /io quay.io/pypa/manylinux_2_28_x86_64 \
#         standalone/build_release_linux.sh OPENBLAS_DIR OUT_DIR
#
# -> OUT_DIR/momwire-nec5-razor2p-linux-x86_64.tar.gz, which unpacks to one
# directory holding
#
#   momwire-nec5-razor2p-linux        the command: picks the build this CPU runs
#   momwire-nec5-razor2p-linux-avx2   the C++ core compiled -mavx2 -mfma
#   momwire-nec5-razor2p-linux-sse2   the x86-64 baseline build
#   README.txt
#
# Each build is `momwire-razor2p-standalone` (MW_EXE_NO_PYTHON), renamed:
# `nec5` in the name is what makes SimNEC write it NEC-5 decks (momwire#1239).
# OpenBLAS, libstdc++, libgcc and (where the toolchain has an archive of it)
# libgomp are linked in; glibc stays shared, at the 2.28 floor, as the wheels
# have it.
set -euo pipefail
OB=$(realpath -m "${1:?usage: build_release_linux.sh OPENBLAS_DIR OUT_DIR}")
OUT=$(realpath -m "${2:?usage: build_release_linux.sh OPENBLAS_DIR OUT_DIR}")
ROOT=$(cd "$(dirname "$0")/.." && pwd)
NAME=momwire-nec5-razor2p-linux
PKG=$NAME-x86_64

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
for v in avx2 sse2; do
    B=$(mktemp -d)
    cmake -S "$ROOT/standalone" -B "$B" -G Ninja -DMW_EXE_NO_PYTHON=ON \
        -DMW_EXE_VARIANT=$v -DMW_OPENBLAS_LIB="$OB/libmwxp.a" \
        -DMW_EXE_STATIC_GOMP=$STATIC_GOMP
    cmake --build "$B"
    cp "$B/momwire-razor2p-standalone" "$STAGE/$NAME-$v"
    strip "$STAGE/$NAME-$v"
done
cp "$ROOT/standalone/release/$NAME" "$STAGE/$NAME"
cp "$ROOT/standalone/release/README.txt" "$STAGE/README.txt"
chmod 755 "$STAGE/$NAME" "$STAGE/$NAME-avx2" "$STAGE/$NAME-sse2"

# What each build needs from the system, and the newest glibc symbol it asks
# for; MW_GLIBC_FLOOR (the workflow's 2.28) fails the build above it.
for v in avx2 sse2; do
    echo "== $NAME-$v (static libgomp: $STATIC_GOMP)"
    ldd "$STAGE/$NAME-$v"
    newest=$(objdump -T "$STAGE/$NAME-$v" | grep -o 'GLIBC_[0-9.]*' | sed 's/GLIBC_//' | sort -uV | tail -1)
    echo "newest glibc symbol: $newest"
    if [ -n "${MW_GLIBC_FLOOR:-}" ] &&
        [ "$(printf '%s\n%s\n' "$newest" "$MW_GLIBC_FLOOR" | sort -V | tail -1)" != "$MW_GLIBC_FLOOR" ]; then
        echo "$NAME-$v asks for glibc $newest, above the $MW_GLIBC_FLOOR floor" >&2
        exit 1
    fi
done

tar -C "$(dirname "$STAGE")" --owner=0 --group=0 -czf "$OUT/$PKG.tar.gz" "$PKG"
echo "$OUT/$PKG.tar.gz"
