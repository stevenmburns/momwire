#!/usr/bin/env bash
#
# Build PyNEC (from the python-necpp submodule) and install it into the
# local project venv at .venv/.
#
# Tool deps (autoconf/automake/libtoolize/m4): system packages, on Ubuntu:
#     sudo apt install autoconf automake libtool m4
# SWIG and numpy come from the local venv. Fortran is NOT required; necpp
# is configured with --without-lapack so the build is C++ only.
#
# After this script: `from PyNEC import nec_context` should work in .venv.
#
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
VENV="$ROOT/.venv"

if [ ! -x "$VENV/bin/python" ]; then
    echo "error: no .venv at $VENV — create one and pip install -e . first." >&2
    exit 1
fi
if [ ! -d "$ROOT/python-necpp/necpp_src" ] || [ -z "$(ls -A "$ROOT/python-necpp/necpp_src" 2>/dev/null)" ]; then
    echo "error: python-necpp submodule not initialised — run:" >&2
    echo "    git submodule update --init --recursive" >&2
    exit 1
fi
if [ ! -x "$VENV/bin/swig" ]; then
    echo "error: swig not in venv — run: .venv/bin/pip install swig" >&2
    exit 1
fi

cd "$ROOT/python-necpp/necpp_src"
# Generate configure (idempotent — skip if config.h already exists).
if [ ! -f config.h ]; then
    make -f Makefile.git
    ./configure --without-lapack
fi

cd "$ROOT/python-necpp/PyNEC"
# Symlink necpp_src into PyNEC/ so setup.py's relative paths work.
if [ ! -e necpp_src ]; then
    ln -s ../necpp_src .
fi
# SWIG wrapper (idempotent — regenerate if .i is newer than the wrapper).
if [ ! -f PyNEC_wrap.cxx ] || [ PyNEC.i -nt PyNEC_wrap.cxx ]; then
    "$VENV/bin/swig" -Wall -c++ -python PyNEC.i
fi

# Build + install into the venv. --no-build-isolation so setup.py sees
# numpy from .venv. -e (editable) so re-running just rebuilds in place.
"$VENV/bin/pip" install --no-build-isolation .

echo "OK — PyNEC built and installed into $VENV"
"$VENV/bin/python" -c "from PyNEC import nec_context; print('PyNEC import OK')"
