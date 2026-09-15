#!/usr/bin/env bash
# momwire#1064 G7, local lanes on the branch (PLAN.md G7), alone on the box.
#
#   run_g7.sh all    make test, then the slow and crossgate lanes on the touched files
#   run_g7.sh test   make test only
#
# PYTHONPATH must name the branch's src: the venv's editable momwire is another
# checkout, and `make test` without it imports that one (the first G7 run did).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
VENV=${VENV:-/home/smburns/stevenmburns/antennaknobs/.venv/bin}
cd "$ROOT" || exit 1
export PATH="$VENV:$PATH"
export PYTHONPATH="$ROOT/src"
step=${1:-all}
echo "start $step $(date -u +%FT%TZ) $(git rev-parse --short HEAD) momwire=$(python -c 'import momwire; print(momwire.__file__)')" >>"$HERE/run_g7.out"
make test >"$HERE/g7_make_test.log" 2>&1
echo "make test rc=$? $(date -u +%FT%TZ)" >>"$HERE/run_g7.out"
[ "$step" = test ] && exit 0
FILES="tests/test_grazing_band_floor_1064.py tests/test_grazing_band_lo_935.py tests/test_grazing_band_838.py tests/test_below_fills_568.py"
python -m pytest $FILES -m slow -q -n 4 -p no:cacheprovider >"$HERE/g7_slow.log" 2>&1
echo "slow (touched files) rc=$? $(date -u +%FT%TZ)" >>"$HERE/run_g7.out"
python -m pytest tests -m crossgate -q -n 4 -p no:cacheprovider >"$HERE/g7_crossgate.log" 2>&1
echo "crossgate rc=$? $(date -u +%FT%TZ)" >>"$HERE/run_g7.out"
