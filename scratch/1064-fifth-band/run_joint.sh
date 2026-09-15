#!/usr/bin/env bash
# momwire#1064, the joint batch's re-runs in the registered order, alone on the box:
# make test (branch src), G5 (TAG=_joint), G1a-sse2 on the branch.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
VENV=/home/smburns/stevenmburns/antennaknobs/.venv/bin
cd "$ROOT" || exit 1
echo "start $(date -u +%FT%TZ) $(git rev-parse --short HEAD)" >"$HERE/run_joint.out"
PATH="$VENV:$PATH" PYTHONPATH="$ROOT/src" "$VENV/python" -c 'import momwire; print("make test momwire", momwire.__file__)' >>"$HERE/run_joint.out"
PATH="$VENV:$PATH" PYTHONPATH="$ROOT/src" make test >"$HERE/g7_make_test_joint.log" 2>&1
echo "make test rc=$? $(date -u +%FT%TZ)" >>"$HERE/run_joint.out"
TAG=_joint "$HERE/run_t5.sh"
echo "G5 rc=$? $(date -u +%FT%TZ)" >>"$HERE/run_joint.out"
MOMWIRE_FORCE_VARIANT=sse2 PYTHONPATH="$ROOT/src" "$VENV/python" "$HERE/g12_grid.py" --tree branch \
    --out "$HERE/g1a_sse2_branch_joint.json" >"$HERE/g1a_sse2_branch_joint.log" 2>&1
echo "G1a-sse2 rc=$? $(date -u +%FT%TZ)" >>"$HERE/run_joint.out"
echo "done $(date -u +%FT%TZ)" >>"$HERE/run_joint.out"
