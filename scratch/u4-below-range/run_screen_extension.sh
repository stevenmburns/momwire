#!/usr/bin/env bash
# Restart of the registered screen extension (MEASUREMENTS.md, "Incident"):
# a transient systemd SERVICE, detached from any task shell, never paused, and
# stopped at 4 h from its own start so unfinished rows are recorded as not run.
#
#   scratch/u4-below-range/run_screen_extension.sh launch   # start the service
#   scratch/u4-below-range/run_screen_extension.sh wait     # block to finish or budget
set -euo pipefail
ROOT=/home/smburns/stevenmburns/momwire-wt-u4
DIR=$ROOT/scratch/u4-below-range
LOG=$DIR/screen_extension.log
UNIT=u4-screen-extension
PY=/home/smburns/stevenmburns/antennaknobs/.venv/bin/python
BUDGET_S=$((4 * 3600))

case "${1:-}" in
launch)
    if systemctl --user is-active --quiet "$UNIT"; then
        echo "$UNIT is already running" >&2
        exit 1
    fi
    date -u +"start %FT%TZ (restart; budget 4 h from here)" >"$LOG"
    date +%s >"$DIR/screen_extension.start_epoch"
    systemd-run --user --unit="$UNIT" -p MemoryMax=24G --nice=15 \
        --working-directory="$ROOT" \
        -p StandardOutput="append:$LOG" -p StandardError="append:$LOG" \
        /bin/bash -c "$PY scratch/u4-below-range/screen_remainder.py --extension --workers 6 --out scratch/u4-below-range/screen_extension.json; rc=\$?; date -u +\"end %FT%TZ exit \$rc\" >> $LOG"
    systemctl --user is-active "$UNIT"
    ;;
wait)
    start=$(cat "$DIR/screen_extension.start_epoch")
    deadline=$((start + BUDGET_S))
    while systemctl --user is-active --quiet "$UNIT"; do
        if [ "$(date +%s)" -ge "$deadline" ]; then
            date -u +"budget reached %FT%TZ: stopping; unfinished rows are not run" >>"$LOG"
            systemctl --user stop "$UNIT"
            break
        fi
        sleep 60
    done
    tail -n 30 "$LOG"
    ;;
*)
    echo "usage: $0 launch|wait" >&2
    exit 2
    ;;
esac
