#!/bin/bash
# momwire#1822 diag leg: start the minimal server, drive it, report.
set -u
LEG=$1; OUT=$2; SECS=${3:-200}
mkdir -p "$OUT"
R=scratch_1822/repro_1822.py
P=$(mktemp -d /tmp/r1822.XXXX)/s.sock
BURN=""
burn() { for i in 1 2 3; do python3 -c "while 1: pass" & BURN="$BURN $!"; done; }
run() { # name, server opts, drive args...
  name=$1; sopts=$2; shift 2
  python3 -I $R server "$P" "$OUT/$name.server.log" $sopts & spid=$!
  sleep 1
  python3 -I $R drive "$P" "$OUT/$name.server.log" 100000000 "$@" --seconds $SECS 2>&1 | tee "$OUT/$name.drive.txt" | tail -30
  kill $spid; wait $spid 2>/dev/null
  # connections the server accepted but never saw EOF on
  python3 - "$OUT/$name.server.log" <<'PY' | tee -a "$OUT/$name.drive.txt"
import sys
acc, first, eof, closed = {}, {}, {}, set()
for ln in open(sys.argv[1]):
    f = ln.split()
    if len(f) < 3: continue
    k, n = f[1], f[2]
    if k == "accept": acc[n] = ln
    elif k == "first": first[n] = ln
    elif k == "eof": eof[n] = ln
    elif k == "closed": closed.add(n)
stuck = [n for n in acc if n not in eof]
print(f"server: accepted={len(acc)} no-eof={len(stuck)}")
for n in stuck[:20]:
    print("  stuck", n, "first:", first.get(n, "-").strip())
PY
}
case $LEG in
  serial)
    run proc1 "" 1 proc
    run thread1 "" 1 thread ;;
  wakeup)
    run retry "--retry-timeout 2" 3 thread ;;
  base)
    run thread3 "" 3 thread
    run ack3 "--ack" 3 thread --ack ;;
  sent)
    run sentinel3 "--sentinel" 3 thread --sentinel
    run ackproc "--ack" 3 proc --ack ;;
  delay)
    run blocking "--blocking" 3 thread
    run ack3b "--ack" 3 thread --ack ;;
  idle)
    run proc3 "" 3 proc
    run thread3 "" 3 thread ;;
  load)
    burn
    run proc3load "" 3 proc
    run thread3load "" 3 thread ;;
  shape)
    burn
    run acceptdelay "--accept-delay 0.02" 6 thread
    run blocking "--blocking" 3 thread ;;
esac
[ -n "$BURN" ] && kill $BURN
exit 0
