#!/usr/bin/env python3
"""Transparent timing shim between SimNEC and a portal process (#375).

Sits on the engine command line, forwards stdin/stdout byte-for-byte
line-wise, and records one JSONL row per deck unit: when its NX terminator
arrived, when the NX echo (the line SimNEC blocks on) went back, the FR
card, and a hash of the geometry/loading cards — enough to separate sweep
decks (one geometry hash, marching FR) from knob decks (changing hash)
after the session, without interpreting anything live.

Usage (always via a wrapper, never typed into SimNEC directly):

    portal_bench_shim.py --log-dir DIR -- /path/momwire-nec2c --cache ...

``-version`` probes exec the real portal directly so the probe stays
prompt.  Every forwarded stdout line is flushed immediately — the shim must
never hold a line SimNEC is blocked on.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

# Cards that decide the moment matrix: geometry, transforms, ground, loading.
_GEOM_PREFIXES = ("GW", "GM", "GS", "GE", "GN", "GD", "LD", "IS", "EK")


def main() -> int:
    argv = sys.argv[1:]
    sep = argv.index("--")
    opts, real_cmd = argv[:sep], argv[sep + 1 :]
    log_dir = Path(opts[opts.index("--log-dir") + 1])

    # The version probe must answer fast and exit: hand straight over.
    for a in real_cmd[1:]:
        if a.strip("-").lower() == "version":
            os.execv(real_cmd[0], real_cmd)

    log_dir.mkdir(parents=True, exist_ok=True)
    log = (log_dir / f"decks.{os.getpid()}.jsonl").open("a")

    child = subprocess.Popen(
        real_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        bufsize=1,
    )
    pending: queue.Queue[dict] = queue.Queue()

    def pump_in() -> None:
        seq = 0
        geom: list[str] = []
        fr = ""
        try:
            for line in sys.stdin:
                child.stdin.write(line)
                card = line.split()[0].upper() if line.split() else ""
                if card in _GEOM_PREFIXES:
                    geom.append(line.strip())
                elif card == "FR":
                    fr = line.strip()
                if card == "NX":
                    child.stdin.flush()
                    pending.put(
                        {
                            "seq": seq,
                            "t_in": time.time(),
                            "fr": fr,
                            "geom": hashlib.sha1("\n".join(geom).encode()).hexdigest()[
                                :12
                            ],
                            "cards": len(geom),
                        }
                    )
                    seq += 1
                    geom, fr = [], ""
        except (BrokenPipeError, ValueError):
            pass
        try:
            child.stdin.close()
        except Exception:
            pass

    threading.Thread(target=pump_in, daemon=True).start()

    for line in child.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        words = line.split()
        if (
            len(words) >= 5
            and words[0] == "DATA"
            and words[1] == "CARD"
            and words[4] == "NX"
        ):
            try:
                rec = pending.get_nowait()
            except queue.Empty:
                rec = {"seq": None, "t_in": None, "fr": "", "geom": "", "cards": 0}
            rec["t_out"] = time.time()
            rec["dt_ms"] = (
                None
                if rec["t_in"] is None
                else round((rec["t_out"] - rec["t_in"]) * 1000, 2)
            )
            log.write(json.dumps(rec) + "\n")
            log.flush()

    log.close()
    return child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
