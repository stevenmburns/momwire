"""The fat executable's printouts, byte for byte against the one-variant builds.

    python scripts/windows_static/fat_compare.py FAT VARIANT SINGLE DIR [DIR ...]

Every `*.nec` under the DIRs is answered by FAT with MOMWIRE_FORCE_VARIANT=
VARIANT and by SINGLE (the one-variant executable of that VARIANT), each
twice: at the size policy's thread count (mostly the serial copy, on these
decks) and through the threaded copy (MW_EXE_COPY=threads), so all four of
the fat build's copies are reached. The printouts must be the same bytes, and
so must the exit status and stdout. Prints one line per mismatch and the
counts; exits 1 on any mismatch, or when nothing was compared.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def run(
    exe: str, deck: Path, out: Path, env: dict[str, str]
) -> tuple[int, bytes, bytes]:
    r = subprocess.run(
        [exe, str(deck), str(out)], env=env, capture_output=True, timeout=900
    )
    body = out.read_bytes() if out.exists() else b""
    out.unlink(missing_ok=True)  # the next run writes the same path
    return r.returncode, r.stdout, body


def main(argv: list[str]) -> int:
    fat, variant, single, *dirs = argv
    decks = sorted({p for d in dirs for p in Path(d).rglob("*.nec")})
    work = Path(tempfile.mkdtemp())
    jobs = []
    for i, deck in enumerate(decks):
        for copy in ("policy", "threads"):
            jobs.append((i, deck, copy))

    def one(job):
        i, deck, copy = job
        env = dict(os.environ)
        env["MOMWIRE_FORCE_VARIANT"] = variant
        env.pop("MOMWIRE_THREADS", None)
        if copy == "threads":
            env["MW_EXE_COPY"] = "threads"
            env["MOMWIRE_THREADS"] = "2"
        else:
            env.pop("MW_EXE_COPY", None)
        # the same printout path for both, in case a printout names it
        out = work / f"{i}-{copy}.out"
        a = run(fat, deck, out, env)
        b = run(single, deck, out, env)
        return deck, copy, a, b

    same = differ = refused = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for deck, copy, a, b in pool.map(one, jobs):
            if a == b:
                same += 1
                if b"not served by this build" in a[2]:
                    refused += 1
            else:
                differ += 1
                what = (
                    "rc" if a[0] != b[0] else "stdout" if a[1] != b[1] else "printout"
                )
                print(f"DIFFER {what} {copy} {deck}")
    print(
        f"{variant}: {same} identical ({refused} of them the same refusal), {differ} different, "
        f"{len(decks)} decks x 2 copies"
    )
    return 1 if differ or not same else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
