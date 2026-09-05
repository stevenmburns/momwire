"""momwire#929: re-capture the buried goldens under the documented ground flag.

The 23 current decks and the 2 anchors were captured with `GE 1,-1` — ground
flag 1 with the segment-check field at -1. antennaknobs#1025 found the wrapper
that wrote those cards had the two fields transposed, and flag 1 is the setting
documented as not usable when wires go below the surface. Every deck here is
FULLY buried, so every one of them is in that class.

Executable only; nothing is read from the licensed source tree. Conclusions
only in anything public.

Pass --verify to reproduce the banked values under the ORIGINAL card, which is
the provenance check on this parser before any of its new numbers are trusted.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "tests")
from golden_buried_anchor_nec5 import (  # noqa: E402
    ANCHOR_DECKS,
    ANCHOR_FOUR_RADIAL,
    ANCHOR_LONE_RADIAL,
)
from golden_buried_currents_nec5 import DECKS  # noqa: E402

EXE = os.environ["NEC5_EXE"]
ANCHOR_Z = {"lone-radial": ANCHOR_LONE_RADIAL, "four-radial": ANCHOR_FOUR_RADIAL}


def run(deck, timeout=900):
    with tempfile.TemporaryDirectory(prefix="nec5_929_") as td:
        (Path(td) / "m.nec").write_text(deck)
        try:
            subprocess.run(
                [EXE],
                input="m.nec\nm.out\n\n",
                text=True,
                capture_output=True,
                cwd=td,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None
        out = Path(td) / "m.out"
        return out.read_text(errors="replace") if out.is_file() else None


def parse_z(text):
    m = re.search(
        r"- - - ANTENNA INPUT PARAMETERS - - -(.*?)(?:\n\s*\n\s*\n|$)", text, re.S
    )
    for line in m.group(1).splitlines() if m else []:
        t = line.split()
        if len(t) >= 12 and re.fullmatch(r"\d+", t[0]):
            return complex(float(t[7]), float(t[8]))
    return None


def parse_currents(text):
    """(element, z_norm, real_A, imag_A) per row, in the printout's own order.

    Two traps, both found by --verify rather than by reading:

    * the section header is followed by an INLINE `- - - Current (Amps) - - -`
      column header, so a non-greedy match to the next `- - -` returns an
      empty section. Scan lines from the header instead.
    * the banked tuples are (element, z, REAL, IMAG) — columns 7 and 8 of the
      row — even though `golden_buried_currents_nec5`'s own docstring calls
      them `magnitude_A, phase_deg`. The magnitude column is 9. Reading the
      docstring rather than the data would have silently compared different
      quantities.
    """
    lines = text.splitlines()
    try:
        start = next(
            i for i, ln in enumerate(lines) if "- - - Wire Currents - - -" in ln
        )
    except StopIteration:
        return ()
    rows = []
    for line in lines[start:]:
        if "Wire Charge Densities" in line or "POWER BUDGET" in line:
            break
        t = line.split()
        if len(t) >= 10 and re.fullmatch(r"\d+", t[0]) and re.fullmatch(r"\d+", t[1]):
            rows.append((int(t[0]), float(t[4]), float(t[6]), float(t[7])))
    return tuple(rows)


def to_minus_one(deck):
    return re.sub(r"^GE\s+1\s*,\s*-1\s*$", "GE -1,0", deck, count=1, flags=re.M)


ap = argparse.ArgumentParser()
ap.add_argument("--verify", action="store_true")
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

if args.verify:
    print("PROVENANCE: reproducing the banked values under the ORIGINAL card\n")
    bad = 0
    items = list(DECKS.items())[: args.limit or None]
    for name, d in items:
        text = run(d["deck"])
        z, cur = parse_z(text), parse_currents(text)
        dz = abs(z - d["input_z"]) if z is not None else float("inf")
        same = cur == d["currents"]
        flag = "ok " if (dz < 1e-6 and same) else "BAD"
        if flag == "BAD":
            bad += 1
        print(
            f"  {flag} {name:26s} dZ={dz:.2e}  currents_match={same}"
            f"  rows={len(cur)}/{len(d['currents'])}"
        )
        if not same and cur:
            print(f"        first row got {cur[0]}  want {d['currents'][0]}")
    for name, deck in ANCHOR_DECKS.items():
        z = parse_z(run(deck))
        dz = abs(z - ANCHOR_Z[name]) if z is not None else float("inf")
        print(f"  {'ok ' if dz < 1e-3 else 'BAD'} anchor {name:19s} dZ={dz:.2e}")
    print(f"\n{bad} deck(s) failed provenance")
    raise SystemExit(1 if bad else 0)

print("RE-CAPTURE under the documented flag: old vs new, 23 current decks\n")
print(
    f"{'deck':28s} {'flag-1 Z':>22s} {'documented-flag Z':>22s} {'spike old':>10s} {'new':>8s}"
)


def spike(rows):
    """|I| at the driven element over |I| at its neighbour — the quantity
    G-U5-8 calls the 10x off-feed collapse. Rows are (elem, z, real, imag)."""
    if len(rows) < 3:
        return float("nan")
    mags = [abs(complex(r[2], r[3])) for r in rows]
    f = max(range(len(mags)), key=lambda i: mags[i])
    nb = [mags[i] for i in (f - 1, f + 1) if 0 <= i < len(mags)]
    return mags[f] / max(nb) if nb and max(nb) else float("nan")


table = []
for name, d in list(DECKS.items())[: args.limit or None]:
    old_t = run(d["deck"])
    new_t = run(to_minus_one(d["deck"]))
    zo, zn = parse_z(old_t), parse_z(new_t)
    co, cn = parse_currents(old_t), parse_currents(new_t)
    so, sn = spike(co), spike(cn)
    table.append((name, zo, zn, so, sn, cn))
    fo = f"{zo.real:10.4f}{zo.imag:+10.4f}j" if zo else "        --"
    fn = f"{zn.real:10.4f}{zn.imag:+10.4f}j" if zn else "        --"
    print(f"{name:28s} {fo:>22s} {fn:>22s} {so:10.2f} {sn:8.2f}")

neg_old = sum(1 for _n, zo, _zn, _so, _sn, _c in table if zo and zo.real < 0)
neg_new = sum(1 for _n, _zo, zn, _so, _sn, _c in table if zn and zn.real < 0)
print(
    f"\nnegative printed R:  flag-1 {neg_old}/{len(table)}   documented {neg_new}/{len(table)}"
)
sp_old = [t[3] for t in table if t[3] == t[3]]
sp_new = [t[4] for t in table if t[4] == t[4]]
if sp_old:
    print(
        f"feed spike ratio:    flag-1 min {min(sp_old):.2f} max {max(sp_old):.2f}"
        f"   documented min {min(sp_new):.2f} max {max(sp_new):.2f}"
    )
