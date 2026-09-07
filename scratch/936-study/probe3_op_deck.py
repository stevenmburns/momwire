"""momwire#936: the OP deck, and the NEC-5 reference re-derived here.

A plumb quarter-wave on a 45 deg slope over four buried radials -- in the
ground's frame a mast leaning 45 deg off the normal, radials 15 cm down,
7.1 MHz, soil 13/0.005. The issue quotes NEC-5 Z = 58.085 - 3.661j; the deck
file itself is not on this box, so the geometry is rebuilt from that
description and the reference RE-DERIVED rather than transcribed. If the two
agree, the deck is the OP's; if they do not, that is worth knowing before any
gate is written against the number.
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

FREQ, LAM = 7.1e6, 299792458.0 / 7.1e6
QW = LAM / 4.0
LEAN = np.radians(45.0)
DEPTH, RAD_A = 0.15, 1e-3
N_RAD = 4
TIP = QW * np.array([np.sin(LEAN), 0.0, np.cos(LEAN)])


def wires():
    """(p0, p1, nseg) list. Hub at -DEPTH, N radials out, a rise to the node
    at z=0, and the leaning mast above it."""
    hub = np.array([0.0, 0.0, -DEPTH])
    node = np.array([0.0, 0.0, 0.0])
    out = []
    for i in range(N_RAD):
        th = 2 * np.pi * i / N_RAD
        tip = np.array([QW * np.cos(th), QW * np.sin(th), -DEPTH])
        out.append((hub, tip, 20))
    out.append((hub, node, 2))
    out.append((node, TIP, 21))
    return out


def nec5_deck():
    lines = ["CM momwire#936 OP deck: plumb mast on a 45 deg slope", "CE"]
    for i, (p0, p1, n) in enumerate(wires(), start=1):
        lines.append(
            f"GW {i} {n} {p0[0]:.6E} {p0[1]:.6E} {p0[2]:.6E} "
            f"{p1[0]:.6E} {p1[1]:.6E} {p1[2]:.6E} {RAD_A:.6E}"
        )
    lines += [
        "GE -1 0",
        "GN 0 0 0 0 1.300000E+01 5.000000E-03 1.000000E+00 0.000000E+00 NOFILE",
        f"EX 0 {len(wires())} 1 0 1.000000E+00 0.000000E+00",
        f"FR 0 1 0 0 {FREQ / 1e6:.6E} 0.000000E+00",
        "XQ 0",
        "EN",
    ]
    return "\n".join(lines) + "\n"


def run_nec5(deck):
    exe = os.environ["NEC5_EXE"]
    with tempfile.TemporaryDirectory(prefix="nec5_936_") as td:
        (Path(td) / "m.nec").write_text(deck)
        subprocess.run(
            [exe],
            input="m.nec\nm.out\n\n",
            text=True,
            capture_output=True,
            cwd=td,
            timeout=2400,
        )
        out = Path(td) / "m.out"
        text = out.read_text(errors="replace") if out.is_file() else ""
    m = re.search(
        r"- - - ANTENNA INPUT PARAMETERS - - -(.*?)(?:\n\s*\n\s*\n|$)", text, re.S
    )
    for line in m.group(1).splitlines() if m else []:
        t = line.split()
        if len(t) >= 12 and re.fullmatch(r"\d+", t[0]):
            return complex(float(t[7]), float(t[8]))
    return None


if __name__ == "__main__":
    d = nec5_deck()
    print(d)
    z = run_nec5(d)
    print(f"NEC-5 re-derived : {z}")
    print("issue quotes     : (58.085-3.661j)")
    if z is not None:
        print(f"|difference|     : {abs(z - (58.085 - 3.661j)):.4f} ohm")
