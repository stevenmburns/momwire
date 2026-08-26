"""Convergence profile: licensed NEC-5, momwire bs2, momwire razor-nec5.

The receipts behind "reproduction is not accuracy" (momwire#604), which the
EZNEC bundle's README and `reference/eznec-nec5` both now quote.

The point is NOT which is closest to the licensed engine at a given mesh —
razor-nec5 is its formulation TWIN, so agreeing is faithfulness by
construction and says nothing about accuracy. The point is what each engine
does to its OWN answer as the mesh refines.

Measured 2026-08-26 on a 0.476 lambda dipole, free space, N = 5..161:

  * razor-nec5 sits 0.037-0.041 ohm from the licensed engine at EVERY
    density. Flat, not improving — which is what a twin looks like.
  * the licensed engine moves 39.89 ohm from N=5 to N=9 and is STILL moving
    0.29 ohm per step at N=161, the O(1/N) walk of razor-blade testing.
  * bs2 moves 0.11 ohm on that first step and 0.03 at the last: it is
    essentially converged at five segments.
  * and the licensed column walks TOWARD the B-spline one — reactance
    -91.78 -> -29.28 while bs2 holds near -28.2.

So on this antenna bs2 at 5 segments is nearer the converged answer than
NEC-5 at 161, and the twin's agreement at coarse mesh is inheritance of a
discretization error rather than evidence of correctness.

Run:
  NEC5_EXE=~/antennas/NEC5-downloads/nec5-linux/nec5cl \
      /home/smburns/antennas/antennaknobs/.venv/bin/python <this>
"""

from __future__ import annotations

import pathlib
import re

import numpy as np

_NUM = re.compile(r"[-+]?\d+\.\d+E[-+]\d+")

FREQ_MHZ = 14.0
WL = 299792458.0 / (FREQ_MHZ * 1e6)
LEN = 10.18946  # the free-space floor study's dipole, ~0.476 lambda
RAD = 1.0e-3
NS = (5, 9, 15, 21, 31, 45, 61, 91, 121, 161)


def deck(n):
    p0, p1 = (0.0, -LEN / 2, 0.0), (0.0, LEN / 2, 0.0)
    return (
        "CM convergence study\nCE\n"
        f"GW 1 {n} {p0[0]:.6E} {p0[1]:.6E} {p0[2]:.6E} "
        f"{p1[0]:.6E} {p1[1]:.6E} {p1[2]:.6E} {RAD:.6E}\n"
        "GE 0\n"
        f"EX 0 1 {n // 2 + 1} 2 1.000000E+00 0.000000E+00\n"
        f"FR 0 1 0 0 {FREQ_MHZ:.6E} 0.000000E+00\nXQ 0\nEN\n"
    )


def momwire_z(n, basis):
    from momwire.deck import build_solver, parse

    built = build_solver(parse(deck(n), dialect="nec2"), basis=basis)
    z, _ = built.solver.compute_impedance()
    return complex(np.atleast_1d(z)[0])


def nec5_z(n):
    """Z from the binary's PRINTED voltage and current — V/I, nothing else."""
    import os
    import subprocess
    import tempfile

    exe = os.path.expanduser(os.environ["NEC5_EXE"])
    with tempfile.TemporaryDirectory(prefix="nec5conv_") as td:
        (pathlib.Path(td) / "m.nec").write_text(deck(n))
        subprocess.run(
            [exe],
            input="m.nec\nm.out\n\n",
            text=True,
            capture_output=True,
            cwd=td,
            timeout=600,
        )
        text = (pathlib.Path(td) / "m.out").read_text(errors="replace")
    lines = text.splitlines()
    i = next(k for k, ln in enumerate(lines) if "ANTENNA INPUT PARAMETERS" in ln)
    for ln in lines[i:]:
        f = ln.split()
        if len(f) >= 9 and all(_NUM.fullmatch(x) for x in f[3:9]):
            v = complex(float(f[3]), float(f[4]))
            c = complex(float(f[5]), float(f[6]))
            return v / c
    raise SystemExit("no source row found")


def main():
    import pathlib as _p  # noqa: F401

    rows = []
    for n in NS:
        z5 = nec5_z(n)
        zb = momwire_z(n, "bspline")
        zr = momwire_z(n, "razor-nec5")
        rows.append((n, z5, zb, zr))
        print(
            f"N={n:<4d} nec5 {z5.real:8.3f}{z5.imag:+8.3f}j   "
            f"bs2 {zb.real:8.3f}{zb.imag:+8.3f}j   "
            f"razor {zr.real:8.3f}{zr.imag:+8.3f}j   "
            f"|razor-nec5| {abs(zr - z5):7.4f}   |bs2-nec5| {abs(zb - z5):7.4f}",
            flush=True,
        )

    print("\n--- step-to-step movement (how far each engine's own answer moved) ---")
    for (n0, a0, b0, c0), (n1, a1, b1, c1) in zip(rows, rows[1:]):
        print(
            f"{n0:>4d}->{n1:<4d}  nec5 {abs(a1 - a0):7.4f}   "
            f"bs2 {abs(b1 - b0):7.4f}   razor {abs(c1 - c0):7.4f}"
        )


if __name__ == "__main__":
    main()
