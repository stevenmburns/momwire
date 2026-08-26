"""Convergence profile: licensed NEC-5, momwire bs2, momwire razor-nec5.

The receipts behind "reproduction is not accuracy" (momwire#604), which the
EZNEC bundle's README and `reference/eznec-nec5` both now quote.

The point is NOT which is closest to the licensed engine at a given mesh —
razor-nec5 is its formulation TWIN, so agreeing is faithfulness by
construction and says nothing about accuracy. The point is what each engine
does to its OWN answer as the mesh refines.

Measured 2026-08-26 on a 0.476 lambda dipole, free space, N = 5..161:

The twin's agreement with the licensed engine is inheritance of a
discretization error, not evidence of correctness. See the two ladders
below for what that means in segments.

THE FEED MUST BE AT A KNOT, AND THAT DECIDES THE PARITY
------------------------------------------------------
NEC-5's basis is the TENT, so its unknowns — and its sources — live at KNOTS,
not at segment centres. Measured, not assumed: `RazorSolver` returns the
IDENTICAL impedance for a feed requested at knot 2, at the mid-segment wire
centre, and at knot 3 (64.2253-91.7383j at N=5, all three), because it snaps
to a knot. `BSplineSolver` does not snap — the same three requests give
75.1897-31.6745j, 68.1410-28.8074j and 75.1897-31.6745j.

So an ODD segment count has no knot at a dipole's centre and CANNOT feed it
there: the source lands half a segment off. "Use odd segments for a
centre-fed dipole" is a NEC-2 convention, where sources sit at segment
centres; NEC-5 inverts it.

An earlier version of this probe ran an all-odd ladder and therefore compared
bs2 fed at the wire's centre against razor and the binary fed at a knot half
a segment away — a feed-position difference read as a basis difference. It
was wrong at coarse mesh, where the offset is a large fraction of the
antenna: |bs2 - nec5| read 63 ohm at N=5 where the corrected N=6 row reads
41.8. Beyond N ~ 20 the two ladders agree (7.1 against 7.5 at N ~ 20, 1.06
against 1.09 at N ~ 160).

Everything below is EVEN N with the centre knot fed, all three engines.

  N     licensed NEC-5      bs2                 |razor-n5|   |bs2-n5|
    4   56.118 -108.593j    69.359 -28.486j       0.039        81.19
   20   66.667  -35.880j    68.045 -28.481j       0.038         7.53
  160   67.670  -29.281j    67.867 -28.208j       0.038         1.09

The twin tracks the reference to 0.038 ohm at EVERY density — flat, not
improving, which is what a twin looks like and says nothing about accuracy.

WHICH CONVERGES FASTER
----------------------
`knot_ladder()` scores each basis against ITS OWN N=320 answer, the only way
to ask the question without nominating one as the truth:

  segments   bs2 error   razor error
      4       1.54          80.63
     20       0.39           7.16
    240       0.025          0.17

Both converge. At a matched mesh the B-spline basis is 20-50x nearer its own
limit — razor needs ~240 segments for the accuracy bs2 has at 20, the O(1/N)
walk of razor-blade testing priced in segments.

NEITHER is converged at a coarse mesh. An earlier draft said bs2 was
"essentially converged at five segments" from its small STEP sizes; that
conflated step with error. bs2's steps are ~0.1 ohm and its cumulative error
at four segments is 1.54 ohm. Small steps, many of them.

FEED PARITY, CONTROLLED
-----------------------
`parity_control` reruns both momwire bases at both parities with the feed
pinned to the exact centre. bs2 is parity-INSENSITIVE (0.35 ohm between N=5
and N=6, 0.025 between 161 and 160), so its behaviour is not a feed-placement
artifact. razor-nec5 is strongly parity-SENSITIVE at coarse mesh (22 ohm
between N=5 and N=6, the parities converging only by N ~ 45) — the same
phenomenon the ladders measure, seen from another angle, and why the twin
tracks the binary at odd N: it inherits that sensitivity too.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np  # noqa: F401  (used by the local imports below)

_NUM = re.compile(r"[-+]?\d+\.\d+E[-+]\d+")

FREQ_MHZ = 14.0
WL = 299792458.0 / (FREQ_MHZ * 1e6)
LEN = 10.18946  # the free-space floor study's dipole, ~0.476 lambda
RAD = 1.0e-3
NS = (4, 6, 10, 20, 40, 60, 120, 160)  # EVEN: the centre must be a KNOT


def deck(n):
    p0, p1 = (0.0, -LEN / 2, 0.0), (0.0, LEN / 2, 0.0)
    return (
        "CM convergence study\nCE\n"
        f"GW 1 {n} {p0[0]:.6E} {p0[1]:.6E} {p0[2]:.6E} "
        f"{p1[0]:.6E} {p1[1]:.6E} {p1[2]:.6E} {RAD:.6E}\n"
        "GE 0\n"
        f"EX 0 1 {n // 2} 2 1.000000E+00 0.000000E+00\n"
        f"FR 0 1 0 0 {FREQ_MHZ:.6E} 0.000000E+00\nXQ 0\nEN\n"
    )


def momwire_z(n, basis):
    """Direct constructors, feed pinned to the centre knot by ARCLENGTH.

    NOT the deck route, deliberately.  For even N no segment CENTRE coincides
    with the centre knot — segment K's centre is (K-0.5)*h, the knot is
    (n/2)*h — so an `EX` card lands equidistant between two knots and the
    snap is a tie.  momwire's deck translation and the binary break that tie
    the OTHER WAY from each other, which at N=4 is a quarter wave apart and
    reads as a 94 ohm formulation difference that is nothing of the kind
    (razor 91.423-195.648j through the deck against 56.120-108.555j through
    the constructor, where the binary says 56.118-108.593j).

    `feed_arclength` names the knot with no tie to break, so momwire is
    asked directly and only the binary goes through a deck.  The deck
    tie-break divergence is real and filed separately; it is not what this
    probe is measuring.
    """
    from momwire import BSplineSolver, RazorSolver

    wires = [np.array([[0.0, -LEN / 2, 0.0], [0.0, LEN / 2, 0.0]])]
    common = dict(wires=wires, wire_radius=RAD, wavelength=WL, feed_arclength=LEN / 2)
    if basis == "bspline":
        z, _ = BSplineSolver(
            **common,
            n_per_edge_per_wire=[[n]],
            degree=2,
            feed_model="point",
            feed_wire_index=0,
        ).compute_impedance()
    else:
        z, _ = RazorSolver(
            **common, n_per_edge_per_wire=[[n]], nec5_quadrature=True
        ).compute_impedance()
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


def parity_control():
    """Both momwire bases at both parities, feed pinned to the exact centre.

    The deck route cannot express this — `EX` names a segment — so the
    solvers are built directly here.  It is the check that the all-odd ladder
    above is measuring convergence and not a feed-placement artifact.
    """
    from momwire import BSplineSolver, RazorSolver

    wires = [np.array([[0.0, -LEN / 2, 0.0], [0.0, LEN / 2, 0.0]])]
    common = dict(wires=wires, wire_radius=RAD, wavelength=WL, feed_arclength=LEN / 2)
    print("\n--- parity control: feed at the EXACT centre in both parities ---")
    print(f"{'N':>4}  {'parity':>5}  {'bs2':>22}  {'razor-nec5':>22}")
    for n in (5, 6, 21, 20, 45, 44, 161, 160):
        zb, _ = BSplineSolver(
            **common,
            n_per_edge_per_wire=[[n]],
            degree=2,
            feed_model="point",
            feed_wire_index=0,
        ).compute_impedance()
        zr, _ = RazorSolver(
            **common, n_per_edge_per_wire=[[n]], nec5_quadrature=True
        ).compute_impedance()
        zb, zr = complex(zb), complex(zr)
        print(
            f"{n:>4}  {'odd' if n % 2 else 'even':>5}  "
            f"{zb.real:10.4f}{zb.imag:+10.4f}j  {zr.real:10.4f}{zr.imag:+10.4f}j"
        )


def knot_ladder():
    """Even N, feed on the centre KNOT, each basis against its own limit.

    The binary is absent by necessity, not by choice — see the module
    docstring.  What is scored is each basis's distance from ITS OWN N=320
    answer, which is the only way to ask "which converges faster" without
    nominating one of them as the truth.
    """
    from momwire import BSplineSolver, RazorSolver

    wires = [np.array([[0.0, -LEN / 2, 0.0], [0.0, LEN / 2, 0.0]])]
    common = dict(wires=wires, wire_radius=RAD, wavelength=WL, feed_arclength=LEN / 2)

    def zb(n):
        z, _ = BSplineSolver(
            **common,
            n_per_edge_per_wire=[[n]],
            degree=2,
            feed_model="point",
            feed_wire_index=0,
        ).compute_impedance()
        return complex(z)

    def zr(n):
        z, _ = RazorSolver(
            **common, n_per_edge_per_wire=[[n]], nec5_quadrature=True
        ).compute_impedance()
        return complex(z)

    ref_b, ref_r = zb(320), zr(320)
    print("\n--- knot-fed, even N, each basis against its OWN N=320 answer ---")
    print(
        f"reference:  bs2 {ref_b.real:.4f}{ref_b.imag:+.4f}j   "
        f"razor {ref_r.real:.4f}{ref_r.imag:+.4f}j"
    )
    print(f"{'N':>4}  {'bs2 err':>9}  {'razor err':>10}  {'|bs2-razor|':>11}")
    for n in (4, 6, 8, 10, 12, 16, 20, 28, 40, 60, 80, 120, 160, 240):
        b, r = zb(n), zr(n)
        print(
            f"{n:>4}  {abs(b - ref_b):9.4f}  {abs(r - ref_r):10.4f}  {abs(b - r):11.4f}"
        )


if __name__ == "__main__":
    main()
    knot_ladder()
    parity_control()
