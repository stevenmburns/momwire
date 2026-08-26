"""Convergence profile: licensed NEC-5, momwire bs2, momwire razor-nec5.

The receipts behind "reproduction is not accuracy" (momwire#604), which the
EZNEC bundle's README and `reference/eznec-nec5` both now quote.

The point is NOT which is closest to the licensed engine at a given mesh —
razor-nec5 is its formulation TWIN, so agreeing is faithfulness by
construction and says nothing about accuracy. The point is what each engine
does to its OWN answer as the mesh refines.

Measured 2026-08-26 on a 0.476 lambda dipole, free space, N = 4..160 (the
ladders are EVEN-N; see the parity section below for why).

The twin's agreement with the licensed engine is inheritance of a
discretization error, not evidence of correctness. See the ladders below
for what that means in segments.

WHICH LADDER PRODUCED WHICH PUBLISHED NUMBER
--------------------------------------------
There are two self-convergence ladders here and they are NOT interchangeable,
because they reach the solvers by different routes:

  `seam_ladder()`  — through `parse_nec5` + `_serve.serve`, the route the
                     EZNEC drop-in takes, scored against each basis's own
                     N=160 rung.  THIS is what the bundle README and
                     `reference/eznec-nec5` print.
  `knot_ladder()`  — the solvers built directly, scored against N=320.

They disagree by about 2x at the coarse end (bs2 2.81 vs 1.54 ohm at N=4)
and that is not noise: a different reference rung and a different feed
spelling are two different questions, and an earlier draft of the docs
labelled the seam numbers with the knot ladder's reference. Whichever is
quoted, the caption has to name the route AND the rung.

THE FEED MUST BE AT A KNOT, AND THAT DECIDES THE PARITY
------------------------------------------------------
NEC-5's basis is the TENT, so its unknowns — and its sources — live at KNOTS.
Measured, not assumed: `RazorSolver` returns the IDENTICAL impedance for a
feed requested at knot 2, at the mid-segment wire centre, and at knot 3
(64.2253-91.7383j at N=5, all three), because it snaps to a knot.
`BSplineSolver` does not snap — the same three give 75.1897-31.6745j,
68.1410-28.8074j and 75.1897-31.6745j.

So an ODD segment count has no knot at a dipole's centre and cannot feed it
there. "Odd segments for a centre-fed dipole" is a NEC-2 convention, where
sources sit at segment CENTRES; NEC-5 inverts it. This ladder is even-N with
`EX` naming the centre knot, and both the binary and the seam map `EX seg K`
to knot K — verified index by index at N=4:

  EX seg   binary                seam (razor-nec5)
     1     91.4170 -195.7123j    91.4128 -195.6994j
     2     56.1179 -108.5930j    56.1162 -108.5864j   <- the centre
     3     91.4170 -195.7123j    91.4128 -195.6994j

THE LADDER
----------
  N     licensed NEC-5      bs2 (seam)          |razor-n5|  |bs2-n5|
    4   56.118 -108.593j    67.645 -31.146j       0.0069      78.30
   20   66.667  -35.880j    67.739 -29.155j       0.0031       6.81
  160   67.670  -29.281j    67.796 -28.340j       0.0029       0.95

The twin reproduces the licensed engine to 0.003 ohm from N=20 up and 0.007
at the coarsest rung — flat, not improving, which is what a twin looks like
and says nothing about accuracy.

WHICH CONVERGES FASTER
----------------------
Both ladders score each basis against ITS OWN converged answer, the only way
to ask the question without nominating one as the truth.

`seam_ladder()`, through the drop-in's own route, against each basis's N=160
rung — the numbers the README and the site page print:

  segments   bs2 error   razor error   ratio
      4       2.8102        80.1436    28.5x
     20       0.8167         6.6736     8.2x
     60       0.2466         1.4279     5.8x

`knot_ladder()`, solvers built directly, against N=320:

  segments   bs2 error   razor error   ratio
      4       1.5433        80.6298    52x
     20       0.3860         7.1554    18x
    240       0.0249         0.1693     6.8x

Both converge. At a matched mesh the B-spline basis is 5.8-52x nearer its own
limit, the O(1/N) walk of razor-blade testing priced in segments. The ratio
SHRINKS with N in both ladders, so the coarse end is where the choice of basis
is worth the most.

An earlier draft said "razor needs ~240 segments for the accuracy bs2 has at
20". It does not: on the knot ladder razor is already at 0.1693 by N=240
against bs2's 0.3860 at N=20, and interpolating its own rows (0.4815 at 160,
0.1693 at 240) puts the crossing near N=200 — the right order of magnitude,
the wrong rung, and only checkable because the row is printed.

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

import os
import pathlib
import re

import numpy as np

from momwire.deck._nec5 import parse_nec5
from momwire.eznec import _serve

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
        "GE 0,-1\n"
        f"EX 0 1 {n // 2} 2 1.000000E+00 0.000000E+00\n"
        f"FR 0 1 0 0 {FREQ_MHZ:.6E} 0.000000E+00\nXQ 0\nEN\n"
    )


def momwire_z(n, basis):
    """Through the NEC-5 SEAM — the same route the EZNEC drop-in takes.

    This is the whole point of the probe: not "what can these solvers do if
    hand-built", but "what does the shipping portal answer for this deck".
    So momwire is asked exactly as EZNEC asks it, through `parse_nec5` and
    `_serve.serve`, on the SAME deck text the binary is handed.

    An earlier version routed momwire through `parse(text, dialect="nec2")`
    instead. That is a different front end with a different convention — it
    maps `EX seg K` to knot K-1 where the NEC-5 dialect maps it to knot K —
    and the mismatch read as a 94 ohm formulation difference at N=4 that was
    nothing of the kind. The lesson is that "the deck front end" is two front
    ends, and the one the portal uses is the one worth measuring.
    """
    return _serve.serve(parse_nec5(deck(n)), basis=basis).sources[0].impedance


def nec5_exe():
    """The licensed binary from ``NEC5_EXE``, or ``None`` — it is OPTIONAL.

    Read through here rather than as ``os.environ[...]`` at the point of use:
    every function in this file except `main` is binary-free, and a bare
    KeyError on the first row used to kill the whole script before either
    self-convergence ladder — the ones the shipped docs quote — printed
    anything at all.
    """
    path = os.environ.get("NEC5_EXE")
    return os.path.expanduser(path) if path else None


def nec5_z(n):
    """Z from the binary's PRINTED voltage and current — V/I, nothing else."""
    import subprocess
    import tempfile

    exe = nec5_exe()
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
    """The licensed comparison — the ONE lane that needs ``NEC5_EXE``."""
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
    solvers are built directly here.  It is the check that the even-N ladders
    above are measuring convergence and not a feed-placement artifact: they
    are all-even BECAUSE the tent basis feeds at knots, and a ladder whose
    parity is fixed cannot tell the two apart on its own.
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


def seam_ladder():
    """The SEAM route, each basis against its own N=160 rung.

    Same route and same decks as `main`'s momwire columns — `parse_nec5` into
    `_serve.serve` — with the licensed column left out, which is what makes it
    runnable without the binary.  Its reference is the finest rung of the
    published ladder rather than a further-out N: the table a reader sees ends
    at 160, and scoring against a rung nobody printed is a number they cannot
    check.  `knot_ladder` asks the same question the other way, further out and
    through the constructors; the two differ by about 2x at the coarse end and
    the docs must say which they quote.
    """
    zb = {n: momwire_z(n, "bspline") for n in NS}
    zr = {n: momwire_z(n, "razor-nec5") for n in NS}
    ref = NS[-1]
    print(f"\n--- seam route, even N, each basis against its OWN N={ref} rung ---")
    print(
        f"reference:  bs2 {zb[ref].real:.4f}{zb[ref].imag:+.4f}j   "
        f"razor {zr[ref].real:.4f}{zr[ref].imag:+.4f}j"
    )
    print(
        f"{'N':>4}  {'bs2':>22}  {'razor-nec5':>22}  {'bs2 err':>9}  {'razor err':>10}"
    )
    for n in NS:
        b, r = zb[n], zr[n]
        print(
            f"{n:>4}  {b.real:10.4f}{b.imag:+10.4f}j  {r.real:10.4f}{r.imag:+10.4f}j  "
            f"{abs(b - zb[ref]):9.4f}  {abs(r - zr[ref]):10.4f}"
        )


if __name__ == "__main__":
    # The binary-free lanes FIRST and unconditionally.  Two of the three
    # ladders need no licensed engine, they are the ones the bundle README and
    # the site page quote, and a reader without the binary has to be able to
    # reproduce what is published — which the old order made impossible by
    # dying in `main` before any of them printed a row.
    seam_ladder()
    knot_ladder()
    parity_control()
    if nec5_exe():
        main()
    else:
        print(
            "\nNEC5_EXE unset: skipping the licensed-engine ladder. "
            "The self-convergence ladders above need no binary; set NEC5_EXE "
            "to the nec5cl path to add the licensed column."
        )
