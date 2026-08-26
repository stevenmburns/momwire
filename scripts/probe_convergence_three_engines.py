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

TWO LADDERS, BECAUSE THEY ASK DIFFERENT QUESTIONS
------------------------------------------------
`main()` is the THREE-ENGINE ladder and it must be SEGMENT-fed: `EX` names a
segment, so only an odd count centres the source, and an even count would
feed the binary half a segment off-centre. That is not a preference — the
licensed engine cannot express a knot feed on a plain `GW` at all. Measured:
`EX 4` at node 10 returns the same impedance as `EX 0` at segment 10 to
0.002 ohm (66.6660-35.8780j against 66.6670-35.8796j, N=20). So this ladder
measures FAITHFULNESS — does the twin reproduce the reference — and nothing
else.

`knot_ladder()` is the TWO-BASIS ladder and it is KNOT-fed, even N, each
basis scored against its OWN N=320 answer. No binary, because the binary
cannot be knot-fed, and that is the point: this is the fair basis-vs-basis
comparison, and it measures CONVERGENCE.

  segments   bs2 error   razor-nec5 error
      4       1.54          80.63
      6       0.74          41.36
     20       0.39           7.16
     60       0.18           1.91
    240       0.025          0.17

Both converge. At a matched mesh the B-spline basis is 20-50x nearer its own
limit — razor needs ~240 segments for the accuracy bs2 has at 20, the O(1/N)
walk of razor-blade testing priced in segments.

And NEITHER is converged at a coarse mesh. An earlier draft of this probe
said bs2 was "essentially converged at five segments" on the strength of its
small STEP sizes; that conflated step with error. bs2's steps are ~0.1 ohm
but its cumulative error at four segments is 1.54 ohm. Small steps, many of
them.

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


def parity_control():
    """Both momwire bases at both parities, feed pinned to the exact centre.

    The deck route cannot express this — `EX` names a segment — so the
    solvers are built directly here.  It is the check that the all-odd ladder
    above is measuring convergence and not a feed-placement artifact.
    """
    import numpy as np

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
    import numpy as np

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
