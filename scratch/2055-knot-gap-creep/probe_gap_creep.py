"""momwire#2055: where bspline's impedance creep under refinement comes from.

The reported deck is an L in free space at 28 MHz, 1 mm wire, a knot source
on the horizontal arm and a series 20 pF load on the knot at the bend. Its
reactance moves ~1.2-1.6 ohm per tripling of the mesh and the steps do not
shrink. The issue's hypothesis was a log-divergent gap term in the knot-LOAD
model. This probe measures four things, each on the solver alone:

1. The deck's 2-port Y (source knot, bend knot) per rung, and the input
   impedance with one entry at a time left free to move. The source's own
   Y11 moves Z_in more than the loaded port's Y22 does, and Y12 moves too.
   Nothing here is particular to the loaded port.
2. The same deck, same ports, same load, on thinner wire. The steps shrink
   once h/a is large. Segment length over radius decides it, not the load.
3. A short straight wire: no load, no bend, |Z|^2 ~ 1.2e6. Its susceptance
   moves by 4 ln(3) w eps0 a per tripling as h approaches a, the zero-width
   gap law the `exact_kernel` docstring already states. |Z|^2 turns that
   6.8e-6 S into ~8 ohm.
4. A closed loop, which has no wire ends: G settles (1e-7 relative at the
   last rung) and B keeps the gap law under a point feed. On an open wire G
   creeps too, under a fixed-WIDTH source as well, so that part belongs to
   the open ends and not to any port.

The reference engine drifts the same way, which is why this is recorded as
a property of the zero-width gap and not fixed here. X steps per tripling,
`antennaknobs ladder`, r = 1, 3, 9, 27, 81, 243 (2026-11-20):

    deck                       engine    steps (ohm)
    the reported deck          nec5      9.15  3.85  2.23  1.76  1.90
                               bspline   1.66  1.25  1.20  1.64  1.46
    short wire, no load, knot  nec5      27.3  13.7  8.70  7.50  8.00
    source 0.75 m along 2 m    bspline   28.4  17.4  10.6  8.07  7.82

As admittance the reported deck's bspline steps are 7-10 uS, SMALLER than
the 15-25 uS of the issue's two "converging" controls at the same rungs;
its |Z|^2 is 1.7e5 against their 8e3.

Run: python scratch/2055-knot-gap-creep/probe_gap_creep.py
Output as committed: results/probe_gap_creep.txt
"""

import warnings

import numpy as np

from momwire.bspline import BSplineSolver

warnings.simplefilter("ignore")

# antennaknobs' wavelength for 28 MHz, so section 1 reproduces its ladder.
LAM = 299.8 / 28.0
OMEGA = 2.0 * np.pi * 28.0e6
EPS0 = 8.8541878128e-12
Y_LOAD = 1j * OMEGA * 20e-12
RUNGS = (1, 3, 9, 27, 81)

# The L as antennaknobs hands it over: the horizontal arm cut at the source
# knot (4 of 12 segments), the bend, the vertical arm. Both ports are series
# node gaps; the load is port algebra on the second.
A, S, B, C = (0, -3, 5), (0, -2, 5), (0, 0, 5), (0, 0, 3)
L_JUNCTIONS = [[(0, "end"), (1, "start")], [(1, "end"), (2, "start")]]


def gap_law(a):
    """Susceptance a zero-width gap gains per tripling of the mesh once the
    segments are short against the radius: 4 ln(3) w eps0 a."""
    return 4.0 * np.log(3.0) * OMEGA * EPS0 * a


def bendload_y(r, a):
    solver = BSplineSolver(
        wires=[[A, S], [S, B], [B, C]],
        n_per_edge_per_wire=[[4 * r], [8 * r], [8 * r]],
        feeds=[],
        junctions=L_JUNCTIONS,
        node_gaps=[(0, "end", 0j), (1, "end", 0j)],
        wavelength=LAM,
        wire_radius=a,
        ground_z=None,
        extended_kernel=True,
        exact_kernel=False,
    )
    return np.asarray(solver.compute_y_matrix())


def z_in(y11, y12, y22):
    return 1.0 / (y11 - y12 * y12 / (y22 + Y_LOAD))


def fmt(z, digits=4):
    return f"{z.real:.{digits}f}{z.imag:+.{digits}f}j"


def section_two_port():
    print("== 1. bendload, a = 1 mm: Z_in with one Y entry free at a time")
    print("   (the others held at their r = 1 values)")
    ys = {r: bendload_y(r, 1e-3) for r in RUNGS}
    y0 = ys[RUNGS[0]]
    print(f"{'r':>4} {'all':>20} {'Y11 only':>20} {'Y12 only':>20} {'Y22 only':>20}")
    for r in RUNGS:
        y = ys[r]
        cols = (
            z_in(y[0, 0], y[0, 1], y[1, 1]),
            z_in(y[0, 0], y0[0, 1], y0[1, 1]),
            z_in(y0[0, 0], y[0, 1], y0[1, 1]),
            z_in(y0[0, 0], y0[0, 1], y[1, 1]),
        )
        print(f"{r:>4} " + " ".join(f"{fmt(c):>20}" for c in cols))
    print("   susceptance step per tripling, in units of the gap law")
    law = gap_law(1e-3)
    for lo, hi in zip(RUNGS, RUNGS[1:], strict=False):
        d = ys[hi] - ys[lo]
        print(
            f"   {lo:>2}->{hi:<2}  dB11 {d[0, 0].imag / law:5.2f}  "
            f"dB22 {d[1, 1].imag / law:5.2f}  dG11 {d[0, 0].real:.2e} S  "
            f"dG22 {d[1, 1].real:.2e} S"
        )


def section_radius():
    print("\n== 2. bendload, same ports and load, thinner wire")
    for a in (1e-3, 1e-4, 1e-5):
        prev = None
        for r in RUNGS:
            y = bendload_y(r, a)
            z = z_in(y[0, 0], y[0, 1], y[1, 1])
            h_over_a = (3.0 / (12 * r)) / a
            step = "" if prev is None else f"  step {fmt(z - prev)}"
            print(f"a={a:g} r={r:>2} h/a={h_over_a:8.1f}  Z_in {fmt(z)}{step}")
            prev = z


def section_short_wire():
    print("\n== 3. short straight wire, a = 1 mm: no load, no bend")
    law = gap_law(1e-3)
    common = dict(
        wavelength=LAM,
        wire_radius=1e-3,
        ground_z=None,
        extended_kernel=True,
        exact_kernel=False,
    )

    def node_gap(r):
        return BSplineSolver(
            wires=[[(0, -1, 5), (0, -0.25, 5)], [(0, -0.25, 5), (0, 1, 5)]],
            n_per_edge_per_wire=[[3 * r], [5 * r]],
            feeds=[],
            junctions=[[(0, "end"), (1, "start")]],
            node_gaps=[(0, "end", 1.0)],
            **common,
        )

    def point_feed(r):
        return BSplineSolver(
            wires=[[(0, -1, 5), (0, 1, 5)]],
            n_per_edge_per_wire=[[8 * r]],
            feeds=[(0, 0.75, 1.0)],
            **common,
        )

    # One place, two spellings of the zero-width gap. The node gap sits on a
    # C0 joint and is near the law from the first rung; the point feed sits
    # inside a C1 spline and comes down to it.
    for label, make in (("node gap", node_gap), ("point feed", point_feed)):
        prev = None
        for r in RUNGS:
            y = complex(np.asarray(make(r).compute_y_matrix())[0, 0])
            z = 1.0 / y
            step = (
                ""
                if prev is None
                else f"  step {fmt(z - 1.0 / prev)}"
                f"  dB/law {(y - prev).imag / law:5.2f}"
            )
            print(f"{label:10s} r={r:>2}  Z {fmt(z)}{step}")
            prev = y


def section_loop():
    print("\n== 4. a 24-gon loop (no wire ends) against an open wire, a = 1 mm")
    n = 24
    radius = 1.1 * LAM / (2.0 * np.pi)
    ring = [
        (
            radius * np.cos(2 * np.pi * i / n),
            radius * np.sin(2 * np.pi * i / n),
            5.0,
        )
        for i in range(n)
    ]
    ring.append(ring[0])
    edge = float(np.hypot(ring[1][0] - ring[0][0], ring[1][1] - ring[0][1]))
    common = dict(
        wavelength=LAM,
        wire_radius=1e-3,
        ground_z=None,
        extended_kernel=True,
        exact_kernel=False,
    )

    def loop(r, **kw):
        return BSplineSolver(
            wires=[ring],
            n_per_edge_per_wire=[[2 * r] * n],
            feeds=[(0, 4.5 * edge, 1.0)],
            junctions=[[(0, "end"), (0, "start")]],
            **common,
            **kw,
        )

    def wire(r, **kw):
        return BSplineSolver(
            wires=[[(0, -2.5, 5), (0, 2.5, 5)]],
            n_per_edge_per_wire=[[20 * r]],
            feeds=[(0, 2.0, 1.0)],
            **common,
            **kw,
        )

    # `feed_smoothing_factor = r` holds the source's physical width at one
    # r = 1 cell on every rung, so what still moves is not the source's.
    cases = (
        ("loop, point feed", loop, {}),
        ("loop, fixed-width source", loop, None),
        ("open wire, point feed", wire, {}),
        ("open wire, fixed-width source", wire, None),
    )
    for label, make, kw in cases:
        prev = None
        for r in RUNGS:
            extra = {"feed_smoothing_factor": float(r)} if kw is None else kw
            y = complex(np.asarray(make(r, **extra).compute_y_matrix())[0, 0])
            step = (
                ""
                if prev is None
                else f"  dG/|Y| {(y - prev).real / abs(y):+.2e}"
                f"  dB/|Y| {(y - prev).imag / abs(y):+.2e}"
            )
            print(f"{label:30s} r={r:>2}  Y {y.real:.6e}{y.imag:+.6e}j{step}")
            prev = y


if __name__ == "__main__":
    print(f"gap law at a = 1 mm: {gap_law(1e-3):.3e} S per tripling")
    section_two_port()
    section_radius()
    section_short_wire()
    section_loop()
