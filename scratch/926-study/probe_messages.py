"""momwire#926: what the two messages SAY, on decks where both rules apply.

The deck in the issue as I first rebuilt it does NOT raise
`CoarseCrossingNode`: four radials sloping up from a node with a vertical mast
are all ABOVE wires, so the junction is grounded but not CROSSING, and the
advisory only walks crossing junctions. Measured -- `_crossing_junctions()`
returns () on it. The h/a refusal fires there; the advisory does not, so that
deck shows only half the collision.

The geometry where BOTH genuinely apply needs a crossing junction (N below
members, exactly ONE above member -- the fan widening's scope) whose ABOVE
member slopes gently away from the node. That is a real antenna: a sloping
wire off a grounded node over a buried radial screen. So the decks here are
`fan_rise_deck`'s screen with the vertical monopole replaced by a sloper.

  * SLOPING above member, ungraded  -- the advisory must fire and name
    L_min = 40 mm rather than the unconditional ~6 mm;
  * the same deck graded to 6 mm    -- the refusal must fire and name the
    advisory;
  * PLUMB above member (the canonical fan) -- L_min is 2 mm, under the ~6 mm
    asked for anyway, so the old text must come back unchanged;
  * the plumb deck's flat buried radials -- level arms, likewise unchanged.
"""

import warnings

import numpy as np

from momwire import _crossing_fill
from momwire.bspline import BSplineSolver

C0 = 299792458.0
WL, A_WIRE, SOIL = C0 / 7e6, 1e-3, (13.0, 0.005)
DEPTH, REACH, RISE = 0.15, 10.0, 0.5


def _graded(finest_mm, span, ratio=1.35):
    """Arclengths from 0 to `span`, geometric from `finest_mm` outward."""
    s, pts, r = finest_mm / 1000.0, [0.0], 1.0
    while pts[-1] < span:
        pts.append(pts[-1] + s * r)
        r *= ratio
    return np.array([p for p in pts if p < span] + [span])


def deck(above="slope", finest_mm=None, n_radials=4):
    """The connected screen: N buried radials rising to a node at z = 0, plus
    ONE above member that is either a plumb mast or a 5 % sloper.

    `finest_mm=None` leaves the above member ungraded (one edge), which is the
    state the advisory is FOR.
    """
    dirs = [
        (np.cos(2 * np.pi * i / n_radials), np.sin(2 * np.pi * i / n_radials))
        for i in range(n_radials)
    ]
    wires = [
        np.array([(5.0 * dx, 5.0 * dy, -DEPTH), (0.0, 0.0, -DEPTH), (0.0, 0.0, 0.0)])
        for dx, dy in dirs
    ]
    npe = [[10, 2] for _ in dirs]
    mono_i = len(wires)
    if above == "plumb":
        tip = np.array([0.0, 0.0, 10.0])
    else:
        tip = np.array([REACH, 0.0, RISE])
    if finest_mm is None:
        wires.append(np.array([(0.0, 0.0, 0.0), tip]))
        npe.append([15])
    else:
        span = float(np.linalg.norm(tip))
        t = _graded(finest_mm, span) / span
        wires.append(np.outer(t, tip))
        npe.append([1] * (len(t) - 1))
    return dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        junctions=[[(i, "end") for i in range(n_radials)] + [(mono_i, "start")]],
        feeds=[(mono_i, 0.25, 1 + 0j)],
        wavelength=WL,
        wire_radius=A_WIRE,
        ground_z=0.0,
        ground_eps=SOIL,
        ground_model="sommerfeld",
    )


def show(name, build):
    print(f"\n=== {name} ===")
    s = BSplineSolver(**build)
    crossing = s._crossing_junctions()
    print(f"  crossing junctions: {crossing}")
    arms = s._crossing_node_members(crossing, s._wire_media())
    for a in arms:
        lm = _crossing_fill.node_panel_floor(a.h_floor, a.slope)
        print(
            f"    wire {a.wire} {a.side:5s} slope={a.slope:7.4f} "
            f"h_adj={a.h_adjacent * 1000:8.2f} mm  "
            f"L_min={'None' if lm is None else f'{lm * 1000:.1f} mm'}"
        )
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _crossing_fill.warn_coarse_node(arms)
    msgs = [
        str(w.message) for w in rec if w.category is _crossing_fill.CoarseCrossingNode
    ]
    print(f"  ADVISORY: {msgs[0] if msgs else '(silent)'}")
    try:
        s.compute_impedance()
        print("  SOLVE: served")
    except (ValueError, NotImplementedError) as e:
        print(f"  SOLVE {type(e).__name__}: {e}")


show("SLOPING above member, ungraded", deck("slope"))
show("SLOPING above member, graded to 6 mm", deck("slope", finest_mm=6.0))
show("SLOPING above member, graded to 40 mm", deck("slope", finest_mm=40.0))
show("PLUMB above member, ungraded (canonical fan)", deck("plumb"))
