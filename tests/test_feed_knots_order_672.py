"""`_feed_knots` hands its knots back in arc order (stevenmburns/momwire#672).

The list used to be built interior-knots-first and then have the junction and
grounded ends APPENDED, so a wire whose start is a junction reported that knot
last. `_snap_to_knot` runs `argmin` over the list and uses the index, which is
invisible wherever one knot is plainly nearest — the same knot wins whatever
its position — and decisive at a tie, where the index is the only thing left
to choose on.

That made an index a property of how the list was built rather than of the
geometry, and the two are not the same: the SAME two physical knots appear as
indices (9, 10) when a bent wire is one polyline and (9, 0) when it is two.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire.razor import RazorSolver

WL = 299792458.0 / 20e6
APEX = [0.0, 0.0, 8.0]
LEFT = [-2.0, 0.0, 6.0]
RIGHT = [2.0, 0.0, 6.0]
LEG = float(np.hypot(2.0, 2.0))


def _split(**kw):
    """The inverted-V as two wires meeting at the apex — so the apex is a
    JUNCTION end on each, which is the case that used to sort last."""
    return RazorSolver(
        wires=[np.array([LEFT, APEX]), np.array([APEX, RIGHT])],
        nsegs=10,
        wire_radius=1.0e-3,
        wavelength=WL,
        ground_z=0.0,
        **kw,
    )


def _merged(**kw):
    """The same antenna as ONE polyline, where the apex is an ordinary
    interior knot and has always sorted where it belongs."""
    return RazorSolver(
        wires=[np.array([LEFT, APEX, RIGHT])],
        nsegs=10,
        wire_radius=1.0e-3,
        wavelength=WL,
        ground_z=0.0,
        **kw,
    )


@pytest.mark.parametrize("build,wire", [(_split, 0), (_split, 1), (_merged, 0)])
def test_feed_knots_come_back_in_arc_order(build, wire):
    """The bug itself. A junction end is a knot like any other and belongs at
    its own arclength, not after every interior one."""
    solver = build(feeds=[(wire, 0.5 * LEG, 1 + 0j)])
    geom = solver._build_geometry()
    arcs = [a for a, _basis, _k in solver._feed_knots(geom, wire)]
    assert arcs == sorted(arcs), arcs


def test_the_same_two_knots_sort_the_same_way_in_both_decompositions():
    """The consequence, and why the order is worth a gate.

    A feed named half a segment from the apex is equidistant from the apex and
    from the first interior knot beyond it — a tie, in both decompositions.
    Whichever way a tie is settled, it must not depend on whether the bend was
    authored as one `GW` card or two: these are the same antenna.
    """
    site = 0.5 * LEG / 10  # half a segment along the second leg

    split = _split(feeds=[(1, site, 1 + 0j)])
    merged = _merged(feeds=[(0, LEG + site, 1 + 0j)])

    def tied_pair(solver, wire, target):
        geom = solver._build_geometry()
        arcs = np.array([a for a, _b, _k in solver._feed_knots(geom, wire)])
        order = np.argsort(np.abs(arcs - target), kind="stable")
        near, second = int(order[0]), int(order[1])
        # the pair really is a tie, or this deck stopped exercising the case
        gap = abs(abs(arcs[second] - target) - abs(arcs[near] - target))
        assert gap <= 1e-9 * arcs[-1], gap
        return near, second, float(arcs[near]), float(arcs[second])

    s_near, s_second, s_a, s_b = tied_pair(split, 1, site)
    m_near, m_second, m_a, m_b = tied_pair(merged, 0, LEG + site)

    # What #672 owes: in each decomposition the index order of the tied pair
    # agrees with their arc order, so "the lower index" and "the one nearer
    # the wire's start" are the same knot. Before the sort that held on the
    # merged reading (both interior, indices 9 and 10) and failed on the
    # split one, where the apex was appended and read as index 9 against its
    # neighbour's 0 — the same two knots, in opposite order.
    assert (s_near < s_second) == (s_a < s_b)
    assert (m_near < m_second) == (m_a < m_b)

    # What #672 does NOT owe, and deliberately is not asserted here: that the
    # two decompositions RESOLVE the tie to the same knot. They still do not.
    # The split tie is exact, so a stable argmin takes the apex; the merged
    # one is inexact by 2.2e-15 and rounding takes the neighbour. Settling
    # that is #623's item 3, which this fix unblocks rather than performs —
    # with the list in arc order, "lowest index" finally means "smallest
    # arclength", which is a property of the geometry and can be the same in
    # both spellings. It could not be stated at all before this.
    assert {round(s_a, 9), round(s_b, 9)} == {0.0, round(LEG / 10, 9)}
    assert {round(m_a - LEG, 9), round(m_b - LEG, 9)} == {0.0, round(LEG / 10, 9)}
