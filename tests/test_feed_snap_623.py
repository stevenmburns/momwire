"""What the feed snap is, per family, and which half of it the tie can reach
(stevenmburns/momwire#623).

Every family but `BSplineSolver` resolves `feed_arclength` by
`argmin(|grid - target|)` — segment centres in `sinusoidal.py`, knots in
`razor.py`'s `_snap_to_knot`, segment centroids in `pulse.py`'s
`_feed_basis_indices`. One spelling, but it means two different things:

* **Position-capable.** `BSplineSolver` (any degree) does not snap at all; it
  excites the exact arclength. `SinusoidalGalerkinSolver` under
  `feed_model="point"` snaps and then carries the remainder in `feed_xi`
  (#648, the default since #654), so the gap sits where it was asked for no
  matter which side of a tie won. The pick is only *whose shape functions do
  I evaluate on* — bookkeeping, and a knot is a correct request.
* **Grid-locked.** `feed_model="segment"` spreads E^app over one whole
  segment and ignores `feed_xi`; a tent basis puts the port at a knot; a
  pulse row IS a segment and has no sub-grid position at all. Here the snap
  is a quantisation: the port moves by h/2 and the pick is the answer.

Each family's tie locus is the midpoint between two adjacent grid points,
which is the OTHER family's grid point: a centre-snapper ties on a knot, a
knot-snapper ties on a centre. So the two loci below name every tie there is
on a uniform mesh.

The gates move two requests 1 nm apart across a locus — physically the same
feed, 1e-10 of the wire — and ask what the family does. Measured, N=20 on a
10 m wire (h = 0.5 m), relative spread in driving-point Z:

    family                 @ knot 3.0    @ centre 2.75    @ midpoint 5.0
    bspline d=1             9.18e-10        1.08e-09          5.8e-16
    bspline d=2             9.20e-10        1.08e-09          1.8e-16
    sin-galerkin point      9.19e-10        1.08e-09          3.5e-16
    sin-galerkin segment    2.06e-01        0.00e+00          5.8e-14
    sinusoidal (pm)         2.06e-01        0.00e+00          4.0e-15
    razor                   0.00e+00        2.38e-01          0.00e+00
    pulse                   8.26e-02        0.00e+00          4.0e-16

Three readings, and each is a gate below:

1. The position-capable families sit at ~1e-9 at BOTH loci, and that number is
   not the snap — `BSplineSolver`, which has no snap, reports the same value
   to 0.3 %. It is what moving a gap 1 nm is worth on this deck. So the tie
   costs the point gap nothing measurable, which is the property that makes
   #623 a non-issue on the default path and the thing nothing asserted before.
2. The grid-locked families return a BIT-IDENTICAL answer for a 1 nm move
   that stays inside one cell, and 8-24 % for one that crosses a locus. Both
   halves of that are the quantisation: they do not track position at all, and
   then they jump a whole cell.
3. On a SYMMETRIC deck every family is at roundoff, grid-locked ones included,
   because the two candidate cells are mirror images. That is why #623 sat
   unnoticed through #421's ladder, and it is why the loci here are off-centre.

The nudge is 1 nm, not an ULP: which side an on-locus request actually lands
on is decided by a margin of tens to tens of thousands of ULPs that grows with
N (see the issue), so a test that tried to ride the real tie would be testing
the rounding. These requests straddle it by ~1e4 ULPs, which makes each side
deterministic while the phenomenon it brackets is not.
"""

import numpy as np
import pytest

from momwire.bspline import BSplineSolver
from momwire.pulse import PulseSolver
from momwire.razor import RazorSolver
from momwire.sinusoidal import SinusoidalSolver
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

# ----------------------------------------------------------------------
# The deck: a straight 10 m wire on a 20 m wavelength, N = 20 uniform
# segments, so h = 0.5 m exactly and both grids are exactly representable.
# Even N puts a KNOT at the wire midpoint, which is what makes 5.0 the
# symmetric locus and 3.0 an off-centre one.
# ----------------------------------------------------------------------
WIRE = np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
WIRE_LEN = 10.0
N = 20
WL = 20.0

KNOT = 3.0  # a knot, off-centre: ties the centre-snappers
CENTRE = 2.75  # a segment centre, off-centre: ties the knot-snapper
MIDPOINT = 5.0  # a knot AND the wire's mirror plane: ties nothing visibly

NUDGE = 1e-9  # 1 nm out of 10 m

# Generous absolute bar for "this family did not jump a cell". Five decades
# below the smallest measured jump (8.3e-2) and three above the largest
# measured non-jump (1.1e-9), so it separates the two classes without
# pinning either number.
FLAT = 1e-6
# A jump is a whole cell of the mesh, so it is a fraction of Z, not a
# tolerance. One decade under the smallest measured (8.3e-2).
JUMP = 1e-2
# How much of the ~1e-9 a snapping family is allowed to owe the SNAP rather
# than to the 1 nm of geometry, measured against the family that has no snap.
# The three position-capable families agree to 0.3 %; 4x leaves room for the
# bases to differ without admitting anything that jumped.
SNAP_FREE = 4.0


def _z(cls, arc, **kw):
    s = cls(
        wires=[WIRE],
        n_per_edge_per_wire=[[N]],
        wavelength=WL,
        feed_arclength=arc,
        **kw,
    )
    return np.atleast_1d(s.compute_impedance()[0])[0]


def _spread(cls, locus, **kw):
    """Relative spread in Z between the two sides of `locus`."""
    lo = _z(cls, locus - NUDGE, **kw)
    hi = _z(cls, locus + NUDGE, **kw)
    return abs(lo - hi) / abs(lo)


# (id, class, kwargs, the locus this family ties on)
POSITION_CAPABLE = [
    ("bspline-d1", BSplineSolver, {"degree": 1}, None),
    ("bspline-d2", BSplineSolver, {"degree": 2}, None),
    ("galerkin-point", SinusoidalGalerkinSolver, {"feed_model": "point"}, None),
]
GRID_LOCKED = [
    ("galerkin-segment", SinusoidalGalerkinSolver, {"feed_model": "segment"}, KNOT),
    ("sinusoidal-pm", SinusoidalSolver, {}, KNOT),
    ("razor", RazorSolver, {}, CENTRE),
    ("pulse", PulseSolver, {}, KNOT),
]
ALL = POSITION_CAPABLE + GRID_LOCKED


def _ids(rows):
    return [r[0] for r in rows]


def test_the_nudge_clears_the_real_tie_by_orders_of_magnitude():
    """The bracket is geometry, not rounding.

    #623's actual tie is decided by a margin of tens to tens of thousands of
    ULPs of the cumsum that builds the grid. If `NUDGE` were anywhere near
    that, these tests would be measuring which way the last bit fell.
    """
    assert NUDGE > 1e4 * np.spacing(WIRE_LEN)


@pytest.mark.parametrize(
    "cls,kw", [(c, k) for _i, c, k, _l in POSITION_CAPABLE], ids=_ids(POSITION_CAPABLE)
)
@pytest.mark.parametrize("locus", [KNOT, CENTRE], ids=["at-knot", "at-centre"])
def test_a_position_capable_feed_does_not_notice_the_tie(cls, kw, locus):
    """Crossing either locus costs a position-capable family nothing.

    This is the property that makes the tie-break in #623 irrelevant on the
    default path, and it is the one a later change to `_basis_value` or to
    #606's folded `AC` could quietly break: if this went from 1e-9 to 1e-3,
    no other gate in the suite would notice.
    """
    assert _spread(cls, locus, **kw) < FLAT


@pytest.mark.parametrize("locus", [KNOT, CENTRE], ids=["at-knot", "at-centre"])
def test_the_point_gap_owes_the_snap_nothing(locus):
    """`SinusoidalGalerkinSolver`'s point gap tracks position as well as the
    family that never snaps at all.

    The bar is DERIVED, not recorded: `BSplineSolver(degree=2)` excites the
    exact arclength with no `argmin` anywhere, so whatever it reports for a
    1 nm move is what the geometry costs. #648's `feed_xi` is doing its job
    exactly when the snapping family matches it.
    """
    no_snap = _spread(BSplineSolver, locus, degree=2)
    snapped = _spread(SinusoidalGalerkinSolver, locus, feed_model="point")
    assert snapped < SNAP_FREE * no_snap, (snapped, no_snap)


@pytest.mark.parametrize(
    "cls,kw,locus", [(c, k, lo) for _i, c, k, lo in GRID_LOCKED], ids=_ids(GRID_LOCKED)
)
def test_a_grid_locked_feed_jumps_a_whole_cell_at_its_own_locus(cls, kw, locus):
    """The half of #623 that is real: 1 nm of request, a cell of answer.

    Not a tolerance failure — the port genuinely moves h/2, so this is the
    size of the ambiguity a caller is exposed to when they name a site on the
    wrong grid for the family. Pinned as a LOWER bound because it must stay
    visible: making it small by snapping harder would only hide it.
    """
    assert _spread(cls, locus, **kw) > JUMP


@pytest.mark.parametrize(
    "cls,kw,locus", [(c, k, lo) for _i, c, k, lo in GRID_LOCKED], ids=_ids(GRID_LOCKED)
)
def test_a_grid_locked_feed_is_deaf_inside_a_cell(cls, kw, locus):
    """The other half of the quantisation, and the cleaner signature of it.

    A 1 nm move that does not cross the family's own locus changes nothing —
    measured bit-identical, because the request reaches the fill ONLY through
    the pick. That is what separates these families from the position-capable
    ones, which report ~1e-9 for the same move because they track the gap.
    """
    other = CENTRE if locus == KNOT else KNOT
    assert _spread(cls, other, **kw) < FLAT


@pytest.mark.parametrize("cls,kw", [(c, k) for _i, c, k, _l in ALL], ids=_ids(ALL))
def test_a_symmetric_deck_hides_every_bit_of_this(cls, kw):
    """Why #623 went unnoticed, as a gate.

    At the wire's mirror plane the two candidate cells are reflections, so
    even the grid-locked families come out at roundoff. Every deck the snap
    had been exercised on was symmetric about its feed — #421's refinement
    ladder included, which is where the issue was found and why it read as
    latent. It also makes the off-centre loci above load-bearing rather than
    arbitrary: move these gates to the midpoint and they all pass vacuously.
    """
    assert _spread(cls, MIDPOINT, **kw) < FLAT
