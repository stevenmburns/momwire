"""momwire#852 — the node's own coordinate lands on either side of the plane.

`hub_deck()`'s rise is a uniform run from the buried hub to the crossing
node, and the node's z is built by segment accumulation. For some segment
counts that accumulation lands a hair ABOVE `ground_z` (+1.04e-17 at 8, 9,
10, 14 and 16 segments) and at or below it for others (7, 11, 12, 13, 20).
The designed tables require z >= 0 >= z', so the below axis's END POINT was
handed to `six_point` on the wrong side and the solve died three frames down
on a bare `ValueError: need z >= 0 >= zp` — an invariant message that says
nothing about the deck.

NON-MONOTONE IN THE SEGMENT COUNT, which is the diagnosis: 12 works and 8,
9, 10, 14 and 16 fail, so it is not a resolution floor being crossed. It is
a coincidence of floating-point placement. The graded catalog spelling
(momwire#674) walks past it, which is why nothing shipped hit it — but
momwire#674's own advice for a coarse node is to grade HARDER, and the first
thing a user does instead is raise the segment count, which is exactly the
path that broke.

WHERE IT IS, precisely, because that took the longest to establish: the
offending point is a wire END on the below axis, not a quadrature node and
not a panel boundary. `_ends_and_corner` already clamped the ABOVE side with
an unconditional `max(pt[2] - gz, 0.0)`; the below side had nothing.

The repair is written so a deck that solves today keeps its bits — a
coordinate already on the required side is passed through UNCHANGED, and
only a wrong-side one is snapped. Measured across the ten rungs that solved
on main before the fix: 10/10 bit-identical.

LANE COVERAGE. `nec5_quadrature` selects the path quadrature inside
`_tables`, strictly downstream of the coordinate this fixes, so the repair
is lane-independent by construction. That is asserted at the coordinate
itself rather than by solving twice: a full Gauss-Legendre solve of one
repaired rung measures 113 s single-threaded against the two-point lane's
22 s (and ~240 s under xdist worker pinning), which is real cost on a
push-only lane for re-testing the same repair.
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from momwire import RazorSolver  # noqa: E402
from momwire import _crossing_fill as CF  # noqa: E402
from momwire import razor as _razor  # noqa: E402
from test_crossing_serve_524 import SOIL_A, hub_deck  # noqa: E402

# 8/9/10/14/16 put the node above the plane, 7/11/12/13 do not. Both halves
# are gated: a change that moved the coincidence would break the pattern.
RISE_WRONG_SIDE = (8, 9, 10, 14, 16)
RISE_RIGHT_SIDE = (7, 11, 12, 13)


@pytest.fixture
def serve_crossing(monkeypatch):
    monkeypatch.setattr(_razor, "_SERVE_CROSSING", True)
    monkeypatch.setattr(_razor, "_SERVE_BELOW_PLANE", True)


def _rise_deck(n_rise, n_radials=4):
    """`hub_deck` with only the RISE's segment count changed. The rise is
    wire `n_radials` — the radials come first, then the rise, then the
    monopole."""
    d = hub_deck(n_radials=n_radials)
    d["n_per_edge_per_wire"] = [list(x) for x in d["n_per_edge_per_wire"]]
    d["n_per_edge_per_wire"][n_radials] = [n_rise]
    return d


def _rise_end_z(n_rise, n_radials=2):
    """The below axis's end z's, relative to the plane — the coordinates
    `_ends_and_corner` hands to `_tables`. Geometry only, no solve."""
    s = RazorSolver(**_rise_deck(n_rise, n_radials), nec5_quadrature=True, n_qp_path=8)
    geom = s._build_geometry()
    ctx = s._crossing_context(geom, ground_eps=SOIL_A)
    so = np.asarray(geom["seg_offsets"])
    axis = CF.axis_data(ctx, np.arange(so[n_radials], so[n_radials + 1]))
    gz = float(ctx.ground_z)
    return np.array([pt[2] - gz for pt, _sign, _fv in axis["ends"]])


def test_the_rule_passes_the_correct_side_through_unchanged():
    """The bit-preservation promise, on the pure function.

    This is what says momwire#852 cannot move a deck that already solved:
    the fix only ever touches a coordinate that was on the WRONG side, and
    such a coordinate used to raise rather than produce an answer.
    """
    for side, vals in (
        ("above", np.array([0.0, 1e-30, 1e-13, 0.5, 9.0])),
        ("below", np.array([0.0, -1e-30, -1e-13, -0.5, -9.0])),
    ):
        got = CF._on_plane_side(vals, side, "test point")
        assert np.array_equal(got, vals), (side, got, vals)


def test_the_rule_snaps_a_wrong_side_ulp_to_the_plane():
    """Inside the tolerance a wrong-side coordinate IS the interface."""
    eps = 1.0408340855860843e-17  # the value momwire#852 actually measured
    got = CF._on_plane_side(np.array([eps, -1.0]), "below", "end point")
    assert got[0] == 0.0 and got[1] == -1.0, got
    got = CF._on_plane_side(np.array([-eps, 1.0]), "above", "end point")
    assert got[0] == 0.0 and got[1] == 1.0, got


def test_the_rule_refuses_a_genuinely_wrong_side_member_by_name():
    """Past the tolerance there is no honest side, so it refuses instead of
    clamping — an above member that crosses the plane is a geometry error,
    and clamping it would model something else silently."""
    for side, bad in (("below", 1e-3), ("above", -1e-3)):
        with pytest.raises(ValueError, match="wrong side of ground_z") as exc:
            CF._on_plane_side(np.array([bad]), side, "end point")
        msg = str(exc.value)
        assert side in msg and "momwire#852" in msg, msg
        assert f"{CF._PLANE_TOL:g}" in msg, msg


def test_the_rule_shares_one_tolerance_with_the_touching_test():
    """`_PLANE_TOL` is the same number `axis_data` uses to decide a segment
    touches the plane. They were separate literals until momwire#852, and a
    point can only be "the interface" for one of those purposes if the two
    agree."""
    assert CF._PLANE_TOL == 1e-12
    src = pathlib.Path(CF.__file__).read_text()
    assert "tol = _PLANE_TOL" in src, (
        "axis_data no longer keys its touching test on _PLANE_TOL"
    )


def test_the_below_axis_end_is_the_wrong_side_and_the_rule_fixes_it(serve_crossing):
    """The bug and the repair at the exact point they happen, with no solve.

    Both halves, so the gate cannot pass vacuously: on rung 8 the rise's end
    really is above the plane and `_on_plane_side` maps it to exactly 0; on
    rung 12 it is already legal and passes through untouched.
    """
    for n_rise, expect_wrong in ((8, True), (12, False)):
        ends_z = _rise_end_z(n_rise)
        assert ends_z.size, "the below axis carries no ends to test"
        worst = float(ends_z.max())
        assert (worst > 0.0) is expect_wrong, (n_rise, worst)
        fixed = CF._on_plane_side(ends_z, "below", "end point")
        assert np.all(fixed <= 0.0), (n_rise, fixed[fixed > 0.0])
        if not expect_wrong:
            assert np.array_equal(fixed, ends_z), "a legal end was modified"


def test_the_uniform_rise_ladder_is_a_coincidence_not_a_floor(serve_crossing):
    """The diagnosis, pinned across the whole ladder without solving: WHICH
    counts land wrong-side is not monotone in the count.

    Cheap, so it sits in the PR lane — if this pattern ever changes, the
    slow ladder below is testing something else.
    """
    wrong = {n: float(_rise_end_z(n).max()) for n in RISE_WRONG_SIDE}
    right = {n: float(_rise_end_z(n).max()) for n in RISE_RIGHT_SIDE}
    assert all(v > 0.0 for v in wrong.values()), wrong
    assert all(v <= 0.0 for v in right.values()), right
    assert all(v < CF._PLANE_TOL for v in wrong.values()), (
        "a wrong-side end past the tolerance is a geometry error, not the "
        f"momwire#852 coincidence: {wrong}"
    )


@pytest.mark.slow
def test_the_uniform_rise_ladder_serves(serve_crossing):
    """The end-to-end half: every rung solves and the answers are smooth
    across the ones that used to raise.

    Monotone in the node segment length is the real check — a rung that
    solved but returned nonsense would break the ordering, and the repaired
    rungs sit between working neighbours on both sides. Two-radial hub and
    four rungs, both trims paid for by the geometry gates above; six rungs
    on the 4-radial deck cost ~200 s.
    """
    r = {}
    for n in (7, 8, 12, 16):
        z, _ = RazorSolver(
            **_rise_deck(n, n_radials=2), nec5_quadrature=True, n_qp_path=8
        ).compute_impedance()
        assert np.isfinite(z), (n, z)
        r[n] = z
    seq = [r[n].real for n in sorted(r)]
    assert all(a > b for a, b in zip(seq, seq[1:], strict=False)), r
