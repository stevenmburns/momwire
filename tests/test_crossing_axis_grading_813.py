"""`axis_data`'s density knobs, and the two plateaus behind them — momwire#813.

`_crossing_fill.axis_data` took its three densities from module constants:
`_NEAR_Q` (plain Gauss, every segment not touching the interface),
`_NEAR_GROWTH` and `_NEAR_GX` (the log-graded a-scale panels on the segments
that DO touch it). Those are a Galerkin axis's settings, and a PATH-tested
row that ends AT the node needs a finer axis than a Galerkin one does — so
they are per-axis arguments now, defaulting to exactly the constants.

**The two error plateaus are separate, and that is the point of this module.**
On razor's node row at `crossing_deck(1)`:

  * sweeping `q` alone leaves the residual at 5.3e-5 for every order from 4
    to 32 — it never moves;
  * sweeping `panel_order` alone takes it to 2.2e-6 and stops there at every
    `growth` from 4.0 down to 1.25;
  * `panel_order` = 8 AND `q` = 8 together reach 7.7e-11.

So a sweep of either knob alone reads as "converged" at the other's plateau.
That is how the residual came to be recorded in
`test_razor_crossing_axis_813.BAR_ROW_HALF` as a property of the source
Gauss, which it is not; the prose beside that bar is corrected in this change
and this module is what holds the correction.

**Which knob dominates is a property of the BLOCK, not of the trunk**, and
that is why the fix is per-axis arguments rather than a new constant. On the
INTERIOR and REVERSED blocks — whose bases exclude the junction, so
`_graded_u`'s interface-touching panels barely enter — the dominance is the
other way round entirely: the panels move nothing at all at any order or
growth, the source order alone plateaus at 1.07e-07, and the two together
reach 3.2e-16. Both structures are gated below, because a reader who
generalises either one gets the other wrong.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

from momwire import RazorSolver
from momwire import _crossing_fill as CF

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from test_crossing_serve_524 import crossing_deck  # noqa: E402


@pytest.fixture(scope="module")
def fs():
    deck = crossing_deck(1)
    d = {
        k: v
        for k, v in deck.items()
        if k not in ("ground_z", "ground_eps", "ground_model")
    }
    rs = RazorSolver(**d, nec5_quadrature=False, n_qp_path=8)
    geom = rs._build_geometry()
    so = np.asarray(geom["seg_offsets"])
    return dict(
        rs=rs,
        geom=geom,
        ctx=rs._crossing_context(geom, ground_eps=(1.0, 0.0)),
        seg_below=np.arange(so[0], so[1]),
        seg_above=np.arange(so[1], so[2]),
    )


def _axes(f, **kw):
    return (
        CF.axis_data(f["ctx"], f["seg_below"], **kw),
        CF.axis_data(f["ctx"], f["seg_above"], **kw),
    )


@pytest.mark.parametrize("coarse", [False, True])
def test_the_defaults_are_the_constants(fs, coarse):
    """Passing nothing must build the axis the module constants build, key
    for key and bit for bit — the property `BSplineSolver`'s crossing fill
    rides on."""
    plain = CF.axis_data(fs["ctx"], fs["seg_below"], coarse=coarse)
    gx = CF._GX4 if coarse else CF._NEAR_GX
    spelt = CF.axis_data(
        fs["ctx"],
        fs["seg_below"],
        coarse=coarse,
        growth=CF._FAR_GROWTH if coarse else CF._NEAR_GROWTH,
        panel_order=len(gx),
        q=CF._FAR_Q if coarse else CF._NEAR_Q,
    )
    for key in ("nodes", "t", "w", "F", "Fd", "segof"):
        assert np.array_equal(np.asarray(plain[key]), np.asarray(spelt[key])), key
    assert len(plain["ends"]) == len(spelt["ends"])


def test_a_finer_axis_is_actually_finer(fs):
    coarse_ax, _ = _axes(fs, growth=4.0, panel_order=4, q=4)
    fine_ax, _ = _axes(fs, growth=2.0, panel_order=8, q=12)
    assert fine_ax["nodes"].shape[0] > coarse_ax["nodes"].shape[0]
    # the graded panels are on the interface-touching segments only, so the
    # extra nodes are not spread evenly
    assert fine_ax["w"].sum() == pytest.approx(coarse_ax["w"].sum(), rel=1e-12)


def _razor_reference(fs):
    """Razor's own chopped node row. Independent of the axis knobs, so it is
    built once -- rebuilding it per setting was this module's whole cost."""
    if "ref" in fs:
        return fs["ref"], fs["cols"], fs["A3"]
    rs, geom = fs["rs"], fs["geom"]
    jn = geom["n_basis_total"] - 1
    bo = np.asarray(geom["basis_offsets"])
    cols = np.arange(bo[0], bo[1])
    k, omega = rs.k, rs.omega
    Z = rs._assemble_Z_from_prepared(geom, rs._assemble_Z_prepare(geom), k, omega)
    gA, gB = dict(geom), dict(geom)
    gA["wing_sigma"] = geom["wing_sigma"].copy()
    gA["wing_sigma"][jn, 1] = 0.0
    gB["wing_sigma"] = geom["wing_sigma"].copy()
    gB["wing_sigma"][jn, 0] = 0.0
    Z_A = rs._assemble_Z_from_prepared(gA, rs._assemble_Z_prepare(gA), k, omega)
    Z_B = rs._assemble_Z_from_prepared(gB, rs._assemble_Z_prepare(gB), k, omega)
    T1_half = Z_B - (Z_A + Z_B - Z)
    seg_h, seg_t, seg_p0 = geom["seg_h"], geom["seg_t"], geom["seg_p0"]
    cent = seg_p0 + 0.5 * seg_h[:, None] * seg_t
    node = rs._knot_points(geom)[jn]
    s_half = int(geom["wing_seg"][jn, 1])
    M0 = rs._seg_moments_from_prepared(
        rs._seg_moments_prepare(
            np.array([node, cent[s_half]]), geom, rs._kernel_radius(geom)
        ),
        k,
        2,
        need_m1=False,
    )[0]
    dM0 = M0[1] - M0[0]
    prep = rs._assemble_Z_prepare(geom)
    T2h = dM0[prep["s_a"]] * prep["q_a"] + dM0[prep["s_b"]] * prep["q_b"]
    fs["ref"] = (T1_half[jn] - T2h / (1j * omega * rs.eps))[cols]
    fs["cols"] = cols
    fs["A3"] = CF.path_test_axis(
        geom["n_basis_total"], rs._path_test_rows(geom, [jn], halves="B")
    )
    return fs["ref"], fs["cols"], fs["A3"]


def _node_row_rel(fs, **kw):
    """Razor's junction row chopped at the node, trunk vs razor's own."""
    ref, cols, A3 = _razor_reference(fs)
    jn = fs["geom"]["n_basis_total"] - 1
    B = CF.axis_data(fs["ctx"], fs["seg_below"], **kw)
    got = -CF.cross_complete_block(fs["ctx"], A3, B, corner=False)[jn, cols]
    return float(np.abs(got - ref).max() / np.abs(ref).max())


def test_the_source_order_alone_never_moves_the_node_row(fs):
    """The measurement that corrects `BAR_ROW_HALF`'s recorded reason."""
    vals = [_node_row_rel(fs, q=q) for q in (4, 8, 16, 32)]
    assert all(4e-5 < v < 7e-5 for v in vals), vals
    assert max(vals) - min(vals) < 1e-6, vals


def test_the_panel_order_alone_stops_at_its_own_plateau(fs):
    # two growths, not three: the claim is that the plateau does not move
    # with growth, and the ends of the range say that as well as three points
    # do while keeping this module inside the 5 s ceiling.
    vals = [_node_row_rel(fs, growth=g, panel_order=16) for g in (4.0, 1.5)]
    assert all(1e-6 < v < 5e-6 for v in vals), vals
    assert max(vals) - min(vals) < 1e-9, vals


def test_both_together_reach_the_floor(fs):
    assert _node_row_rel(fs, growth=2.0, panel_order=8, q=8) < 1e-9
    assert _node_row_rel(fs, growth=2.0, panel_order=8, q=12) < 1e-11


# ---------------------------------------------------------------------------
# the OTHER structure: interior and reversed blocks, where panels are inert
#
# Found in momwire#832's review while checking whether `BAR_REVERSED` had been
# swept against the trunk's density (it had not — the sweep was razor's
# quadrature LANE, which is on razor's path side and cannot see this axis).
# Reproduced here before being recorded.


@pytest.fixture(scope="module")
def interior(fs):
    geom = fs["geom"]
    bo = np.asarray(geom["basis_offsets"])
    rs = fs["rs"]
    Z = rs._assemble_Z_from_prepared(geom, rs._assemble_Z_prepare(geom), rs.k, rs.omega)
    return dict(
        b_below=np.arange(bo[0], bo[1]),
        b_above=np.arange(bo[1], bo[2]),
        Z=Z,
    )


# A CORNER of the block rather than all 22 above bases and their 22 segments:
# the claim is about a plateau in the block's agreement, which every entry of
# it shares, and the full block makes this the most expensive test in the file
# by an order of magnitude. Six above segments, and the bases both of whose
# wings lie inside them.
_SEG_CAP = 6


def _interior_blocks(fs, it, **kw):
    """(reversed, forward) agreement against razor's free-space truth."""
    rs, geom, ctx = fs["rs"], fs["geom"], fs["ctx"]
    n = geom["n_basis_total"]
    seg_a = fs["seg_above"][:_SEG_CAP]
    inside = set(seg_a.tolist())
    above = np.array(
        [
            m
            for m in it["b_above"]
            if all(int(geom["wing_seg"][m, j]) in inside for j in (0, 1))
        ]
    )
    A = CF.path_test_axis(n, rs._path_test_rows(geom, above))
    P = CF.path_test_axis(n, rs._path_test_rows(geom, it["b_below"]))
    ax_a = CF.axis_data(ctx, seg_a, **kw)
    ax_b = CF.axis_data(ctx, fs["seg_below"], **kw)
    out = []
    for blk, rows, cols in (
        (
            -CF.cross_complete_block_reversed(ctx, P, ax_a, corner=False),
            it["b_below"],
            above,
        ),
        (
            -CF.cross_complete_block(ctx, A, ax_b, corner=False),
            above,
            it["b_below"],
        ),
    ):
        ix = np.ix_(rows, cols)
        ref = it["Z"][ix]
        out.append(float(np.abs(blk[ix] - ref).max() / np.abs(ref).max()))
    return out


# `slow`, like `test_interior_rows_against_interior_columns` next door and for
# the same reason: an interior x interior block over 22 above bases is the
# expensive shape in this file, and these run it four times.


@pytest.mark.slow
def test_the_interior_blocks_plateau_is_two_knobs_not_one(fs, interior):
    """The mirror image of the node row, in one test because the three claims
    share the expensive part.

    `_graded_u` fires only on segments touching the interface and these bases
    exclude the junction, so the panels move nothing here — where on the node
    row they are the knob that matters. The source order alone plateaus at
    1.07e-07. Together they leave 6.5613e-06 by four orders, which is what
    makes that number a property of the shipped density rather than of the
    method, and what `BAR_INTERIOR` / `BAR_REVERSED` now say.
    """
    rev, fwd = _interior_blocks(fs, interior)
    assert 5e-6 < rev < 8e-6 and 5e-6 < fwd < 8e-6, (rev, fwd)
    # the two blocks are not independently 6.5e-6; they are the same number
    assert rev == pytest.approx(fwd, rel=1e-6)

    # panels alone: inert to the digits that matter
    p_rev, p_fwd = _interior_blocks(fs, interior, growth=1.5, panel_order=12)
    assert p_rev == pytest.approx(rev, rel=1e-4), (rev, p_rev)
    assert p_fwd == pytest.approx(fwd, rel=1e-4), (fwd, p_fwd)

    # both together: four orders off the plateau
    b_rev, b_fwd = _interior_blocks(fs, interior, growth=1.5, panel_order=12, q=8)
    assert b_rev < 1e-9 and b_fwd < 1e-9, (b_rev, b_fwd)
