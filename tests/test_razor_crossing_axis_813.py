"""The crossing trunk on RAZOR rows — momwire#813, the trunk-facing half.

`_crossing_fill` (post-#801) takes a `CrossingContext` and an axis dict per
side. momwire#651's test-side probe showed that razor's razor-blade testing
can be written in that dict's own language — F = 1 on each row's own path,
Fd = 0, ends = the path's two centroids with the T2 signs — and that the
trunk then reproduces razor's own free-space fill at ε̃ = 1, where the
interface vanishes and razor IS the truth. This module turns the probe into
code: `RazorSolver._crossing_context` (the tents as `BasisPolynomials`),
`RazorSolver._path_test_rows` + `_crossing_fill.path_test_axis` (the test
axis), and the `corner=False` switch on the block functions, with the
probe's numbers as the gates.

Measured 2026-09-02 on `crossing_deck(level=1)` (30 segments, 29 bases):

  * interior rows × interior columns: 6.6e-6 relative, ratio exactly 1;
  * the junction tent's column: 7.2e-9 vs razor's below-wing piece, with the
    node end terms contributing exactly 0 to razor rows;
  * the junction tent's row, above half chopped at the node: 5.3e-5 vs
    razor's kernel chopped the same way — WITH `corner=False`. The corner is
    a Galerkin by-parts term; on a path-tested row it adds 1.9e5 where
    razor's truth has none.

What this module does NOT do: assemble a razor-tested mixed-medium Z. That
is #813's other half, and it needs razor's T2 evaluated AT the node (razor
never does — see `test_the_sigma_trick_does_not_chop_the_path`), which is a
change to the razor fill rather than to the trunk.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import RazorSolver
from momwire import _crossing_fill as CF

from test_crossing_serve_524 import crossing_deck

BAR_INTERIOR = 1e-5  # measured 6.6e-6
BAR_COLUMN = 1e-7  # measured 7.2e-9
BAR_ROW_HALF = (
    1e-4  # measured 5.3e-5 (quadrature: 4-point source Gauss vs razor's 12 + statics)
)


@pytest.fixture(scope="module")
def free_space():
    """The crossing deck's geometry in FREE SPACE on razor: the truth at ε̃ = 1."""
    deck = crossing_deck(1)
    fs = {
        k: v
        for k, v in deck.items()
        if k not in ("ground_z", "ground_eps", "ground_model")
    }
    # n_qp_path=8 rather than the default 32: the path order is razor's own
    # observer set on BOTH sides of every comparison here, so it does not
    # enter the agreement; it does set the size of the trunk's designed-table
    # evaluation, which is what the block tests spend their time on.
    rs = RazorSolver(**fs, nec5_quadrature=False, n_qp_path=8)
    geom = rs._build_geometry()
    prep = rs._assemble_Z_prepare(geom)
    Z = rs._assemble_Z_from_prepared(geom, prep, rs.k, rs.omega)
    # ε̃ = 1 context on the same geometry, with the plane at z = 0 where the
    # two wires meet (the free-space solver has no ground_z; the context's
    # default is 0, which is where the deck puts the node).
    ctx = rs._crossing_context(geom, ground_eps=(1.0, 0.0))
    seg_off = np.asarray(geom["seg_offsets"])
    bas_off = np.asarray(geom["basis_offsets"])
    return dict(
        rs=rs,
        geom=geom,
        prep=prep,
        Z=Z,
        ctx=ctx,
        b_idx=np.arange(seg_off[0], seg_off[1]),  # wire 0 = below
        a_idx=np.arange(seg_off[1], seg_off[2]),  # wire 1 = above
        cols_below=np.arange(bas_off[0], bas_off[1]),
        rows_above=np.arange(bas_off[1], bas_off[2]),
        jn=geom["n_basis_total"] - 1,
    )


def test_the_context_is_the_tent_basis(free_space):
    ctx, geom = free_space["ctx"], free_space["geom"]
    assert ctx.basis.degree == 1
    assert ctx.medium.eps_t == 1 and ctx.medium.c2 == 0 and ctx.medium.a_m == 0
    # An interior tent: rising on wing A, falling on wing B, unit σ.
    m = int(free_space["rows_above"][0])
    h = geom["seg_h"][geom["wing_seg"][m, 0]]
    assert np.allclose(ctx.basis.polys[m, 0], (0.0, 1.0 / h))
    assert np.allclose(ctx.basis.polys[m, 1], (1.0, -1.0 / h))


@pytest.mark.slow
def test_interior_rows_against_interior_columns(free_space):
    f = free_space
    A = CF.path_test_axis(
        f["ctx"].basis.polys.shape[0],
        f["rs"]._path_test_rows(f["geom"], f["rows_above"]),
    )
    B = CF.axis_data(f["ctx"], f["b_idx"])
    t_ab = CF.cross_complete_block(f["ctx"], A, B)
    ref = f["Z"][np.ix_(f["rows_above"], f["cols_below"])]
    got = -t_ab[np.ix_(f["rows_above"], f["cols_below"])]
    rel = np.abs(got - ref).max() / np.abs(ref).max()
    assert rel < BAR_INTERIOR, rel
    big = np.abs(ref) > 1e-3 * np.abs(ref).max()
    ratio = got[big] / ref[big]
    assert abs(np.median(ratio) - 1.0) < 1e-6  # no constant, no sign to absorb


@pytest.mark.slow
def test_the_junction_column_is_the_below_wing_piece(free_space):
    """By wing-linearity: the trunk's below axis sees only the tent's below
    wing, so its column is razor's fill with the above wing zeroed."""
    f = free_space
    rs, geom, jn = f["rs"], f["geom"], f["jn"]
    A = CF.path_test_axis(
        geom["n_basis_total"], rs._path_test_rows(geom, f["rows_above"])
    )
    B = CF.axis_data(f["ctx"], f["b_idx"])
    t_ab = CF.cross_complete_block(f["ctx"], A, B)
    g2 = dict(geom)
    g2["wing_sigma"] = geom["wing_sigma"].copy()
    g2["wing_sigma"][jn, 1] = 0.0
    Z_A = rs._assemble_Z_from_prepared(g2, rs._assemble_Z_prepare(g2), rs.k, rs.omega)
    col_ref = Z_A[f["rows_above"], jn]
    col_got = -t_ab[f["rows_above"], jn]
    assert (
        np.abs(col_got - col_ref).max() / np.abs(f["Z"][f["rows_above"], jn]).max()
        < BAR_COLUMN
    )
    # The node source-end terms contribute nothing to path-tested rows here.
    B_noend = dict(B)
    B_noend["ends"] = [e for e in B["ends"] if abs(e[0][2]) > 1e-9]
    t_noend = CF.cross_complete_block(f["ctx"], A, B_noend)
    assert np.array_equal(t_ab[f["rows_above"], jn], t_noend[f["rows_above"], jn])


def _razor_half_row(f, halves):
    """Razor's junction row chopped at the node, from razor's own kernel:
    T1 of one half (by the σ decomposition) plus T2 between the node and
    that half's centroid (from the moments at those two observers)."""
    rs, geom, prep, Z, jn = f["rs"], f["geom"], f["prep"], f["Z"], f["jn"]
    k, omega = rs.k, rs.omega
    gA = dict(geom)
    gA["wing_sigma"] = geom["wing_sigma"].copy()
    gA["wing_sigma"][jn, 1] = 0.0
    gB = dict(geom)
    gB["wing_sigma"] = geom["wing_sigma"].copy()
    gB["wing_sigma"][jn, 0] = 0.0
    Z_A = rs._assemble_Z_from_prepared(gA, rs._assemble_Z_prepare(gA), k, omega)
    Z_B = rs._assemble_Z_from_prepared(gB, rs._assemble_Z_prepare(gB), k, omega)
    T2t = Z_A + Z_B - Z  # the full-path T2 term, in Z units
    T1_half = (Z_B if halves == "B" else Z_A) - T2t
    seg_h, seg_t, seg_p0 = geom["seg_h"], geom["seg_t"], geom["seg_p0"]
    cent = seg_p0 + 0.5 * seg_h[:, None] * seg_t
    node = rs._knot_points(geom)[jn]
    s_half = int(geom["wing_seg"][jn, 1 if halves == "B" else 0])
    before, after = (node, cent[s_half]) if halves == "B" else (cent[s_half], node)
    obs = np.array([before, after])
    M0 = rs._seg_moments_from_prepared(
        rs._seg_moments_prepare(obs, geom, rs._kernel_radius(geom)), k, 2, need_m1=False
    )[0]
    dM0 = M0[1] - M0[0]
    T2h = dM0[prep["s_a"]] * prep["q_a"] + dM0[prep["s_b"]] * prep["q_b"]
    return T1_half[jn] - T2h / (1j * omega * rs.eps)


@pytest.mark.slow
def test_the_junction_rows_above_half_with_no_corner(free_space):
    f = free_space
    rs, geom, jn = f["rs"], f["geom"], f["jn"]
    A3 = CF.path_test_axis(
        geom["n_basis_total"], rs._path_test_rows(geom, [jn], halves="B")
    )
    B = CF.axis_data(f["ctx"], f["b_idx"])
    ref = _razor_half_row(f, "B")[f["cols_below"]]
    got = -CF.cross_complete_block(f["ctx"], A3, B, corner=False)[jn, f["cols_below"]]
    assert np.abs(got - ref).max() / np.abs(ref).max() < BAR_ROW_HALF
    # With the corner, the (jn, jn) entry moves by an order of magnitude
    # (measured: 1.9e5 added to a 1.4e4 entry).
    with_corner = -CF.cross_complete_block(f["ctx"], A3, B, corner=True)[jn, jn]
    without = -CF.cross_complete_block(f["ctx"], A3, B, corner=False)[jn, jn]
    assert abs(with_corner - without) > 10 * abs(without)


def test_the_sigma_trick_does_not_chop_the_path(free_space):
    """The trap the probe recorded: zeroing a wing's σ zeroes its half of
    the T1 path and its doublet, but T2 still spans the full path's two
    centroids — so "razor with one wing off" is not "razor chopped at the
    node", and razor never evaluates Φ at the node. Pinned so the next
    reader does not rediscover it the hard way."""
    f = free_space
    rs, geom, jn = f["rs"], f["geom"], f["jn"]
    gB = dict(geom)
    gB["wing_sigma"] = geom["wing_sigma"].copy()
    gB["wing_sigma"][jn, 0] = 0.0
    Z_B = rs._assemble_Z_from_prepared(gB, rs._assemble_Z_prepare(gB), rs.k, rs.omega)
    chopped = _razor_half_row(f, "B")
    rel = (
        np.abs(Z_B[jn, f["cols_below"]] - chopped[f["cols_below"]]).max()
        / np.abs(chopped[f["cols_below"]]).max()
    )
    assert rel > 0.1, rel  # measured 0.46: they are different objects
