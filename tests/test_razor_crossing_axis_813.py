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

Half 2 step (1), the REVERSED block (2026-09-03). The block above builds is
above rows x below columns. The other one — BELOW rows (razor paths on the
below wires) x ABOVE columns — is `cross_complete_block_reversed`. bspline
gets it as `t_ab.T` by Galerkin reciprocity; a path-tested fill cannot assume
that, so it is built and the reciprocity is MEASURED:

  * eps~ = 1, path-tested rows vs razor's free-space Z[below, above]:
    6.56e-06 relative, ratio exactly 1 — the same interior class the forward
    block reached, on both quadrature lanes;
  * the main sandwich is the forward's TRANSPOSED, exact to 3e-16 in both
    media: the designed tables take the above axis in the `z` slot whichever
    ROLE it plays, and no kernel swap is needed. What the reversed block
    re-assigns is the BY-PARTS terms, by role: BT on the test axis's ends,
    SW + SQ on the source's;
  * every end term except SW is then bit-identical to the transpose in both
    media. SW is the whole difference, and it is exactly 0 at eps~ = 1
    because W = 0 in a homogeneous medium — which is why the eps~ = 1
    collapse cannot settle it and #813 carried it as open derivation (b).
    5312ca5 settled it by measurement: SW is the by-parts REMNANT of the
    vertical-current coupling (`s_w1 + SW` reproduces the direct dz'W form to
    2.4e-8; `s_w1` alone is 29x off), so it pairs with `s_w1` by geometry —
    the BELOW axis's ends against the ABOVE axis's t_z — whichever side
    tests. Under that pairing (`SW_BY_PARTS`, the default) the reversed block
    reproduces `t_ab.T` bit for bit, so reciprocity comes OUT of the spelling
    instead of being assumed. `SW_BY_ROLE` is the other reading, kept for the
    contrast: 7.94e-04 of the block away at soil A.

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

from test_buried_serve_553 import SOIL_A
from test_crossing_serve_524 import crossing_deck

# 1e-5, measured 6.5613e-06 -- which is a two-knob PLATEAU of the trunk's
# default axis density and not this block's accuracy. Swept properly
# (momwire#836): the source order alone takes it to 1.0746e-07 and stops
# there from q = 6 to 32; the graded panels alone move NOTHING at any order
# or growth, because `_graded_u` fires only on segments touching the
# interface and these bases exclude the junction; both together reach
# 3.2e-16. The block's real agreement with razor's free-space truth is
# machine precision. The bar stands where it is because it gates the
# SHIPPED axis, which is the right thing for it to gate -- but a later bar
# sized from "6.6e-6 is the interior class" would be sized against a
# quadrature setting rather than a method.
BAR_INTERIOR = 1e-5
BAR_COLUMN = 1e-7  # measured 7.2e-9
# 1e-4, measured 5.3e-5. It IS quadrature, and NOT the source Gauss the first
# version of this line named: the chopped half ends AT the node, on the below
# wire's last segment, whose by-parts integrand ~ 1/sqrt(a^2 + s^2) from s = 0
# is carried by `_graded_u`'s a-scale panels rather than by `_NEAR_Q`.
# Measured across the source order 4 -> 32 this number reads 5.326e-5 then
# 5.347e-5 and never moves again; it is `panel_order` and `q` TOGETHER that
# reach 7.7e-11 (momwire#813, `axis_data`'s density knobs). Sizing a later bar
# from the old reason would size it wrong, which is why this says so.
BAR_ROW_HALF = 1e-4


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


# ======================================================================
# The reversed block — below rows x above columns (#813 half 2, step 1)
# ======================================================================
# 1e-5, measured 6.5613e-06. "The same class as BAR_INTERIOR" is true and
# is not the compliment it reads as: the two share a PLATEAU, digit for
# digit at every density setting, and momwire#836's sweep shows both go to
# machine precision (3.2e-16 reversed, 9.5e-16 forward) once the source
# order and the panels move together. The lane sweep that originally stood
# behind this number -- `nec5_quadrature` on/off -- is on razor's path side
# and cannot see the trunk's axis at all, so it was not coverage.
BAR_REVERSED = 1e-5
BAR_LANES = 1e-12  # dense vs split on Galerkin axes: measured 3.3e-19


def _reversed_setup(nec5_quadrature=False):
    """The crossing deck on razor in free space, plus the axes of the
    reversed block: razor's paths on the BELOW wire testing, the ABOVE
    wire's tents sourcing."""
    deck = crossing_deck(1)
    fs = {
        k: v
        for k, v in deck.items()
        if k not in ("ground_z", "ground_eps", "ground_model")
    }
    rs = RazorSolver(**fs, nec5_quadrature=nec5_quadrature, n_qp_path=8)
    geom = rs._build_geometry()
    Z = rs._assemble_Z_from_prepared(geom, rs._assemble_Z_prepare(geom), rs.k, rs.omega)
    so = np.asarray(geom["seg_offsets"])
    bo = np.asarray(geom["basis_offsets"])
    return dict(
        rs=rs,
        geom=geom,
        Z=Z,
        b_idx=np.arange(so[0], so[1]),
        a_idx=np.arange(so[1], so[2]),
        bases_below=np.arange(bo[0], bo[1]),
        bases_above=np.arange(bo[1], bo[2]),
    )


@pytest.mark.slow
@pytest.mark.parametrize("nec5_quadrature", (False, True))
def test_the_reversed_block_is_razors_below_rows_at_eps_one(nec5_quadrature):
    """The gate for step (1): with the plane's two sides swapped, the trunk
    still reproduces razor's own free-space fill at eps~ = 1.

    Both quadrature lanes, because the path axis is razor's observer set on
    both sides of the comparison and must not enter the agreement.
    """
    f = _reversed_setup(nec5_quadrature)
    ctx = f["rs"]._crossing_context(f["geom"], ground_eps=(1.0, 0.0))
    P = CF.path_test_axis(
        f["geom"]["n_basis_total"],
        f["rs"]._path_test_rows(f["geom"], f["bases_below"]),
    )
    Q = CF.axis_data(ctx, f["a_idx"])
    ref = f["Z"][np.ix_(f["bases_below"], f["bases_above"])]
    got = -CF.cross_complete_block_reversed(ctx, P, Q, corner=False)[
        np.ix_(f["bases_below"], f["bases_above"])
    ]
    rel = np.abs(got - ref).max() / np.abs(ref).max()
    assert rel < BAR_REVERSED, rel
    big = np.abs(ref) > 1e-3 * np.abs(ref).max()
    ratio = got[big] / ref[big]
    assert abs(np.median(ratio) - 1.0) < 1e-6  # no constant, no sign to absorb


@pytest.mark.slow
def test_the_main_sandwich_is_the_forward_transposed():
    """No kernel swap: the designed tables take the ABOVE axis in the `z`
    slot whichever role it plays, so the reversed main sandwich is the
    forward's transpose exactly. Measured with the by-parts ends removed
    (`corner=False` and the SW/SQ/BT terms subtracted) — what is left is the
    sandwich, in both media."""
    f = _reversed_setup()
    for ground_eps in ((1.0, 0.0), SOIL_A):
        ctx = f["rs"]._crossing_context(f["geom"], ground_eps=ground_eps)
        A, B = CF.axis_data(ctx, f["a_idx"]), CF.axis_data(ctx, f["b_idx"])
        et, _em, kp, _km, _c2, _am = ctx.medium
        gz, c1 = float(ctx.ground_z), CF._c1_moment(ctx.omega, ctx.mu)
        # The whole block, BIT for bit, under the reciprocity-carried SW.
        # The ends are separately shown bit-equal to the transpose, so this
        # pins the sandwich as the transpose too.
        whole = CF.cross_complete_block_reversed(
            ctx, B, A, corner=False, sw_end=CF.SW_BY_PARTS
        )
        assert np.array_equal(
            whole, CF.cross_complete_block(ctx, A, B, corner=False).T
        ), ground_eps
        # The isolated main part agrees to roundoff — not bit, because
        # `(main + ends) - ends` cancels where main is 0 and ends is not.
        rev_main = whole - CF._ends_and_corner_reversed(
            ctx, B, A, et, kp, c1, gz, corner=False, sw_end=CF.SW_BY_PARTS
        )
        fwd_main = CF._main_sandwich(ctx, A, B, et, kp, c1, gz)
        d = np.abs(rev_main - fwd_main.T).max() / np.abs(fwd_main).max()
        assert d < 1e-14, (ground_eps, d)


@pytest.mark.slow
def test_reciprocity_is_measured_not_assumed():
    """`t_ba` vs `t_ab.T` on Galerkin axes, where reciprocity is the identity
    bspline's fill already rides — so a correct reversed block must reproduce
    the transpose there, and the SW placement is the one free choice.

    The default `SW_BY_PARTS` reproduces it EXACTLY (bit, in both media),
    which is the unit's answer: reciprocity holds for the path-tested block
    once SW is paired with `s_w1` as 5312ca5 measured it. `SW_BY_ROLE` — SW
    read as a source-side end term instead — agrees at eps~ = 1 (W = 0) and
    is 7.94e-04 away at soil A. The number is pinned as a band so a move in
    either direction is a finding rather than silent drift.
    """
    f = _reversed_setup()
    seen = {}
    for label, ground_eps in (("eps1", (1.0, 0.0)), ("soilA", SOIL_A)):
        ctx = f["rs"]._crossing_context(f["geom"], ground_eps=ground_eps)
        A, B = CF.axis_data(ctx, f["a_idx"]), CF.axis_data(ctx, f["b_idx"])
        fwd = CF.cross_complete_block(ctx, A, B, corner=False).T
        for sw in (CF.SW_BY_ROLE, CF.SW_BY_PARTS):
            rev = CF.cross_complete_block_reversed(ctx, B, A, corner=False, sw_end=sw)
            seen[(label, sw)] = np.abs(rev - fwd).max() / np.abs(fwd).max()
    # The by-parts pairing: exact in both media.
    assert seen[("eps1", CF.SW_BY_PARTS)] == 0.0
    assert seen[("soilA", CF.SW_BY_PARTS)] == 0.0
    # The rejected role reading: exact where W vanishes...
    assert seen[("eps1", CF.SW_BY_ROLE)] == 0.0
    # ...and 7.938e-04 away where it does not, as a band.
    assert 5e-4 < seen[("soilA", CF.SW_BY_ROLE)] < 1.2e-3, seen


@pytest.mark.slow
def test_only_the_sw_end_term_separates_the_two_spellings():
    """The whole reciprocity gap is one term. W is the only kernel that is 0
    at eps~ = 1 and nonzero at soil, and SW is the only end term carrying
    it — so the two spellings are bit-identical wherever W vanishes."""
    f = _reversed_setup()
    ctx1 = f["rs"]._crossing_context(f["geom"], ground_eps=(1.0, 0.0))
    A1, B1 = CF.axis_data(ctx1, f["a_idx"]), CF.axis_data(ctx1, f["b_idx"])
    a = CF.cross_complete_block_reversed(ctx1, B1, A1, sw_end=CF.SW_BY_ROLE)
    b = CF.cross_complete_block_reversed(ctx1, B1, A1, sw_end=CF.SW_BY_PARTS)
    assert np.array_equal(a, b)
    # W is exactly zero there, which is WHY.
    tb = CF._tables(
        ctx1,
        ctx1.medium.eps_t,
        ctx1.medium.k_p,
        np.array([0.3]),
        np.array([1.5]),
        np.array([-0.8]),
        CF._CROSS_RTOL,
    )
    assert tb["W"][0] == 0 and tb["dzW"][0] == 0
    # At soil it does not vanish, and neither does the difference.
    ctx2 = f["rs"]._crossing_context(f["geom"], ground_eps=SOIL_A)
    A2, B2 = CF.axis_data(ctx2, f["a_idx"]), CF.axis_data(ctx2, f["b_idx"])
    c = CF.cross_complete_block_reversed(ctx2, B2, A2, sw_end=CF.SW_BY_ROLE)
    d = CF.cross_complete_block_reversed(ctx2, B2, A2, sw_end=CF.SW_BY_PARTS)
    assert not np.array_equal(c, d)


@pytest.mark.slow
def test_the_two_trunk_lanes_agree_on_galerkin_axes():
    """Dense vs the #688 admissibility split, reversed, both media."""
    f = _reversed_setup()
    for ground_eps in ((1.0, 0.0), SOIL_A):
        ctx = f["rs"]._crossing_context(f["geom"], ground_eps=ground_eps)
        A, B = CF.axis_data(ctx, f["a_idx"]), CF.axis_data(ctx, f["b_idx"])
        dense = CF.cross_complete_block_reversed(ctx, B, A)
        split = CF.cross_complete_block_reversed_split(
            ctx, f["b_idx"], f["a_idx"], B, A
        )
        rel = np.abs(dense - split).max() / np.abs(dense).max()
        assert rel < BAR_LANES, (ground_eps, rel)


def test_the_split_refuses_a_path_tested_axis():
    """The trap this unit found and did not leave armed.

    The split's far blocks ride COARSE axes rebuilt by `axis_data` from the
    context's basis, and a testing path has no coarse spelling — so a
    path-tested axis silently gets Galerkin tents on exactly those blocks.
    Measured 2.04e-01 relative, in BOTH directions, on `crossing_deck(1)`.
    It is pre-existing on the forward split (half 1's gates are all dense, so
    nothing hit it) and it would have reached half 2's masked assembly.
    """
    f = _reversed_setup()
    ctx = f["rs"]._crossing_context(f["geom"], ground_eps=(1.0, 0.0))
    P = CF.path_test_axis(
        f["geom"]["n_basis_total"],
        f["rs"]._path_test_rows(f["geom"], f["bases_below"]),
    )
    Q = CF.axis_data(ctx, f["a_idx"])
    assert P.get("path_tested") is True
    assert not Q.get("path_tested")
    with pytest.raises(ValueError, match="path-tested axis"):
        CF.cross_complete_block_reversed_split(ctx, f["b_idx"], f["a_idx"], P, Q)
    with pytest.raises(ValueError, match="path-tested axis"):
        CF.cross_complete_block_split(ctx, f["a_idx"], f["b_idx"], P, Q)
