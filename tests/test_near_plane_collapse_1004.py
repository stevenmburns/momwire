"""momwire#1004: the cross block when the two media come closer than a segment.

The transmitted block's Gauss order was calibrated on a deck whose above and
below wires are never nearer than a segment (`n_qp_buried_field`'s q = 6 table,
momwire#553's fifth inversion). Below that the same 1/R^3 argument bites again
and 6 is badly short. `near_q_factor` scales the order with `seg_h /
separation`; these are the gates that say it works, that it is inert
everywhere else, and that the two trunks do the same thing.

THE LADDER
----------
Two 0.6 m wires, 15 segments each (h = 0.04 m), one at +gap and one at -gap,
at eps-tilde = 1 where the whole mixed fill must reproduce the free-space fill
exactly. The cross quadrant's residual, graded against the free-space block's
own magnitude, measured 2026-09-11:

    gap      h/sep   factor   bspline      SG          before (bspline / SG)
    0.5 m     0.04     1      5.3993e-06   5.3952e-06  unchanged, bit for bit
    0.1 m     0.20     1      3.8203e-05   3.8113e-05  unchanged, bit for bit
    0.02 m    1.00     1      5.5971e-04   5.5844e-04  unchanged, bit for bit
    0.005 m   4.00     4      4.2419e-03   4.1181e-03  2.9536e-01 / 5.4885e-02
    0.002 m  10.00    10      8.9285e-03   1.6163e-02  1.9496e+01 / 1.4924e+00

What is left at the rule's order is the GRID's own near-plane interpolation
floor, which q cannot go below and which grows as the plane is approached —
which is why the bars below grow with it rather than being one number. The
bars sit about 2x over the measured floors: far under the pre-fix readings, so
reverting the rule fails these gates rather than squeaking past them, and
`test_the_raised_order_is_what_buys_it` pins that directly by pinning the
factor back to 1 in-process.

THE TWO SIGN CONVENTIONS
------------------------
bspline's assembled cross quadrant collapses onto the free-space one with a
plain difference. **SG's is stored NEGATED** — its mixed stitch carries the
below class's basis coefficients with the opposite sign, which cancels in the
below/below quadrant and in the solved impedance (measured: both bases collapse
to 1.26e-03 and 1.28e-03 at the 5 mm rung, agreeing with each other to 2 %) and
survives only on the cross quadrants. So `|G_mixed - G_free|` reads a constant
2.0 on SG and IS the residual on bspline, and the other way round for
`|G_mixed + G_free|`. Both are asserted at every rung: the residual on the
family's own convention, and 2.0 on the other one, because a near-2.0 reading
is what an actual sign error would also produce and "the metric could not
move" is how three sweeps came to read inert on this issue.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import momwire.bspline as _bs
from momwire import BSplineSolver, SinusoidalGalerkinSolver
from momwire import _below_interface as BI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_sg_crossing_980_d3 import _JUNCTIONS, _WIRES  # noqa: E402

# One worker for the whole file. Every gate here builds the same handful of
# Sommerfeld grids, and under `--dist loadgroup` an ungrouped file scatters
# across workers so each one pays for its own. Measured: the arbiter gate
# reads 1.55 s alone and 5.63 s scattered, which is over the unmarked ceiling
# for no reason other than a cold cache. `conftest` groups by file elsewhere
# through `_FIXTURE_GROUP_FILES`; a file-level mark is the same thing said
# locally, and it is seen because pytest applies it during collection.
pytestmark = pytest.mark.xdist_group("near_plane_1004")

C0 = 299792458.0
WL7 = C0 / 7e6
EPS_ONE = (1.0, 0.0)
NSEG = 15
LEN = 0.6

# gap -> (bspline bar, SG bar). The floors these sit over are in the module
# docstring; each bar is about 2x its rung's measurement.
LADDER = {
    0.5: (1.1e-05, 1.1e-05),
    0.1: (8.0e-05, 8.0e-05),
    0.02: (1.2e-03, 1.2e-03),
    0.005: (9.0e-03, 9.0e-03),
    0.002: (2.0e-02, 3.5e-02),
}


def _wires(gap):
    return [
        np.array([(0.0, 0.0, gap), (LEN, 0.0, gap)]),
        np.array([(0.0, 0.0, -gap), (LEN, 0.0, -gap)]),
    ]


def mk(cls, gap, *, free=False):
    ground = (
        {}
        if free
        else dict(ground_z=0.0, ground_eps=EPS_ONE, ground_model="sommerfeld")
    )
    return cls(
        wires=_wires(gap),
        n_per_edge_per_wire=[[NSEG], [NSEG]],
        feeds=[(0, 0.3, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        **ground,
    )


def _quiet():
    warnings.simplefilter("ignore")


def G_bs(s):
    with warnings.catch_warnings():
        _quiet()
        geom = s._build_geometry()
        supp_seg, polys, *_ = s._build_basis_polynomials(geom)
        return s._compute_Z_operator(geom, supp_seg, polys)


def G_sg(s):
    with warnings.catch_warnings():
        _quiet()
        geom = s._build_geometry()
        if s._is_mixed(geom):
            return s._assemble_Z(geom, s.k)[0]
        with s._operating_medium(geom):
            return s._assemble_Z(geom, s.k)[0]


def above_mask_bs(s):
    """Which bspline bases live on the ABOVE wire — by support, not by count."""
    with warnings.catch_warnings():
        _quiet()
        geom = s._build_geometry()
        supp_seg, *_ = s._build_basis_polynomials(geom)
        n_above = int(np.count_nonzero(~s._below_segments(geom)))
    return supp_seg.max(axis=1) < n_above


def above_mask_sg(gap, n_total):
    """SG dofs run wire by wire, so the above wire's own solve counts them —
    D2's `test_a_mixed_deck_reproduces_each_class` splits the same way."""
    na = G_sg(
        SinusoidalGalerkinSolver(
            wires=[_wires(gap)[0]],
            n_per_edge_per_wire=[[NSEG]],
            feeds=[(0, 0.3, 1 + 0j)],
            wavelength=WL7,
            wire_radius=0.001,
        )
    ).shape[0]
    assert 0 < na < n_total, (na, n_total)
    m = np.zeros(n_total, bool)
    m[:na] = True
    return m


def cross_residuals(Gm, Gf, is_a):
    """`(same_sign, opposite_sign)` over both cross quadrants, each graded
    against the free-space block's own magnitude."""
    ia, ib = np.nonzero(is_a)[0], np.nonzero(~is_a)[0]
    minus, plus = 0.0, 0.0
    for r, c in ((ia, ib), (ib, ia)):
        m, f = Gm[np.ix_(r, c)], Gf[np.ix_(r, c)]
        scale = np.abs(f).max()
        assert scale > 0.0, "the cross quadrant is empty; nothing was tested"
        minus = max(minus, float(np.abs(m - f).max() / scale))
        plus = max(plus, float(np.abs(m + f).max() / scale))
    return minus, plus


def collapse(base, gap):
    """`(residual, the other sign)` for one base at one rung."""
    if base == "bspline":
        Gm, Gf = G_bs(mk(BSplineSolver, gap)), G_bs(mk(BSplineSolver, gap, free=True))
        is_a = above_mask_bs(mk(BSplineSolver, gap))
        minus, plus = cross_residuals(Gm, Gf, is_a)
        return minus, plus  # bspline's collapse is the plain difference
    Gm = G_sg(mk(SinusoidalGalerkinSolver, gap))
    Gf = G_sg(mk(SinusoidalGalerkinSolver, gap, free=True))
    is_a = above_mask_sg(gap, Gm.shape[0])
    minus, plus = cross_residuals(Gm, Gf, is_a)
    return plus, minus  # SG stores the cross quadrant negated


# ----------------------------------------------------------------------
# G-1004-1: the ladder, both bases
# ----------------------------------------------------------------------


@pytest.mark.parametrize("base", ["bspline", "sg"])
@pytest.mark.parametrize("gap", sorted(LADDER, reverse=True))
def test_the_near_plane_cross_block_collapses(base, gap, record_property):
    bar = LADDER[gap][0 if base == "bspline" else 1]
    rel, other = collapse(base, gap)
    record_property(f"collapse_{base}_{gap}", rel)
    record_property(f"other_sign_{base}_{gap}", other)
    assert rel < bar, f"{base} at gap {gap} m: {rel:.4e}, bar {bar:.1e}"
    # The other sign is the family's convention showing, not a near miss: it
    # has to read 2 + (this residual), and a genuine sign error would put the
    # residual HERE and 2.0 above.
    assert abs(other - 2.0) < 10.0 * bar + 1e-9, (
        f"{base} at gap {gap} m: the opposite sign reads {other:.6e}, not ~2 — "
        "the cross quadrant's sign convention has changed"
    )


# ----------------------------------------------------------------------
# G-1004-2: the raised order is what buys it
# ----------------------------------------------------------------------


@pytest.mark.parametrize("base", ["bspline", "sg"])
def test_the_raised_order_is_what_buys_it(base, monkeypatch, record_property):
    """The gate above with the lever pinned off, in the same process.

    Without this the ladder is a gate that cannot fail: bars chosen over
    measured floors pass whatever the code does as long as it does not get
    worse. Pinning the factor to 1 restores the pre-#1004 order exactly, so
    this is the fix's own before-number, re-measured on every run.
    """
    gap = 0.005
    fixed, _ = collapse(base, gap)
    monkeypatch.setattr(BI, "near_q_factor", lambda separation, seg_h: 1)
    pinned, _ = collapse(base, gap)
    record_property(f"pinned_to_one_{base}", pinned)
    assert pinned > 10.0 * fixed, (
        f"{base}: pinning the order back to 1 reads {pinned:.4e} against "
        f"{fixed:.4e} — the raised order is no longer doing the work"
    )


# ----------------------------------------------------------------------
# G-1004-3: the cross-family arbiter
# ----------------------------------------------------------------------


def test_the_two_bases_agree_on_the_near_plane_deck(record_property):
    """The arbiter the suite uses to adjudicate this family, at the rung the
    fix is for. Before the fix the two disagreed by 6x (bspline 4.24 against
    SG 0.719 on the same deck); they now agree to 2 %."""
    gap = 0.005
    with warnings.catch_warnings():
        _quiet()
        rels = {}
        for tag, cls in (("bspline", BSplineSolver), ("sg", SinusoidalGalerkinSolver)):
            zm = complex(mk(cls, gap).compute_impedance()[0])
            zf = complex(mk(cls, gap, free=True).compute_impedance()[0])
            rels[tag] = abs(zm - zf) / abs(zf)
            record_property(f"z_collapse_{tag}", rels[tag])
    for tag, r in rels.items():
        assert r < 5e-03, f"{tag} impedance collapse {r:.4e}"
    spread = abs(rels["bspline"] - rels["sg"]) / max(rels.values())
    assert spread < 0.10, f"the two bases' collapse residuals differ by {spread:.1%}"


# ----------------------------------------------------------------------
# G-1004-4: the order is inferred, never taken on trust
# ----------------------------------------------------------------------


def test_the_field_block_infers_the_order_from_its_own_arrays():
    """`_field_galerkin_block` reshapes by q. A q from a second source that
    disagreed with the nodes would reshape a correct table into a wrong answer
    with NO exception, which is the whole reason the order is not passed."""
    s = mk(BSplineSolver, 0.005)
    with warnings.catch_warnings():
        _quiet()
        geom = s._build_geometry()
        supp_seg, polys, *_ = s._build_basis_polynomials(geom)
        below = s._below_segments(geom)
    a_idx, b_idx = np.nonzero(~below)[0], np.nonzero(below)[0]
    obs_a, t_a, W_a = s._buried_nodes(geom, a_idx)
    obs_b, t_b, W_b = s._buried_nodes(geom, b_idx)
    hi_a, hi_t, hi_W = s._buried_nodes(geom, a_idx, q_factor=4)

    q = s._n_qp_buried_field()
    assert len(obs_a) == len(a_idx) * q
    assert len(hi_a) == len(a_idx) * 4 * q
    assert W_a.shape[-1] == q and hi_W.shape[-1] == 4 * q

    def proj(o, to, sr, ts):
        return np.zeros((len(o), len(sr)), dtype=np.complex128)

    # nodes at the raised order, weights at the base one: the exact shape of
    # the disagreement the design exists to make impossible.
    with pytest.raises(AssertionError):
        s._field_galerkin_block(
            supp_seg, polys, proj, a_idx, b_idx, hi_a, hi_t, W_a, obs_b, t_b, W_b
        )
    # and the two axes disagreeing with each other
    with pytest.raises(AssertionError):
        s._field_galerkin_block(
            supp_seg, polys, proj, a_idx, b_idx, hi_a, hi_t, hi_W, obs_b, t_b, W_b
        )
    # the honest call goes through
    Q = s._field_galerkin_block(
        supp_seg, polys, proj, a_idx, b_idx, obs_a, t_a, W_a, obs_b, t_b, W_b
    )
    assert Q.shape == (polys.shape[0], polys.shape[0])


# ----------------------------------------------------------------------
# G-1004-5: where the rule is and is not allowed to fire
# ----------------------------------------------------------------------


def test_the_rule_is_inert_at_a_segment_or_more():
    """h/sep = 1 exactly is what a uniform mesh either side of the plane
    produces, and a bare `ceil` would double that deck's order for a rounding.
    The 0.02 m rung IS that deck: h = 0.04, separation = 0.04."""
    s = mk(BSplineSolver, 0.02)
    with warnings.catch_warnings():
        _quiet()
        geom = s._build_geometry()
        below = s._below_segments(geom)
    sep, h = BI.cross_pair_separation(
        np.asarray(geom["seg_l"]),
        np.asarray(geom["seg_r"]),
        np.nonzero(~below)[0],
        np.nonzero(below)[0],
    )
    assert abs(h / sep - 1.0) < 1e-12, (h, sep)
    assert BI.near_q_factor(sep, h) == 1
    assert BI.n_qp_buried_field(3, separation=sep, seg_h=h) == BI.n_qp_buried_field(3)
    # and a single-medium deck, where there is no cross pair at all
    assert BI.near_q_factor(None, None) == 1
    assert BI.cross_pair_separation(
        np.zeros((2, 3)), np.ones((2, 3)), np.array([], dtype=np.int64), np.array([0])
    ) == (None, None)


def test_the_cap_is_the_cap():
    """`q = 6 * h/sep` is unbounded as the separation goes to zero."""
    assert BI.near_q_factor(1e-12, 1.0) == BI._MAX_NEAR_Q_FACTOR
    assert BI.near_q_factor(1.0 / BI._MAX_NEAR_Q_FACTOR, 1.0) == BI._MAX_NEAR_Q_FACTOR


def _crossing(cls, **kw):
    return cls(
        wires=_WIRES,
        n_per_edge_per_wire=[[15], [15]],
        feeds=[(0, 1.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        junctions=_JUNCTIONS,
        ground_z=0.0,
        ground_eps=EPS_ONE,
        ground_model="sommerfeld",
        **kw,
    )


def test_a_crossing_deck_does_not_raise_the_order_on_either_trunk():
    """A crossing deck's cross pair is `_crossing_fill`'s designed direct
    evaluation — no transmitted grid is built for it, and `serve_plan` does not
    even size one. Raising the order there buys nothing and moves the
    below/below extent the plan sizes on the same nodes. This deck WOULD
    trigger the rule on separation alone: h/sep = 1.288.
    """
    sg = _crossing(SinusoidalGalerkinSolver)
    with warnings.catch_warnings():
        _quiet()
        geom = sg._build_geometry()
        below = sg._below_segments(geom)
        medium = sg._fill_medium(geom)
    sep, h = BI.cross_pair_separation(
        np.asarray(geom["seg_l"]),
        np.asarray(geom["seg_r"]),
        np.nonzero(~below)[0],
        np.nonzero(below)[0],
    )
    assert BI.near_q_factor(sep, h) > 1, (sep, h)

    with warnings.catch_warnings():
        _quiet()
        seg_view = sg._stitch_basis_coefs(geom, below, medium.k_p, medium.k_m)
        seg_view = sg._crossing_wing_view(geom, seg_view, below, medium)
        ctx = sg._stitch_test_context(geom, seg_view, below, medium.k_p, medium.k_m)
        plan = sg._mixed_serve_plan(geom, below, medium, ctx, True)
    assert plan["q_buried_field"] == BI.n_qp_buried_field(sg.n_qp_sommerfeld)

    seen = []
    bs = _crossing(BSplineSolver, degree=1)
    real = BSplineSolver._buried_nodes

    def spy(self, geom, seg_idx, *, q_factor=1):
        seen.append(q_factor)
        return real(self, geom, seg_idx, q_factor=q_factor)

    BSplineSolver._buried_nodes = spy
    try:
        with warnings.catch_warnings():
            _quiet()
            gb = bs._build_geometry()
            supp_seg, polys, *_ = bs._build_basis_polynomials(gb)
            bs._compute_Z_operator_buried(gb, supp_seg, polys)
    finally:
        BSplineSolver._buried_nodes = real
    assert seen and set(seen) == {1}, seen


def test_a_near_plane_deck_does_raise_it_on_the_bspline_trunk():
    """The positive half of the gate above: the same spy on the deck the fix
    is for sees the raised factor, and sees it ONLY on the cross node sets."""
    seen = []
    s = mk(BSplineSolver, 0.005)
    real = BSplineSolver._buried_nodes

    def spy(self, geom, seg_idx, *, q_factor=1):
        seen.append(q_factor)
        return real(self, geom, seg_idx, q_factor=q_factor)

    BSplineSolver._buried_nodes = spy
    try:
        with warnings.catch_warnings():
            _quiet()
            geom = s._build_geometry()
            supp_seg, polys, *_ = s._build_basis_polynomials(geom)
            s._compute_Z_operator_buried(geom, supp_seg, polys)
    finally:
        BSplineSolver._buried_nodes = real
    # two classes at the base order for the two same-medium remainders, two at
    # the raised one for the transmitted pair.
    assert sorted(seen) == [1, 1, 4, 4], seen


# ----------------------------------------------------------------------
# G-1004-6: the two assembly routes at the RAISED order
# ----------------------------------------------------------------------


@pytest.mark.skipif(
    not _bs._HAVE_FIELD_GALERKIN_ACCEL,
    reason="built without the #914 field-Galerkin accelerator",
)
@pytest.mark.parametrize("gap,factor", [(0.005, 4), (0.002, 10)])
def test_the_cpp_and_numpy_assemblies_agree_at_the_raised_order(gap, factor):
    """momwire#914's parity gate, at an order it has never seen.

    q enters `_field_galerkin_block` twice — the reshape AND the chunk width,
    `(1 << 19) // (n_src * q * q)` — so raising it by 10x narrows the chunk by
    100x and puts the 2 mm rung over a chunk boundary that q = 6 never
    reaches on a deck this small. #914's own decks are all factor 1, so
    without this the raised order rides whichever route happens to be built.
    """
    seen = []
    real = BSplineSolver._field_galerkin_block

    def spy(self, *args):
        seen.append((self, args))
        return real(self, *args)

    BSplineSolver._field_galerkin_block = spy
    try:
        with warnings.catch_warnings():
            _quiet()
            s = mk(BSplineSolver, gap)
            geom = s._build_geometry()
            supp_seg, polys, *_ = s._build_basis_polynomials(geom)
            s._compute_Z_operator_buried(geom, supp_seg, polys)
    finally:
        BSplineSolver._field_galerkin_block = real

    raised = [
        (solver, args)
        for solver, args in seen
        if len(args[5]) // len(args[3]) == factor * solver._n_qp_buried_field()
    ]
    assert len(raised) == 2, [len(a[1][5]) // len(a[1][3]) for a in seen]

    keep = (_bs._HAVE_FIELD_GALERKIN_ACCEL, _bs._FIELD_GALERKIN_FUSED)
    for solver, args in raised:
        outs = {}
        for tag, accel, fused in (("np", False, True), ("cpp", True, True)):
            _bs._HAVE_FIELD_GALERKIN_ACCEL, _bs._FIELD_GALERKIN_FUSED = accel, fused
            try:
                outs[tag] = real(solver, *args)
            finally:
                _bs._HAVE_FIELD_GALERKIN_ACCEL, _bs._FIELD_GALERKIN_FUSED = keep
        scale = max(float(np.abs(outs["np"]).max()), 1e-300)
        rel = float(np.abs(outs["cpp"] - outs["np"]).max() / scale)
        assert rel <= 1e-13, (gap, rel)
