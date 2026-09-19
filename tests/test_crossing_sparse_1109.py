"""momwire#1029 phase 2 / #1109 unit A: the axis samples are sparse, and
nothing downstream of them builds a dense `(n_basis, n_nodes)` array.

The floor under the crossing fill's 8.55 GB peak at 150 radials was
`axis_data`'s dense F/Fd — 4.13 GB of float64 carrying at most `degree + 1`
nonzeros per COLUMN, because a B-spline basis has local support. The weights
`_row_weights` gathers out of them were the allocation momwire#1109 named:
124 MiB per call, 292 calls per fill.

What is pinned here is the ALGEBRA, not the bits: a sparse product sums the
same nonzero terms in a different order (the same licence momwire#914 and
#919 took on this fill), so the sub-blocks are compared to the dense gather
entry for entry and the fill's own residuals are P2-6 / P2-7's business.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from momwire import _aca
from momwire import _below_interface as bi
from momwire import _crossing_fill as cf
from momwire import bspline
from momwire.bspline import BSplineSolver

from test_crossing_serve_524 import crossing_deck

pytestmark = pytest.mark.filterwarnings(
    "ignore::momwire._crossing_fill.CoarseCrossingNode"
)


@pytest.fixture(scope="module")
def axes():
    s = BSplineSolver(**crossing_deck(1))
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    ctx = s._crossing_context(geom, supp_seg, polys)
    below = np.asarray(s._below_segments(geom))
    a_idx = np.flatnonzero(~below)
    b_idx = np.flatnonzero(below)
    return ctx, a_idx, b_idx, cf.axis_data(ctx, a_idx), cf.axis_data(ctx, b_idx)


def test_p2a_1_the_axis_carries_csr_samples_at_the_dense_shape(axes):
    ctx, _a, _b, ax_a, ax_b = axes
    for ax in (ax_a, ax_b):
        for key in ("F_csr", "Fd_csr"):
            M = ax[key]
            assert isinstance(M, sp.csr_array), f"{key} is {type(M)}"
            assert M.shape == (ax["n_basis"], ax["nodes"].shape[0])
        # the old keys are gone, so no consumer can read them by accident
        assert "F" not in ax and "Fd" not in ax
        # local support: at most degree + 1 live rows over any one node
        per_node = np.diff(ax["F_csr"].tocsc().indptr)
        assert per_node.max() <= ctx.basis.degree + 1
        # …which bounds the whole block: nnz ≤ (degree + 1)·n_nodes whatever
        # n_basis is, so the samples stop scaling with N × N_nodes and the
        # 4.13 GB at 150 radials becomes ~1 MB.
        assert ax["F_csr"].nnz <= (ctx.basis.degree + 1) * ax["nodes"].shape[0]
        assert ax["Fd_csr"].nnz == ax["F_csr"].nnz


def test_p2a_2_no_dense_n_basis_by_n_nodes_array_is_ever_allocated(axes, monkeypatch):
    """The gate of unit A. `axis_data` is allowed every (n_nodes,)-shaped
    array it likes; what it must never build is the full sample block, which
    is what 4.13 GB of the 150-radial peak was."""
    ctx, a_idx, b_idx, _ax_a, _ax_b = axes
    n_basis = ctx.basis.n_basis
    seen = []
    for name in ("zeros", "empty", "ones"):
        orig = getattr(np, name)

        def spy(shape, *a, _orig=orig, _name=name, **kw):
            out = _orig(shape, *a, **kw)
            if out.ndim == 2 and out.shape[0] == n_basis and out.shape[1] > 1:
                seen.append((_name, out.shape))
            return out

        monkeypatch.setattr(np, name, spy)
    ax = cf.axis_data(ctx, b_idx)
    assert ax["F_csr"].shape[1] > 1
    assert seen == [], f"a dense (n_basis, n) array was built: {seen}"


def test_p2a_3_row_weights_are_the_dense_gather_entry_for_entry(axes):
    _ctx, _a, _b, ax_a, ax_b = axes
    rng = np.random.default_rng(1109)
    for ax in (ax_a, ax_b):
        F = ax["F_csr"].toarray()
        Fd = ax["Fd_csr"].toarray()
        n = ax["nodes"].shape[0]
        for _ in range(6):
            ii = np.sort(rng.choice(n, size=min(n, 11), replace=False))
            rows = cf._support_rows(ax, ii)
            got = cf._row_weights(ax, ii, rows)
            assert all(isinstance(M, sp.csr_array) for M in got)
            tx, ty, tz = ax["t"][ii].T
            w = ax["w"][ii]
            Fw = F[np.ix_(rows, ii)] * w
            ref = (Fw * tx, Fw * ty, Fw * tz, Fd[np.ix_(rows, ii)] * w)
            for M, R in zip(got, ref):
                assert np.array_equal(M.toarray(), R)
            # and the unrestricted spelling is the same gather over every row
            for M, R in zip(cf._row_weights(ax, ii), (F[:, ii] * w * tx,)):
                assert np.array_equal(M.toarray(), R)
                break


def test_p2a_4_fdw_sparse_is_fd_times_w(axes):
    _ctx, _a, _b, ax_a, ax_b = axes
    for ax in (ax_a, ax_b):
        ax.pop("_fdw_csr", None)
        got = cf._fdw_sparse(ax)
        assert np.array_equal(got.toarray(), ax["Fd_csr"].toarray() * ax["w"])
        assert cf._fdw_sparse(ax) is got  # still cached on the axis


def test_p2a_5_support_rows_pattern_fallback_covers_the_scan(axes):
    """An axis with no `seg_rows` (a path-test axis is the real one) falls
    back to the samples themselves. The CSR reading is the stored PATTERN,
    which is a superset of the `!= 0` scan and still an exact restriction."""
    _ctx, _a, _b, _ax_a, ax_b = axes
    stripped = {k: v for k, v in ax_b.items() if k != "seg_rows"}
    F = ax_b["F_csr"].toarray()
    Fd = ax_b["Fd_csr"].toarray()
    rng = np.random.default_rng(912)
    n = ax_b["nodes"].shape[0]
    for _ in range(8):
        ii = np.sort(rng.choice(n, size=min(n, 13), replace=False))
        got = cf._support_rows(stripped, ii)
        scan = np.flatnonzero(
            np.any(F[:, ii] != 0, axis=1) | np.any(Fd[:, ii] != 0, axis=1)
        )
        assert set(scan) <= set(got)
        assert set(got) <= set(cf._support_rows(ax_b, ii))


# ---------------------------------------------------------------------------
# Unit B — `rows=` reaches the crossing family
#
# Phase 1 filled the crossing block whole under a `rows=` restriction and
# registered that as the contract (PLAN-phase1.md Amendment 1), because the
# routing reads its transpose. Phase 2 asks for BOTH slices out of one
# evaluation, so the contract becomes "rows outside the request are exactly
# zero" and the crossing family stops being the one term of the route's cost
# that scales with N.
# ---------------------------------------------------------------------------


def _whole_wire_segments(solver, geom, wires):
    off = geom["seg_offsets"]
    per = geom["per_wire"]
    return np.sort(
        np.concatenate([np.arange(off[w], off[w] + per[w]["n_total"]) for w in wires])
    )


@pytest.fixture(scope="module")
def routed():
    """The crossing deck, its basis, and a whole-wire observer restriction."""
    s = BSplineSolver(**crossing_deck(1))
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    seg_rows = _whole_wire_segments(s, geom, [0])  # the BELOW wire
    return s, geom, supp_seg, polys, seg_rows


@pytest.mark.parametrize("aca_guard", [None, 0.0], ids=["shipped", "aca-forced"])
def test_p2b_1_the_pair_is_the_full_blocks_two_slices(
    axes, routed, monkeypatch, aca_guard
):
    ctx, a_idx, b_idx, ax_a, ax_b = axes
    _s, _geom, supp_seg, polys, seg_rows = routed
    if aca_guard is not None:
        # the low-rank branch as well as the direct one: both halves read the
        # same (Uf, Vf) there, and the slices must still be exact
        monkeypatch.setattr(cf, "_ACA_COST_GUARD", aca_guard)
    R = bi._crossing_basis_rows(supp_seg, polys, seg_rows)
    assert R.size and R.size < supp_seg.shape[0]
    full = cf.cross_complete_block_split(ctx, a_idx, b_idx, ax_a, ax_b)
    t_r, t_c = cf.cross_complete_block_split(ctx, a_idx, b_idx, ax_a, ax_b, rows=R)
    assert t_r.shape == (R.size, full.shape[1])
    assert t_c.shape == (full.shape[0], R.size)
    # The same products, gathered rather than recomputed: bit for bit.
    assert np.array_equal(t_r, full[R, :])
    assert np.array_equal(t_c, full[:, R])


@pytest.mark.parametrize("aca_guard", [None, 0.0], ids=["shipped", "aca-forced"])
def test_p2b_2_the_restriction_costs_no_extra_kernel_work(
    axes, routed, monkeypatch, aca_guard
):
    """One evaluation of the kernel tables and one ACA factorisation per
    block serve BOTH halves — the property that makes the transpose free, and
    the reason the crossing family stops scaling with N on the route.

    Driven at the shipped `_ACA_COST_GUARD` (every far block direct on this
    small deck) and with the guard forced to 0 so the low-rank branch runs;
    the guard is a TEST monkeypatch, never a moved constant."""
    ctx, a_idx, b_idx, ax_a, ax_b = axes
    _s, _geom, supp_seg, polys, seg_rows = routed
    R = bi._crossing_basis_rows(supp_seg, polys, seg_rows)
    if aca_guard is not None:
        monkeypatch.setattr(cf, "_ACA_COST_GUARD", aca_guard)
    tally = {"tables": 0, "aca": 0}
    for mod, name, key in ((cf, "_tables", "tables"), (_aca, "aca_partial", "aca")):
        orig = getattr(mod, name)

        def spy(*a, _orig=orig, _key=key, **kw):
            tally[_key] += 1
            return _orig(*a, **kw)

        monkeypatch.setattr(mod, name, spy)

    cf.cross_complete_block_split(ctx, a_idx, b_idx, ax_a, ax_b)
    full = dict(tally)
    tally.update(tables=0, aca=0)
    cf.cross_complete_block_split(ctx, a_idx, b_idx, ax_a, ax_b, rows=R)
    assert full["tables"] > 0
    if aca_guard == 0.0:
        assert full["aca"] > 0, "the ACA branch did not run"
    assert tally == full, (tally, full)


def test_p2b_3_self_completions_answer_their_requested_rows(axes, routed):
    ctx, _a, _b, ax_a, ax_b = axes
    _s, _geom, supp_seg, polys, seg_rows = routed
    R = bi._crossing_basis_rows(supp_seg, polys, seg_rows)
    full = cf.self_completions(ctx, ax_b, ax_a)
    got = cf.self_completions(ctx, ax_b, ax_a, rows=R)
    assert got.shape == (R.size, full.shape[1])
    assert np.array_equal(got, full[R, :])


def test_p2b_4_rows_outside_the_request_are_exactly_zero(routed):
    """P2-8's item 2, at the routing. Phase 1 left the crossing block's rows
    non-zero here and registered it; nothing is exempt now."""
    s, geom, supp_seg, polys, seg_rows = routed
    Z_full = s._compute_Z_operator_buried(geom, supp_seg, polys)
    Z_rows = s._compute_Z_operator_buried(geom, supp_seg, polys, rows=seg_rows)
    R = bi._crossing_basis_rows(supp_seg, polys, seg_rows)
    other = np.setdiff1d(np.arange(supp_seg.shape[0]), R)
    assert np.all(Z_rows[other] == 0.0)
    rel = np.max(np.abs(Z_rows[R] - Z_full[R])) / np.max(np.abs(Z_full[R]))
    assert rel <= 1e-12, rel
    # and it is not vacuous: the restriction really left work out
    assert np.max(np.abs(Z_full[other])) > 0.0


def test_p2b_5_a_split_basis_is_refused_by_name(routed):
    _s, _geom, supp_seg, polys, _seg_rows = routed
    live = np.any(polys != 0.0, axis=2)
    # a segment set that covers one live wing of some basis and not the rest
    m = int(np.flatnonzero(live.sum(axis=1) > 1)[0])
    half = np.array([int(supp_seg[m, np.flatnonzero(live[m])[0]])], dtype=np.int64)
    with pytest.raises(ValueError) as exc:
        bi._crossing_basis_rows(supp_seg, polys, half)
    msg = str(exc.value)
    assert "live support segments" in msg and "whole wires" in msg


def test_p2b_6_two_radius_answers_the_half_of_each_block_the_routing_reads():
    a = 0.25e-3
    s = BSplineSolver(**crossing_deck(2, wire_radius=[a / 2, a]))
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    ctx = s._crossing_context(geom, supp_seg, polys)
    assert ctx.a_below is not None
    below = np.asarray(s._below_segments(geom))
    a_idx, b_idx = np.flatnonzero(~below), np.flatnonzero(below)
    ax_a = cf.axis_data(ctx, a_idx)
    ax_b = cf.axis_data(ctx, b_idx)
    seg_rows = _whole_wire_segments(s, geom, [0])
    R = bi._crossing_basis_rows(supp_seg, polys, seg_rows)
    f_above, f_below = cf.cross_complete_blocks_two_radius(
        ctx, a_idx, b_idx, ax_a, ax_b
    )
    r_above, r_below = cf.cross_complete_blocks_two_radius(
        ctx, a_idx, b_idx, ax_a, ax_b, rows=R
    )
    assert np.array_equal(r_above, f_above[R, :])
    assert np.array_equal(r_below, f_below[:, R])
    fs = cf.self_completions_two_radius(ctx, ax_b, ax_a)
    rs = cf.self_completions_two_radius(ctx, ax_b, ax_a, rows=R)
    assert np.array_equal(rs, fs[R, :])


# ---------------------------------------------------------------------------
# Unit C — the full-size transients go
#
# The crossing family used to build three (n, n) blocks beyond Z and the main
# sandwich: `_ends_and_corner`'s own block (1.06 GB at 150 radials),
# `self_completions`' `total` (1.99 GB with its temporaries), and
# `_field_galerkin_block`'s `Q` (1.05 GB). The first two are the ends' and the
# node's shapes, each confined to a handful of basis rows and columns, so they
# scatter into the caller's block instead — bit-identically, which is what
# these tests pin. The third is no longer blocked at all: momwire#1115 part 3
# takes it, under a tolerance gate rather than a bit one; see the two below.
# ---------------------------------------------------------------------------


def _ends_args(ctx, A, B):
    from momwire._sommerfeld_transmitted import _c1_moment

    eps_t, _eps_m, k_p, _k_m, _c2, _a_m = ctx.medium
    return (ctx, A, B, eps_t, k_p, _c1_moment(ctx.omega, ctx.mu), float(ctx.ground_z))


def test_p2c_1_ends_accumulate_in_place_bit_identically(axes):
    ctx, _a, _b, ax_a, ax_b = axes
    args = _ends_args(ctx, ax_a, ax_b)
    n = ax_a["n_basis"]
    rng = np.random.default_rng(1109)
    base = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) * 1e3
    returned = base + cf._ends_and_corner(*args, memo={})
    in_place = base.copy()
    cf._ends_and_corner(*args, memo={}, out=in_place)
    assert np.array_equal(in_place, returned)


def test_p2c_2_ends_build_no_full_size_block_when_given_one(axes, monkeypatch):
    ctx, _a, _b, ax_a, ax_b = axes
    args = _ends_args(ctx, ax_a, ax_b)
    n = ax_a["n_basis"]
    dest = np.zeros((n, n), dtype=np.complex128)
    seen = []
    orig = np.zeros

    def spy(shape, *a, **kw):
        out = orig(shape, *a, **kw)
        if out.ndim == 2 and out.shape == (n, n):
            seen.append(out.shape)
        return out

    monkeypatch.setattr(np, "zeros", spy)
    cf._ends_and_corner(*args, memo={}, out=dest)
    assert seen == [], seen
    assert np.count_nonzero(dest), "the ends wrote nothing"


def test_p2c_3_self_completions_accumulate_into_z_bit_identically(axes):
    ctx, _a, _b, ax_a, ax_b = axes
    n = ax_b["n_basis"]
    rng = np.random.default_rng(914)
    base = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) * 1e3
    returned = base + cf.self_completions(ctx, ax_b, ax_a)
    # Fortran order on purpose: the buried Z is column-major (momwire#136), so
    # the accumulation has to be correct on a non-C-contiguous destination.
    in_place = np.asfortranarray(base)
    cf.self_completions(ctx, ax_b, ax_a, out=in_place)
    assert np.array_equal(in_place, returned)


def test_p2c_4_self_completions_build_no_full_size_block(axes, monkeypatch):
    ctx, _a, _b, ax_a, ax_b = axes
    n = ax_b["n_basis"]
    dest = np.zeros((n, n), dtype=np.complex128)
    seen = []
    orig = np.zeros

    def spy(shape, *a, **kw):
        out = orig(shape, *a, **kw)
        if out.ndim == 2 and out.shape == (n, n):
            seen.append(out.shape)
        return out

    monkeypatch.setattr(np, "zeros", spy)
    cf.self_completions(ctx, ax_b, ax_a, out=dest)
    assert seen == [], seen
    assert np.count_nonzero(dest), "the completions wrote nothing"


def test_p2c_5_the_field_galerkin_accelerator_takes_a_column_major_target():
    """Unit C's blocker, now gone, and pinned so it cannot come back.

    Two things used to stop `_field_galerkin_block` taking an `out=Z`. The
    first was that `_acc.assemble_field_galerkin` declared `Q` as
    `py::array_t<..., c_style>` without `forcecast`: pybind11 answered a
    column-major array by handing the C++ a C-contiguous COPY, so the call
    returned cleanly and every accumulation was lost. momwire#1115 part 1 made
    that a REFUSAL, and part 3 made it work -- the two scatters address the
    target through its own strides. The buried Z is column-major by
    momwire#136 (`scipy.linalg.solve(overwrite_a=True)` factors in place only
    on one), so that is exactly the shape handed over.

    The second reason still stands and is NOT a blocker, only a change of
    gate: the observer loop chunks, and a basis row whose support straddles a
    chunk boundary accumulates across chunks, so `Z - (c1 + c2)` becomes
    `(Z - c1) - c2`. At 150 radials `chunk` is 1 and every row straddles. The
    lever is therefore a reassociation, and lands under the 1e-12 gate
    `p2_default_path.py` carries rather than the bit gate unit C was
    registered under.

    What this pins is the equality that makes the lever safe to take at all:
    the same numbers whichever way the target is laid out. If a future edit
    reintroduces a copy, the column-major result stops matching and this
    fails -- which the old silent-copy binding could not be made to do.
    """
    acc = pytest.importorskip("momwire._accelerators")
    if not bspline._HAVE_FIELD_GALERKIN_ACCEL:
        pytest.skip("no field-galerkin accelerator in this build")
    if not getattr(acc, "field_galerkin_strided_1115", False):
        pytest.skip("built before momwire#1115 part 3's strided target")
    from test_field_galerkin_914 import _good_args

    args = _good_args()
    n = args["Q"].shape[0]
    args["Q"] = np.zeros((n, n), dtype=np.complex128)
    acc.assemble_field_galerkin(**args)
    assert np.count_nonzero(args["Q"]), "the C-contiguous call wrote nothing"

    args_f = _good_args()
    args_f["Q"] = np.zeros((n, n), dtype=np.complex128, order="F")
    acc.assemble_field_galerkin(**args_f)
    assert np.count_nonzero(args_f["Q"]), (
        "the column-major call wrote nothing -- the copy is back"
    )
    assert np.array_equal(args_f["Q"], args["Q"]), (
        "the column-major target got different numbers from the C-contiguous one"
    )


def test_p2c_6_the_buried_z_is_column_major(routed):
    """The other half of the tripwire: the matrix that would BE that target."""
    s, geom, supp_seg, polys, _rows = routed
    Z = s._compute_Z_operator_buried(geom, supp_seg, polys)
    assert s._buried_chunked_serves
    assert Z.flags["F_CONTIGUOUS"] and not Z.flags["C_CONTIGUOUS"]
