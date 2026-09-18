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

from momwire import _crossing_fill as cf
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
