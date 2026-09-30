"""momwire#1224 stage 3, unit C1: `SinusoidalBasisSampler.samples` builds F / Fd
as CSR from the segment structure instead of two dense (n_basis, n_nodes)
complex arrays. The gate is BYTE equality with the dense build converted the
way `axis_data` converted it (`csr_array(dense)`): same shape, dtype, indptr,
indices and data. That includes the exact-zero entries the dense conversion
drops and the ascending column order it yields.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from momwire.sinusoidal import SinusoidalBasisSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_sin_crossing_recipe_1223 import (  # noqa: E402
    DECKS,
    SOILS,
    _solver,
    _two_k_view,
)


def _dense_reference(sampler, seg_runs, u_phys):
    """Verbatim the dense `samples` this unit replaced."""
    n_nodes = u_phys.shape[0]
    F = np.zeros((sampler.n_basis, n_nodes), dtype=np.complex128)
    Fd = np.zeros((sampler.n_basis, n_nodes), dtype=np.complex128)
    seg_rows: dict[int, np.ndarray] = {}
    for g, (s0, cnt) in seg_runs.items():
        sl, rows = sampler._entries(g)
        if rows.size == 0:
            continue
        xi = u_phys[s0 : s0 + cnt] - 0.5 * sampler._h[g]
        f, fd = sampler._value_and_slope(sl, xi)
        F[rows, s0 : s0 + cnt] = f
        Fd[rows, s0 : s0 + cnt] = fd
        seg_rows[int(g)] = np.sort(rows)
    return F, Fd, seg_rows


def _assert_same_csr(new, ref):
    assert sp.issparse(new) and new.format == "csr"
    assert new.shape == ref.shape
    assert new.dtype == ref.dtype == np.complex128
    assert new.has_sorted_indices and new.has_canonical_format
    assert np.array_equal(new.indptr, ref.indptr)
    assert np.array_equal(new.indices, ref.indices)
    # bytes, not allclose: NaN-free here, and -0.0 / +0.0 must agree too
    assert new.data.tobytes() == ref.data.tobytes()


def _runs(geom, q, order, rng):
    n = int(geom["n_segs"])
    segs = order(list(range(n)))
    runs, u, at = {}, [], 0
    for g in segs:
        runs[g] = (at, q)
        u.append(rng.uniform(0.0, geom["seg_h"][g], q))
        at += q
    return runs, np.concatenate(u)


def _views():
    """(id, geom, view, k) over the crossing decks: single-medium and the
    MIXED per-entry-k view every crossing solve uses."""
    for name in sorted(DECKS):
        for soil in SOILS[:3]:
            s = _solver(name, soil)
            geom, _below, med, mixed = _two_k_view(s)
            tag = f"{name}-er{soil[0]:g}"
            yield f"{tag}-mixed", geom, mixed, med.k_p
            if soil is SOILS[0]:
                yield f"{name}-single", geom, s._basis_coefs(geom, s.k), s.k


VIEWS = list(_views())
ORDERS = {
    "ascending": lambda a: a,
    "reversed": lambda a: a[::-1],
    "shuffled": lambda a: list(np.random.default_rng(7).permutation(a)),
}


@pytest.mark.parametrize("order", sorted(ORDERS))
@pytest.mark.parametrize("view", VIEWS, ids=[v[0] for v in VIEWS])
def test_csr_samples_equal_the_dense_build_byte_for_byte(view, order):
    _id, geom, seg_view, k = view
    sampler = SinusoidalBasisSampler(seg_view, k, geom["seg_h"], int(geom["n_segs"]))
    rng = np.random.default_rng(1224)
    runs, u = _runs(geom, 5, ORDERS[order], rng)
    F, Fd, rows = sampler.samples(runs, u)
    F0, Fd0, rows0 = _dense_reference(sampler, runs, u)
    _assert_same_csr(F, sp.csr_array(F0))
    _assert_same_csr(Fd, sp.csr_array(Fd0))
    assert F.nnz > 0
    assert rows.keys() == rows0.keys()
    for g in rows0:
        assert np.array_equal(rows[g], rows0[g])


def test_exact_zero_entries_are_dropped_like_the_dense_conversion():
    """Zeroing a coefficient set makes whole entries sample to exactly 0."""
    _id, geom, seg_view, k = next(v for v in VIEWS if v[0].endswith("-mixed"))
    seg_view = dict(seg_view)
    for key in ("A", "B", "C", "AC"):
        a = np.array(seg_view[key], copy=True)
        a[::2] = 0.0
        seg_view[key] = a
    sampler = SinusoidalBasisSampler(seg_view, k, geom["seg_h"], int(geom["n_segs"]))
    runs, u = _runs(geom, 4, ORDERS["reversed"], np.random.default_rng(3))
    F, Fd, _ = sampler.samples(runs, u)
    F0, Fd0, _ = _dense_reference(sampler, runs, u)
    structural = sum(sampler._entries(g)[1].size * c for g, (_s0, c) in runs.items())
    assert F.nnz < structural, "no entry sampled to exactly zero: test is vacuous"
    _assert_same_csr(F, sp.csr_array(F0))
    _assert_same_csr(Fd, sp.csr_array(Fd0))


def test_a_perturbed_value_fails_the_gate():
    _id, geom, seg_view, k = VIEWS[0]
    sampler = SinusoidalBasisSampler(seg_view, k, geom["seg_h"], int(geom["n_segs"]))
    runs, u = _runs(geom, 5, ORDERS["ascending"], np.random.default_rng(1))
    F, _Fd, _ = sampler.samples(runs, u)
    F0 = _dense_reference(sampler, runs, u)[0]
    F0[np.nonzero(F0)[0][0], np.nonzero(F0)[1][0]] *= 1.0 + 2**-52
    with pytest.raises(AssertionError):
        _assert_same_csr(F, sp.csr_array(F0))


def test_empty_runs_give_empty_csr_of_the_right_shape():
    _id, geom, seg_view, k = VIEWS[0]
    n = int(geom["n_segs"])
    sampler = SinusoidalBasisSampler(seg_view, k, geom["seg_h"], n)
    F, Fd, rows = sampler.samples({}, np.zeros(0))
    F0, Fd0, _ = _dense_reference(sampler, {}, np.zeros(0))
    assert rows == {}
    _assert_same_csr(F, sp.csr_array(F0))
    _assert_same_csr(Fd, sp.csr_array(Fd0))
