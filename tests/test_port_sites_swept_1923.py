"""`BSplineSolver.port_sites_swept` -- `port_sites` over a sweep (#1923).

The oracle is the single-frequency `port_sites` at each k, on the same solver
with the frequency triple rebound by hand (`_per_k`, spelled out rather than
through the solver's own `_set_k`, as `test_port_solution.py` does). Two
tiers, by the route the sweep takes:

* the per-k loop routes (singular enrichment; the finite-ground fallback with
  its hoisted same-edge moments) run `_port_solution_dense` per k on the same
  fill the single-frequency call makes, so the answer is bit for bit;
* the fully batched route assembles every k of a chunk in one C++ call and
  factors the stack with numpy's gesv where the per-k path uses
  `lu_factor` / `lu_solve`, so it agrees to rounding only. The bound is
  `test_port_solution.py`'s `SWEPT_REASSOC_RTOL` (1e-12), the one the swept Y
  already holds on that route. Measured (laptop, avx2): the junction deck's
  worst is 3.5e-14 on `dy_dx` and about 2e-15 elsewhere; the free deck came
  out bit for bit there, which nothing promises.

`port_sites` itself is gated against fresh solves in
`test_port_sites_1915.py`; this file only has to show the sweep is it.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import BSplineSolver, PortSites, _schedule
from momwire.bspline import _HAVE_BSPLINE_SWEPT_ASSEMBLE_ACCEL

from test_port_sites_1915 import (
    _bent,
    _dipole,
    _feed,
    _over_ground,
    _positions,
    _tee,
)

SWEPT_REASSOC_RTOL = 1e-12

# (deck, site wire, route the sweep takes)
CASES = [
    pytest.param(_dipole, 0, "batched", id="free"),
    pytest.param(_bent, 1, "batched", id="junction"),
    pytest.param(_over_ground, 0, "per-k", id="sommerfeld"),
    pytest.param(_tee, 1, "enrichment", id="enrichment-raw"),
]


def _per_k(solver, ks, fn):
    saved = (solver.k, solver.wavelength, solver.omega)
    out = []
    try:
        for kk in ks:
            solver.k = float(kk)
            solver.omega = solver.k * solver.c
            solver.wavelength = solver.c / (solver.omega / (2 * np.pi))
            out.append(fn())
    finally:
        solver.k, solver.wavelength, solver.omega = saved
    return out


def _ks(solver):
    return solver.k * np.array([0.91, 0.97, 1.0, 1.06, 1.13])


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _readouts(ps, xs):
    return {
        "site_y": ps.site_y,
        "y": ps.y(xs),
        "dy_dx": ps.dy_dx(xs),
        "coeffs": ps.coeffs(xs),
    }


def _route(solver):
    if solver.use_singular_enrichment:
        return "enrichment"
    return "batched" if solver._swept_batched_available() else "per-k"


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("deck_fn, wire, route", CASES)
def test_each_k_is_the_single_frequency_port_sites(deck_fn, wire, route, degree):
    deck = deck_fn()
    solver = BSplineSolver(degree=degree, feeds=_feed(deck), **deck)
    if route == "batched" and not _HAVE_BSPLINE_SWEPT_ASSEMBLE_ACCEL:
        pytest.skip("the batched swept route needs the C++ accelerator")
    assert _route(solver) == route  # the tier below is the route's
    ks = _ks(solver)
    k_before = solver.k
    got = solver.port_sites_swept(wire, ks)
    assert solver.k == k_before
    want = _per_k(solver, ks, lambda: solver.port_sites(wire))
    assert len(got) == len(ks)
    xs = _positions(solver, wire)[:3]
    for g, w in zip(got, want, strict=True):
        assert isinstance(g, PortSites)
        assert (g.n_ports, g.n_sites, g.span) == (w.n_ports, w.n_sites, w.span)
        rg, rw = _readouts(g, xs), _readouts(w, xs)
        for name in rw:
            assert rg[name].shape == rw[name].shape, name
            if route == "batched":
                assert _rel(rg[name], rw[name]) <= SWEPT_REASSOC_RTOL, name
            else:
                assert rg[name].tobytes() == rw[name].tobytes(), name


def test_span_limits_the_swept_sites_too():
    deck = _dipole()
    solver = BSplineSolver(degree=2, feeds=_feed(deck), **deck)
    ks = _ks(solver)[:2]
    part = solver.port_sites_swept(0, ks, span=(1.0, 3.0))
    whole = solver.port_sites_swept(0, ks)
    for p, w in zip(part, whole, strict=True):
        assert p.span == (1.0, 3.0)
        assert p.n_sites < w.n_sites
        assert _rel(p.y([1.0, 2.2, 3.0]), w.y([1.0, 2.2, 3.0])) <= 1e-10


def test_the_chunked_batched_sweep_is_the_one_chunk_sweep(monkeypatch):
    """One k per chunk gives each k the answer the whole-sweep chunk gives,
    to rounding: chunking only re-batches."""
    if not _HAVE_BSPLINE_SWEPT_ASSEMBLE_ACCEL:
        pytest.skip("the batched swept route needs the C++ accelerator")
    deck = _bent()
    solver = BSplineSolver(degree=2, feeds=_feed(deck), **deck)
    ks = _ks(solver)
    one = solver.port_sites_swept(1, ks)
    seen = []

    def one_k(*args, **kw):
        seen.append(1)
        return 1

    monkeypatch.setattr(_schedule, "k_chunk", one_k)
    tiny = solver.port_sites_swept(1, ks)
    assert seen  # the chunk size really was asked for, and cut to one
    for a, b in zip(one, tiny, strict=True):
        assert _rel(a.site_y, b.site_y) <= SWEPT_REASSOC_RTOL


def test_the_site_columns_count_against_the_sweep_budget():
    """`_swept_batched_z_chunks` sizes its k-chunks with the site columns'
    per-k bytes added: the default budget holds this small sweep in one
    chunk alone, and a million site columns (about 1.4 GB per k here) cut
    it to one k per chunk."""
    if not _HAVE_BSPLINE_SWEPT_ASSEMBLE_ACCEL:
        pytest.skip("the batched swept route needs the C++ accelerator")
    deck = _dipole()
    solver = BSplineSolver(degree=2, feeds=_feed(deck), **deck)
    geom = solver._build_geometry()
    supp, polys, _kcl, _wk, _wbg = solver._build_basis_polynomials(geom)
    ks = _ks(solver)

    def n_chunks(n_extra):
        chunks = solver._swept_batched_z_chunks(
            ks, geom, supp, polys, n_extra_columns=n_extra
        )
        return sum(1 for _ in chunks)

    assert n_chunks(0) == 1
    assert n_chunks(10**6) == len(ks)


@pytest.mark.parametrize(
    "kw, match",
    [
        (dict(feed_model="segment"), "port_sites_swept: supported only"),
        (dict(feed_smoothing_factor=0.5), "port_sites_swept: supported only"),
        (
            dict(use_singular_enrichment=True, enrichment_variant="auto"),
            "port_sites_swept: enrichment_variant='auto'",
        ),
    ],
)
def test_refusals_are_port_sites_own(kw, match):
    deck = _dipole(**kw)
    s = BSplineSolver(degree=2, feeds=_feed(deck), **deck)
    with pytest.raises(ValueError, match=match):
        s.port_sites_swept(0, [s.k])
    # Refused before any fill, so an empty sweep refuses too.
    with pytest.raises(ValueError, match=match):
        s.port_sites_swept(0, [])


def test_wire_span_and_complex_k_are_refused():
    deck = _dipole()
    s = BSplineSolver(degree=2, feeds=_feed(deck), **deck)
    with pytest.raises(ValueError, match="port_sites_swept: wire 3 out of range"):
        s.port_sites_swept(3, [s.k])
    with pytest.raises(ValueError, match="port_sites_swept: span"):
        s.port_sites_swept(0, [s.k], span=(3.0, 1.0))
    with pytest.raises((ValueError, TypeError)):
        s.port_sites_swept(0, [s.k * (1 + 0.01j)])
