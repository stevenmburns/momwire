"""The exact-kernel correction's C++ route (momwire#1410).

`_accel_exact_kernel.cpp` ports the ring kernel, the pair moments and the
row windows of `_exact_kernel.CoaxialRows`; the dedup hashes instead of
sorting, and keys a pair on its least image under the coaxial pair's exact
symmetries. It is NOT bit-exact with the numpy route, by decision (Steve,
10-10), so the two are held to DERIVED tolerances:

* the kernel's own accuracy against an independent reference is 1e-11
  (`test_exact_kernel_1408.test_kernel_matches_reference`); C++ vs numpy
  measured 3.2e-16 elementwise on Haswell (avx2 and sse2 alike). The gate,
  1e-14, is ~30x the measurement and three orders below the kernel's own
  error: it cannot resolve a rounding order, and catches any arithmetic slip;
* pair moments, numpy's key (`symmetric=False`): measured 5e-17 of the
  block's max; gated at 1e-14;
* the symmetric key reverses a segment's polynomial index through a binomial
  map, sum_r C(p, r) h^{p-r} (-u)^r, whose terms cancel down to the result:
  for degree 3 the absolute row sum is up to 2^3 = 8 per side, 64 on a
  pair. Measured 9.4e-16; gated at 1e-13 (~64 x a few eps, x2 headroom);
* Z, max|dZ| / max|Z|: measured <= 2.6e-15 over #1409's decks, the dipole
  ladder (d/lambda 0.0085, n = 9 ... 601) and a 3,000-segment fat wire;
  gated at 1e-13, ten times inside the 1e-12 Steve set as the bar.

And to the bit where the C++ route owes the bits to itself: run to run, at
any thread count, at any window height (each unique moment comes from the
same representative pair whatever the windows), and across the avx2 and
sse2 builds on one machine.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pytest

from momwire import BSplineSolver
from momwire import _exact_kernel as EK
from momwire import bspline as _bspline
from momwire._accel import acc

pytestmark = pytest.mark.skipif(
    not EK._HAVE_ACCEL, reason="the C++ accelerator is not built"
)

TOL_KERNEL = 1e-14
TOL_PAIR = 1e-14
TOL_ROWS_SYM = 1e-13
TOL_Z = 1e-13

K0 = 2 * np.pi
A = 0.00425
L = 0.47


def _rel(got, want):
    got, want = np.asarray(got), np.asarray(want)
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


# --------------------------------------------------------------------------
# the kernel and the pair moments
# --------------------------------------------------------------------------
@pytest.mark.parametrize("d_over_lam", [0.001, 0.0085, 0.05])
def test_kernel_matches_numpy(d_over_lam):
    a = d_over_lam / 2
    t = np.concatenate([np.geomspace(1e-4, 1e3, 301), [5.999999, 6.0, 6.000001]])
    got = acc.exact_kernel_ring(t * a, a, K0)
    want = EK.ring_kernel(t * a, a, K0)
    assert np.max(np.abs(got - want) / np.abs(want)) <= TOL_KERNEL
    ts = np.geomspace(1e-6, 5.99, 120) * a
    for got, want in zip(
        acc.exact_kernel_ring_split(ts, a, K0), EK.ring_kernel_split(ts, a, K0)
    ):
        assert np.max(np.abs(got - want) / np.abs(want)) <= TOL_KERNEL
    with pytest.raises(RuntimeError):
        acc.exact_kernel_ring_split(np.array([6.0 * a]), a, K0)


_PAIRS = [  # #1409's pair-shape set: (h_i, h_j, c, sigma_i, sigma_j)
    (6 * A, 6 * A, 0.0, 1, 1),
    (0.7 * A, 0.7 * A, 0.0, 1, 1),
    (3 * A, 1.3 * A, 3 * A, 1, 1),
    (3 * A, 2 * A, 1 * A, 1, 1),
    (4 * A, 4 * A, 4.5 * A, 1, 1),
    (4 * A, 4 * A, 8.5 * A, 1, -1),
    (4 * A, 3 * A, -0.5 * A, -1, 1),
    (2 * A, 2 * A, 30 * A, 1, 1),
    (100 * A, 100 * A, 100 * A, 1, 1),
]


@pytest.mark.parametrize("nd", [3, 4])
@pytest.mark.parametrize("pair", _PAIRS)
def test_pair_moments_match_numpy(pair, nd):
    rules = EK._accel_rules()
    want = EK.pair_moments(*pair, A, K0, nd)
    got = acc.exact_kernel_pair_moments(*pair, A, K0, nd, rules, 0)
    assert _rel(got, want) <= TOL_PAIR


def test_far_rule_matches_numpy():
    rules = EK._accel_rules()
    hi, hj, c, si, sj = _PAIRS[7]
    want = EK.pair_moments_far([hi], [hj], [c], [si], [sj], A, K0, 3)[:, :, 0]
    got = acc.exact_kernel_pair_moments(hi, hj, c, si, sj, A, K0, 3, rules, 1)
    assert _rel(got, want) <= TOL_PAIR


# --------------------------------------------------------------------------
# the row windows
# --------------------------------------------------------------------------
def _group(n, *, flip=(), lengths=None):
    h = np.full(n, L / n) if lengths is None else np.asarray(lengths, float)
    x0 = np.concatenate([[0.0], np.cumsum(h)[:-1]])
    sgn = np.ones(n)
    for i in flip:  # a reversed segment starts at its far end
        sgn[i] = -1.0
        x0[i] += h[i]
    return x0, sgn, h


_SYM_GRADED = np.r_[np.geomspace(0.03, 0.004, 20), np.geomspace(0.004, 0.03, 20)]
_GROUPS = {
    "uniform": _group(41),
    "two runs + reversed": _group(30, flip=(3, 17), lengths=[0.01] * 12 + [0.017] * 18),
    "graded": _group(16, lengths=np.geomspace(0.004, 0.03, 16)),
    "graded, symmetric about the centre": _group(40, lengths=_SYM_GRADED),
}


def _windows(cr, n, rows):
    return np.concatenate(
        [cr.rows(r0, min(n, r0 + rows)) for r0 in range(0, n, rows)], axis=2
    )


@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("name", sorted(_GROUPS))
def test_rows_match_the_numpy_block(name, symmetric):
    x0, sgn, h = _GROUPS[name]
    n = x0.size
    want = EK.coaxial_block(x0, sgn, h, A, K0, 3)
    cr = EK.CoaxialRowsAccel(x0, sgn, h, A, K0, 3, symmetric=symmetric)
    got = _windows(cr, n, 7)
    assert _rel(got, want) <= (TOL_ROWS_SYM if symmetric else TOL_PAIR)
    ref = EK.CoaxialRows(x0, sgn, h, A, K0, 3, max_cached=0)
    ref.rows(0, n)
    if symmetric:
        assert cr.evaluated < ref.evaluated
    else:
        # numpy's own key: the same unique count, pair for pair
        assert cr.evaluated == ref.evaluated


def test_symmetric_key_halves_a_uniform_run():
    """2n - 1 offsets on numpy's key; the transpose folds c onto -c."""
    n = 400
    x0, sgn, h = _group(n)
    cr = EK.CoaxialRowsAccel(x0, sgn, h, A, K0, 3)
    for r0 in range(0, n, 9):
        cr.rows(r0, min(n, r0 + 9))
    assert cr.evaluated == n


def test_every_transform_branch_runs():
    """The groups above reach every branch of the key's transform: the
    transpose, a reversal on either side, and the mirror (a reversal the
    segment's own direction does not explain). Counted on the pairs, so a
    gate that only ever met the identity cannot pass as covering them."""
    seen = set()
    for x0, sgn, h in _GROUPS.values():
        q = 1e-11 * float(np.max(h))
        for i in range(x0.size):
            for j in range(x0.size):
                T, Li, Lj, _key = acc.exact_kernel_canon(
                    h[i], h[j], x0[i], x0[j], sgn[i], sgn[j], q, True
                )
                M = Li != (sgn[i] < 0)
                seen.add(("T", T))
                seen.add(("M", M))
                seen.add(("Li", Li))
                seen.add(("Lj", Lj))
    assert seen == {(f, v) for f in ("T", "M", "Li", "Lj") for v in (False, True)}


@pytest.mark.parametrize("rows", [1, 4, 7, 1000])
@pytest.mark.parametrize("name", sorted(_GROUPS))
def test_windows_are_the_whole_block_bit_for_bit(name, rows):
    x0, sgn, h = _GROUPS[name]
    n = x0.size
    whole = EK.CoaxialRowsAccel(x0, sgn, h, A, K0, 3).rows(0, n)
    cr = EK.CoaxialRowsAccel(x0, sgn, h, A, K0, 3)
    assert np.array_equal(_windows(cr, n, rows), whole)
    one = EK.CoaxialRowsAccel(x0, sgn, h, A, K0, 3)
    one.rows(0, n)
    assert cr.evaluated == one.evaluated


@pytest.mark.parametrize("name", sorted(_GROUPS))
def test_bit_stable_run_to_run_and_across_thread_counts(name):
    x0, sgn, h = _GROUPS[name]
    n = x0.size
    runs = [
        EK.CoaxialRowsAccel(x0, sgn, h, A, K0, 3, n_threads=t).rows(0, n)
        for t in (0, 0, 1, 3)
    ]
    for r in runs[1:]:
        assert np.array_equal(r, runs[0])


def test_capped_cache_agrees():
    x0, sgn, h = _GROUPS["two runs + reversed"]
    n = x0.size
    want = EK.coaxial_block(x0, sgn, h, A, K0, 3)
    cr = EK.CoaxialRowsAccel(x0, sgn, h, A, K0, 3, max_cached=5)
    got = _windows(cr, n, 3)
    assert cr._impl.cached == 5
    assert _rel(got, want) <= TOL_ROWS_SYM
    free = EK.CoaxialRowsAccel(x0, sgn, h, A, K0, 3)
    free.rows(0, n)
    assert cr.evaluated > free.evaluated  # the cap made it re-evaluate


def test_rows_refuse_a_buffer_they_would_not_write():
    x0, sgn, h = _GROUPS["uniform"]
    cr = EK.CoaxialRowsAccel(x0, sgn, h, A, K0, 3)
    n = x0.size
    for bad in (
        np.empty((3, 3, 2, n), dtype=np.complex64),
        np.empty((3, 3, n, 2), dtype=np.complex128).transpose(0, 1, 3, 2),
        np.empty((3, 3, 3, n), dtype=np.complex128),
    ):
        with pytest.raises(RuntimeError):
            cr._impl.rows(0, 2, bad)


def test_window_buffers_are_visible_to_tracemalloc():
    """The memgate lane measures the correction with tracemalloc, which does
    not see a std::vector unless it is told. The per-pair slot table alone
    is 8 bytes a pair; a window of the route must trace at least that, or
    the memory gate is blind to the C++ route."""
    x0, sgn, h = _group(600)
    cr = EK.CoaxialRowsAccel(x0, sgn, h, A, K0, 3)
    out = np.empty((3, 3, 200, 600), dtype=np.complex128)
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        cr._impl.rows(0, 200, out)
        peak = tracemalloc.get_traced_memory()[1] - before
    finally:
        tracemalloc.stop()
    assert peak >= 200 * 600 * 8


# --------------------------------------------------------------------------
# the solver
# --------------------------------------------------------------------------
DIPOLE = [np.array([(0.0, 0.0, -L / 2), (0.0, 0.0, L / 2)])]


def _halves(w):
    return [
        np.array([(0.0, 0.0, -L / 2), (0.0, 0.0, -w / 2)]),
        np.array([(0.0, 0.0, w / 2), (0.0, 0.0, L / 2)]),
    ]


_DECKS = {
    "dipole": dict(n=[[37]]),
    "dipole-segment-gap": dict(n=[[37]], feed_model="segment"),
    "end-port pair": dict(
        wires=_halves(L / 75),
        n=[[37], [37]],
        feeds=[(0, (L - L / 75) / 4, 0j)],
        junctions=[[(0, "end")], [(1, "start")]],
        junction_ports=[(0, 0j), (1, 0j)],
    ),
    "collinear junction + bend": dict(
        wires=[
            np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.2)]),
            np.array([(0.0, 0.0, 0.2), (0.0, 0.0, 0.3), (0.1, 0.0, 0.4)]),
        ],
        n=[[19], [9, 13]],
        junctions=[[(0, "end"), (1, "start")]],
    ),
    "bent wire": dict(
        wires=[np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.25), (0.15, 0.0, 0.40)])],
        n=[[23, 17]],
    ),
    "degree 3": dict(n=[[29]], degree=3),
}


def _solver(wires=DIPOLE, n=None, a=A, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BSplineSolver(
            wires=wires,
            n_per_edge_per_wire=n,
            wavelength=1.0,
            wire_radius=a,
            exact_kernel=True,
            **kw,
        )


def _Z(s):
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.array(s._compute_Z_operator(geom, supp_seg, polys))


def _routes(monkeypatch, s, rows=None):
    """Z through the numpy route and through the C++ route, with a count of
    which row source each one actually walked."""
    ran = {"numpy": 0, "accel": 0}
    for cls, key in ((EK.CoaxialRows, "numpy"), (EK.CoaxialRowsAccel, "accel")):
        orig = cls.rows

        def spy(self, r0, r1, orig=orig, key=key):
            ran[key] += 1
            return orig(self, r0, r1)

        monkeypatch.setattr(cls, "rows", spy)
    if rows is not None:
        monkeypatch.setattr(_bspline, "_EXACT_KERNEL_ROWS_OVERRIDE", rows)
    monkeypatch.setattr(EK, "_USE_ACCEL", False)
    Z0 = _Z(s)
    assert ran["numpy"] > 0 and ran["accel"] == 0, ran
    monkeypatch.setattr(EK, "_USE_ACCEL", True)
    Z1 = _Z(s)
    assert ran["accel"] > 0, "the C++ route did not run"
    Z2 = _Z(s)
    assert np.array_equal(Z1, Z2), "the C++ route is not bit-stable"
    return _rel(Z1, Z0)


@pytest.mark.parametrize("rows", [None, 1, 5])
@pytest.mark.parametrize("ek", [False, True])
@pytest.mark.parametrize("deck", sorted(_DECKS))
def test_solver_Z_matches_the_numpy_route(monkeypatch, deck, ek, rows):
    s = _solver(extended_kernel=ek, **_DECKS[deck])
    assert _routes(monkeypatch, s, rows) <= TOL_Z


@pytest.mark.parametrize(
    "n", [9, 19, 37, 75, 151, pytest.param(301, marks=pytest.mark.slow)]
)
def test_dipole_ladder(monkeypatch, n):
    """#1409's ladder, d/lambda = 0.0085."""
    assert _routes(monkeypatch, _solver(n=[[n]])) <= TOL_Z


@pytest.mark.slow
def test_fat_dipole_601(monkeypatch):
    assert _routes(monkeypatch, _solver(n=[[601]])) <= TOL_Z


def test_numpy_route_where_the_accelerator_does_not_serve(monkeypatch):
    x0, sgn, h = _GROUPS["uniform"]
    assert isinstance(EK.coaxial_rows(x0, sgn, h, A, K0, 3), EK.CoaxialRowsAccel)
    assert isinstance(EK.coaxial_rows(x0, sgn, h, A, K0 + 0.1j, 3), EK.CoaxialRows)
    assert isinstance(EK.coaxial_rows(x0, sgn, h, A, K0, 7), EK.CoaxialRows)
    monkeypatch.setattr(EK, "_HAVE_ACCEL", False)
    assert isinstance(EK.coaxial_rows(x0, sgn, h, A, K0, 3), EK.CoaxialRows)
    monkeypatch.setattr(EK, "_HAVE_ACCEL", True)
    monkeypatch.setattr(EK, "_USE_ACCEL", False)
    assert isinstance(EK.coaxial_rows(x0, sgn, h, A, K0, 3), EK.CoaxialRows)


# --------------------------------------------------------------------------
# both builds
# --------------------------------------------------------------------------
_VARIANT_PROBE = """
import json
import numpy as np
import momwire._accel as _accel
from momwire import _exact_kernel as EK
from test_exact_kernel_accel_1410 import _GROUPS, _rel, _solver, _Z, A, K0
x0, sgn, h = _GROUPS["two runs + reversed"]
rows = EK.CoaxialRowsAccel(x0, sgn, h, A, K0, 3).rows(0, x0.size)
s = _solver(n=[[37]])
EK._USE_ACCEL = False
z_numpy = _Z(s)
EK._USE_ACCEL = True
z = _Z(s)
print(json.dumps({
    "variant": _accel.VARIANT,
    "rows": rows.view(np.float64).ravel().tolist(),
    "rel": _rel(z, z_numpy),
}))
"""


@pytest.mark.integration
def test_both_builds_agree():
    """Each build the wheel carries (momwire#1032) runs the route and agrees
    with the numpy route in its own process (the fill itself differs between
    the builds, so Z is compared within one). The exact moments are the same
    bits from both builds on one machine: no explicit FMA in this TU,
    contraction off, one libm."""
    from momwire import _accel

    if _accel.VARIANT not in ("avx2", "sse2"):
        pytest.skip("a single-build platform")
    got = {}
    for variant in ("avx2", "sse2"):
        env = dict(os.environ, MOMWIRE_FORCE_VARIANT=variant)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(Path(__file__).parent), env.get("PYTHONPATH", "")]
        )
        out = subprocess.run(
            [sys.executable, "-c", _VARIANT_PROBE],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        rec = json.loads(out.stdout.strip().splitlines()[-1])
        assert rec["variant"] == variant, rec["variant"]
        assert rec["rel"] <= TOL_Z, (variant, rec["rel"])
        got[variant] = rec["rows"]
    if sys.platform == "win32":
        # MSVC builds with /fp:fast, which may contract into FMA under
        # /arch:AVX2 and not at the baseline: the builds' bits may differ
        # there (momwire's Windows rule: a derived tolerance, no bit gate).
        a, b = (np.array(got[v]).view(np.complex128) for v in ("avx2", "sse2"))
        assert _rel(a, b) <= TOL_ROWS_SYM
    else:
        assert got["avx2"] == got["sse2"]


# A graded 1,200-segment wire at a 16 MB window budget: every pair is a key
# new to its window, so the C++ dedup's own tables are as large as they get.
# Traced through the production seam, with the C++ buffers reported to
# tracemalloc (`test_window_buffers_are_visible_to_tracemalloc`). The bar is
# the uniform-wire memgate's (tests/test_exact_kernel_rows_1411.py).
@pytest.mark.memgate
def test_graded_wire_peak_is_one_window(monkeypatch):
    n = 1200
    h = np.r_[np.geomspace(0.02, 0.002, n // 2), np.geomspace(0.002, 0.02, n // 2)]
    z = np.r_[0.0, np.cumsum(h)]
    nodes = np.stack([np.zeros_like(z), np.zeros_like(z), z], -1)
    s = _solver(
        wires=[nodes],
        n=[[1] * n],
        a=0.002,
        swept_mem_mb=16,
    )
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    orig = BSplineSolver._add_exact_kernel_correction
    peak, ran = [], []

    def traced(self, *args):
        tracemalloc.start()
        tracemalloc.reset_peak()
        at_entry = tracemalloc.get_traced_memory()[0]
        try:
            return orig(self, *args)
        finally:
            peak.append((tracemalloc.get_traced_memory()[1] - at_entry) / 2**20)
            tracemalloc.stop()

    rows = EK.CoaxialRowsAccel.rows

    def spy(self, r0, r1):
        ran.append(self.evaluated)
        return rows(self, r0, r1)

    monkeypatch.setattr(BSplineSolver, "_add_exact_kernel_correction", traced)
    monkeypatch.setattr(EK.CoaxialRowsAccel, "rows", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s._compute_Z_operator(geom, supp_seg, polys)
    assert len(peak) == 1 and len(ran) > 1, "the windowed C++ route did not run"
    assert peak[0] <= 48
