"""The exact-kernel correction by observer-row windows (momwire#1411).

`_add_exact_kernel_correction` used to build each coaxial group's whole
(d+1)²·n² correction block, plus n² dedup keys and an (n+1)² tangent table:
about 2 GB on one 3,000-segment wire. It now walks the group in row windows
bounded by `swept_mem_mb` and accumulates each straight into Z. Gates:

* the windowed exact moments (`CoaxialRows`) are the whole block's rows BIT
  FOR BIT, at any window height -- each unique moment comes from the same
  representative pair -- and the cross-window cache keeps a uniform run's
  kernel work at the whole block's unique count, O(n) rather than
  O(n · windows);
* with the cache capped, a key evaluated twice from different representatives
  agrees to roundoff;
* the windowed base moments are the whole-group ones' rows;
* Z through the windows equals Z through the pre-#1411 whole-group route
  (`_EXACT_KERNEL_ROW_WINDOWS = False`) within 1e-12 of max|Z|, on #1409's
  decks plus a fat 601-segment dipole and a bent wire, through the C++
  windowed assembler and through its numpy twin (degree 3 always takes the
  twin), with windows forced down to a single row.
"""

import warnings

import numpy as np
import pytest

from momwire import BSplineSolver
from momwire import _exact_kernel as EK
from momwire import bspline as _bspline

L = 0.47
A = 0.00425
K0 = 2 * np.pi
DIPOLE = [np.array([(0.0, 0.0, -L / 2), (0.0, 0.0, L / 2)])]
BENT = [np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.25), (0.15, 0.0, 0.40)])]


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
    "bent wire": dict(wires=BENT, n=[[23, 17]]),
    "degree 3": dict(n=[[29]], degree=3),
}


def _solver(wires=DIPOLE, n=None, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BSplineSolver(
            wires=wires,
            n_per_edge_per_wire=n,
            wavelength=1.0,
            wire_radius=A,
            exact_kernel=True,
            **kw,
        )


def _Z(s):
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.array(s._compute_Z_operator(geom, supp_seg, polys))


def _windowed_vs_whole(monkeypatch, s, rows):
    calls = []
    orig = BSplineSolver._add_exact_kernel_correction_whole

    def spy(self, *args):
        calls.append(1)
        return orig(self, *args)

    monkeypatch.setattr(BSplineSolver, "_add_exact_kernel_correction_whole", spy)
    monkeypatch.setattr(_bspline, "_EXACT_KERNEL_ROW_WINDOWS", False)
    Z0 = _Z(s)
    assert calls, "the reference route did not run"
    monkeypatch.setattr(_bspline, "_EXACT_KERNEL_ROW_WINDOWS", True)
    monkeypatch.setattr(_bspline, "_EXACT_KERNEL_ROWS_OVERRIDE", rows)
    Z1 = _Z(s)
    assert len(calls) == 1, "the windowed route fell back to the whole route"
    return np.max(np.abs(Z1 - Z0)) / np.max(np.abs(Z0))


def _group(n, *, flip=(), lengths=None):
    h = np.full(n, L / n) if lengths is None else np.asarray(lengths, float)
    x0 = np.concatenate([[0.0], np.cumsum(h)[:-1]])
    sgn = np.ones(n)
    for i in flip:  # a reversed segment starts at its far end
        sgn[i] = -1.0
        x0[i] += h[i]
    return x0, sgn, h


_GROUPS = {
    "uniform": _group(41),
    "two runs + reversed": _group(30, flip=(3, 17), lengths=[0.01] * 12 + [0.017] * 18),
    "graded": _group(16, lengths=np.geomspace(0.004, 0.03, 16)),
}


@pytest.mark.parametrize("rows", [1, 4, 7, 1000])
@pytest.mark.parametrize("name", sorted(_GROUPS))
def test_rows_are_the_block_bit_for_bit(name, rows):
    x0, sgn, h = _GROUPS[name]
    n = x0.size
    whole = EK.coaxial_block(x0, sgn, h, A, K0, 3)
    cr = EK.CoaxialRows(x0, sgn, h, A, K0, 3)
    got = np.concatenate(
        [cr.rows(r0, min(n, r0 + rows)) for r0 in range(0, n, rows)], axis=2
    )
    assert got.shape == whole.shape
    assert np.array_equal(got, whole)
    # the cache carries the dedup across windows: exactly the whole
    # block's unique count, not that count per window
    ref = EK.CoaxialRows(x0, sgn, h, A, K0, 3, max_cached=0)
    ref.rows(0, n)
    assert cr.evaluated == ref.evaluated


def test_uniform_run_kernel_work_stays_linear():
    x0, sgn, h = _group(400)
    cr = EK.CoaxialRows(x0, sgn, h, A, K0, 3)
    for r0 in range(0, 400, 9):
        cr.rows(r0, min(400, r0 + 9))
    assert cr.evaluated <= 2 * 400  # 2n - 1 offsets on a uniform run


def test_capped_cache_agrees_to_roundoff():
    x0, sgn, h = _GROUPS["two runs + reversed"]
    n = x0.size
    whole = EK.coaxial_block(x0, sgn, h, A, K0, 3)
    cr = EK.CoaxialRows(x0, sgn, h, A, K0, 3, max_cached=5)
    got = np.concatenate([cr.rows(r0, min(n, r0 + 3)) for r0 in range(0, n, 3)], 2)
    assert len(cr._index) == 5
    assert np.max(np.abs(got - whole)) <= 1e-13 * np.max(np.abs(whole))


@pytest.mark.parametrize("deck", ["end-port pair", "collinear junction + bend"])
def test_base_rows_are_the_whole_base(deck):
    kw = dict(_DECKS[deck])
    s = _solver(**kw)
    geom = s._build_geometry()
    for S, g_edges in s._exact_kernel_groups(geom):
        n = S.size
        whole = s._exact_kernel_base_J(geom, S, g_edges, s.k)
        for r0, r1 in ((0, 1), (3, 11), (n - 5, n)):
            part = s._exact_kernel_base_J(geom, S, g_edges, s.k, rows=slice(r0, r1))
            assert part.shape == (s.degree + 1,) * 2 + (r1 - r0, n)
            scale = np.max(np.abs(whole))
            assert np.max(np.abs(part - whole[:, :, r0:r1])) <= 1e-14 * scale


@pytest.mark.parametrize("rows", [1, 5])
@pytest.mark.parametrize("ek", [False, True])
@pytest.mark.parametrize("deck", sorted(_DECKS))
def test_windows_reproduce_the_whole_route(monkeypatch, deck, ek, rows):
    s = _solver(extended_kernel=ek, **_DECKS[deck])
    assert _windowed_vs_whole(monkeypatch, s, rows) <= 1e-12


@pytest.mark.parametrize("deck", ["end-port pair", "bent wire"])
def test_numpy_twin_reproduces_the_whole_route(monkeypatch, deck):
    monkeypatch.setattr(_bspline, "_HAVE_BSPLINE_WINDOWED_ASSEMBLE_ACCEL", False)
    s = _solver(extended_kernel=True, **_DECKS[deck])
    assert _windowed_vs_whole(monkeypatch, s, 3) <= 1e-12


@pytest.mark.slow
def test_budget_cuts_windows(monkeypatch):
    """The default route (no override) at a 1 MB budget cuts a 601-segment
    wire into many windows and still matches."""
    s = _solver(n=[[601]], swept_mem_mb=1)
    assert _schedule_rows(s, 601) < 601
    assert _windowed_vs_whole(monkeypatch, s, None) <= 1e-12


def _schedule_rows(s, n):
    return _bspline._schedule.mb_rows(s.swept_mem_mb, s._exact_kernel_row_bytes(n))


@pytest.mark.slow
@pytest.mark.parametrize("ek", [False, True])
def test_fat_dipole_601(monkeypatch, ek):
    s = _solver(n=[[601]], extended_kernel=ek)
    assert _windowed_vs_whole(monkeypatch, s, 64) <= 1e-12


# A 1,500-segment fat wire at a 16 MB window budget. Traced inside the call
# on Haswell: 1,391 MB by the whole-group route (`_EXACT_KERNEL_ROW_WINDOWS =
# False`: the (d+1)²·n² block twice, the n² keys, an (n+1)² tangent table),
# 24 MB by the windows -- one window plus O(n) tables. The bar is twice that.
_MEM_N = 1500
_MEM_BUDGET_MB = 16
_MEM_BAR_MB = 48


def _correction_peak_mb(monkeypatch):
    import tracemalloc

    s = _solver(
        wires=[np.array([(0.0, 0.0, -2.25), (0.0, 0.0, 2.25)])],
        n=[[_MEM_N]],
        swept_mem_mb=_MEM_BUDGET_MB,
    )
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    orig = BSplineSolver._add_exact_kernel_correction
    peak = []

    def traced(self, *args):
        tracemalloc.start()
        tracemalloc.reset_peak()
        at_entry = tracemalloc.get_traced_memory()[0]
        try:
            return orig(self, *args)
        finally:
            peak.append((tracemalloc.get_traced_memory()[1] - at_entry) / 2**20)
            tracemalloc.stop()

    monkeypatch.setattr(BSplineSolver, "_add_exact_kernel_correction", traced)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s._compute_Z_operator(geom, supp_seg, polys)
    assert len(peak) == 1, "the correction did not run"
    return peak[0]


@pytest.mark.memgate
def test_correction_peak_is_one_window(monkeypatch):
    assert _correction_peak_mb(monkeypatch) <= _MEM_BAR_MB
