"""The point-matched fill with the contraction fused into the kernel —
momwire#2069 (and the source-axis tiles of #2070 under it).

`_assemble_Z_filled` hands a band to `_acc.sinusoidal_fill_rows` where the
two-pass band would run the real-k C++ kernel and `reduce_phi_band`: free
space or a PEC image, reduced or extended kernel. Each observer row's Φ is
computed into one thread's buffer and contracted there in
`reduce_phi_band`'s order, so no Φ band exists.

Gates:

  * Z with the fused route equals the two-pass route
    (`_FILL_ROWS_ACCEL = False`) BYTE FOR BYTE, on free space and PEC, both
    kernels, a stepped radius, a deck wider than one source tile with a
    ragged last tile, and the row-compact sector fill; a spy counts that
    each route ran what it claims;
  * the thread count moves nothing;
  * what the fused route does not serve (a finite ground, the dense
    regime, pair reuse) keeps the two-pass band.

On Windows the kernel TU builds /fp:fast while the reduction's TU is
precise, so the two routes agree to round-off there, not to the bit
(setup.py; no bit contract on win32).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from test_sin_reduce_tiles_1290 import _free_array_deck

from momwire import sinusoidal as sn
from momwire.sinusoidal import SinusoidalSolver

pytestmark = pytest.mark.skipif(
    not (sn._HAVE_FILL_ROWS and sn._HAVE_PHI_REDUCE_ACCEL),
    reason="the accelerator does not carry sinusoidal_fill_rows",
)

EXACT = sys.platform != "win32"


def _same(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    if a.shape != b.shape:
        return False
    if EXACT:
        return a.tobytes() == b.tobytes()
    return float(np.max(np.abs(a - b))) <= 1e-12 * float(np.max(np.abs(b)))


@pytest.fixture
def calls(monkeypatch):
    """Calls to the fused kernel and to the two-pass reduction, and the
    fused kernel's arguments (for the thread-count gate)."""
    seen = {"fused": 0, "reduce": 0, "args": []}
    acc = sn._acc

    class _Acc:
        def __getattr__(self, name):
            return getattr(acc, name)

        def sinusoidal_fill_rows(self, *a, **kw):
            seen["fused"] += 1
            seen["args"].append(a)
            return acc.sinusoidal_fill_rows(*a, **kw)

        def reduce_phi_band(self, *a, **kw):
            seen["reduce"] += 1
            return acc.reduce_phi_band(*a, **kw)

    monkeypatch.setattr(sn, "_acc", _Acc())
    return seen


def _deck(kind):
    if kind == "free":
        return _free_array_deck()
    if kind == "free-wide":
        # N = 8 * 41 = 328: one full source tile and a ragged one.
        return _free_array_deck(segs=41)
    if kind == "pec":
        return dict(_free_array_deck(), ground_z=0.0)
    if kind == "stepped":
        d = _free_array_deck(segs=23)
        d["wire_radius"] = [1e-3 if j % 2 else 2.5e-3 for j in range(8)]
        return d
    if kind == "stepped-pec":
        d = _free_array_deck(segs=19)
        d["wire_radius"] = [1e-3 if j % 3 else 4e-3 for j in range(8)]
        d["ground_z"] = 0.0
        return d
    raise AssertionError(kind)


def _fill(kw, fused, monkeypatch, obs_rows=None):
    s = SinusoidalSolver(**kw)
    geom = s._build_geometry()
    with monkeypatch.context() as m:
        m.setattr(sn, "_FILL_ROWS_ACCEL", fused)
        if obs_rows is None:
            G, _ = s._assemble_Z(geom, s.k)
        else:
            G, _ = s._assemble_Z(geom, s.k, obs_rows=obs_rows)
    return G


@pytest.mark.parametrize("ek", [False, True])
@pytest.mark.parametrize("kind", ["free", "free-wide", "pec", "stepped", "stepped-pec"])
def test_fused_fill_is_the_two_pass_fill(kind, ek, calls, monkeypatch):
    kw = dict(_deck(kind), extended_kernel=ek)
    ref = _fill(kw, False, monkeypatch)
    assert calls["fused"] == 0 and calls["reduce"] > 0
    calls["reduce"] = 0
    got = _fill(kw, True, monkeypatch)
    assert calls["fused"] > 0 and calls["reduce"] == 0
    assert ref.shape[0] >= sn._DENSE_ASSEMBLY_THRESHOLD
    assert _same(got, ref), kind


def test_fused_fill_banded_and_sector_rows(calls, monkeypatch):
    kw = _deck("pec")
    # Small bands, so the fill runs many fused calls.
    kw["swept_mem_mb"] = 1
    ref = _fill(kw, False, monkeypatch)
    got = _fill(kw, True, monkeypatch)
    assert calls["fused"] > 2
    assert _same(got, ref)
    rows = np.array([0, 1, 2, 40, 41, 97, 150, 175])
    ref_r = _fill(kw, False, monkeypatch, obs_rows=rows)
    got_r = _fill(kw, True, monkeypatch, obs_rows=rows)
    assert _same(got_r, ref_r)
    assert _same(got_r, ref[rows])


@pytest.mark.parametrize("ek", [False, True])
def test_thread_count_moves_nothing(ek, calls, monkeypatch):
    kw = dict(_deck("stepped-pec"), extended_kernel=ek)
    _fill(kw, True, monkeypatch)
    a = calls["args"][0]
    out = a[0]
    rest = a[1:-2]
    outs = []
    for nt in (1, 2, 3, 8):
        o = np.full_like(out, np.nan)
        sn._acc.sinusoidal_fill_rows(o, *rest, nt, 0)
        outs.append(o)
    for o in outs[1:]:
        assert o.tobytes() == outs[0].tobytes()


def test_any_out_layout(calls, monkeypatch):
    """`out` is written through its strides: a Fortran-ordered destination
    holds the same values."""
    kw = _deck("free")
    _fill(kw, True, monkeypatch)
    a = calls["args"][0]
    c = np.full(a[0].shape, np.nan + 0j)
    f = np.asfortranarray(np.full(a[0].shape, np.nan + 0j))
    sn._acc.sinusoidal_fill_rows(c, *a[1:])
    sn._acc.sinusoidal_fill_rows(f, *a[1:])
    assert f.flags.f_contiguous
    assert np.ascontiguousarray(f).tobytes() == c.tobytes()


def test_declines_keep_the_two_pass_band(calls, monkeypatch):
    # A finite ground: the image term is built per band in numpy.
    kw = dict(_deck("free"), ground_z=0.0, ground_eps=(13.0, 0.005))
    _fill(kw, True, monkeypatch)
    assert calls["fused"] == 0 and calls["reduce"] > 0
    # The dense regime (N below the threshold) reduces through zgemm.
    calls["reduce"] = 0
    _fill(_free_array_deck(segs=5), True, monkeypatch)
    assert calls["fused"] == 0
