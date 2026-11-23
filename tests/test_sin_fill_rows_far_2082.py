"""NEC-2's interaction approximation inside the fused fill — momwire#2082,
step 1.

With `nec2_far_approx` on, `_assemble_Z_filled` used to decline the fused
fill (#2069) and take the two-pass band, so turning the approximation on made
the fill slower. The fused kernel now takes `far_r` and lumps the far pairs
itself (`sinusoidal_fill_rows_far`, `_sinusoidal_far_core.cpp`).

Gates:

  * the fused far fill equals the two-pass far fill (`_FILL_ROWS_ACCEL =
    False`: `sinusoidal_field_tensor_far`, then `reduce_phi_band`) BYTE FOR
    BYTE, in free space and over PEC, uniform and stepped radius, on a deck
    whose rows run several near and far flushes with ragged last ones, and on
    the row-compact sector fill; a spy counts that each route ran what it
    claims (the fused one never calls the two-pass far kernel);
  * the thread count moves nothing;
  * a range past every pair is the plain fused fill, to the bit;
  * the extended kernel keeps the two-pass band (it has no lumped twin);
  * the binding refuses `far_r` with the extended kernel's tables, and a
    negative range.

On Windows the kernel TU builds /fp:fast while the reduction's TU is precise,
so the two routes agree to round-off there, not to the bit (setup.py; no bit
contract on win32).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from test_sin_reduce_tiles_1290 import _free_array_deck

from momwire import sinusoidal as sn
from momwire.sinusoidal import SinusoidalSolver

pytestmark = pytest.mark.skipif(
    not (sn._HAVE_FILL_ROWS_FAR and sn._HAVE_PHI_REDUCE_ACCEL),
    reason="the accelerator does not carry sinusoidal_fill_rows' far_r",
)

EXACT = sys.platform != "win32"

# The array's dipoles are 20 m long and 8.5 m apart at 42.8 m: 0.3 wavelength
# (12.8 m) puts each row's own dipole mostly near and the far dipoles far.
RANGE = 0.3


def _same(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    if a.shape != b.shape:
        return False
    if EXACT:
        return a.tobytes() == b.tobytes()
    return float(np.max(np.abs(a - b))) <= 1e-12 * float(np.max(np.abs(b)))


@pytest.fixture
def calls(monkeypatch):
    """Calls to the fused kernel (plain and far), to the two-pass far kernel
    and to the two-pass reduction, and the fused far calls' arguments."""
    seen = {"fused": 0, "fused_far": 0, "far_tensor": 0, "reduce": 0, "args": []}
    acc = sn._acc

    class _Acc:
        def __getattr__(self, name):
            return getattr(acc, name)

        def sinusoidal_fill_rows(self, *a, **kw):
            if kw.get("far_r") is None:
                seen["fused"] += 1
            else:
                seen["fused_far"] += 1
                seen["args"].append((a, kw))
            return acc.sinusoidal_fill_rows(*a, **kw)

        def sinusoidal_field_tensor_far(self, *a, **kw):
            seen["far_tensor"] += 1
            return acc.sinusoidal_field_tensor_far(*a, **kw)

        def reduce_phi_band(self, *a, **kw):
            seen["reduce"] += 1
            return acc.reduce_phi_band(*a, **kw)

    monkeypatch.setattr(sn, "_acc", _Acc())
    return seen


def _deck(kind):
    if kind == "free":
        return _free_array_deck()
    if kind == "free-wide":
        # N = 8 * 160 = 1280: a row has up to ~300 near pairs and ~1000 far
        # ones, so it runs more than one near flush (256) and, on the end
        # dipoles' rows, more than one far flush (1024), each with a ragged
        # last one.
        return _free_array_deck(segs=160)
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


def _solver(kw, far_range=RANGE):
    return SinusoidalSolver(**kw, nec2_far_approx=True, nec2_far_approx_range=far_range)


def _fill(kw, fused, monkeypatch, obs_rows=None, far_range=RANGE, far=True):
    s = _solver(kw, far_range) if far else SinusoidalSolver(**kw)
    geom = s._build_geometry()
    with monkeypatch.context() as m:
        m.setattr(sn, "_FILL_ROWS_ACCEL", fused)
        if obs_rows is None:
            G, _ = s._assemble_Z(geom, s.k)
        else:
            G, _ = s._assemble_Z(geom, s.k, obs_rows=obs_rows)
    return G


def _far_pair_count(kw, far_range=RANGE):
    s = _solver(kw, far_range)
    geom = s._build_geometry()
    c = geom["seg_centers"]
    d2 = np.sum((c[:, None, :] - c[None, :, :]) ** 2, axis=-1)
    far_r = s._far_range_m(s.k)
    return int(np.sum(d2 > far_r * far_r)), d2.size, (d2 > far_r * far_r)


@pytest.mark.parametrize("kind", ["free", "free-wide", "pec", "stepped", "stepped-pec"])
def test_fused_far_fill_is_the_two_pass_far_fill(kind, calls, monkeypatch):
    kw = _deck(kind)
    n_far, n_all, _ = _far_pair_count(kw)
    assert 0 < n_far < n_all  # the range splits the pairs
    ref = _fill(kw, False, monkeypatch)
    assert calls["fused"] == calls["fused_far"] == 0
    assert calls["far_tensor"] > 0 and calls["reduce"] > 0
    calls["far_tensor"] = calls["reduce"] = 0
    got = _fill(kw, True, monkeypatch)
    assert calls["fused_far"] > 0
    assert calls["fused"] == calls["far_tensor"] == calls["reduce"] == 0
    assert ref.shape[0] >= sn._DENSE_ASSEMBLY_THRESHOLD
    assert _same(got, ref), kind
    # And the approximation is in it: the exact fill differs.
    exact = _fill(kw, True, monkeypatch, far=False)
    assert not np.array_equal(exact, got)


def test_the_wide_deck_runs_several_flushes():
    """`free-wide`'s rows overrun one near flush and one far flush, with
    ragged last ones: the case the flush alignment argument is about."""
    _, _, far = _far_pair_count(_deck("free-wide"))
    n_far = far.sum(axis=1)
    n_near = far.shape[1] - n_far
    assert n_near.max() > 256 and (n_near % 256 != 0).any()
    assert n_far.max() > 1024 and (n_far % 1024 != 0).any()


def test_fused_far_fill_banded_and_sector_rows(calls, monkeypatch):
    kw = _deck("pec")
    kw["swept_mem_mb"] = 1
    ref = _fill(kw, False, monkeypatch)
    got = _fill(kw, True, monkeypatch)
    assert calls["fused_far"] > 2
    assert _same(got, ref)
    rows = np.array([0, 1, 2, 40, 41, 97, 150, 175])
    ref_r = _fill(kw, False, monkeypatch, obs_rows=rows)
    got_r = _fill(kw, True, monkeypatch, obs_rows=rows)
    assert _same(got_r, ref_r)
    assert _same(got_r, ref[rows])


def test_the_image_pairs_take_their_own_distance(calls, monkeypatch):
    """Over PEC the image block is far-aware too, decided at the image
    segment's centre: a range between the deck's direct and image extents
    moves the answer only through the image pairs, fused as two-pass."""
    kw = dict(_free_array_deck(), ground_z=-25.0)  # images 70 m below
    # 66 m: no direct pair is that far (the array's diagonal is 62.8 m), and
    # every image pair is (at least 70 m).
    rng = 66.0 / kw["wavelength"]
    n_far, _, _ = _far_pair_count(kw, rng)
    assert n_far == 0
    ref = _fill(kw, False, monkeypatch, far_range=rng)
    got = _fill(kw, True, monkeypatch, far_range=rng)
    assert calls["fused_far"] > 0
    assert _same(got, ref)
    exact = _fill(kw, True, monkeypatch, far=False)
    assert not np.array_equal(exact, got)


def test_thread_count_moves_nothing(calls, monkeypatch):
    kw = _deck("stepped-pec")
    _fill(kw, True, monkeypatch)
    a, kwargs = calls["args"][0]
    out = a[0]
    rest = a[1:-2]
    outs = []
    for nt in (1, 2, 3, 8):
        o = np.full_like(out, np.nan)
        sn._acc.sinusoidal_fill_rows(o, *rest, nt, 0, **kwargs)
        outs.append(o)
    for o in outs[1:]:
        assert o.tobytes() == outs[0].tobytes()


def test_a_range_past_every_pair_is_the_plain_fused_fill(monkeypatch):
    """With every pair near, the compacted near phases are the source axis
    itself and a flush is #2070's tile, so the far kernel is the plain one."""
    kw = _deck("free-wide")
    plain = _fill(kw, True, monkeypatch, far=False)
    got = _fill(kw, True, monkeypatch, far_range=1e3)
    assert _same(got, plain)


def test_the_extended_kernel_keeps_the_two_pass_band(calls, monkeypatch):
    kw = dict(_deck("free"), extended_kernel=True)
    ref = _fill(kw, False, monkeypatch)
    got = _fill(kw, True, monkeypatch)
    assert calls["fused"] == calls["fused_far"] == 0 and calls["reduce"] > 0
    assert _same(got, ref)


def test_the_binding_refuses_ek_tables_and_a_negative_range(calls, monkeypatch):
    _fill(_deck("free"), True, monkeypatch)
    a, kwargs = calls["args"][0]
    a = list(a)
    out = np.empty_like(a[0])
    with pytest.raises(RuntimeError, match="far_r must be a non-negative"):
        sn._acc.sinusoidal_fill_rows(out, *a[1:], far_r=-1.0)
    n = a[4].shape[0]
    # src_a, ind1, ind2 sit at positions 13, 14, 15.
    a[13] = np.full(n, 1e-3)
    a[14] = np.zeros(n, dtype=np.int8)
    a[15] = np.zeros(n, dtype=np.int8)
    with pytest.raises(RuntimeError, match="reduced kernel only"):
        sn._acc.sinusoidal_fill_rows(out, *a[1:], **kwargs)
