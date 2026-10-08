"""Cancellation reaches the below/below Sommerfeld grid fill: momwire#1348.

`get_grid_below` took no token and its contour batch
(`below_six_integrals_batch`) no `cancel_flag`, so on a COLD grid cache a
buried solve could not be cancelled while the grid filled. Measured on Skylake
before this change: razor buried x16 cancelled 2 s in held for 3.2 - 6 s under
load, 0.47 s quiet. A warm cache skips the fill, which is why #1342's sweep did
not see it -- and hosted machines start cold (antennaknobs#1882's watchdog).

The getters now take the solve's `cancel_flag` (the raw address
`_sommerfeld.get_grid` already takes), install it as the ambient token for the
fill (`_cancel.scope_flag`), poll it at every region boundary, and hand it to
the contour batch, which polls it per node. The below->above transmitted grid
(`get_grid_below_above`, `transmitted_six_integrals_batch`) had the same hole
and gets the same plumbing.

What this pins, deterministically (wall time never gates CI):

  * the kernel honours the flag, and an untripped flag changes no bit;
  * a cold fill tripped after its first batch call makes no further one, and
    tripped as that call starts, raises from inside it -- on every solver that
    fills the grid. Each counts the calls it intercepted; a control counts them
    with no trip, so a count of 1 is the poll and not a one-call deck;
  * an aborted fill caches nothing, in memory or on disk, and the next solve
    builds a complete grid: Z bit-identical to an uninterrupted cold run;
  * an aborted lazy band region stays unfilled;
  * (opt-in, MOMWIRE_CANCEL_LATENCY=1) a cancel 1 - 2 s into a cold x16 fill
    returns within 1 s.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback

import numpy as np
import pytest

from momwire import (
    BSplineSolver,
    CancelToken,
    RazorSolver,
    SinusoidalGalerkinSolver,
    SinusoidalSolver,
    SolveAborted,
    _cancel,
    _sommerfeld,
    _sommerfeld_below,
    _sommerfeld_transmitted,
)
from momwire._accel import acc
from momwire._sommerfeld_below import k_medium
from test_cancel_buried_fill_1342 import _hub_x16
from test_crossing_serve_524 import hub_deck

pytestmark = pytest.mark.skipif(
    acc is None or not hasattr(acc, "below_six_integrals_batch"),
    reason="below/below contour accelerator not built",
)

KP = 2.0 * np.pi / 42.831
EPS_T = 13.0 - 12.84j

_MAKE = {
    "razor": lambda **kw: RazorSolver(**kw, nec5_quadrature=True),
    "bs2": lambda **kw: BSplineSolver(**kw),
    "sin": lambda **kw: SinusoidalSolver(**kw),
    "sg": lambda **kw: SinusoidalGalerkinSolver(**kw),
}
_ENGINES = list(_MAKE)


@pytest.fixture
def cold_memory():
    """An empty in-process grid cache for the test, the old one restored after.
    The disk level is already off in the suite (conftest), so this is a cold
    cache; a test that wants the disk sets `MOMWIRE_SOMM_CACHE` itself."""
    cache = _sommerfeld._GRID_CACHE  # one dict, shared by every family
    saved = dict(cache)
    cache.clear()
    yield cache
    cache.clear()
    cache.update(saved)


def _count_batches(monkeypatch, tok=None, when=None):
    """Intercept the below/below batch. `when="after"` trips `tok` once the
    first call has returned; `"before"` trips it as the first call starts, so
    the kernel itself sees the flag. Returns the record."""
    real = acc.below_six_integrals_batch
    rec = {"calls": 0, "kernel_raised": False}

    def counting(*a, **kw):
        rec["calls"] += 1
        if when == "before" and rec["calls"] == 1:
            tok.cancel()
        try:
            out = real(*a, **kw)
        except SolveAborted:
            rec["kernel_raised"] = True
            raise
        if when == "after" and rec["calls"] == 1:
            tok.cancel()
        return out

    monkeypatch.setattr(acc, "below_six_integrals_batch", counting)
    return rec


# --------------------------------------------------------------------------
# The kernels.
# --------------------------------------------------------------------------


def _batch_args():
    rho = np.linspace(0.05, 6.0, 40)
    h = np.linspace(0.3, 0.05, 40)
    return KP, k_medium(EPS_T, KP), rho, h


def test_the_below_batch_honours_the_flag_and_an_untripped_one_changes_nothing():
    k_p, k_m, rho, h = _batch_args()
    ref = _sommerfeld_below._six_below_accel(k_p, k_m, rho, h, 1e-9, True)
    tok = CancelToken()
    with _cancel.scope(tok):
        got = _sommerfeld_below._six_below_accel(k_p, k_m, rho, h, 1e-9, True)
    for a, b in zip(ref, got):
        assert np.array_equal(a, b)
    tok.cancel()
    with _cancel.scope(tok), pytest.raises(SolveAborted):
        _sommerfeld_below._six_below_accel(k_p, k_m, rho, h, 1e-9, True)


def test_the_transmitted_batch_honours_the_flag_and_an_untripped_one_changes_nothing():
    k_p, k_m, rho, h = _batch_args()
    z = np.full_like(rho, 1.0)
    args = (k_p, k_m, rho, z, -h, False, 1e-9, False)
    ref = _sommerfeld_transmitted._six_transmitted_accel(*args)
    tok = CancelToken()
    with _cancel.scope(tok):
        got = _sommerfeld_transmitted._six_transmitted_accel(*args)
    for a, b in zip(ref, got):
        assert np.array_equal(a, b)
    tok.cancel()
    with _cancel.scope(tok), pytest.raises(SolveAborted):
        _sommerfeld_transmitted._six_transmitted_accel(*args)


def test_a_zero_flag_keeps_the_ambient_token():
    tok = CancelToken()
    tok.cancel()
    with _cancel.scope(tok), _cancel.scope_flag(0), pytest.raises(SolveAborted):
        _cancel.poll()
    with _cancel.scope_flag(tok.ptr), pytest.raises(SolveAborted):
        _cancel.poll()
    _cancel.poll()  # restored: no token


# --------------------------------------------------------------------------
# The fill, from a solve, on a cold cache.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", _ENGINES)
def test_a_cold_fill_tripped_after_its_first_batch_makes_no_further_one(
    monkeypatch, cold_memory, engine
):
    """On the base of this change the fill ran every remaining batch (4 at this
    deck) and the solve finished: nothing in the fill read the token."""
    tok = CancelToken()
    rec = _count_batches(monkeypatch, tok, when="after")
    s = _MAKE[engine](**hub_deck(n_radials=4), cancel=tok)
    with pytest.raises(SolveAborted):
        s.compute_impedance()
    assert rec["calls"] == 1
    assert not any(k[0] == "below" for k in cold_memory)


@pytest.mark.parametrize("engine", _ENGINES)
def test_a_cold_fill_tripped_inside_its_first_batch_raises_from_the_kernel(
    monkeypatch, cold_memory, engine
):
    tok = CancelToken()
    rec = _count_batches(monkeypatch, tok, when="before")
    s = _MAKE[engine](**hub_deck(n_radials=4), cancel=tok)
    with pytest.raises(SolveAborted):
        s.compute_impedance()
    assert rec["calls"] == 1
    assert rec["kernel_raised"]


@pytest.mark.parametrize("engine", _ENGINES)
def test_not_cancelled_a_cold_fill_makes_several_batches(
    monkeypatch, cold_memory, engine
):
    """The control: with no trip the same cold solve makes several batch
    calls, so the count of 1 above is the poll."""
    rec = _count_batches(monkeypatch)
    z = _MAKE[engine](**hub_deck(n_radials=4), cancel=CancelToken()).compute_impedance()
    assert rec["calls"] >= 3
    assert np.isfinite(np.atleast_1d(z[0])[0])


def _z(engine):
    return np.atleast_1d(
        _MAKE[engine](
            **hub_deck(n_radials=4), cancel=CancelToken()
        ).compute_impedance()[0]
    )


@pytest.mark.slow
def test_an_aborted_fill_leaves_no_partial_grid_in_memory_or_on_disk(
    monkeypatch, cold_memory, tmp_path
):
    """Abort a cold fill with the disk store ON, then solve again: nothing was
    cached on either level, the next solve fills a complete grid, and its Z is
    bit-identical to an uninterrupted cold run -- and so is a solve served from
    the grid that run persisted."""
    store = tmp_path / "store"
    monkeypatch.setenv("MOMWIRE_SOMM_CACHE", "1")
    monkeypatch.setenv("MOMWIRE_SOMM_CACHE_DIR", str(store))

    with monkeypatch.context() as m:
        tok = CancelToken()
        rec = _count_batches(m, tok, when="after")
        s = _MAKE["razor"](**hub_deck(n_radials=4), cancel=tok)
        with pytest.raises(SolveAborted):
            s.compute_impedance()
        assert rec["calls"] == 1
    assert not any(k[0] == "below" for k in cold_memory)
    # The above/above grid filled before the abort is complete and may be
    # persisted; no below/below file may exist, whole or partial.
    assert not list(store.glob("below-*")), list(store.iterdir())

    z_after = _z("razor")  # cold below/below: the abort left nothing
    assert len(list(store.glob("below-*.npz"))) == 1

    cold_memory.clear()
    z_warm = _z("razor")  # memory cold, disk warm: the persisted grid serves

    cold_memory.clear()
    monkeypatch.setenv("MOMWIRE_SOMM_CACHE_DIR", str(tmp_path / "fresh"))
    z_ref = _z("razor")  # an uninterrupted cold run
    assert np.array_equal(z_after, z_ref), (z_after, z_ref)
    assert np.array_equal(z_warm, z_ref), (z_warm, z_ref)


def test_an_aborted_lazy_band_region_stays_unfilled(monkeypatch, cold_memory):
    """A deferred band region whose fill aborts inside its batch is left
    exactly as it was -- unfilled, NaN -- so the next query that reaches it
    fills it whole."""
    lam_m = 2.0 * np.pi / abs(k_medium(EPS_T, KP))
    grid = _sommerfeld_below.get_grid_below(EPS_T, KP, 0.5 * lam_m, KP * 299792458.0)
    idx = grid._band_idx[0]
    assert not grid._regions[idx]["filled"]
    tok = CancelToken()
    rec = _count_batches(monkeypatch, tok, when="before")
    with _cancel.scope(tok), pytest.raises(SolveAborted):
        grid._ensure_band()
    assert rec["calls"] == 1
    assert rec["kernel_raised"]
    reg = grid._regions[idx]
    assert not reg["filled"]
    assert np.all(np.isnan(reg["vals"]))


# --------------------------------------------------------------------------
# Opt-in: the bound itself, on a cold x16 fill.
# --------------------------------------------------------------------------

BOUND_S = 1.0


def _cold_fill_latency(make, deck, at):
    """(seconds from the flag to the raise, the innermost frames at the flag).
    The flag goes up at the first moment at or after `at` seconds that the
    solve is inside `_sommerfeld_below`'s fill, sampled every 2 ms."""
    tok = CancelToken()
    solver = make(**deck, cancel=tok)
    main = threading.get_ident()
    seen = {}
    done = threading.Event()

    def trip():
        time.sleep(at)
        while not done.is_set():
            frames = traceback.extract_stack(sys._current_frames().get(main))
            names = [f.name for f in frames]
            if "_six_integrals_below_many" in names:
                seen["files"] = [
                    f"{f.filename.rsplit('/', 1)[-1]}:{f.name}" for f in frames[-5:]
                ]
                seen["t"] = time.perf_counter()
                tok.cancel()
                return
            time.sleep(0.002)

    threading.Thread(target=trip, daemon=True).start()
    try:
        solver.compute_impedance()
    except SolveAborted:
        if "t" not in seen:
            pytest.fail("aborted without the flag")
        return time.perf_counter() - seen["t"], seen["files"]
    finally:
        done.set()
    pytest.skip(f"the solve left the grid fill before {at} s, or never entered it")


@pytest.mark.skipif(
    not os.environ.get("MOMWIRE_CANCEL_LATENCY"),
    reason="wall-clock latency gate: CI runners spread ~3x on the same code, so "
    "it runs by hand (MOMWIRE_CANCEL_LATENCY=1), never as a CI gate",
)
@pytest.mark.slow
@pytest.mark.parametrize("at", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("engine", _ENGINES)
def test_a_cancel_mid_cold_fill_returns_within_the_bound(cold_memory, engine, at):
    latency, files = _cold_fill_latency(_MAKE[engine], _hub_x16(), at)
    print(f"\nLATENCY {engine} at={at}: {latency:.4f} s {files}")
    assert latency < BOUND_S, (latency, files)
