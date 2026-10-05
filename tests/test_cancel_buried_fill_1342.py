"""Cancellation reaches the buried fill: momwire#1342.

`cancel_flag` reached the below-interface fills and the bspline / sinusoidal
kernels, but `_crossing_fill` and `_near_interface` had no seam at all, so a
buried or inverted-L solve could not be cancelled in its most expensive phase
-- which is what the hosted app's watchdog (antennaknobs#1882) waits on.
Measured on Skylake at hub_deck(16) x16 on the base of this change, seconds from
the flag (set at 1..7 s into the solve) to the raise:

    SG buried 0.0 - 7.2     SG invl 0.0 - 4.5
    razor buried 0.0 - 2.5  razor invl 0.0 - 9.7

and 0.00 - 0.33 s with the seams. The time was not one long kernel call: every
column-twin / sheet call is already bounded by the callers' batching, and what
a cancel waited out was the SUCCESSION of them (and the numpy census / plan
between them) with no poll anywhere. What this pins:

  * the seams exist and are reached: a token tripped from INSIDE a kernel call
    (so no timer, no race) stops the fill before its next call. A harness that
    never reaches the kernel would pass vacuously, so each test counts the
    calls it intercepted, and a control counts them with no trip;
  * an abort inside `PlaneSheet.cover` leaves the CACHED sheet consistent;
  * SG's own near-correction polls (`_near_pairs`, `_apply_near_correction`),
    by call count;
  * (opt-in, MOMWIRE_CANCEL_LATENCY=1) a cancel in the middle of an x16 fill
    returns within a bound. Wall time never gates CI.
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
    SolveAborted,
    _cancel,
    _near_interface,
)
from momwire._near_interface import k_medium
from test_crossing_serve_524 import hub_deck, invl_deck

pytestmark = pytest.mark.skipif(
    _near_interface._nia is None, reason="near-interface accelerator not built"
)

KP = 2.0 * np.pi / 42.831
EPS_T = 13.0 - 12.84j


# --------------------------------------------------------------------------
# An abort in `cover` leaves the process-cached sheet usable.
# --------------------------------------------------------------------------


def test_abort_inside_cover_leaves_the_sheet_consistent():
    k_m = k_medium(EPS_T, KP)
    lam = _near_interface._LAM_MULT
    tok = CancelToken()
    tok.cancel()
    sheet = _near_interface.PlaneSheet(KP, k_m, -0.15, lam)
    with _cancel.scope(tok), pytest.raises(SolveAborted):
        sheet.cover(6.0, 6.0)
    assert sheet.n_nodes == 0
    assert all(len(v) == 0 for v in sheet.strips.values())
    assert all(len(v) == 0 for v in sheet.cells.values())
    sheet.cover(6.0, 6.0)  # no token now: grows from empty, as a fresh sheet
    fresh = _near_interface.PlaneSheet(KP, k_m, -0.15, lam)
    fresh.cover(6.0, 6.0)
    for a, b in zip(sheet.arrays(), fresh.arrays()):
        assert np.array_equal(a, b)


# --------------------------------------------------------------------------
# The seams are reached from a solve (tripped from inside a kernel call).
# --------------------------------------------------------------------------


_ENGINES = [
    pytest.param(lambda **kw: RazorSolver(**kw, nec5_quadrature=True), id="razor"),
    pytest.param(lambda **kw: SinusoidalGalerkinSolver(**kw), id="sg"),
]


def _count_column_calls(monkeypatch, on_call=None):
    calls = [0]
    real = _near_interface._nia

    class Counting:
        def __getattr__(self, name):
            return getattr(real, name)

        def near_interface_six_columns(self, *a):
            out = real.near_interface_six_columns(*a)
            calls[0] += 1
            if on_call is not None:
                on_call()
            return out

    monkeypatch.setattr(_near_interface, "_nia", Counting())
    return calls


@pytest.mark.parametrize("make", _ENGINES)
def test_a_cancel_inside_the_near_interface_kernel_stops_the_fill(monkeypatch, make):
    """Tripped from inside the first column-twin call, the solve must raise
    before the fill's NEXT one. The solvers' own later checkpoints would abort
    it eventually either way, so only the count says whether the fill's seams
    work: on the base of this change the fill ran every remaining call."""
    tok = CancelToken()
    calls = _count_column_calls(monkeypatch, on_call=tok.cancel)
    s = make(**hub_deck(n_radials=4), cancel=tok)
    with pytest.raises(SolveAborted):
        s.compute_impedance()
    assert calls[0] == 1


@pytest.mark.parametrize("make", _ENGINES)
def test_not_cancelled_the_fill_makes_several_calls(monkeypatch, make):
    """The control for the test above: with no trip the same solve makes
    several column-twin calls (razor 8, SG 3 at this deck), so a count of 1
    there is the seams stopping it and not a deck that only ever calls once."""
    calls = _count_column_calls(monkeypatch)
    z, _cur = make(**hub_deck(n_radials=4), cancel=CancelToken()).compute_impedance()
    assert calls[0] >= 3
    assert np.isfinite(np.atleast_1d(z)[0])


def _near_correction_polls(trip):
    """Solve a small buried deck with SG, recording the callers of
    `_checkpoint` that are SG's near-correction methods; with `trip` the first
    such poll cancels the token. Returns (callers, aborted)."""
    tok = CancelToken()
    s = SinusoidalGalerkinSolver(**hub_deck(n_radials=4), cancel=tok)
    callers = []
    real = s._checkpoint

    def spy():
        who = sys._getframe(1).f_code.co_name
        if who in ("_near_pairs", "_apply_near_correction"):
            callers.append(who)
            if trip and len(callers) == 1:
                tok.cancel()
        real()

    s._checkpoint = spy
    try:
        s.compute_impedance()
    except SolveAborted:
        return callers, True
    return callers, False


def test_sg_near_correction_polls_are_reached_and_stop_the_solve():
    callers, aborted = _near_correction_polls(trip=False)
    assert not aborted
    assert {"_near_pairs", "_apply_near_correction"} <= set(callers), callers
    callers, aborted = _near_correction_polls(trip=True)
    assert aborted
    assert len(callers) == 1, callers  # nothing in the near correction ran on


def test_the_ambient_token_is_restored_after_a_fill():
    tok = CancelToken()
    tok.cancel()
    _cancel.poll()  # no token: a no-op
    with _cancel.scope(tok):
        with pytest.raises(SolveAborted):
            _cancel.poll()
        with _cancel.scope(None):
            _cancel.poll()  # the inner scope hides the outer token
        with pytest.raises(SolveAborted):
            _cancel.poll()  # and restores it
    _cancel.poll()


# --------------------------------------------------------------------------
# slow: the bound itself, on the x16 decks the issue names.
# --------------------------------------------------------------------------

# Seconds from the flag to the raise. Skylake measured under 1 s for every
# fill-phase landing below; the bound carries slack because the suite runs these
# beside xdist workers pinned to a few OpenMP threads each.
BOUND_S = 2.0
_FILL_FILES = ("_crossing_fill.py", "_near_interface.py")


def _hub_x16():
    """`hub_deck(16)` with every wire's segment count x16 and the rise left at
    2, as the benchmark decks scale it (antennaknobs scratch/remeasure-1003)."""
    d = hub_deck(n_radials=16)
    npe = d["n_per_edge_per_wire"]
    d["n_per_edge_per_wire"] = [[n * 16 for n in e] for e in npe[:16]] + [
        npe[16],
        [npe[17][0] * 16],
    ]
    return d


_MAKE = {
    "razor": lambda **kw: RazorSolver(**kw, nec5_quadrature=True),
    "sg": lambda **kw: SinusoidalGalerkinSolver(**kw),
    "bs2": lambda **kw: BSplineSolver(**kw),
}


def _deck(name):
    return _hub_x16() if name == "buried" else invl_deck(n_radials=16, x=16)


def _cancel_latency(make, deck, at):
    """(seconds from the flag to the raise, the files of the innermost frames
    at the flag). The flag goes up at the first moment at or after `at` seconds
    that the solve is INSIDE the crossing fill or the near-interface tables,
    sampled every 2 ms, so every run lands in the phase under test whatever the
    machine's speed."""
    tok = CancelToken()
    solver = make(**deck, cancel=tok)
    main = threading.get_ident()
    seen = {}
    done = threading.Event()

    def trip():
        time.sleep(at)
        while not done.is_set():
            frames = traceback.extract_stack(sys._current_frames().get(main))
            files = [f.filename.rsplit("/", 1)[-1] for f in frames]
            if any(f in _FILL_FILES for f in files):
                seen["files"] = files[-8:]
                seen["t"] = time.perf_counter()
                tok.cancel()
                return
            time.sleep(0.002)

    threading.Thread(target=trip, daemon=True).start()
    try:
        solver.compute_impedance()
    except SolveAborted:
        return time.perf_counter() - seen["t"], seen["files"]
    finally:
        done.set()
    pytest.skip(f"the solve left the fill before {at} s, or never entered it")


@pytest.mark.skipif(
    not os.environ.get("MOMWIRE_CANCEL_LATENCY"),
    reason="wall-clock latency gate: CI runners spread ~3x on the same code, so "
    "it runs by hand (MOMWIRE_CANCEL_LATENCY=1), never as a CI gate",
)
@pytest.mark.slow
@pytest.mark.parametrize("at", [2.0, 5.0])
@pytest.mark.parametrize("deckname", ["buried", "invl"])
@pytest.mark.parametrize("engine", ["razor", "sg", "bs2"])
def test_a_cancel_mid_fill_returns_within_the_bound(engine, deckname, at):
    """Cancel inside the crossing fill or the near-interface tables, at or
    after `at` seconds into an x16 solve. The below/below Sommerfeld grid fill
    (`_sommerfeld_below`) takes no token at all and is a separate hole, so the
    flag is raised only once the solve is in this issue's phases."""
    latency, files = _cancel_latency(_MAKE[engine], _deck(deckname), at)
    assert latency < BOUND_S, (latency, files)


# momwire#1348 scope: the native calls inside these two phases now take the
# flag too (tests/test_cancel_native_crossing_1348.py), so the bound at x32 is
# the target the hosted watchdog asked for. Razor invl is the deck whose
# merge / factorize calls grow fastest with size (0.8 s single calls at x32
# before the flag reached them).
BOUND_X32_S = 0.2


@pytest.mark.skipif(
    not os.environ.get("MOMWIRE_CANCEL_LATENCY"),
    reason="wall-clock latency gate: runs by hand (MOMWIRE_CANCEL_LATENCY=1), "
    "never as a CI gate",
)
@pytest.mark.slow
@pytest.mark.parametrize("at", [2.0, 5.0, 10.0, 20.0])
def test_a_cancel_mid_fill_at_x32_razor_invl_returns_within_the_target(at):
    latency, files = _cancel_latency(_MAKE["razor"], invl_deck(n_radials=16, x=32), at)
    print(f"\nLATENCY razor invl x32 at={at}: {latency:.4f} s {files}")
    assert latency < BOUND_X32_S, (latency, files)
