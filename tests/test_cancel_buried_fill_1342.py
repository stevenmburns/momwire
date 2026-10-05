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
  * (slow) a cancel in the middle of an x16 fill returns within a bound.
"""

from __future__ import annotations

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
    the solve was in when the flag went up)."""
    tok = CancelToken()
    solver = make(**deck, cancel=tok)
    main = threading.get_ident()
    seen = {}

    def trip():
        time.sleep(at)
        frames = traceback.extract_stack(sys._current_frames().get(main))
        seen["files"] = [f.filename.rsplit("/", 1)[-1] for f in frames[-8:]]
        seen["t"] = time.perf_counter()
        tok.cancel()

    threading.Thread(target=trip, daemon=True).start()
    try:
        solver.compute_impedance()
    except SolveAborted:
        return time.perf_counter() - seen["t"], seen["files"]
    pytest.skip(f"the solve finished before the flag at {at} s")


@pytest.mark.slow
@pytest.mark.parametrize("at", [2.0, 5.0])
@pytest.mark.parametrize("deckname", ["buried", "invl"])
@pytest.mark.parametrize("engine", ["razor", "sg", "bs2"])
def test_a_cancel_mid_fill_returns_within_the_bound(engine, deckname, at):
    """Cancel at `at` seconds into the solve. Asserted when the flag lands in
    the crossing fill or the near-interface tables, which are this issue's.
    The below/below Sommerfeld grid fill (`_sommerfeld_below`) takes no token
    at all and is a separate hole (a cancel at 2 s into razor's buried x16
    waits out that fill), so a landing there skips rather than fails."""
    latency, files = _cancel_latency(_MAKE[engine], _deck(deckname), at)
    if not any(f in _FILL_FILES for f in files):
        pytest.skip(f"the flag landed outside the buried fill: {files}")
    assert latency < BOUND_S, (latency, files)
