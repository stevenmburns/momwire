"""The same-edge static-moment path is cancellable — momwire#1006.

`cancel_flag` exists so that changing a knob in the app stops the in-flight
solve and a new one is issued. Twelve entry points took one; the same-edge
static-moment path did not, and `bspline.py` calls it SEPARATELY from the
cancellable fills — so it was an uncancellable window inside an otherwise
cancellable solve.

The window is not small, and it is not the part #1006 made fast. Measured:

    N       table build (parallel, O(N))   whole call (+ gather, O(N^2))
    801                 1.3 ms                       8.3 ms
    1601                2.3 ms                      27.7 ms
    3201                4.4 ms                     107.4 ms

The O(N^2) GATHER is 85% of the call at N=801 and 96% at N=3201, and the total
grows quadratically — 420 ms at N=6401. So the poll goes in the gather; polling
only the table build would drain 4% of the wait. It polls per ROW, so the check
is O(1) per O(N) of copying and worst-case latency is one row.

What this module pins:

  * **both paths cancel** — the C++ square path and the numpy dense branch that
    a non-uniform edge falls to. A fallback that cannot be cancelled hangs the
    app on exactly the geometries the accelerator declines.
  * **the exception type is `SolveAborted`**, not `AcceleratorAborted`. The C++
    kernels raise the latter and `_accel._install_cancel_translation` remaps it
    — but only for names listed in `_CANCELLABLE_KERNELS`. A kernel that polls
    and is not listed raises a type every caller's `except SolveAborted` misses,
    which is a cancel that looks like a crash.
  * **nothing changes without a token.** The default is 0, which the kernels
    read as "no cancellation", so the ordinary path is untouched.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from momwire import CancelToken, SolveAborted
from momwire import _bspline_kernels as K

pytestmark = pytest.mark.skipif(K._acc is None, reason="accelerator not built")

BIG_N = 1601  # ~28 ms uncancelled here: long enough to interrupt, short to run


def _uniform(N):
    return np.arange(N + 1, dtype=float) * (4.0 / N)


def test_the_cpp_square_path_aborts_mid_gather():
    tok = CancelToken()
    threading.Timer(0.01, tok.cancel).start()
    with pytest.raises(SolveAborted):
        K._seg_seg_static_moments(_uniform(BIG_N), 5e-4, 2, cancel=tok)


def test_the_numpy_dense_path_aborts_too():
    """A NON-uniform edge declines the accelerator and takes the numpy branch.
    House policy is that the fallback is cancellable as well."""
    arc = np.cumsum(np.r_[0.0, np.linspace(0.01, 0.02, 600)])
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(SolveAborted):
        K._seg_seg_static_moments(arc, 5e-4, 2, cancel=tok)


def test_an_already_cancelled_token_does_not_run_the_solve():
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(SolveAborted):
        K._seg_seg_static_moments(_uniform(BIG_N), 5e-4, 2, cancel=tok)


def test_the_translation_is_registered_for_both_entries():
    """The remap is by NAME. Polling without registering raises
    `AcceleratorAborted`, which no caller catches."""
    from momwire import _accel

    for name in (
        "seg_seg_static_moments_bspline_uniform",
        "seg_seg_static_moments_bspline_uniform_ek",
    ):
        assert name in _accel._CANCELLABLE_KERNELS
        assert hasattr(K._acc, name)


def test_no_token_is_the_untouched_path():
    """0 means "no cancellation" to the kernels, so the default costs nothing
    and returns what it always did."""
    arc = _uniform(41)
    with_none = np.asarray(K._seg_seg_static_moments(arc, 5e-4, 2))
    with_live = np.asarray(
        K._seg_seg_static_moments(arc, 5e-4, 2, cancel=CancelToken())
    )
    assert np.array_equal(with_none, with_live)
