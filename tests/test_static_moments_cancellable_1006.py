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

import numpy as np
import pytest

from momwire import CancelToken, SolveAborted
from momwire import _bspline_kernels as K

pytestmark = pytest.mark.skipif(K._acc is None, reason="accelerator not built")

BIG_N = 1601  # a realistic size for the already-cancelled check


def _uniform(N):
    return np.arange(N + 1, dtype=float) * (4.0 / N)


PROBE_N = 41
PROBE_MAX_D = 2
# The gather polls once per row of every (p, q) moment block.
PROBE_TOTAL_POLLS = (PROBE_MAX_D + 1) ** 2 * PROBE_N


def _probed_gather(tok, probe):
    """Run the C++ square entry with its poll probe armed (momwire#1158).

    The kernel reads the token's flag as raw memory and never calls back into
    Python, so a Python token that counts its own reads would count nothing.
    The probe lives in the kernel instead: `probe[0]` is the poll on which it
    raises the flag (0 = never), `probe[1]` counts the flag reads the gather
    made. Called on `_acc` directly, with the arguments `_seg_seg_static_moments`
    passes, so the `_accel` SolveAborted translation still wraps it.
    """
    return K._acc.seg_seg_static_moments_bspline_uniform(
        4.0 / PROBE_N,
        5e-4,
        PROBE_N,
        PROBE_MAX_D,
        tok.ptr,
        poll_probe=probe.ctypes.data,
    )


def test_the_cpp_square_path_aborts_mid_gather():
    """Deterministic: the flag rises on a known poll, halfway through the
    gather, and the abort lands on exactly that poll. This replaced a 10 ms
    `threading.Timer` race that the macOS runner lost (momwire#1158)."""
    flip_at = PROBE_TOTAL_POLLS // 2
    tok = CancelToken()
    probe = np.array([flip_at, 0], dtype=np.int64)
    with pytest.raises(SolveAborted):
        _probed_gather(tok, probe)
    polls = int(probe[1])
    assert 1 < polls < PROBE_TOTAL_POLLS
    assert polls == flip_at, "the abort is seen at the poll that raised it"
    assert tok.cancelled


def test_the_probe_counts_every_row_when_nothing_cancels():
    """The denominator above: with the probe armed but never firing, the
    gather reads the flag once per row of every moment block and returns
    the same answer as the unprobed call."""
    tok = CancelToken()
    probe = np.array([0, 0], dtype=np.int64)
    probed = np.asarray(_probed_gather(tok, probe))
    assert int(probe[1]) == PROBE_TOTAL_POLLS
    assert not tok.cancelled
    plain = np.asarray(K._seg_seg_static_moments(_uniform(PROBE_N), 5e-4, PROBE_MAX_D))
    assert np.array_equal(probed, plain)


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
