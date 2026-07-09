"""Phase 0 regression guard: the long C++ accelerator kernels must release the
Python GIL during their compute region.

Motivation: the pybind11 kernels used to hold the GIL for their entire native
call. While a fill ran, no other Python thread could make progress — freezing
the asyncio event loop of the web server for the full duration of every matrix
fill. Each heavy kernel now wraps its OpenMP compute region in
`py::gil_scoped_release`, so concurrent Python threads run while it computes.

What we measure — and why not raw throughput: a background thread that merely
counts is a poor probe here, because pybind11's argument marshalling of the
large input arrays releases the GIL on its own, so *some* concurrent progress
happens even when the compute region holds it. The bug this guards against is
an event-loop *freeze*: a single uninterruptible stretch where no other thread
runs for the whole compute. So we run a watcher thread that timestamps itself
in a tight loop and look at the largest gap between consecutive samples — the
longest interval it was starved — during one big native fill.

- GIL held for the compute region (the old behavior): the watcher is frozen for
  ~the entire fill, so max_gap ≈ fill_duration (ratio ≈ 1).
- GIL released (the fix): the watcher is only ever blocked for ~one CPython
  switch interval (5 ms) while reacquiring, so max_gap ≪ fill_duration
  (ratio ~0.003 observed).

The assertion is *relative* to the measured fill duration so it is independent
of machine speed and core count. This MUST call a bare `_accelerators` kernel
directly, not a high-level solver: `compute_impedance` interleaves Python/numpy
glue between kernel calls (numpy drops the GIL itself), which would mask a
kernel that holds the GIL — a false pass.
"""

import threading
import time

import numpy as np
import pytest

accel = pytest.importorskip("momwire._accelerators")


def test_seg_seg_full_moments_bspline_swept_releases_gil():
    # N large enough that one native call runs for tens of ms — long enough for
    # a frozen watcher (GIL held) to be plainly distinguishable from a
    # cooperatively-yielding one (GIL released). Probes the batched swept
    # off-edge moments kernel (the long fill of the batched sweep path).
    N = 500
    rng = np.random.default_rng(0)
    seg_l = rng.standard_normal((N, 3))
    seg_r = seg_l + 0.01
    gx, gw = np.polynomial.legendre.leggauss(8)
    gl_t = (gx + 1.0) / 2.0
    gl_w = gw / 2.0
    k_array = np.array([1.0, 2.0, 3.0, 4.0])

    samples = []
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            samples.append(time.perf_counter())

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    try:
        time.sleep(0.02)  # let the watcher ramp up on its own core
        i0 = len(samples)
        t0 = time.perf_counter()
        accel.seg_seg_full_moments_bspline_swept(
            seg_l, seg_r, seg_l, seg_r, 1e-6, k_array, 2, gl_t, gl_w
        )
        fill = time.perf_counter() - t0
    finally:
        stop.set()
        thread.join()

    gaps = np.diff(samples[i0:])
    max_gap = float(gaps.max()) if len(gaps) else fill

    # Sanity: the fill must be substantial or the comparison is vacuous. If this
    # trips, the geometry is too small to exercise a meaningful compute region.
    assert fill > 0.02, f"fill too fast ({fill * 1e3:.1f} ms) to be a valid probe"

    # The real assertion: the watcher was never starved for anything close to
    # the whole fill. Held-GIL gives ratio ≈ 1; the fix gives ~0.003. 0.4 sits
    # far from both, so this is robust to machine speed and scheduler jitter.
    ratio = max_gap / fill
    assert ratio < 0.4, (
        f"watcher stalled for {max_gap * 1e3:.1f} ms during a {fill * 1e3:.1f} ms "
        f"fill (ratio {ratio:.2f}); the kernel appears to hold the GIL for its "
        "whole compute region"
    )
