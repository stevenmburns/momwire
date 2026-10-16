"""The Sommerfeld integrands in lanes (momwire#1441 step 3).

The six-integral kernels of both Sommerfeld families evaluate their contour
integrands four Gauss nodes at a time, one node per SIMD lane:

* `somm_six_integrals_batch` (the above-ground table, `_sommerfeld`): every
  node of the 24-point rule, both the Bessel and the Hankel form;
* `below_six_integrals_batch` (the below/below table, `_sommerfeld_below`):
  the head's detour and the real-axis tail, through the contour engine's lane
  hook (`_contour_engine_inline.h`, `mw_has_lanes`).

Each lane runs the scalar walk's own operations in its order, libm's
transcendentals included, so nothing may move a bit and the gates are uint64
(`_somm_lanes_inline.h` states the contract):

- G-1441-L1  above ground, lanes vs the walk (`lanes=False`) over nodes that
             cross both forms, the series/asymptotic switch, rho = 0, h = 0,
             a lossy, a low-loss and a near-PEC soil, and two tolerances.
- G-1441-L2  below/below, the same over head-only and tail-heavy nodes,
             rho = 0 (the Bessel pair's |x| < 1e-6 branch, served by the
             scalar integrand per lane), the coarse self-convergence machine,
             and both tail rules (weighted averages and the plain walk).
- G-1441-L3  a live buried deck: every call the three kernels receive during
             `_shell.main` replayed with `lanes=False`, and the printout with
             the module switches off, byte for byte. The third kernel is the
             near-interface column rule (`near_interface_six_columns`), whose
             Bessel pairs and spectral factors run four nodes at a time too.

The SSE2 baseline build runs the same lanes in two 128-bit halves. The
portable backend (arm64) is exact too but no faster than the walk, so those
builds keep the walk unless MW_SL_FORCE_PORTABLE is defined, and the gates
hold trivially there. On Windows (MSVC, /fp:fast) the value comparisons take
the derived win32 tolerance instead (`_lane_gate.py`), and the byte and
health-array equalities are not asserted; the cp312 win_amd64 wheel smoke
runs this module.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

from _lane_gate import WIN32, assert_lanes_match
from momwire import _near_interface as near
from momwire import _sommerfeld as somm
from momwire import _sommerfeld_below as below

ACC = somm._acc
NIA = near._nia

pytestmark = pytest.mark.skipif(
    ACC is None
    or not hasattr(ACC, "somm_six_lanes_1441")
    or not hasattr(ACC, "below_six_lanes_1441")
    or NIA is None
    or not hasattr(NIA, "six_columns_lanes_1441"),
    reason="the .so predates the Sommerfeld lanes: rebuild",
)

K2 = 2.0 * np.pi  # lengths in free-space wavelengths, as the grids pass them
SOILS = [
    13.0 - 14.420993610649957j,  # Dan's soil at 6 MHz
    4.0 - 0.05j,  # low loss: k1 close to the real axis
    5.0e4 - 9.0e4j,  # |k1| > 200 k2, the PEC-limit escape
]


def _nodes():
    rho = np.array([0.0, 1e-7, 0.003, 0.02, 0.1, 0.17, 0.3, 0.5, 1.4, 3.0, 0.0, 0.25])
    h = np.array([0.05, 0.0, 0.0, 0.004, 0.2, 0.06, 0.7, 0.0, 0.3, 0.01, 2.0, 0.125])
    return rho, h


@pytest.mark.parametrize("eps_t", SOILS)
@pytest.mark.parametrize("form", [0, 1, 2])
@pytest.mark.parametrize("rtol", [1e-6, 1e-9])
def test_g1441_l1_above_ground_lanes_are_the_walk(eps_t, form, rtol):
    rho, h = _nodes()
    if form == 1:  # the Bessel form needs h > 0 to decay
        keep = h > 0
        rho, h = rho[keep], h[keep]
    if form == 2:  # the Hankel form needs rho > 0 (H0 is singular at 0)
        keep = rho > 0
        rho, h = rho[keep], h[keep]
    lanes = ACC.somm_six_integrals_batch(eps_t, K2, rho, h, rtol, form)
    walk = ACC.somm_six_integrals_batch(eps_t, K2, rho, h, rtol, form, 0, False)
    assert np.all(np.isfinite(walk))
    assert_lanes_match(lanes, walk)


def _below_nodes():
    # h = |z| + |z'| (both below), lengths in metres at the soil's own k.
    rho = np.array([0.0, 0.0, 1e-9, 0.01, 0.3, 2.0, 7.5, 25.0, 60.0, 3.0, 0.5])
    h = np.array([0.05, 1.3, 0.2, 0.002, 0.05, 0.2, 0.02, 1.0, 0.4, 3.0, 0.0001])
    return rho, h


@pytest.mark.parametrize(
    "f_hz, er, sigma",
    [(6.0078e6, 13.0, 0.005), (14.0e6, 5.0, 0.001), (1.8e6, 30.0, 0.05)],
)
@pytest.mark.parametrize("selfconv", [False, True])
@pytest.mark.parametrize("wa", [True, False])
def test_g1441_l2_below_lanes_are_the_walk(monkeypatch, f_hz, er, sigma, selfconv, wa):
    if not wa:
        monkeypatch.setattr(below, "_wa_pow", lambda: None)
    w = 2.0 * np.pi * f_hz
    eps_t = er - 1j * sigma / (w * 8.8541878128e-12)
    k_p = w / 299792458.0
    k_m = below.k_medium(eps_t, k_p)
    rho, h = _below_nodes()
    if not wa:  # the plain walk at grazing is thousands of panels: steep only
        keep = h >= 0.05
        rho, h = rho[keep], h[keep]

    def run(lanes):
        monkeypatch.setattr(below, "_SIX_LANES", lanes)
        return below._six_below_accel(k_p, k_m, rho, h, 1e-9, selfconv)

    got, ref = run(True), run(False)
    assert np.all(np.isfinite(ref[0]))
    assert_lanes_match(got[0], ref[0])
    # The health arrays are decisions taken on those values: equal outright
    # where the values are (not under MSVC's /fp:fast, see _lane_gate.py).
    if not WIN32:
        for a, b in zip(got[1:], ref[1:], strict=True):
            np.testing.assert_array_equal(a, b)


DECK = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "eznec_crossing_source_1282"
    / "dan_end1.nec"
)


def _cold_grids():
    """Empty every in-memory Sommerfeld table cache; returns the restore."""
    caches = [somm._GRID_CACHE, somm._NORM_CACHE, below._ANCHOR_CACHE]
    saved = [dict(c) for c in caches]
    for c in caches:
        c.clear()

    def restore():
        for c, s in zip(caches, saved, strict=True):
            c.clear()
            c.update(s)

    return restore


def test_g1441_l3_a_buried_deck_with_and_without_the_lanes(monkeypatch, tmp_path):
    from momwire.eznec import _shell

    monkeypatch.setenv("MOMWIRE_SOMM_CACHE", "0")
    targets = (
        (ACC, "somm_six_integrals_batch"),
        (ACC, "below_six_integrals_batch"),
        (NIA, "near_interface_six_columns"),
    )

    def solve(lanes, out):
        calls = []
        with monkeypatch.context() as mp:
            mp.setattr(somm, "_SIX_LANES", lanes)
            mp.setattr(below, "_SIX_LANES", lanes)
            mp.setattr(near, "_SIX_COLUMNS_LANES", lanes)
            for mod, name in targets:
                real = getattr(mod, name)

                def spy(*a, _real=real, _mod=mod, _name=name, **kw):
                    calls.append((_mod, _name, a, kw))
                    return _real(*a, **kw)

                mp.setattr(mod, name, spy)
            restore = _cold_grids()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rc = _shell.main([str(DECK), str(out)], basis="razor-2p")
            finally:
                restore()
        assert rc == 0
        return calls

    calls = solve(True, tmp_path / "lanes.out")
    solve(False, tmp_path / "walk.out")
    seen = {n for _, n, *_ in calls}
    assert seen == {n for _, n in targets}, seen
    if not WIN32:  # /fp:fast: the values are gated by tolerance below instead
        assert (tmp_path / "lanes.out").read_bytes() == (
            tmp_path / "walk.out"
        ).read_bytes()
    # Every call the live solve made, replayed on the walk.
    for mod, name, a, kw in calls:
        assert kw.get("lanes", True) is True
        kw = {k: v for k, v in kw.items() if k != "cache"}  # rebuild every column
        fn = getattr(mod, name)
        got = fn(*a, **kw)
        ref = fn(*a, **{**kw, "lanes": False})
        if isinstance(got, tuple):
            got, ref = got[0], ref[0]
        assert_lanes_match(got, ref)
