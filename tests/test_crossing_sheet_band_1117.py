"""A sweep's near-interface sheets blended across the band (the opt-in feature
`crossing_sheet_band`).

With the feature on, each plane's and height's sheet is built exact at the
first frequency a process asks for it (the SEED: the flag-off sheet itself)
and at n - 1 more Chebyshev-Lobatto points of a band W wide in frequency;
the other frequencies in the band read the Lagrange blend of the anchors'
node values in t = ln k_p (`_near_interface._band_sheet`, C++
`ni_eval::band_sheet`). The plan measures each band against one more exact
sheet where the node polynomial peaks, and takes anchors until the estimate
is under MOMWIRE_SHEET_BAND_TOL, or declines and builds the band exact.

What is held here (the Python route, and the C++ blend against it):

- off, nothing changes: the feature is off by default, a solver's fill
  without it never reaches the band code, and a sheet built without a band
  grid lays out its strips on its own medium, as before the feature;
- on, a one-frequency solve is off's to the bit (the seed IS the shipped
  sheet), and so is a sweep's first frequency;
- a blended sheet against the exact one at the same medium on the same grid
  converges as n grows, and the C++ blend is the Python route's (to the bit
  off Windows; tests/_lane_gate.py's derived tolerance on win32);
- on, a swept hub's input impedance agrees with off within the declared
  drift, and the route proof: blends were made;
- a seed that already holds more strips than the anchors (a wider sheet of
  the same plane cached by an earlier solve) blends, rather than failing;
- the plan: a high-frequency band takes more anchors than it starts with,
  and a tolerance nothing meets declines every band, which is off's to the
  bit;
- a process holds at most `_BAND_MAX_FAMILIES` families;
- a sweep whose step is wider than the band never blends, and is off's to
  the bit.

The executables (reach 7) are held in tests/test_standalone_nopython.py, on
`radial_vertical_nec` below, under the fixture gates' platform rule.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from momwire import _accel, _features
from momwire import _near_interface as ni
from momwire.razor import RazorSolver

from _lane_gate import assert_lanes_match
from test_crossing_serve_524 import hub_deck

pytestmark = [
    pytest.mark.reach(7),
    pytest.mark.filterwarnings("ignore"),
    pytest.mark.skipif(
        not (ni._use_sheet() and ni._use_column_route() and ni._use_column_accel()),
        reason="the near-interface sheets are not built",
    ),
]

ACC = _accel.acc
HAVE_CXX = ACC is not None and hasattr(ACC, "near_interface_eval_sheet_band_check")
NAME = "crossing_sheet_band"
C0 = 299792458.0
EPS0 = 8.8541878128e-12


def _medium(f):
    k_p = 2.0 * np.pi * f / C0
    eps_t = complex(13.0, -0.005 / (2.0 * np.pi * f * EPS0))
    return k_p, ni.k_medium(eps_t, k_p)


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    """Each test starts with no cached sheet, no band family and zero
    counters."""
    monkeypatch.setattr(ni, "_SHEET_CACHE", type(ni._SHEET_CACHE)())
    monkeypatch.setattr(ni, "_SHEET_STATS", dict.fromkeys(ni._SHEET_STATS, 0))
    monkeypatch.setattr(ni, "_BAND_FAMILIES", [])
    monkeypatch.setattr(ni, "_BAND_STATS", dict.fromkeys(ni._BAND_STATS, 0))
    for var in (
        "MOMWIRE_SHEET_BAND_NODES",
        "MOMWIRE_SHEET_BAND_WIDTH",
        "MOMWIRE_SHEET_BAND_TOL",
    ):
        monkeypatch.delenv(var, raising=False)


def test_the_registry_entry():
    f = _features.FEATURES[NAME]
    assert f.kind == "native" and f.reach == 7 and f.kwarg == NAME
    assert f.env == "MOMWIRE_X_CROSSING_SHEET_BAND"
    assert not f.default and not f.offer
    assert 0.0 < f.drift <= _features.DEFAULT_DRIFT_MAX
    assert f.paths == ("tests/test_crossing_sheet_band_1117.py",)


def test_off_by_default_and_on_when_asked(monkeypatch):
    monkeypatch.delenv(_features.FEATURES[NAME].env, raising=False)
    assert not _features.enabled(NAME)
    assert not RazorSolver(**hub_deck())._crossing_sheet_band
    assert RazorSolver(**hub_deck(), crossing_sheet_band=True)._crossing_sheet_band
    monkeypatch.setenv(_features.FEATURES[NAME].env, "1")
    assert RazorSolver(**hub_deck())._crossing_sheet_band
    assert not RazorSolver(**hub_deck(), crossing_sheet_band=False)._crossing_sheet_band
    assert not ni._CROSSING_SHEET_BAND.get()


def _band_check_py(f0, f, zp, rho_max, s_max, n_a, w):
    """`ni_eval::sheet_band_check` on the Python route: the worst
    |blend - exact| over each key's largest |exact|, and the blended and
    exact node values in the first anchor's layout."""
    k_p0, k_m0 = _medium(f0)
    k_p, k_m = _medium(f)
    er, ei = ni._band_eps(k_p0, k_m0)
    s = ei * k_p0
    t0, t = math.log(k_p0), math.log(k_p)
    ts = ni._band_points(t0, 1 if t >= t0 else -1, n_a, w)
    grid = (k_p0, complex(k_m0))
    anchors = []
    for j, tj in enumerate(ts):
        kp = k_p0 if j == 0 else math.exp(tj)
        km = complex(k_m0) if j == 0 else ni._band_k_m(er, s, kp, complex(k_m0))
        a = ni.PlaneSheet(kp, km, zp, ni._LAM_MULT, grid=grid)
        a.cover(rho_max, s_max)
        anchors.append(a)
    ex = ni.PlaneSheet(k_p, k_m, zp, ni._LAM_MULT, grid=grid)
    ex.cover(rho_max, s_max)
    bl = ni.PlaneSheet(k_p, k_m, zp, ni._LAM_MULT, grid=grid)
    bl.blend(anchors, ni._band_weights(ts, t))
    pp = ni._SHEET_P * ni._SHEET_P
    flat = np.concatenate(ex._vals)
    e = np.concatenate(
        [
            flat[ex._off[k] : ex._off[k] + pp]
            for k in sorted(anchors[0]._off, key=anchors[0]._off.get)
        ]
    )
    b = np.concatenate(bl._vals)
    scale = np.max(np.abs(e), axis=0)
    return float(np.max(np.abs(b - e) / scale)), b.ravel(), e.ravel()


# The target deck's radial plane at 7 MHz (a 4-radial screen at 0.15 m under
# a vertical to 10.7 m), and a frequency 2.3 % up.
F0, F1 = 6.9935e6, 7.155e6
PLANE = dict(zp=-0.15, rho_max=6.34, s_max=10.71)


@pytest.mark.slow
def test_a_blend_converges_in_its_anchors():
    """The unit-level check: a blended sheet against the exact one at the same
    medium, each on the band's grid, at n_a = 2..6 over a 5 % band. The
    error falls by an order or more per anchor (measured 2026-11-17: 3.7e-3,
    4.0e-5, 7.3e-7, 5.9e-9, 1.1e-10 of each key's largest node, worst of
    four in-band frequencies)."""
    errs = [_band_check_py(F0, F1, n_a=n, w=0.05, **PLANE)[0] for n in range(2, 7)]
    print("n_a=2..6:", " ".join(f"{e:.2e}" for e in errs))
    assert all(b < a / 5 for a, b in zip(errs, errs[1:])), errs
    assert errs[2] < 1e-6, errs


@pytest.mark.skipif(not HAVE_CXX, reason="needs the extension's TEST-ONLY hook")
@pytest.mark.parametrize("n_a", [3, 4])
def test_the_cxx_blend_is_pythons(n_a):
    """The C++ route's anchors and blend against the Python route's, at a
    plane and a reach small enough to run in the PR lane: the same exact
    nodes and the same blended nodes, to the bit."""
    small = dict(zp=-0.15, rho_max=1.0, s_max=1.5)
    k_p0, k_m0 = _medium(F0)
    k_p, k_m = _medium(F1)
    cx, cx_blend, cx_exact = ACC.near_interface_eval_sheet_band_check(
        k_p0,
        k_m0,
        k_p,
        k_m,
        small["zp"],
        False,
        -1.0,
        small["rho_max"],
        small["s_max"],
        n_a,
        0.05,
    )
    py, py_blend, py_exact = _band_check_py(F0, F1, n_a=n_a, w=0.05, **small)
    # Bit-equal off Windows. On win32, MSVC's /fp:fast reassociates the C++
    # sums, so the house's derived tolerance applies (tests/_lane_gate.py),
    # as crossing_sheet_retune's C++-vs-Python sheet values do.
    assert_lanes_match(np.asarray(cx_exact), py_exact, f"exact nodes, n_a={n_a}")
    assert_lanes_match(np.asarray(cx_blend), py_blend, f"blended nodes, n_a={n_a}")
    assert 0.0 < py < 1e-3


def _hub_sweep(freqs, on, **kw):
    out = []
    for f in freqs:
        s = RazorSolver(
            **hub_deck(n_radials=8, wavelength=C0 / f, **kw),
            crossing_sheet_retune=True,
            crossing_sheet_band=on,
        )
        out.append(complex(s.compute_impedance()[0]))
    return np.asarray(out)


def test_one_frequency_on_is_off_to_the_bit():
    off = _hub_sweep([7.0e6], False)
    ni._SHEET_CACHE.clear()
    on = _hub_sweep([7.0e6], True)
    assert ni._BAND_STATS["blends"] == 0
    assert ni._BAND_STATS["families"] >= 1
    assert np.array_equal(off, on), (off, on)


@pytest.mark.slow
def test_a_swept_hub_moves_within_the_drift_and_blends():
    """Five frequencies over 3 %: the first is off's to the bit (its sheets
    are the seeds), the rest blended, each within the declared drift."""
    freqs = [7.0e6 * (1.0 + 0.0075 * i) for i in range(5)]
    off = _hub_sweep(freqs, False)
    ni._SHEET_CACHE.clear()
    on = _hub_sweep(freqs, True)
    assert ni._BAND_STATS["blends"] >= len(freqs) - 1, ni._BAND_STATS
    assert on[0] == off[0]
    drift = np.max(np.abs(on - off) / np.abs(off))
    print(f"hub8 drift {drift:.3e}")
    assert 0.0 < drift <= _features.FEATURES[NAME].drift, drift


@pytest.mark.slow
def test_a_sweep_wider_than_the_band_never_blends():
    freqs = [7.0e6, 7.7e6, 8.47e6]  # 10 % steps, W = 5 %
    off = _hub_sweep(freqs, False)
    ni._SHEET_CACHE.clear()
    on = _hub_sweep(freqs, True)
    assert ni._BAND_STATS["blends"] == 0
    assert np.array_equal(off, on), (off, on)


def _scaled_hub(q, n_radials=8, wavelength=C0 / 7.0e6):
    """`hub_deck` with its radials q m long instead of 5 m."""
    d = hub_deck(n_radials=n_radials, wavelength=wavelength)
    for i in range(n_radials):
        d["wires"][i] = d["wires"][i] * np.array([q / 5.0, q / 5.0, 1.0])
    return d


def _zin(build, on):
    s = RazorSolver(**build, crossing_sheet_retune=True, crossing_sheet_band=on)
    return complex(s.compute_impedance()[0])


@pytest.mark.slow
def test_a_seed_wider_than_its_anchors_blends():
    """The seed is the cached sheet, with every strip an earlier solve grew
    it to: here a hub with 10 m radials, band off, at the frequency the band
    is then seeded at by the 5 m hub. The anchors grow to the seed's extent,
    so the second frequency blends (it raised KeyError before) and moves
    within the declared drift of off."""
    f0, f1 = 7.0e6, 7.0525e6
    _zin(_scaled_hub(10.0), False)
    seeded = dict(ni._SHEET_CACHE)
    assert seeded
    on = [_zin(_scaled_hub(5.0, wavelength=C0 / f), True) for f in (f0, f1)]
    assert ni._BAND_STATS["blends"] >= 1, ni._BAND_STATS
    # the wider sheet seeded the band: its strips outnumber what 5 m asks for
    wide = [
        f
        for f in ni._BAND_FAMILIES
        if f.dir != 0
        and len(f.seed.strips["rho"])
        > len(ni._sheet_axis(f.k_p0, f.k_m0, abs(f.zp), "rho", f.rho_reach))
    ]
    assert wide, [(f.dir, len(f.seed.strips["rho"])) for f in ni._BAND_FAMILIES]
    for f in wide:
        for a in [*f.anchors, f.check]:
            assert a.strips == f.seed.strips
    ni._SHEET_CACHE.clear()
    ni._BAND_FAMILIES.clear()
    off = [_zin(_scaled_hub(5.0, wavelength=C0 / f), False) for f in (f0, f1)]
    assert on[0] == off[0]
    drift = abs(on[1] - off[1]) / abs(off[1])
    assert drift <= _features.FEATURES[NAME].drift, drift


@pytest.mark.slow
def test_a_tolerance_nothing_meets_declines_to_off(monkeypatch):
    """A band whose plan cannot meet MOMWIRE_SHEET_BAND_TOL declines: every
    frequency of it is built exact, so the sweep is off's to the bit."""
    monkeypatch.setenv("MOMWIRE_SHEET_BAND_TOL", "1e-300")
    freqs = [7.0e6, 7.0525e6]
    off = _hub_sweep(freqs, False)
    ni._SHEET_CACHE.clear()
    on = _hub_sweep(freqs, True)
    assert ni._BAND_STATS["declined"] >= 1, ni._BAND_STATS
    assert ni._BAND_STATS["blends"] == 0, ni._BAND_STATS
    assert all(f.declined for f in ni._BAND_FAMILIES if f.dir != 0)
    assert np.array_equal(off, on), (off, on)


@pytest.mark.slow
def test_the_plan_takes_anchors_where_the_band_is_long_in_ground_wavelengths():
    """At 28 MHz a 5 % band is four times longer in the ground's wavelengths
    than at 7, so the four anchors a band starts with do not meet the
    tolerance where they do at 7 MHz, and the plan takes more (measured
    2026-11-17 on the 10 m hub: 5 anchors at 7 MHz, 6 at 28). Both stay within the declared drift (`_features`)."""
    took = {}
    for f0 in (7.0e6, 28.0e6):
        ni._SHEET_CACHE.clear()
        ni._BAND_FAMILIES.clear()
        on = [_zin(_scaled_hub(10.0, wavelength=C0 / f), True) for f in (f0, f0 * 1.02)]
        took[f0] = max(len(f.ts) for f in ni._BAND_FAMILIES if f.dir != 0)
        assert all(f.err <= ni._band_tol() for f in ni._BAND_FAMILIES if f.dir != 0)
        ni._SHEET_CACHE.clear()
        ni._BAND_FAMILIES.clear()
        off = [
            _zin(_scaled_hub(10.0, wavelength=C0 / f), False) for f in (f0, f0 * 1.02)
        ]
        drift = abs(on[1] - off[1]) / abs(off[1])
        assert drift <= _features.FEATURES[NAME].drift, (f0, drift)
    assert took[28.0e6] > took[7.0e6] >= 4, took


@pytest.mark.slow
def test_a_process_holds_at_most_its_families(monkeypatch):
    """Band state is bounded in a resident process: past `_BAND_MAX_FAMILIES`
    (or `_BAND_MAX_VALUES`) the oldest family is dropped, and what a request
    reads stays within the drift."""
    monkeypatch.setattr(ni, "_BAND_MAX_FAMILIES", 1)
    for depth in (0.15, 0.3, 0.45):
        for f in (7.0e6, 7.0525e6):
            RazorSolver(
                **hub_deck(n_radials=8, depth=depth, wavelength=C0 / f),
                crossing_sheet_retune=True,
                crossing_sheet_band=True,
            ).compute_impedance()
            assert len(ni._BAND_FAMILIES) <= 1, len(ni._BAND_FAMILIES)
    assert ni._BAND_STATS["families"] >= 3, ni._BAND_STATS
    monkeypatch.setattr(ni, "_BAND_MAX_FAMILIES", 32)
    monkeypatch.setattr(ni, "_BAND_MAX_VALUES", 0)
    _hub_sweep([7.0e6, 7.0525e6], True)
    # a family that holds nothing of its own (a seed alone) costs nothing
    holding = [f for f in ni._BAND_FAMILIES if f.held() > 0]
    assert len(holding) <= 1, len(holding)


def test_a_sheet_without_a_band_grid_is_laid_out_on_its_own_medium():
    """Flag off, the band's grid medium is the sheet's own: its strips are
    `_sheet_axis` of its own (k_p, k_m), its cache key carries no grid, and
    its nodes are the ones a sheet given its own medium as the grid holds,
    to the bit."""
    k_p, k_m = _medium(F1)
    a = ni.PlaneSheet(k_p, k_m, -0.15, ni._LAM_MULT)
    assert a.grid == (a.k_p, a.k_m)
    a.cover(2.0, 3.0)
    for axis, reach in (("rho", 2.0), ("s", 3.0)):
        assert a.strips[axis] == ni._sheet_axis(k_p, k_m, 0.15, axis, reach)
    b = ni.PlaneSheet(k_p, k_m, -0.15, ni._LAM_MULT, grid=(a.k_p, a.k_m))
    b.cover(2.0, 3.0)
    assert a.strips == b.strips and a._off == b._off
    assert np.array_equal(np.concatenate(a._vals), np.concatenate(b._vals))
    sh = ni._plane_sheet(k_p, k_m, -0.15, ni._LAM_MULT, 2.0, 3.0)
    assert sh.grid == (sh.k_p, sh.k_m)
    (key,) = [k for k, v in ni._SHEET_CACHE.items() if v is sh]
    assert len(key) == 13 and key[0] == "grid", key


# --- the executables' deck (tests/test_standalone_nopython.py) ---------------


def radial_vertical_nec(nfr=21, step=0.01065, f0=6.9935, radial=6.333643, n_radial=54):
    """A synthetic NEC-5 deck (not a user's model): four radials at 0.15 m
    under a 13 / 0.005 S/m Sommerfeld ground, a vertical crossing the
    interface with short segments near z = 0, fed just above it."""
    a = 5e-4
    lines = ["CM synthetic buried radial vertical (crossing_sheet_band)", "CE"]
    t = 1
    for r in range(4):
        ph = 2 * math.pi * r / 4
        x, y = radial * math.cos(ph), radial * math.sin(ph)
        x = 0.0 if abs(x) < 1e-12 else x
        y = 0.0 if abs(y) < 1e-12 else y
        lines.append(f"GW {t} {n_radial} 0 0 -0.15 {x:.6E} {y:.6E} -0.15 {a}")
        t += 1
    ex = None
    for z1, z2, n in (
        (-0.15, -0.05, 2),
        (-0.05, -0.0125, 2),
        (-0.0125, 0.0, 2),
        (0.0, 0.05, 4),
        (0.05, 0.0625, 2),
        (0.0625, 0.1, 2),
        (0.1, 0.25, 2),
        (0.25, 0.85, 2),
        (0.85, 3.25, 2),
        (3.25, 10.55607, 15),
    ):
        if z1 == 0.0:
            ex = t
        lines.append(f"GW {t} {n} 0 0 {z1} 0 0 {z2} {a}")
        t += 1
    lines += [
        "GE -1 0",
        "GN 0 0 0 0 13 0.005 1 0 NOFILE",
        f"EX 0 {ex} 2 2 1 0",
        f"FR 0 {nfr} 0 0 {f0} {step}",
        "XQ 0",
        "EN",
        "",
    ]
    return "\n".join(lines)
