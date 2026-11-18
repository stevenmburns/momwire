"""A sweep's near-interface sheets blended across the band (the opt-in feature
`crossing_sheet_band`, prototype of 2026-11-17).

With the feature on, each plane's and height's sheet is built exact at the
first frequency a process asks for it (the SEED: the flag-off sheet itself)
and at n_a - 1 more Chebyshev-Lobatto points of a band W wide in frequency;
the other frequencies in the band read the Lagrange blend of the anchors'
node values in t = ln k_p (`_near_interface._band_sheet`, C++
`ni_eval::band_sheet`).

What is held:

- off, nothing changes: the feature is off by default, and a solver's fill
  without it never reaches the band code;
- on, a one-frequency solve is off's to the bit (the seed IS the shipped
  sheet), and so is a sweep's first frequency;
- a blended sheet against the exact one at the same medium on the same grid
  converges as n_a grows, and the C++ blend is the Python route's to the bit;
- on, a swept hub's input impedance agrees with off within the declared
  drift, and the route proof: blends were made;
- a sweep whose step is wider than the band never blends, and is off's to
  the bit;
- reach 7: the executable's printout is the launcher's, byte for byte, with
  the feature on (MOMWIRE_RAZOR2P_STANDALONE_EXE), and its full-precision
  Zin (MW_EXE_ARRAYS) on against off is within the declared drift on the
  synthetic buried-radial vertical swept over 3 %.
"""

from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from momwire import _accel, _features
from momwire import _near_interface as ni
from momwire.razor import RazorSolver

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
EXE = os.environ.get("MOMWIRE_RAZOR2P_STANDALONE_EXE")


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
    for var in ("MOMWIRE_SHEET_BAND_NODES", "MOMWIRE_SHEET_BAND_WIDTH"):
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
    assert np.array_equal(cx_exact, py_exact)
    assert np.array_equal(cx_blend, py_blend), np.max(np.abs(cx_blend - py_blend))
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


# --- the executables (reach 7) -----------------------------------------------


def radial_vertical_nec(nfr=21, step=0.01065, f0=6.9935):
    """A synthetic NEC-5 deck (not a user's model): four 54-segment radials
    at 0.15 m under a 13 / 0.005 S/m Sommerfeld ground, a vertical crossing
    the interface with short segments near z = 0, fed just above it."""
    a = 5e-4
    lines = ["CM synthetic buried radial vertical (crossing_sheet_band)", "CE"]
    t = 1
    for r in range(4):
        ph = 2 * math.pi * r / 4
        x, y = 6.333643 * math.cos(ph), 6.333643 * math.sin(ph)
        x = 0.0 if abs(x) < 1e-12 else x
        y = 0.0 if abs(y) < 1e-12 else y
        lines.append(f"GW {t} 54 0 0 -0.15 {x:.6E} {y:.6E} -0.15 {a}")
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


def _exe_zin(tmp_path, tag, env_on, text):
    d = tmp_path / tag
    d.mkdir()
    (d / "deck.nec").write_text(text, newline="")
    (d / "arrays").mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("MOMWIRE_X_")}
    env.update(env_on)
    env.update({"MOMWIRE_SOMM_CACHE": "0", "MW_EXE_ARRAYS": str(d / "arrays")})
    r = subprocess.run(
        [EXE, "deck.nec", "deck.out"], cwd=d, env=env, capture_output=True, check=False
    )
    assert r.returncode == 0, r.stderr
    z = []
    for i in range(10_000):
        p = d / "arrays" / f"{i}_i_source.bin"
        if not p.exists():
            break
        (i_src,) = np.fromfile(p, dtype=np.complex128)
        z.append(1.0 / complex(i_src))
    return np.asarray(z), (d / "deck.out").read_bytes()


RETUNE = {"MOMWIRE_X_CROSSING_SHEET_RETUNE": "1"}
BAND = {**RETUNE, "MOMWIRE_X_CROSSING_SHEET_BAND": "1"}


@pytest.mark.skipif(
    not EXE, reason="MOMWIRE_RAZOR2P_STANDALONE_EXE does not name a built executable"
)
@pytest.mark.slow
def test_the_executable_moves_within_the_drift(tmp_path):
    text = radial_vertical_nec(nfr=8, step=0.0304)
    z_off, out_off = _exe_zin(tmp_path, "off", RETUNE, text)
    z_on, out_on = _exe_zin(tmp_path, "on", BAND, text)
    assert len(z_on) == len(z_off) == 8
    assert z_on[0] == z_off[0]
    drift = np.max(np.abs(z_on - z_off) / np.abs(z_off))
    print(f"exe drift {drift:.3e}")
    assert 0.0 < drift <= _features.FEATURES[NAME].drift, drift
    assert b"+crossing_sheet_band" in out_on.split(b"\n")[1]
    assert b"+crossing_sheet_band" not in out_off.split(b"\n")[1]


def _launcher():
    import shutil
    import sys

    name = "momwire-nec5-razor-2p"
    found = shutil.which(name) or str(Path(sys.executable).parent / name)
    assert Path(found).exists(), found
    return found


@pytest.mark.skipif(
    not EXE, reason="MOMWIRE_RAZOR2P_STANDALONE_EXE does not name a built executable"
)
@pytest.mark.slow
def test_the_executable_answers_as_the_launcher_with_the_band_on(tmp_path):
    text = radial_vertical_nec(nfr=5, step=0.05)
    outs = []
    for tag, prog in (("exe", EXE), ("launcher", _launcher())):
        d = tmp_path / tag
        d.mkdir()
        (d / "deck.nec").write_text(text, newline="")
        env = {k: v for k, v in os.environ.items() if not k.startswith("MOMWIRE_X_")}
        env.update({**BAND, "MOMWIRE_SOMM_CACHE": "0"})
        r = subprocess.run(
            [prog, "deck.nec", "deck.out"],
            cwd=d,
            env=env,
            capture_output=True,
            check=False,
        )
        assert r.returncode == 0, r.stderr
        outs.append((d / "deck.out").read_bytes())
    assert outs[0] == outs[1]
    assert b"+crossing_sheet_band" in outs[0].split(b"\n")[1]
