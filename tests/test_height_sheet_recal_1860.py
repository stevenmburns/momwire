"""The near-interface height census recalibrated (momwire#1860 L2, the opt-in
feature `height_sheet_recal`).

On the EFHW deck of #1860, the 1 cm counterpoise's rows (its nodes x the
buried rod's) were declined a height sheet by the census's row worth, and
nec5-bspline's knots put that height at three values a few ulp apart. On,
the census values a horizontal wire's rows (a fixed (x, y) group holding no
other height) at `_SHEET_HEIGHT_LONE_WORTH`, keeps a mast's at
`_SHEET_ROW_WORTH`, and snaps heights within `_SHEET_HEIGHT_SNAP_ULPS`.

What is held:

- off, nothing changes: the feature is off by default, and `sheet_plan`
  outside the scope is the shipped census (no snapped windows);
- a few-ulp run of heights is one height, its window spanning the run, and
  heights further apart stay their own;
- on, a counterpoise-shaped census takes its height and a mast-shaped one
  does not, though the mast's rows alone would pay at the lone worth;
- a row in a snapped window is served AT the window's height;
- on, the EFHW deck takes the counterpoise's sheet and its input impedance
  agrees with off.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _features, _ground_refl
from momwire import _near_interface as ni
from momwire.bspline import BSplineSolver
from momwire.razor import RazorSolver

from test_crossing_serve_524 import crossing_deck

# An experimental feature (`momwire._features`, reach 1).
pytestmark = [
    pytest.mark.reach(1),
    pytest.mark.skipif(
        not (ni._use_sheet() and ni._use_column_route() and ni._use_column_accel()),
        reason="the near-interface sheets are not built",
    ),
]

C0 = 299792458.0
F = 3.62e6
OMEGA = 2.0 * np.pi * F
K_P = OMEGA / C0
EPS_T = complex(_ground_refl.eps_tilde((13.0, 0.005), OMEGA, 8.8541878128e-12))
A_ROD = 0.008


def _line(p0, p1, n):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return np.asarray(p0, float) + t * (np.asarray(p1, float) - np.asarray(p0, float))


def _rod(n=38):
    return _line((0.0, 0.0, -3.0), (0.0, 0.0, -0.04), n)


def _counterpoise(n=624, z=0.01):
    return _line((-0.02, 0.0, z), (-4.0, 0.0, z), n)


def _plan(specs, on):
    with ni.height_sheet_recal_scope(on):
        return ni.sheet_plan(EPS_T, K_P, specs, A_ROD)


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("MOMWIRE_X_HEIGHT_SHEET_RECAL", raising=False)
    assert not _features.enabled("height_sheet_recal")
    assert not BSplineSolver(**crossing_deck())._height_sheet_recal
    assert not RazorSolver(**crossing_deck())._height_sheet_recal
    assert not ni._HEIGHT_SHEET_RECAL.get()
    plan = ni.sheet_plan(EPS_T, K_P, [(_counterpoise(), _rod())], A_ROD)
    assert plan.lo is None and plan.hi is None


def test_a_few_ulp_run_is_one_height_and_its_window_spans_it():
    h = 0.01
    lo, hi = np.nextafter(h, 0.0), np.nextafter(h, 1.0)
    far = h * (1.0 + 1e-12)
    n10 = np.nextafter(10.0, 0.0)
    z = np.array([h, lo, h, hi, h, far, 10.0, n10])
    snapped, rep, run_lo, run_hi = ni._snap_heights(z)
    # The most frequent value names its run; a tie takes the least.
    assert snapped.tolist() == [h, h, h, h, h, far, n10, n10]
    assert rep.tolist() == [h, far, n10]
    assert run_lo.tolist() == [lo, far, n10]
    assert run_hi.tolist() == [hi, far, 10.0]


def test_on_takes_the_counterpoise_height_and_off_declines_it():
    specs = [(_counterpoise(), _rod())]
    off = _plan(specs, False)
    on = _plan(specs, True)
    assert 0.01 not in off.heights.tolist()
    assert 0.01 in on.heights.tolist()
    k = on.heights.tolist().index(0.01)
    assert on.lo[k] == on.hi[k] == 0.01


def test_a_mast_keeps_the_shipped_worth():
    # Every height of a vertical over a rod's 600 depths: each height's
    # fixed group (the vertical's (x, y)) holds every other height, so its
    # rows keep `_SHEET_ROW_WORTH` -- valued at the lone worth they would pay.
    mast = _line((0.0, 0.0, 3.0), (0.0, 0.0, 10.0), 60)
    specs = [(mast, _rod(600))]
    on = _plan(specs, True)
    off = _plan(specs, False)
    assert on.heights.tolist() == off.heights.tolist()
    old = ni._SHEET_ROW_WORTH
    try:
        ni._SHEET_ROW_WORTH = ni._SHEET_HEIGHT_LONE_WORTH
        assert _plan(specs, False).heights.size > off.heights.size
    finally:
        ni._SHEET_ROW_WORTH = old


def test_a_snapped_row_is_served_at_its_window_height():
    h = 0.01
    up = np.nextafter(np.nextafter(h, 1.0), 1.0)
    cp = np.concatenate([_counterpoise(), _counterpoise(8, up)])
    plan = _plan([(cp, _rod())], True)
    k = plan.heights.tolist().index(h)
    assert plan.hi[k] == up
    k_m = ni.k_medium(EPS_T, K_P)
    rho = np.array([0.5, 1.7])
    zp = np.array([-1.2, -0.3])
    at_h = np.stack([rho, np.full(2, h), zp], axis=1)
    at_up = np.stack([rho, np.full(2, up), zp], axis=1)
    sub = np.concatenate([at_h, at_up])
    take = ni._sheet_take(sub, plan)
    assert take is not None and take[0].all()
    out = np.empty((4, ni.N_KEYS), dtype=np.complex128)
    ni._serve_from_sheets(K_P, k_m, sub, ni._LAM_MULT, take, plan, out, "six")
    assert np.array_equal(out[2:], out[:2])


EFHW_HI = """CM EFHW-like inverted L, buried 3 m rod, 4 m counterpoise at 1 cm
CE
GW 1 120 0 0 0 0 0 10 0.001
GW 2 38 0 0 0 0 0 -3 0.008
GW 3 380 0 0 10 30 0 10 0.001
GW 4 52 -0.02 0 0.01 -4 0 0.01 0.001
GE -1
GN 2 0 0 0 13 0.005
FR 0 1 0 0 3.62 0
EX 0 1 1 0 1 0
XQ
EN
"""


def _zin(text):
    for i, line in enumerate(text.splitlines()):
        if "INPUT PARAMETERS" in line:
            rows = text.splitlines()[i + 1 : i + 8]
            for r in rows:
                v = r.split()
                if len(v) >= 10 and v[0] == "1":
                    return complex(float(v[6]), float(v[7]))
    raise AssertionError("no input impedance in the printout")


# Slow lane at the 0.89.0 pre-tag check: 8.5 s median on the Linux test job.
@pytest.mark.slow
def test_on_the_efhw_deck_takes_the_counterpoise_sheet_and_zin_agrees(
    monkeypatch, tmp_path
):
    from momwire.dropin.nec5 import _shell

    monkeypatch.setenv("MOMWIRE_SOMM_CACHE_DIR", str(tmp_path))
    plans = []
    real = ni.sheet_plan

    def spy(*a, **kw):
        plans.append(real(*a, **kw))
        return plans[-1]

    monkeypatch.setattr(ni, "sheet_plan", spy)
    monkeypatch.delenv("MOMWIRE_X_HEIGHT_SHEET_RECAL", raising=False)
    off = _shell.render(EFHW_HI, basis="razor-2p", dialect="nec5")
    assert plans and all(0.01 not in p.heights.tolist() for p in plans)
    plans.clear()
    monkeypatch.setenv("MOMWIRE_X_HEIGHT_SHEET_RECAL", "1")
    on = _shell.render(EFHW_HI, basis="razor-2p", dialect="nec5")
    assert plans and all(0.01 in p.heights.tolist() for p in plans)
    z_off, z_on = _zin(off), _zin(on)
    assert abs(z_on - z_off) <= 1e-9 * abs(z_off)
