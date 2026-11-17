"""The near-interface sheets' build weights retuned (momwire#1956, the opt-in
feature `crossing_sheet_retune`).

The census takes a plane (or height) sheet when the rows it would serve pay
for its nodes at `_SHEET_BUILD_WEIGHT` (`_SHEET_HEIGHT_WEIGHT`) of a column
setup each. The 0.15 both carry was measured with every sheet built from
scratch; with the strips cached (a warm process, the hosted app's) a node
costs far less, and on the crossing decks of #1956 a lower weight takes the
radials' plane the shipped census declines. On, the census reads
`_SHEET_RETUNE_BUILD_WEIGHT` and `_SHEET_RETUNE_HEIGHT_WEIGHT`, constants:
the plan is still never decided on what is cached.

What is held:

- off, nothing changes: the feature is off by default, and outside a fill's
  scope the census reads the shipped weights (and the tests' handle on them);
- on, a vertical over a radial screen takes the plane off declines;
- the C++ fills' plan and values are the Python route's to the bit, on and
  off (the TEST-ONLY `near_interface_eval_designed`);
- on, a hub deck's input impedance agrees with off within the declared drift
  on both formulations, and the plan did change; so does the deep, high-
  frequency hub where the drift was measured largest (razor-2p).
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _accel, _features
from momwire import _near_interface as ni
from momwire.bspline import BSplineSolver
from momwire.razor import RazorSolver

from test_crossing_serve_524 import _radial_dirs, crossing_deck, hub_deck

# An experimental feature (`momwire._features`, reach 1).
pytestmark = [
    pytest.mark.reach(1),
    # The decks are the fill tests' shapes, not converged designs: their
    # advisories (a coarse crossing node, an ambiguous feed site) are not
    # what is held here.
    pytest.mark.filterwarnings("ignore"),
    pytest.mark.skipif(
        not (ni._use_sheet() and ni._use_column_route() and ni._use_column_accel()),
        reason="the near-interface sheets are not built",
    ),
]

ACC = _accel.acc
HAVE_CXX = ACC is not None and hasattr(ACC, "near_interface_eval_designed")
NAME = "crossing_sheet_retune"

F = 7e6
K_P = 2.0 * np.pi * F / 299792458.0
EPS_T = complex(13.0, -0.005 / (2.0 * np.pi * F * 8.8541878128e-12))
K_M = ni.k_medium(EPS_T, K_P)
A = 0.001


@pytest.fixture(autouse=True)
def _fresh_sheets(monkeypatch):
    """Each test starts with no cached sheet and zero counters."""
    monkeypatch.setattr(ni, "_SHEET_CACHE", type(ni._SHEET_CACHE)())
    monkeypatch.setattr(ni, "_SHEET_STATS", dict.fromkeys(ni._SHEET_STATS, 0))


def _vertical_over_radials(n_rad=16, depth=0.15, length=5.0, nseg=10, nvert=20):
    """A vertical's Gauss nodes on the axis over `n_rad` flat radials at
    `depth`: the (above, below) node pair, z relative to the interface, and
    every row the pair asks."""
    gl = np.polynomial.legendre.leggauss(4)[0]

    def nodes(n):
        return np.concatenate([(k + 0.5 + 0.5 * gl) / n for k in range(n)])

    tv = nodes(nvert)
    above = np.stack([np.zeros_like(tv), np.zeros_like(tv), 10.0 * tv], axis=1)
    tr = nodes(nseg)
    below = np.concatenate(
        [
            np.stack([tr * length * dx, tr * length * dy, np.full_like(tr, -depth)], 1)
            for dx, dy in _radial_dirs(n_rad)
        ]
    )
    rho = np.hypot(
        np.hypot(
            above[:, 0][:, None] - below[:, 0][None, :],
            above[:, 1][:, None] - below[:, 1][None, :],
        ),
        A,
    ).ravel()
    z = np.repeat(above[:, 2], below.shape[0])
    zp = np.tile(below[:, 2], above.shape[0])
    return (above, below), np.ascontiguousarray(np.stack([rho, z, zp], axis=1))


def _plan(specs, on):
    with ni.crossing_sheet_retune_scope(on):
        return ni.sheet_plan(EPS_T, K_P, [specs], A)


def test_off_by_default_and_on_when_asked(monkeypatch):
    monkeypatch.delenv(_features.FEATURES[NAME].env, raising=False)
    assert not _features.enabled(NAME)
    assert not RazorSolver(**crossing_deck())._crossing_sheet_retune
    assert not BSplineSolver(**crossing_deck())._crossing_sheet_retune
    assert RazorSolver(
        **crossing_deck(), crossing_sheet_retune=True
    )._crossing_sheet_retune
    monkeypatch.setenv(_features.FEATURES[NAME].env, "1")
    assert _features.enabled(NAME)
    assert RazorSolver(**crossing_deck())._crossing_sheet_retune
    assert BSplineSolver(**crossing_deck())._crossing_sheet_retune
    assert not BSplineSolver(
        **crossing_deck(), crossing_sheet_retune=False
    )._crossing_sheet_retune
    # Outside a solver's fill scope the census reads the shipped weights.
    assert not ni._CROSSING_SHEET_RETUNE.get()
    assert ni._sheet_weights() == (ni._SHEET_BUILD_WEIGHT, ni._SHEET_HEIGHT_WEIGHT)


def test_the_weights_are_constants_and_off_keeps_the_tests_handle(monkeypatch):
    with ni.crossing_sheet_retune_scope(True):
        assert ni._sheet_weights() == (
            ni._SHEET_RETUNE_BUILD_WEIGHT,
            ni._SHEET_RETUNE_HEIGHT_WEIGHT,
        )
    assert ni._SHEET_RETUNE_BUILD_WEIGHT < ni._SHEET_BUILD_WEIGHT
    assert ni._SHEET_RETUNE_HEIGHT_WEIGHT < ni._SHEET_HEIGHT_WEIGHT
    monkeypatch.setattr(ni, "_SHEET_BUILD_WEIGHT", 0.0)
    assert ni._sheet_weights()[0] == 0.0


def test_on_takes_the_radials_plane_off_declines():
    specs, _rows = _vertical_over_radials()
    off = _plan(specs, False)
    on = _plan(specs, True)
    assert -0.15 not in off.planes.tolist()
    assert -0.15 in on.planes.tolist()


@pytest.mark.skipif(not HAVE_CXX, reason="needs the extension's TEST-ONLY hooks")
@pytest.mark.parametrize("on", [False, True])
def test_the_cxx_plan_and_values_are_pythons(on):
    """`sheet_plan` and `designed_tables` through a memo carrying it, as the
    razor and bs2 crossing cores call them, with the feature asked of the C++
    route as an executable's fill would have it: the same planes and heights
    and the same floats at every row."""
    specs, rows = _vertical_over_radials()
    with ni.crossing_sheet_retune_scope(on):
        plan = ni.sheet_plan(EPS_T, K_P, [specs], A)
        memo = ni.TripleMemo()
        memo.sheet_plan = plan
        py = ni.designed_tables(
            EPS_T, K_P, rows[:, 0], rows[:, 1], rows[:, 2], memo=memo
        )
    py = np.stack([py[k] for k in ni.KEYS], axis=1)
    cx = ACC.near_interface_eval_designed(
        K_P,
        K_M,
        [specs],
        A,
        rows[:, 0],
        rows[:, 1],
        rows[:, 2],
        features=[NAME] if on else [],
    )
    assert np.array_equal(np.asarray(cx["planes"]), np.asarray(plan.planes))
    assert np.array_equal(np.asarray(cx["heights"]), np.asarray(plan.heights))
    cv = np.asarray(cx["values"])
    assert np.array_equal(py.view(np.uint64), cv.view(np.uint64))


@pytest.mark.parametrize("cls", [RazorSolver, BSplineSolver])
def test_on_agrees_with_off_within_the_declared_drift(cls):
    deck = hub_deck(n_radials=16)
    zin = {}
    planes = {}
    for on in (False, True):
        before = ni._SHEET_STATS["planes_planned"]
        s = cls(**deck, crossing_sheet_retune=on)
        zin[on] = complex(np.atleast_1d(s.compute_impedance()[0])[0])
        planes[on] = ni._SHEET_STATS["planes_planned"] - before
    # The plan moved (else this measured nothing), and the answer did not.
    assert planes[True] > planes[False]
    drift = abs(zin[True] - zin[False]) / abs(zin[False])
    assert drift <= _features.FEATURES[NAME].drift, drift


def _deep_hub(f, depth, q, n=16):
    """A hub deck scaled to `q`: `n` radials of length q at `depth`, a
    2-segment rise, a 15-segment vertical of length q fed at 0.4333 q. At
    28 MHz, 1.5 m deep, q = 2.5 m the plan takes the most planes and the
    drift is the largest measured (9.77e-12 on razor-2p, momwire#1956)."""
    wires = [
        np.array([(q * dx, q * dy, -depth), (0.0, 0.0, -depth)])
        for dx, dy in _radial_dirs(n)
    ]
    wires.append(np.array([(0.0, 0.0, -depth), (0.0, 0.0, 0.0)]))
    wires.append(np.array([(0.0, 0.0, q), (0.0, 0.0, 0.0)]))
    return dict(
        wires=wires,
        n_per_edge_per_wire=[[10]] * n + [[2], [15]],
        junctions=[
            [(i, "end") for i in range(n)] + [(n, "start")],
            [(n, "end"), (n + 1, "end")],
        ],
        feeds=[(n + 1, 0.43333333333 * q, 1 + 0j)],
        wavelength=299792458.0 / f,
        wire_radius=A,
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
    )


def test_deep_high_frequency_hub_within_the_declared_drift():
    deck = _deep_hub(28e6, 1.5, 2.5)
    zin = {}
    planes = {}
    for on in (False, True):
        before = ni._SHEET_STATS["planes_planned"]
        s = RazorSolver(**deck, crossing_sheet_retune=on)
        zin[on] = complex(np.atleast_1d(s.compute_impedance()[0])[0])
        planes[on] = ni._SHEET_STATS["planes_planned"] - before
    assert planes[True] > planes[False]
    drift = abs(zin[True] - zin[False]) / abs(zin[False])
    # Over the old 2e-12, so this deck is the one the declared bound is for.
    assert 2e-12 < drift <= _features.FEATURES[NAME].drift, drift
