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

Reach 8 (the two nec5 executables): `DECKS` below are two hubs as NEC-5
decks, which tests/test_standalone_nopython.py gives each executable off and
on, against the launcher (byte for byte on Linux, the fixture gates'
tolerance on Windows and macOS), and the executable's own Zin, on against
off, against the declared drift. The launcher's route on the same decks is
held here (`test_the_launchers_route_moves_its_plan_within_its_drift`): its
census plans more planes on than off, so the byte gate compared the retune
and not the shipped weights twice. `DECKS["deep_hub"]` is the executables'
worst case and is NOT `_deep_hub` below (the Python route's worst case on
the dict-built decks); each drift figure below names its deck.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from momwire import _accel, _features
from momwire import _near_interface as ni
from momwire.bspline import BSplineSolver
from momwire.razor import RazorSolver

from test_crossing_serve_524 import _radial_dirs, crossing_deck, hub_deck
from _lane_gate import assert_lanes_match

# An experimental feature (`momwire._features`, reach 8).
pytestmark = [
    pytest.mark.reach(8),
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


def hub_nec(f_mhz, depth, q, h, label, n=16):
    """A synthetic NEC-5 deck (not a user's model) of the hub shape above:
    `n` radials of length `q`, 10 segments each, at `depth` under a 13 /
    0.005 S/m Sommerfeld ground; a 2-segment rise; a 15-segment vertical of
    height `h`, fed on its 7th segment. Kept as text here, not under
    tests/fixtures, which the C++ driver tests sweep whole."""
    lines = [f"CM {label}", "CE"]
    for i in range(n):
        a = 2.0 * math.pi * i / n
        x, y = q * math.cos(a), q * math.sin(a)
        lines.append(f"GW {i + 1} 10 0 0 {-depth} {x:.9f} {y:.9f} {-depth} 0.001")
    lines.append(f"GW {n + 1} 2 0 0 {-depth} 0 0 0 0.001")
    lines.append(f"GW {n + 2} 15 0 0 0 0 0 {h} 0.001")
    lines += ["GE -1", "GN 2 0 0 0 13 0.005", f"FR 0 1 0 0 {f_mhz} 0"]
    lines += [f"EX 0 {n + 2} 7 0 1 0", "XQ", "EN", ""]
    return "\n".join(lines)


# The executables' decks, and the launcher route's in
# `test_the_launchers_route_moves_its_plan_within_its_drift`. hub16 is
# `hub_deck(n_radials=16)`'s shape at 7 MHz: off, no route plans its
# radials' plane; on, every one does (drift 2.3e-13 / 2.4e-13 razor-2p /
# bspline). deep_hub is the largest drift the executables measured over a
# scan of these `hub_nec` hubs (7-30 MHz, depth 0.15-3 m, radials and
# vertical 2.5-7.5 m; Rocketlake, T1, 2026-11-17): 1.93e-11 on nec5-bspline
# (two planes on, none off), 1.40e-11 on nec5-razor-2p there, whose own worst
# was 1.56e-11 (28 MHz, 0.5 m, 5 m); the launcher's route on it prints the
# same 1.93e-11 / 1.40e-11. It is a different geometry from `_deep_hub`
# below (30 MHz and 5 m, not 28 MHz and 2.5 m; radials laid from the hub
# out; the vertical and its feed counted from the base), whose 9.77e-12 is
# the Python route's figure on the dict-built decks.
DECKS = {
    "hub16": hub_nec(
        7.0, 0.15, 5.0, 10.0, "16 radials at 0.15 m, 7 MHz (momwire#1956)"
    ),
    "deep_hub": hub_nec(
        30.0, 1.5, 5.0, 5.0, "16 radials at 1.5 m, 30 MHz (momwire#1956)"
    ),
}


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
    # Bit-equal off Windows. On win32, MSVC's /fp:fast reassociates the C++
    # sums, so the house's derived tolerance applies (tests/_lane_gate.py);
    # measured on the i5-1240P at 8.6e-16 off and 2.5e-15 on.
    assert_lanes_match(cv, py, f"sheet values, retune {'on' if on else 'off'}")


def _cxx_plan(args, features):
    """The C++ census's plan through the TEST-ONLY hook."""
    got = ACC.near_interface_eval_designed(*args, features=features)
    return np.asarray(got["planes"]).tolist(), np.asarray(got["heights"]).tolist()


def _hook_args():
    """The vertical over radials, on few rows: the plan is what is held."""
    specs, rows = _vertical_over_radials()
    return (K_P, K_M, [specs], A, rows[:32, 0], rows[:32, 1], rows[:32, 2])


@pytest.mark.skipif(not HAVE_CXX, reason="needs the extension's TEST-ONLY hooks")
def test_a_refused_feature_name_leaves_nothing_requested():
    """momwire#2007: the hook asks for its features one name at a time, and a
    name the C++ route does not carry throws part way through. The names
    already asked must not stay on for the rest of the process."""
    args = _hook_args()
    off, on = _cxx_plan(args, []), _cxx_plan(args, [NAME])
    assert off != on, "the feature must move this plan for the test to see it"
    with pytest.raises(ValueError, match="not a C\\+\\+ feature"):
        ACC.near_interface_eval_designed(*args, features=[NAME, "no_such_feature"])
    assert _cxx_plan(args, []) == off


@pytest.mark.skipif(not HAVE_CXX, reason="needs the extension's TEST-ONLY hooks")
def test_the_hook_scope_on_two_threads_neither_leaks_nor_crosses():
    """momwire#2007, in the shape of the kcl port's
    `test_feature_scopes_on_two_threads_neither_leak_nor_cross`. The requested
    set is process-global. One thread holds a scope with the GIL released
    (`bspline_solve_ports`, asking for this feature) while another plans
    through this hook asking for nothing. The plan must be the flag-off plan
    every time, and nothing may stay requested afterwards. A save / restore
    outside the scopes' lock fails both ways: the plan reads the other thread's
    request, and the restore puts back a request that thread has since
    dropped."""
    import threading

    from test_kcl_nullspace_cxx_1898 import CRAFTED, N, _numpy_blas, _random_system
    from threadpoolctl import threadpool_limits

    _syrk, four = _numpy_blas()
    args = _hook_args()
    off, on = _cxx_plan(args, []), _cxx_plan(args, [NAME])
    assert off != on, "the feature must move this plan for the test to see it"
    Z, B = _random_system(np.random.default_rng(5), N, 1)
    kcl, _route = CRAFTED["reducible"]

    rounds = 60
    start = threading.Barrier(2)
    planned = threading.Event()
    plans = []
    errors = []

    def plan():
        try:
            start.wait()
            for _ in range(rounds):
                plans.append(_cxx_plan(args, []))
        except BaseException as exc:  # noqa: BLE001 — re-raised on the main thread below
            errors.append(exc)
        finally:
            planned.set()

    def solve():
        try:
            start.wait()
            while not planned.is_set():
                ACC.bspline_solve_ports(Z, B, kcl, four, [NAME])
        except BaseException as exc:  # noqa: BLE001 — re-raised on the main thread below
            errors.append(exc)

    with threadpool_limits(limits=1):
        threads = [threading.Thread(target=f) for f in (plan, solve)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    if errors:
        raise errors[0]
    assert len(plans) == rounds
    crossed = sum(p != off for p in plans)
    assert crossed == 0, f"{crossed} of {rounds} plans read the other thread's request"
    assert _cxx_plan(args, []) == off, "the feature stayed requested"


# Razor's case moved to the slow lane at 0.94.0 (median 6.6 s over 12 runs);
# bspline's stays in the default lane, so every PR still runs one basis.
@pytest.mark.parametrize(
    "cls", [pytest.param(RazorSolver, marks=pytest.mark.slow), BSplineSolver]
)
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
    drift is the largest the Python route measured on these dict-built decks
    (9.77e-12 on razor-2p, momwire#1956). Not `DECKS["deep_hub"]`, the
    executables' worst case (1.93e-11, nec5-bspline), which
    `test_the_launchers_route_moves_its_plan_within_its_drift` runs."""
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


# Slow lane at 0.94.0: median 7.7 s over 12 runs.
@pytest.mark.slow
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


def _launcher_route(text, basis, on, monkeypatch):
    """`momwire-nec5-<basis>`'s own route for one deck text, in process
    (`_shell.render`, what the launcher's server answers it with), the
    feature off or on by its variable as the launcher reads it: (planes the
    census planned, Zin at full precision, the printout)."""
    from momwire.dropin.nec5 import _serve, _shell

    monkeypatch.setenv(_features.FEATURES[NAME].env, "1" if on else "0")
    states = []
    orig = _serve._port_state

    def port_state(*args, **kwargs):
        states.append(state := orig(*args, **kwargs))
        return state

    monkeypatch.setattr(_serve, "_port_state", port_state)
    monkeypatch.setattr(ni, "_SHEET_CACHE", type(ni._SHEET_CACHE)())
    before = ni._SHEET_STATS["planes_planned"]
    printout = _shell.render(text, basis=basis)
    (state,) = states
    (i_source,) = np.atleast_1d(state.i_source)
    return (
        ni._SHEET_STATS["planes_planned"] - before,
        1.0 / complex(i_source),
        printout,
    )


@pytest.mark.parametrize("basis", ["razor-2p", "bspline"])
@pytest.mark.parametrize("deck", sorted(DECKS))
def test_the_launchers_route_moves_its_plan_within_its_drift(deck, basis, monkeypatch):
    """The launcher's half of the executables' route proof
    (tests/test_standalone_nopython.py's
    `test_the_sheet_retune_moves_the_executables_plan_within_its_drift`), on
    the same deck text: on, the census plans more planes than off, so the
    byte gate against the launcher held the retune and not two runs of the
    shipped weights (on and off print the same digits); and Zin moves by no
    more than the declared drift. On deep_hub this is the measured worst
    case of the executables, 1.93e-11 on bspline."""
    text = DECKS[deck]
    off_planes, z_off, off_text = _launcher_route(text, basis, False, monkeypatch)
    on_planes, z_on, on_text = _launcher_route(text, basis, True, monkeypatch)
    assert on_planes > off_planes, (off_planes, on_planes)
    assert "+crossing_sheet_retune" in on_text.split("\n")[1]
    assert "+crossing_sheet_retune" not in off_text.split("\n")[1]
    drift = abs(z_on - z_off) / abs(z_off)
    assert drift <= _features.FEATURES[NAME].drift, drift
