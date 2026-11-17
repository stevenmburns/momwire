"""razor-2p's fill-pair reuse (momwire#1971, `_razor_pair_reuse_core.h`).

The fill finds its own classes of (centroid observer, source segment) pairs:
the direct and image moments keyed on what `seg_moments` reads (the axis
frame's u_r and perpendicular offset, the source's length and radius, the
extended kernel's eligibility and radii), orientation-canonical (#1969's
rule: a reversed source takes M1 -> h M0 - M1), and the composing ground's
remainder on the height-aware key of #1970 (the height SUM, the horizontal
Gram invariants, the vertical tangent parts). One representative per class
is evaluated with the shipped kernels and the rest gathered, by one C++ core
the Python route and the C++ routes (the front route here, the executables
in `test_standalone_nopython.py`) share.

The gates, and why each is not green by construction:

* every agreement test first proves the reuse branch RAN (the route counter
  and the census's class count, which is below the pair count) -- a fill
  that silently fell back would agree to the bit and measure nothing;
* on against off within the registry's declared drift, |dZin|/|Zin|;
* `verify` checks every window's reused moments and remainder rows against
  the shipped fill and raises past its tolerance;
* the adversarial gates build plans with a deliberately WRONG key
  (`razor_reuse_probe`: no height sum, no orientation flip, no radius type,
  no remainder code) on a geometry where that key merges pairs that differ:
  `verify` raises, and the answer moves by orders of magnitude more than the
  drift. Each probe is also shown to change the census, so the right key is
  what keeps those pairs apart.

Knot type (a wing's rise or fall, a junction's) is NOT in the key, and that
is not an omission: the moments are of the source SEGMENT (M0 = int g,
M1 = int tau g over the segment), and which combination a wing takes
(M1/h rising, M0 - M1/h falling) is applied after the gather from the pair's
own tables (`assemble_t1`, `t2_rows`), so two segments with different knots
share their moments exactly. `test_a_junction_deck_reuses` holds that.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from momwire import _features, razor
from momwire._accel import acc
from momwire.razor import RazorSolver

pytestmark = [
    pytest.mark.reach(7),
    pytest.mark.skipif(
        acc is None or not getattr(acc, "razor_pair_reuse_1971", False),
        reason="needs the razor pair-reuse kernels",
    ),
    pytest.mark.filterwarnings("ignore"),
]

DRIFT = _features.FEATURES["razor_pair_reuse"].drift


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("MOMWIRE_X_RAZOR_PAIR_REUSE", raising=False)
    monkeypatch.setattr(razor, "_RAZOR_CXX", False)
    acc.razor_reuse_probe(0)
    acc.razor_reuse_cache_clear()
    razor._PAIR_REUSE_ROUTE.clear()
    yield
    acc.razor_reuse_probe(0)
    acc.razor_reuse_cache_clear()


SOMM = {"ground_z": 0.0, "ground_eps": (13.0, 0.005), "ground_model": "sommerfeld"}
REFL = {"ground_z": 0.0, "ground_eps": (13.0, 0.005), "ground_model": "refl-coef"}
PEC = {"ground_z": 0.0}
GROUNDS = {"free": {}, "pec": PEC, "sommerfeld": SOMM, "refl-coef": REFL}


def _dipoles(heights=(8.0, 8.0), xs=(0.0, 3.0), reverse_second=False):
    w = []
    for i, (h, x) in enumerate(zip(heights, xs)):
        a, b = [x, -5.0, h], [x, 5.0, h]
        if reverse_second and i == 1:
            a, b = b, a
        w.append(np.array([a, b]))
    return w


def _triangle(h=6.0, side=8.0):
    """An equilateral loop: three congruent sides, each a rotation of the
    first by 120 degrees, so mirrored pairs reverse both tangents."""
    r = side / np.sqrt(3.0)
    ang = np.deg2rad([90.0, 210.0, 330.0])
    pts = np.stack([r * np.cos(ang), r * np.sin(ang), np.full(3, h)], axis=1)
    return [np.array([pts[i], pts[(i + 1) % 3]]) for i in range(3)]


def _solve(wires, mode, *, nsegs=31, radius=0.001, ground=None, **kw):
    s = RazorSolver(
        wires=wires,
        nsegs=nsegs,
        wire_radius=radius,
        wavelength=20.0,
        nec5_quadrature=True,
        extended_kernel=kw.pop("extended_kernel", True),
        razor_pair_reuse=mode,
        **(ground or {}),
        **kw,
    )
    return s.compute_port_solution().y


def _zin(y):
    return 1.0 / np.diag(y)


def _drift(a, b):
    za, zb = _zin(a), _zin(b)
    return float(np.max(np.abs(za - zb) / np.abs(zb)))


def _ran():
    return razor._PAIR_REUSE_ROUTE.get("reuse", 0)


# ------------------------------------------------------------ off / on


def _cache():
    # process totals: the plans built and found, as deltas across a test
    c = acc.razor_reuse_cache()
    return c["built"], c["found"]


def test_off_never_enters_the_reuse_branch():
    before = _cache()
    _solve(_dipoles(), None)
    _solve(_dipoles(), False)
    assert razor._PAIR_REUSE_ROUTE == {}
    assert _cache() == before


def test_the_environment_alone_turns_it_on(monkeypatch):
    monkeypatch.setenv("MOMWIRE_X_RAZOR_PAIR_REUSE", "1")
    _solve(_dipoles(), None)
    assert _ran() == 1
    monkeypatch.setenv("MOMWIRE_X_RAZOR_PAIR_REUSE", "1")
    _solve(_dipoles(), False)  # the keyword wins
    assert _ran() == 1


def test_a_bad_value_is_refused():
    with pytest.raises(ValueError, match="razor_pair_reuse"):
        _solve(_dipoles(), "yes")


@pytest.mark.parametrize("ground", sorted(GROUNDS))
@pytest.mark.parametrize("ek", [True, False], ids=["ek", "reduced"])
def test_on_matches_off_within_the_drift(ground, ek):
    wires = _triangle() + _dipoles(heights=(9.0, 9.0), xs=(0.0, 1.5))
    off = _solve(wires, False, ground=GROUNDS[ground], extended_kernel=ek)
    on = _solve(wires, True, ground=GROUNDS[ground], extended_kernel=ek)
    assert _ran() == 1
    st = razor._PAIR_REUSE_STATS[0]
    assert st["classes"] < st["pairs"] / 2, st
    if ground == "sommerfeld":
        assert 0 < st["q_classes"] < st["q_pairs"] / 2, st
    else:
        assert st["q_pairs"] == 0, st
    assert _drift(on, off) <= DRIFT


@pytest.mark.parametrize("ground", sorted(GROUNDS))
def test_verify_checks_every_window(ground):
    wires = _triangle()
    verified = _solve(wires, "verify", ground=GROUNDS[ground])
    assert _ran() == 1
    on = _solve(wires, True, ground=GROUNDS[ground])
    # verify answers with the reused values, so it IS the on fill
    np.testing.assert_array_equal(verified, on)


def test_mixed_radii_and_a_junction_deck_reuses():
    """Two radii (Eq 89's two-radius tube on the EK) and a junction (rising
    and falling wings, a through knot): the moments do not read a knot, so
    the census merges across them and the answer holds."""
    v = [
        np.array([[0.0, 0.0, 6.0], [0.0, 3.0, 9.0]]),
        np.array([[0.0, 0.0, 6.0], [0.0, -3.0, 9.0]]),
    ]
    wires = v + _dipoles(heights=(8.0, 8.0), xs=(2.0, 4.0))
    radius = [0.001, 0.001, 0.002, 0.001]
    off = _solve(wires, False, radius=radius, ground=SOMM)
    on = _solve(wires, True, radius=radius, ground=SOMM)
    assert _ran() == 1
    assert _drift(on, off) <= DRIFT
    _solve(wires, "verify", radius=radius, ground=SOMM)


def test_the_plan_is_found_again_for_the_same_geometry():
    b0, f0 = _cache()
    _solve(_dipoles(), True)
    _solve(_dipoles(), True)
    assert _cache() == (b0 + 1, f0 + 1)


def test_reversed_wires_share_their_classes():
    """A straight run's +k and -k offsets, and a wire listed reversed, are
    the same couplings with the source tangent reversed: the same classes
    (and the same answer) either way round."""
    _solve(_dipoles(), True, ground=SOMM)
    fwd = razor._PAIR_REUSE_STATS[0]
    _solve(_dipoles(reverse_second=True), True, ground=SOMM)
    rev = razor._PAIR_REUSE_STATS[0]
    assert (fwd["classes"], fwd["q_classes"]) == (rev["classes"], rev["q_classes"])
    acc.razor_reuse_probe(5)
    unfolded_y = _solve(_dipoles(), True, ground=SOMM)
    unfolded = razor._PAIR_REUSE_STATS[0]
    # without the fold a straight run keeps +k and -k apart: about twice the
    # shared classes (the near pairs are their own either way)
    shared, shared_unfolded = (
        st["classes"] - st["near_pairs"] for st in (fwd, unfolded)
    )
    assert shared_unfolded > 1.6 * shared, (fwd, unfolded)
    acc.razor_reuse_probe(0)
    assert _drift(_solve(_dipoles(), True, ground=SOMM), unfolded_y) <= DRIFT


# ------------------------------------------------------------ C++ route


@pytest.mark.parametrize("ground", ["free", "pec", "sommerfeld"])
def test_the_cxx_front_route_is_the_python_route_to_the_bit(ground, monkeypatch):
    """The C++ front route (the executables' fill, from Python) gathers from
    the same plan and class values: its answer with reuse on is the Python
    route's bit for bit, and its counters show the reuse ran."""
    wires = _triangle()
    py = _solve(wires, True, ground=GROUNDS[ground])
    monkeypatch.setattr(razor, "_RAZOR_CXX", True)
    ctx = razor._cxx_context()
    before = dict(ctx.counters)
    cx = _solve(wires, True, ground=GROUNDS[ground])
    after = ctx.counters
    assert razor._CXX_FRONT_ROUTE.get("front", 0) > 0
    assert after["reuse_fills"] == before["reuse_fills"] + 1
    assert after["reuse_windows"] > before["reuse_windows"]
    if ground == "sommerfeld":
        assert after["reuse_q_windows"] > before["reuse_q_windows"]
    np.testing.assert_array_equal(cx, py)
    # and its verify mode checks the windows too
    _solve(wires, "verify", ground=GROUNDS[ground])
    assert ctx.counters["reuse_verify_worst"] <= acc.razor_reuse_verify_tol


# ------------------------------------------------------------ adversarial

PROBES = {
    # a remainder key without the height sum merges a pair on the low dipole
    # with its twin on the high one
    "no-height": (1, _dipoles(heights=(3.0, 9.0), xs=(0.0, 0.0)), SOMM),
    # a direct key without the orientation fold takes +k's M1 for -k's
    "no-flip": (2, _dipoles(), {}),
    # a direct key without the radius type merges a thin wire with a thick one
    "no-type": (3, _dipoles(), {}),
    # a remainder key whose orientation code is dropped keeps the wrong sign
    "no-q-code": (4, _dipoles(), SOMM),
}


@pytest.mark.parametrize("name", sorted(PROBES))
def test_a_wrong_key_is_caught(name):
    probe, wires, ground = PROBES[name]
    radius = [0.001, 0.008] if name == "no-type" else 0.001
    off = _solve(wires, False, radius=radius, ground=ground)
    _solve(wires, True, radius=radius, ground=ground)
    right = razor._PAIR_REUSE_STATS[0]
    acc.razor_reuse_probe(probe)
    wrong = _solve(wires, True, radius=radius, ground=ground)
    st = razor._PAIR_REUSE_STATS[0]
    key = "q_classes" if name in ("no-height", "no-q-code") else "classes"
    if name in ("no-height", "no-type"):
        # the wrong key merges what the right one keeps apart (the two
        # orientation probes keep the classes and lose the signs)
        assert st[key] < right[key], (st, right)
    moved = _drift(wrong, off)
    assert moved > 1e4 * DRIFT, moved
    with pytest.raises(RuntimeError, match="pair_reuse verify"):
        _solve(wires, "verify", radius=radius, ground=ground)
    acc.razor_reuse_probe(0)
    assert _drift(_solve(wires, True, radius=radius, ground=ground), off) <= DRIFT


def test_a_wrong_key_is_caught_on_the_cxx_route(monkeypatch):
    monkeypatch.setattr(razor, "_RAZOR_CXX", True)
    acc.razor_reuse_probe(1)
    wires = _dipoles(heights=(3.0, 9.0), xs=(0.0, 0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError, match="pair_reuse verify"):
            _solve(wires, "verify", ground=SOMM)
