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

# A native feature refuses below its reach. At reach 7 every Python route is
# within it (macOS is level 5, the drop-in 4), so this guard skips nothing
# today; it stays so a narrowed reach skips the on-tests rather than failing
# them, as test_point_cplx_staged_1849 does. The off and refusal tests run
# everywhere.
_needs_cert = pytest.mark.skipif(
    not _features.certified("razor_pair_reuse"),
    reason="razor_pair_reuse is not certified on this platform",
)
DRIFT = _features.FEATURES["razor_pair_reuse"].drift
# The decks here are well conditioned: on against off moves them ~1e-13, so
# they hold the reuse to this, tighter than the registry's drift (which the
# close-spaced decks set, `_features`). A wrong key moves them by orders more.
CLEAN = 1e-11


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("MOMWIRE_X_RAZOR_PAIR_REUSE", raising=False)
    monkeypatch.setattr(razor, "_RAZOR_CXX", False)
    acc.razor_reuse_probe(0)
    acc.razor_reuse_class_limit(0)
    acc.razor_reuse_plan_budget(0)
    acc.razor_reuse_cache_clear()
    # The decks here are cheap on purpose, below the small-deck thresholds,
    # and test the reuse itself: lift them (test_small_decks_step_aside
    # holds them).
    acc.razor_reuse_set_small_deck_segs(0, 0)
    razor._PAIR_REUSE_ROUTE.clear()
    yield
    acc.razor_reuse_probe(0)
    acc.razor_reuse_class_limit(0)
    acc.razor_reuse_set_small_deck_segs(-1, -1)
    acc.razor_reuse_plan_budget(0)
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


@_needs_cert
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


@_needs_cert
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
    assert _drift(on, off) <= CLEAN


@_needs_cert
@pytest.mark.parametrize("ground", sorted(GROUNDS))
def test_verify_checks_every_window(ground):
    wires = _triangle()
    verified = _solve(wires, "verify", ground=GROUNDS[ground])
    assert _ran() == 1
    on = _solve(wires, True, ground=GROUNDS[ground])
    # verify answers with the reused values, so it IS the on fill
    np.testing.assert_array_equal(verified, on)


@_needs_cert
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
    assert _drift(on, off) <= CLEAN
    _solve(wires, "verify", radius=radius, ground=SOMM)


@_needs_cert
def test_the_plan_is_found_again_for_the_same_geometry():
    b0, f0 = _cache()
    _solve(_dipoles(), True)
    _solve(_dipoles(), True)
    assert _cache() == (b0 + 1, f0 + 1)


@_needs_cert
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
    assert _drift(_solve(_dipoles(), True, ground=SOMM), unfolded_y) <= CLEAN


def _hairpin(x0=3.3):
    """Two 10 m wires 2 mm apart joined by a 2 mm bridge; with 21 segments
    per wire the bridge's h = 95 um is under the 1 mm radius (h/a = 0.095)
    and cond(Z) is ~7e9. Translated off the origin, its coordinates carry
    rounding that a key against the pair's distance cannot see."""
    ws = [
        [[0, -5, 8], [0, 5, 8]],
        [[0, 5, 8], [0.002, 5, 8]],
        [[0.002, 5, 8], [0.002, -5, 8]],
    ]
    return [np.array(w, dtype=float) + [x0, 0.0, 0.0] for w in ws]


def _hairpin_zin(mode, wires):
    s = RazorSolver(
        wires=wires,
        n_per_edge_per_wire=[[21], [21], [21]],
        feeds=[(0, 5.238095238095238, 1 + 0j)],
        wire_radius=0.001,
        wavelength=20.0,
        nec5_quadrature=True,
        extended_kernel=False,
        exact_kernel=False,
        razor_pair_reuse=mode,
    )
    return complex(1.0 / s.compute_port_solution().y[0, 0])


@_needs_cert
def test_segments_shorter_than_their_radius_keep_their_near_pairs():
    """The near rule is in units of max(h, a), not h: the hairpin's
    bridge-on-bridge pairs a few h apart are within a radius, where the
    reduced kernel varies over a. Under the rule in h alone (the probe)
    they are shared, every pair still passes `verify` (2e-13), and Zin
    moves 1.5e-5. With the right rule it moves ~4e-11."""
    wires = _hairpin()
    off = _hairpin_zin(False, wires)
    on = _hairpin_zin(True, wires)
    assert _ran() == 1
    right = razor._PAIR_REUSE_STATS[0]
    assert right["classes"] < 0.6 * right["pairs"], right
    good = abs(on - off) / abs(off)
    assert good < 1e-9, good
    acc.razor_reuse_probe(6)
    wrong = _hairpin_zin(True, wires)
    st = razor._PAIR_REUSE_STATS[0]
    assert st["near_pairs"] < right["near_pairs"], (st, right)
    bad = abs(wrong - off) / abs(off)
    assert bad > 1e-7, bad


# ------------------------------------------------------------ C++ route


@_needs_cert
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


@_needs_cert
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
    assert moved > 1e3 * CLEAN, moved
    with pytest.raises(RuntimeError, match="pair_reuse verify"):
        _solve(wires, "verify", radius=radius, ground=ground)
    acc.razor_reuse_probe(0)
    assert _drift(_solve(wires, True, radius=radius, ground=ground), off) <= CLEAN


@_needs_cert
def test_a_wrong_key_is_caught_on_the_cxx_route(monkeypatch):
    monkeypatch.setattr(razor, "_RAZOR_CXX", True)
    acc.razor_reuse_probe(1)
    wires = _dipoles(heights=(3.0, 9.0), xs=(0.0, 0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError, match="pair_reuse verify"):
            _solve(wires, "verify", ground=SOMM)


# ------------------------------------------------------------ declines


def _solver(wires, mode, ground=None):
    return RazorSolver(
        wires=wires,
        nsegs=31,
        wire_radius=0.001,
        wavelength=20.0,
        nec5_quadrature=True,
        extended_kernel=True,
        razor_pair_reuse=mode,
        **(ground or {}),
    )


@_needs_cert
@pytest.mark.parametrize("family", ["direct", "remainder"])
def test_a_census_it_cannot_code_declines_and_fills_as_off(family):
    """A deck with more classes than the census's int32 entries code (2^30
    direct, 2^29 remainder; each of P thread parts a 2^-ceil(log2 P) share)
    used to THROW out of the fill. It declines now, with its reason, and fills as
    off bit for bit. The test hook lowers the limit so a small deck reaches
    it: one short of the direct classes the direct census declines, and one
    short of the remainder's (over Sommerfeld) the remainder's does."""
    wires = _triangle() + _dipoles(heights=(9.0, 9.0), xs=(0.0, 1.5))
    off = _solver(wires, False, SOMM).compute_port_solution().y
    _solver(wires, True, SOMM).compute_port_solution()
    st = razor._PAIR_REUSE_STATS[0]
    assert st["declined"] is None and st["q_classes"] > 0, st
    c, q = st["classes"], st["q_classes"]
    if family == "direct":
        limits, why = (c - 1, 0), "too many classes to code"
    else:
        limits, why = (c, q - 1), "too many remainder classes to code"
    acc.razor_reuse_class_limit(*limits)
    razor._PAIR_REUSE_ROUTE.clear()
    s = _solver(wires, True, SOMM)
    y = s.compute_port_solution().y
    assert razor._PAIR_REUSE_ROUTE == {"declined": 1}
    assert s.pair_reuse_declined == why
    assert razor._PAIR_REUSE_STATS[0]["declined"] == why
    assert razor._PAIR_REUSE_STATS[0]["classes"] == 0
    np.testing.assert_array_equal(y, off)
    # the declined plan is cached: a second fill does not re-run the census
    b0, f0 = _cache()
    _solver(wires, True, SOMM).compute_port_solution()
    assert _cache() == (b0, f0 + 1)
    # and the limit is part of the plan's key: lifted, the reuse serves again
    acc.razor_reuse_class_limit(0)
    s = _solver(wires, True, SOMM)
    s.compute_port_solution()
    assert s.pair_reuse_declined is None
    assert razor._PAIR_REUSE_ROUTE.get("reuse", 0) == 1


@_needs_cert
def test_a_census_it_cannot_code_declines_on_the_cxx_route(monkeypatch):
    wires = _triangle()
    monkeypatch.setattr(razor, "_RAZOR_CXX", True)
    off = _solve(wires, False, ground=SOMM)
    ctx = razor._cxx_context()
    acc.razor_reuse_class_limit(4)
    before = dict(ctx.counters)
    y = _solve(wires, True, ground=SOMM)
    after = ctx.counters
    assert razor._CXX_FRONT_ROUTE.get("front", 0) > 0
    assert after["reuse_declined"] == before["reuse_declined"] + 1
    assert after["reuse_fills"] == before["reuse_fills"]
    assert after["reuse_windows"] == before["reuse_windows"]
    assert ctx.reuse_declined_reason == "too many classes to code"
    np.testing.assert_array_equal(y, off)


def _random_wires(n=8, seed=3):
    """Wires of random place, direction and length: hardly two pairs alike,
    so nearly every pair is its own class."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        p = rng.uniform([-4.0, -4.0, 6.0], [4.0, 4.0, 10.0])
        d = rng.normal(size=3)
        out.append(np.array([p, p + rng.uniform(2.0, 4.0) * d / np.linalg.norm(d)]))
    return out


POOR = "too few shared pairs to repay the plan"


@_needs_cert
@pytest.mark.parametrize("ground", ["free", "pec", "sommerfeld"])
def test_a_deck_that_compresses_poorly_declines(ground):
    """The census stops once its classes pass MAX_CLASS_FRACTION (3/4) of
    its pairs (`_razor_pair_reuse_core.h`): the plan's memory grows with its
    classes (~200 B each at the census's peak), so a deck whose pairs are
    nearly all distinct would cost that for little gain. It declines with
    its reason, reported in `pair_reuse_declined`, and fills as off bit for
    bit; a deck that compresses well (the triangle and dipoles, well under 3/4)
    still reuses."""
    wires = _random_wires()
    off = _solver(wires, False, GROUNDS[ground]).compute_port_solution().y
    razor._PAIR_REUSE_ROUTE.clear()
    s = _solver(wires, True, GROUNDS[ground])
    y = s.compute_port_solution().y
    assert razor._PAIR_REUSE_ROUTE == {"declined": 1}
    assert s.pair_reuse_declined == POOR
    st = razor._PAIR_REUSE_STATS[0]
    assert st["declined"] == POOR and st["classes"] == 0, st
    np.testing.assert_array_equal(y, off)
    # the declined plan is cached: a second fill does not re-run the census
    b0, f0 = _cache()
    _solver(wires, True, GROUNDS[ground]).compute_port_solution()
    assert _cache() == (b0, f0 + 1)
    # not every deck: one that compresses is served
    razor._PAIR_REUSE_ROUTE.clear()
    good = _solver(_triangle(), True, GROUNDS[ground])
    good.compute_port_solution()
    assert good.pair_reuse_declined is None
    assert razor._PAIR_REUSE_ROUTE.get("reuse", 0) == 1


@_needs_cert
def test_a_deck_that_compresses_poorly_declines_on_the_cxx_route(monkeypatch):
    wires = _random_wires()
    monkeypatch.setattr(razor, "_RAZOR_CXX", True)
    off = _solve(wires, False, ground=SOMM)
    ctx = razor._cxx_context()
    before = dict(ctx.counters)
    y = _solve(wires, True, ground=SOMM)
    after = ctx.counters
    assert razor._CXX_FRONT_ROUTE.get("front", 0) > 0
    assert after["reuse_declined"] == before["reuse_declined"] + 1
    assert after["reuse_fills"] == before["reuse_fills"]
    assert ctx.reuse_declined_reason == POOR
    np.testing.assert_array_equal(y, off)


def test_the_plan_budget_is_256_mb():
    # 512 MB hosted bound = this budget + one dense Z (144 MB at the
    # 3000-segment cap) + headroom (`_razor_pair_reuse_core.h`, Steve 11-17).
    assert acc.razor_reuse_plan_budget_bytes == 256 << 20


@_needs_cert
@pytest.mark.parametrize("ground", ["free", "pec", "sommerfeld"])
def test_a_plan_past_its_byte_budget_declines(ground):
    """The plan's bytes (maps, groups and flags, one wavenumber's class
    values) are held to PLAN_BUDGET_BYTES. Lowered by the test hook to one
    byte under what this deck's plan takes, the census stops and the plan
    declines with the budget named, fills as off bit for bit, and is cached;
    lowered under the maps alone, it declines before any census; at the
    plan's own size it is served."""
    wires = _triangle() + _dipoles(heights=(9.0, 9.0), xs=(0.0, 1.5))
    g = GROUNDS[ground]
    off = _solver(wires, False, g).compute_port_solution().y
    _solver(wires, True, g).compute_port_solution()
    st = razor._PAIR_REUSE_STATS[0]
    assert st["declined"] is None, st
    need = st["budget_bytes"]
    assert need == st["nbytes"] + 32 * (st["classes"] + st["q_classes"]), st
    n_maps = 2 if ground != "free" else 1
    if ground == "sommerfeld":
        n_maps += 1
    maps = 4 * n_maps * (st["pairs"] // (2 if ground != "free" else 1))
    for budget in (need - 1, maps - 1):
        acc.razor_reuse_plan_budget(budget)
        why = f"the plan would exceed its {budget / 2**20:.6g} MB budget"
        razor._PAIR_REUSE_ROUTE.clear()
        s = _solver(wires, True, g)
        y = s.compute_port_solution().y
        assert razor._PAIR_REUSE_ROUTE == {"declined": 1}
        assert s.pair_reuse_declined == why
        assert razor._PAIR_REUSE_STATS[0]["classes"] == 0
        np.testing.assert_array_equal(y, off)
        b0, f0 = _cache()
        _solver(wires, True, g).compute_port_solution()
        assert _cache() == (b0, f0 + 1)
    acc.razor_reuse_plan_budget(need)
    s = _solver(wires, True, g)
    s.compute_port_solution()
    assert s.pair_reuse_declined is None


@_needs_cert
def test_a_plan_past_its_byte_budget_declines_on_the_cxx_route(monkeypatch):
    wires = _triangle()
    monkeypatch.setattr(razor, "_RAZOR_CXX", True)
    off = _solve(wires, False, ground=SOMM)
    acc.razor_reuse_plan_budget(1024)
    ctx = razor._cxx_context()
    before = dict(ctx.counters)
    y = _solve(wires, True, ground=SOMM)
    after = ctx.counters
    assert razor._CXX_FRONT_ROUTE.get("front", 0) > 0
    assert after["reuse_declined"] == before["reuse_declined"] + 1
    assert after["reuse_fills"] == before["reuse_fills"]
    assert ctx.reuse_declined_reason == (
        f"the plan would exceed its {1024 / 2**20:.6g} MB budget"
    )
    np.testing.assert_array_equal(y, off)


# ------------------------------------------------------------ the declared drift


def _line_zin(mode, spacing, ground, exact):
    """A 10 m two-wire line `spacing` apart (1 mm wire, 41 segments a side,
    one-segment bridges), fed at the middle of one bridge."""
    s = spacing
    wires = [
        np.array(w, dtype=float)
        for w in (
            [[0, -5, 4], [0, 5, 4]],
            [[0, 5, 4], [s, 5, 4]],
            [[s, 5, 4], [s, -5, 4]],
            [[s, -5, 4], [0, -5, 4]],
        )
    ]
    sol = RazorSolver(
        wires=wires,
        n_per_edge_per_wire=[[41], [1], [41], [1]],
        feeds=[(3, spacing / 2, 1 + 0j)],
        wire_radius=0.001,
        wavelength=299.792458 / 14.0,
        nec5_quadrature=True,
        extended_kernel=True,
        exact_kernel=exact,
        razor_pair_reuse=mode,
        **GROUNDS[ground],
    )
    return complex(1.0 / sol.compute_port_solution().y[0, 0])


@_needs_cert
@pytest.mark.parametrize("ground", ["free", "pec", "sommerfeld"])
@pytest.mark.parametrize("exact", [False, True])
def test_a_close_spaced_line_is_within_the_declared_drift(ground, exact):
    """The shape that sets the registry's drift: a line 2.5 radii apart, a
    near-short at its feed (|Zin| ~ 27 ohm), whose flag-off fill itself moves
    ~1e-10 when the deck is translated. On against off stays within the
    declared drift (here ~3e-12; the deck forms of these lines measured up to
    4.3e-10, `_features`)."""
    off = _line_zin(False, 0.0025, ground, exact)
    on = _line_zin(True, 0.0025, ground, exact)
    assert _ran() == 1
    assert abs(on - off) / abs(off) <= DRIFT


# ------------------------------------------------------------ small decks


@_needs_cert
@pytest.mark.parametrize("ground", sorted(GROUNDS))
def test_small_decks_step_aside(ground):
    """Below SMALL_DECK_SEGS_FREE (free space, PEC) or SMALL_DECK_SEGS_GROUND
    (Sommerfeld, a reflection coefficient) segments the reuse declines with
    its reason and fills as off, bit for bit; "verify" still runs it; N - 1
    declines and N serves."""
    acc.razor_reuse_set_small_deck_segs(-1, -1)
    finite = ground in ("sommerfeld", "refl-coef")
    want = 100 if finite else 400
    assert acc.razor_reuse_small_deck_segs(finite) == want
    wires = _triangle()  # 93 segments: below both
    off = _solver(wires, False, GROUNDS[ground]).compute_port_solution().y
    s = _solver(wires, True, GROUNDS[ground])
    y = s.compute_port_solution().y
    why = f"too few segments to repay the plan (N < {want})"
    assert s.pair_reuse_declined == why
    assert razor._PAIR_REUSE_ROUTE == {why: 1}
    np.testing.assert_array_equal(y, off)
    _solver(wires, "verify", GROUNDS[ground]).compute_port_solution()
    assert razor._PAIR_REUSE_ROUTE.get("reuse", 0) == 1
    # the threshold is the comparison it says: 93 serves at 93, not at 94
    for n, served in ((93, True), (94, False)):
        acc.razor_reuse_set_small_deck_segs(n, n)
        s = _solver(wires, True, GROUNDS[ground])
        s.compute_port_solution()
        assert (s.pair_reuse_declined is None) == served, s.pair_reuse_declined


@_needs_cert
def test_small_decks_step_aside_on_the_cxx_route(monkeypatch):
    acc.razor_reuse_set_small_deck_segs(-1, -1)
    monkeypatch.setattr(razor, "_RAZOR_CXX", True)
    wires = _triangle()
    off = _solve(wires, False, ground=SOMM)
    ctx = razor._cxx_context()
    before = dict(ctx.counters)
    y = _solve(wires, True, ground=SOMM)
    after = ctx.counters
    assert razor._CXX_FRONT_ROUTE.get("front", 0) > 0
    assert after["reuse_declined"] == before["reuse_declined"] + 1
    assert after["reuse_fills"] == before["reuse_fills"]
    assert ctx.reuse_declined_reason == "too few segments to repay the plan (N < 100)"
    np.testing.assert_array_equal(y, off)
    # verify ignores it
    _solve(wires, "verify", ground=SOMM)
    assert ctx.counters["reuse_fills"] == after["reuse_fills"] + 1


# ------------------------------------------------------------ source length


def _small_loop_zin(mode, d, side=0.3):
    """A square loop 0.3 m a side at 137 kHz (radiation resistance ~1e-11
    ohm against ~1.16 ohm of reactance, so R is the answer's cancelling
    part), 110 segments a side, one side lengthened by `d`."""
    p = [[0, 0, 2], [side, 0, 2], [side, side + d, 2], [0, side, 2]]
    wires = [np.array([p[i], p[(i + 1) % 4]], dtype=float) for i in range(4)]
    sol = RazorSolver(
        wires=wires,
        n_per_edge_per_wire=[[110]] * 4,
        feeds=[(0, 0.37 * side, 1 + 0j)],
        wire_radius=0.0005,
        wavelength=2190.0,
        nec5_quadrature=True,
        extended_kernel=True,
        exact_kernel=False,
        razor_pair_reuse=mode,
    )
    return complex(1.0 / sol.compute_port_solution().y[0, 0])


@_needs_cert
def test_segments_of_different_lengths_never_share_a_class():
    """The key carries the source length h exactly, by type, as it does the
    radii (momwire#1971 review). M0 is proportional to h, so h's error counts
    relative to h; a step of 2^-40 times the pair distance D was coarser than
    h's own by D/h, and merged segments whose lengths differ by ~1e-11.

    On this loop the lengthened side's segments shared classes with the
    others', and R moved 8.1e-5 (identically under translation, so not
    noise), while every pair passed `verify` at ~4e-11. With h exact it
    moves ~2e-9, the flag-off fill's own translation noise (1e-9 - 2e-9)."""
    d = 4e-11 * 0.3
    off = _small_loop_zin(False, d)
    on = _small_loop_zin(True, d)
    assert _ran() == 1
    assert abs(on.real - off.real) / off.real < 1e-7


@_needs_cert
def test_the_keys_carry_the_source_length():
    """Both keys, the direct one and the composing ground's remainder one,
    keep sources of different lengths apart. Two parallel horizontal dipoles
    3 m apart over Sommerfeld ground, the second 1e-12 m longer (1e-13
    relative): its pairs then match the first's to well under the position
    step, and under a key quantising h against the pair distance they merged
    with the first's (the shared classes grew 1.3x, the remainder's 1.2x).
    With h exact each length has classes of its own, at least twice the
    twin deck's."""

    def stats(dy):
        w = [
            np.array([[0.0, -5.0, 4.0], [0.0, 5.0, 4.0]]),
            np.array([[3.0, -5.0, 4.0], [3.0, 5.0 + dy, 4.0]]),
        ]
        _solve(w, True, nsegs=41, ground=SOMM)
        return dict(razor._PAIR_REUSE_STATS[0])

    twin, apart = stats(0.0), stats(1e-12)
    assert _ran() == 2
    assert twin["q_classes"] > 0, twin
    assert apart["q_classes"] >= 2 * twin["q_classes"], (twin, apart)
    shared = [st["classes"] - st["near_pairs"] for st in (twin, apart)]
    assert shared[1] >= 2 * shared[0], (twin, apart)
