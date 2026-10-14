"""momwire#1471: a frequency sweep of a buried deck is served from anchor
tables, interpolated in Im(eps_t), instead of one contour fill per frequency.

Three kinds of test, and which is which matters:

* **Policy and bookkeeping, on a stand-in kernel.** `iv_surfaces_direct_below`
  is replaced by a closed form with the real kernel's scaling law, so a grid
  costs microseconds and every evaluation is COUNTED. These say when anchors
  are built, how many, that a grown anchor evaluates only its new rows, and
  that the blend lands on the right lattice with the right scale. The stand-in
  cannot say anything about the real surfaces' smoothness.
* **The real kernel, small.** The scaling law itself, and a grown anchor
  against a fresh one, bit for bit.
* **The real kernel, slow.** Per-surface interpolation error over every region
  the issue's deck reads, and a sweep of that deck through the served seam
  (`momwire.eznec._shell`), counting fills and comparing feed impedance with
  the exact fill.

Every tolerance below is a multiple of a number measured on 2026-10-13
(Rocketlake, the AVX2 accelerator) and written beside it.
"""

import math

import numpy as np
import pytest

from momwire import _somm_disk_cache as dc
from momwire import _sommerfeld as sm
from momwire import _sommerfeld_below as sb
from momwire._sommerfeld import _C_LIGHT, _MU0, _SURF_KEYS

EPS0 = 8.8541878128e-12
EPS_R, SIGMA = 13.0, 0.005  # the issue's ground


def _medium(f_hz, eps_r=EPS_R, sigma=SIGMA):
    omega = 2.0 * math.pi * f_hz
    return complex(eps_r, -sigma / (omega * EPS0)), omega / _C_LIGHT, omega


def _grid(f_hz, r1_wl, **kw):
    """`get_grid_below` for the issue's soil at `f_hz`, sized `r1_wl`
    in-medium wavelengths."""
    eps_t, k2, omega = _medium(f_hz)
    return sb.get_grid_below(eps_t, k2, r1_wl * sb._lambda_m(eps_t, k2), omega, **kw)


@pytest.fixture
def clean(monkeypatch):
    """No grids, no anchors, no history, counters at zero."""
    monkeypatch.setattr(sm, "_GRID_CACHE", {})
    monkeypatch.setattr(sb, "_GRID_CACHE", sm._GRID_CACHE)
    sb.reset_sweep_state()
    for k in sb.SWEEP_STATS:
        monkeypatch.setitem(sb.SWEEP_STATS, k, 0)
    monkeypatch.delenv(sb.ENV_ANCHORS, raising=False)
    yield
    sb.reset_sweep_state()


# The stand-in's phase runs at RATE*(1 + R1/lam_m) radians per unit of
# ln Im(eps_t). The real surfaces' fastest term measures about 6 at the 4 lam_m
# cap; this is the same order on the small grids used here.
RATE = 2.0


@pytest.fixture
def standin(monkeypatch, clean):
    """A closed-form kernel obeying S = omega*mu*G(eps_t; R1/lam_m, theta),
    smooth in ln Im(eps_t), with every evaluated node counted."""
    count = {"nodes": 0, "calls": 0, "eps": []}

    def fake(eps_t, k2, R1, theta, rtol=1e-9, omega=None, mu=_MU0, **_kw):
        R1b, thb = np.broadcast_arrays(np.asarray(R1, float), np.asarray(theta, float))
        count["nodes"] += R1b.size
        count["calls"] += 1
        count["eps"].append(complex(eps_t))
        r = R1b / sb._lambda_m(eps_t, k2)
        s = -complex(eps_t).imag
        phase = RATE * math.log(s) * (1.0 + r) + thb if s > 0 else thb
        g = (1.0 + 0.3 * r) * np.exp(1j * phase) * (1.0 + 0.01 * complex(eps_t).real)
        return {key: omega * mu * (i + 1) * g for i, key in enumerate(_SURF_KEYS)}

    monkeypatch.setattr(sb, "iv_surfaces_direct_below", fake)
    return count


def _outermost_row_wl(grid):
    """The largest R1 any filled region tabulates, in in-medium wavelengths:
    the table edge plus its padding rows."""
    return (
        max(float(reg["r_nodes"][-1]) for reg in grid._regions if reg["filled"])
        / grid.lam_m
    )


def _interp_bound(r1_wl):
    """The Lagrange remainder for the stand-in at mid-interval: |f^(n)|/n!
    times the node polynomial, for a unit-modulus phasor of rate c."""
    n = sb._SOMM_BELOW_ANCHOR_POINTS
    c = RATE * (1.0 + r1_wl)
    node_poly = np.prod([abs(n / 2 - 0.5 - j) for j in range(n)])
    return (c * sb._SOMM_BELOW_ANCHOR_STEP) ** n * node_poly / math.factorial(n)


# ---------------------------------------------------------------------------
# Small parts
# ---------------------------------------------------------------------------


def test_lagrange_weights_reproduce_a_polynomial_and_are_one_hot_on_a_node():
    n = 8
    for t in (3.0, 3.25, 3.5, 3.999):
        w = sb._lagrange_weights(t, n)
        assert abs(w.sum() - 1.0) < 1e-12
        for power in range(1, n):
            assert abs(w @ (np.arange(n) ** power) - t**power) < 1e-9 * 7.0**power
    assert np.array_equal(sb._lagrange_weights(3.0, n), np.eye(n)[3])


def test_the_plan_brackets_the_target_on_a_geometric_ladder():
    eps_t, _k2, _om = _medium(9.65e6)
    plan = sb._anchor_plan(eps_t)
    n = sb._SOMM_BELOW_ANCHOR_POINTS
    s = [-e.imag for e in plan.rungs]
    assert len(s) == n and all(e.real == EPS_R for e in plan.rungs)
    assert np.allclose(np.diff(np.log(s)), sb._SOMM_BELOW_ANCHOR_STEP, rtol=1e-12)
    # A centred stencil: the target sits in the middle interval.
    assert s[n // 2 - 1] <= plan.s < s[n // 2]
    # The same target in a neighbouring process plans the same rungs, bit for
    # bit: the rungs are the disk keys.
    assert sb._anchor_plan(eps_t).rungs == plan.rungs


def test_a_lossless_ground_has_no_ladder_and_keeps_the_exact_fill(standin, monkeypatch):
    assert sb._anchor_plan(4.0 + 0j) is None
    monkeypatch.setenv(sb.ENV_ANCHORS, "on")
    k2 = 2 * math.pi / 30.0
    grid = sb.get_grid_below(4.0 + 0j, k2, 0.4 * sb._lambda_m(4.0, k2), k2 * _C_LIGHT)
    assert type(grid) is sb.SommerfeldGridBelow
    assert sb.SWEEP_STATS["anchor_fills"] == 0


@pytest.mark.parametrize(
    "value,mode",
    [("", "auto"), ("auto", "auto"), ("off", "off"), ("0", "off"), ("on", "on")],
)
def test_the_switch(monkeypatch, value, mode):
    monkeypatch.setenv(sb.ENV_ANCHORS, value)
    assert sb.anchor_mode() == mode


# ---------------------------------------------------------------------------
# Policy, counted (stand-in kernel)
# ---------------------------------------------------------------------------


def _sweep_freqs(n, f0=9.6e6, step=0.003):
    return [f0 * (1.0 + step) ** i for i in range(n)]


def test_frequencies_short_of_the_trigger_are_each_one_exact_fill(standin):
    trigger = sb._SOMM_BELOW_ANCHOR_TRIGGER
    grids = [_grid(f, 0.45) for f in _sweep_freqs(trigger - 1)]
    assert all(type(g) is sb.SommerfeldGridBelow for g in grids)
    assert sb.SWEEP_STATS["exact_grids"] == trigger - 1
    assert sb.SWEEP_STATS["anchor_fills"] == 0
    assert sb.SWEEP_STATS["blended_grids"] == 0


def test_the_trigger_builds_one_stencil_of_anchors_and_then_evaluates_nothing(standin):
    trigger, points = sb._SOMM_BELOW_ANCHOR_TRIGGER, sb._SOMM_BELOW_ANCHOR_POINTS
    freqs = _sweep_freqs(trigger + 20)
    for f in freqs[: trigger - 1]:
        _grid(f, 0.45)
    before = standin["nodes"]
    exact_nodes = before // (trigger - 1)

    first = _grid(freqs[trigger - 1], 0.45)
    assert type(first) is sb.SommerfeldGridBelowBlended
    assert sb.SWEEP_STATS["anchor_fills"] == points
    # An anchor is the size of the exact grid it stands in for, give or take
    # the one row the lattice's own ceil() moves by.
    built = standin["nodes"] - before
    assert 0.9 * points * exact_nodes < built < 1.1 * points * exact_nodes

    # Twenty more points of the sweep span 6 % in frequency, well inside one
    # 10 % ladder step: at most one more rung is ever needed.
    before = standin["nodes"]
    rest = [_grid(f, 0.45) for f in freqs[trigger:]]
    assert all(type(g) is sb.SommerfeldGridBelowBlended for g in rest)
    assert sb.SWEEP_STATS["exact_grids"] == trigger - 1
    assert sb.SWEEP_STATS["anchor_fills"] <= points + 1
    assert standin["nodes"] - before <= 1.1 * exact_nodes
    assert sb.SWEEP_STATS["blended_grids"] == 21


def test_frequencies_far_apart_never_trigger(standin):
    # An octave apart each: no two inside the window, however many there are.
    for i in range(sb._SOMM_BELOW_ANCHOR_TRIGGER + 3):
        assert type(_grid(1.8e6 * 2.0**i, 0.45)) is sb.SommerfeldGridBelow
    assert sb.SWEEP_STATS["anchor_fills"] == 0


def test_another_deck_on_the_same_soil_is_not_this_sweep(standin):
    """A table of a different size is a different deck: its first frequency
    is a lone frequency however long the other deck's sweep was."""
    for f in _sweep_freqs(sb._SOMM_BELOW_ANCHOR_TRIGGER + 2):
        _grid(f, 0.45)
    assert sb.SWEEP_STATS["blended_grids"] == 3
    big = _grid(9.7e6, 3.0)
    assert type(big) is sb.SommerfeldGridBelow


def test_off_is_always_exact_and_on_is_blended_from_the_first_frequency(
    standin, monkeypatch
):
    monkeypatch.setenv(sb.ENV_ANCHORS, "off")
    for f in _sweep_freqs(sb._SOMM_BELOW_ANCHOR_TRIGGER + 2):
        assert type(_grid(f, 0.45)) is sb.SommerfeldGridBelow
    assert sb.SWEEP_STATS["anchor_fills"] == 0
    monkeypatch.setenv(sb.ENV_ANCHORS, "on")
    assert type(_grid(11.0e6, 0.45)) is sb.SommerfeldGridBelowBlended


def test_a_watched_fill_is_exact_whatever_the_switch_says(standin, monkeypatch):
    monkeypatch.setenv(sb.ENV_ANCHORS, "on")
    grid = _grid(9.65e6, 0.45, health=sb.Health())
    assert type(grid) is sb.SommerfeldGridBelow


def test_an_exact_grid_already_held_is_what_is_returned(standin, monkeypatch):
    exact = _grid(9.65e6, 0.45)
    monkeypatch.setenv(sb.ENV_ANCHORS, "on")
    assert _grid(9.65e6, 0.45) is exact


# ---------------------------------------------------------------------------
# The blend lands on the right lattice with the right scale (stand-in kernel)
# ---------------------------------------------------------------------------


def _region_errors(blended, exact):
    """Worst |blend - exact| per region, relative to that surface's largest
    value in the region, over the rows the blended grid serves."""
    out = {}
    for idx, (b, e) in enumerate(zip(blended._regions, exact._regions)):
        if not (b["filled"] and e["filled"]):
            continue
        rows = b["n_r"]
        ev = e["vals"][:, :rows, :]
        scale = np.abs(ev).reshape(len(_SURF_KEYS), -1).max(axis=1)
        err = np.abs(b["vals"] - ev).reshape(len(_SURF_KEYS), -1).max(axis=1)
        out[idx] = err / scale
    return out


@pytest.mark.parametrize("r1_wl", [0.45, 1.3, 3.0])
def test_the_blend_is_the_exact_table_to_the_interpolation_bound(
    standin, monkeypatch, r1_wl
):
    # A frequency whose Im(eps_t) sits mid-interval, where the error peaks.
    plan0 = sb._anchor_plan(_medium(9.65e6)[0])
    n = sb._SOMM_BELOW_ANCHOR_POINTS
    s_mid = math.sqrt(plan0.rungs[n // 2 - 1].imag * plan0.rungs[n // 2].imag)
    f_mid = SIGMA / (2 * math.pi * EPS0 * s_mid)

    monkeypatch.setenv(sb.ENV_ANCHORS, "off")
    exact = _grid(f_mid, r1_wl)
    monkeypatch.setenv(sb.ENV_ANCHORS, "on")
    sm._GRID_CACHE.clear()
    blended = _grid(f_mid, r1_wl)
    assert type(blended) is sb.SommerfeldGridBelowBlended
    for g in (exact, blended):
        g._ensure_for(g.r1_max, g.th_min, 0.5 * math.pi)

    errs = _region_errors(blended, exact)
    # Five theta bands in the inner and near zones, and in the far annulus
    # once the grid reaches past 2 lam_m.
    assert len(errs) == (15 if r1_wl > 2.0 else 10), sorted(errs)
    worst = max(float(e.max()) for e in errs.values())
    bound = _interp_bound(_outermost_row_wl(blended))
    # Interpolated, not copied (a zero here would mean the anchors were never
    # read), and no worse than the remainder term allows.
    assert 1e-3 * bound < worst <= 1.05 * bound + 1e-13, (worst, bound)
    assert sb.SWEEP_STATS["exact_regions_on_blended"] == 0


def test_the_blend_carries_frequency_and_mu_through_the_scale(standin, monkeypatch):
    """Anchors are tabulated at one wavelength and mu0. A target at another
    frequency AND another permeability must come out scaled by omega*mu and
    on its own lam_m lattice."""
    eps_t, k2, omega = _medium(3.6e6)
    mu = 1.7 * _MU0
    r1 = 0.6 * sb._lambda_m(eps_t, k2)
    monkeypatch.setenv(sb.ENV_ANCHORS, "off")
    exact = sb.get_grid_below(eps_t, k2, r1, omega, mu=mu)
    monkeypatch.setenv(sb.ENV_ANCHORS, "on")
    sm._GRID_CACHE.clear()
    blended = sb.get_grid_below(eps_t, k2, r1, omega, mu=mu)
    assert type(blended) is sb.SommerfeldGridBelowBlended
    assert blended.lam_m == exact.lam_m and blended.r1_max == exact.r1_max
    worst = max(float(e.max()) for e in _region_errors(blended, exact).values())
    assert worst <= 1.05 * _interp_bound(_outermost_row_wl(blended)) + 1e-13

    rng = np.random.default_rng(1471)
    r = rng.uniform(0.0, exact.r1_max, 200)
    th = rng.uniform(exact._th_split, 0.5 * math.pi, 200)
    a, b = exact.eval(r, th), blended.eval(r, th)
    for key in _SURF_KEYS:
        assert np.max(np.abs(a[key] - b[key]) / np.abs(a[key])) < 1e-6


def test_every_wavelength_gets_a_centred_stencil_at_its_zone_edges(
    standin, monkeypatch
):
    """`n_r` is a ceil of a float quotient and comes out a row long on some
    lam_m, on the target or on an anchor. Whichever has the extra row, the
    blended grid must still read the same four rows the exact one reads at
    the outer edge of every zone. Scanned over 60 frequencies."""
    row_counts = set()
    for i in range(60):
        f = 3.5e6 * 1.0371**i
        monkeypatch.setenv(sb.ENV_ANCHORS, "off")
        exact = _grid(f, 3.0)
        monkeypatch.setenv(sb.ENV_ANCHORS, "on")
        sm._GRID_CACHE.pop(next(reversed(sm._GRID_CACHE)))
        blended = _grid(f, 3.0)
        assert type(blended) is sb.SommerfeldGridBelowBlended
        edges = np.array([blended.r_break, blended.r_near, blended.r1_max])
        r = np.concatenate([edges, edges * (1 - 1e-12), edges * 0.999])
        th = np.full(r.shape, math.radians(40.0))
        a, b = exact.eval(r, th), blended.eval(r, th)
        for key in _SURF_KEYS:
            assert np.max(np.abs(a[key] - b[key]) / np.abs(a[key])) < 1e-3
        row_counts.add(tuple(reg["n_r"] for reg in exact._regions))
    # The hazard is real on this scan: the exact lattice itself does not have
    # one row count.
    assert len(row_counts) > 1
    assert sb.SWEEP_STATS["exact_regions_on_blended"] == 0


def test_a_region_off_the_anchor_lattice_is_filled_exactly(standin, monkeypatch):
    """The far annulus of a grid smaller than 2 lam_m starts at the grid's own
    edge, not at 2 lam_m, so no anchor row lines up with it. Nothing routes
    there, and if something did the answer would be the exact fill."""
    monkeypatch.setenv(sb.ENV_ANCHORS, "on")
    grid = _grid(9.65e6, 1.3)
    far = sb.region_index(sb._ZONE_FAR, sb._BAND_STEEP)
    grid._fill_region(far)
    assert sb.SWEEP_STATS["exact_regions_on_blended"] == 1
    assert grid._regions[far]["filled"]


# ---------------------------------------------------------------------------
# Anchors grow (stand-in for the count, real kernel for the bits)
# ---------------------------------------------------------------------------


def test_a_larger_deck_grows_the_anchors_by_their_new_rows_only(standin, monkeypatch):
    monkeypatch.setenv(sb.ENV_ANCHORS, "on")
    points = sb._SOMM_BELOW_ANCHOR_POINTS
    _grid(9.65e6, 0.45)
    small = standin["nodes"]
    assert sb.SWEEP_STATS["anchor_fills"] == points
    _grid(9.65e6, 1.3)
    grown = standin["nodes"] - small
    assert sb.SWEEP_STATS["anchor_fills"] == 2 * points

    sb.reset_sweep_state()
    sm._GRID_CACHE.clear()
    standin["nodes"] = 0
    _grid(9.65e6, 1.3)
    fresh = standin["nodes"]
    # Growing 0.45 -> 1.3 evaluates what a fresh 1.3 fill evaluates, less what
    # the 0.45 anchors already held.
    assert grown == fresh - small
    assert 0 < grown < fresh


@pytest.mark.slow
def test_a_grown_grid_is_the_fresh_grid_bit_for_bit(clean):
    """REAL kernel. A node's value does not depend on which batch evaluated
    it, so a grid seeded with a smaller one's rows is the grid a fresh fill
    builds -- which is what lets a grown anchor be stored under one key."""
    eps_t = complex(EPS_R, -8.66)
    k2 = 2 * math.pi
    lam_m = sb._lambda_m(eps_t, k2)
    small = sb.SommerfeldGridBelow(eps_t, k2, 0.35 * lam_m)
    before = sb.SWEEP_STATS["direct_nodes"]
    grown = sb.SommerfeldGridBelow(eps_t, k2, 0.6 * lam_m, seed=small)
    grown_nodes = sb.SWEEP_STATS["direct_nodes"] - before
    fresh = sb.SommerfeldGridBelow(eps_t, k2, 0.6 * lam_m)
    fresh_nodes = sb.SWEEP_STATS["direct_nodes"] - before - grown_nodes
    assert 0 < grown_nodes < 0.3 * fresh_nodes
    assert dc.grid_digest(grown) == dc.grid_digest(fresh)


# ---------------------------------------------------------------------------
# The scaling law (real kernel)
# ---------------------------------------------------------------------------


def test_the_normalised_table_does_not_depend_on_frequency(clean):
    """S/(omega*mu) on the lam_m lattice is a function of eps_t alone. This is
    the law the anchors rest on. Measured on this test's grids (eps_t =
    13 - 9.31j, free-space wavelengths 15 and 160 m, the four eager regions):
    4.8e-10 of the surface's largest value at worst -- the contour's own
    rtol = 1e-9, not a law that holds only roughly."""
    eps_t = complex(EPS_R, -9.314943868663876)
    tables = []
    for lam0 in (15.0, 160.0):
        k2 = 2 * math.pi / lam0
        g = sb.SommerfeldGridBelow(eps_t, k2, 0.35 * sb._lambda_m(eps_t, k2))
        tables.append([r["vals"] / (g.omega * g.mu) for r in g._regions if r["filled"]])
    worst = 0.0
    for a, b in zip(*tables):
        rows = min(a.shape[1], b.shape[1])
        a, b = a[:, :rows], b[:, :rows]
        scale = np.abs(a).reshape(4, -1).max(axis=1)[:, None, None]
        worst = max(worst, float((np.abs(a - b) / scale).max()))
    assert 0.0 < worst < 1e-8, worst


# ---------------------------------------------------------------------------
# Across processes: the disk level
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv(dc.ENV_ENABLE, "1")
    monkeypatch.setenv(dc.ENV_DIR, str(tmp_path / "somm"))
    monkeypatch.setattr(dc, "_WRITE_WARNED", set())
    return tmp_path / "somm"


def _new_process():
    """What a second worker process starts with: nothing in memory."""
    sm._GRID_CACHE.clear()
    sb.reset_sweep_state()


def test_a_second_process_reads_the_anchors_and_fills_nothing(standin, store):
    trigger = sb._SOMM_BELOW_ANCHOR_TRIGGER
    freqs = _sweep_freqs(trigger + 2)
    for f in freqs[:-1]:
        _grid(f, 0.45)
    assert sb.SWEEP_STATS["blended_grids"] == 2

    _new_process()
    fills, nodes = sb.SWEEP_STATS["anchor_fills"], standin["nodes"]
    grid = _grid(freqs[-1], 0.45)
    assert type(grid) is sb.SommerfeldGridBelowBlended
    assert sb.SWEEP_STATS["anchor_fills"] == fills
    assert standin["nodes"] == nodes
    # Blended grids are never written: only exact grids and anchors are.
    names = sorted(p.name for p in store.iterdir())
    assert sum(n.endswith(".npz") for n in names) == (trigger - 1) + (
        sb._SOMM_BELOW_ANCHOR_POINTS
    )


def test_a_crew_of_processes_counts_as_one_sweep(standin, store):
    """Each worker sees a few of the sweep's frequencies. The count that
    trips the trigger is the crew's, read from the shared directory."""
    trigger = sb._SOMM_BELOW_ANCHOR_TRIGGER
    freqs = _sweep_freqs(trigger)
    for f in freqs[:-1]:
        _new_process()  # every frequency lands on a fresh worker
        assert type(_grid(f, 0.45)) is sb.SommerfeldGridBelow
    _new_process()
    assert type(_grid(freqs[-1], 0.45)) is sb.SommerfeldGridBelowBlended


def test_an_exact_grid_on_disk_is_preferred_to_a_blend(standin, store):
    freqs = _sweep_freqs(sb._SOMM_BELOW_ANCHOR_TRIGGER + 1)
    for f in freqs:
        _grid(f, 0.45)
    _new_process()
    nodes = standin["nodes"]
    assert type(_grid(freqs[0], 0.45)) is sb.SommerfeldGridBelow
    assert standin["nodes"] == nodes


def test_a_lazy_region_another_process_filled_is_adopted_not_refilled(
    standin, store, monkeypatch
):
    monkeypatch.setenv(sb.ENV_ANCHORS, "on")
    mid = sb.region_index(sb._ZONE_INNER, sb._BAND_MID)
    first = _grid(9.65e6, 0.45)
    keys = list(sb._ANCHOR_CACHE)
    first._fill_region(mid)
    # A worker that built its anchors before the band was filled elsewhere.
    sm._GRID_CACHE.clear()
    sb.reset_sweep_state()
    for key in keys:
        stale = dc.fetch(key)
        stale._regions[mid]["filled"] = False
        stale._regions[mid]["vals"] = np.full_like(stale._regions[mid]["vals"], np.nan)
        sb._ANCHOR_CACHE[key] = stale
    nodes = standin["nodes"]
    second = _grid(9.66e6, 0.45)
    second._fill_region(mid)
    assert standin["nodes"] == nodes
    assert np.all(np.isfinite(second._regions[mid]["vals"]))


def test_the_seen_list_is_shared_and_bounded(store):
    for i in range(dc._SEEN_MAX + 5):
        seen = dc.note_seen("tag", (float(i), 1.0))
    assert len(seen) == dc._SEEN_MAX and seen[-1] == (float(dc._SEEN_MAX + 4), 1.0)
    assert dc.note_seen("tag", seen[3]) == seen
    assert dc.note_seen("other", (1.0, 2.0)) == [(1.0, 2.0)]


def test_each_worker_starts_its_walk_at_its_own_anchor(standin, monkeypatch):
    """A crew that needs one stencil spreads out over it: the walk starts at
    the process's own offset and wraps around."""
    monkeypatch.setenv(sb.ENV_ANCHORS, "on")
    n = sb._SOMM_BELOW_ANCHOR_POINTS
    monkeypatch.setattr(sb.os, "getpid", lambda: 3 * n + 5)
    del standin["eps"][:]
    _grid(9.65e6, 0.45)
    plan = sb._anchor_plan(_medium(9.65e6)[0])
    order = []
    for e in standin["eps"]:
        if e not in order:
            order.append(e)
    # The first rung filled is the one whose GLOBAL ladder index is this
    # process's offset, so workers at shifted stencils still spread out.
    first = (5 - plan.n0) % n
    assert (plan.n0 + first) % n == 5
    assert order == [plan.rungs[(first + j) % n] for j in range(n)]


def test_an_anchor_another_worker_stored_mid_walk_is_read_not_filled(
    standin, store, monkeypatch
):
    """Worker A stores the whole stencil. Worker B, already holding a stale
    view of none of it, reads each anchor from disk just before it would fill
    it, and fills nothing."""
    monkeypatch.setenv(sb.ENV_ANCHORS, "on")
    _grid(9.65e6, 0.45)
    _new_process()
    monkeypatch.setattr(sb.os, "getpid", lambda: 12345)
    fills, nodes = sb.SWEEP_STATS["anchor_fills"], standin["nodes"]
    assert type(_grid(9.66e6, 0.45)) is sb.SommerfeldGridBelowBlended
    assert sb.SWEEP_STATS["anchor_fills"] == fills
    assert standin["nodes"] == nodes


def test_with_the_store_off_nothing_is_written(standin, tmp_path, monkeypatch):
    monkeypatch.setenv(dc.ENV_ENABLE, "0")
    monkeypatch.setenv(dc.ENV_DIR, str(tmp_path / "somm"))
    for f in _sweep_freqs(sb._SOMM_BELOW_ANCHOR_TRIGGER + 1):
        _grid(f, 0.45)
    assert sb.SWEEP_STATS["blended_grids"] == 2
    assert not (tmp_path / "somm").exists()


# ---------------------------------------------------------------------------
# The real kernel, slow
# ---------------------------------------------------------------------------

# Per-surface interpolation error of the shipped scheme (10 % ladder, eight
# rungs), mid-interval, eps_t = 13 - 8.66j: the hardest of three soils
# measured on the ladder study (20 - 77j and 5 - 0.9j read the same or
# lower). Worst node of each region relative to its surface's largest value,
# this test's own configuration: 1.5e-8, I_z^V in the far annulus's grazing
# band; the four eager regions read 0.4e-9 to 7.7e-9. The tables' own spatial
# interpolation error is 3.6e-5 on that scale.
_SURFACE_TOL = 1e-7  # 6.6x the worst measured


@pytest.mark.slow
def test_per_surface_interpolation_error_over_the_issues_regions(clean, monkeypatch):
    plan0 = sb._anchor_plan(complex(EPS_R, -8.66))
    n = sb._SOMM_BELOW_ANCHOR_POINTS
    s_mid = math.sqrt(plan0.rungs[n // 2 - 1].imag * plan0.rungs[n // 2].imag)
    f_mid = SIGMA / (2 * math.pi * EPS0 * s_mid)

    monkeypatch.setenv(sb.ENV_ANCHORS, "off")
    exact = _grid(f_mid, 3.05)
    monkeypatch.setenv(sb.ENV_ANCHORS, "on")
    sm._GRID_CACHE.clear()
    blended = _grid(f_mid, 3.05)
    # What the issue's deck reads: theta from 1 deg up, out to 3 lam_m.
    for g in (exact, blended):
        g._ensure_for(g.r1_max, math.radians(5.0), 0.5 * math.pi)
    errs = _region_errors(blended, exact)
    assert len(errs) == 6
    worst = {idx: float(e.max()) for idx, e in errs.items()}
    assert max(worst.values()) < _SURFACE_TOL, worst
    # Interpolated, not copied.
    assert max(worst.values()) > 1e-12, worst
    for idx, e in errs.items():
        assert e.shape == (len(_SURF_KEYS),)


# The issue's deck (Ward AE6TY's geometry, synthetic): a wire 10 m up, a wire
# 1 m down, joined by a vertical wire through the surface.
_DECK = """CM momwire#1471
CE
GW 1 26 -10 0 10 10 0 10 6.45426e-4
GW 2 26 -10 0 -1 10 0 -1 6.45426e-4
GW 3 2 -10 0 -1 -10 0 0 6.45426e-4
GW 4 13 -10 0 10 -10 0 0 6.45426e-4
GE -1
GN 2 0 0 0 13.0 0.005
FR 0 1 0 0 {f:.9f} 1
EX 0 1 13 2 1.0
XQ
EN
"""


def _feed_z(f_mhz):
    from momwire.eznec import _shell

    answered = _shell._answer(_DECK.format(f=f_mhz), basis="bspline")
    assert not isinstance(answered, str), answered[-800:]
    port = answered[1][0].sources[0]
    return complex(port.voltage) / complex(port.current)


# Feed impedance from blended tables against the exact fill, measured over
# 9-12 MHz on Ward's deck and 13-15 MHz on Dan AC6LA's buried-radial vertical
# (tests/fixtures/eznec_crossing_source_1282/dan_end1.nec), worst |dZ|/|Z|:
# 2.8e-12 and 5.1e-12; this test's five points read 1.1e-12 to 2.1e-12. (A
# cubic on the same ladder reads 7.4e-9 and a linear blend 1.6e-6.) The bar
# the issue set is 1e-6.
_Z_TOL = 1e-9  # 200x the worst measured, 1000x inside the bar


@pytest.mark.slow
def test_a_sweep_of_the_issues_deck_counts_its_fills_and_keeps_its_impedance(
    clean, monkeypatch
):
    """Through the served seam, in the default mode. Fills are COUNTED: a
    sweep of trigger + 4 frequencies costs trigger - 1 exact grids and one
    stencil of anchors, where it used to cost trigger + 4 exact grids."""
    trigger, points = sb._SOMM_BELOW_ANCHOR_TRIGGER, sb._SOMM_BELOW_ANCHOR_POINTS
    freqs = [9.60 + 0.03 * i for i in range(trigger + 4)]
    swept = [_feed_z(f) for f in freqs]
    assert sb.SWEEP_STATS["exact_grids"] == trigger - 1
    assert sb.SWEEP_STATS["blended_grids"] == 5
    assert points <= sb.SWEEP_STATS["anchor_fills"] <= 2 * (points + 1)
    assert sb.SWEEP_STATS["exact_regions_on_blended"] == 0
    blended_nodes = sb.SWEEP_STATS["direct_nodes"]

    # The same five frequencies from exact fills.
    monkeypatch.setenv(sb.ENV_ANCHORS, "off")
    sm._GRID_CACHE.clear()
    for k in sb.SWEEP_STATS:
        sb.SWEEP_STATS[k] = 0
    exact = [_feed_z(f) for f in freqs[trigger - 1 :]]
    assert sb.SWEEP_STATS["exact_grids"] == 5
    per_fill = sb.SWEEP_STATS["direct_nodes"] / 5
    # The anchors are sized like the exact grids, so the whole sweep's contour
    # work is about (trigger - 1) + points fills' worth, plus one rung.
    assert blended_nodes < (trigger - 1 + points + 1.5) * per_fill

    rel = [abs(a - b) / abs(b) for a, b in zip(swept[trigger - 1 :], exact)]
    assert max(rel) < _Z_TOL, rel
    assert max(rel) > 0.0  # two different tables were read
