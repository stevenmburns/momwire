"""`pair_reuse`'s allowance on the sinusoidal fill (momwire#2123, which
replaced #2019's fixed 256 MB budget).

Each plan (the direct block's, the PEC image's, the reflected term's) keeps
at most half its pairs as classes, `_sin_pair_reuse.class_cap` (2x
compression), and its fill is held to `plan_allowance`, 4 B per pair plus
56 B per class the cap allows. A deck past either says so in
`pair_reuse_declined` and fills bit for bit as with reuse off; a deck inside
both keeps its classes. The allowance is moved here (`ALLOWANCE_BYTES`) so
that small decks reach both sides.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from momwire import _sin_pair_reuse as pr
from momwire._accel import acc
from momwire.sinusoidal import SinusoidalSolver

# An experimental feature (`momwire._features`, reach 4: through the drop-in
# on Linux and Windows).
pytestmark = pytest.mark.reach(4)

LAM = 10.7
GROUNDS = {
    "free": {},
    "pec": dict(ground_z=0.0),
    "refl-coef": dict(ground_z=0.0, ground_eps=(13.0, 0.005)),
    "sommerfeld": dict(
        ground_z=0.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld"
    ),
}
BIG = 1 << 40


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("MOMWIRE_X_PAIR_REUSE", raising=False)
    pr.clear_shared_plans()
    yield
    pr.clear_shared_plans()


def _yagi(taper=0.0):
    """Five elements, 405 segments: equal (`taper` 0), whose pairs compress
    far past the allowance's bound, or each its own length (`taper` 0.03),
    few pairs shared."""
    wires = [
        np.array(
            [
                [0.21 * LAM * i, -0.25 * LAM * (1 - taper * i), 8.0],
                [0.21 * LAM * i, 0.25 * LAM * (1 - taper * i), 8.0],
            ]
        )
        for i in range(5)
    ]
    return dict(
        wires=wires,
        n_per_edge_per_wire=[[81]] * 5,
        feed_wire_index=1,
        wavelength=LAM,
        wire_radius=2e-3,
    )


def _fill(pair_reuse=None, taper=0.0, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = SinusoidalSolver(**_yagi(taper), **kw, pair_reuse=pair_reuse)
        geom = s._build_geometry()
        with s._operating_medium(geom) as medium:
            G, _ = s._assemble_Z(geom, s.k, s._medium_eta(medium))
    return G, s


def _direct_plan(allowance=None):
    """The free-space deck's direct plan, through `build_plan` itself."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = SinusoidalSolver(**_yagi())
        geom = s._build_geometry()
    c = np.ascontiguousarray(geom["seg_centers"], dtype=np.float64)
    t = np.ascontiguousarray(geom["seg_tangents"], dtype=np.float64)
    h = np.ascontiguousarray(geom["seg_h"], dtype=np.float64)
    a = np.full(c.shape[0], 2e-3)
    src_type, obs_type, src_rev = pr.reversible_segment_types(h, a)
    return pr.build_plan(
        c, t, c, t, h, a, src_type, obs_type, 64, src_rev=src_rev, allowance=allowance
    )


def _random_wires(n=14, segs=31, seed=2123):
    """Straight wires in random directions: nearly every far pair its own
    class, 434 segments (past SMALL_DECK_SEGS_FREE)."""
    rng = np.random.default_rng(seed)
    wires = []
    for _ in range(n):
        c = rng.uniform(-3.0, 3.0, 3) + np.array([0.0, 0.0, 8.0])
        d = rng.normal(size=3)
        d *= 0.2 * LAM / np.linalg.norm(d)
        wires.append(np.stack([c - d, c + d]))
    return dict(wires=wires, n_per_edge_per_wire=[[segs]] * n, feed_wire_index=n // 2)


def test_the_allowance_is_the_cores():
    """One allowance for every basis and route (`_pair_reuse_core.h`)."""
    assert pr.CLASS_CAP_COMPRESSION == acc.pair_reuse_class_cap_compression == 2
    assert pr.PLAN_FLOOR_BYTES_PER_CLASS == acc.pair_reuse_plan_floor_bytes_per_class
    assert pr.PLAN_FLOOR_BYTES_PER_CLASS == pr.VALUE_BYTES + 8 == 56
    assert acc.pair_reuse_cap_compression() == 2
    n = 405 * 405
    assert pr.class_cap(n) == n // 2
    # 4 B a pair for the map, 56 B for each class the cap allows: 32 B a pair
    assert pr.plan_allowance(n) == 4 * n + 56 * (n // 2)
    assert pr.plan_allowance(2 * n) == 64 * n


def test_the_reasons_are_the_cores_words():
    assert pr.cap_reason(164_025) == (
        "more than 82012 classes for 164025 pairs (the cap is 2x compression)"
    )
    assert pr.allowance_reason(1 << 19) == (
        "the kept plan would exceed its allowance of 0.5 MB"
    )


@pytest.mark.parametrize("ground", sorted(GROUNDS))
def test_a_deck_past_the_allowance_declines_and_fills_as_off(ground, monkeypatch):
    base, _ = _fill(**GROUNDS[ground])
    allowance = 700_000  # the class map fits, the plan's fill does not
    monkeypatch.setattr(pr, "ALLOWANCE_BYTES", allowance)
    on, s = _fill(pair_reuse=True, **GROUNDS[ground])
    assert s.pair_reuse_declined == pr.allowance_reason(allowance)
    assert s.pair_reuse_census == {}
    assert on.tobytes() == base.tobytes()


@pytest.mark.parametrize("ground", sorted(GROUNDS))
def test_a_deck_inside_the_allowance_reuses_as_without_one(ground, monkeypatch):
    """The allowance's checks move no class: the same bytes as an allowance
    that could never bind, and the reuse ran."""
    on, s = _fill(pair_reuse=True, **GROUNDS[ground])
    assert s.pair_reuse_declined is None
    assert "direct" in s.pair_reuse_census
    for plan in (p for _, p in s._sin_reuse_plans.values()):
        assert pr.fill_nbytes(plan) <= pr.plan_allowance(plan.n_pairs)
        assert plan.n_classes <= pr.class_cap(plan.n_pairs)
    pr.clear_shared_plans()
    monkeypatch.setattr(pr, "ALLOWANCE_BYTES", BIG)
    lifted, s = _fill(pair_reuse=True, **GROUNDS[ground])
    assert s.pair_reuse_declined is None
    assert on.tobytes() == lifted.tobytes()


def test_verify_is_held_to_the_allowance(monkeypatch):
    base, _ = _fill()
    monkeypatch.setattr(pr, "ALLOWANCE_BYTES", 700_000)
    on, s = _fill(pair_reuse="verify")
    assert s.pair_reuse_declined == pr.allowance_reason(700_000)
    assert on.tobytes() == base.tobytes()


def test_the_class_map_alone_declines_before_any_census():
    got = _direct_plan(allowance=4 * 405 * 405 - 1)
    assert got == pr.Declined(pr.allowance_reason(4 * 405 * 405 - 1))
    # the map fits: the census runs, and the plan's fill does not
    got = _direct_plan(allowance=4 * 405 * 405 + 1)
    assert got == pr.Declined(pr.allowance_reason(4 * 405 * 405 + 1))


def test_a_plan_is_held_to_its_fill_exactly():
    plan = _direct_plan()
    need = pr.fill_nbytes(plan)
    assert need == pr.plan_nbytes(plan) + 48 * (plan.n_classes + plan.n_near)
    assert isinstance(_direct_plan(allowance=need - 1), pr.Declined)
    served = _direct_plan(allowance=need)
    assert not isinstance(served, pr.Declined)
    assert served.cls.tobytes() == plan.cls.tobytes()
    assert served.src_c.tobytes() == plan.src_c.tobytes()


@pytest.mark.parametrize("ground", ["free", "sommerfeld"])
def test_a_deck_short_of_2x_compression_declines_on_the_cap(ground, monkeypatch):
    """Random wires: nearly every far pair is its own class. At the default
    allowance the census stops first on the allowance (the least its plan
    could keep, 104 B a class, passes 32 B a pair at ~0.27 M N classes,
    before the cap's 0.5 M N): that is the sinusoidal plan's common case.
    With the allowance lifted it passes M N / 2 classes and stops there, the
    cap named; either way the fill is the fill without the reuse. Under a
    cap that binds nothing too (`pair_reuse_cap_compression(1)`), the same
    deck's plan keeps more than half its pairs: the cap is what declined it."""
    kw = {**_random_wires(), **GROUNDS[ground]}

    def fill(pair_reuse=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = SinusoidalSolver(**{**_yagi(), **kw}, pair_reuse=pair_reuse)
            geom = s._build_geometry()
            with s._operating_medium(geom) as medium:
                G, _ = s._assemble_Z(geom, s.k, s._medium_eta(medium))
        return G, s

    base, s = fill()
    n = s._build_geometry()["n_segs"]
    on, s = fill(pair_reuse=True)
    assert s.pair_reuse_declined == pr.allowance_reason(pr.plan_allowance(n * n))
    assert on.tobytes() == base.tobytes()
    pr.clear_shared_plans()
    monkeypatch.setattr(pr, "ALLOWANCE_BYTES", BIG)
    on, s = fill(pair_reuse=True)
    assert s.pair_reuse_declined == pr.cap_reason(n * n)
    assert s.pair_reuse_census == {}
    assert on.tobytes() == base.tobytes()
    pr.clear_shared_plans()
    acc.pair_reuse_cap_compression(1)
    try:
        _, s = fill(pair_reuse=True)
        assert s.pair_reuse_declined is None
        # the direct block: more than half its pairs are classes
        assert s.pair_reuse_census["direct"][1] > n * n // 2
    finally:
        acc.pair_reuse_cap_compression(-1)
        pr.clear_shared_plans()


@pytest.mark.parametrize("ground", sorted(GROUNDS))
def test_a_deck_short_of_the_plans_bytes_declines_on_the_allowance(ground):
    """The tapered elements compress 3.04x (free space: 53,895 classes of
    164,025 pairs), past the cap, but the sinusoidal direct plan keeps
    ~103 B a class beyond its map (one wavenumber's values 48 B, the
    representative 8 B, the canonical sources 48 B), where the allowance
    counts 56 B: its 6.55 MB fill passes the 5.25 MB allowance, so it
    declines on bytes, not on the cap, and fills as off. (It was served
    under #2019's 256 MB budget.)"""
    base, _ = _fill(taper=0.03, **GROUNDS[ground])
    on, s = _fill(pair_reuse=True, taper=0.03, **GROUNDS[ground])
    assert s.pair_reuse_declined == pr.allowance_reason(pr.plan_allowance(405**2))
    assert on.tobytes() == base.tobytes()


def test_each_block_is_held_to_its_own_allowance(monkeypatch):
    """The direct plan and the PEC image's are each held to their own
    allowance, not to one shared between them: an allowance one of them
    passes declines the fill whole (so no block is reused alone), and one
    both fit serves both, though their sum passes it."""
    base, _ = _fill(ground_z=0.0)
    _, s = _fill(pair_reuse=True, ground_z=0.0)
    needs = [pr.fill_nbytes(p) for _, p in s._sin_reuse_plans.values()]
    assert len(needs) == 2 and needs[0] != needs[1], needs
    pr.clear_shared_plans()
    monkeypatch.setattr(pr, "ALLOWANCE_BYTES", min(needs))
    on, s = _fill(pair_reuse=True, ground_z=0.0)
    assert s.pair_reuse_declined == pr.allowance_reason(min(needs))
    assert s.pair_reuse_census == {}
    assert on.tobytes() == base.tobytes()
    pr.clear_shared_plans()
    monkeypatch.setattr(pr, "ALLOWANCE_BYTES", max(needs))
    assert sum(needs) > max(needs)
    _, s = _fill(pair_reuse=True, ground_z=0.0)
    assert s.pair_reuse_declined is None
    assert set(s.pair_reuse_census) == {"direct", "image"}


def test_a_declined_census_is_not_repeated(monkeypatch):
    """A solver per frequency finds the first one's verdict in the cache."""
    acc.sin_pair_reuse_stats(True)
    monkeypatch.setattr(pr, "ALLOWANCE_BYTES", 700_000)
    _fill(pair_reuse=True)
    _, s = _fill(pair_reuse=True)
    stats = acc.sin_pair_reuse_stats()
    assert stats["built"] == stats["declined"] == 1 and stats["found"] == 1
    assert s.pair_reuse_declined == pr.allowance_reason(700_000)


def _census(n, budget, band_bytes, base=0, per=0, floor_per=0, workers=8):
    """The census's band budget alone (`pair_reuse::BandBudget`), the census
    holding `base` plus `per` for each band folded and its floor
    `floor_per` for each."""
    return acc.pair_reuse_band_budget(
        n, -1 if budget is None else budget, band_bytes, workers, base, per, floor_per
    )


def test_no_budget_folds_every_band_in_order():
    out = _census(40, None, 0)
    assert out["peak"] is not None
    assert out["started"] == out["folded"] == 40


def test_a_census_that_fits_folds_every_band_in_order():
    out = _census(50, 100 + 7 * 50 + 30, 10, base=100, per=7)
    assert out["peak"] is not None and out["peak"] <= 100 + 7 * 50 + 30
    assert out["folded"] == 50


def test_the_count_stops_the_census_once_it_passes_the_budget():
    out = _census(80, 100 + 7 * 20, 0, base=100, per=7)
    assert out["peak"] is None
    assert out["folded"] == 21  # the fold that passed it is the last
    assert out["started"] < 80


def test_the_plans_floor_stops_the_census():
    out = _census(80, 5 * 30, 0, floor_per=5)
    assert out["peak"] is None
    assert out["folded"] == 31


def test_a_band_starts_only_where_it_fits():
    """Each band in flight counts `band_bytes` until it is folded: under a
    budget with room for two beside what is held, no more than two count at
    once, however many workers there are."""
    out = _census(30, 1000 + 2 * 50, 50, base=1000)
    assert out["peak"] is not None and out["peak"] <= 1000 + 2 * 50
    assert out["max_in_flight"] == 2
    assert out["folded"] == 30


def test_a_census_with_no_room_for_one_band_runs_none():
    out = _census(10, 1000 + 49, 50, base=1000)
    assert out["peak"] is None
    assert out["started"] == 0


def test_the_plans_kept_between_solvers_are_a_fixed_cap():
    """A long-lived process keeps 256 MB of old plans whatever the decks'
    allowances, on the sinusoidal solver and on razor2p: one cache, the
    common core's (momwire#2026, #2123)."""
    assert pr.SHARED_PLAN_BYTES == 256 << 20
    header = Path(pr.__file__).with_name("_pair_reuse_core.h").read_text()
    assert "constexpr size_t SHARED_PLAN_BYTES = size_t(256) << 20;" in header


def test_a_fills_second_block_keeps_its_first_in_the_cache():
    """Under a cache of one byte, a PEC deck's two plans (the direct block
    and the image, each its own census) are both kept for the next solver:
    the image's is built with `keep=2`, so the direct one, just used, is not
    dropped for it (and then re-censused at every frequency)."""

    def fill():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = SinusoidalSolver(**_yagi(), ground_z=0.0, pair_reuse=True)
            s._assemble_Z(s._build_geometry(), s.k)
        assert set(s.pair_reuse_census) == {"direct", "image"}

    acc.sin_pair_reuse_configure(cache_bytes=1)
    try:
        acc.sin_pair_reuse_stats(True)
        fill()
        fill()  # a solver per frequency: both found
        stats = acc.sin_pair_reuse_stats()
        assert (stats["built"], stats["found"]) == (2, 2)
    finally:
        acc.sin_pair_reuse_configure(cache_bytes=-1, clear=True)


def test_the_cache_drops_the_oldest_plan_past_its_budget():
    """Under a cache of one byte the newest plan is kept and the one before
    it dropped (the common core's, `_pair_reuse_core.h`)."""

    def fill(radius):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = SinusoidalSolver(**{**_yagi(), "wire_radius": radius}, pair_reuse=True)
            s._assemble_Z(s._build_geometry(), s.k)

    acc.sin_pair_reuse_configure(cache_bytes=1)
    try:
        acc.sin_pair_reuse_stats(True)
        fill(2e-3)
        fill(3e-3)  # another deck
        fill(3e-3)  # the newest is kept
        assert acc.sin_pair_reuse_stats()["built"] == 2
        fill(2e-3)  # the first was dropped
        assert acc.sin_pair_reuse_stats()["built"] == 3
    finally:
        acc.sin_pair_reuse_configure(cache_bytes=-1, clear=True)
