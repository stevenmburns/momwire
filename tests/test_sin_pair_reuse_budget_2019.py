"""`pair_reuse`'s byte budget on the sinusoidal fill (momwire#2019).

The census and the plan of one fill are held to
`_sin_pair_reuse.PLAN_BUDGET_BYTES`, razor-2p's budget. A deck past it says so
in `pair_reuse_declined` and fills bit for bit as with reuse off; a deck
inside it keeps its classes. The budget is moved here so that small decks
reach both sides.
"""

from __future__ import annotations

import threading
import time
import warnings
from pathlib import Path

import numpy as np
import pytest

from momwire import _sin_pair_reuse as pr
from momwire import razor
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


def _yagi():
    """Five elements, each its own length: 405 segments, few pairs shared."""
    wires = [
        np.array(
            [
                [0.21 * LAM * i, -0.25 * LAM * (1 - 0.03 * i), 8.0],
                [0.21 * LAM * i, 0.25 * LAM * (1 - 0.03 * i), 8.0],
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


def _fill(pair_reuse=None, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = SinusoidalSolver(**_yagi(), **kw, pair_reuse=pair_reuse)
        geom = s._build_geometry()
        with s._operating_medium(geom) as medium:
            G, _ = s._assemble_Z(geom, s.k, s._medium_eta(medium))
    return G, s


def _reason(budget):
    return f"the plan and its census would exceed the {budget / 2**20:.6g} MB budget"


def _direct_plan(budget=None):
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
        c, t, c, t, h, a, src_type, obs_type, 64, src_rev=src_rev, budget=budget
    )


def test_the_budget_is_razors():
    assert pr.PLAN_BUDGET_BYTES == 256 << 20
    assert pr.PLAN_BUDGET_BYTES == acc.razor_reuse_plan_budget_bytes


def test_the_reason_is_razors_words():
    assert pr.over_budget_reason() == _reason(256 << 20)


@pytest.mark.parametrize("ground", sorted(GROUNDS))
def test_a_deck_past_the_budget_declines_and_fills_as_off(ground, monkeypatch):
    base, _ = _fill(**GROUNDS[ground])
    budget = 700_000  # the class map fits, the census does not
    monkeypatch.setattr(pr, "PLAN_BUDGET_BYTES", budget)
    on, s = _fill(pair_reuse=True, **GROUNDS[ground])
    assert s.pair_reuse_declined == _reason(budget)
    assert s.pair_reuse_census == {}
    assert on.tobytes() == base.tobytes()


@pytest.mark.parametrize("ground", sorted(GROUNDS))
def test_a_deck_inside_the_budget_reuses_as_without_one(ground, monkeypatch):
    """The budget's checks move no class: the same bytes as a budget that
    could never bind, and the reuse ran."""
    on, s = _fill(pair_reuse=True, **GROUNDS[ground])
    assert s.pair_reuse_declined is None
    assert "direct" in s.pair_reuse_census
    pr.clear_shared_plans()
    monkeypatch.setattr(pr, "PLAN_BUDGET_BYTES", BIG)
    lifted, s = _fill(pair_reuse=True, **GROUNDS[ground])
    assert s.pair_reuse_declined is None
    assert on.tobytes() == lifted.tobytes()


def test_verify_is_held_to_the_budget(monkeypatch):
    base, _ = _fill()
    monkeypatch.setattr(pr, "PLAN_BUDGET_BYTES", 700_000)
    on, s = _fill(pair_reuse="verify")
    assert s.pair_reuse_declined == _reason(700_000)
    assert on.tobytes() == base.tobytes()


def test_the_class_map_alone_declines_before_any_band(monkeypatch):
    def no_census(*_):
        raise AssertionError("the census ran")

    monkeypatch.setattr(pr, "_census_bands", no_census)
    assert _direct_plan(budget=4 * 405 * 405 - 1) is pr.OVER_BUDGET


def test_a_plan_is_held_to_its_fill_exactly(monkeypatch):
    plan = _direct_plan()
    need = pr.fill_nbytes(plan)
    assert need == pr.plan_nbytes(plan) + 48 * (plan.n_classes + plan.n_near)
    monkeypatch.setattr(pr, "BAND_BYTES_PER_PAIR", 0)
    assert _direct_plan(budget=need - 1) is pr.OVER_BUDGET
    served = _direct_plan(budget=need)
    assert served is not pr.OVER_BUDGET
    assert served.cls.tobytes() == plan.cls.tobytes()
    assert served.src_c.tobytes() == plan.src_c.tobytes()


def test_the_blocks_share_one_budget(monkeypatch):
    """The direct plan fits alone, and the PEC image's does not fit beside
    it: the fill declines whole rather than reusing one block."""
    base, _ = _fill(ground_z=0.0)
    _, s = _fill(pair_reuse=True, ground_z=0.0)
    direct = s.pair_reuse_census["direct"]
    assert direct[1] > 0
    need = pr.fill_nbytes(next(iter(s._sin_reuse_plans.values()))[1])
    pr.clear_shared_plans()
    monkeypatch.setattr(pr, "BAND_BYTES_PER_PAIR", 0)
    monkeypatch.setattr(pr, "GROUND_BAND_BYTES_PER_PAIR", 0)
    monkeypatch.setattr(pr, "PLAN_BUDGET_BYTES", need + 1)
    on, s = _fill(pair_reuse=True, ground_z=0.0)
    assert s.pair_reuse_declined == _reason(need + 1)
    assert s.pair_reuse_census == {}
    assert on.tobytes() == base.tobytes()


def test_a_declined_census_is_not_repeated(monkeypatch):
    """A solver per frequency finds the first one's verdict in the cache."""
    runs = []
    census = pr._census_bands

    def counted(*a):
        runs.append(1)
        return census(*a)

    monkeypatch.setattr(pr, "_census_bands", counted)
    monkeypatch.setattr(pr, "PLAN_BUDGET_BYTES", 700_000)
    _fill(pair_reuse=True)
    _, s = _fill(pair_reuse=True)
    assert len(runs) == 1
    assert s.pair_reuse_declined == _reason(700_000)


class _Held:
    """A `_Fold` stand-in: `held()` is `base` plus `per` for each band
    folded, `floor()` `floor_per` for each."""

    def __init__(self, base=0, per=0, floor_per=0):
        self.base, self.per, self.floor_per, self.folded = base, per, floor_per, []
        self.map_bytes = 0

    def __len__(self):
        return len(self.folded)

    def held(self):
        return self.base + self.per * len(self.folded)

    def floor(self):
        return self.floor_per * len(self.folded)

    def fold(self, i, r):
        self.folded.append((i, r))


def _census(census, n, budget, band_bytes, band=None):
    calls = []

    def default(s):
        calls.append(s)
        return s * 10

    out = pr._census_bands(
        band or default, list(range(n)), census, census.fold, budget, band_bytes
    )
    return out, calls


def test_no_budget_folds_every_band_in_order():
    held = _Held()
    out, calls = _census(held, 40, None, 0)
    assert out is not None
    assert sorted(calls) == list(range(40))
    assert held.folded == [(i, 10 * i) for i in range(40)]


def test_a_census_that_fits_folds_every_band_in_order():
    held = _Held(base=100, per=7)
    out, _ = _census(held, 50, 100 + 7 * 50 + 30, 10)
    assert out is not None and out <= 100 + 7 * 50 + 30
    assert [i for i, _ in held.folded] == list(range(50))


def test_the_count_stops_the_census_once_it_passes_the_budget():
    held = _Held(base=100, per=7)
    out, calls = _census(held, 80, 100 + 7 * 20, 0)
    assert out is None
    assert len(held.folded) == 21  # the fold that passed it is the last
    assert len(calls) < 80


def test_the_plans_floor_stops_the_census():
    held = _Held(floor_per=5)
    out, _ = _census(held, 80, 5 * 30, 0)
    assert out is None
    assert len(held.folded) == 31


def test_a_band_starts_only_where_it_fits():
    """Each band in flight counts `band_bytes` until it is folded: under a
    budget with room for two beside what is held, no more than two run at
    once, however many workers the pool has."""
    lock = threading.Lock()
    now = [0, 0]

    def band(s):
        with lock:
            now[0] += 1
            now[1] = max(now)
        time.sleep(0.01)
        with lock:
            now[0] -= 1
        return s

    held = _Held(base=1000)
    out, _ = _census(held, 30, 1000 + 2 * 50, 50, band)
    assert out is not None and out <= 1000 + 2 * 50
    assert now[1] <= 2
    assert len(held.folded) == 30


def test_a_census_with_no_room_for_one_band_runs_none():
    held = _Held(base=1000)
    out, calls = _census(held, 10, 1000 + 49, 50)
    assert out is None
    assert calls == []


def test_a_census_hands_its_pages_back_once(monkeypatch):
    trims = []
    monkeypatch.setattr(razor, "_trim_heap", lambda: trims.append(1))
    assert pr.shared_plan(("k",), lambda: pr.OVER_BUDGET) is pr.OVER_BUDGET
    assert pr.shared_plan(("k",), lambda: pr.OVER_BUDGET) is pr.OVER_BUDGET
    assert len(trims) == 1


def test_the_plans_kept_between_solvers_are_one_fills_budget():
    """A long-lived process holds no more plans than one fill may use, on the
    sinusoidal solver and on razor-2p (momwire#2026)."""
    assert pr.SHARED_PLAN_BYTES == pr.PLAN_BUDGET_BYTES == 256 << 20
    header = Path(pr.__file__).with_name("_razor_pair_reuse_core.h").read_text()
    assert "SHARED_PLAN_BYTES = PLAN_BUDGET_BYTES;" in header
    assert "PLAN_BUDGET_BYTES = size_t(256) << 20;" in header


def test_the_cache_drops_the_oldest_plan_past_its_budget(monkeypatch):
    class Sized:
        near_blocks = ()

        def __init__(self, n):
            self.values = np.zeros(n, np.uint8)

    monkeypatch.setattr(pr, "_shared_plans", type(pr._shared_plans)())
    monkeypatch.setattr(pr, "SHARED_PLAN_BYTES", 100)
    monkeypatch.setattr(razor, "_trim_heap", lambda: None)
    pr.shared_plan(("a",), lambda: Sized(60))
    pr.shared_plan(("b",), lambda: Sized(60))
    assert list(pr._shared_plans) == [("b",)]
