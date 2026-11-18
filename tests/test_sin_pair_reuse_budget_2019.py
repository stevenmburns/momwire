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

import numpy as np
import pytest

from momwire import _sin_pair_reuse as pr
from momwire import razor
from momwire._accel import acc
from momwire.sinusoidal import SinusoidalSolver

# An experimental feature (`momwire._features`, reach 1).
pytestmark = pytest.mark.reach(1)

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
    monkeypatch.setattr(pr, "CENSUS_BYTES_PER_BAND_CLASS", 0)
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
    monkeypatch.setattr(pr, "CENSUS_BYTES_PER_BAND_CLASS", 0)
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


def _bands(sizes):
    calls = []

    def band(i):
        calls.append(i)
        return ([np.zeros(sizes[i], np.int64)],)

    return band, calls


def test_no_budget_runs_every_band_in_order():
    band, calls = _bands([3] * 40)
    out = pr._census_bands(band, list(range(40)), 0, None)
    assert sorted(calls) == list(range(40))
    assert [r[0][0].size for r in out] == [3] * 40


def test_the_pilot_stops_a_census_it_projects_past_the_budget():
    band, calls = _bands([100] * 80)
    budget = pr.CENSUS_BYTES_PER_BAND_CLASS * 100 * 79
    assert pr._census_bands(band, list(range(80)), 0, budget) is None
    assert len(calls) == pr.PILOT_BANDS


def test_the_running_count_stops_what_the_pilot_missed():
    """The pilot's bands are light and the others heavy: the projection
    passes and the count of the classes actually kept declines."""
    n = 80
    light = set(np.linspace(0, n - 1, pr.PILOT_BANDS).astype(np.int64).tolist())
    sizes = [1 if i in light else 100 for i in range(n)]
    band, _ = _bands(sizes)
    budget = pr.CENSUS_BYTES_PER_BAND_CLASS * 1000
    assert pr._census_bands(band, list(range(n)), 0, budget) is None


def test_a_census_that_fits_keeps_every_band_in_order():
    sizes = [i % 7 + 1 for i in range(50)]
    band, calls = _bands(sizes)
    budget = pr.CENSUS_BYTES_PER_BAND_CLASS * sum(sizes)
    out = pr._census_bands(band, list(range(50)), 0, budget)
    assert sorted(calls) == list(range(50))
    assert [r[0][0].size for r in out] == sizes


def test_the_pilot_runs_its_bands_a_few_at_a_time():
    """A pool's worth of bands at once is the peak the pilot is there to
    avoid paying on a deck that will decline."""
    lock = threading.Lock()
    now = [0, 0]

    def band(_):
        with lock:
            now[0] += 1
            now[1] = max(now)
        time.sleep(0.02)
        with lock:
            now[0] -= 1
        return ([np.zeros(100, np.int64)],)

    assert pr._census_bands(band, list(range(80)), 0, 1) is None
    assert now[1] <= 2


def test_a_census_hands_its_pages_back_once(monkeypatch):
    trims = []
    monkeypatch.setattr(razor, "_trim_heap", lambda: trims.append(1))
    assert pr.shared_plan(("k",), lambda: pr.OVER_BUDGET) is pr.OVER_BUDGET
    assert pr.shared_plan(("k",), lambda: pr.OVER_BUDGET) is pr.OVER_BUDGET
    assert len(trims) == 1
