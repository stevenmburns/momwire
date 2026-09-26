"""Plane sheets for the crossing fill (momwire#1173 Design E, phase 1).

A buried deck whose above side is not one vertical line asks one near-interface
row per (above node, buried node) pair, nearly all on ONE source depth z′ and
each with its own ρ, so the column route used to pay a whole column per row.
`_near_interface.PlaneSheet` tabulates such a plane once (its nodes evaluated
by the column twin, each a column of one) and `_evaluate_fresh` interpolates
every row on it in C++ (`near_interface_plane_sheet`).

This is a GATED change, not a bit-identical one, so the tests here are of
three kinds:

  * accuracy — the sheet against the twin at the same rows, and a razor fill
    with the sheets on against the same fill with them off, at tolerances
    orders below the Design E gate (1 % of razor's own mesh-halving
    difference, and 0.01 Ω); a coarse table must FAIL the same check;
  * the switch — `_SHEET = False` is the exact route, to the bit: it IS the
    code on main (`_column_twin`, unchanged), and a fill with it is the
    fill with the sheets on but no plane qualifying;
  * the guard and the counters — a plane shallower than `_SHEET_MIN_DEPTH`
    (the corner z, z′ → 0 is the one point a sheet must never serve) is left
    exact and counted as such, and `_SHEET_STATS` accounts for every fresh
    row, so a green accuracy row cannot be a sheet that never ran.

The deck-level gates (hub_deck(16), the inverted-L and its leaning variant at
x2 / x4 / x8, and the soil / frequency / depth ladder) were measured by script
for the PR; see there.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _near_interface as ni
from momwire._near_interface import k_medium

from test_crossing_serve_524 import SOIL_A, WL7, hub_deck, invl_deck
from test_triple_memo_1168 import _razor

pytestmark = pytest.mark.skipif(
    not (ni._HAVE_PLANE_SHEET_ACCEL and ni._use_column_accel()),
    reason="the plane sheet needs the C++ near-interface extension",
)

K_P = 2.0 * np.pi / WL7
EPS_T = complex(
    SOIL_A[0], -SOIL_A[1] / (2.0 * np.pi * (299792458.0 / WL7) * 8.8541878128e-12)
)
K_M = k_medium(EPS_T, K_P)


@pytest.fixture(autouse=True)
def _fresh_sheets(monkeypatch):
    """Each test starts with no cached sheet and zero counters."""
    monkeypatch.setattr(ni, "_SHEET_CACHE", type(ni._SHEET_CACHE)())
    monkeypatch.setattr(ni, "_SHEET_STATS", dict.fromkeys(ni._SHEET_STATS, 0))


def _twin(sub):
    """The exact route at each row, each a column of one."""
    n = sub.shape[0]
    return np.asarray(
        ni._nia.near_interface_six_columns(
            K_P,
            K_M,
            np.ascontiguousarray(sub[:, 0]),
            np.arange(n + 1, dtype=np.intp),
            np.ascontiguousarray(sub[:, 1]),
            np.ascontiguousarray(sub[:, 2]),
            float(ni._LAM_MULT),
            int(ni._COLUMN_P),
            float(ni._DETOUR),
            ni._physical_cpu_count(),
            ni._GX,
            ni._GW,
        )
    )


def _plane_rows(zp, n=600, rmax=12.0, seed=3):
    """Rows on the plane z′ = zp: the open box, the interface (z → 0) and the
    axis (ρ → a), which are where a sheet is hardest."""
    rng = np.random.default_rng(seed)
    rho = np.concatenate(
        [rng.uniform(0, rmax, n), rng.uniform(0, rmax, n), 10 ** rng.uniform(-3, 1, n)]
    )
    z = np.concatenate(
        [
            rng.uniform(0, rmax, n),
            10 ** rng.uniform(-5, 0, n),
            10 ** rng.uniform(-4, 1, n),
        ]
    )
    rho = np.hypot(rho, 1e-3)  # the radius fold, as the crossing fill asks
    return np.ascontiguousarray(np.stack([rho, z, np.full_like(z, zp)], axis=1))


def _envelope_error(sub, got, ref):
    """Max error per kernel relative to its 1/R^pw envelope at that R (a
    pointwise relative error is meaningless where a kernel crosses zero)."""
    R = np.hypot(sub[:, 0], sub[:, 1] - sub[:, 2])
    pw = np.array([1, 1, 1, 2, 2, 2])
    env = np.abs(ref * R[:, None] ** pw).max(axis=0) / R[:, None] ** pw
    return float((np.abs(got - ref) / env).max())


def _sheet_at(sub):
    zp = float(sub[0, 2])
    R = np.hypot(sub[:, 0], sub[:, 1] - zp)
    sheet = ni.PlaneSheet(K_P, K_M, zp, ni._LAM_MULT)
    sheet.cover(float(R.max()))
    out = np.empty((sub.shape[0], 6), dtype=np.complex128)
    sheet.interpolate(sub, np.arange(sub.shape[0]), out)
    return sheet, out


# ----------------------------------------------------------------------
# the table against the twin, at the rows
# ----------------------------------------------------------------------


def test_the_sheet_is_the_twin_at_its_rows():
    """Soil A, 7 MHz, the radials' 0.15 m plane: 1.1e-9 measured."""
    sub = _plane_rows(-0.15)
    _sheet, got = _sheet_at(sub)
    assert _envelope_error(sub, got, _twin(sub)) < 1e-8


def test_a_coarse_table_fails_the_same_check(monkeypatch):
    """The negative control: a p = 3 table (Design E's, uncapped, reads
    4.8e-2 Ω on invl x2) must fail the tolerance the production table
    passes."""
    monkeypatch.setattr(ni, "_SHEET_P", 3)
    monkeypatch.setattr(ni, "_SHEET_PT", 3)
    sub = _plane_rows(-0.15)
    _sheet, got = _sheet_at(sub)
    assert _envelope_error(sub, got, _twin(sub)) > 1e-4


def test_a_grown_sheet_keeps_every_node():
    """Growing a sheet appends panels and never moves a node, so a row's
    value does not depend on how far the sheet had been grown before."""
    sub = _plane_rows(-0.15, n=200, rmax=3.0)
    zp = float(sub[0, 2])
    rmax = float(np.hypot(sub[:, 0], sub[:, 1] - zp).max())
    once = ni.PlaneSheet(K_P, K_M, zp, ni._LAM_MULT)
    once.cover(rmax)
    twice = ni.PlaneSheet(K_P, K_M, zp, ni._LAM_MULT)
    twice.cover(1.0)
    small = twice.n_nodes
    twice.cover(rmax)
    assert twice.n_nodes > small
    for a, b in zip(once.arrays(), twice.arrays(), strict=True):
        assert np.array_equal(a, b)
    idx = np.arange(sub.shape[0])
    got = [np.empty((sub.shape[0], 6), dtype=np.complex128) for _ in range(2)]
    once.interpolate(sub, idx, got[0])
    twice.interpolate(sub, idx, got[1])
    assert np.array_equal(got[0], got[1])


def test_the_table_refuses_a_row_it_cannot_answer():
    """Off the plane, below the interface or beyond the table: refused in
    C++, never extrapolated."""
    sub = _plane_rows(-0.15, n=50, rmax=3.0)
    sheet, _ = _sheet_at(sub)
    for bad in ([1.0, 1.0, -0.2], [1.0, -1e-3, -0.15], [1e4, 1.0, -0.15]):
        rows = np.ascontiguousarray(np.vstack([sub, bad]))
        out = np.empty((rows.shape[0], 6), dtype=np.complex128)
        with pytest.raises(ValueError, match="off the plane sheet"):
            sheet.interpolate(rows, np.arange(rows.shape[0]), out)


# ----------------------------------------------------------------------
# the guard
# ----------------------------------------------------------------------


def test_no_sheet_serves_the_corner(monkeypatch):
    """The distance guard at the seam: rows at z′ = 0 (the corner) and on a
    plane shallower than `_SHEET_MIN_DEPTH` never take a sheet, whatever
    their number; the shallow plane's rows are counted as guarded; a plane
    just deep enough does take one."""
    monkeypatch.setattr(ni, "_SHEET_MIN_ROWS", 64)
    monkeypatch.setattr(ni, "_SHEET_BUILD_WEIGHT", 0.0)
    g = ni._SHEET_MIN_DEPTH
    rows = np.concatenate(
        [_plane_rows(0.0, n=40), _plane_rows(-0.5 * g, n=40), _plane_rows(-g, n=40)]
    )
    planes, take = ni._sheet_planes(rows, K_P, K_M)
    assert [v for v, _r in planes] == [-g]
    assert np.all(rows[take, 2] == -g)
    assert ni._SHEET_STATS["guarded_rows"] == 120
    plan = ni.SheetPlan([v for v, _r in planes])
    got = ni._evaluate_fresh(EPS_T, K_P, rows, 1e-10, ni._LAM_MULT, plan=plan)
    assert ni._SHEET_STATS["sheet_rows"] == 120
    assert ni._SHEET_STATS["exact_rows"] == 240
    ref = ni._column_twin(K_P, K_M, rows[~take], ni._LAM_MULT)
    assert np.array_equal(got[~take], ref)


def test_a_plane_under_the_guard_is_left_exact(monkeypatch):
    """The guard through a real fill: with the guard moved below the hub's
    0.15 m radials (razor refuses radials shallow enough to fall under the
    shipped 1 mm, at the grazing floor), every row of theirs is asked, none
    takes a sheet, they are counted as guarded, and Z is the switched-off
    fill's to the bit."""
    monkeypatch.setattr(ni, "_SHEET_MIN_ROWS", 16)
    monkeypatch.setattr(ni, "_SHEET_BUILD_WEIGHT", 0.0)
    monkeypatch.setattr(ni, "_SHEET_MIN_DEPTH", 0.2)
    deck = hub_deck(n_radials=2)
    got = _fill(deck)
    assert ni._SHEET_STATS["sheet_rows"] == 0
    assert ni._SHEET_STATS["guarded_rows"] > 1000
    monkeypatch.setattr(ni, "_SHEET", False)
    assert np.array_equal(got, _fill(deck))


def test_a_plane_is_tabulated_only_when_its_rows_pay_for_it():
    """The cost rule, on one call's rows alone: the same number of rows on the
    same plane, reaching as far, is declined when it is one exact-rho column
    (the exact route pays one setup) and taken when every row is its own
    column (the inverted-L's top wire over its radials)."""
    n_nodes = ni._sheet_nodes_to(K_P, K_M, 0.15, 10.0)
    n = n_nodes + 64
    z = np.linspace(0.0, 10.0, n)
    shared = np.stack([np.full(n, 3.0), z, np.full(n, -0.15)], axis=1)
    shared[-1, 0] = 10.0  # the farthest row, as in the other call
    distinct = np.stack([np.linspace(0.01, 10.0, n), z, np.full(n, -0.15)], axis=1)
    planes, _take = ni._sheet_planes(np.ascontiguousarray(shared), K_P, K_M)
    assert planes == [] and ni._SHEET_STATS["declined_rows"] == n
    planes, take = ni._sheet_planes(np.ascontiguousarray(distinct), K_P, K_M)
    assert [v for v, _r in planes] == [-0.15] and take.all()
    assert planes[0][1] == pytest.approx(np.hypot(10.0, 10.15))


# ----------------------------------------------------------------------
# the razor fill: sheets against the exact route, the switch, the counters
# ----------------------------------------------------------------------


def _fill(deck):
    s = _razor(deck)
    return np.array(s._assemble_Z(s._build_geometry(), s.k), copy=True)


@pytest.fixture(scope="module")
def invl_fills():
    """The inverted-L over one radial, filled with the sheets off and on (the
    radials' plane carries a few thousand rows in each of razor's two fills,
    so a 512-row pre-filter with the cost rule off takes it, and the rise's
    planes with it; at the shipped rule a deck this small builds no table).
    Returns
    ({name: Z}, {name: stats})."""
    deck = invl_deck(n_radials=1)
    Z, stats = {}, {}
    for name, sheet in (("off", False), ("on", True)):
        with pytest.MonkeyPatch.context() as m:
            m.setattr(ni, "_SHEET", sheet)
            m.setattr(ni, "_SHEET_MIN_ROWS", 512)
            m.setattr(ni, "_SHEET_BUILD_WEIGHT", 0.0)
            m.setattr(ni, "_SHEET_CACHE", type(ni._SHEET_CACHE)())
            m.setattr(ni, "_SHEET_STATS", dict.fromkeys(ni._SHEET_STATS, 0))
            Z[name] = _fill(deck)
            stats[name] = dict(ni._SHEET_STATS)
    return Z, stats


def test_the_sheet_fill_is_the_exact_fill_at_the_tolerance(invl_fills):
    """Every entry within 1e-6 relative (entries above 1e-6 of max|Z|), and
    far inside the gate: the Design E prototype measured 4.7e-11 Ω on Z_in."""
    Z, stats = invl_fills
    ref, got = Z["off"], Z["on"]
    assert not np.array_equal(got, ref)  # the sheet ran and moved something
    big = np.abs(ref) > 1e-6 * np.abs(ref).max()
    assert (np.abs(got - ref)[big] / np.abs(ref[big])).max() < 1e-6
    assert np.abs(got - ref).max() < 1e-9 * np.abs(ref).max()
    assert stats["on"]["sheet_rows"] > stats["on"]["exact_rows"]


def test_the_counters_account_for_every_row(invl_fills):
    """Each fill's fresh rows are counted once, sheet or exact, and the same
    fill asks the same fresh rows whichever serves them."""
    _Z, stats = invl_fills
    on, off = stats["on"], stats["off"]
    assert on["sheet_rows"] + on["exact_rows"] == off["exact_rows"]
    assert on["nodes_built"] >= on["sheets_built"] * ni._SHEET_P * ni._SHEET_PT
    # Two fills (razor's forward and reversed blocks), each planning once;
    # a plane both take is one sheet, built once and serving both.
    assert on["fills_planned"] == 2 and on["planes_planned"] >= 2
    assert 1 <= on["sheets_built"] < on["planes_planned"]
    assert off["sheet_rows"] == off["sheet_planes"] == off["nodes_built"] == 0
    assert off["fills_planned"] == 0


def test_the_switch_off_evaluates_through_the_twin(monkeypatch):
    """The switch at the seam: with `_SHEET = False`, `_evaluate_fresh` is
    `_column_twin` -- the column-twin branch as it stands on main, moved and
    not edited -- to the bit, `permuted` too, with every row counted exact,
    on rows a sheet would otherwise take."""
    monkeypatch.setattr(ni, "_SHEET", False)
    monkeypatch.setattr(ni, "_SHEET_MIN_ROWS", 16)
    monkeypatch.setattr(ni, "_SHEET_BUILD_WEIGHT", 0.0)
    rows = _plane_rows(-0.15, n=30)
    plan = ni.SheetPlan([-0.15])
    got = ni._evaluate_fresh(EPS_T, K_P, rows, 1e-10, ni._LAM_MULT, plan=plan)
    assert np.array_equal(got, ni._column_twin(K_P, K_M, rows, ni._LAM_MULT))
    vals, pos = ni._evaluate_fresh(
        EPS_T, K_P, rows, 1e-10, ni._LAM_MULT, permuted=True, plan=plan
    )
    assert np.array_equal(vals[pos], got)
    assert ni._SHEET_STATS["sheet_rows"] == 0
    assert ni._SHEET_STATS["exact_rows"] == 2 * rows.shape[0]


# ----------------------------------------------------------------------
# the plan is the fill's (Design E phase 2): no call can decide differently
# ----------------------------------------------------------------------


def test_a_plane_row_is_the_same_in_any_cut_of_the_rows(monkeypatch):
    """Under one plan a row's value does not depend on the call it arrives
    in: the same rows asked as one call, as two halves, and interleaved with
    rows of another depth give the same floats, sheet rows and exact rows
    alike. Phase 1 decided per call, and a half below the pre-filter took the
    twin for the rows its sibling interpolated."""
    monkeypatch.setattr(ni, "_SHEET_MIN_ROWS", 64)
    on = _plane_rows(-0.15, n=200)
    off = _plane_rows(-0.4, n=40, seed=5)
    rows = np.ascontiguousarray(np.concatenate([on, off]))
    plan = ni.SheetPlan([-0.15])
    whole = ni._evaluate_fresh(EPS_T, K_P, rows, 1e-10, ni._LAM_MULT, plan=plan)
    h = rows.shape[0] // 3
    parts = [
        ni._evaluate_fresh(EPS_T, K_P, rows[a:b], 1e-10, ni._LAM_MULT, plan=plan)
        for a, b in ((0, h), (h, rows.shape[0]))
    ]
    assert np.array_equal(np.concatenate(parts)[: on.shape[0]], whole[: on.shape[0]])
    alone = ni._evaluate_fresh(EPS_T, K_P, on[:50], 1e-10, ni._LAM_MULT, plan=plan)
    assert np.array_equal(alone, whole[:50])
    # The exact rows are the twin's on exactly the rows left exact.
    assert np.array_equal(
        whole[on.shape[0] :], ni._column_twin(K_P, K_M, off, ni._LAM_MULT)
    )


def test_no_plan_is_the_exact_route(monkeypatch):
    """A call outside a fill (no memo, so no plan) is the twin to the bit,
    however many rows of a plane it carries."""
    monkeypatch.setattr(ni, "_SHEET_MIN_ROWS", 16)
    monkeypatch.setattr(ni, "_SHEET_BUILD_WEIGHT", 0.0)
    rows = _plane_rows(-0.15, n=300)
    got = ni._evaluate_fresh(EPS_T, K_P, rows, 1e-10, ni._LAM_MULT)
    assert np.array_equal(got, ni._column_twin(K_P, K_M, rows, ni._LAM_MULT))
    assert ni._SHEET_STATS["sheet_rows"] == 0


def test_the_census_is_the_rule_on_the_fills_rows(monkeypatch):
    """`sheet_plan`'s grouped census decides as the rule does on the fill's
    rows spelled out (`_sheet_planes`), on a mast (one (x, y) group, many z),
    a horizontal wire (many groups, one z) and both, and at build weights
    either side of each plane's score, so both bounds and the exact count
    are exercised."""
    rng = np.random.default_rng(1173)
    mast = np.stack([np.zeros(40), np.zeros(40), np.linspace(0.0, 10.0, 40)], 1)
    wire = np.stack([np.linspace(0.0, 5.0, 60), np.zeros(60), np.full(60, 10.0)], 1)
    ang = rng.uniform(0, 2 * np.pi, 8)
    rad = np.concatenate(
        [
            np.stack([t * np.cos(a), t * np.sin(a), np.full(t.size, -0.15)], 1)
            for a in ang
            for t in (np.linspace(0.05, 9.0, 30),)
        ]
    )
    rod = np.stack([np.full(12, 3.0), np.zeros(12), np.linspace(-0.05, -1.0, 12)], 1)
    below = np.concatenate([rad, rod])
    a_wire = 1e-3
    for above in (mast, wire, np.concatenate([mast, wire])):
        rho = np.hypot(
            above[:, 0][:, None] - below[:, 0][None, :],
            above[:, 1][:, None] - below[:, 1][None, :],
        )
        z = np.broadcast_to(above[:, 2][:, None], rho.shape)
        zp = np.broadcast_to(below[:, 2][None, :], rho.shape)
        rows = np.stack([ni.radius_fold(rho, a_wire), z, zp], -1).reshape(-1, 3)
        rows = np.unique(rows, axis=0)
        for weight in (0.02, 0.1, 0.5, 1.0, 5.0):
            monkeypatch.setattr(ni, "_SHEET_BUILD_WEIGHT", weight)
            monkeypatch.setattr(ni, "_SHEET_MIN_ROWS", 64)
            want, _ = ni._sheet_planes(rows, K_P, K_M)
            plan = ni.sheet_plan(EPS_T, K_P, [(above, below)], a_wire)
            assert plan.planes.tolist() == [v for v, _r in want], (weight, plan)
