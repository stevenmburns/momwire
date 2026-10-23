"""Cone sheets: the crossing tables along a SLOPED wire (momwire#1595 option A).

Seen from an observer on its axis, a sloped straight buried wire is a cone:
every row it asks lies on z' = zf + beta * u(rho), u = sqrt(rho^2 - a^2).
`_near_interface.ConeSheet` tabulates that surface along u at the fill's
exact observer heights, and `_evaluate_fresh` interpolates every row on it
(`near_interface_cone_line`). A deck's radials slope a little differently
from one another (coordinates printed to a few decimals), so a cone is a FAN
read linearly in slope between its two edges.

GATED, not bit-identical, as the plane sheets are, so the tests are of four
kinds:

  * accuracy -- the table against the twin at the rows it serves, below the
    plane sheets' own bound; a coarser table (p = 8) must FAIL the same
    check, and so must the fan read without its slope step;
  * the twins -- the C++ fills' plan and values are the Python route's to the
    bit (the TEST-ONLY `near_interface_eval_designed` and
    `near_interface_eval_cone_table`);
  * the rule -- what the census takes and declines, the corner and the
    off-set heights left exact, the switch off as the exact route;
  * a razor fill with cones on against the same fill with them off, at a
    tolerance far inside the mesh's own error.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _accel
from momwire import _near_interface as ni

from test_crossing_serve_524 import A_WIRE, SOIL_A, WL7, _radial_dirs
from test_triple_memo_1168 import _razor

pytestmark = pytest.mark.skipif(
    not (ni._use_cone() and ni._use_column_accel()),
    reason="the cone sheets need the C++ near-interface extension",
)

ACC = _accel.acc
HAVE_CXX = ACC is not None and hasattr(ACC, "near_interface_eval_designed")

F = 14e6
K_P = 2.0 * np.pi * F / 299792458.0
EPS_T = complex(13.0, -0.005 / (2.0 * np.pi * F * 8.8541878128e-12))
K_M = ni.k_medium(EPS_T, K_P)
A = 0.001
PW = np.asarray(ni.KEY_RPOW)
# The plane sheets' measured bound (`_SHEET_P`'s comment: 1.8e-9 to 2.4e-9
# against the envelope); the cones measured 4.8e-10 to 1.1e-9 on these rows
# and on the synthetic Slant's fills.
BOUND = 2.5e-9


@pytest.fixture(autouse=True)
def _fresh_sheets(monkeypatch):
    """Each test starts with no cached sheet and zero counters."""
    monkeypatch.setattr(ni, "_SHEET_CACHE", type(ni._SHEET_CACHE)())
    monkeypatch.setattr(ni, "_SHEET_STATS", dict.fromkeys(ni._SHEET_STATS, 0))


def _vertical_over_slant(
    n_rad=16, apex=0.0, deep=-0.0508, length=5.0, nseg=30, decimals=6
):
    """A vertical's Gauss nodes on the axis over `n_rad` straight radials
    from (0, 0, apex) to endpoints at `deep`, printed to `decimals`: the
    (above, below) node pair, z relative to the interface, and every row the
    pair asks."""
    gl = np.polynomial.legendre.leggauss(4)[0]
    t = np.concatenate([(k + 0.5 + 0.5 * gl) / nseg for k in range(nseg)])
    above = np.stack([np.zeros_like(t), np.zeros_like(t), 5.0 * t], axis=1)
    below = []
    for r in range(n_rad):
        th = 2.0 * np.pi * r / n_rad + 0.05 * r  # irregular: no mirror pairs
        ex = round(length * np.cos(th), decimals)
        ey = round(length * np.sin(th), decimals)
        below.append(np.stack([t * ex, t * ey, apex + t * (deep - apex)], axis=1))
    below = np.concatenate(below)
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


def _envelope_error(sub, got, ref):
    """Max error per kernel relative to its 1/R^pw envelope (a pointwise
    relative error is meaningless where a kernel crosses zero)."""
    R = np.hypot(sub[:, 0], sub[:, 1] - sub[:, 2])
    env = np.abs(ref * R[:, None] ** PW).max(axis=0) / R[:, None] ** PW
    return float((np.abs(got - ref) / env).max())


def _twin(sub):
    """The exact route at each row, each a column of one."""
    return np.asarray(
        ni._nia.near_interface_six_columns(
            K_P,
            K_M,
            np.ascontiguousarray(sub[:, 0]),
            np.arange(sub.shape[0] + 1, dtype=np.intp),
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


def _plan_and_rows(**geom):
    specs, rows = _vertical_over_slant(**geom)
    plan = ni.sheet_plan(EPS_T, K_P, [specs], A)
    sub = np.unique(rows, axis=0)
    return specs, plan, sub


def _served(sub, plan):
    """The cone rows of `sub` and their values from the plan's sheets."""
    take = ni._sheet_take(sub, plan)
    assert take is not None and take[2] is not None
    idx = np.flatnonzero(take[2] >= 0)
    out = np.zeros((sub.shape[0], ni.N_KEYS), dtype=np.complex128)
    ni._serve_from_cones(K_P, K_M, sub, take[2], plan, out)
    return idx, out[idx]


# ----------------------------------------------------------------------
# the table against the twin, at the rows
# ----------------------------------------------------------------------


@pytest.mark.parametrize("apex", [0.0, -0.0254])
def test_the_cone_is_the_twin_at_its_rows(apex):
    """Every row a cone serves, against the twin at that row: below the
    plane sheets' bound. Both apex kinds: at the interface (Dan's Slant, a
    radial from the feed) and buried (a radial from a buried hub)."""
    _specs, plan, sub = _plan_and_rows(n_rad=8, apex=apex)
    assert len(plan.cones) == 1, plan
    zf, blo, bhi, _d, _zs, _p = plan.cones[0]
    assert blo < bhi  # the 6-decimal endpoints make it a fan
    idx, got = _served(sub, plan)
    assert idx.size > 0.9 * sub.shape[0]
    assert _envelope_error(sub[idx], got, _twin(sub[idx])) < BOUND
    assert ni._SHEET_STATS["cone_rows"] == idx.size


def test_a_coarse_cone_fails_the_same_check(monkeypatch):
    """At p = 8 the same rows read ~7e-7: the check can fail."""
    monkeypatch.setattr(ni, "_CONE_P", 8)
    _specs, plan, sub = _plan_and_rows(n_rad=8)
    idx, got = _served(sub, plan)
    assert _envelope_error(sub[idx], got, _twin(sub[idx])) > 10 * BOUND


def test_the_fan_needs_its_slope_step():
    """Endpoints printed to 4 decimals spread the slopes ~1e-5 apart, a fan
    inside its bound whose edges are ~5e-7 of K apart. Read with the slope
    step the rows meet the bound; read at beta_lo alone (no step) they fail
    it, so the step is doing the work."""
    _specs, plan, sub = _plan_and_rows(n_rad=16, decimals=4)
    assert len(plan.cones) == 1, plan
    idx, stepped = _served(sub, plan)
    ref = _twin(sub[idx])
    assert _envelope_error(sub[idx], stepped, ref) < BOUND
    zf, blo, _bhi, d, zs, _p = plan.cones[0]
    u = ni._cone_u(sub[idx, 0], A)
    zi = np.searchsorted(zs, sub[idx, 1]).astype(np.int64)
    flat = np.empty((idx.size, ni.N_KEYS), dtype=np.complex128)
    ni._cone_sheet(K_P, K_M, d, zf, blo, A, zs, float(u.max())).interpolate(u, zi, flat)
    assert _envelope_error(sub[idx], flat, ref) > 10 * BOUND


def test_a_grown_cone_keeps_every_node():
    """Growing a sheet appends u cells and never re-evaluates one: grown in
    three steps it is the sheet grown in one, to the bit."""
    zs = np.sort(np.concatenate([[0.0, 0.003, 0.01], np.linspace(0.02, 5.0, 40)]))
    one = ni.ConeSheet(K_P, K_M, 1 / 64, 0.0, -0.01016, A, zs)
    one.cover(5.0)
    three = ni.ConeSheet(K_P, K_M, 1 / 64, 0.0, -0.01016, A, zs)
    for reach in (0.1, 1.0, 5.0):
        three.cover(reach)
    a, b = one.arrays(), three.arrays()
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[2].view(np.uint64), b[2].view(np.uint64))


def test_the_table_refuses_a_row_it_cannot_answer():
    """A height the table was not built at, or a u past its cells, raises:
    the interpolator never extrapolates."""
    zs = np.array([0.5, 1.0])
    sheet = ni.ConeSheet(K_P, K_M, 1 / 64, 0.0, -0.01016, A, zs)
    sheet.cover(1.0)
    out = np.empty((1, ni.N_KEYS), dtype=np.complex128)
    with pytest.raises(ValueError, match="off the cone sheet"):
        sheet.interpolate(np.array([0.5]), np.array([2]), out)
    with pytest.raises(ValueError, match="off the cone sheet"):
        sheet.interpolate(np.array([50.0]), np.array([0]), out)


# ----------------------------------------------------------------------
# the twins: the C++ fills' plan and values are the Python route's
# ----------------------------------------------------------------------


@pytest.mark.skipif(not HAVE_CXX, reason="needs the extension's TEST-ONLY hooks")
@pytest.mark.parametrize("n_rad,apex", [(16, 0.0), (8, -0.0254)])
def test_the_cxx_plan_and_values_are_pythons(n_rad, apex):
    """`sheet_plan` and `designed_tables` through a memo carrying it, as the
    razor and bs2 crossing cores call them: the same cones (edges, corner,
    heights) and the same floats at every row, cone and exact alike."""
    specs, rows = _vertical_over_slant(n_rad=n_rad, apex=apex)
    plan = ni.sheet_plan(EPS_T, K_P, [specs], A)
    memo = ni.TripleMemo()
    memo.sheet_plan = plan
    py = ni.designed_tables(EPS_T, K_P, rows[:, 0], rows[:, 1], rows[:, 2], memo=memo)
    py = np.stack([py[k] for k in ni.KEYS], axis=1)
    cx = ACC.near_interface_eval_designed(
        K_P, K_M, [specs], A, rows[:, 0], rows[:, 1], rows[:, 2]
    )
    assert ni._SHEET_STATS["cone_rows"] > 0
    assert cx["fold"] == plan.fold
    assert len(cx["cones"]) == len(plan.cones) == 1
    for (zf, blo, bhi, d, zs, p_fan), c in zip(plan.cones, cx["cones"], strict=True):
        assert (zf, blo, bhi, d, p_fan) == (*c[:4], c[5])
        assert np.array_equal(zs, np.asarray(c[4]))
    cv = np.asarray(cx["values"])
    assert np.array_equal(py.view(np.uint64), cv.view(np.uint64))


@pytest.mark.skipif(not HAVE_CXX, reason="needs the extension's TEST-ONLY hooks")
@pytest.mark.parametrize("p", [ni._CONE_P, 2, 6])
def test_the_cxx_cone_table_is_pythons(p):
    """One surface's table, the full one and the fan's coarse pair: the C++
    `ConeSheet`'s edges and values are Python's to the bit. (Its Chebyshev
    nodes are scalar libm's: a vectorised cosine moved 1 node in 8 by an
    ulp.)"""
    zs = np.sort(np.concatenate([[0.0, 0.003, 0.01], np.linspace(0.02, 5.0, 40)]))
    sheet = ni.ConeSheet(K_P, K_M, 1 / 64, 0.0, -0.01016, A, zs, p)
    sheet.cover(5.0)
    edges, _off, vals = sheet.arrays()
    e_cx, v_cx = ACC.near_interface_eval_cone_table(
        K_P, K_M, 1 / 64, 0.0, -0.01016, A, zs, p, 5.0
    )
    assert np.array_equal(edges, np.asarray(e_cx))
    v_cx = np.asarray(v_cx).reshape(vals.shape)
    assert np.array_equal(vals.view(np.uint64), v_cx.view(np.uint64))


# ----------------------------------------------------------------------
# the rule
# ----------------------------------------------------------------------


def test_the_corner_and_heights_off_the_plan_stay_exact():
    """A row in the corner (u < d and z < d), or at a height the plan was not
    built at, or off the fan, is on no cone."""
    _specs, plan, sub = _plan_and_rows(n_rad=8)
    zf, blo, bhi, d, zs, _p = plan.cones[0]
    u = 0.5 * d
    rows = np.array(
        [
            [np.hypot(u, A), 0.5 * d, zf + blo * u],  # the corner
            [np.hypot(1.0, A), 1234.5, zf + blo * 1.0],  # not a plan height
            [np.hypot(1.0, A), zs[-1], zf + 2.0 * blo * 1.0],  # off the fan
            [np.hypot(1.0, A), zs[-1], zf + blo * 1.0],  # on it
        ]
    )
    cone = ni._cone_take(rows, plan, np.ones(4, dtype=bool))
    assert cone.tolist() == [-1, -1, -1, 0]


def test_the_census_declines_what_does_not_pay():
    """Four radials along the axes ask every rho twice (the mirrored pairs
    are the same rows), so the exact route already shares their columns and
    a cone does not pay: declined, and counted as such."""
    ni._CONE_TRACE = []
    try:
        gl = np.polynomial.legendre.leggauss(4)[0]
        t = np.concatenate([(k + 0.5 + 0.5 * gl) / 30 for k in range(30)])
        above = np.stack([0 * t, 0 * t, 5.0 * t], axis=1)
        below = np.concatenate(
            [
                np.stack([t * ex, t * ey, -0.0625 * t], axis=1)
                for ex, ey in ((4.0, 0.0), (0.0, 4.0), (-4.0, 0.0), (0.0, -4.0))
            ]
        )
        plan = ni.sheet_plan(EPS_T, K_P, [(above, below)], A)
        assert plan.cones == ()
        (trace,) = ni._CONE_TRACE
        assert trace["gain"] < trace["cost"]
        assert ni._SHEET_STATS["cone_declined_rows"] == trace["rows"] > 0
    finally:
        ni._CONE_TRACE = None


def test_a_fan_past_its_bound_is_declined(monkeypatch):
    """The fan's linear step is bounded; a bound below the fan's own (here
    1e-30) declines it whatever it would save."""
    monkeypatch.setattr(ni, "_CONE_FAN_ERR", 1e-30)
    _specs, plan, _sub = _plan_and_rows(n_rad=16)
    assert plan.cones == ()
    assert ni._SHEET_STATS["cone_declined_rows"] > 0


def test_the_switch_off_is_the_exact_route(monkeypatch):
    """`_CONE = False`: no cone is planned and every row is the twin's, to
    the bit -- the code on main."""
    specs, rows = _vertical_over_slant(n_rad=8)
    monkeypatch.setattr(ni, "_CONE", False)
    plan = ni.sheet_plan(EPS_T, K_P, [specs], A)
    assert plan.cones == ()
    memo = ni.TripleMemo()
    memo.sheet_plan = plan
    got = ni.designed_tables(EPS_T, K_P, rows[:, 0], rows[:, 1], rows[:, 2], memo=memo)
    ref = ni.designed_tables(EPS_T, K_P, rows[:, 0], rows[:, 1], rows[:, 2])
    for k in ni.KEYS:
        assert np.array_equal(got[k].view(np.uint64), ref[k].view(np.uint64))
    assert ni._SHEET_STATS["cone_rows"] == 0


# ----------------------------------------------------------------------
# the razor fill: cones against the exact route
# ----------------------------------------------------------------------


def slant_deck(n_radials=16, depth=0.0254, deep=0.0762, **override):
    """`hub_deck`'s sloped twin: N radials junction-joined at a buried hub
    (0, 0, -depth), each sloping down to -deep at 5 m (endpoints at the
    printed precision of `_radial_dirs`), a rise to the interface and a
    10 m monopole."""
    dirs = _radial_dirs(n_radials)
    wires = [
        np.array([(5.0 * dx, 5.0 * dy, -deep), (0.0, 0.0, -depth)]) for dx, dy in dirs
    ]
    npe = [[10] for _ in dirs]
    rise_i = len(wires)
    wires.append(np.array([(0.0, 0.0, -depth), (0.0, 0.0, 0.0)]))
    npe.append([2])
    mono_i = rise_i + 1
    wires.append(np.array([(0.0, 0.0, 10.0), (0.0, 0.0, 0.0)]))
    npe.append([15])
    build = dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        junctions=[
            [(i, "end") for i in range(n_radials)] + [(rise_i, "start")],
            [(rise_i, "end"), (mono_i, "end")],
        ],
        feeds=[(mono_i, 4.3333333333, 1 + 0j)],
        wavelength=WL7,
        wire_radius=A_WIRE,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )
    build.update(override)
    return build


def _fill(deck):
    s = _razor(deck)
    return np.array(s._assemble_Z(s._build_geometry(), s.k), copy=True)


@pytest.fixture(scope="module")
def slant_fills():
    """The sloped hub filled with the cones off and on: ({name: Z},
    {name: stats})."""
    deck = slant_deck()
    Z, stats = {}, {}
    for name, cone in (("off", False), ("on", True)):
        with pytest.MonkeyPatch.context() as m:
            m.setattr(ni, "_CONE", cone)
            m.setattr(ni, "_SHEET_CACHE", type(ni._SHEET_CACHE)())
            m.setattr(ni, "_SHEET_STATS", dict.fromkeys(ni._SHEET_STATS, 0))
            Z[name] = _fill(deck)
            stats[name] = dict(ni._SHEET_STATS)
    return Z, stats


def test_the_cone_fill_is_the_exact_fill_at_the_tolerance(slant_fills):
    """Every entry within 1e-7 relative (entries above 1e-6 of max|Z|), and
    the whole matrix within 1e-9 of max|Z|: the plane sheets' gate."""
    Z, stats = slant_fills
    ref, got = Z["off"], Z["on"]
    assert not np.array_equal(got, ref)  # the cones ran and moved something
    big = np.abs(ref) > 1e-6 * np.abs(ref).max()
    assert (np.abs(got - ref)[big] / np.abs(ref[big])).max() < 1e-7
    assert np.abs(got - ref).max() < 1e-9 * np.abs(ref).max()
    assert stats["on"]["cone_rows"] > 0.25 * stats["off"]["exact_rows"]


def test_the_counters_account_for_every_row(slant_fills):
    """Each fill's fresh rows are counted once, sheet or exact, and the same
    fill asks the same fresh rows whichever serves them."""
    _Z, stats = slant_fills
    on, off = stats["on"], stats["off"]
    assert on["sheet_rows"] + on["exact_rows"] == off["exact_rows"]
    assert on["cone_rows"] <= on["sheet_rows"]
    assert on["cones_planned"] >= 1 and on["cone_sheets_built"] >= 1
    assert off["cone_rows"] == off["cones_planned"] == off["cone_nodes_built"] == 0
