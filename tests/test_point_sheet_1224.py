"""Plane sheets on the point-matched crossing rows (momwire#1224 option A).

`SinusoidalSolver`'s crossing rows (`_crossing_fill.point_observer_block`)
used to pay one exact near-interface column per distinct rho_eff, for the six
keys AND the four point keys: on the buried inverted-L nearly every (top node,
radial node) pair is its own column, 98.7 % of the invl x16 wall. The block
now plans its fill's sheets once (`_plan_point_sheets`, the Galerkin fill's
`_plan_sheets` on the point grid), the six keys read them through the memo,
and the point family reads the same plan from width-4 sheets
(`PlaneSheet(family="point")`: the same strips, nodes and interpolator, the
nodes from the point column twin, stored times R^`POINT_KEY_RPOW`).

GATED, not bit-identical, as the six-key sheets are (`test_plane_sheet_1173`):

  * accuracy -- each family's sheet against its exact column twin, per
    decade of R, on planes down to the 1 mm guard and on a height; a coarse
    table must fail the same check;
  * the route ran -- counters prove most point rows came from sheets on a
    real invl block, so a silent fallback cannot pass the accuracy rows;
  * the block against `_SHEET = False` (the exact route) at 1e-9, and the
    solve's Z at 1e-9 on invl x4 / x8 and hub x8 (slow);
  * the cut and the history do not show -- dense vs chunked, and a sheet
    grown further by someone else, are the same bits;
  * the guard -- a plane under `_SHEET_MIN_DEPTH` is left exact, counted,
    and the block is the switched-off block's to the bit.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _crossing_fill as cf
from momwire import _near_interface as ni
from momwire._near_interface import k_medium
from momwire.sinusoidal import SinusoidalSolver

from test_crossing_serve_524 import SOIL_A, WL7, hub_deck, invl_deck

pytestmark = pytest.mark.skipif(
    not (
        ni._HAVE_PLANE_SHEET_ACCEL
        and ni._HAVE_POINT_COLUMNS_ACCEL
        and ni._use_column_accel()
    ),
    reason="the point sheet needs both C++ near-interface extensions",
)

K_P = 2.0 * np.pi / WL7
EPS_T = complex(
    SOIL_A[0], -SOIL_A[1] / (2.0 * np.pi * (299792458.0 / WL7) * 8.8541878128e-12)
)
K_M = k_medium(EPS_T, K_P)
A_WIRE = 1e-3
FAMILIES = {"six": ni.KEY_RPOW, "point": ni.POINT_KEY_RPOW}


@pytest.fixture(autouse=True)
def _fresh_sheets(monkeypatch):
    """Each test starts with no cached sheet and zero counters."""
    monkeypatch.setattr(ni, "_SHEET_CACHE", type(ni._SHEET_CACHE)())
    monkeypatch.setattr(ni, "_SHEET_STATS", dict.fromkeys(ni._SHEET_STATS, 0))


# ----------------------------------------------------------------------
# the kernels: each family's sheet against its twin
# ----------------------------------------------------------------------


def _twin(sub, family):
    """The exact route at each row, each a column of one."""
    fn = (
        ni._nia.near_interface_six_columns
        if family == "six"
        else ni._nipa.near_interface_point_columns
    )
    n = sub.shape[0]
    return np.asarray(
        fn(
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


def _plane_rows(zp, n=400, rmax=14.0, seed=11):
    """Rows on the plane z' = zp, stratified in R (the distance from the
    sheet's singular point) by decade from |zp| to `rmax`: in each decade R
    log-uniform with s = z - z' uniform in [|zp|, R], plus the interface
    (z -> 0) and the axis (rho -> a). rho is folded with the wire radius, as
    the crossing rows ask it."""
    rng = np.random.default_rng(seed)
    d = abs(zp)
    rows = []
    lo = np.log10(d)
    for a in np.arange(np.floor(lo), np.log10(rmax)):
        a0, a1 = max(a, lo), min(a + 1.0, np.log10(rmax))
        if a1 <= a0:
            continue
        R = 10 ** rng.uniform(a0, a1, n)
        s = rng.uniform(d, np.maximum(R, d * (1 + 1e-12)))
        rows.append(np.stack([np.sqrt(np.maximum(R * R - s * s, 0.0)), s - d], 1))
    rho = rng.uniform(0, rmax, n)
    rows.append(np.stack([rho, 10 ** rng.uniform(-6, 0, n)], 1))  # the interface
    rows.append(np.stack([10 ** rng.uniform(-4, 0, n), rng.uniform(0, rmax, n)], 1))
    rz = np.concatenate(rows)
    rho = np.hypot(rz[:, 0], A_WIRE)
    return np.ascontiguousarray(np.stack([rho, rz[:, 1], np.full(rho.size, zp)], 1))


def _height_rows(h, n=600, rmax=40.0, dmax=3.0, seed=7):
    """`test_plane_sheet_1173`'s height rows: z = h over (rho, z' <= 0)."""
    rng = np.random.default_rng(seed)
    rho = np.concatenate(
        [rng.uniform(0, rmax, n), 10 ** rng.uniform(-3, np.log10(rmax), n)]
    )
    zp = -np.concatenate([rng.uniform(0, dmax, n), 10 ** rng.uniform(-6, 0, n)])
    zp[:5] = 0.0
    rho = np.hypot(rho, A_WIRE)
    return np.ascontiguousarray(np.stack([rho, np.full_like(zp, h), zp], axis=1))


def _decade_error(sub, got, ref, rpow, fixed):
    """Worst error over the rows, each relative to its kernel's envelope in
    its OWN decade of R: max over that decade's rows of |f R^pw|, over R^pw
    (a kernel crosses zero, so a pointwise ratio is meaningless there; the
    per-decade envelope is far stricter than one envelope over the sample)."""
    R = np.hypot(sub[:, 0], sub[:, 1] - sub[:, 2])
    pw = np.asarray(rpow)[None, :]
    dec = np.floor(np.log10(R / fixed)).astype(int)
    worst = 0.0
    for k in np.unique(dec):
        m = dec == k
        env = np.abs(ref[m] * R[m, None] ** pw).max(axis=0) / R[m, None] ** pw
        worst = max(worst, float((np.abs(got[m] - ref[m]) / env).max()))
    return worst


def _sheet_values(sub, family, height=False):
    if height:
        fixed = float(sub[0, 1])
        sheet = ni.PlaneSheet(
            K_P,
            K_M,
            fixed,
            ni._LAM_MULT,
            height=True,
            depth=ni._height_reach(float(-sub[:, 2].min())),
            family=family,
        )
    else:
        fixed = float(sub[0, 2])
        sheet = ni.PlaneSheet(K_P, K_M, fixed, ni._LAM_MULT, family=family)
    sheet.cover(float(sub[:, 0].max()), float((sub[:, 1] - sub[:, 2]).max()))
    out = np.empty((sub.shape[0], sheet.width), dtype=np.complex128)
    sheet.interpolate(sub, np.arange(sub.shape[0]), out)
    return sheet, out


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("zp", [-1e-3, -1e-2, -0.15])
def test_the_sheet_is_the_twin_on_every_plane_to_the_guard(family, zp):
    """Soil A, 7 MHz, per decade of R, on the radials' 0.15 m plane and on
    shallower planes down to the 1 mm guard (the corner is the twin's): the
    prototype read 5e-10 on the six keys and 1.4e-9 on the point keys."""
    sub = _plane_rows(zp)
    sheet, got = _sheet_values(sub, family)
    assert sheet.width == len(FAMILIES[family])
    err = _decade_error(sub, got, _twin(sub, family), FAMILIES[family], abs(zp))
    assert err < 1e-8, err


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_the_height_sheet_is_the_twin(family):
    """A 2.5 m height over rods to 3 m, reaching 40 m, per decade of R."""
    sub = _height_rows(2.5)
    _sheet, got = _sheet_values(sub, family, height=True)
    err = _decade_error(sub, got, _twin(sub, family), FAMILIES[family], 2.5)
    assert err < 1e-8, err


def test_a_coarse_point_table_fails_the_same_check(monkeypatch):
    """The negative control: a p = 3 point table must fail the tolerance the
    production table passes."""
    monkeypatch.setattr(ni, "_SHEET_P", 3)
    sub = _plane_rows(-0.15, n=150)
    _sheet, got = _sheet_values(sub, "point")
    err = _decade_error(sub, got, _twin(sub, "point"), ni.POINT_KEY_RPOW, 0.15)
    assert err > 1e-4, err


def test_the_point_sheet_shares_the_six_sheets_layout():
    """Same strips, cells and node offsets; only the width and the stored
    values differ. And the families are cached apart."""
    sub = _plane_rows(-0.15, n=60, rmax=3.0)
    six, _ = _sheet_values(sub, "six")
    pt, _ = _sheet_values(sub, "point")
    a, b = six.arrays(), pt.arrays()
    for i in range(3):
        assert np.array_equal(a[i], b[i])
    assert a[3].shape == (six.n_nodes, ni.N_KEYS)
    assert b[3].shape == (pt.n_nodes, ni.N_POINT_KEYS)
    s1 = ni._plane_sheet(K_P, K_M, -0.15, ni._LAM_MULT, 1.0, 1.0)
    s2 = ni._plane_sheet(K_P, K_M, -0.15, ni._LAM_MULT, 1.0, 1.0, family="point")
    assert s1.family == "six" and s2.family == "point" and s1 is not s2
    assert ni._SHEET_STATS["sheets_built"] == 1
    assert ni._SHEET_STATS["point_sheets_built"] == 1
    with pytest.raises(ValueError, match="no sheet family"):
        ni.PlaneSheet(K_P, K_M, -0.15, ni._LAM_MULT, family="seven")


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_a_node_stores_its_twin_value_times_r_to_its_power(family):
    """The storage convention, pinned at the nodes themselves: a node holds
    the twin's value there times R^pw with the family's own powers
    (`POINT_KEY_RPOW` = 2 for every point key). The interpolation divides the
    same powers back out, so a wrong power in the family table is invisible
    to the accuracy rows (f R and f R^2 are both smooth on a cell) and has to
    be caught here."""
    sub = _plane_rows(-0.15, n=30, rmax=2.0)
    sheet, _ = _sheet_values(sub, family)
    assert sheet.rpow == FAMILIES[family]
    p = ni._SHEET_P
    (a, b), off = sorted(sheet._off.items())[len(sheet._off) // 2]
    ra, rb, _i = sheet.cells["rho"][a]
    sa, sb, _j = sheet.cells["s"][b]
    rn = 0.5 * (ra + rb) + 0.5 * (rb - ra) * sheet.x
    sn = 0.5 * (sa + sb) + 0.5 * (sb - sa) * sheet.x
    rho, s = (g.ravel() for g in np.meshgrid(rn, sn, indexing="ij"))
    z = np.maximum(sheet.zp + s, 0.0)
    nodes = np.ascontiguousarray(np.stack([rho, z, np.full_like(z, sheet.zp)], 1))
    ref = _twin(nodes, family)
    R = np.hypot(rho, z - sheet.zp)
    stored = sheet.arrays()[3][off : off + p * p]
    want = ref * R[:, None] ** np.asarray(FAMILIES[family])[None, :]
    assert np.abs(stored - want).max() < 1e-10 * np.abs(want).max()


def test_a_plan_splits_the_point_rows_as_it_splits_the_six(monkeypatch):
    """`point_designed_rows` under a plan: plane rows from the point sheet,
    height rows (on no plane) from the height's, the rest the exact twin's
    grouped among themselves -- and every row within the tolerance."""
    plane = _plane_rows(-0.15, n=60)
    height = _height_rows(2.5, n=60, rmax=12.0)
    other = _plane_rows(-0.4, n=20, seed=9)
    rows = np.ascontiguousarray(np.concatenate([plane, height, other]))
    plan = ni.SheetPlan([-0.15], [(2.5, 4.0)])
    got = ni.point_designed_rows(EPS_T, K_P, rows, plan=plan)
    got = np.stack([got[k] for k in ni.POINT_KEYS], axis=1)
    n_p, n_h = plane.shape[0], height.shape[0]
    _s, want_p = _sheet_values(plane, "point")
    assert np.array_equal(got[:n_p], want_p)
    _s, want_h = _sheet_values(height, "point", height=True)
    assert np.array_equal(got[n_p : n_p + n_h], want_h)
    want_o = ni._point_columns_exact(EPS_T, K_P, other[:, 0], other[:, 1], other[:, 2])
    assert np.array_equal(got[n_p + n_h :], want_o)
    assert ni._SHEET_STATS["point_sheet_rows"] == n_p + n_h
    assert ni._SHEET_STATS["point_height_rows"] == n_h
    assert ni._SHEET_STATS["point_exact_rows"] == other.shape[0]
    # Off, or on the numpy route, the plan is ignored: every row exact.
    for flag, v in (("_SHEET", False), ("_FORCE_NUMPY", True)):
        with monkeypatch.context() as m:
            m.setattr(ni, flag, v)
            m.setattr(ni, "_SHEET_STATS", dict.fromkeys(ni._SHEET_STATS, 0))
            ni.point_designed_rows(EPS_T, K_P, rows[:5], plan=plan)
            assert ni._SHEET_STATS["point_sheet_rows"] == 0


# ----------------------------------------------------------------------
# the crossing rows of a real deck
# ----------------------------------------------------------------------


def _cross_rows(deck):
    """`SinusoidalSolver._crossing_point_rows` for `deck`, with the counters
    it ran up."""
    s = SinusoidalSolver(**deck)
    geom = s._build_geometry()
    below = np.asarray(geom["seg_centers"])[:, 2] < s.ground_z
    med = cf.buried_medium(s.ground_eps, s.omega, s.eps, s.k)
    view = s._basis_coefs(geom, np.where(below, med.k_m, med.k_p))
    stats = dict.fromkeys(ni._SHEET_STATS, 0)
    with pytest.MonkeyPatch.context() as m:
        m.setattr(ni, "_SHEET_STATS", stats)
        rows = s._crossing_point_rows(geom, view, med, below)
    return rows, stats


@pytest.fixture(scope="module")
def invl_rows():
    """The 8-radial inverted-L's crossing rows: sheets off, on (fresh
    cache), on through the dense reference route, and on over a cache whose
    sheets another caller had grown further first."""
    deck = invl_deck(n_radials=8, x=1)
    out = {}
    with pytest.MonkeyPatch.context() as m:
        m.setattr(ni, "_SHEET_CACHE", type(ni._SHEET_CACHE)())
        m.setattr(ni, "_SHEET", False)
        out["off"] = _cross_rows(deck)
        m.setattr(ni, "_SHEET", True)
        out["on"] = _cross_rows(deck)
        # (family, height, depth, k_p, k_m, zp, lam_mult) of every sheet
        # the fill used, to pre-grow below.
        used = [key[1:8] for key in ni._SHEET_CACHE]
        m.setattr(ni, "_SHEET_CACHE", type(ni._SHEET_CACHE)())
        m.setattr(cf, "_POINT_CHUNKED", False)
        out["dense"] = _cross_rows(deck)
        m.setattr(cf, "_POINT_CHUNKED", True)
        m.setattr(ni, "_SHEET_CACHE", type(ni._SHEET_CACHE)())
        for fam, height, depth, k_p, k_m, zp, lam in used:
            ni._plane_sheet(k_p, k_m, zp, lam, 40.0, 40.0, height, depth, fam)
        out["grown"] = _cross_rows(deck)
    return out, used


def test_the_point_rows_came_from_sheets(invl_rows):
    """The route ran: most of the block's point rows AND six-key rows were
    interpolated, from one sheet of each family; off, none was, and no fill
    was planned. A fallback to the exact twin cannot pass this row."""
    rows, used = invl_rows
    _r, on = rows["on"]
    _r, off = rows["off"]
    assert sorted(u[0] for u in used) == ["point", "six"]
    n_pt = on["point_sheet_rows"] + on["point_exact_rows"]
    n_six = on["sheet_rows"] + on["exact_rows"]
    assert on["fills_planned"] == 2 and on["planes_planned"] >= 1
    assert on["point_sheet_rows"] > 0.9 * n_pt, on
    assert on["sheet_rows"] > 0.9 * n_six, on
    assert on["point_sheets_built"] >= 1 and on["point_nodes_built"] > 0
    assert off["point_sheet_rows"] == off["sheet_rows"] == off["fills_planned"] == 0
    assert off["point_exact_rows"] == n_pt


def test_the_block_is_the_exact_block_at_the_tolerance(invl_rows):
    ref, _s = invl_rows[0]["off"]
    got, _s = invl_rows[0]["on"]
    assert not np.array_equal(got, ref)
    assert np.abs(got - ref).max() < 1e-9 * np.abs(ref).max()


@pytest.mark.parametrize("cut", ["dense", "grown"])
def test_the_cut_and_the_cache_do_not_show(invl_rows, cut):
    """The plan is the block's and a sheet node depends only on its key: the
    dense reference route, and a cache grown further by another caller,
    serve the same rows from the same values -- the block to the bit."""
    got, stats = invl_rows[0][cut]
    assert stats["point_sheet_rows"] > 0
    if cut == "grown":
        # The pre-grown sheets were the ones served (same key, reach
        # already covered): nothing was built during the fill.
        assert stats["sheets_built"] == stats["point_sheets_built"] == 0
        assert stats["nodes_built"] == stats["point_nodes_built"] == 0
    assert np.array_equal(got, invl_rows[0]["on"][0])


@pytest.fixture(scope="module")
def guarded_rows():
    """The 8-radial inverted-L's block with the guard moved below its
    radials' 0.15 m plane: heights priced as shipped, heights priced out,
    and the switched-off block."""
    deck = invl_deck(n_radials=8, x=1)
    out = {}
    with pytest.MonkeyPatch.context() as m:
        m.setattr(ni, "_SHEET_CACHE", type(ni._SHEET_CACHE)())
        m.setattr(ni, "_SHEET_MIN_DEPTH", 0.2)
        out["height"] = _cross_rows(deck)
        m.setattr(ni, "_SHEET_HEIGHT_WEIGHT", 1e9)
        out["guarded"] = _cross_rows(deck)
        m.setattr(ni, "_SHEET", False)
        out["off"] = _cross_rows(deck)
    return out


def test_a_plane_under_the_guard_is_left_exact(guarded_rows):
    """Under the guard the radials' rows are counted as guarded, no plane is
    planned, and (heights priced out) no row of either family takes a
    sheet: the block is the switched-off block's to the bit."""
    got, stats = guarded_rows["guarded"]
    assert stats["guarded_rows"] > 1000 and stats["planes_planned"] == 0
    assert stats["point_sheet_rows"] == stats["sheet_rows"] == 0
    assert np.array_equal(got, guarded_rows["off"][0])


def test_under_the_guard_the_top_wires_height_takes_over(guarded_rows):
    """With no plane, the top wire's 10 m height takes its height sheet for
    both families (the plan's second rule), within the tolerance of the
    exact block."""
    got, stats = guarded_rows["height"]
    ref, _s = guarded_rows["off"]
    assert stats["guarded_rows"] > 1000 and stats["planes_planned"] == 0
    assert stats["heights_planned"] == 2
    assert stats["point_sheet_rows"] == stats["point_height_rows"] > 0
    assert stats["sheet_rows"] == stats["height_rows"] > 0
    assert not np.array_equal(got, ref)
    assert np.abs(got - ref).max() < 1e-9 * np.abs(ref).max()


# ----------------------------------------------------------------------
# end to end: the solve's Z, sheets against the exact route
# ----------------------------------------------------------------------


def _hub(x):
    d = hub_deck(n_radials=16)
    npe = d["n_per_edge_per_wire"]
    d["n_per_edge_per_wire"] = [[n * x for n in e] for e in npe[:16]] + [
        npe[16],
        [npe[17][0] * x],
    ]
    return d


def _solve(deck):
    stats = dict.fromkeys(ni._SHEET_STATS, 0)
    with pytest.MonkeyPatch.context() as m:
        m.setattr(ni, "_SHEET_STATS", stats)
        z, _cur = SinusoidalSolver(**deck).compute_impedance()
    return complex(np.atleast_1d(z)[0]), stats


@pytest.mark.slow
@pytest.mark.parametrize(
    "name,deck",
    [
        ("invl x4", lambda: invl_deck(x=4)),
        ("invl x8", lambda: invl_deck(x=8)),
        ("hub x8", lambda: _hub(8)),
    ],
)
def test_the_solve_is_the_exact_solve_at_the_tolerance(monkeypatch, name, deck):
    """|dZ|/|Z| <= 1e-9 against `MOMWIRE_NEAR_INTERFACE_SHEET=0`; the
    prototype measured 5.8e-11 (invl x4) and 7.3e-12 (x8)."""
    d = deck()
    got, on = _solve(d)
    monkeypatch.setattr(ni, "_SHEET", False)
    ref, off = _solve(d)
    assert on["point_sheet_rows"] > 0.9 * (
        on["point_sheet_rows"] + on["point_exact_rows"]
    )
    assert off["point_sheet_rows"] == 0
    assert got != ref
    assert abs(got - ref) <= 1e-9 * abs(ref), (name, got, ref)
