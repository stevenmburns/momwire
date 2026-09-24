"""The crossing fill's tables from their product structure — momwire#1173
design B.

The main sandwich asks (ρ_eff, z, z′) at every (above node, below node) pair.
Grouping one side's nodes by exact (x, y) makes ρ a function of (group, other
node), so the distinct triples are a union of per-group products {z} ×
{(ρ_eff, z_line)}: `_product_plan` builds them from their O(N) factors
instead of sorting the asked grid, `ProductMemo` holds them as factors, and
an end-loop ask that is wholly product rows is served by a gather
(`_fast_end_rows`). The below-plane fill's two all-pairs checks are answered
by bounds first (`RazorSolver._below_remainder_clear_of_floor`,
`_potential_ground._r1_max_below`), falling back to the exact walk.

Everything is BIT-IDENTICAL by construction, so the gate is the bits: Z of a
razor fill through its real constructor (`_assemble_Z`, where it is filled)
with every switch on, against the same fill with every switch off — the grid
dedup, the lookup path and the two exact walks, i.e. the code on main — at
`np.array_equal`. The route counters prove which route ran on each deck, so a
green row cannot be a fallback measuring nothing, and each deck family is
chosen to reach a different route:

  crossing1, detached         one group on each side: product, slots z and z′
  fan_rise, sloped            one above group, a below line spanning depths
  detached_hub                three source radii: one product per partition
  two_node (cap lifted)       TWO above groups: the multi-group merge
  lean20                      a leaning mast: every node its own group, grid route
  buried_dipole               no crossing fill: the two below-plane bounds only

The negative controls make the route wrong on purpose and require the same
comparison to FAIL: rows mapped back to the grid in the wrong (transposed)
order, the evaluation split in two calls (which changes columns' s_min), and
every fast end served one product row off.

Measured 2026-09-24 by script against origin/main 9479305, Z to the bit (see
the PR): razor hub_deck(16) x2 / x4 / x8 (x16 on Skylake), crossing_deck(1),
the detached pair and hub, lean20, two_node, fan_rise, sloped radials, the
buried dipole and WA7ARK's ground-rod EFHW, nec5 lane and the default lane.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from momwire import _crossing_fill as cf
from momwire import _near_interface as ni
from momwire import _potential_ground as pg
from momwire import _sommerfeld_below as sb
from momwire import razor as rz
from momwire.razor import RazorSolver

from test_crossing_serve_524 import (
    A_WIRE,
    SOIL_A,
    WL7,
    crossing_deck,
    fan_rise_deck,
    two_node_deck,
)
from test_razor_detached_1149 import detached, detached_hub
from test_tilted_crossing_936 import _crossing_deck as tilted_deck
from test_triple_memo_1168 import _hub, _razor

# ----------------------------------------------------------------------
# decks
# ----------------------------------------------------------------------


def sloped_radials_deck(depths=(0.10, 0.20, 0.30, 0.40), length=5.0, n_seg=10):
    """Radials from the mast base IN the plane down to a per-radial depth at
    their far ends, so every radial spans many depths (no hub, no rise), and
    a 10 m mast above the same node."""
    wires = []
    for i, d in enumerate(depths):
        a = 2 * math.pi * i / len(depths) + 0.3
        wires.append(
            np.array(
                [(0.0, 0.0, 0.0), (length * math.cos(a), length * math.sin(a), -d)]
            )
        )
    npe = [[n_seg] for _ in depths]
    wires.append(np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 10.0)]))
    npe.append([15])
    return dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        junctions=[[(i, "start") for i in range(len(depths) + 1)]],
        feeds=[(len(depths), 4.3333333333, 1 + 0j)],
        wavelength=WL7,
        wire_radius=A_WIRE,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )


def buried_dipole_razor():
    """test_buried_serve_553's phase-0 vertical dipole, as razor-2p."""
    n, length, depth = 11, 1.0, 0.15
    return RazorSolver(
        wires=[np.array([(0.0, 0.0, -(depth + length)), (0.0, 0.0, -depth)])],
        n_per_edge_per_wire=[[n]],
        feeds=[(0, ((n + 1) // 2 - 0.5) / n * length, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
        nec5_quadrature=True,
    )


# deck -> (maker, the main-sandwich route each cross block must take)
DECKS = {
    "crossing1": (lambda: _razor(crossing_deck(1)), "product"),
    "detached": (lambda: _razor(detached()), "product"),
    "detached_hub": (lambda: _razor(detached_hub()), "product"),
    "fan_rise": (lambda: _razor(fan_rise_deck()), "product"),
    "sloped": (lambda: _razor(sloped_radials_deck()), "product"),
    "lean20": (lambda: _razor(tilted_deck(20.0, n_mast=6)), "groups"),
    "buried_dipole": (buried_dipole_razor, None),
}

_OFF = {
    (cf, "_PRODUCT_TABLES"): False,
    (cf, "_PRODUCT_ENDS"): False,
    (rz, "_GRAZING_BOUND"): False,
    (pg, "_R1_BRACKET"): False,
}


def _reset_counters():
    for d in (cf._ROUTES, rz._GRAZING_ROUTES, pg._R1_ROUTES):
        for k in d:
            d[k] = 0


def _fill(make, mp=None, off=False, **flags):
    """Z of a fresh solver's fill, the route counters after it."""
    _reset_counters()
    with pytest.MonkeyPatch.context() as m:
        if off:
            for (mod, name), v in _OFF.items():
                m.setattr(mod, name, v)
        for key, v in flags.items():
            mod, name = key.split(".")
            m.setattr({"cf": cf, "rz": rz, "pg": pg}[mod], name, v)
        s = make()
        Z = s._assemble_Z(s._build_geometry(), s.k)
    routes = {
        **{f"cf.{k}": v for k, v in cf._ROUTES.items()},
        **{f"rz.{k}": v for k, v in rz._GRAZING_ROUTES.items()},
        **{f"pg.{k}": v for k, v in pg._R1_ROUTES.items()},
    }
    return np.array(Z, copy=True), routes


# ----------------------------------------------------------------------
# the integrated gate: Z to the bit, with the route proven
# ----------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(DECKS))
def test_the_fill_is_the_grid_route_to_the_bit(case):
    make, main_route = DECKS[case]
    ref, r_off = _fill(make, off=True)
    got, r_on = _fill(make)
    assert np.array_equal(got, ref)
    # The reference ran none of the new routes...
    assert r_off["cf.main_product"] == 0 and r_off["cf.ends_fast"] == 0
    assert r_off["rz.bound"] == 0 and r_off["pg.bracket"] == 0
    # ...and the gated fill ran the ones this deck is for.
    if main_route == "product":
        assert r_on["cf.main_product"] >= 2 and r_on["cf.main_generic"] == 0, r_on
        assert r_on["cf.ends_fast"] > 0, r_on
    elif main_route is not None:
        assert r_on["cf.main_product"] == 0, r_on
        assert r_on[f"cf.main_generic_{main_route}"] >= 2, r_on
    # Every deck here passes the below fill's two bounds.
    assert r_on["rz.bound"] >= 1 and r_on["rz.exact"] == 0, r_on
    assert r_on["pg.bracket"] >= 1 and r_on["pg.exact"] == 0, r_on


@pytest.mark.slow
def test_two_groups_take_the_merged_product_to_the_bit():
    """`two_node_deck`'s two masts are two above groups. Its asked triples
    barely repeat (each mast sees its own rod at ρ = a and the other's at
    12 m), so the candidate cap sends it to the grid route by default; lifted,
    the multi-group merge runs and must give the same Z."""
    make = lambda: _razor(two_node_deck(separation=12.0))  # noqa: E731
    ref, _r = _fill(make, off=True)
    got, r = _fill(make)
    assert r["cf.main_generic_candidates"] == 2 and r["cf.main_product"] == 0, r
    assert np.array_equal(got, ref)
    got, r = _fill(make, **{"cf._PRODUCT_MAX_CAND_FRAC": 1.0})
    assert r["cf.main_product"] == 2 and r["cf.main_product_groups"] == 2, r
    assert r["cf.ends_fast"] > 0, r
    assert np.array_equal(got, ref)


def test_the_below_side_groups_when_it_is_the_cheaper_side():
    """A mast leaning 20° over ONE vertical buried rod: the above nodes are
    all distinct in (x, y), the below rod is one group, so the product runs
    grouped on the z′ slot, key-major."""
    deck = tilted_deck(20.0, n_mast=6)
    deck["wires"][0] = np.array([(0.0, 0.0, -2.0), (0.0, 0.0, 0.0)])
    deck["n_per_edge_per_wire"][0] = [8]
    make = lambda: _razor(deck)  # noqa: E731
    ref, _r = _fill(make, off=True)
    got, r = _fill(make)
    assert r["cf.main_product_zp"] >= 1, r
    assert np.array_equal(got, ref)


@pytest.mark.parametrize("budget", [None, 20_000])
def test_chunked_product_tables_are_the_grid_routes(budget):
    """At a tiny budget the product serves hundreds of chunks through the
    streamed contraction; Z does not move."""
    kw = {} if budget is None else {"cf._MAIN_CHUNK_BYTES": budget}
    make = lambda: _razor(detached())  # noqa: E731
    ref, _r = _fill(make, off=True, **kw)
    got, r = _fill(make, **kw)
    assert r["cf.main_product"] == 2
    assert np.array_equal(got, ref)


# ----------------------------------------------------------------------
# negative controls: the same gate FAILS on a wrong route
# ----------------------------------------------------------------------


@pytest.mark.parametrize("control", ["transpose", "split", "row"])
def test_negative_controls_move_z(control):
    """ "transpose" maps a one-group product's rows back to the grid in the
    other factor's order (the product order permuted), "split" evaluates the
    rows as two calls (columns' s_min change), "row" serves every fast end one
    product row off. Each must move Z, so the gate above can see each."""
    make = lambda: _razor(crossing_deck(1))  # noqa: E731
    ref, _r = _fill(make, off=True)
    got, r = _fill(make, **{"cf._PRODUCT_NEG_CONTROL": control})
    assert r["cf.main_product"] == 2, r
    moved = int(np.count_nonzero(got != ref))
    assert moved > 0, f"{control}: the gate is blind to it"


# ----------------------------------------------------------------------
# unit rows: the plan against the grid dedup it replaces
# ----------------------------------------------------------------------


def _grid_rows(ctx, A, B, gz):
    """The rows `_chunked_tables` hands `designed_rows`: the grid's distinct
    folded triples in first-appearance order (one `_unique_rows` call)."""
    nA, nB = A["nodes"].shape[0], B["nodes"].shape[0]
    rho, (zA,), (zB,), _s = cf._direct_coords(
        [(A, B, np.arange(nA), np.arange(nB))], gz
    )
    r = ni.radius_fold(rho.reshape(nA, nB), float(ctx.a_wire))
    rows, _inv = ni._unique_rows(
        r, np.broadcast_to(zA[:, None], r.shape), np.broadcast_to(zB[None, :], r.shape)
    )
    return rows


def _axes(xy_a, z_a, xy_b, z_b):
    A = {"nodes": np.column_stack([np.asarray(xy_a, float), np.asarray(z_a, float)])}
    B = {"nodes": np.column_stack([np.asarray(xy_b, float), np.asarray(z_b, float)])}
    return A, B


class _Ctx:
    a_wire = 0.001


def _plan_rows(A, B, monkeypatch, **flags):
    seen = []
    real = ni.designed_rows_permuted

    def spy(eps_t, k2, rows, rtol=1e-10, lam_mult=ni._LAM_MULT):
        seen.append(np.array(rows, copy=True))
        v = np.zeros((rows.shape[0], 6), dtype=np.complex128)
        v[:, 0] = np.arange(rows.shape[0])  # a value naming its row
        return v, None

    with monkeypatch.context() as mp:
        mp.setattr(ni, "designed_rows_permuted", spy)
        for k, v in flags.items():
            mp.setattr(cf, k, v)
        plan = cf._product_plan(_Ctx(), 1.0, 1.0, A, B, 0.0)
    del real
    return plan, seen


@pytest.mark.parametrize("shape", ["one_group", "below_group", "two_groups"])
def test_plan_rows_are_the_grid_rows_awkward_values(shape, monkeypatch):
    """The rows handed to the evaluation are `_unique_rows`' own — same
    floats (sign of zero included), same order — on node sets that repeat z
    values, carry −0.0 against 0.0 in x, y and z, and (two_groups) share
    triples between groups; and the served tables' value rows name the grid
    pair's own row."""
    rng = np.random.default_rng(1173)
    zs_a = rng.choice([0.5, 1.0, 1.5, 2.0], 12)
    zs_a[3] = 0.0
    zs_b = -rng.choice([0.0, 0.1, 0.2], 9)
    zs_b[0] = -0.0
    xy_b = rng.choice([0.0, 1.0, 2.0], (9, 2))
    xy_b[2] = (-0.0, 0.0)
    if shape == "one_group":
        xy_a = np.zeros((12, 2))
        xy_a[5] = (-0.0, -0.0)
        A, B = _axes(xy_a, zs_a, xy_b, zs_b)
    elif shape == "below_group":
        A, B = _axes(rng.choice([0.0, 3.0, 4.0], (12, 2)), zs_a, np.zeros((9, 2)), zs_b)
    else:
        xy_a = np.zeros((12, 2))
        xy_a[6:] = (2.0, 0.0)  # a second mast; triples at ρ = 2 repeat across
        A, B = _axes(xy_a, zs_a, xy_b, zs_b)
    flags = {"_PRODUCT_MAX_CAND_FRAC": 10.0, "_PRODUCT_MAX_GROUP_FRAC": 1.0}
    plan, seen = _plan_rows(A, B, monkeypatch, **flags)
    assert not isinstance(plan, str), plan
    product, fast, chunk_idx = plan
    want = _grid_rows(_Ctx(), A, B, 0.0)
    (got,) = seen
    assert np.array_equal(got, want)
    assert np.array_equal(np.signbit(got), np.signbit(want))
    # Each grid pair's served value row is its triple's row.
    nA, nB = A["nodes"].shape[0], B["nodes"].shape[0]
    idx = chunk_idx(slice(0, nB))
    rows_of = product.value_rows(want)
    assert np.array_equal(rows_of, np.arange(want.shape[0]))
    rho, (zA,), (zB,), _s = cf._direct_coords(
        [(A, B, np.arange(nA), np.arange(nB))], 0.0
    )
    r = ni.radius_fold(rho.reshape(nA, nB), 0.001)
    asked = np.stack([r.ravel(), np.repeat(zA, nB), np.tile(zB, nA)], axis=1)
    assert np.array_equal(idx.ravel(), product.value_rows(asked + 0.0))
    assert (product.slot == "zp") == (shape == "below_group")


def test_product_set_lookup_folds_signed_zero_and_misses_nan(monkeypatch):
    A, B = _axes(
        np.zeros((3, 2)), [0.0, 1.0, 1.0], [(1.0, 0.0), (2.0, 0.0)], [-0.0, -1.0]
    )
    (product, _fast, _idx), _seen = _plan_rows(A, B, monkeypatch)
    r1 = float(ni.radius_fold(1.0, 0.001))
    ask = np.array(
        [
            [r1, -0.0, 0.0],  # both zeros flipped: held
            [r1, 1.0, -1.0],  # held (a's z 1.0 against the second line node's z)
            [r1, 0.5, -0.0],  # z not in the product
            [np.nan, 0.0, 0.0],
            [r1, np.nan, 0.0],
        ]
    )
    got = product.value_rows(ask)
    assert got[0] >= 0 and got[2:].tolist() == [-1, -1, -1]
    assert got[1] == -1  # (ρ to node 0, z′ of node 1) is no grid pair's key


def test_product_memo_is_a_triple_memo_holding_the_union(monkeypatch):
    """Lookup through a `ProductMemo` against a `TripleMemo` that stored the
    same rows and values: the same hits, the same floats; inserted rows go to
    the array part and are found; keys and values list the union in order."""
    A, B = _axes(
        np.zeros((4, 2)), [0.5, 1.0, 1.0, 2.0], [(1.0, 0.0), (0.0, 2.0)], [-0.1, -0.2]
    )
    (product, _f, _i), (rows,) = _plan_rows(A, B, monkeypatch)
    memo = ni.ProductMemo()
    memo.set_product(product)
    ref = ni.TripleMemo()
    ref.insert(rows, product.vals[product.row_vrow])
    extra = np.array([[3.0, 0.5, -0.1], [3.0, 7.0, -0.1]])
    memo.insert(extra, np.full((2, 6), 9 + 1j))
    ref.insert(extra, np.full((2, 6), 9 + 1j))
    ask = np.concatenate([rows[::-1], extra, [[5.0, 5.0, -5.0]]]) + 0.0
    ask[0, 2] = -0.0 if ask[0, 2] == 0 else ask[0, 2]
    h1, b1 = memo.lookup(ask)
    h2, b2 = ref.lookup(ask)
    assert np.array_equal(h1, h2) and np.array_equal(b1[h1], b2[h2])
    assert memo.keys() == ref.keys() and len(memo) == len(ref)
    assert all(np.array_equal(x, y) for x, y in zip(memo.values(), ref.values()))
    with pytest.raises(ValueError, match="once"):
        memo.set_product(product)


def test_first_groups_is_the_unique_tri_grouping():
    rng = np.random.default_rng(904)
    a = rng.integers(0, 3, 300).astype(float)
    b = rng.integers(0, 3, 300).astype(float)
    a[rng.random(300) < 0.1] = -0.0
    b[rng.random(300) < 0.05] = np.nan
    first, rank = cf._first_groups(a, b)
    rows, inv = ni._unique_tri(np.stack([a, b, np.zeros(300)], axis=1))
    assert np.array_equal(inv, rank)
    assert np.array_equal(rows[:, :2], np.stack([a, b], 1)[first], equal_nan=True)


# ----------------------------------------------------------------------
# the below-plane bounds: the same decision as the exact walk, both outcomes
# ----------------------------------------------------------------------


def _grazing_deck(depth=1e-4):
    """test_razor_grazing_shared_1168's deck: two buried wires 3 m apart at
    1e-4 m, whose pairs sit under the floor."""
    return dict(
        wires=[
            np.array([(0.0, 0.0, -depth), (1.0, 0.0, -depth)]),
            np.array([(4.0, 0.0, -depth), (5.0, 0.0, -depth)]),
        ],
        n_per_edge_per_wire=[[4], [4]],
        feeds=[(0, 0.5, 1 + 0j)],
        wavelength=WL7,
        wire_radius=A_WIRE,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )


@pytest.mark.parametrize("depth", [1e-4, 2e-3, 0.15])
def test_the_grazing_bound_decides_as_the_walk(depth):
    """Bound on vs off: the same refusal (the same sentence, or None). At
    1e-4 m the bound is inconclusive and the walk refuses; at 0.15 m the bound
    proves the deck clear; and whenever it does, the walk agrees."""
    s = RazorSolver(**_grazing_deck(depth), n_qp_path=8)
    geom = s._build_geometry()
    floor = math.radians(sb._SOMM_BELOW_TH_MIN_DEG)
    clear = s._below_remainder_clear_of_floor(geom, floor)
    th, _hh = s._below_remainder_th_min(geom)
    if clear:
        assert th >= floor
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(rz, "_GRAZING_BOUND", False)
        want = s._below_plane_grazing_refusal(geom)
    _reset_counters()
    got = s._below_plane_grazing_refusal(geom)
    assert got == want
    assert rz._GRAZING_ROUTES == (
        {"bound": 1, "exact": 0} if clear else {"bound": 0, "exact": 1}
    )
    if depth == 1e-4:
        assert not clear and got is not None
    if depth == 0.15:
        assert clear and got is None


def _cloud(rng, n, span, depth):
    p = rng.uniform(-span, span, (n, 3))
    p[:, 2] = -rng.uniform(0.01, depth, n)
    return p


@pytest.mark.parametrize("seed", range(6))
def test_the_r1_bracket_builds_the_same_grid(seed):
    """`_r1_max_below` against the exact all-pairs value, through the bucket
    the grid is keyed on; the exact route is taken (and counted) whenever the
    bracket straddles an edge, which the scaled clouds force."""
    rng = np.random.default_rng(seed)
    eps_t, k_p = 13.0 - 12.84j, 2 * math.pi / WL7
    lam_m = sb._lambda_m(eps_t, k_p)
    obs, src = _cloud(rng, 200, 3.0, 0.4), _cloud(rng, 150, 3.0, 0.4)
    exact = pg._r1_max_exact(obs, src, -obs[:, 2], -src[:, 2])
    # Scale both clouds so the exact radius sits just under a bucket edge.
    edge = sb._r1_bucket_below_wl(exact, lam_m) * lam_m
    for target in (edge * (1 - 1e-4), edge * (1 - 0.1)):
        f = target / exact
        o, e = obs * f, src * f
        d_o, d_s = -o[:, 2], -e[:, 2]
        _reset_counters()
        got = pg._r1_max_below(o, e, d_o, d_s, eps_t, k_p)
        want = pg._r1_max_exact(o, e, d_o, d_s)
        assert sb._r1_bucket_below_wl(got, lam_m) == sb._r1_bucket_below_wl(want, lam_m)
        if pg._R1_ROUTES["exact"]:
            assert got == want
    # And the one-shot spelling the chunked walk replaced.
    rho = np.hypot(obs[:, 0][:, None] - src[:, 0], obs[:, 1][:, None] - src[:, 1])
    one = float(np.max(np.hypot(rho, -obs[:, 2][:, None] - src[:, 2][None, :]))) * 1.001
    assert pg._r1_max_exact(obs, src, -obs[:, 2], -src[:, 2]) == one


def test_the_r1_bracket_takes_both_routes():
    """Near an edge the bracket straddles it and the exact walk answers; far
    from one the bracket answers."""
    rng = np.random.default_rng(7)
    eps_t, k_p = 13.0 - 12.84j, 2 * math.pi / WL7
    lam_m = sb._lambda_m(eps_t, k_p)
    obs, src = _cloud(rng, 100, 3.0, 0.4), _cloud(rng, 100, 3.0, 0.4)
    exact = pg._r1_max_exact(obs, src, -obs[:, 2], -src[:, 2])
    edge = sb._r1_bucket_below_wl(exact, lam_m) * lam_m / 1.001
    seen = {}
    for name, target in (("exact", edge * (1 - 1e-6)), ("bracket", edge * 0.9)):
        f = target / exact * 1.001
        o, e = obs * f, src * f
        _reset_counters()
        pg._r1_max_below(o, e, -o[:, 2], -e[:, 2], eps_t, k_p)
        seen[name] = dict(pg._R1_ROUTES)
    assert seen["exact"] == {"bracket": 0, "exact": 1}, seen
    assert seen["bracket"] == {"bracket": 1, "exact": 0}, seen


def test_one_bucket_is_read_across_an_edge():
    eps_t, k_p = 13.0 - 12.84j, 2 * math.pi / WL7
    lam_m = sb._lambda_m(eps_t, k_p)
    e = 1.25**3 * lam_m
    assert sb.r1_bracket_one_bucket(e * 0.85, e * 0.99, eps_t, k_p)
    assert not sb.r1_bracket_one_bucket(e * 0.85, e * 1.01, eps_t, k_p)
    assert not sb.r1_bracket_one_bucket(e * 0.9, e * (1 - 1e-12), eps_t, k_p)
    assert not sb.r1_bracket_one_bucket(e * 0.9, np.nan, eps_t, k_p)
    # Past the cap every radius is one grid.
    cap = sb._SOMM_BELOW_R1_CAP_LAMBDA_M * lam_m
    assert sb.r1_bracket_one_bucket(cap * 1.1, cap * 9.0, eps_t, k_p)


# ----------------------------------------------------------------------
# memory hygiene: what the path axis no longer pins
# ----------------------------------------------------------------------


def test_path_axis_ends_pin_no_caller_array():
    """The T2 endpoints are copies and the one-hots windows of one read-only
    buffer: an axis over many rows keeps no per-row whole-geometry array."""
    cent = np.arange(30.0).reshape(10, 3)
    rows = [
        (
            m,
            np.zeros((2, 3)),
            np.ones((2, 3)),
            np.ones(2),
            np.zeros(2),
            cent[m],
            cent[m + 1],
        )
        for m in range(5)
    ]
    ax = cf.path_test_axis(7, rows)
    for (pt, _s, fv), m in zip(ax["ends"], np.repeat(np.arange(5), 2)):
        assert not np.shares_memory(pt, cent)
        assert fv.dtype == np.float64 and fv.shape == (7,) and not fv.flags.writeable
        assert np.flatnonzero(fv).tolist() == [m] and fv[m] == 1.0


@pytest.mark.slow
@pytest.mark.parametrize("x", [2, 4])
def test_hub16_is_the_grid_route_to_the_bit(x):
    """The headline deck at x2 and x4: product on both blocks, the ends fast,
    both bounds, Z to the bit."""
    make = lambda: _razor(_hub(x))  # noqa: E731
    ref, _r = _fill(make, off=True)
    got, r = _fill(make)
    assert r["cf.main_product"] == 2 and r["cf.ends_fast"] > 1000, r
    assert r["rz.bound"] >= 1 and r["pg.bracket"] >= 1, r
    assert np.array_equal(got, ref)


@pytest.mark.parametrize("corner", [False, True])
def test_the_support_block_is_the_full_blocks_slice(corner, monkeypatch):
    """`support=` against the full block sliced, both orientations, through
    razor's own calls (their axes and contexts): the same bits. `corner=True`
    reaches the corner's scatter too (razor itself passes False)."""
    seen = []
    for name in ("cross_complete_block", "cross_complete_block_reversed"):
        real = getattr(cf, name)

        def spy(ctx, T, S, *, corner_, support=None, _real=real, **kw):
            del corner_
            compact = _real(ctx, T, S, corner=corner, support=support, **kw)
            full = _real(ctx, T, S, corner=corner, **kw)
            rows, cols = support
            seen.append(np.array_equal(compact, full[np.ix_(rows, cols)]))
            return _real(ctx, T, S, corner=False, support=support, **kw)

        monkeypatch.setattr(
            rz._crossing_fill,
            name,
            lambda ctx, T, S, corner=False, _spy=spy, **kw: _spy(
                ctx, T, S, corner_=corner, **kw
            ),
        )
    s = _razor(detached_hub())
    s._assemble_Z(s._build_geometry(), s.k)
    # One forward call per below radius (three) and one reversed (the mast).
    assert len(seen) == 4 and all(seen), seen
