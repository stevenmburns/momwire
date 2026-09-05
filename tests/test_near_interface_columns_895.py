"""momwire#895 — the FIXED per-(ρ, z′) column rule for the designed
near-interface integrals.

`six_columns` is the same integrand on the same contour as `six_point`,
with the panels decided ONCE per column instead of once per point: in
`_core` z enters only as e^{−γ₊z} and ρ only through the Bessel factor, so
a column shares the path, the nodes, the weights, the path derivative and
the Bessel factor exactly. It is not a grid and not an interpolation, which
is why the corner column is served like every other one.

Two things a fixed rule must own that an adaptive walk gets for free, and
both are gated here because both were measured wrong first (`_sub_seed`):

  * panels spanning at most a DOUBLING in λ — without it γ_p's branch
    point at λ = k_p, which `_head` brackets and then leaves in a single
    interval reaching a_head/7, reads 1.8e-8 at σ = 3 S/m class and 4e-9
    at ε̃ = 1 (where the two branch points coincide and u = 2λ/(γ₊+γ₋)
    becomes 1/√(λ−k_p) rather than a kink);
  * panels spanning at most ONE oscillation of J₀(λρ) — without it a 41 m
    radial puts 12 oscillations in one 24-point panel and reads 110 %
    wrong. `test_g895_7` is the red/green proof that both still bind.

The ladder is #680's LEDGER (the pinned structural set: corner, ε̃ = 1
identities, high-σ far pair, ρ = 0, kill cap, tiny s) plus `WIDE`, the
rows that DECIDED the two sub-seeding rules. Relative error is on the
VECTOR scale max|ref| over the six components, never per-component: W ≡ 0
at ε̃ = 1 and a per-component scale would divide by an exact zero.

Two tolerances, and they say different things. Against `six_point` the bar
is 1e-10, the REFERENCE's own rtol — past that the reference is the
uncertain one (measured: it moves 3.8e-12 between rtol 1e-12 and 1e-14 on
a 37 m near-interface pair). What says the column rule is converged is the
p / p+1 PAIR at 1e-12, which needs no reference at all.

Parity gates run BEFORE integrated gates (the #568 lesson: an integrated
gate cannot see a conditioning defect).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from momwire import _ground_refl, _near_interface as ni
from momwire.bspline import BSplineSolver

from test_ble_1937_838 import ble_deck
from test_crossing_serve_524 import fan_rise_deck

C0 = 299792458.0
F7 = 7e6
K7 = 2.0 * np.pi * F7 / C0
OM7 = 2.0 * np.pi * F7
EPS0 = 8.8541878128e-12
A_WIRE = 0.001

SOIL_A = _ground_refl.eps_tilde((13.0, 0.005), OM7, EPS0)
HIGH_SIGMA = _ground_refl.eps_tilde((13.0, 5.0), OM7, EPS0)

# The #680 structural ledger, unchanged — one row per branch of the walk.
LEDGER = [
    ("corner", SOIL_A, K7, (A_WIRE, 0.0, 0.0)),
    ("eps1-corner", 1.0, K7, (A_WIRE, 0.0, 0.0)),
    ("eps1-generic", 1.0, K7, (0.3, 0.2, -0.4)),
    ("eps1-rho0", 1.0, K7, (0.0, 1.0, -0.5)),
    ("high-sigma-far", HIGH_SIGMA, K7, (0.5, 3.0, -3.0)),
    ("rho0", SOIL_A, K7, (0.0, 0.05, -0.03)),
    ("kill-cap", SOIL_A, K7, (1.0, 8.0, -6.0)),
    ("generic", SOIL_A, K7, (0.3, 0.5, -0.2)),
    ("tiny-s", SOIL_A, K7, (1e-5, 1e-6, -1e-6)),
]

# The rows a fixed rule needs and a per-point walk does not: each one broke
# an earlier spelling of `_sub_seed`, and the comment says which half.
# They are gated against the reference, NOT as a converged pair — at ρ of
# tens of metres the mid alternates sign over hundreds of radians and both
# machines sit on a cancellation floor near 1e-11 (measured non-monotone in
# p, and the reference moves 3.8e-12 across its own rtol ladder there).
WIDE = [
    # oscillation half: a 41 m radial, the BRV-class screen's outer node.
    ("radial-41m", SOIL_A, K7, (41.0, 0.5, -0.2)),
    ("radial-41m-deep", SOIL_A, K7, (41.0, 3.0, -0.9)),
    # doubling half: |k_m| ≫ k_p squeezes the whole k_p region into the
    # first 0.8 % of the head, leaving one interval spanning a factor 12.5.
    (
        "sigma-3-midband",
        _ground_refl.eps_tilde((18.13, 2.982), 2.0 * np.pi * 4.28e6, EPS0),
        2.0 * np.pi * 4.28e6 / C0,
        (3.457, 0.06177, -0.002989),
    ),
    # doubling half again, from the other side: nearly lossless ground at
    # VHF puts k_m within 1.6 k_p, so the two branch points nearly merge.
    (
        "dry-vhf",
        _ground_refl.eps_tilde((2.5, 0.001), 2.0 * np.pi * 30e6, EPS0),
        2.0 * np.pi * 30e6 / C0,
        (0.3, 0.2, -0.4),
    ),
]


def _rel(got, ref):
    """Relative on the VECTOR scale — see the module docstring."""
    return float(np.max(np.abs(got - ref))) / max(float(np.max(np.abs(ref))), 1e-300)


def _embedding(z, zp):
    """The ledger point plus 30 log-spaced z between 1e-4 and 20 m, so the
    row is read BOTH as a column of one and as a member of a tall column
    whose s_min some other member sets."""
    return np.concatenate([[z], np.geomspace(1e-4, 20.0, 30)])


# ----------------------------------------------------------------------
# The parity gates
# ----------------------------------------------------------------------


@pytest.mark.parametrize("label,eps_t,k2,pt", LEDGER, ids=[r[0] for r in LEDGER])
def test_g895_1_pointwise_parity_alone_and_in_a_column(label, eps_t, k2, pt):
    """Every ledger point against `six_point`, as a one-point column and
    embedded in a 31-z column sharing its (ρ, z′). The bar is the
    reference's own rtol, 1e-10; measured worst 1.5e-15 either way, which
    is why the pair gate below and not this one is the convergence
    statement."""
    rho, z, zp = pt
    ref = ni.six_point(eps_t, k2, rho, z, zp)
    alone = ni.six_columns(eps_t, k2, rho, [z], zp)
    assert alone.shape == (1, 6)
    assert _rel(alone[0], ref) <= 1e-10, f"{label}: one-point column departs"
    embedded = ni.six_columns(eps_t, k2, rho, _embedding(z, zp), zp)[0]
    assert _rel(embedded, ref) <= 1e-10, (
        f"{label}: the row moves when other z share its column"
    )


@pytest.mark.parametrize("label,eps_t,k2,pt", LEDGER, ids=[r[0] for r in LEDGER])
def test_g895_2_converged_pair_on_the_ledger(label, eps_t, k2, pt):
    """p against p+1 over the whole embedded column, 1e-12 relative. This
    is the gate that says the rule is converged INDEPENDENTLY of the
    reference's tolerance — `_COLUMN_P` carries no structure (`_sub_seed`
    does), so refining it must move nothing. Measured worst 1.4e-13, on
    the ε̃ = 1 rows."""
    rho, z, zp = pt
    zs = _embedding(z, zp)
    lo = ni.six_columns(eps_t, k2, rho, zs, zp, p=ni._COLUMN_P)
    hi = ni.six_columns(eps_t, k2, rho, zs, zp, p=ni._COLUMN_P + 1)
    worst = max(_rel(lo[i], hi[i]) for i in range(len(zs)))
    assert worst <= 1e-12, f"{label}: p / p+1 disagree at {worst:.3e}"


@pytest.mark.parametrize("label,eps_t,k2,pt", WIDE, ids=[r[0] for r in WIDE])
def test_g895_3_the_rows_that_decided_the_sub_seeding(label, eps_t, k2, pt):
    """`WIDE` against `six_point`. Each row is a regression on one half of
    `_sub_seed` (see its comment); without the rule the same rows read
    1.8e-8 and 110 %, which `test_g895_7` proves red/green."""
    rho, z, zp = pt
    ref = ni.six_point(eps_t, k2, rho, z, zp)
    got = ni.six_columns(eps_t, k2, rho, [z], zp)[0]
    assert _rel(got, ref) <= 1e-10, f"{label}: departs the reference walk"


def test_g895_4_eps1_identity_through_the_column_route():
    """test_g524_3 / test_g680_2's collapse identity, on the COLUMN route:
    at ε̃ = 1, U_T = k²V_T = e^{−jkR}/R exactly and W ≡ ∂zW ≡ 0. The rule
    must own the identity itself, not merely track the numpy walk — and it
    must own it for a point read out of a tall column too, since that is
    how production reads it."""
    pts = [(A_WIRE, 0.0, 0.0), (0.3, 0.2, -0.4), (0.0, 1.0, -0.5)]
    for rho, z, zp in pts:
        for zs, i in ((np.array([z]), 0), (_embedding(z, zp), 0)):
            six = ni.six_columns(1.0, K7, rho, zs, zp)[i]
            R = np.hypot(rho, z - zp)
            g = np.exp(-1j * K7 * R) / R
            assert abs(six[0] - g) <= 1e-12 * abs(g)
            assert abs(K7 * K7 * six[1] - g) <= 1e-12 * abs(g)
            assert abs(six[2]) <= 1e-12 * abs(g)
            assert abs(six[3]) <= 1e-12 * abs(g) / max(R, A_WIRE)


def test_g895_5_domain_contract_matches_six_point():
    """The walk's refusals, unchanged: z < 0, z′ > 0 and R = 0 refuse
    rather than evaluate, and a column refuses on ANY member — one bad z
    among good ones is still a bad column, with that z in the message."""
    with pytest.raises(ValueError, match="need z >= 0 >= zp"):
        ni.six_columns(SOIL_A, K7, 0.1, [-0.2], -0.3)
    with pytest.raises(ValueError, match="need z >= 0 >= zp"):
        ni.six_columns(SOIL_A, K7, 0.1, [0.2], 0.3)
    with pytest.raises(ValueError, match="need R > 0"):
        ni.six_columns(SOIL_A, K7, 0.0, [0.0], 0.0)
    with pytest.raises(ValueError, match=r"need z >= 0 >= zp, got \(-0.5"):
        ni.six_columns(SOIL_A, K7, 0.1, [0.2, -0.5, 0.4], -0.3)
    with pytest.raises(ValueError, match="need R > 0"):
        ni.six_columns(SOIL_A, K7, 0.0, [0.3, 0.0], 0.0)


def test_g895_6_route_switch_is_read_at_call_time_and_refuses_typos():
    """The dispatch handle. `column` is the default; `point` selects the
    per-point walk; anything else REFUSES rather than falling back, because
    a typo in the whole-run env switch would otherwise silently select the
    route it was set to avoid — which is the bisect the switch exists for."""
    assert ni._ROUTE == "column"
    assert ni._use_column_route() is True
    was = ni._ROUTE
    try:
        ni._ROUTE = "point"
        assert ni._use_column_route() is False
        ni._ROUTE = "colum"
        with pytest.raises(ValueError, match="MOMWIRE_NEAR_INTERFACE_ROUTE"):
            ni._use_column_route()
    finally:
        ni._ROUTE = was


def test_g895_7_both_sub_seeding_rules_are_load_bearing(monkeypatch):
    """Red/green on `_sub_seed`: with it neutered to a pass-through, the
    two `WIDE` halves go wrong by the amounts its docstring names — a 41 m
    radial by more than 1 % and the σ = 3 S/m head by more than 1e-9 —
    and with it they are inside 1e-10. Without this the rule could quietly
    stop firing and every other gate here would still pass, because the
    near-interface ledger does not need it."""
    rows = {label: (eps_t, k2, pt) for label, eps_t, k2, pt in WIDE}
    checks = (("radial-41m", 1e-2), ("sigma-3-midband", 1e-9))
    refs = {}
    for label, floor in checks:
        eps_t, k2, (rho, z, zp) = rows[label]
        refs[label] = ni.six_point(eps_t, k2, rho, z, zp)
        assert _rel(ni.six_columns(eps_t, k2, rho, [z], zp), refs[label]) <= 1e-10

    monkeypatch.setattr(ni, "_sub_seed", lambda edges, rho: list(edges))
    for label, floor in checks:
        eps_t, k2, (rho, z, zp) = rows[label]
        got = ni.six_columns(eps_t, k2, rho, [z], zp)[0]
        assert _rel(got, refs[label]) > floor, (
            f"{label}: neutering `_sub_seed` did not break it, so the rule "
            "it encodes is no longer what makes this row converge"
        )


# ----------------------------------------------------------------------
# The integrated gates (after the parity ones — the #568 lesson)
# ----------------------------------------------------------------------


def _routed(build_or_mesh, route):
    """`designed_tables` on `route` over one mesh, plus its memo."""
    eps_t, k2, rho, z, zp = build_or_mesh
    memo = {}
    was = ni._ROUTE
    try:
        ni._ROUTE = route
        tables = ni.designed_tables(eps_t, k2, rho, z, zp, rtol=1e-10, memo=memo)
    finally:
        ni._ROUTE = was
    return tables, memo


def test_g895_8_designed_tables_routes_and_matches_on_a_crossing_mesh():
    """The routing gate, on a mesh shaped like a crossing block: a mast's
    z against a radial's ρ, with IEEE-exact duplicate triples. Every
    triple within 1e-10 of the point route, and the memo contract holds on
    both — same keys, same FIRST-APPEARANCE order, every entry filled. The
    grouping is free to reorder the work; it must not reorder the memo."""
    rho = np.concatenate([np.geomspace(1e-3, 13.6, 12), [1e-3, 13.6]])
    z = np.geomspace(1e-4, 9.0, 8)[:, None] * np.ones((1, rho.size))
    zp = np.array([-0.1524])
    mesh = (SOIL_A, K7, rho[None, :] * np.ones((8, 1)), z, zp)

    got, memo_c = _routed(mesh, "column")
    ref, memo_p = _routed(mesh, "point")

    assert list(memo_c) == list(memo_p)
    assert len(memo_c) == 8 * 12  # the two duplicate ρ columns dedup away
    assert all(v is not None for v in memo_c.values())
    for key in ni.KEYS:
        scale = max(float(np.max(np.abs(ref[key]))), 1e-300)
        rel = float(np.max(np.abs(got[key] - ref[key]))) / scale
        assert rel <= 1e-10, (key, rel)
    # the duplicated columns are the SAME floats on the column route too:
    # the memo, not the arithmetic, is what makes a repeat a repeat
    for key in ni.KEYS:
        assert np.array_equal(got[key][:, 0], got[key][:, 12])
        assert np.array_equal(got[key][:, 11], got[key][:, 13])


@pytest.mark.slow
def test_g895_9_a_served_crossing_deck_solves_the_same_both_routes():
    """`fan_rise_deck` — a served crossing deck, ~5,000 unique triples in
    ~60 real columns — solved through both routes. The kernel agrees at
    1e-10 over every triple the fill asked for and the impedance to the
    digit. This is the gate the anchors' crossgate rows then re-run at
    their own pinned constants."""
    build = fan_rise_deck()
    seen, real = {}, ni.designed_tables

    def spy(eps_t, k2, rho, z, zp, **kw):
        out = real(eps_t, k2, rho, z, zp, **kw)
        r, zz, zpz = np.broadcast_arrays(
            np.asarray(rho, float), np.asarray(z, float), np.asarray(zp, float)
        )
        vals = np.stack([out[kk] for kk in ni.KEYS])
        it = np.nditer(r, flags=["multi_index"])
        for _ in it:
            ix = it.multi_index
            seen[(float(r[ix]), float(zz[ix]), float(zpz[ix]))] = vals[
                (slice(None),) + ix
            ]
        return out

    tables = {}
    for route in ("column", "point"):
        seen.clear()
        was = ni._ROUTE
        ni.designed_tables, ni._ROUTE = spy, route
        try:
            z_in, _ = BSplineSolver(**build).compute_impedance()
        finally:
            ni.designed_tables, ni._ROUTE = real, was
        tables[route] = (z_in, dict(seen))

    z_col, col = tables["column"]
    z_pt, pt = tables["point"]
    assert set(col) == set(pt)
    assert len(col) > 3000
    worst = max(_rel(col[k], pt[k]) for k in pt)
    assert worst <= 1e-10, f"kernel departs at {worst:.3e}"
    assert abs(z_col - z_pt) <= 1e-6, f"{z_col} vs {z_pt}"


@pytest.mark.slow
def test_g895_10_real_ble_columns_are_a_converged_pair():
    """The converged-pair gate on REAL columns rather than synthetic ones:
    `ble_deck(4)`'s own asked set, taken the way probe1's census spy takes
    it, then its on-axis z′ = 0 column (the corner column, 108 mast
    heights, s_min 1.1e-4 — the pair the module docstring says no grid can
    hold) and its radial-end column (ρ = 13.6 m, 109 heights) read at p and
    p+1. Measured 1e-15 class on both."""
    asked, real = [], ni.designed_tables

    def spy(eps_t, k2, rho, z, zp, **kw):
        r, zz, zpz = np.broadcast_arrays(
            np.asarray(rho, float), np.asarray(z, float), np.asarray(zp, float)
        )
        asked.extend(zip(r.ravel().tolist(), zz.ravel().tolist(), zpz.ravel().tolist()))
        spy.eps_t, spy.k2 = eps_t, k2
        return real(eps_t, k2, rho, z, zp, **kw)

    ni.designed_tables = spy
    try:
        BSplineSolver(**ble_deck(4)).compute_impedance()
    finally:
        ni.designed_tables = real

    columns = defaultdict(list)
    for r, z, zp in set(asked):
        columns[(r, zp)].append(z)
    on_axis = min(r for r, zp in columns if zp == 0.0)
    radial_end = max(r for r, _ in columns)
    picks = [
        ("on-axis, z' = 0", (on_axis, 0.0)),
        ("radial end", max((k for k in columns if k[0] == radial_end))),
    ]
    for label, key in picks:
        zs = np.sort(np.asarray(columns[key], dtype=float))
        assert zs.size >= 32, f"{label} is not a column: {zs.size} z"
        lo = ni.six_columns(spy.eps_t, spy.k2, key[0], zs, key[1], p=ni._COLUMN_P)
        hi = ni.six_columns(spy.eps_t, spy.k2, key[0], zs, key[1], p=ni._COLUMN_P + 1)
        worst = max(_rel(lo[i], hi[i]) for i in range(zs.size))
        assert worst <= 1e-12, f"{label} (rho={key[0]:.4g}): pair at {worst:.3e}"
