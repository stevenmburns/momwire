"""momwire#899 item 1 — the C++ twin of the FIXED per-ρ column rule.

`_near_interface_accel.near_interface_six_columns` is a WALK port of
`six_columns` (shared walk + limit, the house twin rule): the same head
detour with the same H rule, sevenths and marks, the same `_sub_seed`
doubling and J₀-oscillation halves, the same fixed Gauss panels at 2^p, the
same mid, the same geometric ray tail out to 60 decay lengths, the same
far-pair kill cap on the column's s_min — and then, per member, the same one
exponential per node and the same six dot products. So parity is rounding
class, and the gates here are RELATIVE at 1e-12, never bit (house rule: no
cross-build bit equality; the transcendental libraries and the dot-product
orders differ in the last bits, momwire#249/#270).

The reference is the numpy column rule, not the point walk: a twin is the
twin of ITS OWN walk. `six_columns` is already the numpy machine, so calling
it is already the bypass; where the dispatch is what is under test, the
reference side is forced with `_FORCE_NUMPY`.

Two ledgers, both imported rather than re-listed so they cannot drift:

  * `LEDGER` (#680's structural set: corner, ε̃ = 1 identities, high-σ far
    pair, ρ = 0, kill cap, tiny s) — one row per branch of the walk;
  * `WIDE` (#895's) — the rows that DECIDED `_sub_seed`. They are the ones
    that matter most here, because a fixed rule transcribed into a second
    language is exactly where a seeding rule goes quietly missing, and
    because at ρ of tens of metres the mid alternates sign over hundreds of
    radians: the 41 m column's Σ|terms| is 20× its own vector scale, so a
    last-bit difference in one Bessel factor arrives amplified. Measured
    worst on that row, 5.6e-14 — three decades inside the gate and the
    largest number in this file by an order.

Relative error is on the VECTOR scale max|ref| over the six components,
never per-component: W ≡ 0 at ε̃ = 1 and a per-component scale would divide
by an exact zero.

Parity gates run BEFORE integrated gates (the #568 lesson: an integrated
gate cannot see a conditioning defect).
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _ground_refl, _near_interface as ni
from momwire.bspline import BSplineSolver

from test_crossing_serve_524 import fan_rise_deck
from test_near_interface_columns_895 import LEDGER, WIDE, _embedding, _rel

C0 = 299792458.0
F7 = 7e6
K7 = 2.0 * np.pi * F7 / C0
OM7 = 2.0 * np.pi * F7
EPS0 = 8.8541878128e-12
A_WIRE = 0.001

SOIL_A = _ground_refl.eps_tilde((13.0, 0.005), OM7, EPS0)
HIGH_SIGMA = _ground_refl.eps_tilde((13.0, 5.0), OM7, EPS0)

# test_g899_1's mixed-z′ columns: one ρ, many (z, z′), including the corner
# column (z′ = 0 and buried z′ together) and the high-σ far pair.
MIXED = [
    (SOIL_A, A_WIRE, [0.0, 0.05, 1.0, 1.0, 8.0], [0.0, -0.1524, -0.1524, -3.0, 0.0]),
    (SOIL_A, 0.3, [0.5, 0.2, 4.0, 0.05], [-0.2, -0.4, -0.02, -1.5]),
    (HIGH_SIGMA, 0.5, [3.0, 0.1, 3.0], [-3.0, -0.05, 0.0]),
    (1.0, 0.3, [0.2, 1.0, 0.2], [-0.4, -0.5, -0.05]),
]

pytestmark = pytest.mark.skipif(
    not ni._HAVE_NEAR_INTERFACE_COLUMNS_ACCEL,
    reason="near-interface column accel not built (pure-Python install)",
)


def _twin(eps_t, k2, columns, p=None):
    """The twin over a list of (ρ, zs, zps) columns, concatenated the way
    `designed_tables` concatenates a whole fill's grouping. Returns the
    (n, 6) table in member order."""
    k_p = float(k2)
    k_m = ni.k_medium(complex(eps_t), k_p)
    rho = np.asarray([c[0] for c in columns], dtype=float)
    zs = np.concatenate([np.atleast_1d(np.asarray(c[1], float)) for c in columns])
    zps = np.concatenate(
        [
            np.broadcast_to(
                np.asarray(c[2], float), np.atleast_1d(np.asarray(c[1], float)).shape
            )
            for c in columns
        ]
    )
    offsets = np.zeros(len(columns) + 1, dtype=np.intp)
    offsets[1:] = np.cumsum(
        [np.atleast_1d(np.asarray(c[1], float)).size for c in columns]
    )
    return ni._nia.near_interface_six_columns(
        k_p,
        k_m,
        rho,
        offsets,
        np.ascontiguousarray(zs),
        np.ascontiguousarray(zps),
        ni._LAM_MULT,
        ni._COLUMN_P if p is None else int(p),
        ni._DETOUR,
        ni._physical_cpu_count(),
        ni._GX,
        ni._GW,
    )


def _worst(got, ref):
    return max(_rel(got[i], ref[i]) for i in range(len(ref)))


# ----------------------------------------------------------------------
# The parity gates
# ----------------------------------------------------------------------


@pytest.mark.parametrize("label,eps_t,k2,pt", LEDGER, ids=[r[0] for r in LEDGER])
def test_g899c_1_ledger_parity_alone_and_in_a_column(label, eps_t, k2, pt):
    """#680's structural ledger against the numpy column rule, as a
    one-point column and embedded in a 31-z column whose s_min some other
    member sets. Measured worst 5.5e-15."""
    rho, z, zp = pt
    alone = _twin(eps_t, k2, [(rho, [z], zp)])
    assert alone.shape == (1, 6)
    ref = ni.six_columns(eps_t, k2, rho, [z], zp)
    assert _rel(alone[0], ref[0]) <= 1e-12, f"{label}: one-point column departs"

    zs = _embedding(z, zp)
    got = _twin(eps_t, k2, [(rho, zs, zp)])
    ref = ni.six_columns(eps_t, k2, rho, zs, zp)
    assert _worst(got, ref) <= 1e-12, f"{label}: embedded, {_worst(got, ref):.3e}"


@pytest.mark.parametrize("label,eps_t,k2,pt", WIDE, ids=[r[0] for r in WIDE])
def test_g899c_2_the_rows_that_decided_the_sub_seeding(label, eps_t, k2, pt):
    """`WIDE` — the rows that decided `_sub_seed`, so the rows that say the
    transcription kept both of its halves. Neutered, they go wrong by 110 %
    (the 41 m radial's oscillation half) and 1.8e-8 (the σ = 3 S/m doubling
    half); here they must sit at rounding class. Measured worst 5.6e-14,
    which is the cancellation on the 41 m column, not the seeding."""
    rho, z, zp = pt
    alone = _twin(eps_t, k2, [(rho, [z], zp)])
    ref = ni.six_columns(eps_t, k2, rho, [z], zp)
    assert _rel(alone[0], ref[0]) <= 1e-12, f"{label}: one-point column departs"

    zs = _embedding(z, zp)
    got = _twin(eps_t, k2, [(rho, zs, zp)])
    ref = ni.six_columns(eps_t, k2, rho, zs, zp)
    assert _worst(got, ref) <= 1e-12, f"{label}: embedded, {_worst(got, ref):.3e}"


def test_g899c_3_mixed_zp_columns_and_a_hundred_member_one():
    """test_g899_1's mixed-z′ columns — z′ rides in the exponential exactly
    as z does, so a column is a ρ and nothing but s_min is shared — plus a
    128-member column, which is past the point where the twin stops
    parallelising over columns and starts parallelising over members. All
    four columns go in ONE call, so the offsets contract is under test too:
    a member must be answered from its own column's rule."""
    # k_m is per CALL, so the columns are batched by ε̃ — which is what
    # `designed_tables` does too, one call per fill.
    for eps_t in dict.fromkeys(row[0] for row in MIXED):
        cols = [(rho, zs, zps) for e, rho, zs, zps in MIXED if e == eps_t]
        got = _twin(eps_t, K7, cols)
        lo = 0
        for rho, zs, zps in cols:
            ref = ni.six_columns(eps_t, K7, rho, zs, zps)
            worst = _worst(got[lo : lo + len(zs)], ref)
            assert worst <= 1e-12, (eps_t, rho, worst)
            lo += len(zs)

    zs = np.geomspace(1e-4, 30.0, 128)
    zps = -np.geomspace(1e-3, 3.0, 128)
    ref = ni.six_columns(SOIL_A, K7, 13.6, zs, zps)
    got = _twin(SOIL_A, K7, [(13.6, zs, zps)])
    assert got.shape == (128, 6)
    assert _worst(got, ref) <= 1e-12, _worst(got, ref)


def test_g899c_4_a_column_is_answered_the_same_whichever_call_carries_it():
    """The offsets are the whole contract of the concatenated form: a column
    handed over alone, first of many and last of many is the same column, so
    it must read the same to the bit. Not a tolerance — the rule and the
    members are identical arithmetic in identical order; only the OpenMP
    schedule differs, and a schedule that changed an answer would be a
    shared-state bug."""
    a = (0.3, [0.5, 0.2, 4.0], [-0.2, -0.4, -0.02])
    b = (13.6, list(np.geomspace(1e-4, 9.0, 40)), -0.1524)
    c = (A_WIRE, [0.0, 0.05], [0.0, -0.03])
    alone = _twin(SOIL_A, K7, [a])
    first = _twin(SOIL_A, K7, [a, b, c])[:3]
    last = _twin(SOIL_A, K7, [b, c, a])[-3:]
    assert np.array_equal(alone, first)
    assert np.array_equal(alone, last)


def test_g899c_5_domain_refusals_hold_on_any_member_of_any_column():
    """The walk's contract raises, on the twin's own entry: z < 0, z′ > 0
    and R = 0 refuse rather than evaluate, wherever in the concatenation the
    bad member sits — including the LAST member of the LAST column, which is
    the one a validation pass placed inside the parallel region would miss
    (a throw cannot cross an omp boundary, so it would abort instead)."""
    good = (0.3, [0.5, 0.2], [-0.2, -0.4])
    for bad in (
        (0.1, [-0.2], -0.3),  # z < 0
        (0.1, [0.2], 0.3),  # z' > 0
        (0.0, [0.0], 0.0),  # R = 0
        (0.1, [0.2, -0.5, 0.4], -0.3),  # mid-column
    ):
        with pytest.raises(ValueError, match="need (z >= 0 >= zp|R > 0)"):
            _twin(SOIL_A, K7, [bad])
        with pytest.raises(ValueError, match="need (z >= 0 >= zp|R > 0)"):
            _twin(SOIL_A, K7, [good, good, bad])


def test_g899c_6_the_capability_flag_is_this_entrys_own_attr(monkeypatch):
    """The flag keys on `near_interface_columns_899`, not on the point
    twin's `near_interface_680` and not on a shared one: a .so built between
    the two arcs exports the point entry and NOT this one, and a shared flag
    would claim a contract it cannot serve. With the column flag down the
    point twin is still live and the column route still serves — through
    `six_columns`."""
    assert ni._HAVE_NEAR_INTERFACE_COLUMNS_ACCEL is (
        ni._nia is not None and bool(getattr(ni._nia, "near_interface_columns_899", 0))
    )
    assert ni._use_column_accel() is True

    monkeypatch.setattr(ni, "_HAVE_NEAR_INTERFACE_COLUMNS_ACCEL", False)
    assert ni._use_column_accel() is False
    assert ni._use_near_interface_accel() is True  # the other twin is untouched
    calls = []
    real = ni.six_columns
    monkeypatch.setattr(
        ni, "six_columns", lambda *a, **kw: (calls.append(1), real(*a, **kw))[1]
    )
    ni.designed_tables(SOIL_A, K7, 0.3, 0.2, -0.2)
    assert calls, "the column route did not fall back to the numpy rule"


def test_g899c_7_force_numpy_turns_this_twin_off_too(monkeypatch):
    """`_FORCE_NUMPY` means "the numpy walks are the machine". A timing
    comparison or a bisect that turned off one twin and not the other would
    answer neither question, so the switch has to reach both — and the point
    route must not reach this entry at all."""
    seen = []
    monkeypatch.setattr(
        ni._nia,
        "near_interface_six_columns",
        lambda *a, **kw: seen.append(1),
    )
    monkeypatch.setattr(ni, "_FORCE_NUMPY", True)
    ni.designed_tables(SOIL_A, K7, np.array([0.3, 13.6]), 0.2, -0.2)
    assert seen == [], "the twin served with _FORCE_NUMPY set"

    monkeypatch.setattr(ni, "_FORCE_NUMPY", False)
    monkeypatch.setattr(ni, "_ROUTE", "point")
    ni.designed_tables(SOIL_A, K7, np.array([0.3, 13.6]), 0.2, -0.2)
    assert seen == [], "the point route reached the column twin"


def test_g899c_7b_the_twin_serves_by_default_and_the_numpy_rule_does_not(monkeypatch):
    """The positive half of the dispatch: with nothing forced, one call to
    `designed_tables` on the column route is ONE call to the twin carrying
    every column, and `six_columns` is never reached. Without this the
    routing gate below could pass with the twin silently unreachable — both
    sides would then be the numpy rule agreeing with itself."""
    twin_calls, numpy_calls = [], []
    real_twin = ni._nia.near_interface_six_columns

    def spy_twin(k_p, k_m, rho, offsets, *a, **kw):
        twin_calls.append((len(rho), int(offsets[-1])))
        return real_twin(k_p, k_m, rho, offsets, *a, **kw)

    monkeypatch.setattr(ni._nia, "near_interface_six_columns", spy_twin)
    monkeypatch.setattr(ni, "six_columns", lambda *a, **kw: numpy_calls.append(1))
    monkeypatch.setattr(ni, "_FORCE_NUMPY", False)
    monkeypatch.setattr(ni, "_ROUTE", "column")
    rho = np.array([0.3, 2.0, 13.6])[None, :, None]
    z = np.array([0.01, 0.5, 2.0, 9.0])[:, None, None]
    zp = np.array([-0.1524, -0.9])[None, None, :]
    out = ni.designed_tables(SOIL_A, K7, rho, z, zp)
    assert twin_calls == [(3, 24)], twin_calls  # three columns, 24 members, one call
    assert numpy_calls == []
    assert all(out[k].shape == (4, 3, 2) for k in ni.KEYS)


def test_g899c_8_designed_tables_refuses_before_it_hands_anything_over():
    """A bad member anywhere in the ask refuses in the WALK's words, with
    the offending member's own numbers, and before the twin is called at
    all — the C++ entry refuses the same set, but from there it cannot
    spell the values."""
    with pytest.raises(ValueError, match=r"need z >= 0 >= zp, got \(-0.5, -0.3\)"):
        ni.designed_tables(SOIL_A, K7, 0.1, np.array([0.2, -0.5]), -0.3)
    with pytest.raises(ValueError, match=r"need R > 0, got rho=0.0, s=0.0"):
        ni.designed_tables(SOIL_A, K7, 0.0, np.array([0.3, 0.0]), 0.0)


# ----------------------------------------------------------------------
# The integrated gates (after the parity ones — the #568 lesson)
# ----------------------------------------------------------------------


def _routed(mesh, force_numpy):
    eps_t, k2, rho, z, zp = mesh
    memo = {}
    was = ni._FORCE_NUMPY
    try:
        ni._FORCE_NUMPY = force_numpy
        tables = ni.designed_tables(eps_t, k2, rho, z, zp, rtol=1e-10, memo=memo)
    finally:
        ni._FORCE_NUMPY = was
    return tables, memo


def test_g899c_9_designed_tables_routes_and_matches_on_a_crossing_mesh():
    """test_g895_8's routing gate with the twin on one side and the forced
    numpy column route on the other: a mesh shaped like a crossing block, a
    mast's z against a radial's ρ, with IEEE-exact duplicate triples. Same
    memo keys in the same FIRST-APPEARANCE order, every entry filled, every
    value within 1e-12 — and the duplicated columns still the SAME floats,
    because the memo and not the arithmetic is what makes a repeat a repeat."""
    rho = np.concatenate([np.geomspace(1e-3, 13.6, 12), [1e-3, 13.6]])
    z = np.geomspace(1e-4, 9.0, 8)[:, None] * np.ones((1, rho.size))
    mesh = (SOIL_A, K7, rho[None, :] * np.ones((8, 1)), z, np.array([-0.1524]))

    got, memo_t = _routed(mesh, force_numpy=False)
    ref, memo_n = _routed(mesh, force_numpy=True)

    assert list(memo_t) == list(memo_n)
    assert len(memo_t) == 8 * 12  # the two duplicate ρ columns dedup away
    assert all(v is not None for v in memo_t.values())
    for key in ni.KEYS:
        scale = max(float(np.max(np.abs(ref[key]))), 1e-300)
        rel = float(np.max(np.abs(got[key] - ref[key]))) / scale
        assert rel <= 1e-12, (key, rel)
        assert np.array_equal(got[key][:, 0], got[key][:, 12])
        assert np.array_equal(got[key][:, 11], got[key][:, 13])


@pytest.mark.slow
def test_g899c_10_a_served_crossing_deck_solves_the_same_both_machines():
    """test_g895_9's shape with the two COLUMN machines in it: `fan_rise_deck`
    — a served crossing deck, ~5,000 unique triples in ~60 real columns —
    solved through the twin and through the forced numpy column rule. The
    kernel agrees at 1e-12 over every triple the fill asked for and the
    impedance to well inside a microhm. This is the gate the anchors'
    crossgate rows then re-run at their own pinned constants."""
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
    for force in (False, True):
        seen.clear()
        was = ni._FORCE_NUMPY
        ni.designed_tables, ni._FORCE_NUMPY = spy, force
        try:
            z_in, _ = BSplineSolver(**build).compute_impedance()
        finally:
            ni.designed_tables, ni._FORCE_NUMPY = real, was
        tables[force] = (z_in, dict(seen))

    z_tw, tw = tables[False]
    z_np, npy = tables[True]
    assert set(tw) == set(npy)
    assert len(tw) > 3000
    worst = max(_rel(tw[k], npy[k]) for k in npy)
    assert worst <= 1e-12, f"kernel departs at {worst:.3e}"
    assert abs(z_tw - z_np) <= 1e-6, f"{z_tw} vs {z_np}"
