"""momwire#899 (items 2 and 3) — a column is a ρ, and the dedup is one
`np.unique`.

Item 2: `six_columns` takes z′ per member. In `_core` z and z′ both enter
only through e^{γ₋z′ − γ₊z}, so nothing in the rule depends on z′ beyond
`s_min`, and `designed_tables` groups the unique triples by ρ alone — the
per-z′ columns of one radial merge (138 → 84 groups on BLE N = 4) and the
singletons halve. The gates: a mixed-z′ column matches the reference walk
at the reference's own 1e-10 on the #680 ledger's shapes, and the fill
makes exactly one column call per distinct ρ.

Item 3: the two Python passes over the FULL asked set (~3× the unique set
on a crossing fill, 10–15 % of the route) are one `np.unique`. The memo
contract it feeds is unchanged and pinned here directly: Python-float
triple keys in first-appearance order, −0.0 folding with 0.0 as it does in
a dict, every asked point scattered from its unique row, and the empty ask
served.
"""

import numpy as np

from momwire import _ground_refl, _near_interface as ni

C0 = 299792458.0
F7 = 7e6
K7 = 2.0 * np.pi * F7 / C0
OM7 = 2.0 * np.pi * F7
EPS0 = 8.8541878128e-12
A_WIRE = 0.001
SOIL_A = _ground_refl.eps_tilde((13.0, 0.005), OM7, EPS0)
HIGH_SIGMA = _ground_refl.eps_tilde((13.0, 5.0), OM7, EPS0)


def _rel(got, ref):
    return float(np.max(np.abs(got - ref))) / max(float(np.max(np.abs(ref))), 1e-300)


def test_g899_1_a_mixed_zp_column_matches_the_reference_walk():
    """One ρ, many (z, z′): the corner column (ρ = a, z′ = 0 and buried z′
    together), a generic column and the high-σ far pair, each against
    `six_point` at 1e-10. The rule sees only s_min; z′ rides in the
    exponential exactly as z does."""
    cases = [
        (
            SOIL_A,
            A_WIRE,
            [0.0, 0.05, 1.0, 1.0, 8.0],
            [0.0, -0.1524, -0.1524, -3.0, 0.0],
        ),
        (SOIL_A, 0.3, [0.5, 0.2, 4.0, 0.05], [-0.2, -0.4, -0.02, -1.5]),
        (HIGH_SIGMA, 0.5, [3.0, 0.1, 3.0], [-3.0, -0.05, 0.0]),
        (1.0, 0.3, [0.2, 1.0, 0.2], [-0.4, -0.5, -0.05]),
    ]
    for eps_t, rho, zs, zps in cases:
        got = ni.six_columns(eps_t, K7, rho, zs, zps)
        assert got.shape == (len(zs), 6)
        for i, (z, zp) in enumerate(zip(zs, zps)):
            ref = ni.six_point(eps_t, K7, rho, z, zp)
            assert _rel(got[i], ref) <= 1e-10, (rho, z, zp, _rel(got[i], ref))


def test_g899_2_scalar_zp_is_the_broadcast_of_the_array_form():
    """The #895 signature (one z′ for the column) is the array form with
    that z′ repeated — the same bits, not merely the same answer."""
    zs = np.geomspace(1e-4, 20.0, 31)
    a = ni.six_columns(SOIL_A, K7, 0.3, zs, -0.2)
    b = ni.six_columns(SOIL_A, K7, 0.3, zs, np.full(zs.shape, -0.2))
    assert np.array_equal(a, b)


def test_g899_3_refusals_name_the_offending_member():
    import pytest

    with pytest.raises(ValueError, match=r"need z >= 0 >= zp, got \(0.4, 0.1\)"):
        ni.six_columns(SOIL_A, K7, 0.1, [0.2, 0.4], [-0.3, 0.1])
    with pytest.raises(ValueError, match="need R > 0"):
        ni.six_columns(SOIL_A, K7, 0.0, [0.3, 0.0], [-0.1, 0.0])


def test_g899_4_designed_tables_makes_one_column_call_per_distinct_rho(monkeypatch):
    """A crossing-shaped mesh: 3 radial ρ × 4 mast z × 2 buried z′. Before
    #899 that was 6 columns; now 3, one per ρ, each carrying 8 members —
    and the answer, the memo keys and their order are the point route's."""
    calls, real = [], ni.six_columns

    def spy(eps_t, k2, rho, zs, zp, **kw):
        calls.append((float(rho), len(np.atleast_1d(zs))))
        return real(eps_t, k2, rho, zs, zp, **kw)

    monkeypatch.setattr(ni, "six_columns", spy)
    rho = np.array([0.3, 2.0, 13.6])[None, :, None]
    z = np.array([0.01, 0.5, 2.0, 9.0])[:, None, None]
    zp = np.array([-0.1524, -0.9])[None, None, :]
    memo_c, memo_p = {}, {}
    monkeypatch.setattr(ni, "_ROUTE", "column")
    got = ni.designed_tables(SOIL_A, K7, rho, z, zp, memo=memo_c)
    monkeypatch.setattr(ni, "_ROUTE", "point")
    ref = ni.designed_tables(SOIL_A, K7, rho, z, zp, memo=memo_p)

    assert sorted(calls) == [(0.3, 8), (2.0, 8), (13.6, 8)]
    assert list(memo_c) == list(memo_p)
    assert len(memo_c) == 24
    for key in ni.KEYS:
        assert got[key].shape == (4, 3, 2)
        scale = max(float(np.max(np.abs(ref[key]))), 1e-300)
        assert float(np.max(np.abs(got[key] - ref[key]))) / scale <= 1e-10


def test_g899_5_unique_triples_contract():
    """First-appearance order, float-tuple keys, −0.0 folded with 0.0, and
    an inverse that scatters every asked point from its own row."""
    rho = np.array([[0.5, 0.3, 0.5, -0.0], [0.3, 0.5, 0.0, 0.3]])
    z = np.array([[1.0, 2.0, 1.0, 0.0], [2.0, 3.0, 0.0, 2.0]])
    zp = np.array([[-1.0, -2.0, -1.0, 0.0], [-2.0, -1.0, 0.0, -2.0]])
    keys, inverse = ni._unique_triples(rho, z, zp)
    assert keys == [
        (0.5, 1.0, -1.0),
        (0.3, 2.0, -2.0),
        (0.0, 0.0, 0.0),
        (0.5, 3.0, -1.0),
    ]
    assert all(type(v) is float for k in keys for v in k)
    assert inverse.tolist() == [0, 1, 0, 2, 1, 3, 2, 1]
    tri = np.stack([rho.ravel(), z.ravel(), zp.ravel()], axis=1)
    assert np.array_equal(np.asarray(keys)[inverse], tri)  # −0.0 == 0.0 here too


def test_g899_6_the_scatter_and_the_memo_are_the_same_floats():
    """Every asked point is the memo row of its triple, bit for bit, on a
    mesh with IEEE-exact duplicates and a cross-call memo — and an empty
    ask returns the right shapes without touching anything."""
    rho = np.array([0.3, 0.5, 0.3, 0.5])
    z = np.array([[0.2], [1.0]])
    memo = {}
    first = ni.designed_tables(SOIL_A, K7, rho, z, -0.2, memo=memo)
    again = ni.designed_tables(SOIL_A, K7, rho, z, -0.2, memo=memo)
    assert len(memo) == 4
    for i, key in enumerate(ni.KEYS):
        assert first[key].shape == (2, 4)
        assert np.array_equal(first[key], again[key])
        assert np.array_equal(first[key][:, 0], first[key][:, 2])
        for zi, zz in enumerate([0.2, 1.0]):
            for ri, rr in enumerate([0.3, 0.5, 0.3, 0.5]):
                assert first[key][zi, ri] == memo[(rr, zz, -0.2)][i]
    empty = ni.designed_tables(SOIL_A, K7, np.empty((0, 3)), 0.2, -0.2)
    assert all(empty[k].shape == (0, 3) for k in ni.KEYS)
