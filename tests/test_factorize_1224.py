"""The hash grouping kernels of momwire#1224 (`_accel_factorize.cpp`).

The crossing fill's dedups and its exact-triple memo answered with sorts and
sorted runs; the kernels answer the same questions with one hash table. What
they answer is integers and copied floats -- each row's group in
first-appearance order, the stored row an exact-`==` query names, the values
an earlier insert stored -- so the kernels and the numpy spellings they
replace must agree EXACTLY, and every row here compares with `array_equal`
(on the bits where a float is handed back), never a tolerance.

The unit rows drive both spellings in one process on the values the
equality rule is about: -0.0 against 0.0 (one value), NaN (never equal, a
row of its own), repeats, strided columns. The integrated rows solve each
lane's buried and inverted-L crossing decks with the kernels on and off and
pin Z and the currents bit for bit, with counters proving the kernels ran.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _accel
from momwire import _crossing_fill as cf
from momwire import _near_interface as ni
from momwire.bspline import BSplineSolver
from momwire.razor import RazorSolver
from momwire.sinusoidal import SinusoidalSolver
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

from test_crossing_serve_524 import hub_deck, invl_deck

pytestmark = pytest.mark.skipif(
    not ni._HAVE_FACTORIZE_ACCEL,
    reason="the accelerator does not carry the factorize kernels",
)


def _awkward(n, k, seed, *, dup=3):
    """(n, k) float rows drawn from n // dup distinct ones, with -0.0 and 0.0
    planted in the same column, NaNs, and repeats of the NaN rows."""
    if n == 0:
        return np.empty((0, k))
    rng = np.random.default_rng(seed)
    base = np.round(rng.standard_normal((max(1, n // dup), k)), 1)
    rows = base[rng.integers(0, base.shape[0], n)].copy()
    rows[rng.integers(0, n, n // 7), rng.integers(0, k, n // 7)] = 0.0
    rows[rng.integers(0, n, n // 7), rng.integers(0, k, n // 7)] = -0.0
    nan_at = rng.integers(0, n, max(1, n // 50))
    rows[nan_at, rng.integers(0, k, nan_at.size)] = np.nan
    rows[rng.integers(0, n, nan_at.size)] = rows[nan_at]
    return rows


def _sorted_route(monkeypatch):
    monkeypatch.setattr(ni, "_FACTORIZE", False)


@pytest.mark.parametrize("n", [0, 1, 2, 7, 1000, 50_000])
def test_unique_tri_is_the_lexsort_spelling(n, monkeypatch):
    tri = _awkward(n, 3, 1224 + n)
    got_u, got_inv = ni._unique_tri(tri)
    _sorted_route(monkeypatch)
    want_u, want_inv = ni._unique_tri(tri)
    assert np.array_equal(got_inv, want_inv)
    # The representatives are the first occurrence's floats, sign of zero
    # and NaN payload included.
    assert np.array_equal(got_u.view(np.uint64), want_u.view(np.uint64))


@pytest.mark.parametrize("k", [1, 2])
def test_first_groups_on_strided_columns_is_the_lexsort_spelling(k, monkeypatch):
    rows = _awkward(20_000, 3, 7 + k)
    cols = tuple(rows[:, c] for c in range(k))  # strided views, read in place
    got = cf._first_groups(*cols)
    _sorted_route(monkeypatch)
    want = cf._first_groups(*cols)
    for g, w in zip(got, want):
        assert g.dtype == w.dtype and np.array_equal(g, w)


def test_first_ints_is_the_unique_spelling(monkeypatch):
    ids = np.random.default_rng(3).integers(0, 500, 40_000)
    got = cf._first_ints(ids)
    _sorted_route(monkeypatch)
    want = cf._first_ints(ids)
    for g, w in zip(got, want):
        assert np.array_equal(g, w)


def test_row_groups_is_one_grouping_over_the_parts(monkeypatch):
    """Fed in parts, `RowGroups` numbers and represents the groups as one
    `_unique_tri` over the concatenation (the sorted spelling)."""
    tri = _awkward(30_000, 3, 99)
    groups = _accel.acc.RowGroups()
    cuts = [0, 1, 7, 7, 5000, 12_345, 30_000]
    ids = np.concatenate(
        [
            groups.add([tri[a:b, 0], tri[a:b, 1], tri[a:b, 2]])
            for a, b in zip(cuts[:-1], cuts[1:])
        ]
    )
    _sorted_route(monkeypatch)
    want_u, want_inv = ni._unique_tri(tri)
    assert np.array_equal(ids, want_inv)
    assert len(groups) == want_u.shape[0]
    assert np.array_equal(groups.rows().view(np.uint64), want_u.view(np.uint64))


def test_the_kernel_refuses_shapes_it_cannot_group():
    acc = _accel.acc
    with pytest.raises(RuntimeError):
        acc.factorize_rows([])
    with pytest.raises(RuntimeError):
        acc.factorize_rows([np.zeros(3)] * 4)
    with pytest.raises(RuntimeError):
        acc.factorize_rows([np.zeros(3), np.zeros(4)])


def test_key_index_ids_match_the_sorted_searches(monkeypatch):
    rng = np.random.default_rng(11)
    rows = np.unique(np.round(rng.standard_normal((5000, 2)), 2), axis=0)
    key_r, key_zl = rows[:, 0].copy(), rows[:, 1].copy()
    q = np.concatenate(
        [
            rows[rng.integers(0, rows.shape[0], 3000)],
            np.round(rng.standard_normal((500, 2)), 2),
        ]
    )
    q[::97, 0] = np.nan
    q[np.flatnonzero(q[:, 1] == 0.0), 1] = -0.0  # the fold, query side
    hashed = ni.KeyIndex(key_r, key_zl)
    assert hashed._index is not None, "the hash index did not serve"
    _sorted_route(monkeypatch)
    sorted_ = ni.KeyIndex(key_r, key_zl)
    assert sorted_._index is None
    got, want = hashed.ids(q[:, 0], q[:, 1]), sorted_.ids(q[:, 0], q[:, 1])
    assert got.dtype == want.dtype and np.array_equal(got, want)
    for g, w in zip(hashed.take_r_classes(), sorted_.take_r_classes()):
        assert np.array_equal(g, w)


def _vals(rows):
    r = np.nan_to_num(np.asarray(rows, dtype=float), nan=0.5)
    s = r[:, 0] + 3.0 * r[:, 1] + 7.0 * r[:, 2]
    return s[:, None] * np.arange(1.0, ni.N_KEYS + 1.0) * (1 + 1j)


def test_the_table_store_answers_as_the_sorted_runs(monkeypatch):
    """The same inserts and lookups through both stores: hits, the floats they
    return, `contains`, length, and keys/values in insertion order -- with a
    repeated key inserted twice (the first insertion answers), NaN rows
    (stored, never found) and -0.0 asked for a stored 0.0. Small merge
    thresholds walk the runs through every transition."""
    monkeypatch.setattr(ni, "_MEMO_MERGE_MIN_ROWS", 40)
    monkeypatch.setattr(ni, "_MEMO_MAX_PENDING_RUNS", 3)
    rng = np.random.default_rng(1224)
    table = ni.TripleMemo()
    assert table._table is not None, "the table store did not serve"
    monkeypatch.setattr(ni, "_HASH_MEMO", False)
    runs = ni.TripleMemo()
    assert runs._table is None
    seen = []
    for size in (1, 5, 30, 2, 80, 1, 13, 200, 3):
        rows = _awkward(size, 3, int(rng.integers(1 << 30)), dup=1)
        vals = _vals(rows) * (1 + len(seen))  # a repeat's later value differs
        for m in (table, runs):
            m.insert(rows, vals)
        seen.append(rows)
        ask = np.concatenate(seen + [_awkward(50, 3, size)])
        ask[ask == 0.0] = -0.0
        h_t, b_t = table.lookup(ask)
        h_r, b_r = runs.lookup(ask)
        assert np.array_equal(h_t, h_r)
        assert np.array_equal(b_t[h_t].view(np.uint64), b_r[h_r].view(np.uint64))
        assert np.array_equal(table.contains(ask), runs.contains(ask))
        assert len(table) == len(runs)
    assert runs.stats["merges"] >= 1 and runs.stats["compactions"] >= 1
    k_t, k_r = np.array(table.keys()), np.array(runs.keys())
    assert np.array_equal(k_t.view(np.uint64), k_r.view(np.uint64))
    assert np.array_equal(np.array(table.values()), np.array(runs.values()))
    for key in ("lookups", "hits", "inserts"):
        assert table.stats[key] == runs.stats[key]


# ----------------------------------------------------------------------
# Integrated: every lane's crossing decks, kernels on and off
# ----------------------------------------------------------------------


def _solve(make, names):
    """(Z, currents) of a fresh solve, Z copied at the last call of the
    lane's assembly entry `names` (the solve factors it in place)."""
    s = make()
    grabbed = []
    for name in names:
        real = getattr(s, name, None)
        if real is None:
            continue

        def spy(*a, _real=real, **kw):
            out = _real(*a, **kw)
            grabbed.append(np.array(out[0] if isinstance(out, tuple) else out))
            return out

        setattr(s, name, spy)
    _z, currents = s.compute_impedance()
    assert grabbed, "the assembly entry never ran"
    return grabbed[-1], np.asarray(currents)


# Each lane, and the entry its full Z leaves through.
LANES = {
    "bs2": (lambda d: BSplineSolver(**d), ("_compute_Z_operator",)),
    "razor": (lambda d: RazorSolver(**d, nec5_quadrature=True), ("_assemble_Z",)),
    "sin": (lambda d: SinusoidalSolver(**d), ("_assemble_Z",)),
    "sg": (
        lambda d: SinusoidalGalerkinSolver(**d),
        ("_compute_Z_operator", "_assemble_Z_ported"),
    ),
}
# The inverted-L's above side is two members, so razor's product plan has
# many groups and its end loops search `KeyIndex`.
DECKS = {
    "buried": lambda: hub_deck(n_radials=4),
    "invl": lambda: invl_deck(n_radials=4),
}


@pytest.mark.parametrize("deck", sorted(DECKS))
@pytest.mark.parametrize("lane", sorted(LANES))
def test_the_kernels_move_no_bit_of_z(lane, deck, monkeypatch):
    calls = {"rows": 0, "ints": 0, "index": 0, "table": 0, "groups": 0}
    acc = _accel.acc

    class _Counted:
        """`acc` with its factorize entries counted."""

        def __getattr__(self, name):
            return getattr(acc, name)

        def factorize_rows(self, cols):
            calls["rows"] += 1
            return acc.factorize_rows(cols)

        def factorize_ints(self, cols):
            calls["ints"] += 1
            return acc.factorize_ints(cols)

        def RowIndex(self, cols):
            calls["index"] += 1
            return acc.RowIndex(cols)

        def TripleTable(self, width):
            calls["table"] += 1
            return acc.TripleTable(width)

        def RowGroups(self):
            calls["groups"] += 1
            return acc.RowGroups()

    build, names = LANES[lane]
    monkeypatch.setattr(ni, "_FACTORIZE", False)
    z_ref, i_ref = _solve(lambda: build(DECKS[deck]()), names)
    monkeypatch.setattr(ni, "_FACTORIZE", True)
    monkeypatch.setattr(_accel, "acc", _Counted())
    z_new, i_new = _solve(lambda: build(DECKS[deck]()), names)
    assert calls["rows"] > 0, calls
    if lane in ("bs2", "sg"):
        assert calls["table"] > 0, calls
    if lane == "sin":
        assert calls["groups"] > 0, calls
    if lane == "razor" and deck == "invl":
        assert calls["ints"] > 0 and calls["index"] > 0, calls
    assert np.array_equal(z_new, z_ref)
    assert np.array_equal(i_new, i_ref)
