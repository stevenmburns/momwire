"""The array memo of `designed_tables` — momwire#1168 U1.

`TripleMemo` replaced the dict keyed on float tuples that `designed_tables`
carried across calls (momwire#688); the dict is kept, unchanged, as
`_designed_tables_reference`, and these rows gate the one against the other.

The memo decides HOW a stored row is found, never which rows are fresh nor
their order, and on the column route that is the whole contract: a member's
value depends on its column's membership (`six_columns`, 7.9e-16 between
groupings), so a memo that served a different fresh set would move bits even
with every stored value exact. The integrated rows therefore compare the
assembled Z of the three trunks, built by their real constructors, at
`np.array_equal` — and prove the array memo ran (lookups and hits counted
through it) and the dict route did not.

The unit rows pin what the integrated ones cannot reach on a real deck: a
hash collision (forced, since the audit's 5.4 M lookups met none), the
−0.0 / 0.0 fold, lookups spread over pending and main runs, and the dict
refusal.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _near_interface as ni
from momwire.bspline import BSplineSolver
from momwire.razor import RazorSolver
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

from test_crossing_serve_524 import crossing_deck, hub_deck

# ----------------------------------------------------------------------
# Unit rows
# ----------------------------------------------------------------------


def _vals(rows):
    """A value per row that names it, so a wrong lookup cannot pass."""
    # Elementwise, never a matmul: a BLAS product's blocking depends on the
    # batch size, and the same row must get the same value in any batch.
    r = np.asarray(rows, dtype=float)
    s = r[:, 0] + 3.0 * r[:, 1] + 7.0 * r[:, 2]
    return s[:, None] * np.arange(1.0, 7.0) * (1 + 1j)


def test_lookup_returns_the_stored_floats_and_misses_the_rest():
    m = ni.TripleMemo()
    stored = np.array([[0.3, 0.2, -0.1], [0.5, 0.2, -0.1], [0.3, 0.4, 0.0]])
    m.insert(stored, _vals(stored))
    ask = np.array([[0.5, 0.2, -0.1], [0.7, 0.2, -0.1], [0.3, 0.4, 0.0]])
    hit, block = m.lookup(ask)
    assert hit.tolist() == [True, False, True]
    assert np.array_equal(block[hit], _vals(ask[hit]))
    assert m.stats["lookups"] == 3 and m.stats["hits"] == 2


def test_negative_zero_is_the_same_key_as_zero():
    """The dict folded them (−0.0 == 0.0 and hash equal); so does the memo."""
    m = ni.TripleMemo()
    m.insert(np.array([[0.3, 0.0, -0.0]]), _vals([[0.3, 0.0, 0.0]]))
    hit, _ = m.lookup(np.array([[0.3, -0.0, 0.0]]))
    assert hit.tolist() == [True]
    assert m.keys() == [(0.3, 0.0, 0.0)]


def test_a_forced_hash_collision_is_resolved_on_the_full_row(monkeypatch):
    """Every row hashes alike, so every probe walks the equal-hash range and
    the row compare alone decides. A miss is a miss, a hit is its own row."""
    monkeypatch.setattr(
        ni, "_row_hash", lambda keys: np.zeros(keys.shape[0], dtype=np.uint64)
    )
    m = ni.TripleMemo()
    stored = np.array([[float(i), 0.1, -0.2] for i in range(1, 6)])
    m.insert(stored, _vals(stored))
    ask = np.array([[4.0, 0.1, -0.2], [9.0, 0.1, -0.2], [1.0, 0.1, -0.2]])
    hit, block = m.lookup(ask)
    assert hit.tolist() == [True, False, True]
    assert np.array_equal(block[hit], _vals(ask[hit]))
    # 4.0 passes three other keys, 9.0 all five, 1.0 none.
    assert m.stats["collisions"] == 3 + 5 + 0


def test_rows_are_found_across_pending_runs_merges_and_compactions(monkeypatch):
    """Small thresholds force every store transition; each inserted row stays
    findable through all of them, and iteration keeps insertion order."""
    monkeypatch.setattr(ni, "_MEMO_MERGE_MIN_ROWS", 12)
    monkeypatch.setattr(ni, "_MEMO_MERGE_FRACTION", 2)
    monkeypatch.setattr(ni, "_MEMO_MAX_PENDING_RUNS", 3)
    rng = np.random.default_rng(1168)
    m = ni.TripleMemo()
    inserted = []
    for size in (1, 2, 1, 3, 1, 1, 5, 2, 1, 1, 20, 1, 2, 3, 1, 1):
        rows = rng.standard_normal((size, 3))
        m.insert(rows, _vals(rows))
        inserted.append(rows)
        every = np.concatenate(inserted)
        hit, block = m.lookup(every)
        assert hit.all()
        assert np.array_equal(block, _vals(every))
    assert m.stats["merges"] >= 2 and m.stats["compactions"] >= 1
    assert m.keys() == [tuple(r) for r in np.concatenate(inserted).tolist()]
    assert len(m) == sum(r.shape[0] for r in inserted)


def test_a_dict_memo_is_refused_by_name():
    with pytest.raises(TypeError, match="_designed_tables_reference"):
        ni.designed_tables(4.0 - 0.5j, 0.9, 0.3, 0.2, -0.1, memo={})


# ----------------------------------------------------------------------
# Integrated rows: the three trunks' Z against the dict reference
# ----------------------------------------------------------------------


def _hub(m, n_radials=16):
    """hub_deck(16) refined x`m` as the #1168 audit ran it: radials and the
    monopole times m, the two-edge rise held."""
    d = hub_deck(n_radials=n_radials)
    npe = d["n_per_edge_per_wire"]
    d["n_per_edge_per_wire"] = [[n * m for n in e] for e in npe[:n_radials]] + [
        npe[n_radials],
        [npe[n_radials + 1][0] * m],
    ]
    return d


def _razor(deck):
    return RazorSolver(
        **{k: v for k, v in deck.items() if k != "junctions"}, nec5_quadrature=True
    )


def _solve(make):
    """(Z, currents) of a fresh solver from `make()`. Razor keeps its Z; the
    Galerkin trunks' Z is taken off their own assembly entry."""
    s = make()
    grabbed = []
    for name in ("_compute_Z_operator", "_assemble_Z_ported"):
        real = getattr(s, name, None)
        if real is None:
            continue

        def spy(*a, _real=real, **kw):
            out = _real(*a, **kw)
            grabbed.append(out[0] if isinstance(out, tuple) else out)
            return out

        setattr(s, name, spy)
    _z, currents = s.compute_impedance()
    Z = s.z if isinstance(s, RazorSolver) else grabbed[-1]
    return np.array(Z, copy=True), np.asarray(currents)


class _Routes:
    """Counts on both routes of one solve: the reference dict route, and
    every `TripleMemo` the production fill creates."""

    def __init__(self, monkeypatch):
        self.memos = []
        self.reference_calls = 0
        self._mp = monkeypatch
        real_init = ni.TripleMemo.__init__

        def init(memo):
            real_init(memo)
            self.memos.append(memo)

        monkeypatch.setattr(ni.TripleMemo, "__init__", init)
        real_ref = ni._designed_tables_reference

        def counted(*a, **kw):
            self.reference_calls += 1
            return real_ref(*a, **kw)

        monkeypatch.setattr(ni, "_designed_tables_reference", counted)

    def stat(self, key):
        return sum(m.stats[key] for m in self.memos)

    def use_reference(self):
        """Serve every `designed_tables` call through the dict reference, one
        dict per TripleMemo the fill hands in — the pre-#1168 route."""
        array_route = ni.designed_tables
        dicts = {}

        def reference(
            eps_t, k2, rho, z, zp, rtol=1e-10, lam_mult=ni._LAM_MULT, memo=None
        ):
            d = None
            if memo is not None:
                # Keyed on the object (held alive by `self.memos`), so an id
                # is never reused inside one solve.
                d = dicts.setdefault(id(memo), {})
            return ni._designed_tables_reference(
                eps_t, k2, rho, z, zp, rtol=rtol, lam_mult=lam_mult, memo=d
            )

        self._mp.setattr(ni, "designed_tables", reference)
        return lambda: self._mp.setattr(ni, "designed_tables", array_route)


CASES = {
    "razor-hub16-x2": lambda: _razor(_hub(2)),
    "razor-hub16-x4": lambda: _razor(_hub(4)),
    "razor-crossing1": lambda: _razor(crossing_deck(1)),
    "bspline-hub16-x2": lambda: BSplineSolver(**_hub(2)),
    "sg-hub16-x2": lambda: SinusoidalGalerkinSolver(**_hub(2)),
}


@pytest.mark.slow
@pytest.mark.parametrize("case", sorted(CASES))
def test_the_array_memo_is_bit_identical_to_the_dict_reference(case, monkeypatch):
    """Z and the solved currents to the bit, the dict route first, then the
    array route with the dict route counted at zero calls on it.

    Z is compared and not only the currents because the currents alone miss
    what this row exists to catch. Negative control, run 2026-09-23 with the
    memo's lookup patched to add one ulp to the real part of a hit's six
    values: on bspline x2 it moved 975 Z entries and ZERO currents. On
    razor x2 one such row moved 1 Z entry and all 351 currents; on
    crossing_deck(1) one row rounded away entirely, and one ulp on every hit
    moved 294 of 841 entries. A memo that misses where it should hit (every
    row fresh, so the column grouping changes) moved 208 / 6,873 / 11,073
    entries on the three."""
    routes = _Routes(monkeypatch)
    restore = routes.use_reference()
    z_ref, i_ref = _solve(CASES[case])
    assert routes.reference_calls > 0, "the reference route never ran"
    restore()

    calls_before = routes.reference_calls
    routes.memos.clear()
    z_new, i_new = _solve(CASES[case])
    assert routes.reference_calls == calls_before, "the dict route ran"
    assert routes.memos, "the fill built no TripleMemo"
    assert routes.stat("lookups") > 0 and routes.stat("hits") > 0

    assert z_new.shape == z_ref.shape
    assert np.array_equal(z_new, z_ref)
    assert np.array_equal(i_new, i_ref)
