"""The column twin's head + mid cache (`mw899::ColumnCache`).

A column's rule is head + mid + tail; the head and mid depend on rho and the
extents alone, so a fill-held cache serves them to every later column of the
same rho. It decides WHEN those factors are computed, never what they are,
so the gates are uint64 equality against the uncached twin:

* kernel level: a sequence of calls revisiting rho with different members
  (different s_min, so different tails) -- hits counted, every value equal;
  a budget too small for one entry (never stores, still equal); a budget that
  forces evictions (equal, evictions counted); a context change (k_m) that
  must empty the cache rather than serve another medium's factors; a column
  whose s_min triggers the far-pair kill cap (different extents, so a
  different key, never a false hit);
* end to end: a buried crossing solve (bs2) with and without the cache is
  the same Z_in and currents bit for bit, and the fill's cache was hit.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from momwire import _near_interface as NI

nia = NI._nia

pytestmark = pytest.mark.skipif(
    nia is None or not hasattr(nia, "ColumnCache"),
    reason="needs the near-interface C++ twin with the column cache",
)

K_P = 2 * np.pi / (299792458.0 / 7e6)
K_M = NI.k_medium(complex(13.0, -12.84), K_P)


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


def _call(rho, members, cache=None, k_m=K_M):
    """One twin call: columns at `rho`, each with its (z, zp) members."""
    rho = np.asarray(rho, float)
    sizes = [len(m) for m in members]
    offsets = np.zeros(len(rho) + 1, dtype=np.intp)
    offsets[1:] = np.cumsum(sizes)
    z = np.ascontiguousarray([zz for m in members for zz, _ in m], dtype=float)
    zp = np.ascontiguousarray([pp for m in members for _, pp in m], dtype=float)
    return nia.near_interface_six_columns(
        K_P,
        k_m,
        rho,
        offsets,
        z,
        zp,
        float(NI._LAM_MULT),
        int(NI._COLUMN_P),
        float(NI._DETOUR),
        4,
        NI._GX,
        NI._GW,
        cache=cache,
    )


def _calls(seed):
    """Five calls over a small pool of rho, each column a few fresh
    members; one column per call deep enough (s > 60 / lam_top) for the
    kill cap to move its extents."""
    rng = np.random.default_rng(seed)
    pool = rng.uniform(0.05, 6.0, 7)
    out = []
    for c in range(5):
        rho = rng.choice(pool, size=4, replace=False)
        members = [
            [
                (rng.uniform(0.0, 0.5), -rng.uniform(0.01, 3.0))
                for _ in range(rng.integers(1, 4))
            ]
            for _ in rho
        ]
        members[0] = [(0.2, -14.0 - c)]  # kill-capped extents
        out.append((rho, members))
    return out


def test_a_cached_sequence_is_the_uncached_one_to_the_bit():
    cache = nia.ColumnCache(1 << 30)
    for rho, members in _calls(0):
        want = _call(rho, members)
        got = _call(rho, members, cache)
        assert np.array_equal(_bits(got), _bits(want))
    assert cache.hits > 0 and cache.misses > 0
    assert cache.entries > 0 and cache.evictions == 0


@pytest.mark.parametrize("budget", [1, 200_000])
def test_small_budgets_store_less_and_change_nothing(budget):
    cache = nia.ColumnCache(budget)
    for rho, members in _calls(1):
        assert np.array_equal(
            _bits(_call(rho, members, cache)), _bits(_call(rho, members))
        )
    assert cache.bytes <= budget
    if budget == 1:
        assert cache.entries == 0 and cache.hits == 0
    else:
        assert cache.evictions > 0


def test_a_new_medium_empties_the_cache():
    cache = nia.ColumnCache(1 << 30)
    rho, members = _calls(2)[0]
    _call(rho, members, cache)
    other = K_M * (1.0 + 1e-3)
    want = _call(rho, members, k_m=other)
    hits = cache.hits
    got = _call(rho, members, cache, k_m=other)
    assert cache.hits == hits  # nothing served across the change
    assert np.array_equal(_bits(got), _bits(want))


def test_the_gate_sees_one_ulp():
    rho, members = _calls(3)[0]
    want = _call(rho, members)
    got = _call(np.nextafter(rho, np.inf), members)
    assert (_bits(got) != _bits(want)).mean() > 0.25


def _hub(x=1):
    from test_crossing_serve_524 import hub_deck

    d = hub_deck(n_radials=2)
    d["n_per_edge_per_wire"] = [[n * x for n in e] for e in d["n_per_edge_per_wire"]]
    return d


def test_a_crossing_solve_is_the_uncached_one_to_the_bit(monkeypatch):
    from momwire.bspline import BSplineSolver

    caches = []
    real = NI._new_column_cache

    def spy():
        c = real()
        caches.append(c)
        return c

    def solve():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z, cur = BSplineSolver(**_hub()).compute_impedance()
        return np.atleast_1d(np.asarray(z)), np.asarray(cur)

    monkeypatch.setattr(NI, "_new_column_cache", spy)
    z, cur = solve()
    assert caches and all(c is not None for c in caches)
    assert sum(c.hits for c in caches) > 0
    monkeypatch.setattr(NI, "_new_column_cache", lambda: None)
    z_ref, cur_ref = solve()
    assert np.array_equal(_bits(z), _bits(z_ref))
    assert np.array_equal(_bits(cur), _bits(cur_ref))
