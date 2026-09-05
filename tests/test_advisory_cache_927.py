"""The surface advisory survives the basis cache (momwire#927).

`warn_surface_height` is reached from `_wire_endpoint_status`, which
`_build_basis_polynomials` runs only PAST its `_BASIS_POLY_CACHE` hit return.
That cache is module-level and deliberately outlives the solver instance — its
own comment says the engine wrapper recreates the solver per `impedance()`
call — so a second solve of the same deck skipped the check entirely and said
nothing. Measured before the fix: run 1 warned, runs 2 and 3 did not, and the
advisory came back only after 40 other decks evicted the entry.

That is the worst shape for an advisory: intermittent rather than absent. On a
long-running server the first user to solve a deck saw it and nobody after
did, so antennaknobs#1144's channel would have been silently unreliable.

The fix stores the `(h_min, a_at, count)` summary in the cache VALUE and
re-emits from it on a hit. The returned 5-tuple is unchanged — callers unpack
it, and the cache is read and written in one method, so the summary rides
along without touching that contract.

Gates:

- G-927-1  the same deck solved repeatedly, a new solver each time, warns
           EVERY time — and the second solve is proved to have taken the
           cache-hit path, without which the gate would pass on a cache miss
           and prove nothing.
- G-927-2  once per solve, not once per `_build_basis_polynomials` call.
- G-927-3  a deck with nothing near the interface stays silent however many
           times it is solved — the fix must not turn a cache into a warning.
- G-927-4  the cached basis itself is unchanged: same numbers, hit or miss.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import momwire.bspline as _bs
from momwire._surface_height import SurfaceRadialHeight
from momwire.bspline import BSplineSolver

WL = 299.792458 / 7.1
A_WIRE = 0.0005
H_LOW = 0.00105  # h/a = 2.1 — inside the advisory band
H_CLEAR = 1.0  # well clear of the interface


def _deck(h):
    return dict(
        wires=[
            np.array([(0.0, 0.0, h), (5.0, 0.0, h)]),
            np.array([(0.0, 0.0, h), (0.0, 0.0, 10.0)]),
        ],
        n_per_edge_per_wire=[[20], [20]],
        junctions=[[(0, "start"), (1, "start")]],
        feeds=[(1, 5.0, 1 + 0j)],
        wavelength=WL,
        wire_radius=A_WIRE,
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
    )


@pytest.fixture(autouse=True)
def _cold_caches():
    """Every gate here is about cache STATE, so none may inherit another's."""
    _bs._GEOMETRY_CACHE.clear()
    _bs._BASIS_POLY_CACHE.clear()
    yield
    _bs._GEOMETRY_CACHE.clear()
    _bs._BASIS_POLY_CACHE.clear()


def _solve_recording(h):
    """One solve on a FRESH solver — the shape the engine wrapper makes."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        z, _ = BSplineSolver(**_deck(h)).compute_impedance()
    return z, [w for w in rec if issubclass(w.category, SurfaceRadialHeight)]


# --- G-927-1 --------------------------------------------------------------


def test_g927_1_a_repeat_solve_of_one_deck_still_warns():
    counts = [len(_solve_recording(H_LOW)[1]) for _ in range(4)]
    assert all(c >= 1 for c in counts), counts


def test_g927_1_the_second_solve_really_did_hit_the_cache():
    """Without this the gate above is worthless: a solve that MISSED the
    cache would warn through the cold path and look identical. The cache
    being non-empty and unchanged in size across the second solve is what
    makes "it warned from the hit path" a measurement rather than a hope.
    """
    _solve_recording(H_LOW)
    assert _bs._BASIS_POLY_CACHE, "nothing was cached; the gate proves nothing"
    size_after_first = len(_bs._BASIS_POLY_CACHE)
    keys = set(_bs._BASIS_POLY_CACHE)

    _, hits = _solve_recording(H_LOW)
    assert len(_bs._BASIS_POLY_CACHE) == size_after_first
    assert set(_bs._BASIS_POLY_CACHE) == keys, "the second solve keyed differently"
    assert hits, "cache hit, and the advisory was swallowed — #927 is back"


def test_g927_1_the_advisory_text_survives_the_round_trip():
    """Re-emitted from the stored summary, so the sentence a second user sees
    must be the sentence the first one saw — not a truncated stand-in."""
    _, first = _solve_recording(H_LOW)
    _, second = _solve_recording(H_LOW)
    assert str(first[0].message) == str(second[0].message)
    assert "h/a" in str(second[0].message)


# --- G-927-2 --------------------------------------------------------------


def test_g927_2_one_advisory_per_solve_not_one_per_call():
    """`_build_basis_polynomials` runs several times in a solve. Re-emitting
    on each hit would turn one advisory into a handful."""
    for _ in range(3):
        _, hits = _solve_recording(H_LOW)
        assert len(hits) == 1, [str(h.message)[:60] for h in hits]


# --- G-927-3 --------------------------------------------------------------


def test_g927_3_a_deck_clear_of_the_interface_never_warns():
    """The fix must not convert a cache into a warning. Repeated, because a
    stale summary would show up on the second solve, not the first."""
    for _ in range(3):
        _, hits = _solve_recording(H_CLEAR)
        assert hits == []


def test_g927_3_a_clear_deck_after_a_low_one_stays_clear():
    """The two share a module cache. A summary leaking between entries would
    put the low deck's advisory on the clear deck's answer."""
    _solve_recording(H_LOW)
    _, hits = _solve_recording(H_CLEAR)
    assert hits == [], [str(h.message)[:80] for h in hits]


# --- G-927-4 --------------------------------------------------------------


def test_g927_4_the_cached_basis_is_unchanged_by_carrying_the_summary():
    """The cache VALUE changed shape; the RETURN did not. Callers unpack a
    5-tuple, and a hit must still give exactly what a miss gave."""
    z_miss, _ = _solve_recording(H_LOW)
    z_hit, _ = _solve_recording(H_LOW)
    assert z_hit == z_miss

    key = next(iter(_bs._BASIS_POLY_CACHE))
    entry = _bs._BASIS_POLY_CACHE[key]
    assert len(entry) == 2, "the cache value is (result, advisory)"
    result, advisory = entry
    assert len(result) == 5, "the returned basis tuple must stay a 5-tuple"
    assert advisory is not None and len(advisory) == 3
