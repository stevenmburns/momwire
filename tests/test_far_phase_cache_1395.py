"""The far-field phase-factor cache (momwire#1395).

``exp(j k r_hat . r_n)`` depends only on the direction grid, the element
centres and k, so a multi-run deck's runs share it. A hit must return what a
fresh build gives, bit for bit, and a factor over the size cap must not be
kept.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _far_readout as fr


@pytest.fixture(autouse=True)
def _empty_cache():
    fr._PHASE_CACHE.clear()
    yield
    fr._PHASE_CACHE.clear()


def _case(seed=0, n=40):
    rng = np.random.default_rng(seed)
    mid = rng.normal(size=(n, 3)) + np.array([0.0, 0.0, 3.0])
    moment = rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3))
    theta = np.deg2rad(np.arange(0.0, 91.0, 5.0))
    phi = np.deg2rad(np.arange(0.0, 360.0, 5.0))
    return mid, moment, 2 * np.pi / 7.5, theta, phi


@pytest.mark.parametrize(
    "ground", [fr.Ground("free"), fr.Ground("pec"), fr.Ground("refl", 13.0, 0.005)]
)
def test_a_hit_is_bit_identical_to_a_fresh_build(ground):
    mid, moment, k, theta, phi = _case()
    first = fr._far_moments(mid, moment, k, theta, phi, ground, 0.0, 14e6)
    assert len(fr._PHASE_CACHE) >= 1
    # A second run: same structure, other currents, served from the cache.
    other = moment[::-1].copy()
    cached = fr._far_moments(mid, other, k, theta, phi, ground, 0.0, 14e6)
    fr._PHASE_CACHE.clear()
    fresh = fr._far_moments(mid, other, k, theta, phi, ground, 0.0, 14e6)
    for a, b in zip(cached, fresh, strict=True):
        assert np.array_equal(a, b)
    assert not np.array_equal(first[0], cached[0])


def test_other_geometry_or_k_misses():
    mid, moment, k, theta, phi = _case()
    fr._far_moments(mid, moment, k, theta, phi, fr.Ground("free"), 0.0, 14e6)
    fr._far_moments(mid + 0.1, moment, k, theta, phi, fr.Ground("free"), 0.0, 14e6)
    fr._far_moments(mid, moment, 1.01 * k, theta, phi, fr.Ground("free"), 0.0, 14e6)
    assert len(fr._PHASE_CACHE) == 3


def test_a_factor_over_the_cap_is_not_kept(monkeypatch):
    monkeypatch.setattr(fr, "_PHASE_CACHE_MAX_ELEMENTS", 10)
    mid, moment, k, theta, phi = _case()
    fr._far_moments(mid, moment, k, theta, phi, fr.Ground("free"), 0.0, 14e6)
    assert len(fr._PHASE_CACHE) == 0


def test_the_cache_holds_a_few_entries_at_most():
    mid, moment, k, theta, phi = _case()
    for i in range(fr._PHASE_CACHE_ENTRIES + 3):
        fr._far_moments(
            mid, moment, k * (1 + 0.01 * i), theta, phi, fr.Ground("free"), 0.0, 14e6
        )
    assert len(fr._PHASE_CACHE) == fr._PHASE_CACHE_ENTRIES
