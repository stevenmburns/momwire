"""The point-observer key family — momwire#1223 U3a.

The point-matched crossing node tests the transmitted field at a POINT, so
it needs the gradients the Galerkin trunk moves off the observer by parts:
∂ρV_T, ∂zV_T, ∂ρW_T and ∂z′V_T, a key family of their own (#1223 decision;
the six shared keys do not widen). Same contour and same refusals as the
six; the ρ rows take J₁ where the six take J₀.

The gates, parity first (the #568 lesson):
  * each key is the finite difference of the six's own V or W, at ordinary,
    near-corner and corner triples over three soils;
  * the column route is the point route's twin, and converged in p;
  * the ρ keys are an exact 0 at ρ = 0 (a coaxial pair) and ∂ρW an exact 0
    at ε̃ = 1, where W_T ≡ 0;
  * the refusals are the six's.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _ground_refl
from momwire import _near_interface as ni

C0 = 299792458.0
F7 = 7e6
K7 = 2.0 * np.pi * F7 / C0
OM7 = 2.0 * np.pi * F7
EPS0 = 8.8541878128e-12
SOILS = {
    "eps1": 1.0 + 0j,
    "soilA": _ground_refl.eps_tilde((13.0, 0.005), OM7, EPS0),
    "highsigma": _ground_refl.eps_tilde((13.0, 5.0), OM7, EPS0),
}
TRIPLES = [
    (0.3, 0.2, -0.1),
    (1.5, 0.8, -0.4),
    (0.01, 0.005, -0.002),  # near-corner
    (0.002, 0.001, -0.0005),  # the corner class a crossing node's midpoints reach
]
IDS = [f"r{r:g}-z{z:g}-zp{zp:g}" for r, z, zp in TRIPLES]


def _six(eps, rho, z, zp):
    return ni.six_point(eps, K7, rho, z, zp, rtol=1e-12)


def _central(f, x, h):
    return (f(x + h) - f(x - h)) / (2.0 * h)


def test_the_family_is_its_own_and_the_six_do_not_widen():
    assert ni.POINT_KEYS == ("gRhoV", "gzV", "gRhoW", "gzpV")
    assert len(ni.POINT_KEY_RPOW) == ni.N_POINT_KEYS == 4
    assert ni.KEYS == ("U", "V", "W", "dzW", "dzpV", "dzpW")
    assert ni.N_KEYS == 6


@pytest.mark.slow
@pytest.mark.parametrize("soil", sorted(SOILS))
@pytest.mark.parametrize("triple", TRIPLES, ids=IDS)
def test_each_key_is_the_derivative_of_the_six(soil, triple):
    """Central differences of the six's V and W, on the vector scale
    (#895's convention). The bar is the difference's own truncation: the
    worst measured is 7.4e-8, on the high-σ far pair."""
    eps = SOILS[soil]
    rho, z, zp = triple
    got = ni.point_keys_point(eps, K7, rho, z, zp, rtol=1e-12)
    hr = 1e-4 * max(rho, 1e-3)
    hz = 1e-4 * max(z, 1e-3)
    hzp = 1e-4 * max(-zp, 1e-3)
    fd = np.array(
        [
            _central(lambda r: _six(eps, r, z, zp)[1], rho, hr),
            _central(lambda q: _six(eps, rho, q, zp)[1], z, hz),
            _central(lambda r: _six(eps, r, z, zp)[2], rho, hr),
            _central(lambda q: _six(eps, rho, z, q)[1], zp, hzp),
        ]
    )
    assert np.max(np.abs(fd - got)) <= 1e-6 * np.max(np.abs(got))


@pytest.mark.slow
@pytest.mark.parametrize("soil", sorted(SOILS))
@pytest.mark.parametrize("triple", TRIPLES, ids=IDS)
def test_the_column_route_is_the_point_routes_twin(soil, triple):
    """Against the point walk the bar is its own rtol (#895); measured
    ≤ 1.3e-14."""
    eps = SOILS[soil]
    rho, z, zp = triple
    ref = ni.point_keys_point(eps, K7, rho, z, zp, rtol=1e-12)
    col = ni.point_keys_columns(eps, K7, rho, [z], zp)[0]
    assert np.max(np.abs(col - ref)) <= 1e-10 * np.max(np.abs(ref))


def test_the_column_rule_is_converged_in_p():
    """The p / p+1 pair, which needs no reference (#895)."""
    eps = SOILS["soilA"]
    zs = np.array([0.001, 0.2, 0.8])
    a = ni.point_keys_columns(eps, K7, 0.3, zs, -0.1, p=0)
    b = ni.point_keys_columns(eps, K7, 0.3, zs, -0.1, p=1)
    assert np.max(np.abs(a - b)) <= 1e-12 * np.max(np.abs(b))


def test_a_column_of_many_is_each_point():
    """Several z and z′ in one ρ column, paired element-wise."""
    eps = SOILS["soilA"]
    zs = np.array([0.001, 0.05, 0.2, 0.8])
    zps = np.array([-0.0005, -0.1, -0.1, -0.4])
    col = ni.point_keys_columns(eps, K7, 0.3, zs, zps)
    for i, (z, zp) in enumerate(zip(zs, zps, strict=True)):
        ref = ni.point_keys_point(eps, K7, 0.3, z, zp, rtol=1e-12)
        assert np.max(np.abs(col[i] - ref)) <= 1e-10 * np.max(np.abs(ref))


@pytest.mark.parametrize("soil", sorted(SOILS))
def test_the_rho_keys_are_an_exact_zero_on_the_axis(soil):
    """ρ = 0 is a coaxial pair: ρ̂ is undefined and the J₁ spelling carries
    ρ explicitly, so the two ρ keys are 0 exactly, on both routes."""
    eps = SOILS[soil]
    pt = ni.point_keys_point(eps, K7, 0.0, 0.2, -0.1, rtol=1e-12)
    col = ni.point_keys_columns(eps, K7, 0.0, [0.2], -0.1)[0]
    for got in (pt, col):
        assert got[0] == 0 and got[2] == 0
        assert got[1] != 0 and got[3] != 0


def test_at_eps_one_the_w_gradient_vanishes():
    """ε̃ = 1: γ₊ = γ₋, W̃ = (γ₊ − γ₋)Ṽ = 0, so ∂ρW_T is exactly 0."""
    pt = ni.point_keys_point(1.0 + 0j, K7, 0.3, 0.2, -0.1, rtol=1e-12)
    col = ni.point_keys_columns(1.0 + 0j, K7, 0.3, [0.2], -0.1)[0]
    assert pt[2] == 0 and col[2] == 0


@pytest.mark.parametrize(
    ("rho", "z", "zp", "match"),
    [
        (0.3, -0.1, -0.1, "need z >= 0 >= zp"),
        (0.3, 0.1, 0.1, "need z >= 0 >= zp"),
        (0.0, 0.0, 0.0, "need R > 0"),
    ],
)
def test_the_refusals_are_the_sixs(rho, z, zp, match):
    eps = SOILS["soilA"]
    for call in (
        lambda: ni.point_keys_point(eps, K7, rho, z, zp),
        lambda: ni.point_keys_columns(eps, K7, rho, [z], zp),
        lambda: ni.six_point(eps, K7, rho, z, zp),
    ):
        with pytest.raises(ValueError, match=match):
            call()
