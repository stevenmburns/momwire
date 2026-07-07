"""Sommerfeld-integral engine tests (docs/sommerfeld-ground-plan.md Phase 1).

Layers, from machinery to physics:
  1. contour + branch machinery against a closed form (Sommerfeld
     identity: the module's own contours must reproduce e^{-jkR}/R);
  2. integrand internal identity (Laplacian of J0);
  3. cross-form agreement (Bessel vs Hankel contours at the same point —
     the strongest contour-invariance check, the paths share nothing);
  4. exact limits (free-space eps=1 -> 0, PEC scaling, R1 -> 0 limits);
  5. the NEC-2 theory manual's published surface extrema (figs 7-11).

Working units: wavelengths (k2 = 2*pi) unless a test says otherwise.
"""

import numpy as np
import pytest

from momwire import _sommerfeld as som

from oracle_sommerfeld_figs import FIG_FREQ_MHZ, FIG_ORACLES

K2 = 2.0 * np.pi
EPS_GROUND = 10.0 - 3.6j  # generic lossy ground for machinery tests
EPS0 = 8.8541878128e-12


# ---------------------------------------------------------------------------
# 1. Sommerfeld identity: contours reproduce the closed-form Green's function
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rho,h",
    [(0.0, 0.1), (0.02, 0.1), (0.1, 0.5), (0.05, 2.0), (1e-5, 1e-4)],
)
def test_sommerfeld_identity_bessel_form(rho, h):
    num, exact = som.greens_free_space_check(K2, rho, h, "J")
    assert abs(num - exact) / abs(exact) < 1e-8


@pytest.mark.parametrize(
    "rho,h",
    [(0.1, 0.1), (0.5, 0.1), (1.0, 0.02), (0.3, 0.0), (0.05, 0.08), (1e-4, 0.0)],
)
def test_sommerfeld_identity_hankel_form(rho, h):
    num, exact = som.greens_free_space_check(K2, rho, h, "H")
    assert abs(num - exact) / abs(exact) < 1e-8


# ---------------------------------------------------------------------------
# 2. Integrand internal identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form", ["J", "H"])
def test_integrand_laplacian_identity(form):
    """d2/drho2 + (1/rho) d/drho acting on J0(lam*rho) is -lam^2*J0, so
    integrands 0 + 3 + lam^2 * 4 must vanish pointwise."""
    k1 = K2 * np.sqrt(EPS_GROUND)
    lam = np.array([0.3 + 0.1j, 5.0 + 1.0j, 20.0 - 4.0j, 100.0 - 1.0j])
    six = som._integrand_six(lam, 0.3, 0.2, np.conj(k1), K2, form)
    resid = six[0] + six[3] + lam**2 * six[4]
    assert np.max(np.abs(resid)) < 1e-13 * np.max(np.abs(six[0]))


# ---------------------------------------------------------------------------
# 3. Cross-form agreement at identical points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("r1", [0.02, 0.1, 0.5, 1.0])
def test_bessel_vs_hankel_same_point(r1):
    """Both forms converge in the band h/2 < rho < 2h; they share no
    contour geometry, so agreement validates both."""
    theta = np.arctan2(1.0, 1.0)  # rho == h
    rho = r1 * np.cos(theta)
    h = r1 * np.sin(theta)
    sj = som._six_integrals(EPS_GROUND, K2, rho, h, form="J")
    sh = som._six_integrals(EPS_GROUND, K2, rho, h, form="H")
    assert np.max(np.abs(sj - sh)) < 1e-7 * np.max(np.abs(sj))


# ---------------------------------------------------------------------------
# 4. Exact limits
# ---------------------------------------------------------------------------


def test_free_space_limit_is_zero():
    """eps_t = 1 kills D1 and D2 identically: no ground, no remainder."""
    out = som.iv_surfaces_direct(1.0 + 0.0j, K2, [0.05, 0.3], [0.4, 1.2])
    ref = som.iv_surfaces_direct(EPS_GROUND, K2, [0.05, 0.3], [0.4, 1.2])
    scale = max(np.max(np.abs(v)) for v in ref.values())
    assert max(np.max(np.abs(v)) for v in out.values()) < 1e-12 * scale


def test_pec_limit_scales_away():
    """|I| ~ O(1/sqrt(eps)) as eps -> infinity at fixed R1 > 0 (D1's
    leading term is 2/gamma_1 ~ 2/k1); check the trend and smallness."""
    ref = som.iv_surfaces_direct(EPS_GROUND, K2, [0.2], [0.7])
    scale = max(np.max(np.abs(v)) for v in ref.values())
    prev = None
    for mag in (1e4, 1e6, 1e8):
        out = som.iv_surfaces_direct(mag * (1 - 1e-3j), K2, [0.2], [0.7])
        size = max(np.max(np.abs(v)) for v in out.values()) / scale
        if prev is not None:
            assert size < 0.15 * prev  # ~1/sqrt(1e2) per decade-squared step
        prev = size
    assert size < 1e-3


def test_r1_zero_limits_continuity():
    """Direct evaluation at R1 = 1e-4 wavelengths lands on the analytic
    eqs 169-172 limits (relative to each surface's limit scale)."""
    th = np.array([0.05, 0.4, 1.0, np.pi / 2])
    lim = som._limits_r1_zero(EPS_GROUND, K2, th, K2 * som._C_LIGHT, som._MU0)
    near = som.iv_surfaces_direct(EPS_GROUND, K2, np.full_like(th, 1e-4), th)
    for kk in ("IrhoV", "IzV", "IrhoH", "IphiH"):
        dev = np.max(np.abs(near[kk] - lim[kk])) / np.max(np.abs(lim[kk]))
        assert dev < 5e-3, (kk, dev)


def test_iv_surfaces_r1_zero_uses_limits():
    """R1 = 0 entries route to the analytic limits (no integration)."""
    th = np.array([0.3, np.pi / 2])
    out = som.iv_surfaces_direct(EPS_GROUND, K2, np.zeros_like(th), th)
    lim = som._limits_r1_zero(EPS_GROUND, K2, th, K2 * som._C_LIGHT, som._MU0)
    for kk in ("IrhoV", "IzV", "IrhoH", "IphiH"):
        np.testing.assert_allclose(out[kk], lim[kk], rtol=1e-12)


def test_near_free_space_stability():
    """eps_t = 1 + delta exercises the near-singularity at k2 the manual
    warns about (virtual D poles when k1 -> k2); values must be small,
    finite, and roughly proportional to delta."""
    outs = {}
    for delta in (1e-2, 1e-3):
        out = som.iv_surfaces_direct(1.0 + delta - 1e-4j, K2, [0.3], [0.6])
        m = max(np.max(np.abs(v)) for v in out.values())
        assert np.isfinite(m)
        outs[delta] = m
    ratio = outs[1e-2] / outs[1e-3]
    assert 3.0 < ratio < 30.0  # ~linear in delta


# ---------------------------------------------------------------------------
# 5. Manual figure extrema (figs 7-11)
# ---------------------------------------------------------------------------

# Unit-moment conversion: the manual plots Idl = 1 A*wavelength at 10 MHz
# (pinned in oracle_sommerfeld_figs.py) -> multiply our A*m surfaces by
# lambda(10 MHz) in meters.
_LAMBDA_M = som._C_LIGHT / (FIG_FREQ_MHZ * 1e6)


@pytest.fixture(scope="module")
def fig_meshes():
    """One surface mesh per (eps_r, sigma) ground in the oracle table.
    31x13 samples R1 in [1e-3, 1] wavelengths x theta in [0, 90] deg —
    comparable to the manual's plot mesh."""
    r1, th = np.meshgrid(
        np.linspace(1e-3, 1.0, 31),
        np.radians(np.linspace(0.0, 90.0, 13)),
        indexing="ij",
    )
    omega = 2.0 * np.pi * FIG_FREQ_MHZ * 1e6
    meshes = {}
    for _comp, er, sig in FIG_ORACLES:
        if (er, sig) in meshes:
            continue
        eps_t = er - 1j * sig / (omega * EPS0)
        meshes[(er, sig)] = som.iv_surfaces_direct(
            eps_t, K2, r1, th, rtol=1e-7, omega=omega
        )
    return meshes


@pytest.mark.parametrize("comp,er,sig", sorted(FIG_ORACLES))
def test_manual_figure_extrema(fig_meshes, comp, er, sig):
    """Computed surface extrema against the NEC-2 manual's printed
    Max/Min (figs 7-11). Tolerance is dominated by mesh sampling of the
    small-theta ridge: 6% of the surface's own extremum scale for the
    lossy eps_r=4 figures, 15% for the oscillatory lossless fig 11."""
    rng = FIG_ORACLES[(comp, er, sig)]
    v = fig_meshes[(er, sig)][comp] * _LAMBDA_M
    tol = 0.15 if sig == 0.0 else 0.06
    for part, arr in (("re", v.real), ("im", v.imag)):
        lo, hi = rng[part]
        scale = max(abs(lo), abs(hi))
        assert abs(arr.min() - lo) < tol * scale, (part, "min", arr.min(), lo)
        assert abs(arr.max() - hi) < tol * scale, (part, "max", arr.max(), hi)
