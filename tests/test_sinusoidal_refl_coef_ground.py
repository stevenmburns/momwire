"""Reflection-coefficient finite ground (`SinusoidalSolver(ground_eps=...)`).

Phase 6 of docs/refl-coef-ground-plan.md: the sinusoidal solver is
field-based (Eqs 76-79 give the TOTAL E-field per source shape), so NEC's
field dyad applies exactly and there is no `ground_phi_mode` knob. Golden
references are the same PyNEC gn 0 captures the bspline tests use
(tests/golden_refl_coef_ground.py); geometry kwargs are constructor-
compatible across solvers (tests/fixtures_refl_coef_geoms.py).

Thresholds carry the measured Phase 6 residuals plus ~3x slack: they are
regression guards against the physics drifting, not re-derivations of the
acceptance analysis. Measured at capture (2026-07-06), |Z - gn0| over the
0.1-0.5 lambda window x all three ground constants:
  dipole      0.107 .. 0.114 ohm  (max at h=0.2, eps=13)
  inverted_l  0.066 .. 0.067 ohm
i.e. the residual is essentially flat at the sinusoidal-vs-NEC cross-solver
floor — as the plan predicted for a shared basis, well under bspline's
2.45 ohm window max.
"""

import numpy as np
import pytest

from momwire import SinusoidalSolver

from fixtures_refl_coef_geoms import GEOMS
from golden_refl_coef_ground import GOLDEN


def _solve(name, frac, ground_eps=None, **extra):
    kw = dict(GEOMS[(name, frac)])
    z, _ = SinusoidalSolver(
        **kw, ground_z=0.0, ground_eps=ground_eps, **extra
    ).compute_impedance()
    return z


def test_ground_eps_requires_ground_z():
    kw = dict(GEOMS[("dipole", 0.2)])
    with pytest.raises(ValueError, match="ground_eps requires ground_z"):
        SinusoidalSolver(**kw, ground_eps=(10.0, 0.002))


def test_free_space_default_kwarg_is_inert():
    """ground_eps=None (the default) must not perturb the free-space solve:
    passing it explicitly gives bit-identical Z."""
    kw = dict(GEOMS[("dipole", 0.2)])
    z_a, _ = SinusoidalSolver(**kw).compute_impedance()
    z_b, _ = SinusoidalSolver(**kw, ground_eps=None).compute_impedance()
    assert z_a == z_b


def test_pec_ground_path_deterministic_and_inert_kwarg():
    """PEC-ground regression guard: with ground_z set and ground_eps=None
    the solve must take the untouched PEC image path — two identically
    constructed solvers (one with the explicit ground_eps=None kwarg) give
    bit-identical Z."""
    kw = dict(GEOMS[("dipole", 0.2)])
    z_a, _ = SinusoidalSolver(**kw, ground_z=0.0).compute_impedance()
    z_b, _ = SinusoidalSolver(**kw, ground_z=0.0, ground_eps=None).compute_impedance()
    assert z_a == z_b


@pytest.mark.parametrize("name", ["dipole", "inverted_l"])
def test_pec_limit_reproduces_pec_image(name):
    """eps -> inf must collapse the Fresnel dyad (rho_v -> 1, rho_h -> -1)
    to the PEC image tensor. Measured collapse at ground_eps=1e16: 1.1e-8
    relative (dipole), 4.8e-9 (inverted_l); guard at 1e-5 relative."""
    kw = dict(GEOMS[(name, 0.2)])
    z_pec, _ = SinusoidalSolver(**kw, ground_z=0.0).compute_impedance()
    z_lim, _ = SinusoidalSolver(
        **kw, ground_z=0.0, ground_eps=1e16 + 0j
    ).compute_impedance()
    assert abs(z_lim - z_pec) < 1e-5 * abs(z_pec)


def test_complex_eps_matches_tuple_spec():
    """(eps_r, sigma) folds to eps_t = eps_r - j*sigma/(omega*eps0); passing
    that eps_t directly must give the identical solve (measured: exactly
    equal — same eps_tilde() call either way)."""
    kw = dict(GEOMS[("dipole", 0.2)])
    sim = SinusoidalSolver(**kw, ground_z=0.0, ground_eps=(10.0, 0.002))
    eps_t = complex(10.0, -0.002 / (sim.omega * sim.eps))
    z_tuple, _ = sim.compute_impedance()
    z_complex, _ = SinusoidalSolver(
        **kw, ground_z=0.0, ground_eps=eps_t
    ).compute_impedance()
    np.testing.assert_allclose(z_complex, z_tuple, rtol=0, atol=1e-9)


@pytest.mark.parametrize("frac", [0.1, 0.2, 0.35, 0.5])
@pytest.mark.parametrize("ground", [(10.0, 0.002), (13.0, 0.005), (3.0, 0.001)])
def test_dipole_matches_nec_gn0_in_window(frac, ground):
    """Acceptance window: |Z - gn0| within the measured residuals (max
    0.1143 ohm at capture, at h=0.2 eps=13; guard at ~3x = 0.35), and
    strictly better than the PEC-image solve whose error this feature
    removes (PEC residuals are 7.5-41.6 ohm across the window)."""
    eps_r, sigma = ground
    gn0 = GOLDEN[("dipole", frac, eps_r, sigma)]["finite-fast"]
    z = _solve("dipole", frac, ground_eps=(eps_r, sigma))
    z_pec = _solve("dipole", frac)
    assert abs(z - gn0) < 0.35
    assert abs(z - gn0) < abs(z_pec - gn0)


@pytest.mark.parametrize("frac", [0.1, 0.2, 0.35, 0.5])
@pytest.mark.parametrize("ground", [(10.0, 0.002), (13.0, 0.005), (3.0, 0.001)])
def test_inverted_l_matches_nec_gn0_in_window(frac, ground):
    """Junction/KCL geometry through the weighted image path. Measured max
    0.0670 ohm at capture (flat across the window); guard at ~3x = 0.2.
    Strictly-better-than-PEC held on every inverted_l window case at
    capture (tightest margin: 0.067 vs 0.414 ohm at h=0.5 eps=13), so it
    is asserted here too — no cross-solver-floor tie materialized."""
    eps_r, sigma = ground
    gn0 = GOLDEN[("inverted_l", frac, eps_r, sigma)]["finite-fast"]
    z = _solve("inverted_l", frac, ground_eps=(eps_r, sigma))
    z_pec = _solve("inverted_l", frac)
    assert abs(z - gn0) < 0.2
    assert abs(z - gn0) < abs(z_pec - gn0)


@pytest.mark.parametrize("ground", [(10.0, 0.002), (13.0, 0.005), (3.0, 0.001)])
def test_low_height_improves_on_pec_but_not_gated(ground):
    """0.05 lambda: NEC gn 0 is itself shaky this close to ground, so no
    tight bound — but finite must still beat PEC decisively (measured:
    ~0.11 ohm vs PEC 41.8-61.6 ohm)."""
    eps_r, sigma = ground
    gn0 = GOLDEN[("dipole", 0.05, eps_r, sigma)]["finite-fast"]
    z = _solve("dipole", 0.05, ground_eps=(eps_r, sigma))
    z_pec = _solve("dipole", 0.05)
    assert abs(z - gn0) < abs(z_pec - gn0)


def test_impedance_swept_matches_single_k():
    """Both swept loops rebind self.omega per k before _assemble_Z, so the
    eps_t / Fresnel tables are per-frequency automatically. A 1-point sweep
    at the construction k must reproduce compute_impedance (measured:
    exactly equal), and in a 3-k sweep the middle entry (== construction k)
    must match too — this is what guards per-k eps_t handling: a stale
    eps_t(omega) from the first sweep point would show up in the middle
    entry."""
    kw = dict(GEOMS[("dipole", 0.2)])
    ground = dict(ground_z=0.0, ground_eps=(10.0, 0.002))
    z_single, _ = SinusoidalSolver(**kw, **ground).compute_impedance()
    sim = SinusoidalSolver(**kw, **ground)
    k0 = sim.k
    z_sw1 = sim.compute_impedance_swept(np.array([k0]))
    np.testing.assert_allclose(z_sw1[0], z_single, rtol=1e-12)
    z_sw3 = sim.compute_impedance_swept(np.array([0.97 * k0, k0, 1.03 * k0]))
    np.testing.assert_allclose(z_sw3[1], z_single, rtol=1e-12)


def test_y_matrix_swept_matches_single_k():
    """Same per-k eps_t guard for the Y-matrix sweep path, plus the
    Y[0,0] = 1/Z consistency for the single-feed geometry."""
    kw = dict(GEOMS[("dipole", 0.2)])
    ground = dict(ground_z=0.0, ground_eps=(10.0, 0.002))
    Y_single = SinusoidalSolver(**kw, **ground).compute_y_matrix()
    sim = SinusoidalSolver(**kw, **ground)
    k0 = sim.k
    Y_sw = sim.compute_y_matrix_swept(np.array([0.97 * k0, k0, 1.03 * k0]))
    np.testing.assert_allclose(Y_sw[1], Y_single, rtol=1e-12)
    z_single, _ = SinusoidalSolver(**kw, **ground).compute_impedance()
    np.testing.assert_allclose(1.0 / Y_single[0, 0], z_single, rtol=1e-9)


def test_accel_toggle_consistent_under_ground_eps():
    """The grounded-finite image block is pure numpy either way (the C++
    sinusoidal_field_tensor kernel projects pre-dyad, so ground_eps never
    uses it) — only the free-space block differs between accel on/off. This
    guards the _field_components refactor: the numpy fallback must still
    agree with the C++ free-space tensor through the full grounded solve.
    Measured: 1.9e-10 relative on this junction case (1.2e-11 on the
    plain dipole — the KCL-constrained system is a touch worse
    conditioned); guard at ~5x = 1e-9."""
    import momwire.sinusoidal as sin_mod

    if not sin_mod._HAVE_FIELD_TENSOR:
        pytest.skip("C++ accelerator not built")

    z_accel = _solve("inverted_l", 0.2, ground_eps=(10.0, 0.002))
    sin_mod._HAVE_FIELD_TENSOR = False
    try:
        z_numpy = _solve("inverted_l", 0.2, ground_eps=(10.0, 0.002))
    finally:
        sin_mod._HAVE_FIELD_TENSOR = True
    np.testing.assert_allclose(z_numpy, z_accel, rtol=1e-9)
