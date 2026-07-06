"""Reflection-coefficient finite ground (`BSplineSolver(ground_eps=...)`).

Golden references are PyNEC gn 0 impedances captured by
scripts/capture_refl_coef_ground_golden.py; geometries are the exact solver
inputs antennaknobs hands momwire, captured by scripts/dump_refl_coef_geoms.py.
No PyNEC / antennaknobs dependency here — everything is literals.

Thresholds carry the measured Phase 1 residuals (see
docs/refl-coef-ground-plan.md) plus a little slack: they are regression
guards against the physics drifting, not re-derivations of the acceptance
analysis.
"""

import numpy as np
import pytest

from momwire import BSplineSolver

from fixtures_refl_coef_geoms import GEOMS
from golden_refl_coef_ground import GOLDEN


def _solve(name, frac, ground_eps=None, **extra):
    kw = dict(GEOMS[(name, frac)])
    z, _ = BSplineSolver(
        **kw, ground_z=0.0, ground_eps=ground_eps, **extra
    ).compute_impedance()
    return z


def test_ground_eps_requires_ground_z():
    kw = dict(GEOMS[("dipole", 0.2)])
    with pytest.raises(ValueError, match="ground_eps requires ground_z"):
        BSplineSolver(**kw, ground_eps=(10.0, 0.002))


def test_bad_phi_mode_rejected():
    kw = dict(GEOMS[("dipole", 0.2)])
    with pytest.raises(ValueError, match="ground_phi_mode"):
        BSplineSolver(
            **kw, ground_z=0.0, ground_eps=(10.0, 0.002), ground_phi_mode="nope"
        )


def test_pec_limit_reproduces_pec_image():
    """ε̃ → ∞ must collapse the Fresnel dyad + Φ weight to the PEC image."""
    kw = dict(GEOMS[("dipole", 0.2)])
    z_pec, _ = BSplineSolver(**kw, ground_z=0.0).compute_impedance()
    z_lim, _ = BSplineSolver(
        **kw, ground_z=0.0, ground_eps=1e16 + 0j
    ).compute_impedance()
    assert abs(z_lim - z_pec) < 1e-5


def test_free_space_unaffected_by_ground_params():
    """ground_eps=None + ground_z=None is bit-exact free space."""
    kw = dict(GEOMS[("dipole", 0.2)])
    z_a, _ = BSplineSolver(**kw).compute_impedance()
    z_b, _ = BSplineSolver(**kw, ground_phi_mode="image").compute_impedance()
    assert z_a == z_b


@pytest.mark.parametrize("frac", [0.1, 0.2, 0.35, 0.5])
@pytest.mark.parametrize("ground", [(10.0, 0.002), (3.0, 0.001)])
def test_dipole_matches_nec_gn0_in_window(frac, ground):
    """Acceptance window: |ΔZ| vs NEC gn 0 within the measured residuals
    (max 2.45 Ω at capture; guard at 3.0), and strictly better than the
    PEC-image solve whose error this feature removes."""
    eps_r, sigma = ground
    gn0 = GOLDEN[("dipole", frac, eps_r, sigma)]["finite-fast"]
    z = _solve("dipole", frac, ground_eps=(eps_r, sigma))
    z_pec = _solve("dipole", frac)
    assert abs(z - gn0) < 3.0
    assert abs(z - gn0) < abs(z_pec - gn0)


def test_inverted_l_junction_geometry_in_window():
    """Junction/KCL geometry runs through the weighted image path; residual
    at the measured ~1.5 Ω level (guard at 2.5)."""
    gn0 = GOLDEN[("inverted_l", 0.2, 10.0, 0.002)]["finite-fast"]
    z = _solve("inverted_l", 0.2, ground_eps=(10.0, 0.002))
    assert abs(z - gn0) < 2.5


def test_low_height_improves_on_pec_but_not_gated():
    """0.05λ: outside the accuracy window, but must still beat PEC by a wide
    margin (measured: 11.6 Ω vs PEC 45.2 Ω on this case)."""
    gn0 = GOLDEN[("dipole", 0.05, 10.0, 0.002)]["finite-fast"]
    z = _solve("dipole", 0.05, ground_eps=(10.0, 0.002))
    z_pec = _solve("dipole", 0.05)
    assert abs(z - gn0) < 0.5 * abs(z_pec - gn0)


def test_complex_eps_matches_tuple_spec():
    """(eps_r, sigma) folds to ε̃ = εr − jσ/(ωε₀); passing that ε̃ directly
    must give the identical solve."""
    kw = dict(GEOMS[("dipole", 0.2)])
    sim = BSplineSolver(**kw, ground_z=0.0, ground_eps=(10.0, 0.002))
    eps_t = complex(10.0, -0.002 / (sim.omega * sim.eps))
    z_tuple, _ = sim.compute_impedance()
    z_complex, _ = BSplineSolver(
        **kw, ground_z=0.0, ground_eps=eps_t
    ).compute_impedance()
    np.testing.assert_allclose(z_complex, z_tuple, rtol=0, atol=1e-9)


def test_y_matrix_path_consistent_with_impedance():
    """compute_y_matrix (what MomwireEngine.impedance() drives) must agree
    with compute_impedance for a single-feed geometry."""
    kw = dict(GEOMS[("dipole", 0.2)])
    sim = BSplineSolver(**kw, ground_z=0.0, ground_eps=(10.0, 0.002))
    z_direct, _ = sim.compute_impedance()
    Y = BSplineSolver(
        **kw, ground_z=0.0, ground_eps=(10.0, 0.002)
    ).compute_y_matrix()
    np.testing.assert_allclose(1.0 / Y[0, 0], z_direct, rtol=1e-9)


def test_impedance_swept_matches_single_k():
    """The swept impedance path delegates per-k to compute_impedance and
    rebinds ω, so ε̃(ω) must track — check against two single-k solves."""
    kw = dict(GEOMS[("dipole", 0.2)])
    sim = BSplineSolver(**kw, ground_z=0.0, ground_eps=(10.0, 0.002))
    k0 = sim.k
    k_arr = np.array([k0 * 0.98, k0 * 1.02])
    z_swept = sim.compute_impedance_swept(k_arr)
    for kk, z_k in zip(k_arr, z_swept):
        wl = 2 * np.pi / kk
        kw_k = dict(kw)
        kw_k["wavelength"] = wl
        z_single, _ = BSplineSolver(
            **kw_k, ground_z=0.0, ground_eps=(10.0, 0.002)
        ).compute_impedance()
        np.testing.assert_allclose(z_k, z_single, rtol=1e-9)


def test_accel_weighted_assembly_matches_numpy_reference(monkeypatch):
    """The C++ assemble_Z_bspline_weighted kernel must agree with the numpy
    einsum fallback (the bit-exact reference). Only the weighted-image flag
    is patched, so the free-space assembly is identical in both solves."""
    from momwire import bspline as bspline_mod

    if not bspline_mod._HAVE_BSPLINE_ASSEMBLE_W_ACCEL:
        pytest.skip("weighted C++ assembler not built")
    z_accel = _solve("inverted_l", 0.2, ground_eps=(10.0, 0.002))
    monkeypatch.setattr(bspline_mod, "_HAVE_BSPLINE_ASSEMBLE_W_ACCEL", False)
    z_numpy = _solve("inverted_l", 0.2, ground_eps=(10.0, 0.002))
    np.testing.assert_allclose(z_accel, z_numpy, rtol=1e-9)


def test_y_matrix_swept_matches_single_k():
    """Swept Y with ground_eps must equal per-k single solves: ε̃(ω) has to
    track the rebound omega while the specular prep is reused across k."""
    kw = dict(GEOMS[("dipole", 0.2)])
    sim = BSplineSolver(**kw, ground_z=0.0, ground_eps=(10.0, 0.002))
    k0 = sim.k
    k_arr = np.array([k0 * 0.97, k0, k0 * 1.03])
    Y_swept = sim.compute_y_matrix_swept(k_arr)
    for kk, Y_k in zip(k_arr, Y_swept):
        kw_k = dict(kw)
        kw_k["wavelength"] = 2 * np.pi / kk
        Y_single = BSplineSolver(
            **kw_k, ground_z=0.0, ground_eps=(10.0, 0.002)
        ).compute_y_matrix()
        np.testing.assert_allclose(Y_k, Y_single, rtol=1e-9)
