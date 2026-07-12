"""Distributed series wire impedance (stevenmburns/momwire#131).

Oracles, cheapest-first:

* `_wire_loading.wire_internal_impedance` against both closed-form limits
  of the exact round-conductor solution: DC (R = 1/(σπa²), internal
  X = ωμ₀/(8π)) and strong skin effect (R = X = 1/(2πaσδ)).
* First-order perturbation: for small loading, ΔZ_in = c₀ᵀ·L·c₀ / I₀²
  with c₀ the UNLOADED solve's coefficients (transpose, not conjugate —
  the Galerkin Z is complex-symmetric/reciprocal, so dZ_in/dZ_mn is the
  unconjugated current product). Pins the loading matrix L itself, wired
  through the actual solve, against an independent expression.
* Physics windows on a 20 m half-wave copper dipole: feed-referred loss
  resistance ≈ R'·L/2 (sinusoidal current weighting), efficiency and
  internal-reactance behavior in the strong-skin regime.
* Insulation loading against King's quasi-static jacket inductance, and
  the velocity-factor direction (insulated wire tunes long → X rises at
  fixed frequency).
* Path parity: swept vs per-k, y-matrix vs impedance, HMatrix zblock vs
  dense assembly — the loading must ride every Z consumer identically.
"""

import numpy as np
import pytest

from momwire import BSplineSolver, SinusoidalSolver
from momwire import _wire_loading
from momwire.hmatrix import HMatrixSolver

MU0 = _wire_loading.MU0
SIGMA_CU = 5.8e7

# 20 m-band half-wave dipole in 28 AWG copper — the POTA driving example.
L_DIP = 10.15
A_28 = 0.16e-3
WL = 21.264  # ≈ 14.1 MHz
DIPOLE = [np.array([[0.0, 0.0, -L_DIP / 2], [0.0, 0.0, L_DIP / 2]])]


def _solver(**kw):
    base = dict(wires=DIPOLE, nsegs=81, wavelength=WL, wire_radius=A_28)
    base.update(kw)
    return BSplineSolver(**base)


# ----------------------------------------------------------------------
# Physics helpers (unit level)
# ----------------------------------------------------------------------


def test_internal_impedance_dc_limit():
    a, sigma = 1e-3, SIGMA_CU
    omega = 2 * np.pi * 10.0  # 10 Hz: a/δ ≈ 0.05, deep in the DC regime
    z = _wire_loading.wire_internal_impedance(omega, a, sigma)
    r_dc = 1.0 / (sigma * np.pi * a**2)
    x_dc = omega * MU0 / (8 * np.pi)  # internal inductance μ₀/8π
    assert z.real == pytest.approx(r_dc, rel=1e-4)
    assert z.imag == pytest.approx(x_dc, rel=1e-2)


def test_internal_impedance_strong_skin_limit():
    a, sigma = 1e-3, SIGMA_CU
    omega = 2 * np.pi * 100e6  # a/δ ≈ 150
    z = _wire_loading.wire_internal_impedance(omega, a, sigma)
    delta = np.sqrt(2.0 / (omega * MU0 * sigma))
    r_hf = 1.0 / (2 * np.pi * a * sigma * delta)
    assert z.real == pytest.approx(r_hf, rel=2e-2)
    assert z.imag == pytest.approx(r_hf, rel=2e-2)  # R ≈ X in strong skin


def test_internal_impedance_no_overflow_extreme_skin():
    # a/δ ~ 5·10⁴ — unscaled Bessels overflow around a/δ ≈ 350.
    z = _wire_loading.wire_internal_impedance(2 * np.pi * 1e13, 1e-2, SIGMA_CU)
    assert np.isfinite(z.real) and np.isfinite(z.imag) and z.real > 0


def test_insulation_inductance_value():
    # b/a = 2.5, εr = 3: L' = μ₀/2π · (1 − 1/3) · ln 2.5
    L = _wire_loading.insulation_inductance(1e-3, 2.5e-3, 3.0)
    assert L == pytest.approx(MU0 / (2 * np.pi) * (2 / 3) * np.log(2.5), rel=1e-12)


# ----------------------------------------------------------------------
# Loading through the solve: perturbation oracle
# ----------------------------------------------------------------------


def test_loading_matches_first_order_perturbation():
    """ΔZ_in for a small loading equals c₀ᵀ L c₀ / I₀² from the unloaded
    solve — an independent route to the same number that pins both the
    Gram structure and the Z'(ω) scaling as they enter the real solve."""
    sigma_big = 5.8e10  # tiny loss → first-order error ~(ΔZ/Z)² negligible
    s0 = _solver()
    z0, c0 = s0.compute_impedance()

    s1 = _solver(wire_conductivity=sigma_big)
    z1, _ = s1.compute_impedance()

    n = c0.shape[0]  # single wire, no KCL rows: coeffs are all basis coeffs
    L = np.zeros((n, n), dtype=np.complex128)
    s1._apply_loading(L)
    i0 = 1.0 / z0  # unit-V delta gap: I = V/Z
    dz_pred = (c0 @ L @ c0) / i0**2
    dz = z1 - z0
    assert dz == pytest.approx(dz_pred, rel=1e-3)


def test_copper_dipole_loss_physics_window():
    """28 AWG copper 20 m dipole: ΔR within 15% of the R'·L/2 hand value,
    ΔX ≈ ΔR (strong skin), efficiency in the 0.91–0.95 window."""
    z0, _ = _solver().compute_impedance()
    s = _solver(wire_conductivity=SIGMA_CU)
    z1, c1 = s.compute_impedance()

    rp = np.real(_wire_loading.wire_internal_impedance(s.omega, A_28, SIGMA_CU))
    dr_hand = rp * L_DIP / 2
    assert z1.real - z0.real == pytest.approx(dr_hand, rel=0.15)
    assert z1.imag - z0.imag == pytest.approx(z1.real - z0.real, rel=0.20)

    p_wire, per_wire = s.wire_loss_power(c1)
    p_in = 0.5 * np.real(1.0 / np.conj(z1))
    eff = 1.0 - p_wire / p_in
    assert 0.91 < eff < 0.95
    assert per_wire.shape == (1,)
    assert per_wire[0] == pytest.approx(p_wire)


def test_insulation_reactance_shift_and_direction():
    """PVC-ish jacket: ΔX ≈ ωL'·L/2 (same current weighting as the loss
    row) and positive — the insulated wire looks electrically longer."""
    z0, _ = _solver().compute_impedance()
    b, eps_r = 0.4e-3, 3.0
    s = _solver(insulation_radius=b, insulation_eps_r=eps_r)
    z1, c1 = s.compute_impedance()

    lp = _wire_loading.insulation_inductance(A_28, b, eps_r)
    dx_hand = s.omega * lp * L_DIP / 2
    assert z1.imag - z0.imag == pytest.approx(dx_hand, rel=0.15)
    assert z1.imag > z0.imag
    # purely reactive: no dissipation
    p_wire, _ = s.wire_loss_power(c1)
    assert p_wire == 0.0
    # loss unchanged to first order
    assert z1.real == pytest.approx(z0.real, rel=0.1)


# ----------------------------------------------------------------------
# Defaults, per-wire selection, validation
# ----------------------------------------------------------------------


def test_lossless_default_bit_identical():
    z_default, c_default = _solver().compute_impedance()
    z_nan, c_nan = _solver(wire_conductivity=np.nan).compute_impedance()
    assert z_nan == z_default
    np.testing.assert_array_equal(c_nan, c_default)


def test_per_wire_sequence_matches_scalar():
    z_scalar, _ = _solver(wire_conductivity=SIGMA_CU).compute_impedance()
    z_seq, _ = _solver(wire_conductivity=[SIGMA_CU]).compute_impedance()
    assert z_seq == z_scalar


def test_validation_errors():
    with pytest.raises(ValueError, match="must exceed"):
        _solver(insulation_radius=A_28 / 2, insulation_eps_r=2.0)
    with pytest.raises(ValueError, match="eps_r must be >= 1"):
        _solver(insulation_radius=1e-3, insulation_eps_r=0.5)
    with pytest.raises(ValueError, match="given together"):
        _solver(insulation_radius=1e-3)
    with pytest.raises(ValueError, match="given together"):
        _solver(insulation_eps_r=2.0)
    with pytest.raises(ValueError, match="length-1"):
        _solver(wire_conductivity=[SIGMA_CU, SIGMA_CU])
    with pytest.raises(ValueError, match="> 0 S/m"):
        _solver(wire_conductivity=-1.0)


def test_enrichment_gated():
    with pytest.raises(NotImplementedError, match="enrichment"):
        _solver(wire_conductivity=SIGMA_CU, use_singular_enrichment=True)


def test_sinusoidal_rejects_loading():
    with pytest.raises(NotImplementedError, match="SinusoidalSolver"):
        SinusoidalSolver(
            wires=DIPOLE,
            nsegs=81,
            wavelength=WL,
            wire_radius=A_28,
            wire_conductivity=SIGMA_CU,
        )


# ----------------------------------------------------------------------
# Every-Z-consumer parity
# ----------------------------------------------------------------------

LOSSY_KW = dict(
    wire_conductivity=SIGMA_CU, insulation_radius=0.4e-3, insulation_eps_r=3.0
)


def test_swept_matches_per_k():
    ks = 2 * np.pi / np.array([WL * 0.98, WL, WL * 1.02])
    zs = _solver(**LOSSY_KW).compute_impedance_swept(ks)
    for i, kk in enumerate(ks):
        z_i, _ = _solver(wavelength=2 * np.pi / kk, **LOSSY_KW).compute_impedance()
        assert zs[i] == pytest.approx(z_i, rel=1e-9)


def test_swept_fallback_matches_per_k(monkeypatch):
    """Force the non-batched sweep (the finite-ground/enrichment route) so
    the per-k loop's loading is exercised even where the batched C++
    kernels are installed."""
    import momwire.bspline as mod

    monkeypatch.setattr(mod, "_HAVE_BSPLINE_SWEPT_ASSEMBLE_ACCEL", False)
    ks = 2 * np.pi / np.array([WL * 0.99, WL * 1.01])
    s = _solver(**LOSSY_KW)
    assert not s._swept_batched_available()
    zs = s.compute_impedance_swept(ks)
    for i, kk in enumerate(ks):
        z_i, _ = _solver(wavelength=2 * np.pi / kk, **LOSSY_KW).compute_impedance()
        assert zs[i] == pytest.approx(z_i, rel=1e-9)


def test_y_matrix_and_swept_include_loading():
    z, _ = _solver(**LOSSY_KW).compute_impedance()
    Y = _solver(**LOSSY_KW).compute_y_matrix()
    assert 1.0 / Y[0, 0] == pytest.approx(z, rel=1e-9)

    ks = 2 * np.pi / np.array([WL * 0.99, WL * 1.01])
    Ys = _solver(**LOSSY_KW).compute_y_matrix_swept(ks)
    for i, kk in enumerate(ks):
        z_i, _ = _solver(wavelength=2 * np.pi / kk, **LOSSY_KW).compute_impedance()
        assert 1.0 / Ys[i, 0, 0] == pytest.approx(z_i, rel=1e-9)


def test_hmatrix_zblock_matches_dense():
    h = HMatrixSolver(
        wires=DIPOLE, nsegs=81, wavelength=WL, wire_radius=A_28, **LOSSY_KW
    )
    n = h._context()["n_basis"]
    idx = np.arange(n)
    Z_blocks = h.zblock(idx, idx)

    s = _solver(**LOSSY_KW)
    geom = s._build_geometry()
    supp, polys, _kcl, _wk, _wbg = s._build_basis_polynomials(geom)
    J = s._build_J_blocks(geom, s.k)
    Z_ref = s._apply_loading(s._assemble_Z(J, supp, polys, geom))
    assert np.max(np.abs(Z_blocks - Z_ref)) < 1e-9 * np.max(np.abs(Z_ref))


def test_hmatrix_iterative_solve_carries_loading():
    z_dense, _ = _solver(**LOSSY_KW).compute_impedance()
    h = HMatrixSolver(
        wires=DIPOLE, nsegs=81, wavelength=WL, wire_radius=A_28, **LOSSY_KW
    )
    z_h, _ = h.compute_impedance()
    assert z_h == pytest.approx(z_dense, rel=1e-5)  # GMRES rtol, not roundoff
