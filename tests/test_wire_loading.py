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

Three testing schemes carry the same physical Z'(ω), each in its own form —
BSpline's polynomial Galerkin Gram, the point-matched sinusoidal impedance
boundary condition, and (momwire#395) the sinusoidal Galerkin overlap — so
each gets the same oracle set, and the cross-scheme ΔZ rows tie them
together.
"""

import numpy as np
import pytest

from momwire import BSplineSolver, SinusoidalGalerkinSolver, SinusoidalSolver
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


# ----------------------------------------------------------------------
# SinusoidalSolver (momwire#134): point-matched loading — NEC's impedance
# boundary condition at the match points, not the Galerkin overlap.
# ----------------------------------------------------------------------


def _ssolver(**kw):
    base = dict(wires=DIPOLE, nsegs=81, wavelength=WL, wire_radius=A_28)
    base.update(kw)
    return SinusoidalSolver(**base)


def test_sinusoidal_lossless_default_bit_identical():
    z_default, c_default = _ssolver().compute_impedance()
    z_nan, c_nan = _ssolver(wire_conductivity=np.nan).compute_impedance()
    assert z_nan == z_default
    np.testing.assert_array_equal(c_nan, c_default)


def test_sinusoidal_loading_matches_adjoint_perturbation():
    """The collocation G is NOT complex-symmetric, so the Galerkin
    perturbation oracle (c₀ᵀLc₀/I₀²) doesn't apply. The adjoint form does:
    with α₀ = G⁻¹v, I₀ = rᵀα₀ (r the feed-centre readout vector), and
    w = G⁻ᵀr, a perturbation G → G − L gives ΔZ = −V·(wᵀLα₀)/I₀² to
    first order. Independent route to the same number — pins the loading
    matrix structure, its sign, and the Z'(ω) scaling through the solve."""
    import scipy.linalg

    sigma_big = 5.8e10  # tiny loss → first-order error negligible
    s0 = _ssolver()
    z0, a0 = s0.compute_impedance()

    s1 = _ssolver(wire_conductivity=sigma_big)
    z1, _ = s1.compute_impedance()

    geom = s0._build_geometry()
    seg_view = s0._basis_coefs(geom, s0.k)
    n = geom["n_segs"]
    Lneg = np.zeros((n, n), dtype=np.complex128)
    s1._apply_loading(Lneg, geom, seg_view, s0.k)  # writes −L into zeros
    fi = geom["feed_segs"][0]
    st, en = seg_view["starts"][fi], seg_view["starts"][fi + 1]
    r = np.zeros(n, dtype=np.complex128)
    r[seg_view["jbasis"][st:en]] = seg_view["sigma"][st:en] * (
        seg_view["A"][st:en] + seg_view["C"][st:en]
    )
    # Gᵀw = r straight from the stashed forward-solve factors (LU of Gᵀ,
    # trans=0) — no second factorization, no retained raw matrix.
    w = scipy.linalg.lu_solve(s0.Z_factors, r)
    i0 = 1.0 / z0
    dz_pred = (w @ Lneg @ a0) / i0**2  # ΔZ = −V·wᵀLα₀/I₀², V=1, Lneg=−L
    assert z1 - z0 == pytest.approx(dz_pred, rel=1e-3)


def test_sinusoidal_copper_dipole_physics_window():
    """Same physics windows as the BSpline test: ΔR within 15% of R'·L/2,
    ΔX ≈ ΔR (strong skin), efficiency in the 0.91–0.95 window from the
    closed-form ∫|I|² wire_loss_power readout."""
    z0, _ = _ssolver().compute_impedance()
    s = _ssolver(wire_conductivity=SIGMA_CU)
    z1, c1 = s.compute_impedance()

    rp = np.real(_wire_loading.wire_internal_impedance(s.omega, A_28, SIGMA_CU))
    assert z1.real - z0.real == pytest.approx(rp * L_DIP / 2, rel=0.15)
    assert z1.imag - z0.imag == pytest.approx(z1.real - z0.real, rel=0.20)

    p_wire, per_wire = s.wire_loss_power(c1)
    p_in = 0.5 * np.real(1.0 / np.conj(z1))
    assert 0.91 < 1.0 - p_wire / p_in < 0.95
    assert per_wire.shape == (1,)
    assert per_wire[0] == pytest.approx(p_wire)


def test_sinusoidal_matches_bspline_delta_z():
    """Cross-basis: the copper ΔZ from the point-matched sinusoidal system
    and the Galerkin BSpline system agree to ~1% at N=81 (measured 0.04%
    on ΔR / 0.1% on ΔX) — two testing schemes, one physical loading."""
    z0s, _ = _ssolver().compute_impedance()
    z1s, _ = _ssolver(wire_conductivity=SIGMA_CU).compute_impedance()
    z0b, _ = _solver().compute_impedance()
    z1b, _ = _solver(wire_conductivity=SIGMA_CU).compute_impedance()
    assert (z1s - z0s) == pytest.approx(z1b - z0b, rel=0.01)


def test_sinusoidal_insulation_shift_and_direction():
    z0, _ = _ssolver().compute_impedance()
    b, eps_r = 0.4e-3, 3.0
    s = _ssolver(insulation_radius=b, insulation_eps_r=eps_r)
    z1, c1 = s.compute_impedance()

    lp = _wire_loading.insulation_inductance(A_28, b, eps_r)
    assert z1.imag - z0.imag == pytest.approx(s.omega * lp * L_DIP / 2, rel=0.15)
    assert z1.imag > z0.imag
    p_wire, _ = s.wire_loss_power(c1)
    assert p_wire == 0.0  # purely reactive
    assert z1.real == pytest.approx(z0.real, rel=0.1)


def test_sinusoidal_validation_errors():
    with pytest.raises(ValueError, match="must exceed"):
        _ssolver(insulation_radius=A_28 / 2, insulation_eps_r=2.0)
    with pytest.raises(ValueError, match="given together"):
        _ssolver(insulation_radius=1e-3)
    with pytest.raises(ValueError, match="> 0 S/m"):
        _ssolver(wire_conductivity=-1.0)


def test_sinusoidal_swept_and_y_matrix_parity():
    """Both swept loops and the Y-matrix path funnel through _assemble_Z,
    so the loading must ride every consumer identically."""
    ks = 2 * np.pi / np.array([WL * 0.98, WL, WL * 1.02])
    zs = _ssolver(**LOSSY_KW).compute_impedance_swept(ks)
    for i, kk in enumerate(ks):
        z_i, _ = _ssolver(wavelength=2 * np.pi / kk, **LOSSY_KW).compute_impedance()
        # rel 1e-8, not 1e-9: the per-k constructor re-derives k from
        # wavelength=2π/k (one ulp of k), and the basis trig near the feed
        # resonance amplifies that to ~1e-9 relative on Z.
        assert zs[i] == pytest.approx(z_i, rel=1e-8)

    Y = _ssolver(**LOSSY_KW).compute_y_matrix()
    z, _ = _ssolver(**LOSSY_KW).compute_impedance()
    assert 1.0 / Y[0, 0] == pytest.approx(z, rel=1e-9)


# ----------------------------------------------------------------------
# SinusoidalGalerkinSolver (momwire#395): the GALERKIN overlap form of the
# same three-term shapes — the third testing scheme, and the one that
# closes the family's only shipping capability hole.
# ----------------------------------------------------------------------


def _gsolver(**kw):
    base = dict(wires=DIPOLE, nsegs=81, wavelength=WL, wire_radius=A_28)
    base.update(kw)
    return SinusoidalGalerkinSolver(**base)


def test_sg_lossless_default_bit_identical():
    """The overlap term must be a true no-op when loading is off — not a
    multiply by zero — so the whole assembled matrix, not just the solved
    impedance, has to come back bit-identical."""
    s_default, s_nan = _gsolver(), _gsolver(wire_conductivity=np.nan)
    geom = s_default._build_geometry()
    G_default, _ = s_default._assemble_Z(geom, s_default.k)
    G_nan, _ = s_nan._assemble_Z(s_nan._build_geometry(), s_nan.k)
    assert np.array_equal(G_nan, G_default)

    z_default, c_default = s_default.compute_impedance()
    z_nan, c_nan = s_nan.compute_impedance()
    assert z_nan == z_default
    np.testing.assert_array_equal(c_nan, c_default)


def test_sg_loading_matches_variational_perturbation():
    """Unlike the collocation G, this one IS complex-symmetric (asserted
    below, measured 1.4e-13 relative at N=81), so the Galerkin perturbation
    oracle applies rather than the adjoint one: ΔZ_in = c₀ᵀ·L·c₀ / I₀² from
    the UNLOADED solve, transpose and not conjugate.

    The identity also wants the readout to be the exact dual of the drive,
    and the DEFAULT gap readout is the segment-centre current while the
    drive is the gap-integrated one — they differ at O((kΔ)²). That shows
    up as the oracle's own floor: 1.1e-4 relative here against the 6.6e-5
    the exact adjoint form (wᵀLα₀ with w = G⁻ᵀr) gives on the same numbers,
    both far inside the gate. Pins the overlap Gram, its sign, and the
    Z'(ω) scaling through the real solve.
    """
    sigma_big = 5.8e10  # tiny loss → first-order error ~(ΔZ/Z)² negligible
    s0 = _gsolver()
    z0, c0 = s0.compute_impedance()
    s1 = _gsolver(wire_conductivity=sigma_big)
    z1, _ = s1.compute_impedance()

    geom = s0._build_geometry()
    seg_view = s0._basis_coefs(geom, s0.k)
    G, _ = s0._assemble_Z(geom, s0.k)
    assert np.linalg.norm(G - G.T) < 1e-11 * np.linalg.norm(G)

    n = G.shape[0]  # single wire, no ports: coeffs are all basis coeffs
    L = np.zeros((n, n), dtype=np.complex128)
    s1._apply_loading(L, geom, seg_view, s0.k)  # writes −L into zeros
    i0 = 1.0 / z0  # unit-V delta gap: I = V/Z
    dz_pred = -(c0 @ L @ c0) / i0**2
    assert z1 - z0 == pytest.approx(dz_pred, rel=1e-3)


def test_sg_copper_dipole_physics_window():
    """Same physics windows as the other two schemes: ΔR within 15% of
    R'·L/2, ΔX ≈ ΔR (strong skin), efficiency in the 0.91–0.95 window from
    the inherited closed-form ∫|I|² readout — which needs no Galerkin
    variant, a dissipated-power integral being basis-scheme-independent."""
    z0, _ = _gsolver().compute_impedance()
    s = _gsolver(wire_conductivity=SIGMA_CU)
    z1, c1 = s.compute_impedance()

    rp = np.real(_wire_loading.wire_internal_impedance(s.omega, A_28, SIGMA_CU))
    assert z1.real - z0.real == pytest.approx(rp * L_DIP / 2, rel=0.15)
    assert z1.imag - z0.imag == pytest.approx(z1.real - z0.real, rel=0.20)

    p_wire, per_wire = s.wire_loss_power(c1)
    p_in = 0.5 * np.real(1.0 / np.conj(z1))
    assert 0.91 < 1.0 - p_wire / p_in < 0.95
    assert per_wire.shape == (1,)
    assert per_wire[0] == pytest.approx(p_wire)


def test_sg_matches_other_schemes_delta_z():
    """Cross-testing-scheme: one physical loading, three testing schemes.
    Measured at N=81 — Galerkin-sinusoidal vs point-matched sinusoidal
    2.1e-4 relative on ΔZ (2.1e-4 on ΔR, 2.0e-4 on ΔX), vs Galerkin
    BSpline 7.8e-4 (1.5e-4 on ΔR, 1.2e-3 on ΔX). The tolerance stays the
    sin↔bspline precedent's 1% rather than tightening onto today's
    numbers: what the gate must catch is a wrong operator, and the spread
    it is allowed is the basis gap between the schemes at this mesh."""
    z0g, _ = _gsolver().compute_impedance()
    z1g, _ = _gsolver(wire_conductivity=SIGMA_CU).compute_impedance()
    z0s, _ = _ssolver().compute_impedance()
    z1s, _ = _ssolver(wire_conductivity=SIGMA_CU).compute_impedance()
    z0b, _ = _solver().compute_impedance()
    z1b, _ = _solver(wire_conductivity=SIGMA_CU).compute_impedance()
    assert (z1g - z0g) == pytest.approx(z1s - z0s, rel=0.01)
    assert (z1g - z0g) == pytest.approx(z1b - z0b, rel=0.01)


def test_sg_insulation_shift_and_direction():
    z0, _ = _gsolver().compute_impedance()
    b, eps_r = 0.4e-3, 3.0
    s = _gsolver(insulation_radius=b, insulation_eps_r=eps_r)
    z1, c1 = s.compute_impedance()

    lp = _wire_loading.insulation_inductance(A_28, b, eps_r)
    assert z1.imag - z0.imag == pytest.approx(s.omega * lp * L_DIP / 2, rel=0.15)
    assert z1.imag > z0.imag
    p_wire, _ = s.wire_loss_power(c1)
    assert p_wire == 0.0  # purely reactive
    assert z1.real == pytest.approx(z0.real, rel=0.1)


def test_sg_swept_and_y_matrix_parity():
    """Every solve on this family reaches its matrix through `_assemble_Z`,
    so one call site has to carry the loading to all of them."""
    ks = 2 * np.pi / np.array([WL * 0.98, WL, WL * 1.02])
    zs = _gsolver(**LOSSY_KW).compute_impedance_swept(ks)
    for i, kk in enumerate(ks):
        z_i, _ = _gsolver(wavelength=2 * np.pi / kk, **LOSSY_KW).compute_impedance()
        # rel 1e-8 for the same reason as the collocation parity row: the
        # per-k constructor re-derives k from wavelength=2π/k.
        assert zs[i] == pytest.approx(z_i, rel=1e-8)

    Y = _gsolver(**LOSSY_KW).compute_y_matrix()
    z, _ = _gsolver(**LOSSY_KW).compute_impedance()
    assert 1.0 / Y[0, 0] == pytest.approx(z, rel=1e-9)


def test_sg_junction_port_loading_matches_perturbation():
    """The ported path takes the loading through the same seg-view: a port
    basis is a current distribution on real segments, so its entries carry
    the same three-term shape, and M5b's node-charge correction acts on the
    port's CHARGE and cannot interact with a current-only term.

    Oracle is the variational one again — `_assemble_Z_ported` keeps G
    symmetric (asserted; the correction adds the same scalars to both
    halves) and a junction port's readout is the exact dual of its drive,
    so this form is if anything cleaner here than at a gap feed. The
    one-terminal port's Z is not a physical antenna impedance (it carries
    none of the node's self-capacitance — see `_assemble_Z_ported`); the
    perturbation identity is algebra on the assembled system and does not
    care.
    """
    half = np.array([[0.0, 0.0, -L_DIP / 2], [0.0, 0.0, 0.0]])
    wires = [half, half[::-1] * np.array([1.0, 1.0, -1.0])]

    def ported(**kw):
        base = dict(
            wires=wires,
            n_per_edge_per_wire=[[40], [40]],
            feeds=[],
            junctions=[[(0, "end"), (1, "start")]],
            junction_ports=[(0, 1.0 + 0j)],
            wavelength=WL,
            wire_radius=A_28,
        )
        base.update(kw)
        return SinusoidalGalerkinSolver(**base)

    s0 = ported()
    z0, c0 = s0.compute_impedance()
    s1 = ported(wire_conductivity=5.8e10)
    z1, _ = s1.compute_impedance()

    geom = s0._build_geometry()
    seg_view = s0._basis_coefs(geom, s0.k)
    G, _ = s0._assemble_Z_ported(geom, s0.k)
    assert np.linalg.norm(G - G.T) < 1e-10 * np.linalg.norm(G)
    n = G.shape[0]
    assert n == geom["n_segs"] + 1  # the port basis is a column of G

    L = np.zeros((n, n), dtype=np.complex128)
    s1._apply_loading(L, geom, seg_view, s0.k)
    assert np.any(L[n - 1] != 0.0)  # the port row is loaded like any other
    i0 = 1.0 / z0
    assert z1 - z0 == pytest.approx(-(c0 @ L @ c0) / i0**2, rel=1e-3)

    # …and the inherited power readout survives the extra basis column.
    s_cu = ported(wire_conductivity=SIGMA_CU)
    _z, c_cu = s_cu.compute_impedance()
    p_wire, per_wire = s_cu.wire_loss_power(c_cu)
    assert per_wire.shape == (2,)
    assert p_wire == pytest.approx(per_wire.sum())
    assert per_wire[0] == pytest.approx(per_wire[1], rel=1e-6)  # symmetric halves
