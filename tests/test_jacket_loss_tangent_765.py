"""A jacket's dielectric loss tangent — momwire#765.

**The term.** With e^{jωt} a lossy jacket has ε* = εr·(1 − j·tan δ). Its loss
lives in the jacket's elastance, the shell `equivalent_radius` already gives
the kernel at the real εr, so a′ and King's L′ stay as they are and the loss
is one further charge-side term (`_wire_loading.jacket_loss_elastance`)

    ΔS′_loss = ln(b/a)/(2π ε₀) · (1/ε* − 1/εr),       zq = ΔS′/(jω),

whose real part is the dissipation per |dI/dl|²: ½·Re(zq)·∫|dI/dl|² watts.
Its long-wire (TEM) series equivalent is R′_d = Re(zq)·k₀²
= ωμ₀·ln(b/a)·tan δ / (2π εr (1 + tan² δ)) Ω/m.

**The gates** (real constructors, all four families that serve loading):

* the closed form and its check value, and linearity in small tan δ;
* tan δ = 0 / None / NaN is bit-identical to no kwarg, free space and buried,
  with zq structurally None on a free-space wire;
* a jacketed free-space half-wave dipole at tan δ = 0.01: ΔR_in agrees
  across bs2 / sin / SG / razor-2p, and with the first-order estimates from
  the solved current;
* the power budget closes against a FAR-FIELD radiated power (independent of
  the fill), efficiency falls with tan δ, and a flipped sign — in the term
  or in the ledger alone — fails it;
* the ledger's ∫|dI/dl|² equals the sampled `current_slopes` integral on a
  buried lossy jacket (the complex-k closed form of the sinusoidal ledger);
* validation, the deck seam, swept / H-matrix parity, the symmetry check.
"""

from __future__ import annotations

import math
import pathlib
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from momwire import (
    BSplineSolver,
    SinusoidalGalerkinSolver,
    SinusoidalSolver,
    _wire_loading,
)
from momwire._rotational_symmetry import RotationalSymmetryRefused
from momwire._wire_loading import MU0
from momwire.deck._solver import _wire_loading as deck_wire_loading
from momwire.deck.model import WireMaterial
from momwire.hmatrix import HMatrixSolver
from momwire.razor import RazorSolver
from test_rotational_symmetry_1029 import solver as screen_solver

C0 = 299792458.0
EPS0 = BSplineSolver.eps
ETA0 = 376.730313668
WL7 = C0 / 7.0e6
L_DIP = 0.48 * WL7
A, B, ER = 1e-3, 2e-3, 3.5
TAN_D = 0.01

FAMILIES = {
    "bs2": BSplineSolver,
    "sin": SinusoidalSolver,
    "sg": SinusoidalGalerkinSolver,
    "razor-2p": RazorSolver,
}


def _build(family, d):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if family == "razor-2p":
            return RazorSolver(**d, nec5_quadrature=True)
        return FAMILIES[family](**d)


def free_dipole(n=31, **kw):
    """A jacketed centre-fed half-wave-ish dipole on z in free space."""
    pts = np.array([(0.0, 0.0, -0.5 * L_DIP), (0.0, 0.0, 0.5 * L_DIP)])
    d = dict(
        wires=[pts],
        n_per_edge_per_wire=[[n]],
        feeds=[(0, 0.5 * L_DIP, 1 + 0j)],
        wavelength=WL7,
        wire_radius=A,
        insulation_radius=B,
        insulation_eps_r=ER,
    )
    d.update(kw)
    return d


def buried_dipole(n=12, length=5.0, depth=0.5, **kw):
    """test_jacket_in_soil_1154's buried dipole, in soil A."""
    pts = np.array([(-0.5 * length, 0.0, -depth), (0.5 * length, 0.0, -depth)])
    d = dict(
        wires=[pts],
        n_per_edge_per_wire=[[n]],
        feeds=[(0, 0.5 * length, 1 + 0j)],
        wavelength=WL7,
        wire_radius=A,
        insulation_radius=1.8e-3,
        insulation_eps_r=ER,
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
    )
    d.update(kw)
    return d


def solve(family, d):
    s = _build(family, d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z, c = s.compute_impedance()
    return s, complex(np.ravel(z)[0]), np.asarray(c)


def p_rad_far(s, c):
    """Radiated watts of a z-directed wire from its far field — the current
    moments only, nothing the fill or a ledger computed:

        P = η k² / (32π²) ∫ |M_z(θ)|² sin²θ dΩ,   M_z = Σ I·dl e^{jk z cosθ}.
    """
    mid, moment, _n, _d = s.element_currents(c, subdiv=8)
    k = 2.0 * math.pi / WL7
    th = np.linspace(0.0, math.pi, 721)
    mz = (moment[:, 2][None, :] * np.exp(1j * k * np.cos(th)[:, None] * mid[:, 2])).sum(
        axis=1
    )
    f = np.abs(mz) ** 2 * np.sin(th) ** 3
    return ETA0 * k * k / (32.0 * math.pi**2) * 2.0 * math.pi * np.trapezoid(f, th)


_CACHE: dict = {}


def free_solve(family, tan_d):
    key = (family, tan_d)
    if key not in _CACHE:
        kw = {} if tan_d is None else {"insulation_loss_tangent": tan_d}
        _CACHE[key] = solve(family, free_dipole(**kw))
    return _CACHE[key]


# ----------------------------------------------------------------------
# the term
# ----------------------------------------------------------------------


def test_the_check_value_and_the_series_equivalent():
    """1 mm wire, 2 mm jacket, εr 3.5, tan δ 1e-3, 7 MHz: zq ≈ 0.0809 Ω·m and
    zq·k₀² is the TEM series equivalent R′_d ≈ 1.742e-3 Ω/m to 1e-12."""
    omega = 2.0 * math.pi * 7.0e6
    t = 1e-3
    zq = _wire_loading.jacket_loss_elastance(A, B, ER, t, EPS0) / (1j * omega)
    assert zq.real == pytest.approx(0.0809, abs=5e-5)
    k0 = omega * math.sqrt(EPS0 * MU0)
    r_d = omega * MU0 * math.log(B / A) * t / (2.0 * math.pi * ER * (1.0 + t * t))
    assert (zq * k0 * k0).real == pytest.approx(r_d, rel=1e-12)
    assert r_d == pytest.approx(1.742e-3, rel=1e-3)
    # The definition, 1/ε* − 1/εr, is the closed form the function spells.
    eps_star = ER * (1.0 - 1j * t)
    want = math.log(B / A) / (2.0 * math.pi * EPS0) * (1.0 / eps_star - 1.0 / ER)
    got = _wire_loading.jacket_loss_elastance(A, B, ER, t, EPS0)
    assert got == pytest.approx(want, rel=1e-9)
    # Dissipative: Im ΔS′ > 0 is ½·ω·Im(ΔS′)·|q|² > 0 watts per metre.
    assert got.imag > 0.0 and zq.real > 0.0
    assert _wire_loading.jacket_loss_elastance(A, B, ER, 0.0, EPS0) == 0.0


def test_the_term_is_linear_in_a_small_loss_tangent():
    unit = _wire_loading.jacket_loss_elastance(A, B, ER, 1.0, EPS0)
    for t in (1e-6, 1e-4, 1e-2):
        got = _wire_loading.jacket_loss_elastance(A, B, ER, t, EPS0)
        # Im is t/(1 + t²) of the unit's slope, i.e. linear to O(t²).
        assert got.imag / t == pytest.approx(2.0 * unit.imag, rel=2 * t * t)


def test_the_solved_resistance_is_linear_in_a_small_loss_tangent():
    _s, z0, _c = free_solve("bs2", None)
    _s, z1, _c = free_solve("bs2", 1e-3)
    _s, z2, _c = free_solve("bs2", TAN_D)
    ratio = (z2 - z0).real / (10.0 * (z1 - z0).real)
    # Measured 1.00046: the second-order part of a 0.3 % resistance change.
    assert abs(ratio - 1.0) < 2e-3


# ----------------------------------------------------------------------
# bit-identity
# ----------------------------------------------------------------------


@pytest.mark.parametrize("family", list(FAMILIES))
@pytest.mark.parametrize("off", [0.0, [float("nan")]], ids=["zero", "nan"])
def test_a_lossless_tangent_is_bit_identical_in_free_space(family, off):
    s0, z0, c0 = free_solve(family, None)
    s1, z1, c1 = solve(family, free_dipole(insulation_loss_tangent=off))
    assert z1 == z0
    np.testing.assert_array_equal(c1, c0)
    # Structurally absent, not zero.
    assert _wire_loading.loading_for(s1, s1.omega).zq_wire is None
    assert _wire_loading.jacket_loss_charge(s1, s1.omega) is None


def test_a_lossless_tangent_leaves_the_bspline_z_matrix_bit_identical():
    def z_matrix(d):
        s = _build("bs2", d)
        geom = s._build_geometry()
        supp, polys, _kcl, _wk, _wbg = s._build_basis_polynomials(geom)
        J = s._build_J_blocks(geom, s.k)
        return s._apply_loading(s._assemble_Z(J, supp, polys, geom))

    np.testing.assert_array_equal(
        z_matrix(free_dipole(insulation_loss_tangent=0.0)), z_matrix(free_dipole())
    )


@pytest.mark.parametrize("family", list(FAMILIES))
def test_a_lossless_tangent_is_bit_identical_on_a_buried_jacket(family):
    _s0, z0, c0 = solve(family, buried_dipole())
    s1, z1, c1 = solve(family, buried_dipole(insulation_loss_tangent=0.0))
    assert z1 == z0
    np.testing.assert_array_equal(c1, c0)
    np.testing.assert_array_equal(
        _wire_loading.loading_for(s1, s1.omega).zq_wire,
        _wire_loading.buried_jacket_charge(s1, s1.omega),
    )


def test_a_lossy_buried_jacket_carries_both_charge_terms_as_summands():
    s = _build("bs2", buried_dipole(insulation_loss_tangent=TAN_D))
    buried = _wire_loading.buried_jacket_charge(s, s.omega)
    loss = _wire_loading.jacket_loss_charge(s, s.omega)
    np.testing.assert_array_equal(
        _wire_loading.loading_for(s, s.omega).zq_wire, buried + loss
    )
    assert loss[0].real > 0.0


# ----------------------------------------------------------------------
# across the families, and against the first-order estimates
# ----------------------------------------------------------------------


def test_the_resistance_shift_agrees_across_every_family():
    """Measured ΔR_in: bs2 0.20417, sin 0.20378, SG 0.20417, razor-2p
    0.20292 Ω (spread 0.6 %); ΔX ≈ −0.011 Ω on all four."""
    dr = {}
    for family in FAMILIES:
        _s, z0, _c = free_solve(family, None)
        _s, z1, _c = free_solve(family, TAN_D)
        dr[family] = (z1 - z0).real
        assert dr[family] > 0.0
    lo, hi = min(dr.values()), max(dr.values())
    assert hi / lo - 1.0 < 0.02, dr


@pytest.mark.parametrize("family", list(FAMILIES))
def test_the_resistance_shift_matches_the_first_order_estimates(family):
    """ΔR against the solved current I(l) (feed current I_f):

    * the charge reading ∫Re(zq)|dI/dl|²/|I_f|² — what the term IS — within
      2 % (measured 0.9–1.0 %);
    * its TEM series equivalent ∫R′_d|I|²/|I_f|² within 12 % (measured
      8 %): on a 0.48 λ jacketed dipole |I′|²/k₀² outweighs |I|² (the slow
      wave's k exceeds k₀, and cos² outweighs sin² short of resonance), so
      the series reading is the long-wire approximation the module calls it.
    """
    _s, z0, _c = free_solve(family, None)
    s, z1, c = free_solve(family, TAN_D)
    dr = (z1 - z0).real
    arcs = [np.linspace(0.0, L_DIP, 4001)]
    i_l = s.currents_at_knots(c, arcs)[0]
    di_l = s.current_slopes(c, arcs)[0]
    i_feed2 = abs(1.0 / z1) ** 2
    zq = _wire_loading.jacket_loss_charge(s, s.omega)[0]
    k0 = s.omega / C0
    charge = zq.real * np.trapezoid(np.abs(di_l) ** 2, arcs[0]) / i_feed2
    series = zq.real * k0 * k0 * np.trapezoid(np.abs(i_l) ** 2, arcs[0]) / i_feed2
    assert charge == pytest.approx(dr, rel=0.02)
    assert series == pytest.approx(dr, rel=0.12)


# ----------------------------------------------------------------------
# the power budget
# ----------------------------------------------------------------------


def _budget(family, tan_d):
    s, z, c = free_solve(family, tan_d)
    p_in = 0.5 * (1.0 / np.conj(z)).real
    p_wire = s.wire_loss_power(c)[0] if tan_d is not None else 0.0
    return p_in, p_rad_far(s, c), p_wire


def _budget_gate(family, budgets):
    """The gate, as a predicate so the sign-flip test can watch it fail.

    `budgets` is {tan δ: (P_in, P_rad, P_wire)} for None, 1e-3 and 1e-2. The
    lossless budget's own residual (far-field quadrature and the family's
    discretisation) is the floor; a lossy budget must close to that floor
    within 1 % of its dielectric watts, and efficiency must fall with tan δ.
    """
    p_in0, p_rad0, _ = budgets[None]
    floor = (p_in0 - p_rad0) / p_in0
    ok = abs(floor) < 1e-3
    eff = [1.0]
    for t in (1e-3, TAN_D):
        p_in, p_rad, p_wire = budgets[t]
        ok &= p_wire > 0.0
        ok &= abs((p_in - p_rad - p_wire) / p_in - floor) < 0.01 * p_wire / p_in
        eff.append(1.0 - p_wire / p_in)
    ok &= eff[0] > eff[1] > eff[2]
    return bool(ok)


@pytest.mark.parametrize("family", list(FAMILIES))
def test_the_power_budget_closes_and_efficiency_falls(family):
    """Measured: lossless floors 3.6e-5 (bs2, SG), 4.1e-4 (sin), −2.6e-4
    (razor-2p) of P_in; each lossy budget sits on its floor to < 1e-3 of its
    1.7e-5 W of dielectric loss; efficiency 1 → 0.99971 → 0.99716."""
    budgets = {t: _budget(family, t) for t in (None, 1e-3, TAN_D)}
    assert _budget_gate(family, budgets)


def test_a_flipped_sign_fails_the_budget_gate(monkeypatch):
    family = "bs2"
    good = {t: _budget(family, t) for t in (None, 1e-3, TAN_D)}
    assert _budget_gate(family, good)

    # The ledger alone flipped: the budget no longer closes.
    flipped_ledger = dict(good)
    for t in (1e-3, TAN_D):
        s, _z, c = free_solve(family, t)
        real = _wire_loading.jacket_loss_charge
        with monkeypatch.context() as m:
            m.setattr(
                _wire_loading, "jacket_loss_charge", lambda sv, w, f=real: -f(sv, w)
            )
            p_wire = s.wire_loss_power(c)[0]
        p_in, p_rad, _ = good[t]
        flipped_ledger[t] = (p_in, p_rad, p_wire)
    assert not _budget_gate(family, flipped_ledger)

    # The term itself flipped (fill and ledger together): the budget still
    # closes, but the jacket now amplifies — ΔR < 0 and efficiency > 1.
    real = _wire_loading.jacket_loss_elastance
    monkeypatch.setattr(
        _wire_loading, "jacket_loss_elastance", lambda *a, f=real: -f(*a)
    )
    flipped = {None: good[None]}
    for t in (1e-3, TAN_D):
        s, z, c = solve(family, free_dipole(insulation_loss_tangent=t))
        p_in = 0.5 * (1.0 / np.conj(z)).real
        flipped[t] = (p_in, p_rad_far(s, c), s.wire_loss_power(c)[0])
    assert not _budget_gate(family, flipped)


@pytest.mark.parametrize("family", list(FAMILIES))
def test_the_ledger_is_the_sampled_slope_integral_on_a_buried_jacket(family):
    """½·Re(zq)·∫|dI/dl|² from each ledger's closed form against the same
    integral of `current_slopes` sampled at 20001 points — on a BURIED
    jacket, so the sinusoidal ledger runs its complex-k form. Measured
    ≤ 7e-9 relative (razor 3.7e-6: its slope jumps at every knot)."""
    s, _z, c = solve(family, buried_dipole(insulation_loss_tangent=TAN_D))
    arcs = [np.linspace(0.0, 5.0, 20001)]
    zq = _wire_loading.jacket_loss_charge(s, s.omega)[0]
    di_l = s.current_slopes(c, arcs)[0]
    sampled = 0.5 * zq.real * np.trapezoid(np.abs(di_l) ** 2, arcs[0])
    assert s.wire_loss_power(c)[0] == pytest.approx(sampled, rel=1e-5)


# ----------------------------------------------------------------------
# the other Z consumers
# ----------------------------------------------------------------------


def test_swept_matches_per_k():
    lossy = dict(insulation_loss_tangent=TAN_D)
    ks = 2.0 * np.pi / np.array([WL7 * 0.98, WL7, WL7 * 1.02])
    zs = _build("bs2", free_dipole(**lossy)).compute_impedance_swept(ks)
    for i, kk in enumerate(ks):
        _s, z_i, _c = solve("bs2", free_dipole(wavelength=2 * np.pi / kk, **lossy))
        assert zs[i] == pytest.approx(z_i, rel=1e-9)


def test_hmatrix_zblock_carries_the_loss_term():
    d = free_dipole(insulation_loss_tangent=TAN_D)
    d.pop("n_per_edge_per_wire")
    d.pop("feeds")
    h = HMatrixSolver(nsegs=31, **d)
    n = h._context()["n_basis"]
    idx = np.arange(n)
    s = BSplineSolver(nsegs=31, **d)
    geom = s._build_geometry()
    supp, polys, _kcl, _wk, _wbg = s._build_basis_polynomials(geom)
    J = s._build_J_blocks(geom, s.k)
    z_ref = s._apply_loading(s._assemble_Z(J, supp, polys, geom))
    assert np.max(np.abs(h.zblock(idx, idx) - z_ref)) < 1e-9 * np.max(np.abs(z_ref))


# ----------------------------------------------------------------------
# validation and the seams
# ----------------------------------------------------------------------


@pytest.mark.parametrize("family", list(FAMILIES))
def test_validation_errors(family):
    with pytest.raises(ValueError, match="needs a jacket"):
        _build(
            family,
            free_dipole(
                insulation_radius=None,
                insulation_eps_r=None,
                insulation_loss_tangent=TAN_D,
            ),
        )
    with pytest.raises(ValueError, match=r"\[0\] must be finite and >= 0"):
        _build(family, free_dipole(insulation_loss_tangent=-1e-3))
    with pytest.raises(ValueError, match=r"\[0\] must be finite and >= 0"):
        _build(family, free_dipole(insulation_loss_tangent=float("inf")))
    with pytest.raises(ValueError, match="length-1"):
        _build(family, free_dipole(insulation_loss_tangent=[TAN_D, TAN_D]))


def test_a_tangent_on_an_unjacketed_wire_is_refused():
    pts = [
        np.array([(0.0, 0.0, -5.0), (0.0, 0.0, 5.0)]),
        np.array([(1.0, 0.0, -5.0), (1.0, 0.0, 5.0)]),
    ]
    d = dict(
        wires=pts,
        n_per_edge_per_wire=[[11], [11]],
        feeds=[(0, 5.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=A,
        insulation_radius=[B, float("nan")],
        insulation_eps_r=[ER, float("nan")],
    )
    with pytest.raises(ValueError, match=r"\[1\] is set on a wire with no jacket"):
        _build("bs2", dict(d, insulation_loss_tangent=TAN_D))
    s = _build("bs2", dict(d, insulation_loss_tangent=[TAN_D, float("nan")]))
    zq = _wire_loading.jacket_loss_charge(s, s.omega)
    assert zq[0] != 0.0 and zq[1] == 0.0


def test_the_deck_seam_carries_the_tangent_only_when_one_is_named():
    lossless = (
        WireMaterial(insulation_radius=2e-3, insulation_eps_r=2.3),
        None,
    )
    assert "insulation_loss_tangent" not in deck_wire_loading(lossless)
    lossy = (
        WireMaterial(
            insulation_radius=2e-3, insulation_eps_r=2.3, insulation_loss_tangent=0.02
        ),
        WireMaterial(conductivity=5.8e7),
    )
    kwargs = deck_wire_loading(lossy)
    assert kwargs["insulation_loss_tangent"][0] == 0.02
    assert np.isnan(kwargs["insulation_loss_tangent"][1])
    with pytest.raises(ValueError, match="needs a jacket"):
        WireMaterial(conductivity=5.8e7, insulation_loss_tangent=0.02)


def test_the_symmetry_check_reads_the_tangent_as_part_of_the_jacket():
    n_w = 6
    jacket = dict(insulation_radius=[2e-3] * n_w, insulation_eps_r=[3.0] * n_w)
    tan_d = [TAN_D] * n_w
    tan_d[1] = 0.02
    with pytest.raises(RotationalSymmetryRefused) as exc:
        screen_solver(insulation_loss_tangent=tan_d, **jacket)
    assert "sector 1's jacket loss tangent is 0.02" in str(exc.value)
