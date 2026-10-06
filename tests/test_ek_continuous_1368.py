"""momwire#1368: extended-kernel eligibility is continuous in geometry.

Until #1368 every solver extended only COAXIAL, EQUAL-RADIUS pairs, so Z
jumped at the first non-zero bend angle and at the first radius ratio above
1 — bs2 +0.11+0.23j ohm at a 0.02 degree bend on a 50 mm wire, razor
+0.024+0.049j at a 1.0001:1 step — where licensed NEC-4.2 and NEC-5 move by
print resolution (#1367 / #1368, Haswell `~/ek1367` / `~/ek1368`). Steve's
decisions (2026-10-06): extend EVERY pair, and give unequal radii NEC Eq 89's
two-radius factor (source tube b, observer ρ):

    T1 = b²ρ²/(4R⁴),   T2 = b²/(2R²),

which is the equal-radius factor at b = ρ. bs2 and SG regularise R by the
observer's radius and take b = the source's; razor regularises by the
source's and takes the transposed spelling (b = the observer's), which gives
the same Z_in to 1e-4 ohm on the #1368 step decks.

The Galerkin fills put an observer INSIDE the source tube at a radius-step
junction (the thin wire's nodes within b of the fat wire), where b²/R² > 1
and the expansion Eq 89 is stops holding: bs2 on the raw factor diverged
under refinement at 5:1 and 20:1. Their factor is evaluated no closer than
the tube, R_f = max(R, b) (`_ek_factor_floored`); it never bites where
b <= ρ, so every equal-radius pair is the pre-#1368 arithmetic to the bit.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import momwire._bspline_kernels as _bk
import momwire._sinusoidal_mp as mp
import momwire.bspline as _bspline_mod
import momwire.razor as _razor_mod
from momwire import BSplineSolver
from momwire._kernel_moments import _axis_frame, _static_axis_moments_ek
from momwire._quadrature import leggauss
from momwire.razor import RazorSolver
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

C0 = 299792458.0
WL = C0 / 7e6
K0 = 2 * math.pi / WL
A_FAT = 0.05


def _knot(m):
    """NEC-5's fed knot on wire 0 (`EX 0 1 k 2`), the probes' feed."""
    h = 10.0 / (11 * m)
    return math.floor(5.0 / h + 1e-9) * h


def _bend(alpha_deg, m=1, a=A_FAT):
    al = math.radians(alpha_deg)
    p0 = np.array([0.0, 0.0, 10.0])
    p1 = np.array([10.0, 0.0, 10.0])
    p2 = p1 + 11.0 * np.array([math.cos(al), math.sin(al), 0.0])
    return dict(
        wires=[np.array([p0, p1, p2])],
        n_per_edge_per_wire=[[11 * m, 12 * m]],
        feeds=[(0, _knot(m), 1 + 0j)],
        wavelength=WL,
        wire_radius=a,
    )


def _step(r, m=1):
    p0, p1, p2 = (
        np.array(p) for p in ((-10.0, 0, 10.0), (0.0, 0, 10.0), (10.0, 0, 10.0))
    )
    return dict(
        wires=[np.array([p0, p1]), np.array([p1, p2])],
        n_per_edge_per_wire=[[11 * m], [11 * m]],
        junctions=[[(0, "end"), (1, "start")]],
        feeds=[(0, _knot(m), 1 + 0j)],
        wavelength=WL,
        wire_radius=[A_FAT, A_FAT / r],
    )


def _z(cls, deck, ek=True, **kw):
    if cls is RazorSolver:
        kw.setdefault("nec5_quadrature", True)
    return complex(
        np.atleast_1d(cls(**deck, extended_kernel=ek, **kw).compute_impedance()[0])[0]
    )


@pytest.fixture
def old_rule(monkeypatch):
    """The pre-#1368 coaxial-and-equal-radius labels, for the negative
    controls: bspline and razor import the rule by name."""
    for mod in (_bspline_mod, _razor_mod):
        monkeypatch.setattr(mod, "_ek_axis_groups", _bk._ek_axis_groups_coaxial)


# ----------------------------------------------------------------------
# continuity — and the jump the old rule had, as the control
# ----------------------------------------------------------------------

# Read as the EK RESPONSE, ΔE = [Z_EK − Z_red](x) − [Z_EK − Z_red](x0), so
# the geometry's own change (1.3e-3 ohm for a 1.0001:1 step on the reduced
# kernel) cancels. Measured under 1e-4 ohm on every fill at a 0.02 degree
# bend and a 1.0001:1 step, against 0.046-0.25 ohm under the old rule.
_CONTINUITY_BAR = 1e-3
_JUMP_FLOOR = 0.02


def _dE(cls, deck, deck0, **kw):
    e = _z(cls, deck, **kw) - _z(cls, deck, ek=False, **kw)
    e0 = _z(cls, deck0, **kw) - _z(cls, deck0, ek=False, **kw)
    return abs(e - e0)


@pytest.mark.parametrize("cls", [BSplineSolver, RazorSolver, SinusoidalGalerkinSolver])
def test_a_bend_is_continuous_at_zero_angle(cls):
    jump = _dE(cls, _bend(0.02), _bend(0.0))
    assert jump <= _CONTINUITY_BAR, jump


@pytest.mark.parametrize("cls", [BSplineSolver, RazorSolver])
def test_the_old_rule_jumps_at_a_bend(cls, old_rule):
    jump = _dE(cls, _bend(0.02), _bend(0.0))
    assert jump >= _JUMP_FLOOR, jump


def test_sg_direct_keeps_the_old_rule_and_its_jump():
    """The direct fill keeps the coaxial rule (its field form is not
    symmetric under the continuous one; `_ek_axis_labels`), so its jump is
    still there — the control for SG's continuity above."""
    jump = _dE(SinusoidalGalerkinSolver, _bend(0.02), _bend(0.0), fill="direct")
    assert jump >= _JUMP_FLOOR, jump


@pytest.mark.parametrize("cls", [BSplineSolver, RazorSolver])
def test_a_radius_step_is_continuous_at_ratio_one(cls):
    jump = _dE(cls, _step(1.0001), _step(1.0))
    assert jump <= _CONTINUITY_BAR, jump


@pytest.mark.parametrize("cls", [BSplineSolver, RazorSolver])
def test_the_old_rule_jumps_at_a_radius_step(cls, old_rule):
    jump = _dE(cls, _step(1.0001), _step(1.0))
    assert jump >= _JUMP_FLOOR, jump


# ----------------------------------------------------------------------
# razor against NEC-5's step response
# ----------------------------------------------------------------------

# NEC-5 (nec5t, black box), printed Z_in = V/I on the collinear step deck at
# x1, fed at the knot 4.545 m (Haswell `~/ek1368/results.jsonl`).
_NEC5_STEP_X1 = {
    1: 128.78387699155107 - 56.29084515064637j,
    2: 124.32718836681396 - 63.7546292385047j,
    20: 112.6048303283927 - 86.74172283505371j,
}
# Measured |ΔE_razor − ΔE_NEC-5| = 0.0087 ohm worst at x1 (0.016 at x5):
# NEC-5's print resolution at these Z. razor-2p's reduced row is NEC-5's
# reduced twin, so the response is read as the EK-minus-reduced change.
_NEC5_STEP_BAR = 0.02


@pytest.mark.parametrize("r", [2, 20])
def test_razor_tracks_nec5s_step_response(r):
    d_ek = _z(RazorSolver, _step(r)) - _z(RazorSolver, _step(1))
    d_red = _z(RazorSolver, _step(r), ek=False) - _z(RazorSolver, _step(1), ek=False)
    d_nec5 = _NEC5_STEP_X1[r] - _NEC5_STEP_X1[1]
    resid = abs((d_ek - d_red) - (d_nec5 - d_red))
    assert resid <= _NEC5_STEP_BAR, resid


# ----------------------------------------------------------------------
# the two-radius factor and statics
# ----------------------------------------------------------------------


def test_the_two_radius_factor_is_the_equal_radius_one_at_b_equal_a():
    R = np.linspace(0.002, 3.0, 101)
    a = 0.01
    for k in (K0, K0 * (2.0 - 0.3j)):
        assert np.array_equal(_bk._ek_factor(R, a, k, a), _bk._ek_factor(R, a, k))
        assert np.array_equal(_bk._ek_reg_extra(R, a, k, a), _bk._ek_reg_extra(R, a, k))
        assert np.array_equal(
            _bk._ek_factor_floored(R + a, a, k, a), _bk._ek_factor(R + a, a, k)
        )


def test_the_floor_bites_only_inside_the_tube():
    a, b = 0.0025, 0.05
    R = np.array([a, 0.5 * b, b, 2 * b, 1.0])
    f = _bk._ek_factor_floored(R, a, K0, b)
    raw = _bk._ek_factor(R, a, K0, b)
    assert np.array_equal(f[2:], raw[2:])  # R >= b: the factor itself
    assert np.array_equal(f[:2], _bk._ek_factor(np.full(2, b), a, K0, b))
    assert np.abs(raw[0]) > 50 and np.abs(f).max() < 2  # what the floor removes


@pytest.mark.parametrize("b_over_a", [1.0, 0.05, 20.0])
def test_the_two_radius_statics_match_their_integrand(b_over_a):
    """`_static_axis_moments_ek(..., b)` against Gauss-Legendre of the static
    integrand 1/R − b²/(2R³) + 3b²a²/(4R⁵) on observers off the segment's end
    (a smooth integrand there, so the rule converges)."""
    a = 0.01
    b = a * b_over_a
    seg_p0 = np.array([[0.0, 0.0, 0.0]])
    seg_t = np.array([[0.0, 0.0, 1.0]])
    h = np.array([0.3])
    obs = np.array([[0.0, 0.0, 0.6], [0.05, 0.0, -0.4], [0.2, 0.1, 0.15]])
    u_r, rho2 = _axis_frame(obs, seg_p0, seg_t, a)
    m0, m1 = _static_axis_moments_ek(u_r, rho2, h, a, b=b)
    x, w = leggauss(400)
    tau = 0.15 * (x + 1.0)
    w = 0.15 * w
    R = np.sqrt((tau[None, :] - u_r) ** 2 + rho2)
    f = 1.0 / R - b * b / (2 * R**3) + 3 * b * b * a * a / (4 * R**5)
    assert np.allclose(m0[:, 0], (f * w).sum(1), rtol=1e-12, atol=0)
    assert np.allclose(m1[:, 0], (f * tau * w).sum(1), rtol=1e-12, atol=0)
    if b_over_a == 1.0:
        e0, e1 = _static_axis_moments_ek(u_r, rho2, h, a)
        assert np.array_equal(m0, e0) and np.array_equal(m1, e1)


# ----------------------------------------------------------------------
# the C++ lanes against their numpy references, on unequal radii
# ----------------------------------------------------------------------


@pytest.mark.parametrize("r", [5.0, 0.2])
def test_razor_fused_matches_numpy_on_a_step(monkeypatch, r):
    deck = _step(r, m=3)
    fused = _z(RazorSolver, deck)
    monkeypatch.setattr(_razor_mod, "_FORCE_NUMPY", True)
    numpy_ = _z(RazorSolver, deck)
    assert abs(fused - numpy_) <= 1e-10 * abs(numpy_), (fused, numpy_)


@pytest.mark.parametrize("r", [20.0, 0.05])
def test_bs2_lanes_match_numpy_on_a_step_with_the_floor_biting(monkeypatch, r):
    """At 20:1 the thin wire's nodes reach inside the fat tube (the floor
    bites on the thin-observer rows); 1:20 is the same deck mirrored."""
    deck = _step(r, m=3)
    lanes = _z(BSplineSolver, deck)
    monkeypatch.setattr(_bk, "_HAVE_BSPLINE_OFFEDGE_EK_ACCEL", False)
    numpy_ = _z(BSplineSolver, deck)
    assert abs(lanes - numpy_) <= 1e-9 * abs(numpy_), (lanes, numpy_)


@pytest.mark.skipif(not mp._HAVE_SIN_TIERED_EK, reason="needs the accelerator")
def test_sg_tiered_kernel_two_radii_match_numpy():
    rng = np.random.default_rng(1368)
    n = 7
    c = rng.uniform(-1.5, 1.5, size=(n, 3))
    t = rng.normal(size=(n, 3))
    t /= np.linalg.norm(t, axis=1)[:, None]
    h = rng.uniform(0.2, 0.6, size=n)
    sl, sr = mp.segment_ends(c, t, h)
    g = np.zeros(n, dtype=np.int64)
    a = 0.02
    b_j = rng.choice([0.004, 0.02, 0.06], size=n)
    ek = (g, g, a, b_j)
    got = mp.pair_moments_tiered(sl, sr, sl, sr, a * a, K0, 8, ((2.0, 4),), ek=ek)
    ref = mp._pair_moments_tiered_numpy(
        sl, sr, sl, sr, a * a, K0, 8, ((2.0, 4),), ek=ek
    )
    assert np.abs(got - ref).max() <= 1e-12 * np.abs(ref).max()
    same = mp.pair_moments_tiered(
        sl, sr, sl, sr, a * a, K0, 8, ((2.0, 4),), ek=(g, g, a)
    )
    eq = b_j == a
    assert np.array_equal(got[:, :, :, eq], same[:, :, :, eq])


@pytest.mark.skipif(not mp._HAVE_SIN_PARALLEL, reason="needs the accelerator")
def test_sg_parallel_reduction_two_radii_match_numpy():
    rng = np.random.default_rng(13)
    n = 10
    t = np.tile([0.0, 0.0, 1.0], (n, 1))
    c_i = np.zeros((n, 3))
    c_i[:, 2] = rng.uniform(-1.0, 1.0, n)
    c_j = np.zeros((n, 3))
    c_j[:, 0] = rng.uniform(0.0, 0.1, n)
    c_j[:, 2] = rng.uniform(-1.0, 1.0, n)
    h_i = rng.uniform(0.3, 0.7, n)
    h_j = rng.uniform(0.3, 0.7, n)
    a2 = np.full(n, 0.01**2)
    a_ek = np.full(n, 0.01)
    b_ek = rng.choice([0.002, 0.01, 0.05], size=n)
    kw = dict(a_ek=a_ek, b_ek=b_ek)
    got = mp.parallel_pair_moments(c_i, t, h_i, c_j, t, h_j, a2, K0, **kw)
    ref = mp._parallel_pair_moments_numpy(c_i, t, h_i, c_j, t, h_j, a2, K0, **kw)
    assert np.abs(got - ref).max() <= 1e-12 * np.abs(ref).max()


# ----------------------------------------------------------------------
# SG serves the corner, symmetrically
# ----------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [20.0, 90.0])
def test_sg_mixed_potential_ek_is_reciprocal_at_a_bend(alpha):
    s = SinusoidalGalerkinSolver(**_bend(alpha), extended_kernel=True)
    geom = s._build_geometry()
    assert s._mp_serves(geom)
    with s._operating_medium(geom) as medium:
        G, _ = s._assemble_Z(geom, s.k, s._medium_eta(medium))
    assert np.abs(G - G.T).max() <= 1e-9 * np.abs(G).max()
