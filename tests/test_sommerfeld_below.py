"""The below/below Sommerfeld remainder: both points in the ground
(momwire#553 unit 2).

ORACLES (three, none of them the licensed engine)
-------------------------------------------------
1. **The #524 phase-0 prototype's direct evaluation**, committed as numbers
   in `golden_below_below_524.py`. It is an independent implementation of
   the same physics from the same open-literature equation kit, on the same
   contour recipe, written before any of this module existed — and its
   ±=+ anchor reproduces momwire's OWN four interpolation surfaces to
   1.4e-10, which is what makes it an oracle for the ±=− ones rather than
   a second opinion.
2. **empymod** (`ht='quad'`, pts_per_dec 600) on the SPEC M-line, also
   committed, each grid carrying its own recorded quad-vs-quad spread. A
   tolerance here is named against that spread and never below it.
3. **momwire itself, at ±=+.** The generalized R₁ → 0 limit is written once
   for both regimes and must reproduce `_sommerfeld._limits_r1_zero` — eqs
   169–172 — exactly. The static-coefficient groupings are where a swapped
   sign hides, so that identity is a gate and not a comment.

WHY THE GRID GATES NEVER COMPARE GRID TO GRID
---------------------------------------------
`_sommerfeld`'s own history: a coarse far lattice once measured ≤ 7e-4
against a direct evaluation that was itself wrong past R₁ ≈ 2.84 λ at
grazing — a mis-branched fig-14 waypoint returned a nearly FLAT surface,
and a flat surface is exactly what a coarse lattice interpolates well
(the addendum at `_sommerfeld.py`'s `_SOMM_R1_NEAR_LAMBDA`). Every grid
gate below therefore runs grid → direct → committed goldens. A grid that
agreed with a direct evaluator and nothing else would prove only that the
two share a bug.

WHAT THE SIGN SCAN IS FOR
-------------------------
On the M-line the remainder is not a correction: phase 0 measured
|E_rem|/|E_direct| rising from 0.10 at ρ = 1 m to 2.3 at ρ = 10 m, and
this unit measured it rising past 1e5 by 4 λ_m. So the eight-way scan
over {image dyad literal/mirror} × {sign A_m} × {sign remainder} is a
live test: the winner must be the measured one AND every loser must fail
by O(1). A scan whose losers stopped failing would mean the remainder had
gone quiet, which is itself the failure.
"""

import math

import numpy as np
import pytest

from momwire import _sommerfeld as som
from momwire import _sommerfeld_below as below
from momwire._ground_mirror import MIRROR, mirror_positions, mirror_tangents

import golden_below_below_524 as gold

EPS0 = 8.8541878128e-12
MU0 = som._MU0
C0 = 299792458.0
KEYS = som._SURF_KEYS
GROUND_Z = 0.0


# ----------------------------------------------------------------------
# The media, and the closed forms the composition is assembled from
# ----------------------------------------------------------------------


def eps_tilde(eps_r, sigma, f_hz):
    """ε̃ = ε_r − jσ/(ωε₀) (scratch/524-phase0/proto/EQUATIONS.md)."""
    return eps_r - 1j * sigma / (2.0 * np.pi * f_hz * EPS0)


def medium(cell):
    """(ε̃, k_p, ω, k_m) for a golden cell key like "A/7MHz"."""
    sname, fname = cell.split("/")
    er, sg = gold.SOILS[sname]
    f = float(fname[:-3]) * 1e6
    om = 2.0 * np.pi * f
    et = eps_tilde(er, sg, f)
    kp = om / C0
    return et, kp, om, below.k_medium(et, kp)


def fs_field(c1, k, p_hat, d):
    """E = C₁(∇∇/k² + I)·(p̂ e^{−jkR}/R) at displacement d = r − r′, complex
    k — EQUATIONS.md's regime-2 direct term, transcribed, not remembered."""
    d = np.asarray(d, dtype=float)
    R = float(np.sqrt(np.dot(d, d)))
    rh = d / R
    a = 1.0 / (k * R)
    g = np.exp(-1j * k * R) / R
    pr = np.dot(rh, p_hat)
    return (
        c1
        * g
        * (
            np.asarray(p_hat, dtype=np.complex128) * (1.0 - 1j * a - a * a)
            + rh * pr * (-1.0 + 3j * a + 3.0 * a * a)
        )
    )


def image_field(c1, k, p_hat, obs, src, dyad="mirror"):
    """AGARD (4b)'s image term, both readings.

    `mirror` — momwire's own shipped convention (`_ground_mirror`: mirrored
    positions, mirrored tangents, one global minus) — is the one phase 0
    measured, against the oracle AND against a σ → ∞ PEC limit with no
    oracle in it at all. `literal` is the left-contraction reading of the
    paper's notation, kept so the scan can keep measuring that it loses.
    """
    src_img = mirror_positions(np.asarray(src, dtype=float), GROUND_Z)
    d = np.asarray(obs, dtype=float) - src_img
    if dyad == "mirror":
        return -fs_field(c1, k, mirror_tangents(np.asarray(p_hat, dtype=float)), d)
    if dyad == "literal":
        return -MIRROR * fs_field(c1, k, p_hat, d)
    raise ValueError(dyad)


class DirectSurfaces:
    """A grid-shaped view on `iv_surfaces_direct_below`.

    The projection takes anything that can `eval(R1, theta)`, so the same
    dyad algebra can be driven off the exact surfaces instead of a
    tabulation. That is what makes the three-way grid → direct → goldens
    comparison possible with ONE combination routine: if the grid and the
    direct evaluator disagree, the dyad is not the suspect.
    """

    regime = "below"

    def __init__(self, eps_t, k2, omega, r1_max=1e6, rtol=1e-9):
        self.eps_t = eps_t
        self.k2 = k2
        self.omega = omega
        self.r1_max = r1_max
        self.rtol = rtol

    def eval(self, R1, theta):
        return below.iv_surfaces_direct_below(
            self.eps_t, self.k2, R1, theta, rtol=self.rtol, omega=self.omega
        )


def compose(eps_t, kp, omega, obs, src, p_hat, surfaces, dyad="mirror", sA=1.0, s=1.0):
    """E_total = E_direct(k_m) + sA·A_m·E_image + s·E_remainder at one point."""
    km = below.k_medium(eps_t, kp)
    c1 = -1j * omega * MU0 / (4.0 * np.pi)
    obs = np.asarray(obs, dtype=float)
    src = np.asarray(src, dtype=float)
    e_dir = fs_field(c1, km, p_hat, obs - src)
    e_img = image_field(c1, km, p_hat, obs, src, dyad=dyad)
    a_m = sA * below.image_coefficient_below(eps_t)
    e_rem = below.remainder_field_below(
        obs[None, :],
        src[None, :],
        np.asarray(p_hat, dtype=complex)[None, :],
        GROUND_Z,
        kp,
        km,
        surfaces,
    )[0]
    return e_dir + a_m * e_img + s * e_rem


# `record_property` values must be plain Python scalars or strings, never
# numpy ones. execnet serializes each test report to the xdist controller and
# its `_Serializer` has no dispatch for `np.float64` — the worker dies mid-
# report and the run ends in an INTERNALERROR naming the test, with no
# failure and no traceback pointing here. Serial runs never notice, because
# nothing is serialized. Hence the `float(...)` on every value below.


def rel(got, ref):
    got = np.asarray(got)
    ref = np.asarray(ref)
    den = max(float(np.max(np.abs(ref))), 1e-300)
    return float(np.max(np.abs(got - ref))) / den


# ======================================================================
# G-U2-1 — the surfaces against the phase-0 prototype
# ======================================================================
#
# 144 (R1, theta) points, six SPEC cells, R1 from 0.01 to 3.5 in-medium
# wavelengths and theta from 2 to 80 degrees. This is the core formulation
# gate: the swapped `_d12`, the (5a)-(5d) components at the lower sign, the
# one flipped z-derivative, the k_o^2/k_s^2 prefactor pattern and the g_m
# divide-out, all at once against an evaluator that shares no line of code
# with this one.
#
# MEASURED worst over every cell: 1.50e-9 (a separate dense 336-point
# sweep at rtol 1e-11, over R1 0.005-3.5 lambda_m and theta 0.5-89 deg,
# read 3.1e-10 against the same oracle). Pinned at 1e-7 — ~67x margin, and
# still four decades under any tolerance that could absorb a sign.

G_U2_1_TOL = 1e-7


@pytest.mark.parametrize("cell", sorted(gold.SURFACES))
def test_gu2_1_surfaces_vs_the_phase0_prototype(cell, record_property):
    et, kp, om, km = medium(cell)
    lam_m = 2.0 * np.pi / abs(km)
    rows = gold.SURFACES[cell]
    r1 = np.array([r * lam_m for r, _, *_ in rows])
    th = np.radians(np.array([t for _, t, *_ in rows]))
    got = below.iv_surfaces_direct_below(et, kp, r1, th, rtol=1e-9, omega=om)
    worst = 0.0
    for i, row in enumerate(rows):
        ref = dict(zip(KEYS, row[2:]))
        scale = max(abs(v) for v in ref.values())
        for k in KEYS:
            worst = max(worst, abs(got[k][i] - ref[k]) / scale)
    record_property("worst_rel", float(worst))
    assert worst < G_U2_1_TOL, f"{cell}: worst rel {worst:.3e}"


def test_gu2_1_the_kernel_pair_is_the_swapped_d12():
    """`_d12(λ, k_p, k_m)` IS the below/below D-pair — the claim the whole
    module rests on, spelled out against the generalized form.

    `_d12(λ, k1, k2)` builds around γ₂ as the decay γ and k1 as the other
    medium. Below, the decay medium is the GROUND, so k_m goes in the k2
    slot. Reading the arguments the other way round is the ±=+ pair, and
    the two are numerically different everywhere — asserted, so that a
    future "simplification" that unswaps them cannot pass.
    """
    et, kp, om, km = medium("A/7MHz")
    lam = np.linspace(1e-3, 4.0 * abs(km), 400).astype(np.complex128)
    d1_below, d2_below, g_below = som._d12(lam, kp, km)
    d1_above, d2_above, g_above = som._d12(lam, km, kp)
    assert np.allclose(g_below, som._gamma(lam, km), rtol=0, atol=0)
    assert np.allclose(g_above, som._gamma(lam, kp), rtol=0, atol=0)
    # Not a sign flip of each other, not equal, not proportional.
    assert rel(d1_below, d1_above) > 0.1
    assert rel(d2_below, d2_above) > 0.1


def test_gu2_1_the_integrand_z_derivative_flips_and_nothing_else_does():
    """Exactly one of the six integrands changes sign under the swap.

    Built as the ±=+ integrand evaluated with the SAME (D₁, D₂, γ) — i.e.
    `_integrand_six(λ, ρ, h, k_m, k_p)`, whose `_d12(λ, k_m, k_p)` call is
    the swapped pair spelled the other way — so any difference between the
    two stacks is the z-derivative sign alone and not a kernel difference.
    """
    et, kp, om, km = medium("B/7MHz")
    lam = np.linspace(0.05, 3.0 * abs(km), 200).astype(np.complex128)
    rho, h = 3.0, 0.7
    mine = below._integrand_six_below(lam, rho, h, kp, km)
    # `_integrand_six(lam, rho, h, k1, k2)` -> `_d12(lam, k1, k2)`, so
    # (k1, k2) = (k_p, k_m) is the same swapped kernel pair.
    theirs = som._integrand_six(lam, rho, h, kp, km, "J")
    for i in range(6):
        if i == 2:
            assert np.allclose(mine[i], -theirs[i], rtol=1e-14, atol=0)
            assert np.max(np.abs(mine[i])) > 0.0
        else:
            assert np.allclose(mine[i], theirs[i], rtol=1e-14, atol=0)


# ======================================================================
# G-U2-2 — the ±=+ path keeps its bytes
# ======================================================================


def test_gu2_2_the_below_family_never_enters_six_integrals(monkeypatch):
    """A spy, not an argument: `_six_integrals` and its batch twin carry
    every fig-13/fig-14 landmark that assumes a REAL decay-medium k, and
    the below family must not reach one of them."""
    calls = []
    monkeypatch.setattr(
        som,
        "_six_integrals",
        lambda *a, **k: calls.append(("six", a)) or np.zeros(6, dtype=complex),
    )
    monkeypatch.setattr(
        som,
        "_six_integrals_batch",
        lambda *a, **k: calls.append(("batch", a)) or np.zeros((1, 6), dtype=complex),
    )
    et, kp, om, km = medium("A/7MHz")
    lam_m = 2.0 * np.pi / abs(km)
    below.iv_surfaces_direct_below(
        et,
        kp,
        np.array([0.0, 0.3 * lam_m, 1.1 * lam_m]),
        np.radians(np.array([10.0, 45.0, 80.0])),
        rtol=1e-9,
        omega=om,
    )
    assert calls == []


def test_gu2_2_the_above_family_still_answers_unchanged():
    """The ±=+ evaluator's own numbers, next to the below call that shares
    its `_gamma`/`_d12`. Anything this file did to those two shows up here
    before it shows up in the shipped suite."""
    k2 = 2.0 * np.pi
    eps = 10.0 - 3.6j
    a = som.iv_surfaces_direct(eps, k2, 0.4, np.radians(35.0), rtol=1e-9)
    below.iv_surfaces_direct_below(eps, k2, 0.4, np.radians(35.0), rtol=1e-9)
    b = som.iv_surfaces_direct(eps, k2, 0.4, np.radians(35.0), rtol=1e-9)
    for k in KEYS:
        assert complex(a[k]) == complex(b[k])


# ======================================================================
# G-U2-3 — the R1 -> 0 limits
# ======================================================================


@pytest.mark.parametrize(
    "eps_t",
    [
        complex(13.0, -12.839),
        complex(20.0, -77.036),
        complex(5.0, -2.568),
        complex(80.0, 0.0),
        complex(3.0, -1.2),
    ],
)
@pytest.mark.parametrize("k2", [0.14671, 0.44013, 2.0 * np.pi])
def test_gu2_3_the_generalized_limit_reproduces_eqs_169_172(eps_t, k2):
    """The one gate that makes the ±=− groupings trustworthy.

    `_limits_r1_zero_shared` is written once for both regimes in terms of a
    shared medium s and an other medium o. Read at (s, o) = (k₂, k₁) with
    the ±=+ z-derivative sign it must BE eqs 169–172 — the shipped
    `_limits_r1_zero`, whose `c2`/`c3` groupings are the ±=+ spellings of
    the same static coefficients. If a below-side sign were wrong, it would
    have to be wrong here too.
    """
    om = k2 * C0
    th = np.linspace(0.0, 0.5 * np.pi, 61)
    ref = som._limits_r1_zero(eps_t, k2, th, om, MU0)
    k1 = below.k_medium(eps_t, k2)
    got = below._limits_r1_zero_shared(k2, k1, +1.0, th, om, MU0)
    for k in KEYS:
        assert rel(got[k], ref[k]) < 1e-14


def test_gu2_3_the_below_limit_is_not_the_above_limit():
    """The swap is not a relabelling: A_m = −C₂ and the ∂/∂z sign both
    turn over, so I_ρ^V flips outright and the H rows change magnitude."""
    et, kp, om, km = medium("A/7MHz")
    th = np.radians(np.array([5.0, 30.0, 60.0, 85.0]))
    ab = som._limits_r1_zero(et, kp, th, om, MU0)
    be = below._limits_r1_zero_below(et, kp, th, om, MU0)
    for k in KEYS:
        assert rel(be[k], ab[k]) > 0.5


# MEASURED: the analytic limit against the prototype's DIRECT evaluation at
# R1 = 1e-4 lambda_m reads 3.90e-4 ... 4.69e-4 over the six SPEC cells. That
# residual is not an error, it is the O(R1) term a zero-radius limit drops:
# |k_m|*R1 = 2*pi*1e-4 = 6.28e-4 there, and every cell sits at 0.62-0.75 of
# that first neglected term. Pinned at 2e-3, ~4x the worst.
G_U2_3_TOL = 2e-3


@pytest.mark.parametrize("cell", sorted(gold.LIMITS))
def test_gu2_3_limits_land_on_the_prototype_at_small_r1(cell, record_property):
    """Continuity AND an oracle in one: the analytic limits must equal the
    PROTOTYPE's direct evaluation at R₁ = 1e-4 λ_m. `_sommerfeld`'s own
    :256 pattern gates continuity against momwire's evaluator; this gates
    it against a different one, so a shared small-R₁ bug cannot hide."""
    et, kp, om, km = medium(cell)
    rows = gold.LIMITS[cell]
    th = np.radians(np.array([t for t, *_ in rows]))
    lim = below._limits_r1_zero_below(et, kp, th, om, MU0)
    worst = 0.0
    for i, row in enumerate(rows):
        ref = dict(zip(KEYS, row[1:]))
        scale = max(abs(v) for v in ref.values())
        for k in KEYS:
            worst = max(worst, abs(lim[k][i] - ref[k]) / scale)
    record_property("worst_rel", float(worst))
    assert worst < G_U2_3_TOL, f"{cell}: worst rel {worst:.3e}"


def test_gu2_3_r1_zero_row_comes_from_the_limits():
    """`iv_surfaces_direct_below` at R₁ = 0 must be the analytic limit and
    not a quadrature attempt (there is no contour at R₁ = 0)."""
    et, kp, om, km = medium("C/7MHz")
    th = np.radians(np.array([0.0, 12.0, 47.0, 90.0]))
    got = below.iv_surfaces_direct_below(et, kp, np.zeros_like(th), th, omega=om)
    ref = below._limits_r1_zero_below(et, kp, th, om, MU0)
    for k in KEYS:
        assert np.array_equal(got[k], ref[k])


def test_gu2_3_the_iphi_identity_holds_on_axis():
    """I_ρ^H(θ = 90°) = −I_φ^H, the identity the ρ → 0 azimuth degeneracy
    in both projections leans on. At ρ = 0 the two integrands coincide
    term for term, so this is exact and not a tolerance."""
    et, kp, om, km = medium("A/7MHz")
    lam_m = 2.0 * np.pi / abs(km)
    for r1 in (0.0, 0.05 * lam_m, 0.9 * lam_m):
        s = below.iv_surfaces_direct_below(et, kp, r1, 0.5 * np.pi, omega=om)
        assert abs(complex(s["IrhoH"]) + complex(s["IphiH"])) <= 1e-12 * abs(
            complex(s["IrhoH"])
        )


# ======================================================================
# G-U2-6 — the free-space short circuit and the sigma ladder
# ======================================================================


def test_gu2_6_eps_one_is_exactly_zero():
    """ε̃ = 1: D₁ = D₂ = 0 identically, so the six integrals are EXACTLY
    zero — not small. Phase 0 measured what happens without the
    short-circuit: 1164 s instead of 0.4 s, the tail chasing ulp noise
    with a purely relative convergence test that can never trip."""
    h = below.Health()
    six = below._six_integrals_below(1.0, 2.0 * np.pi, 0.3, 0.2, health=h)
    assert np.count_nonzero(six) == 0
    assert h.zero_kernel == 1
    surf = below.iv_surfaces_direct_below(
        1.0, 2.0 * np.pi, np.array([0.3, 1.0]), np.radians([20.0, 70.0])
    )
    for k in KEYS:
        assert np.count_nonzero(surf[k]) == 0


@pytest.mark.slow
def test_gu2_6_sigma_ladder_extinguishes_the_total_in_medium_field():
    """The G3 lineage, read below the interface: drive σ up and the TOTAL
    below/below field at a fixed in-medium point must fall monotonically
    toward zero. This is the whole composition — direct, image and
    remainder — so a sign error in any one of them shows up as a ladder
    that stops falling."""
    f = 7e6
    om = 2.0 * np.pi * f
    kp = om / C0
    obs = np.array([2.0, 0.0, -0.4])
    src = np.array([0.0, 0.0, -0.2])
    p_hat = np.array([1.0, 0.0, 0.0])
    mags = []
    for sigma in (0.005, 0.05, 0.5, 5.0, 50.0):
        et = eps_tilde(13.0, sigma, f)
        tot = compose(et, kp, om, obs, src, p_hat, DirectSurfaces(et, kp, om))
        mags.append(float(np.max(np.abs(tot))))
    assert all(b < a for a, b in zip(mags, mags[1:])), mags
    assert mags[-1] < 1e-3 * mags[0]


# ======================================================================
# G-U2-7 — reciprocity and azimuthal covariance of the composed field
# ======================================================================


@pytest.mark.slow
def test_gu2_7_the_composed_field_is_reciprocal():
    """Swap source and observer and project on the other's orientation:
    t_a·E(b→a) = t_b·E(a→b). Cheap, and it catches asymmetric plumbing —
    an h built from the wrong endpoint, a dyad row transposed."""
    et, kp, om, km = medium("A/7MHz")
    surf = DirectSurfaces(et, kp, om)
    a = np.array([1.5, 0.7, -0.3])
    b = np.array([-2.0, 1.1, -0.9])
    ta = np.array([0.6, -0.8, 0.0])
    tb = np.array([0.2, 0.3, 0.93])
    tb = tb / np.linalg.norm(tb)
    fwd = np.dot(ta, compose(et, kp, om, a, b, tb, surf))
    rev = np.dot(tb, compose(et, kp, om, b, a, ta, surf))
    assert abs(fwd - rev) <= 1e-9 * max(abs(fwd), abs(rev))


@pytest.mark.slow
def test_gu2_7_the_remainder_rotates_with_the_geometry():
    """Rotate source, observer and orientation about ẑ and the field must
    rotate with them. The SPEC grids all sit at y = 0 where sinφ = 0, so
    I_φ^H never enters them — this is the gate that exercises it."""
    et, kp, om, km = medium("B/7MHz")
    surf = DirectSurfaces(et, kp, om)
    obs = np.array([3.0, 0.0, -0.5])
    src = np.array([0.0, 0.0, -0.2])
    p = np.array([1.0, 0.0, 0.0])
    e0 = below.remainder_field_below(
        obs[None, :], src[None, :], p[None, :].astype(complex), GROUND_Z, kp, km, surf
    )[0]
    c, s = np.cos(0.7), np.sin(0.7)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    e1 = below.remainder_field_below(
        (rot @ obs)[None, :],
        (rot @ src)[None, :],
        (rot @ p)[None, :].astype(complex),
        GROUND_Z,
        kp,
        km,
        surf,
    )[0]
    assert rel(e1, rot @ e0) < 1e-9


@pytest.mark.slow
def test_gu2_7_the_two_projection_spellings_are_one_algebra():
    """`remainder_field_proj_below` is the tangent projection of
    `remainder_field_below` — the `_field_point` discipline, gated rather
    than promised."""
    et, kp, om, km = medium("A/7MHz")
    surf = DirectSurfaces(et, kp, om)
    obs = np.array([[2.0, 0.5, -0.3], [4.0, -1.0, -0.8]])
    src = np.array([[0.0, 0.0, -0.15], [1.0, 1.0, -0.6]])
    t_obs = np.array([[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]])
    t_src = np.array([[0.0, 0.0, 1.0], [0.7, 0.7, 0.14]])
    proj = below.remainder_field_proj_below(
        obs, t_obs, src, t_src, GROUND_Z, kp, km, surf
    )
    for j in range(src.shape[0]):
        vec = below.remainder_field_below(
            obs,
            src[j : j + 1],
            t_src[j : j + 1].astype(complex),
            GROUND_Z,
            kp,
            km,
            surf,
        )
        for i in range(obs.shape[0]):
            assert abs(proj[i, j] - np.dot(t_obs[i], vec[i])) <= 1e-12 * abs(proj[i, j])


# ======================================================================
# G-U2-8 — the refusals
# ======================================================================


def test_gu2_8_a_wrong_side_endpoint_raises_by_name():
    """Never the silent θ-clamp. A point at or above the interface is not
    a below/below geometry, and absorbing it into a plausible θ is how a
    contact case becomes a confident wrong answer."""
    et, kp, om, km = medium("A/7MHz")
    surf = DirectSurfaces(et, kp, om)
    good = np.array([[1.0, 0.0, -0.4]])
    for bad in (np.array([[1.0, 0.0, 0.0]]), np.array([[1.0, 0.0, +0.4]])):
        with pytest.raises(ValueError, match="strictly below"):
            below.remainder_field_below(
                bad, good, np.array([[1.0, 0.0, 0.0 + 0j]]), GROUND_Z, kp, km, surf
            )
        with pytest.raises(ValueError, match="strictly below"):
            below.remainder_field_below(
                good, bad, np.array([[1.0, 0.0, 0.0 + 0j]]), GROUND_Z, kp, km, surf
            )


def test_gu2_8_an_above_grid_is_refused_by_the_below_projection():
    """The two families tabulate different surfaces over different
    divide-outs; handing one to the other's projection is a silent decade,
    so it is a named error instead."""
    grid = som.get_grid(complex(13.0, -12.8), 2.0 * np.pi, 1.0, 2.0 * np.pi * C0)
    with pytest.raises(ValueError, match="SommerfeldGridBelow"):
        below.remainder_field_below(
            np.array([[1.0, 0.0, -0.4]]),
            np.array([[0.0, 0.0, -0.2]]),
            np.array([[1.0, 0.0, 0.0 + 0j]]),
            GROUND_Z,
            2.0 * np.pi,
            2.0 * np.pi * 3.6,
            grid,
        )


def test_gu2_8_the_below_grid_refuses_a_normalized_rescale():
    """No frequency-axis reuse: the ±=+ family's rescaling rests on a
    measured linear k₂-scaling and a λ-proportional lattice, and this unit
    measured neither below."""
    et, kp, om, km = medium("C/21MHz")
    grid = below.SommerfeldGridBelow(et, kp, 0.4 * (2 * np.pi / abs(km)), omega=om)
    with pytest.raises(NotImplementedError, match="normalized-master"):
        grid.scaled_to(kp * 2, om * 2, MU0)


def test_gu2_8_the_two_regimes_do_not_collide_in_the_grid_cache():
    """One cache, one FIFO bound, two entries: the key carries a regime
    discriminator so an above grid and a below grid at IDENTICAL (ε̃, k₂,
    r1, ω, μ) are two tables and not one."""
    et = complex(5.0, -2.568)
    k2 = 2.0 * np.pi
    om = k2 * C0
    r1 = 0.5
    som._GRID_CACHE.clear()
    som._NORM_CACHE.clear()
    a = som.get_grid(et, k2, r1, om)
    b = below.get_grid_below(et, k2, r1, om)
    assert a is not b
    assert type(a) is som.SommerfeldGrid
    assert type(b) is below.SommerfeldGridBelow
    assert som.get_grid(et, k2, r1, om) is a
    assert below.get_grid_below(et, k2, r1, om) is b
    keys = list(som._GRID_CACHE)
    assert len(keys) == 2
    assert {k[0] for k in keys} == {"above", "below"}
    som._GRID_CACHE.clear()
    som._NORM_CACHE.clear()


# ======================================================================
# G-U2-5 — the composed field against empymod, and the eight-way sign scan
# ======================================================================
#
# MEASURED (170 M-line points, 17 SPEC cells): the winning combination
# reads 3.16e-04 worst over the 50 points whose grid's OWN oracle spread is
# <= 1e-3, and 3.66e-03 over all grids — every one of the latter having a
# larger recorded spread than our disagreement. Those are, to three digits,
# the phase-0 prototype's own G4c numbers (3.160e-04 / 3.663e-03) reached
# through a completely separate implementation. The runner-up sign
# combination reads 1.76e+00, a margin of 481x. Tolerances below are named
# against the oracle, not against us: the well-conditioned bar is the
# phase-0 gate's 1e-3, and each grid is additionally allowed its own
# recorded spread.

WINNER = ("mirror", +1.0, +1.0)
SIGN_SCAN = [
    (dyad, sa, s)
    for dyad in ("mirror", "literal")
    for sa in (+1.0, -1.0)
    for s in (+1.0, -1.0)
]
WELL_CONDITIONED_SPREAD = 1e-3
G_U2_5_TOL_WELL = 1e-3
G_U2_5_LOSER_FLOOR = 0.5


@pytest.fixture(scope="module")
def mline_scan():
    """Compose every M-line point once, scoring all eight sign combinations.

    The remainder is evaluated ONCE per point and reused across the eight —
    the scan varies only how the three terms are combined, which is exactly
    the thing under test.
    """
    rows = []
    for entry in gold.MLINE:
        f = entry["freq_hz"]
        om = 2.0 * np.pi * f
        kp = om / C0
        et = eps_tilde(entry["eps_r"], entry["sigma"], f)
        km = below.k_medium(et, kp)
        surf = DirectSurfaces(et, kp, om)
        src = np.asarray(entry["src_xyz"], dtype=float)
        p_hat = (
            np.array([1.0, 0.0, 0.0])
            if entry["kind"] == "HED"
            else np.array([0.0, 0.0, 1.0])
        )
        c1 = -1j * om * MU0 / (4.0 * np.pi)
        for pt, e_ref in zip(entry["points"], entry["E"]):
            obs = np.asarray(pt, dtype=float)
            e_dir = fs_field(c1, km, p_hat, obs - src)
            e_rem = below.remainder_field_below(
                obs[None, :],
                src[None, :],
                p_hat[None, :].astype(complex),
                GROUND_Z,
                kp,
                km,
                surf,
            )[0]
            a_m = below.image_coefficient_below(et)
            imgs = {
                d: image_field(c1, km, p_hat, obs, src, dyad=d)
                for d in ("mirror", "literal")
            }
            errs = {
                combo: rel(
                    e_dir + combo[1] * a_m * imgs[combo[0]] + combo[2] * e_rem,
                    np.asarray(e_ref),
                )
                for combo in SIGN_SCAN
            }
            rows.append(
                {
                    "id": entry["id"],
                    "spread": entry["oracle_spread"],
                    "errs": errs,
                    "rem_share": float(np.max(np.abs(e_rem)))
                    / max(float(np.max(np.abs(e_dir))), 1e-300),
                }
            )
    return rows


@pytest.mark.slow
def test_gu2_5_composed_field_vs_empymod(mline_scan, record_property):
    """The whole below/below answer — direct(k_m) + A_m·image + remainder —
    against the banked quad oracle at all 170 M-line points."""
    worst_all = max(r["errs"][WINNER] for r in mline_scan)
    well = [r for r in mline_scan if r["spread"] <= WELL_CONDITIONED_SPREAD]
    worst_well = max(r["errs"][WINNER] for r in well)
    record_property("worst_all_grids", float(worst_all))
    record_property("worst_well_conditioned", float(worst_well))
    record_property("n_well_conditioned", int(len(well)))
    assert well, "no well-conditioned M-line grid in the goldens"
    assert worst_well < G_U2_5_TOL_WELL, f"well-conditioned worst {worst_well:.3e}"
    # Every remaining point is allowed its own grid's recorded oracle spread
    # (a factor 3 of headroom over it, which the phase-0 table shows we sit
    # at 0.25-0.42x of) and nothing more.
    for r in mline_scan:
        assert r["errs"][WINNER] < max(3.0 * r["spread"], G_U2_5_TOL_WELL), (
            f"{r['id']}: {r['errs'][WINNER]:.3e} vs oracle spread {r['spread']:.3e}"
        )


@pytest.mark.slow
def test_gu2_5_the_sign_scan_winner_is_the_measured_one(mline_scan, record_property):
    """All eight combinations scored; the measured one must win and every
    loser must fail by O(1).

    The second assertion is the one that keeps this honest. If the
    remainder ever went quiet — a divide-out mismatch, a grid that froze,
    a projection that returned zeros — the losers would stop failing and
    this test would notice before any envelope did.
    """
    scored = sorted(
        ((max(r["errs"][c] for r in mline_scan), c) for c in SIGN_SCAN),
        key=lambda t: t[0],
    )
    record_property("scan", [(f"{v:.3e}", c) for v, c in scored])
    assert scored[0][1] == WINNER, f"winner is {scored[0][1]}, not {WINNER}"
    assert scored[1][0] > 100.0 * scored[0][0], (
        f"winner {scored[0][0]:.3e} vs runner-up {scored[1][0]:.3e}: the "
        "margin has collapsed"
    )
    for value, combo in scored[1:]:
        assert value > G_U2_5_LOSER_FLOOR, (
            f"{combo} only fails by {value:.3e} — a loser that stopped "
            "failing means the remainder went quiet, which is the real bug"
        )


@pytest.mark.slow
def test_gu2_5_the_remainder_is_not_a_correction(mline_scan):
    """Phase 0's headline, re-measured here: on the M-line the remainder
    runs from a tenth of the direct term to more than twice it. Everything
    about this family's tolerances follows from that."""
    shares = [r["rem_share"] for r in mline_scan]
    assert max(shares) > 1.0, max(shares)


# ======================================================================
# G-U2-4 — the grid, three ways: grid -> direct -> committed goldens
# ======================================================================


@pytest.fixture(scope="module")
def small_below_grid():
    """A deliberately small below grid (soil A, 7 MHz) so the numpy fill
    stays inside a test's patience. The layout decisions it exercises are
    the shipped ones; only `r1_max` is small."""
    et, kp, om, km = medium("A/7MHz")
    lam_m = 2.0 * np.pi / abs(km)
    health = below.Health()
    grid = below.SommerfeldGridBelow(
        et, kp, 0.6 * lam_m, rtol=1e-9, omega=om, health=health
    )
    return grid, health, (et, kp, om, km, lam_m)


# The R1 x theta bands the full-rectangle probe reports separately. A single
# pooled worst hides which corner is weak, and this family's weak corner moved
# once the domain was probed properly: at a 4 lambda_m cap the [2, 4) annulus
# read 2.4e-3 against 1.6e-4 everywhere inside it, which is what drove the cap
# down to 2 lambda_m (see `_SOMM_BELOW_R1_CAP_LAMBDA_M`).
_R_BANDS = ((0.0, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0))
_TH_BANDS = ((1.0, 2.0), (2.0, 5.0), (5.0, 30.0), (30.0, 90.0))
_RECT_FLOOR_REL = 1e-10
# theta in [1, 5] deg at R1 in [0.5, 2] lambda_m: the geometry the product
# brings, a 15 cm-deep radial pair read at rho 4-20 m. Graded on its own,
# because it must never be inside the floor's skip set and it is the band
# G-U2-9 then composes at.
_PRODUCT_BANDS = frozenset(
    (rb, tb) for rb in _R_BANDS for tb in _TH_BANDS if rb[0] >= 0.5 and tb[1] <= 5.0
)


def _rect_probe(grid, et, kp, om, lam_m, per_band, rng):
    """grid -> direct over the FULL served rectangle, PER SURFACE, per band.

    The metric is |grid_k - direct_k| / |direct_k| for each surface
    separately, NOT the ±=+ house convention |grid_k - direct_k| divided by
    the largest of the four at that point. Below the interface the four
    surfaces span decades at a single point and the remainder is not a small
    correction, so a surface that is small at a point still carries its own
    share of the composed field; of-family normalisation would grade it
    against a neighbour. (Measured, the two differ by ~2.5x here, not by
    decades — but the stricter one is the one that means something.)

    NAMED FLOOR: a surface is skipped at a point when |direct_k| falls below
    `_RECT_FLOOR_REL` times that surface's own maximum over the band, i.e.
    when it is a numerical zero and a relative error is meaningless. The
    skipped fraction is returned so it can be seen to be small, and the
    product band is reported separately so it can be seen NOT to be in the
    skip set.

    Boundaries are FORCED into the sample: theta at the grazing floor
    exactly, theta = 90, R1 = 0 (the analytic-limit row) and R1 = r1_max.
    """
    th_min_deg = np.degrees(grid.th_min)
    out = {}
    skipped = total = 0
    for rb in _R_BANDS:
        rlo, rhi = rb[0] * lam_m, min(rb[1] * lam_m, grid.r1_max)
        if rhi <= rlo:
            continue
        for tb in _TH_BANDS:
            tlo = max(tb[0], th_min_deg)
            rq = np.concatenate([[rlo, rhi], rng.uniform(rlo, rhi, per_band)])
            tq = np.radians(
                np.concatenate([[tlo, tb[1]], rng.uniform(tlo, tb[1], per_band)])
            )
            rq, tq = np.meshgrid(rq, tq, indexing="ij")
            rq = np.clip(rq.ravel(), 0.0, grid.r1_max)
            tq = tq.ravel()
            got = grid.eval(rq, tq)
            ref = below.iv_surfaces_direct_below(et, kp, rq, tq, rtol=1e-9, omega=om)
            worst = 0.0
            for k in KEYS:
                d = np.abs(ref[k])
                live = d >= _RECT_FLOOR_REL * d.max()
                total += d.size
                skipped += int((~live).sum())
                if not live.any():
                    continue
                worst = max(
                    worst, float(np.max(np.abs(got[k] - ref[k])[live] / d[live]))
                )
            out[(rb, tb)] = worst
    return out, skipped, total


@pytest.fixture(scope="module")
def full_below_grid(request):
    """A below grid at the FULL cap, so the probe covers what the grid
    serves rather than a comfortable interior patch."""
    cell = request.param
    et, kp, om, km = medium(cell)
    lam_m = 2.0 * np.pi / abs(km)
    health = below.Health()
    grid = below.SommerfeldGridBelow(
        et,
        kp,
        below._SOMM_BELOW_R1_CAP_LAMBDA_M * lam_m,
        rtol=1e-9,
        omega=om,
        health=health,
    )
    return cell, grid, health, (et, kp, om, km, lam_m)


@pytest.mark.slow
@pytest.mark.parametrize(
    "full_below_grid", ["B/7MHz", "A/21MHz", "C/21MHz"], indirect=True
)
def test_gu2_4_grid_matches_direct_over_the_whole_rectangle(
    full_below_grid, record_property
):
    """grid -> direct over R1 in [0, r1_max] x theta in [th_min, 90], edges
    included, PER SURFACE, on the three stressor cells.

    This gate replaces an interior probe that ran to 0.6 lambda_m on one
    cell under of-family normalisation. Two things were wrong with that: it
    did not cover the served domain, and it graded each surface against its
    largest neighbour. Both are fixed here.

    Two pins, because the domain is not uniform in what it owes anyone. The
    PRODUCT band — deep grazing at working range, where G-U2-9 then composes
    — is held to the same 1e-3 as the composition itself. The rest of the
    rectangle, whose worst cell is the steep band at R1 ~ 0.5-1 lambda_m, is
    held to 2e-3. Densifying that corner was measured and REJECTED: theta
    2.5 -> 1.25 deg costs +44 % nodes and moves the grand worst from 5.27e-4
    only to 4.29e-4, making one cell (C/21) worse. The residual is not
    theta-resolution any more.
    """
    cell, grid, health, (et, kp, om, km, lam_m) = full_below_grid
    bands, skipped, total = _rect_probe(
        grid, et, kp, om, lam_m, 3, np.random.default_rng(553)
    )
    for (rb, tb), w in sorted(bands.items()):
        record_property(f"R1[{rb[0]},{rb[1]})_th[{tb[0]},{tb[1]})", float(w))
    worst = max(bands.values())
    product = max(v for k, v in bands.items() if k in _PRODUCT_BANDS)
    record_property("worst_per_surface", float(worst))
    record_property("worst_product_band", float(product))
    record_property("floor_skipped_frac", float(skipped) / float(total))
    record_property("nodes", int(sum(r["n_r"] * r["n_th"] for r in grid._regions)))
    record_property("health", repr(health.as_dict()))
    assert health.nonconvergent == 0, health.as_dict()
    # The floor must stay a rounding detail, and must never swallow the
    # product band — every product band reports a live number above.
    assert skipped < 0.02 * total, f"{skipped}/{total} skipped by the floor"
    assert product < G_U2_4_PRODUCT_TOL, f"{cell}: product band {product:.3e}"
    assert worst < G_U2_4_TOL, f"{cell}: full-rectangle worst {worst:.3e}"


@pytest.mark.slow
def test_gu2_4_grid_matches_direct_and_the_goldens(small_below_grid, record_property):
    """The three-way. A grid checked only against a direct evaluator proves
    the two agree; `_sommerfeld`'s #161 addendum is the standing reminder
    that they can agree on a mis-branched contour. So: grid vs direct, AND
    direct vs the committed prototype goldens at the same points.
    """
    grid, health, (et, kp, om, km, lam_m) = small_below_grid
    rng = np.random.default_rng(553)
    n = 40
    r1 = rng.uniform(0.02 * lam_m, 0.58 * lam_m, n)
    th = rng.uniform(np.radians(2.0), 0.5 * np.pi, n)
    got = grid.eval(r1, th)
    ref = below.iv_surfaces_direct_below(et, kp, r1, th, rtol=1e-9, omega=om)
    scale = np.max(np.abs(np.stack([ref[k] for k in KEYS])), axis=0)
    worst_grid = max(float(np.max(np.abs(got[k] - ref[k]) / scale)) for k in KEYS)
    record_property("grid_vs_direct", float(worst_grid))

    # ... and the same direct evaluator against the OTHER implementation.
    rows = [r for r in gold.SURFACES["A/7MHz"] if r[0] * lam_m <= 0.58 * lam_m]
    assert rows, "no golden row inside the small grid"
    gr1 = np.array([r[0] * lam_m for r in rows])
    gth = np.radians(np.array([r[1] for r in rows]))
    gdir = below.iv_surfaces_direct_below(et, kp, gr1, gth, rtol=1e-9, omega=om)
    worst_gold = 0.0
    for i, row in enumerate(rows):
        refg = dict(zip(KEYS, row[2:]))
        sc = max(abs(v) for v in refg.values())
        for k in KEYS:
            worst_gold = max(worst_gold, abs(gdir[k][i] - refg[k]) / sc)
    record_property("direct_vs_goldens", float(worst_gold))
    record_property("health", repr(health.as_dict()))

    assert worst_gold < G_U2_1_TOL, f"direct vs goldens {worst_gold:.3e}"
    assert worst_grid < G_U2_4_TOL, f"grid vs direct {worst_grid:.3e}"
    assert health.nonconvergent == 0, health.as_dict()


# MEASURED over the FULL served rectangle, PER SURFACE (|g_k - d_k|/|d_k|)
# -- R1 in [0, 2 lambda_m] x theta in [1, 90] deg, boundaries forced into
# the sample, all six SPEC cells, 23,520 surface-points. Worst per cell:
#
#   A/7 1.64e-4   A/21 2.17e-4   B/7 5.27e-4
#   B/21 2.71e-4  C/7  1.83e-4   C/21 2.62e-4
#
# Grand worst 5.27e-4, in R1 [0.5, 1.0) x theta [30, 90). PRODUCT band worst
# 2.71e-4. Floor skipped 210/23520 = 0.89 %, none of it in a product band.
#
# The of-family metric this gate used to carry reads 2.14e-4 on the same
# lattice, so per-surface costs a factor 2.5 -- not the decades an of-family
# convention can hide when a family spans decades, but it is the honest one
# and it is what is pinned.
#
# Densification was measured and rejected: steep band 2.5 -> 1.25 deg is
# +44 % nodes (3520 -> 5056) and +30 % fill for grand worst 5.27e-4 ->
# 4.29e-4, with C/21 getting WORSE (2.62e-4 -> 4.29e-4). The residual is no
# longer theta-resolution-limited, so more theta nodes buy nothing.
G_U2_4_PRODUCT_TOL = 1e-3
G_U2_4_TOL = 2e-3


@pytest.mark.slow
def test_gu2_4_the_r1_zero_row_and_the_boundary(small_below_grid):
    """The two lattice edges a query can actually land on: R₁ = 0 (served
    by the analytic limits) and R₁ = r1_max exactly (served, not refused)."""
    grid, _, (et, kp, om, km, lam_m) = small_below_grid
    th = np.radians(np.array([2.0, 20.0, 55.0, 90.0]))
    at_zero = grid.eval(np.zeros_like(th), th)
    lim = below._limits_r1_zero_below(et, kp, th, om, MU0)
    for k in KEYS:
        assert rel(at_zero[k], lim[k]) < 5e-2
    edge = grid.eval(np.full(4, grid.r1_max), th)
    ref = below.iv_surfaces_direct_below(
        et, kp, np.full(4, grid.r1_max), th, rtol=1e-9, omega=om
    )
    scale = np.max(np.abs(np.stack([ref[k] for k in KEYS])), axis=0)
    for k in KEYS:
        assert float(np.max(np.abs(edge[k] - ref[k]) / scale)) < G_U2_4_TOL


@pytest.mark.slow
def test_gu2_4_past_the_cap_refuses_instead_of_clamping(small_below_grid):
    """The ±=+ grid clamps R₁ to r1_max because out there its remainder is
    a negligible tail whose amplitude may as well freeze. Below, this unit
    measured |remainder|/|direct+image| RISING through 1.2 at 1 λ_m, 29 at
    2 λ_m and 6.4e3 at 4 λ_m on soil A — past the tabulation the remainder
    IS the field, so a frozen amplitude would be a fabrication. The cap
    itself sits at 2 λ_m because that is where the tabulation was measured
    accurate, not because the physics stops there."""
    grid, _, (et, kp, om, km, lam_m) = small_below_grid
    with pytest.raises(ValueError, match="past the tabulation"):
        grid.eval(np.array([grid.r1_max * 1.05]), np.array([0.6]))
    with pytest.raises(ValueError, match="non-negative"):
        grid.eval(np.array([-0.1]), np.array([0.6]))


@pytest.mark.slow
def test_gu2_4_below_the_grazing_floor_refuses(small_below_grid):
    """The lattice stops at a measured grazing floor and says so.

    Between θ = 5° and θ = 0.05° at R₁ = 3.5 λ_m the surfaces move by 2.4
    to 3.3 times their own scale, and the drift grows ~linearly in log θ —
    a lateral-wave logarithm that no uniform lattice resolves and that has
    no θ = 0 node at all (h = 0 kills the tail's exponential decay, leaving
    a conditionally convergent integral this contour does not evaluate).
    So the floor is a domain edge, named."""
    grid, _, _ = small_below_grid
    with pytest.raises(ValueError, match="grazing floor"):
        grid.eval(np.array([0.5 * grid.r1_max]), np.array([np.radians(0.2)]))


@pytest.mark.slow
def test_gu2_4_the_projection_reads_the_grid_and_the_direct_alike(
    small_below_grid,
):
    """The last link: the same dyad algebra driven off the tabulation and
    off the exact surfaces must agree to the grid's own interpolation bar.
    If it does not, the divide-out in `remainder_field_proj_below` and the
    one in `iv_surfaces_direct_below` have drifted apart."""
    grid, _, (et, kp, om, km, lam_m) = small_below_grid
    obs = np.array([[1.2, 0.4, -0.3], [2.5, -0.9, -0.7]])
    src = np.array([[0.0, 0.0, -0.2], [0.6, 0.2, -0.5]])
    t_obs = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t_src = np.array([[0.0, 0.0, 1.0], [0.8, 0.6, 0.0]])
    a = below.remainder_field_proj_below(obs, t_obs, src, t_src, GROUND_Z, kp, km, grid)
    b = below.remainder_field_proj_below(
        obs, t_obs, src, t_src, GROUND_Z, kp, km, DirectSurfaces(et, kp, om)
    )
    assert rel(a, b) < G_U2_4_TOL


# ======================================================================
# G-U2-9 — the composed field at DEEP GRAZING, grid against direct
# ======================================================================
#
# The class G-U2-5's M-line does not reach. The SPEC M-line sits at
# theta = atan2(0.52..0.65, 1..10) = 3..33 deg and R1 <= ~2 lambda_m only on
# the highest-loss soil; the geometry the product actually brings — a pair of
# 15 cm-deep radials, hh = 0.30 m, read out to tens of metres — runs to
# theta ~ 1-2 deg at R1 ~ 1-2 lambda_m, where the remainder is not merely
# comparable to direct+image but many times it. A tabulation error there is a
# TOTAL-FIELD error at the same size, which is why this gate scores against
# the composed total and not against surface scale.
#
# Both sides compose identically and differ only in where the remainder's
# surfaces come from: the shipped grid, or `iv_surfaces_direct_below`. So the
# number below is tabulation accuracy expressed in the units the caller cares
# about.

G_U2_9_TOL = 1e-3


@pytest.mark.slow
@pytest.mark.parametrize("cell", ["A/7MHz", "B/7MHz", "C/21MHz"])
def test_gu2_9_deep_grazing_composed_field_grid_vs_direct(cell, record_property):
    """A buried-radial pair at 15 cm, read from 4 m out to wherever the
    grid's own floor or cap stops it."""
    et, kp, om, km = medium(cell)
    lam_m = 2.0 * np.pi / abs(km)
    grid = below.SommerfeldGridBelow(et, kp, 2.0 * lam_m, rtol=1e-9, omega=om)
    direct = DirectSurfaces(et, kp, om)

    hh = 0.30  # two 15 cm depths
    # Stay inside BOTH the grazing floor and the cap — the two named
    # refusals, honoured here rather than tested here (G-U2-4 tests them).
    rho_floor = hh / math.tan(grid.th_min)
    rho_max = min(math.sqrt(max(grid.r1_max**2 - hh * hh, 0.0)), rho_floor)
    assert rho_max > 4.0, f"{cell}: nothing served past 4 m (rho_max {rho_max:.2f})"
    rhos = np.linspace(4.0, rho_max * 0.999, 8)

    worst = 0.0
    worst_at = None
    min_share, max_share = np.inf, 0.0
    for rho in rhos:
        obs = np.array([float(rho), 0.0, -0.15])
        src = np.array([0.0, 0.0, -0.15])
        for p_hat in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])):
            e_grid = compose(et, kp, om, obs, src, p_hat, grid)
            e_dir = compose(et, kp, om, obs, src, p_hat, direct)
            # remainder share, so a quiet remainder cannot make this pass
            c1 = -1j * om * MU0 / (4.0 * np.pi)
            rem = below.remainder_field_below(
                obs[None, :],
                src[None, :],
                p_hat[None, :].astype(complex),
                GROUND_Z,
                kp,
                km,
                direct,
            )[0]
            di = fs_field(c1, km, p_hat, obs - src) + below.image_coefficient_below(
                et
            ) * image_field(c1, km, p_hat, obs, src)
            share = float(np.max(np.abs(rem))) / max(float(np.max(np.abs(di))), 1e-300)
            min_share = min(min_share, share)
            max_share = max(max_share, share)
            e = float(np.max(np.abs(e_grid - e_dir))) / max(
                float(np.max(np.abs(e_dir))), 1e-300
            )
            if e > worst:
                worst, worst_at = e, (float(rho), math.degrees(math.atan2(hh, rho)))
    record_property("worst_of_total_field", float(worst))
    record_property("worst_at_rho_theta", repr(worst_at))
    record_property("min_remainder_share", float(min_share))
    record_property("max_remainder_share", float(max_share))
    record_property("rho_served_m", float(rho_max))
    record_property(
        "theta_deg_at_rho_max", float(math.degrees(math.atan2(hh, rho_max)))
    )
    # The class is only meaningful if the sweep actually REACHES remainder
    # dominance. It does not start there — at the near end (rho = 4 m, R1
    # well under a wavelength) direct+image still lead, and the share climbs
    # with range. Requiring dominance at every point would have been
    # requiring the wrong thing; requiring it nowhere would let a quiet
    # remainder pass.
    assert max_share > 1.0, f"{cell}: remainder never dominates (max {max_share:.3g})"
    assert worst < G_U2_9_TOL, f"{cell}: {worst:.3e} of total field at {worst_at}"
