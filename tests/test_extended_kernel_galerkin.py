"""The extended thin-wire kernel for the Galerkin family (momwire#246).

Unit A is the delta kernel and the seam: `SinusoidalSolver
._folded_ek_delta_fields` computes the EK-minus-reduced FIELD correction by
Gauss-Legendre quadrature of a smooth kernel, and `_field_components_bcast`
dispatches on the EK payload's type so that the folded (`cos_shape="cos-1"`)
and point-matched (`"cos"`) contracts can no longer be silently confused.
Unit B wires it into `SinusoidalGalerkinSolver` — its gates start at G-B1,
about two-thirds of the way down.

  G-A1  the float64 delta against an independent 80-bit, very-high-N
        quadrature of the same integrand, over the (Δ/a, kH, observer) box
        the fill visits — 5e-14 of the const-shape field's size;
  G-A2  reduced + delta against `_extended_kernel_fields` (momwire#233's
        EKSCX port) on the LITERAL shapes, which pins the field-operator
        conventions against a completely separate implementation;
  G-A3  a → 0 collapses the delta to exact zeros, structurally;
  G-A4  the node count is converged (N = 16 against N = 48);
  G-A5  nothing the extended kernel added can run on a path that did not ask
        for it — the reduced and point-matched paths never reach the delta.

Two things in this half changed when unit B put a fill behind it, and both
are recorded here rather than quietly. The probe box now includes observers
INSIDE the source segment, which is where a fill spends its self and
node-sharing pairs and which the original box — one, two and ten segment
lengths away — never visited; and the quadrature runs in the sinh-mapped
variable, without which no node count reaches those probes at all. See
`Z_OVER_DELTA` and `ON_SEGMENT_FLOOR`.
"""

import contextlib
import functools

import numpy as np
import pytest

from momwire.sinusoidal import (
    _EKPairs,
    _N_PANEL_EK_DELTA_NEAR,
    _N_QP_EK_DELTA,
    SinusoidalSolver,
)

C0 = 299792458.0
FREQ = 30e6
WL = C0 / FREQ
K = 2.0 * np.pi * FREQ / C0

# The box the design's E1 experiment swept, and the one the fill visits:
# Δ/a from the thick-wire limit NEC's EK exists for, out to where reduced and
# extended agree; kH from a λ/40 half-length down to λ/4000; the observer at
# one, two and ten segment lengths.
DELTA_OVER_A = (2.0, 6.0, 20.0)
KH = (7.5e-2, 7.5e-3, 7.5e-4)
# The observer's axial offset in SEGMENT lengths. The three outer ones are the
# far pairs; 0, 0.25 and 0.5 put the observer INSIDE the source segment and at
# its endpoint, which is where a fill spends its self and node-sharing pairs
# and where the delta's structure lives — the extended-minus-reduced field is
# O(1) relative to the reduced one within a radius of a segment END. A probe
# box that stops outside the segment says nothing about the pairs that matter
# most, and momwire#246 unit B found that out the hard way.
Z_OVER_DELTA = (0.0, 0.25, 0.5, 1.0, 2.0, 10.0)

# G-A1: how far the float64 delta may sit from the 80-bit reference, in units
# of the const-shape reduced field (the scale at which a field-table error
# reaches G). Measured worst 2.6e-15 over the whole box, both components, all
# three shapes — and that number belongs to the REFERENCE, not to the routine:
# run the 80-bit reference against itself at N = 16 and N = 512 and the two
# differ by 2.6e-15 of the same scale, while the float64 routine sits 1.2e-16
# from the 80-bit answer on its OWN nodes. (Truncation is not in it either
# way: the integrand is entire, and the nearest singularity — the kernel's
# ζ = ±jρ — puts the 16-point rule's Bernstein rate at 1e-20 on the closest
# probe in this box.) Both statements are asserted below.
DELTA_VS_LONGDOUBLE = 5e-14
# The same measurement with the reference on the routine's own 16 nodes, so
# what is left is arithmetic and nothing else. Measured 1.2e-16.
DELTA_ARITHMETIC = 1e-15

# G-A2: how far reduced + delta may sit from the EKSCX port on the literal
# shapes, same scale. This is not a tolerance on the physics — the two are the
# same quantity — it is the size of EKSCX's OWN cancellation noise, which is
# what the comparison can resolve. Measured worst 7.3e-13, on E_ρ of the
# literal cos shape at the finest mesh, where the closed forms differ two
# nearly equal endpoint groups; E_z, which has no such group, lands at 2.4e-15.
DELTA_VS_EKSCX = 3e-12

# G-A4: N = 16 against N = 48. Measured 1.6e-15 — the same size as G-A1's
# number and for the same reason: with truncation 1e-20 away, what separates
# two node counts is the rounding of two different sums, and the longer sum
# carries more of it. (The design's E1 reached 1e-15 with 8-12 nodes.)
NODE_CONVERGENCE = 1e-14

# The ON-SEGMENT tier's own floor, and it is neither the reference's nor the
# rule's: it is reduced-plus-delta's. The delta's whole-line integral very
# nearly vanishes — extended and reduced kernels agree away from the wire — so
# a pair whose observer sits on the source segment sums contributions ~(H/ρ)²
# larger than their total, and float64 leaves ~ε·(H/ρ)² of the peak behind.
# At this box's thinnest probe (Δ/a = 20, i.e. H/ρ = 10) that is ~1e-13 of the
# const field, and every measurement below sits inside it: 6.5e-14 against the
# 80-bit reference, 2.6e-14 on the reference's own nodes, 8.7e-14 between node
# counts, 5.6e-13 against EKSCX. Refining anything moves none of them. The
# FAR probes, which have no such cancellation, keep unit A's original gates —
# which is why the two tiers are asserted separately rather than under one
# widened bar.
ON_SEGMENT_FLOOR = 3e-13
ON_SEGMENT_VS_EKSCX = 3e-12
ON_SEGMENT_NODE_CONVERGENCE = 5e-13


def _solver(**over):
    """A half-wave dipole at 30 MHz — geometry irrelevant here, the solver is
    only a carrier for η, k and the Gauss-Legendre cache."""
    kw = dict(
        wires=[np.array([(0.0, 0.0, -0.25 * WL), (0.0, 0.0, 0.25 * WL)])],
        n_per_edge_per_wire=[[11]],
        nsegs=11,
        wavelength=WL,
        wire_radius=1e-3,
    )
    kw.update(over)
    return SinusoidalSolver(**kw)


def _probes():
    """(H, z, a, n_panels) over the box. On an EK-eligible pair the observer is
    coaxial with the source and carries the same radius, so ρ_eval == a
    exactly.

    `n_panels` is the tier the FILL would use at that probe, so the gates below
    measure the rule the solver actually runs rather than a rule nothing calls.
    An observer inside or on the source segment takes the near tier — and in
    the fill that is not a choice about accuracy but about which path owns the
    pair: coaxial segments whose spans overlap are at separation zero, i.e.
    near pairs, and `SinusoidalGalerkinSolver._apply_near_correction` computes
    them. Everything a segment length away is converged on one panel.
    """
    for da in DELTA_OVER_A:
        for kh in KH:
            H = kh / K
            seg = 2.0 * H
            a = seg / da
            for zf in Z_OVER_DELTA:
                z = zf * seg
                yield H, z, a, (_N_PANEL_EK_DELTA_NEAR if abs(z) <= 1.05 * H else 1)


def _coaxial_tables(sim, H, z, a, **kw):
    """`_field_components_bcast` on one coaxial equal-radius pair: source
    centred at the origin along +z with half-length H, observer on the same
    axis at height z, both of radius a."""
    return sim._field_components_bcast(
        K,
        obs_c=np.array([[[0.0, 0.0, z]]]),
        obs_t=np.array([[[0.0, 0.0, 1.0]]]),
        a=a,
        src_c=np.array([[[0.0, 0.0, 0.0]]]),
        src_t=np.array([[[0.0, 0.0, 1.0]]]),
        src_hh=np.array([[H]]),
        **kw,
    )


def _const_scale(tables):
    """The const-shape field's size — where an ε·‖T_const‖ rounding sits, and
    therefore the scale at which a field-table error reaches G. Both
    components together, since E_ρ vanishes on some of these pairs."""
    return abs(tables["Ez_const"][0, 0]) + abs(tables["Erho_const"][0, 0])


_LD_NODES = {}


def _ld_leggauss(n):
    if n not in _LD_NODES:
        gx, gw = np.polynomial.legendre.leggauss(n)
        _LD_NODES[n] = (gx.astype(np.longdouble), gw.astype(np.longdouble))
    return _LD_NODES[n]


def _ld_delta(eta, H, z, rho, a, shape, n=512, panels=1):
    """(ΔE_z, ΔE_ρ) of one source shape, at 80 bits and N nodes.

    Independent of the production routine in precision, in node count and in
    spelling: the four u-derivatives of the reduced kernel are written out
    from their own algebra (each Aₙ as an explicit polynomial in kR with the
    j's resolved by hand into real and imaginary parts) rather than through
    the reverse-Bessel recursion, and the operator brackets are assembled in
    a different grouping. What it shares with the production routine is the
    SCHEME — the sinh map ζ = ρ·sinh t, without which no node count reaches
    the on-segment probes at all — which is the point: this measures the
    float64 arithmetic, in the idiom of `test_sinusoidal_galerkin.py`'s
    60-digit `cos kξ − 1` reference.
    """
    gx, gw = _ld_leggauss(n)
    if panels > 1:
        edges = np.linspace(np.longdouble(-1), np.longdouble(1), panels + 1)
        mid = 0.5 * (edges[:-1] + edges[1:])
        half = 0.5 * (edges[1:] - edges[:-1])
        gx = (mid[:, None] + half[:, None] * gx[None, :]).ravel()
        gw = (half[:, None] * gw[None, :]).ravel()
    kL = np.longdouble(K)
    HL, zL, rL, aL = (np.longdouble(v) for v in (H, z, rho, a))
    # ζ = ρ·sinh t on t ∈ [asinh((z−H)/ρ), asinh((z+H)/ρ)], so R = ρ·cosh t.
    # Spelled through log1p-free arcsinh and an explicit Jacobian, i.e. the
    # same change of variables assembled differently.
    lo = np.arcsinh((zL - HL) / rL)
    hi = np.arcsinh((zL + HL) / rL)
    t = 0.5 * (hi + lo) + 0.5 * (hi - lo) * gx
    zeta = rL * np.sinh(t)
    xi = zL - zeta
    w = (0.5 * (hi - lo) * gw) * (rL * np.cosh(t))
    R = np.sqrt(rL * rL + zeta * zeta)
    y = kL * R  # kR, real
    e = np.exp(-1j * y) / R  # the reduced kernel itself
    # Aₙ(jkR) with j resolved: A₁ = 1 + jy, A₂ = (3 − y²) + 3jy,
    # A₃ = (15 − 6y²) + j(15y − y³), A₄ = (105 − 45y² + y⁴) + j(105y − 10y³).
    p1 = 1.0 + 1j * y
    p2 = (3.0 - y * y) + 3j * y
    p3 = (15.0 - 6.0 * y * y) + 1j * (15.0 * y - y**3)
    p4 = (105.0 - 45.0 * y * y + y**4) + 1j * (105.0 * y - 10.0 * y**3)
    g1 = -p1 * e / (2.0 * R**2)
    g2 = p2 * e / (4.0 * R**4)
    g3 = -p3 * e / (8.0 * R**6)
    g4 = p4 * e / (16.0 * R**8)
    r2 = rL * rL
    l_z = kL * kL * g1 + 2.0 * g2 + 4.0 * zeta**2 * g3
    l_z = l_z + r2 * (kL * kL * g2 + 2.0 * g3 + 4.0 * zeta**2 * g4)
    l_r = 8.0 * rL * zeta * g3 + 4.0 * rL * r2 * zeta * g4
    if shape == "const":
        s = np.ones_like(xi)
    elif shape == "sin":
        s = np.sin(kL * xi)
    elif shape == "cos":
        s = np.cos(kL * xi)
    else:  # the folded shape, pointwise
        s = -2.0 * np.sin(0.5 * kL * xi) ** 2
    gain = -1j * np.longdouble(eta) * aL * aL / (4.0 * np.longdouble(np.pi) * kL)
    return (
        complex(np.sum(s * l_z * w) * gain),
        complex(np.sum(s * l_r * w) * gain),
    )


# ---------------------------------------------------------------------------
# G-A1 — the differential test
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cos_shape", ["cos", "cos-1"])
def test_delta_matches_an_independent_longdouble_reference(cos_shape):
    """Every probe, both components, all three source shapes.

    The delta's integrand is analytic along the segment — bounded at ζ = 0,
    where the two kernels' difference tends to a finite multiple of G_red(a),
    and with its nearest pole a wire radius off the real axis — so a 16-point
    rule is not an approximation to be tolerated but a converged one, and
    what this measures is the float64 arithmetic underneath it.

    Two statements, because the reference has a floor of its own: against the
    N = 512 reference (the stated gate) and against the SAME 16 nodes at 80
    bits, which is arithmetic and nothing else. The second is twenty times
    the smaller of the two, which is how one knows the first is the
    reference's summation noise rather than the routine's.
    """
    sim = _solver()
    worst = dict.fromkeys(("far", "near"), 0.0)
    worst_same_nodes = dict.fromkeys(("far", "near"), 0.0)
    for H, z, a, panels in _probes():
        tier = "near" if panels > 1 else "far"
        scale = _const_scale(_coaxial_tables(sim, H, z, a))
        got = sim._folded_ek_delta_fields(
            K,
            np.array([H]),
            np.array([z]),
            np.array([a]),
            np.array([a]),
            cos_shape=cos_shape,
            n_panels=panels,
        )
        for shape in ("const", "sin", "cos"):
            ref_shape = shape
            if shape == "cos" and cos_shape == "cos-1":
                ref_shape = "folded"
            for n, bound in ((512, "worst"), (_N_QP_EK_DELTA, "same")):
                ez, er = _ld_delta(sim.eta, H, z, a, a, ref_shape, n=n, panels=panels)
                err = max(
                    abs(got[f"Ez_{shape}"][0] - ez) / scale,
                    abs(got[f"Erho_{shape}"][0] - er) / scale,
                )
                if bound == "worst":
                    worst[tier] = max(worst[tier], err)
                else:
                    worst_same_nodes[tier] = max(worst_same_nodes[tier], err)
    assert worst["far"] < DELTA_VS_LONGDOUBLE, worst
    assert worst_same_nodes["far"] < DELTA_ARITHMETIC, worst_same_nodes
    assert worst["near"] < ON_SEGMENT_FLOOR, worst
    assert worst_same_nodes["near"] < ON_SEGMENT_FLOOR, worst_same_nodes


# ---------------------------------------------------------------------------
# G-A2 — the conventions, against the EKSCX port
# ---------------------------------------------------------------------------
def test_reduced_plus_delta_reproduces_the_point_matched_extended_kernel():
    """On a fully-eligible straight-wire pair the two routes to the extended
    kernel's field must agree, and they do on all six tables.

    Route one is momwire#233's port of NEC's EKSCX (`_extended_kernel_fields`)
    — closed forms, per-end IND gating, both ends extended here and no swap,
    since the observer's ρ is exactly the source radius. Route two is the
    reduced closed forms plus this arc's quadrature. They share no code and
    no derivation, so agreeing to a part in 10¹² pins every sign, prefactor
    and shape convention in the new routine against a NEC transcription.

    That agreement is also what settles E_ρ. NEC Eq 89's scalar factor is a
    function of R and a alone and has no honest ρ-derivative; differentiating
    it as though R² = ζ² + ρ² lands E_ρ at half the value both EKSCX and the
    exact circumferential average give. `_folded_ek_delta_fields` therefore
    carries the kernel with its ρ-dependence intact (a²g′ + a²ρ²g″, which at
    ρ = a IS Eq 89), and this test is what would catch a regression to the
    other spelling — E_z would not notice.
    """
    sim = _solver()
    keys = ("Ez_const", "Erho_const", "Ez_sin", "Erho_sin", "Ez_cos", "Erho_cos")
    worst = {tier: dict.fromkeys(keys, 0.0) for tier in ("far", "near")}
    for H, z, a, panels in _probes():
        tier = "near" if panels > 1 else "far"
        red = _coaxial_tables(sim, H, z, a)
        ekscx = _coaxial_tables(
            sim, H, z, a, ek=(np.array([a]), np.array([0]), np.array([0]))
        )
        scale = _const_scale(red)
        delta = sim._folded_ek_delta_fields(
            K,
            np.array([[H]]),
            np.array([[z]]),
            red["rho_eval"],
            np.array([a]),
            cos_shape="cos",
            n_panels=panels,
        )
        for key in keys:
            got = red[key][0, 0] + delta[key][0, 0]
            worst[tier][key] = max(
                worst[tier][key], abs(got - ekscx[key][0, 0]) / scale
            )
    far, near = worst["far"], worst["near"]
    assert max(far.values()) < DELTA_VS_EKSCX, far
    # E_z carries no near-cancelling endpoint group in either route, so it is
    # the digit-for-digit statement; E_ρ is held to EKSCX's own noise above.
    assert max(far[k] for k in ("Ez_const", "Ez_sin", "Ez_cos")) < 1e-14, far
    # On the segment the two roles swap — it is E_z that carries the delta's
    # own (H/ρ)² cancellation there, and E_ρ that has none — so the tier is
    # held at its floor and the SHAPE of the residual is the statement.
    assert max(near.values()) < ON_SEGMENT_VS_EKSCX, near


# ---------------------------------------------------------------------------
# G-A3 — the a → 0 collapse
# ---------------------------------------------------------------------------
def test_delta_is_exactly_zero_at_zero_radius():
    """Structural, not asymptotic: the whole correction is linear in a², and
    that factor is applied as one multiplication at the end, so a = 0 gives
    IEEE zeros rather than something that rounds to them."""
    sim = _solver()
    H, z, a, panels = next(iter(_probes()))
    out = sim._folded_ek_delta_fields(
        K,
        np.array([H]),
        np.array([z]),
        np.array([a]),  # the regularized ρ stays finite; only the EK a vanishes
        np.array([0.0]),
    )
    for key, val in out.items():
        assert np.array_equal(val, np.zeros_like(val)), key


# ---------------------------------------------------------------------------
# G-A4 — the node count
# ---------------------------------------------------------------------------
def test_delta_node_count_is_converged():
    """16 nodes against 48. Spectral convergence on an entire integrand means
    the difference is not truncation but the rounding of two different sums —
    which is why this number is the same size as G-A1's."""
    sim = _solver()
    worst = dict.fromkeys(("far", "near"), 0.0)
    for H, z, a, panels in _probes():
        tier = "near" if panels > 1 else "far"
        scale = _const_scale(_coaxial_tables(sim, H, z, a))
        args = (K, np.array([H]), np.array([z]), np.array([a]), np.array([a]))
        lo = sim._folded_ek_delta_fields(*args, n_panels=panels)
        hi = sim._folded_ek_delta_fields(*args, n_gl=48, n_panels=panels)
        for key in lo:
            worst[tier] = max(worst[tier], abs(lo[key][0] - hi[key][0]) / scale)
    assert worst["far"] < NODE_CONVERGENCE, worst
    assert worst["near"] < ON_SEGMENT_NODE_CONVERGENCE, worst


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------
def test_seam_serves_the_folded_delta_through_field_components():
    """`_EKPairs` + `cos_shape="cos-1"` returns the #205 folded reduced tables
    plus the delta, and nothing else moves: `td`, the ρ̂ projection and the
    geometry columns are EFLD's, computed before the two kernels part."""
    sim = _solver()
    H, z, a, panels = next(iter(_probes()))
    red = _coaxial_tables(sim, H, z, a, cos_shape="cos-1")
    got = _coaxial_tables(
        sim, H, z, a, cos_shape="cos-1", ek=_EKPairs(np.array([a]), None)
    )
    delta = sim._folded_ek_delta_fields(
        K, np.array([[H]]), np.array([[z]]), red["rho_eval"], np.array([a])
    )
    for key in ("Ez_const", "Erho_const", "Ez_sin", "Erho_sin", "Ez_cos", "Erho_cos"):
        assert np.array_equal(got[key], red[key] + delta[key]), key
    for key in ("td", "rho_proj_factor", "rho_vec", "rho_eval"):
        assert np.array_equal(got[key], red[key]), key


def test_seam_ineligible_pairs_are_bit_identical_to_the_reduced_fill():
    """The eligibility mask is what unit B's pair rule will drive. An
    all-False mask has to give back the reduced tables to the bit, or EK-on
    would perturb pairs it never claimed."""
    sim = _solver()
    H, z, a, panels = next(iter(_probes()))
    red = _coaxial_tables(sim, H, z, a, cos_shape="cos-1")
    got = _coaxial_tables(
        sim,
        H,
        z,
        a,
        cos_shape="cos-1",
        ek=_EKPairs(np.array([a]), np.array([[False]])),
    )
    for key, val in red.items():
        assert np.array_equal(got[key], val), key


@pytest.mark.parametrize(
    "ek,cos_shape",
    [
        (_EKPairs(np.array([1e-3]), None), "cos"),
        ((np.array([1e-3]), np.array([0]), np.array([0])), "cos-1"),
    ],
)
def test_seam_refuses_the_mismatched_payload(ek, cos_shape):
    """Before momwire#246 the EK branch early-returned whatever `cos_shape`
    said, so `ek=` with the folded shape was silently served literal-cos EK
    tables. Both crossed combinations now raise and name the issue."""
    sim = _solver()
    H, z, a, panels = next(iter(_probes()))
    with pytest.raises(NotImplementedError, match="246"):
        _coaxial_tables(sim, H, z, a, cos_shape=cos_shape, ek=ek)


# ---------------------------------------------------------------------------
# G-A5 — the armor
# ---------------------------------------------------------------------------
def test_no_other_path_can_reach_the_delta(monkeypatch):
    """The reduced paths and the point-matched extended kernel must not so
    much as call the new routine — the EK-off (and EK-point-matched) fills
    have to stay byte-frozen, and a routine that never runs cannot move them.

    A full free-space fill each way, with the delta booby-trapped.
    """

    def trap(*a, **kw):
        raise AssertionError("the EK delta ran on a path that did not ask for it")

    monkeypatch.setattr(SinusoidalSolver, "_folded_ek_delta_fields", trap)
    z_reduced = _solver().compute_impedance()[0]
    z_ek = _solver(extended_kernel=True).compute_impedance()[0]
    # …and the extended kernel is not a no-op, so the trap was not vacuous.
    assert abs(z_ek - z_reduced) > 1e-9 * abs(z_reduced)


# ===========================================================================
# Unit B — the solver wiring
# ===========================================================================
#
# `SinusoidalGalerkinSolver(extended_kernel=True)` now assembles: the
# free-space block, the PEC image, the reflection-coefficient image and the
# near-pair correction all carry the delta, on the pairs momwire#249 §4's
# symmetric rule admits. The gates below are, in order:
#
#   G-B1  the Δ/a ladder's SHIFT — δZ = Z(EK on) − Z(EK off) — against
#         nec2c's own δZ column and against the point-matched solver's δZ on
#         the same mesh. Absolute Z is not gated: that is the basis's
#         accuracy, which #182's instrument report already measures.
#   G-B2  ‖G−Gᵀ‖/‖G‖ survives, and where EK-on sits above the reduced fill's
#         own ratio it is the shared TEST rule that is short, not the pair
#         rule — refining `n_qp_test` drives it back under. The falsifying
#         contrast is an asymmetric mask, which no refinement can rescue.
#   G-B3  a → 0 collapses the EK fill onto the reduced one.
#   G-B4  EK OFF is the pre-#246 code path, bit for bit, and enters no EK
#         code at all.
#   G-B5  the PEC image block is doing the work, not riding along.
#
# And the two decisions that are not tolerances: which ground models are
# served, and which pairs an IMAGE block extends.

from momwire import sinusoidal_galerkin as _sg  # noqa: E402
from momwire.sinusoidal_galerkin import (  # noqa: E402
    SinusoidalGalerkinSolver,
    _plain_projection,
)

NS = 41
LEN = 5.0
LAM_NEC = 299.792458 / 30.0  # nec2c's CVEL, so the oracle meshes as we do

# radius -> (Δ/a, nec2c EK-OFF, nec2c EK-ON). The pinned oracle deck of
# `test_extended_kernel.py` — a free-space dipole, L = 5 m, NS = 41, 30 MHz,
# centre fed — reused here verbatim so that the Galerkin shift is measured
# against the same column the collocation shift was.
LADDER = {
    0.001: (121.951, 80.146 + 46.432j, 80.146 + 46.430j),
    0.0048780: (25.000, 83.291 + 48.053j, 83.281 + 48.016j),
    0.005: (24.390, 83.364 + 48.092j, 83.353 + 48.053j),
    0.02: (6.0976, 90.040 + 50.737j, 89.774 + 50.190j),
    0.05: (2.4390, 101.010 + 50.114j, 98.660 + 48.113j),
    0.1: (1.2195, 123.470 + 34.444j, 111.020 + 37.528j),
    0.15: (0.81301, 130.460 - 30.093j, 122.240 + 18.377j),
    0.3: (0.40650, 0.40269 - 7.5778j, 37.990 - 56.910j),
}

# Rungs on which nec2c's own δZ is RESOLVED by the pinned oracle. The table is
# quoted to three decimals of an ~80 Ω impedance, and at Δ/a ≥ 24 the EK card
# moves nec2c by 0.04 % and below — so its δZ there is 0.000 ± half a digit,
# and its real part is literally 0. A sign or a ratio taken against that is a
# statement about the transcription's last digit, not about a kernel.
_RESOLVED = (0.02, 0.05, 0.1, 0.15, 0.3)

# Rungs where the extended kernel is the DOMINANT term rather than a
# correction — nec2c moves by 2.7 % to 817 % — and where the two testings can
# therefore be compared on the shift itself. This is #249 §7's cross-basis
# bar, and DESIGN246 §6 gate 2.
_EK_DOMINANT = (0.05, 0.1, 0.15, 0.3)
_SHIFT_BAR = 0.25

_DIPOLE_W = [np.array([[0.0, 0.0, -LEN / 2], [0.0, 0.0, LEN / 2]])]


def _ladder_deck(cls, radius, **kw):
    return cls(
        wires=_DIPOLE_W,
        n_per_edge_per_wire=[[NS]],
        wavelength=LAM_NEC,
        wire_radius=radius,
        nsegs=NS,
        feed_arclength=LEN / 2,
        **kw,
    )


def _shift(cls, radius, **kw):
    """δZ = Z(EK on) − Z(EK off) on ONE mesh, so the basis's own
    discretization gap — which is what absolute Z would measure — cancels."""
    off, _ = _ladder_deck(cls, radius, **kw).compute_impedance()
    on, _ = _ladder_deck(cls, radius, extended_kernel=True, **kw).compute_impedance()
    return on - off


# ---------------------------------------------------------------------------
# G-B1 — the nec2c ladder, as a SHIFT
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("radius", _EK_DOMINANT)
def test_gb1_galerkin_shift_tracks_nec2c_where_the_kernel_dominates(radius):
    """Measured against the oracle's own δZ column:

        Δ/a     δZ nec2c            δZ galerkin          off by
        2.439   −2.350 − 2.001j     −2.289 − 2.015j       2.0 %
        1.220  −12.450 + 3.084j    −11.273 + 3.422j       9.5 %
        0.813   −8.220 + 48.470j    −7.607 + 47.350j      2.6 %
        0.407  +37.587 − 49.332j   +36.105 − 48.502j      2.7 %

    The point-matched solver on the same rungs is off by 51 %, 7.7 %, 2.5 %
    and 0.7 % — so this is not a weaker statement than #233's, it is a
    stronger one at the fat end.
    """
    da, z_off, z_on = LADDER[radius]
    d_nec = z_on - z_off
    d_gal = _shift(SinusoidalGalerkinSolver, radius)
    err = abs(d_gal - d_nec) / abs(d_nec)
    assert err < _SHIFT_BAR, f"Δ/a={da}: δZ {d_gal} vs nec2c {d_nec} ({err:.1%})"


@pytest.mark.parametrize("radius", [0.1, 0.15, 0.3])
def test_gb1_galerkin_and_collocation_shifts_agree(radius):
    """The cross-basis bar (#249 §7): on the rungs where the extended kernel
    carries the answer, the two testings must move Z by the same amount.
    Measured 3.5 %, 1.0 % and 2.8 %.

    The bar is deliberately not applied above Δ/a ≈ 2.5. There the shift is a
    small correction and the two testings genuinely differ on it — see
    `test_gb1_thin_wire_galerkin_shift_exceeds_the_point_matched_one`, which
    measures that difference and shows it is the fill and not an error.
    """
    da = LADDER[radius][0]
    d_gal = _shift(SinusoidalGalerkinSolver, radius)
    d_col = _shift(SinusoidalSolver, radius)
    err = abs(d_gal - d_col) / abs(d_col)
    assert err < _SHIFT_BAR, f"Δ/a={da}: galerkin {d_gal} vs collocation {d_col}"


@pytest.mark.parametrize("radius", _RESOLVED)
def test_gb1_shift_has_nec2c_sign_structure(radius):
    """Both components of δZ point the way the oracle's do, on every rung
    where the oracle resolves them. A kernel wired with a sign error would
    survive a magnitude bar on the fat rungs and die here."""
    _, z_off, z_on = LADDER[radius]
    d_nec = z_on - z_off
    d_gal = _shift(SinusoidalGalerkinSolver, radius)
    assert np.sign(d_gal.real) == np.sign(d_nec.real), (d_gal, d_nec)
    assert np.sign(d_gal.imag) == np.sign(d_nec.imag), (d_gal, d_nec)


@pytest.mark.parametrize("radius", [0.005, 0.001])
def test_gb1_thin_wire_galerkin_shift_exceeds_the_point_matched_one(radius):
    """A measured property of the Galerkin fill, pinned so that it cannot
    drift silently, and shown to be the fill rather than a defect.

    At Δ/a ≥ 24 the Galerkin δZ runs several times the point-matched one
    (0.146 Ω against 0.046 Ω at Δ/a = 24.4; 0.021 Ω against 0.0014 Ω at
    Δ/a = 122). That is not quadrature error — refining the delta rule, the
    near rule and the test rule each move it by less than 1e-7 — it is what
    testing does to NEC's kernel. The extended-minus-reduced field is O(1)
    relative to the reduced one in a region of width `a` around each source
    segment END: `_folded_ek_delta_fields`' own probe puts the ratio at 0.75
    on the endpoint at Δ/a = 122 and at 4e-4 in the segment's middle. A
    collocation point sits at a segment CENTRE and never visits that region;
    a test integral sweeps through it and picks up its O(a/h) weight.

    The witness is a second construction of the same fill that shares no
    quadrature with the first: rebuild the extended tables from NEC's own
    EKSCX closed forms (`_extended_kernel_fields`, momwire#233's port) and
    fold them by subtraction — the spelling momwire#205 rejected on
    precision grounds, which is exactly why it is an independent check here.
    It reproduces δZ to eight figures.
    """
    d_delta = _shift(SinusoidalGalerkinSolver, radius)
    d_col = _shift(SinusoidalSolver, radius)
    assert abs(d_delta) > 3.0 * abs(d_col)

    original = SinusoidalSolver._field_components_bcast

    def ekscx_built(self, k, obs_c, obs_t, a, src_c, src_t, src_hh, **kw):
        ek = kw.pop("ek", None)
        if ek is None:
            return original(self, k, obs_c, obs_t, a, src_c, src_t, src_hh, **kw)
        # Straight wire: every pair is coaxial and equal-radius, so both ends
        # of every source segment are extended (NEC's IND = 0/1) and the mask
        # is all-True — the one geometry on which the two gatings coincide.
        kw["cos_shape"] = "cos"
        zeros = np.zeros(1, dtype=np.int8)
        tables = dict(
            original(
                self,
                k,
                obs_c,
                obs_t,
                a,
                src_c,
                src_t,
                src_hh,
                ek=(np.asarray(ek.src_a), zeros, zeros),
                **kw,
            )
        )
        tables["Ez_cos"] = tables["Ez_cos"] - tables["Ez_const"]
        tables["Erho_cos"] = tables["Erho_cos"] - tables["Erho_const"]
        return tables

    try:
        SinusoidalSolver._field_components_bcast = ekscx_built
        d_ekscx = _shift(SinusoidalGalerkinSolver, radius)
    finally:
        SinusoidalSolver._field_components_bcast = original
    assert abs(d_ekscx - d_delta) < 1e-6 * abs(d_delta), (d_ekscx, d_delta)


# ---------------------------------------------------------------------------
# The scope decisions
# ---------------------------------------------------------------------------
_GROUND_KW = {
    "free space": {},
    "PEC": dict(ground_z=0.0),
    "refl-coef": dict(ground_z=0.0, ground_eps=(13.0, 0.005)),
    # momwire#287. The remainder half of this model stays reduced, on the
    # measurement G-S1 makes at the bottom of this file.
    "sommerfeld": dict(
        ground_z=0.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld"
    ),
}


def _monopole(**kw):
    """A FAT monopole — Δ/a ≈ 1.07 — so the extended kernel is a first-order
    term on every deck below rather than a rounding difference."""
    return SinusoidalGalerkinSolver(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.4]])],
        n_per_edge_per_wire=[[15]],
        wavelength=LAM_NEC,
        wire_radius=0.15,
        nsegs=15,
        feed_arclength=0.08,
        **kw,
    )


@pytest.mark.parametrize("ground", list(_GROUND_KW))
def test_galerkin_accepts_the_extended_kernel_on_every_served_ground(ground):
    """momwire#233's blanket refusal is now gone entirely: free space, the
    PEC image, the reflection-coefficient image and the Sommerfeld ground
    (momwire#287) all solve, and the answer moves on every one of them."""
    kw = _GROUND_KW[ground]
    z_red, _ = _monopole(**kw).compute_impedance()
    z_ext, _ = _monopole(extended_kernel=True, **kw).compute_impedance()
    assert abs(z_ext - z_red) > 1e-3 * abs(z_red)


def test_galerkin_still_serves_sommerfeld_with_the_reduced_kernel():
    """The kernel is a choice on this ground, not a requirement of it."""
    z, _ = _monopole(
        ground_z=0.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld"
    ).compute_impedance()
    assert np.isfinite(z)


def test_galerkin_refuses_the_extended_kernel_without_the_near_correction():
    """M1 mode would leave the on-segment pairs on the far tier's rule, which
    is not a coarser answer there but a wrong one."""
    with pytest.raises(NotImplementedError, match="near_correction"):
        _monopole(extended_kernel=True, near_correction=False)


def test_image_pairs_are_scored_against_the_mirrored_geometry():
    """The image-block eligibility decision, stated as a test.

    A wire STANDING on the plane maps onto its own axis, so every real/image
    pair is coaxial and equal-radius and the image block is extended — NEC's
    IND = 0 ground-contact branch, reached here through the mirror instead of
    through a neighbour table. A wire PARALLEL to the plane maps onto a
    different line, offset by twice its height, so no real/image pair is
    coaxial and the image block stays reduced.

    Both fall out of scoring the labels in ONE scan over real ∧ mirrored
    segments. Two independent scans would label a horizontal wire and its
    image identically and extend every pair of them.
    """

    def labels(wire, **kw):
        sim = SinusoidalGalerkinSolver(
            wires=[wire],
            n_per_edge_per_wire=[[6]],
            wavelength=LAM_NEC,
            wire_radius=0.05,
            nsegs=6,
            feed_arclength=0.2,
            ground_z=0.0,
            extended_kernel=True,
            **kw,
        )
        geom = sim._build_geometry()
        gi, gj = sim._ek_axis_labels(geom, mirror=True)
        return gi[:, None] == gj[None, :]

    standing = labels(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.4]]))
    lying = labels(np.array([[-1.2, 0.0, 3.0], [1.2, 0.0, 3.0]]))
    assert standing.all(), "a wire on the plane must extend against its image"
    assert not lying.any(), "a wire parallel to the plane must not"


def test_the_pair_mask_is_symmetric_by_construction():
    """The free-space mask is label equality against ONE label array, so
    eligible(i, j) == eligible(j, i) is structural rather than measured — and
    the image mask inherits it, reflection being an isometry that maps the
    coaxial-and-equal-radius relation onto itself."""
    sim = _monopole(extended_kernel=True, ground_z=0.0)
    geom = sim._build_geometry()
    n = geom["n_segs"]
    idx = np.arange(n)
    for mirror in (False, True):
        mask = sim._ek_pairs(geom, idx[:, None], idx[None, :], mirror).eligible
        assert np.array_equal(mask, mask.T), mirror


# ---------------------------------------------------------------------------
# G-B2 — reciprocity
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _without_the_282_contact_correction():
    """Assemble with #282's ground-contact charge correction off — see the
    same-named helper in test_sinusoidal_galerkin.py."""
    cls = SinusoidalGalerkinSolver
    orig = cls._contact_charge_correction_tested
    cls._contact_charge_correction_tested = lambda self, G, geom, k, sv, ctx: G
    try:
        yield
    finally:
        cls._contact_charge_correction_tested = orig


def _sym_ratio(sim):
    geom = sim._build_geometry()
    G, _ = sim._assemble_Z(geom, sim.k)
    return np.linalg.norm(G - G.T) / np.linalg.norm(G)


@pytest.mark.parametrize("radius,bar", [(0.05, 2.0), (0.02, 2.5)])
def test_gb2_symmetry_survives_the_extended_kernel(radius, bar):
    """On the ladder's own mesh the EK-on fill is as reciprocal as the
    reduced one: 5.8e-13 against 3.5e-13 at Δ/a = 2.4 (1.7×), 4.6e-13
    against 2.1e-13 at Δ/a = 6.1 (2.2×)."""
    r_red = _sym_ratio(_ladder_deck(SinusoidalGalerkinSolver, radius))
    r_ext = _sym_ratio(
        _ladder_deck(SinusoidalGalerkinSolver, radius, extended_kernel=True)
    )
    assert r_ext < bar * r_red, f"{r_ext:.2e} vs reduced {r_red:.2e}"


@pytest.mark.parametrize(
    "ground",
    # Sommerfeld is deliberately absent, and G-S3 says why in its own terms:
    # that ground's asymmetry floor is not the TEST rule's but the
    # remainder's SOURCE rule's (`n_qp_sommerfeld`), so refining n_qp_test
    # does not move it — the claim this test makes is not the claim that
    # ground supports. `test_g4_sommerfeld_symmetry_near_the_plane_is_source_
    # quadrature_limited` in test_sommerfeld_ground_sinusoidal.py owns that.
    [g for g in _GROUND_KW if g != "sommerfeld"],
)
def test_gb2_fat_wire_asymmetry_is_test_quadrature_limited(ground):
    """At Δ/a ≈ 1 the EK-on ratio sits ABOVE the reduced fill's — 1.9e-12
    against 4.4e-14 in free space — and this is what that means.

    The delta varies on the scale of the wire radius, which on this deck is
    the segment length, so the SHARED far-pair test rule is the short one:
    at `n_qp_test` = 16 the ratio falls to 5.1e-15, under the reduced fill's
    own. A pair rule that broke reciprocity structurally — NEC's per-END
    gating is the live example — would floor instead, exactly as
    `test_uniform_rule_symmetry_is_quadrature_limited` shows for the M1 rule.
    The next test does that floor deliberately.
    """
    kw = _GROUND_KW[ground]
    # This deck STANDS IN the plane, so over a finite ground #282's
    # contact-charge correction applies — and that correction is one-sided
    # by construction (it removes a source-model charge; a test function has
    # none to remove), which makes the fill non-self-adjoint at 8.5e-2 here.
    # The quadrature statement this test makes is about the EK fill and is
    # measured with the correction off; `test_the_282_contact_correction_is
    # _not_self_adjoint` in test_sinusoidal_galerkin.py owns the trade.
    with _without_the_282_contact_correction():
        coarse = _sym_ratio(_monopole(extended_kernel=True, **kw))
        fine = _sym_ratio(
            _monopole(extended_kernel=True, n_qp_test=16, n_qp_near=16, **kw)
        )
    assert coarse < 1e-10, coarse
    assert fine < 0.1 * coarse, f"{fine:.2e} did not improve on {coarse:.2e}"
    assert fine <= 10.0 * _sym_ratio(_monopole(n_qp_test=16, n_qp_near=16, **kw))


def test_gb2_an_asymmetric_mask_breaks_reciprocity_and_refinement_cannot_fix_it(
    monkeypatch,
):
    """The falsifying contrast, and the reason the pair rule exists.

    Extend only the pairs with j > i — a caricature of a per-source-segment
    decision — and ‖G−Gᵀ‖/‖G‖ jumps by orders of magnitude AND stops
    responding to the test rule, because the asymmetry is now in what is
    being integrated rather than in how well.

    On the NUMPY backend, forced. The caricature is `mask & (j > i)`, and
    since momwire#358 the fused fill no longer takes a mask: it takes the
    pair rule's group LABELS and scores `g_obs[m] == g_src[n]` itself, which
    is symmetric by construction and cannot express an upper-triangular
    mutation at all. Forcing the backend that still evaluates `_ek_pairs`
    keeps the caricature exactly as written; the two backends agree to 1e-15
    on the honest fill (G-C1/G-C3), so nothing about the statement is
    backend-specific.
    """
    monkeypatch.setattr(_sg, "_HAVE_GALERKIN_FAR_FILL", False)
    original = SinusoidalGalerkinSolver._ek_pairs

    def upper_only(self, geom, m_idx, n_idx, mirror, n_panels=1):
        pairs = original(self, geom, m_idx, n_idx, mirror, n_panels)
        return pairs._replace(eligible=pairs.eligible & (n_idx > m_idx))

    try:
        SinusoidalGalerkinSolver._ek_pairs = upper_only
        coarse = _sym_ratio(_monopole(extended_kernel=True))
        fine = _sym_ratio(_monopole(extended_kernel=True, n_qp_test=16, n_qp_near=16))
    finally:
        SinusoidalGalerkinSolver._ek_pairs = original
    assert coarse > 1e-3, coarse
    assert fine > 0.5 * coarse, f"{fine:.2e} vs {coarse:.2e}: not structural"


# ---------------------------------------------------------------------------
# G-B3 — a → 0
# ---------------------------------------------------------------------------
def _fill(radius, **kw):
    sim = _ladder_deck(SinusoidalGalerkinSolver, radius, **kw)
    geom = sim._build_geometry()
    return sim._assemble_Z(geom, sim.k)[0]


def test_gb3_the_fill_collapses_onto_the_reduced_one_as_the_wire_thins():
    """‖G_ek − G_red‖/‖G_red‖ over four decades of radius on the ladder mesh:

        a = 0.05    1.6e-1
        a = 0.005   2.5e-3
        a = 5e-4    3.7e-5
        a = 5e-5    2.6e-6

    Monotone, and three and a half decades of collapse for three decades of
    radius. It is not a clean a² law at the thin end and it is not supposed
    to be: the delta's own whole-line near-cancellation leaves ~ε·(H/ρ)² of
    its peak behind, which is O(1) in these units — the floor
    `_folded_ek_delta_fields` documents, and the reason this gate is a
    collapse rather than an exponent.
    """
    prev = None
    for radius in (0.05, 0.005, 5e-4, 5e-5):
        red = _fill(radius)
        ext = _fill(radius, extended_kernel=True)
        rel = np.linalg.norm(ext - red) / np.linalg.norm(red)
        assert prev is None or rel < 0.1 * prev, (radius, rel, prev)
        prev = rel
    assert prev < 1e-5, prev


def test_gb3_zero_radius_gives_the_reduced_fill_to_the_bit():
    """Structural, at the seam the fill actually uses: hand the payload a
    zero source radius and every table comes back the reduced one — the a²
    prefactor is one multiplication at the end, so this is IEEE equality and
    not a limit. The mask branch is exercised too (half the pairs
    ineligible), since selecting the SUM is what keeps those bit-identical.
    """
    sim = _monopole(extended_kernel=True)
    geom = sim._build_geometry()
    ctx = sim._test_context(geom, sim._basis_coefs(geom, sim.k), sim.k)
    n = ctx["N"]
    kwargs = dict(
        obs_c=ctx["obs_c"][:, None, :],
        obs_t=ctx["obs_t"][:, None, :],
        a=ctx["a_obs"],
        src_c=geom["seg_centers"][None, :, :],
        src_t=geom["seg_tangents"][None, :, :],
        src_hh=ctx["hh"][None, :],
        cos_shape="cos-1",
    )
    red = sim._field_components_bcast(sim.k, **kwargs)
    half = np.zeros((1, n), dtype=bool)
    half[0, : n // 2] = True
    got = sim._field_components_bcast(
        sim.k,
        ek=_EKPairs(np.zeros((1, n)), half, _N_PANEL_EK_DELTA_NEAR),
        **kwargs,
    )
    for key, val in red.items():
        assert np.array_equal(got[key], val), key


# ---------------------------------------------------------------------------
# G-B4 — EK off is the pre-#246 path
# ---------------------------------------------------------------------------
_G4_DECKS = {
    "straight dipole": dict(
        wires=_DIPOLE_W,
        n_per_edge_per_wire=[[21]],
        nsegs=21,
        wire_radius=0.05,
        feed_arclength=LEN / 2,
    ),
    "vee": dict(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [1.5, 0.0, 2.0]])],
        n_per_edge_per_wire=[[8, 6]],
        nsegs=14,
        wire_radius=0.02,
        feed_arclength=0.125,
    ),
    "PEC ground": dict(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.4]])],
        n_per_edge_per_wire=[[14]],
        nsegs=14,
        wire_radius=0.02,
        feed_arclength=0.085,
        ground_z=0.0,
    ),
    "refl-coef ground": dict(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.4]])],
        n_per_edge_per_wire=[[14]],
        nsegs=14,
        wire_radius=0.02,
        feed_arclength=0.085,
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
    ),
    "sommerfeld ground": dict(
        wires=[np.array([[0.0, 0.0, 0.3], [0.0, 0.0, 2.7]])],
        n_per_edge_per_wire=[[14]],
        nsegs=14,
        wire_radius=0.02,
        feed_arclength=1.2,
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
    ),
}


@pytest.mark.parametrize("name", list(_G4_DECKS))
def test_gb4_ek_off_is_bit_identical_to_the_default(name):
    """Numerical identity is necessary and not sufficient (#233's argument),
    so this is half of the armor and the counter below is the other."""
    kw = dict(_G4_DECKS[name], wavelength=LAM_NEC)
    z_def, c_def = SinusoidalGalerkinSolver(**kw).compute_impedance()
    z_off, c_off = SinusoidalGalerkinSolver(
        **kw, extended_kernel=False
    ).compute_impedance()
    assert z_def == z_off, f"{name}: {z_def!r} vs {z_off!r}"
    assert np.array_equal(c_def, c_off)


_EK_ENTRY_POINTS = [
    (SinusoidalSolver, "_folded_ek_delta_fields"),
    (SinusoidalSolver, "_ek_delta_rule"),
    (SinusoidalSolver, "_ek_end_bracket_fields"),
    (SinusoidalGalerkinSolver, "_ek_axis_labels"),
    (SinusoidalGalerkinSolver, "_ek_pairs"),
    (SinusoidalGalerkinSolver, "_ek_reduced_ends"),
    (SinusoidalGalerkinSolver, "_ek_bracket_plans"),
    (SinusoidalGalerkinSolver, "_ek_bracket_block"),
]

# The fused fill's own eligibility payload (momwire#358) joins the roster only
# on a build that has the C++ EK twin to consume it: without one an EK-on fill
# correctly takes the numpy block loop and never builds it, and the "they all
# fire with EK on" control below would be demanding a call that must not
# happen. The "EK off enters no EK code" half covers it either way.
if _sg._HAVE_GALERKIN_FAR_FILL and _sg._HAVE_GALERKIN_FAR_FILL_EK:
    _EK_ENTRY_POINTS.append((SinusoidalGalerkinSolver, "_ek_far_labels"))


@pytest.fixture
def ek_call_counts(monkeypatch):
    counts = {}
    for owner, attr in _EK_ENTRY_POINTS:
        counts[attr] = 0
        original = getattr(owner, attr)

        def wrapper(*args, _a=attr, _f=original, **kwargs):
            counts[_a] += 1
            return _f(*args, **kwargs)

        monkeypatch.setattr(owner, attr, wrapper)
    return counts


def test_gb4b_the_counters_fire_when_ek_is_on(ek_call_counts):
    """The control: a monkeypatch that failed to bind would make the gate
    below pass vacuously. Two decks, because momwire#299's end-bracket
    correction only evaluates a bracket on a geometry with a SPLIT node — the
    monopole reaches `_ek_bracket_block` and returns from it."""
    _monopole(extended_kernel=True, ground_z=0.0).compute_impedance()
    SinusoidalGalerkinSolver(
        **dict(_G4_DECKS["vee"], wavelength=LAM_NEC), extended_kernel=True
    ).compute_impedance()
    for attr, n in ek_call_counts.items():
        assert n > 0, f"{attr} never called with EK on"


@pytest.mark.parametrize("name", list(_G4_DECKS))
def test_gb4b_ek_off_enters_no_ek_code(ek_call_counts, name):
    kw = dict(_G4_DECKS[name], wavelength=LAM_NEC)
    SinusoidalGalerkinSolver(**kw).compute_impedance()
    assert ek_call_counts == dict.fromkeys(ek_call_counts, 0)


def test_gb4_the_fused_far_fill_refuses_an_ek_payload_it_cannot_serve(monkeypatch):
    """The reduced C++ far fill takes no eligibility payload, so routing an
    EK-on block through it would drop the delta silently. Unit C added the twin
    and flipped `_HAVE_GALERKIN_FAR_FILL_EK`; on a build WITHOUT it — a
    pure-Python install, or an extension predating #246 — `_tested_contribs`
    must take the numpy path and `_far_fill_accel` must refuse an EK payload
    rather than ignore it. That is what this simulates."""
    monkeypatch.setattr(_sg, "_HAVE_GALERKIN_FAR_FILL_EK", False)
    sim = _monopole(extended_kernel=True)
    payload = _sg._EKFarLabels(
        np.zeros(1), np.zeros(1, np.int64), np.zeros(1, np.int64)
    )
    with pytest.raises(NotImplementedError, match="unit C"):
        sim._far_fill_accel(sim.k, None, None, None, ek=payload)


# ---------------------------------------------------------------------------
# G-B5 — the image block is doing the work
# ---------------------------------------------------------------------------
def test_gb5_pec_image_shift_matches_the_point_matched_one():
    """A fat monopole STANDING on a PEC plane — the deck whose image block is
    eligible at all. δZ = −4.192 − 1.605j against the point-matched solver's
    −4.210 − 1.191j, 9.5 % apart on the shift.

    That the image block is what is being measured is pinned by the free-space
    control: the same wire off the ground moves by −158.6j, two orders away.
    """
    kw = dict(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.4]])],
        n_per_edge_per_wire=[[15]],
        wavelength=LAM_NEC,
        wire_radius=0.15,
        nsegs=15,
        feed_arclength=0.08,
        ground_z=0.0,
    )

    def shift(cls):
        off, _ = cls(**kw).compute_impedance()
        on, _ = cls(**kw, extended_kernel=True).compute_impedance()
        return on - off

    d_gal = shift(SinusoidalGalerkinSolver)
    d_col = shift(SinusoidalSolver)
    err = abs(d_gal - d_col) / abs(d_col)
    assert err < _SHIFT_BAR, f"galerkin {d_gal} vs collocation {d_col} ({err:.1%})"


def test_gb5_the_image_block_carries_its_own_delta(monkeypatch):
    """Not a ride-along: force the image block's payload to None and the
    grounded answer moves, by more than the free-space block's own EK shift
    would explain.

    BOTH payload builders are neutered, because since momwire#358 the fused
    far fill takes `_ek_far_labels` while the near correction and the numpy
    block loop still take `_ek_pairs`. Patching only one would leave the
    image block half-extended and make the gate measure the wrong mutation.
    """
    orig_pairs = SinusoidalGalerkinSolver._ek_pairs
    orig_labels = SinusoidalGalerkinSolver._ek_far_labels

    def pairs_free_space_only(self, geom, m_idx, n_idx, mirror, n_panels=1):
        if mirror:
            return None
        return orig_pairs(self, geom, m_idx, n_idx, mirror, n_panels)

    def labels_free_space_only(self, geom, mirror, n_panels=1):
        if mirror:
            return None
        return orig_labels(self, geom, mirror, n_panels)

    z_both, _ = _monopole(extended_kernel=True, ground_z=0.0).compute_impedance()
    monkeypatch.setattr(SinusoidalGalerkinSolver, "_ek_pairs", pairs_free_space_only)
    monkeypatch.setattr(
        SinusoidalGalerkinSolver, "_ek_far_labels", labels_free_space_only
    )
    z_free_only, _ = _monopole(extended_kernel=True, ground_z=0.0).compute_impedance()
    assert abs(z_both - z_free_only) > 1e-3 * abs(z_both)


def test_gb5_the_refl_coef_image_rides_the_same_delta():
    """The Fresnel dyad is a per-segment-pair weight applied AFTER the field
    tables, so the reflection-coefficient block needs no EK code of its own —
    and this is the statement that it got the delta anyway."""
    original = SinusoidalGalerkinSolver._ek_pairs
    seen = []

    def spy(self, geom, m_idx, n_idx, mirror, n_panels=1):
        seen.append(mirror)
        return original(self, geom, m_idx, n_idx, mirror, n_panels)

    try:
        SinusoidalGalerkinSolver._ek_pairs = spy
        sim = _monopole(extended_kernel=True, ground_z=0.0, ground_eps=(13.0, 0.005))
        geom = sim._build_geometry()
        ctx = sim._test_context(geom, sim._basis_coefs(geom, sim.k), sim.k)
        nnz, n_src = ctx["w_entry"].shape[0], ctx["N"]
        dest = tuple(
            np.zeros((nnz, n_src), dtype=np.complex128) for _ in range(3)
        )  # the free-space triple the ground folds into (#332)
        sim._fold_ground_block(geom, sim.k, ctx, dest)
    finally:
        SinusoidalGalerkinSolver._ek_pairs = original
    assert seen and all(seen), seen
    assert _plain_projection is not None  # the refl block used its own projector


def test_the_pair_mask_equals_an_independent_coaxiality_predicate():
    """Armor against over-extension (reviewer's probe: flipping the mask's
    `&` to `|` — every labeled pair eligible — passed every shift and
    symmetry gate, because the spurious delta on well-separated non-coaxial
    pairs is symmetric and small). The mask is therefore pinned directly:
    on a bent deck it must equal a predicate computed here from raw segment
    geometry — same line (parallel tangents AND no perpendicular offset
    between centers) and equal radii — with both eligible and ineligible
    pairs present in the deck."""
    kw = dict(_G4_DECKS["vee"], wavelength=LAM_NEC, extended_kernel=True)
    sim = SinusoidalGalerkinSolver(**kw)
    geom = sim._build_geometry()
    n = geom["n_segs"]
    idx = np.arange(n)
    mask = sim._ek_pairs(geom, idx[:, None], idx[None, :], mirror=False).eligible

    seg_c = geom["seg_centers"]
    seg_t = geom["seg_tangents"]
    seg_a = sim._seg_radius(geom)
    expect = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            parallel = abs(float(np.dot(seg_t[i], seg_t[j]))) >= 1.0 - 1e-6
            d = seg_c[j] - seg_c[i]
            perp = d - np.dot(d, seg_t[i]) * seg_t[i]
            on_line = float(np.linalg.norm(perp)) <= 1e-6 * (
                1.0 + float(np.linalg.norm(d))
            )
            radii = abs(float(seg_a[i]) - float(seg_a[j])) <= 1e-6 * float(seg_a[i])
            expect[i, j] = parallel and on_line and radii
    # The vee must exercise both answers or the pin is vacuous.
    assert expect.any() and not expect.all()
    assert np.array_equal(mask, expect)


# ===========================================================================
# G-C — the C++ twin (momwire#246 unit C)
# ===========================================================================
# The fused far fill got an extended-kernel twin,
# `_accelerators.sinusoidal_galerkin_far_fill_ek`, so an EK-on Galerkin fill
# runs at the accelerated path's speed instead of falling back to the numpy
# block loop. The gates below are the #249 parity rule applied to it:
#
#   G-C1  the two backends agree at TOLERANCE on the EK far fill — never bit,
#         because the C++ path reassociates the same arithmetic.
#   G-C2  the REDUCED far fill is untouched: an all-ineligible EK call is bit
#         identical to the reduced entry point, and EK off never reaches the
#         twin at all. (Byte-freeze against a pre-#246 BUILD is measured at
#         review time — `np.array_equal` of the reduced fill's raw output
#         across the two .so builds — and is not reproducible from inside the
#         suite, which only ever has one build.)
#   G-C3  end to end: the same Z either way, and the G-B1 shift gates above
#         now run with the C++ path active.

_HAVE_ACCEL = _sg._HAVE_GALERKIN_FAR_FILL and _sg._HAVE_GALERKIN_FAR_FILL_EK

_GC_DECKS = {
    "fat dipole": _G4_DECKS["straight dipole"],
    "thin dipole": dict(
        wires=_DIPOLE_W,
        n_per_edge_per_wire=[[41]],
        nsegs=41,
        wire_radius=0.001,
        feed_arclength=LEN / 2,
    ),
    "vee": _G4_DECKS["vee"],
    "PEC ground": _G4_DECKS["PEC ground"],
    "refl-coef ground": _G4_DECKS["refl-coef ground"],
}

# The two backends agree to reassociation, which on these decks measures
# 7.2e-15 of the block's largest entry at worst (the fat ladder rungs) — the
# same level the reduced far fill has agreed at since #194.
_GC1_BAR = 1e-13


def _gc_solver(name, **over):
    return SinusoidalGalerkinSolver(
        **dict(_GC_DECKS[name], wavelength=LAM_NEC), extended_kernel=True, **over
    )


def _far_blocks(name, accel, monkeypatch):
    """Every plainly-projected source block's three contribution arrays, with
    the near correction (numpy on both backends, and an OVERWRITE) switched
    off so what is compared is the far fill alone."""
    monkeypatch.setattr(_sg, "_HAVE_GALERKIN_FAR_FILL", accel)
    sim = _gc_solver(name)
    sim.near_correction = False
    geom = sim._build_geometry()
    ctx = sim._test_context(geom, sim._basis_coefs(geom, sim.k), sim.k)
    out = list(sim._tested_contribs(geom, sim.k, ctx, _plain_projection))
    if sim.ground_z is not None and sim.ground_eps is None:
        src_c, src_t = sim._image_source_centers_tangents(geom)
        out += list(
            sim._tested_contribs(
                geom, sim.k, ctx, _plain_projection, src_c, src_t, mirror=True
            )
        )
    return out


@pytest.mark.skipif(not _HAVE_ACCEL, reason="C++ accelerator not built")
@pytest.mark.parametrize("name", list(_GC_DECKS))
def test_gc1_the_cpp_ek_far_fill_matches_the_numpy_one(name, monkeypatch):
    """Measured, worst entry of any block relative to that block's largest:

        fat dipole        1.6e-15
        thin dipole       3.9e-15
        vee               1.9e-15
        PEC ground        2.2e-15
        refl-coef ground  2.2e-15

    A tolerance and not a bit comparison, deliberately (#249's rule): the C++
    fill contracts the test quadrature as it goes and the numpy one contracts
    it afterwards, so the two cannot be bit-equal and pinning them there would
    pin a compiler rather than a kernel.
    """
    cpp = _far_blocks(name, True, monkeypatch)
    npy = _far_blocks(name, False, monkeypatch)
    for i, (c, n) in enumerate(zip(cpp, npy)):
        rel = np.max(np.abs(c - n)) / np.max(np.abs(n))
        assert rel < _GC1_BAR, f"{name} block {i}: {rel:.2e}"


@pytest.mark.skipif(not _HAVE_ACCEL, reason="C++ accelerator not built")
@pytest.mark.parametrize("name", list(_GC_DECKS))
def test_gc1_the_ek_twin_actually_moves_the_far_fill(name, monkeypatch):
    """The control for the gate above: an EK twin that dropped the delta
    would agree with the numpy EK fill only if the numpy fill had dropped it
    too — but it would also equal the REDUCED fill, which this refuses."""
    monkeypatch.setattr(_sg, "_HAVE_GALERKIN_FAR_FILL", True)
    ext = _far_blocks(name, True, monkeypatch)
    sim = SinusoidalGalerkinSolver(**dict(_GC_DECKS[name], wavelength=LAM_NEC))
    sim.near_correction = False
    geom = sim._build_geometry()
    ctx = sim._test_context(geom, sim._basis_coefs(geom, sim.k), sim.k)
    red = list(sim._tested_contribs(geom, sim.k, ctx, _plain_projection))
    rel = np.max(np.abs(ext[0] - red[0])) / np.max(np.abs(red[0]))
    assert rel > 1e-9, f"{name}: EK far fill is the reduced one ({rel:.2e})"


@pytest.mark.skipif(not _HAVE_ACCEL, reason="C++ accelerator not built")
def test_gc2_an_all_ineligible_label_set_gives_the_reduced_fill_to_the_bit():
    """The delta's code path is entered per PAIR, so labels that make nothing
    eligible must leave the reduced arithmetic alone — not to a tolerance, to
    the bit. This is the C++ half of `test_gb3_zero_radius_gives_the_reduced_
    fill_to_the_bit`, and it is what says the twin's reduced half is the
    frozen one.

    Both ways of emptying the set, because momwire#358 moved the predicate
    `g_obs[m] >= 0 and g_obs[m] == g_src[n]` into the kernel and either half
    of it can now be the thing that fails: labels that MISMATCH (0 against 1)
    and labels that carry the never-extend marker on both sides (−1 against
    −1, equal and still ineligible). The third row is the control — labels
    that DO match, with zero radii, which reaches the delta and finds it
    identically zero.
    """
    from momwire._accel import acc

    sim = _gc_solver("fat dipole")
    geom = sim._build_geometry()
    ctx = sim._test_context(geom, sim._basis_coefs(geom, sim.k), sim.k)
    n_obs, n_src = ctx["obs_c"].shape[0], ctx["N"]
    n_test = ctx["starts"].shape[0] - 1
    args = (
        np.ascontiguousarray(ctx["obs_c"]),
        np.ascontiguousarray(ctx["obs_t"]),
        np.full(n_obs, float(sim._uniform_radius)),
        np.ascontiguousarray(geom["seg_centers"]),
        np.ascontiguousarray(geom["seg_tangents"]),
        np.ascontiguousarray(ctx["hh"]),
        float(sim.k),
        float(sim.eta),
        *[np.ascontiguousarray(v) for v in sim._leggauss_cached(sim.n_qp_const)],
        np.ascontiguousarray(ctx["w_entry"], dtype=np.complex128),
        np.ascontiguousarray(ctx["starts"], dtype=np.int64),
    )
    ek_gx, ek_gw = sim._ek_delta_rule(_N_QP_EK_DELTA, 1)
    red = acc.sinusoidal_galerkin_far_fill(*args)
    zeros_t = np.zeros(n_test, dtype=np.int64)
    zeros_s = np.zeros(n_src, dtype=np.int64)
    for why, radius, g_obs, g_src in (
        ("mismatched labels", sim._seg_radius(geom), zeros_t, zeros_s + 1),
        ("the never-extend marker", sim._seg_radius(geom), zeros_t - 1, zeros_s - 1),
        ("eligible, zero radius", np.zeros(n_src), zeros_t, zeros_s),
    ):
        got = acc.sinusoidal_galerkin_far_fill_ek(
            *args,
            np.ascontiguousarray(radius, dtype=np.float64),
            np.ascontiguousarray(g_obs, dtype=np.int64),
            np.ascontiguousarray(g_src, dtype=np.int64),
            np.ascontiguousarray(ek_gx),
            np.ascontiguousarray(ek_gw),
        )
        for a, b in zip(got, red):
            assert np.array_equal(a, b), why


def _ek_selection(sim, geom, ctx, g_obs, g_src):
    """Which (test segment, source segment) pairs the C++ twin actually gave
    the delta to, read back EXACTLY rather than inferred.

    G-C2 is what makes the readback exact: an ineligible pair's entry is bit
    identical to the reduced fill, because the delta's code path is not
    entered for it at all. So `ek != reduced`, cell by cell, IS the eligible
    set — no tolerance anywhere — and folding the entry axis onto its test
    segment turns it back into the (M, N) shape the old mask had.
    """
    from momwire._accel import acc

    args, ek = _gc4_args(sim, geom, ctx)
    payload = list(ek)
    payload[1] = np.ascontiguousarray(g_obs, dtype=np.int64)
    payload[2] = np.ascontiguousarray(g_src, dtype=np.int64)
    red = acc.sinusoidal_galerkin_far_fill(*args)
    got = acc.sinusoidal_galerkin_far_fill_ek(*args, *payload)
    moved = np.zeros(red[0].shape, dtype=bool)
    for a, b in zip(got, red):
        moved |= a != b
    sel = np.zeros((ctx["starts"].shape[0] - 1, ctx["N"]), dtype=bool)
    np.logical_or.at(sel, ctx["m_of_entry"], moved)
    return sel


def _label_patterns(sim, geom):
    """Adversarial (observer, source) label pairs for the predicate gate."""
    n = geom["n_segs"]
    z = np.zeros(n, dtype=np.int64)
    plain, plain_src = sim._ek_axis_labels(geom, False)
    out = {
        # The two controls: everything and nothing.
        "all eligible": (z, z.copy()),
        # Equal labels that must STILL be ineligible — the `>= 0` half of the
        # predicate, which a kernel that only compared labels would drop and
        # no shift or symmetry gate would notice.
        "the never-extend marker, both sides": (z - 1, z - 1),
        "never-extend observers against real sources": (z - 1, z.copy()),
        "never-extend sources against real observers": (z.copy(), z - 1),
        # Mixed −1s: alternate observers opt out, sources all in one group.
        "alternating never-extend observers": (
            np.where(np.arange(n) % 2 == 0, 0, -1).astype(np.int64),
            z.copy(),
        ),
        # A junction fan: three groups, permuted between the two axes, so the
        # eligible set is a scattered pattern and not a block.
        "three groups, permuted": (
            (np.arange(n) % 3).astype(np.int64),
            ((np.arange(n) + 1) % 3).astype(np.int64),
        ),
        "three groups, aligned": (
            (np.arange(n) % 3).astype(np.int64),
            (np.arange(n) % 3).astype(np.int64),
        ),
        # An observer group with no source in it at all.
        "observer group absent from the sources": (z + 7, z.copy()),
        # The real thing, unmirrored.
        "the deck's own labels": (plain, plain_src),
    }
    # And mirrored, where the deck has a ground to mirror through. That pair
    # is the asymmetric one: `_ek_axis_labels(geom, True)` scans the real and
    # reflected segments TOGETHER and splits the result, so its two halves are
    # different arrays and passing either one twice is a different eligible
    # set — which is exactly what the swapped row below shows.
    if sim.ground_z is not None:
        real, image = sim._ek_axis_labels(geom, True)
        out["the deck's own MIRRORED labels"] = (real, image)
        out["mirrored labels, halves swapped"] = (image, real)
    return out


@pytest.mark.skipif(not _HAVE_ACCEL, reason="C++ accelerator not built")
@pytest.mark.parametrize("deck", ["fat dipole", "vee", "PEC ground"])
def test_gc2_the_in_kernel_predicate_is_the_mask_formula(deck):
    """momwire#358's whole correctness claim, pinned per pair.

    The twin used to be handed `_ek_pairs`' materialized (n_obs, N) mask; it
    is now handed the group labels and scores `g_obs[m] >= 0 and g_obs[m] ==
    g_src[n]` itself. So the eligible set it reaches must equal that formula
    evaluated in numpy — the mask builder's own expression, spelled here
    exactly as `_ek_pairs` spells it — on every label pattern that could tell
    the two apart: mixed −1s, permuted junction-fan groups, an observer group
    with no sources in it, and the mirror-asymmetric label pair whose two
    halves are different arrays.
    """
    sim = _gc_solver(deck)
    geom = sim._build_geometry()
    ctx = sim._test_context(geom, sim._basis_coefs(geom, sim.k), sim.k)

    seen = set()
    for why, (g_obs, g_src) in _label_patterns(sim, geom).items():
        expect = (g_obs[:, None] == g_src[None, :]) & (g_obs[:, None] >= 0)
        got = _ek_selection(sim, geom, ctx, g_obs, g_src)
        assert np.array_equal(got, expect), (
            f"{deck} / {why}: kernel selected {got.sum()} pairs, "
            f"the mask formula {expect.sum()}"
        )
        seen.add(expect.tobytes())

    # The gate is only worth anything if the patterns disagree with each other
    # and if both answers occur — an all-True and an all-False readback would
    # pass against a kernel that ignored the labels entirely.
    assert len(seen) > 4, "the label patterns did not separate"


@pytest.mark.skipif(not _HAVE_ACCEL, reason="C++ accelerator not built")
def test_gc2_the_selection_readback_is_not_vacuous():
    """The control for the gate above: its two extremes really are extreme, so
    `ek != reduced` is measuring the delta's presence and not float noise."""
    sim = _gc_solver("fat dipole")
    geom = sim._build_geometry()
    ctx = sim._test_context(geom, sim._basis_coefs(geom, sim.k), sim.k)
    n = geom["n_segs"]
    z = np.zeros(n, dtype=np.int64)
    assert _ek_selection(sim, geom, ctx, z, z).all(), "no pair took the delta"
    assert not _ek_selection(sim, geom, ctx, z - 1, z - 1).any(), "every pair did"


@pytest.mark.skipif(not _HAVE_ACCEL, reason="C++ accelerator not built")
def test_gc2_the_twin_checks_its_label_lengths():
    """The mask carried its own shape and the kernel checked it against the
    observer rows; the labels carry one length each, and getting either wrong
    would read off the end of the array rather than mis-select a pair
    (momwire#358). So both are refused."""
    from momwire._accel import acc

    sim = _gc_solver("fat dipole")
    geom = sim._build_geometry()
    ctx = sim._test_context(geom, sim._basis_coefs(geom, sim.k), sim.k)
    args, ek = _gc4_args(sim, geom, ctx)
    n_test, n_src = ek[1].shape[0], ek[2].shape[0]
    assert n_test == ctx["starts"].shape[0] - 1 and n_src == ctx["N"]
    bad = {
        "obs labels per observer ROW": (1, np.zeros(ctx["obs_c"].shape[0], np.int64)),
        "obs labels one short": (1, np.zeros(n_test - 1, np.int64)),
        "src labels one long": (2, np.zeros(n_src + 1, np.int64)),
    }
    for why, (slot, labels) in bad.items():
        payload = list(ek)
        payload[slot] = labels
        try:
            acc.sinusoidal_galerkin_far_fill_ek(*args, *payload)
        except (RuntimeError, TypeError, ValueError):
            continue
        pytest.fail(f"{why}: accepted")


@pytest.mark.skipif(not _HAVE_ACCEL, reason="C++ accelerator not built")
@pytest.mark.parametrize("name", list(_G4_DECKS))
def test_gc2_ek_off_never_reaches_the_twin(name, monkeypatch):
    """Numerical identity is necessary and not sufficient (the G-B4
    argument), so the counter is the other half here too: with the extended
    kernel off, the EK symbol is not called even once."""
    from momwire._accel import acc

    calls = []
    original = acc.sinusoidal_galerkin_far_fill_ek
    monkeypatch.setattr(
        acc,
        "sinusoidal_galerkin_far_fill_ek",
        lambda *a, **kw: (calls.append(1), original(*a, **kw))[1],
        raising=False,
    )
    kw = dict(_G4_DECKS[name], wavelength=LAM_NEC)
    SinusoidalGalerkinSolver(**kw).compute_impedance()
    assert calls == []


@pytest.mark.skipif(not _HAVE_ACCEL, reason="C++ accelerator not built")
def test_gc3_the_fused_path_is_what_an_ek_solve_takes(monkeypatch):
    """The control for the two gates above: on a deck the accelerator serves,
    an EK-on solve DOES reach the twin. A capability flag that never flipped,
    or a `_tested_contribs` that kept falling back, would make them vacuous.
    """
    from momwire._accel import acc

    calls = []
    original = acc.sinusoidal_galerkin_far_fill_ek
    monkeypatch.setattr(
        acc,
        "sinusoidal_galerkin_far_fill_ek",
        lambda *a, **kw: (calls.append(1), original(*a, **kw))[1],
        raising=False,
    )
    _monopole(extended_kernel=True, ground_z=0.0).compute_impedance()
    assert len(calls) == 2, calls  # the free-space block and the PEC image


@pytest.mark.skipif(not _HAVE_ACCEL, reason="C++ accelerator not built")
@pytest.mark.parametrize("name", list(_GC_DECKS))
def test_gc3_the_two_backends_solve_the_same_deck(name, monkeypatch):
    """End to end, near correction and all: |ΔZ|/|Z| measured 3.5e-15 at
    worst over these decks (2.7e-14 on the thinnest ladder rung, where the
    delta's own near-cancellation floor is widest)."""
    monkeypatch.setattr(_sg, "_HAVE_GALERKIN_FAR_FILL", True)
    z_cpp, c_cpp = _gc_solver(name).compute_impedance()
    monkeypatch.setattr(_sg, "_HAVE_GALERKIN_FAR_FILL", False)
    z_npy, c_npy = _gc_solver(name).compute_impedance()
    assert abs(z_cpp - z_npy) < 1e-12 * abs(z_npy), (name, z_cpp, z_npy)
    assert np.allclose(c_cpp, c_npy, rtol=1e-10, atol=1e-14)


# ===========================================================================
# G-C4/G-C5 — the fill folds into the caller's triple (momwire#356)
# ===========================================================================
# Both fused entry points took no destination, so they allocated and returned
# three (nnz, N) arrays whatever the caller wanted. A grounded accelerated
# block therefore floored at TWO triples live — the free-space destination
# plus the kernel's return — where the numpy path has held one since #332
# unit C. `out=` (three arrays) plus `scale` fixes that: each finished entry
# is folded on as `out += scale * value`, scale on the LEFT.
#
#   G-C4  the fold is the allocating call's arithmetic, to the bit: scale 1
#         into zeros is the return value, scale −1 off a triple is
#         `np.subtract`, and a complex scale is `np.multiply(scale, value)`
#         — the operand order `sinusoidal.py`'s C2 convention pins. Plus the
#         default preservation both halves of this rest on: `out=None` is the
#         allocating path, and `scale` without `out` is inert.
#   G-C5  the residency the whole thing is for, in triples.


def _gc4_args(sim, geom, ctx):
    """The positional argument tuple both entry points take, and the EK
    payload the twin takes after it — built exactly as `_far_fill_accel`
    does, so what these gates exercise is the shipped call."""
    n_obs, n_src = ctx["obs_c"].shape[0], ctx["N"]
    args = (
        np.ascontiguousarray(ctx["obs_c"]),
        np.ascontiguousarray(ctx["obs_t"]),
        np.full(n_obs, float(sim._uniform_radius)),
        np.ascontiguousarray(geom["seg_centers"]),
        np.ascontiguousarray(geom["seg_tangents"]),
        np.ascontiguousarray(ctx["hh"]),
        float(sim.k),
        float(sim.eta),
        *[np.ascontiguousarray(v) for v in sim._leggauss_cached(sim.n_qp_const)],
        np.ascontiguousarray(ctx["w_entry"], dtype=np.complex128),
        np.ascontiguousarray(ctx["starts"], dtype=np.int64),
    )
    ek_gx, ek_gw = sim._ek_delta_rule(_N_QP_EK_DELTA, 1)
    # All-eligible, in the label vocabulary momwire#358 gave the twin: one
    # label per TEST segment and one per source segment, all equal and all
    # non-negative.
    n_test = ctx["starts"].shape[0] - 1
    ek = (
        np.ascontiguousarray(sim._seg_radius(geom), dtype=np.float64),
        np.zeros(n_test, dtype=np.int64),
        np.zeros(n_src, dtype=np.int64),
        np.ascontiguousarray(ek_gx),
        np.ascontiguousarray(ek_gw),
    )
    return args, ek


@pytest.mark.skipif(not _HAVE_ACCEL, reason="C++ accelerator not built")
@pytest.mark.parametrize("twin", [False, True], ids=["reduced", "ek"])
def test_gc4_the_fold_is_the_allocating_fill_to_the_bit(twin):
    """`out=`/`scale` may not be a second arithmetic (momwire#356).

    The fold applies `scale` ONCE per entry, to the finished test-quadrature
    sum, which is why the kernel accumulates into a per-test-segment scratch
    band when a destination is given. Folding node by node instead would be
    `((dst − t1) − t2) − …` against the caller's `dst − (t1 + t2 + …)`: a
    reassociation, and one no tolerance-level gate would ever notice. So all
    three comparisons below are `array_equal`.

    The complex case is `np.multiply(scale, value)` and NOT `value * scale`:
    complex128 multiply evaluates the imaginary part as `x.re*y.im +
    x.im*y.re`, so the operand order moves the last bit, and the whole reason
    `scale` exists is the Sommerfeld ground's C2 — which `sinusoidal.py`
    documents as being on the LEFT.
    """
    from momwire._accel import acc

    fill = (
        acc.sinusoidal_galerkin_far_fill_ek
        if twin
        else acc.sinusoidal_galerkin_far_fill
    )
    sim = _gc_solver("fat dipole")
    geom = sim._build_geometry()
    ctx = sim._test_context(geom, sim._basis_coefs(geom, sim.k), sim.k)
    args, ek = _gc4_args(sim, geom, ctx)
    args = args + ek if twin else args

    ref = fill(*args)

    zeros = tuple(np.zeros_like(a) for a in ref)
    got = fill(*args, out=zeros, scale=1.0)
    assert all(g is z for g, z in zip(got, zeros)), "out= must return out"
    for a, b in zip(ref, zeros):
        assert np.array_equal(a, b)

    # scale −1 off a triple of the fill's own magnitude: the ground fold.
    rng = np.random.default_rng(356)
    base = tuple(
        (rng.standard_normal(a.shape) + 1j * rng.standard_normal(a.shape))
        * np.max(np.abs(a))
        for a in ref
    )
    folded = tuple(b.copy() for b in base)
    fill(*args, out=folded, scale=-1.0)
    for b, r, f in zip(base, ref, folded):
        assert np.array_equal(b - r, f)

    # A complex scale, spelled the C2 way.
    c2 = complex(0.3129384756, -0.7182736451)
    scaled = tuple(np.zeros_like(a) for a in ref)
    fill(*args, out=scaled, scale=c2)
    for r, s in zip(ref, scaled):
        assert np.array_equal(np.multiply(c2, r), s)


@pytest.mark.skipif(not _HAVE_ACCEL, reason="C++ accelerator not built")
@pytest.mark.parametrize("twin", [False, True], ids=["reduced", "ek"])
def test_gc4_the_defaults_are_the_pre_356_call(twin):
    """The additive half of G-C4, and what lets G-C2's byte-freeze survive a
    signature change: `out=None` allocates and returns as it always did, and
    `scale` alone is inert — there is no destination for it to weight, and a
    caller who passes one without the other must not silently get a scaled
    fill back.
    """
    from momwire._accel import acc

    fill = (
        acc.sinusoidal_galerkin_far_fill_ek
        if twin
        else acc.sinusoidal_galerkin_far_fill
    )
    sim = _gc_solver("fat dipole")
    geom = sim._build_geometry()
    ctx = sim._test_context(geom, sim._basis_coefs(geom, sim.k), sim.k)
    args, ek = _gc4_args(sim, geom, ctx)
    args = args + ek if twin else args

    ref = fill(*args)
    for kw in ({}, {"out": None}, {"scale": 1.0}, {"scale": complex(-4.0, 9.0)}):
        for a, b in zip(ref, fill(*args, **kw)):
            assert np.array_equal(a, b), kw


@pytest.mark.skipif(not _HAVE_ACCEL, reason="C++ accelerator not built")
def test_gc4_a_destination_the_kernel_cannot_write_is_refused():
    """The kernel folds into `out`'s buffer directly, so anything pybind11
    would have had to COPY to accept — wrong dtype, wrong shape, a non-C
    layout, a read-only array — has to raise rather than land the fold in the
    copy and hand the caller back an untouched destination.
    """
    from momwire._accel import acc

    sim = _gc_solver("fat dipole")
    geom = sim._build_geometry()
    ctx = sim._test_context(geom, sim._basis_coefs(geom, sim.k), sim.k)
    args, _ek = _gc4_args(sim, geom, ctx)
    ref = acc.sinusoidal_galerkin_far_fill(*args)
    shape = ref[0].shape

    def _ok():
        return [np.zeros(shape, dtype=np.complex128) for _ in range(3)]

    bad = {
        "two arrays": _ok()[:2],
        "float dtype": _ok()[:2] + [np.zeros(shape, dtype=np.float64)],
        "wrong shape": _ok()[:2] + [np.zeros((shape[0] + 1, shape[1]), np.complex128)],
        "fortran order": _ok()[:2] + [np.zeros(shape, np.complex128, order="F")],
        "a transposed view": _ok()[:2] + [np.zeros(shape[::-1], np.complex128).T],
        "read-only": _ok()[:2] + [_readonly(np.zeros(shape, np.complex128))],
    }
    for why, out in bad.items():
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            acc.sinusoidal_galerkin_far_fill(*args, out=out)
        assert all(not a.any() for a in out[:2]), f"{why}: it wrote anyway"


def _readonly(a):
    a.flags.writeable = False
    return a


# The ground block's OWN peak, in triples of 3 x 16 x nnz x N, extended
# kernel on, at the N = 300 bend `_gd8_bend` builds. Measured, main -> #356:
#
#   PEC image         1.18 -> 0.23      (15.20 -> 3.00 MB of a 12.93 MB triple)
#   PEC image, N=400  1.18 -> 0.23      (27.03 -> 5.31 MB of a 23.00 MB triple)
#
# and the two grounds this does not move, for the record:
#
#   refl-coef         1.90 -> 1.90      (numpy projector: never came here)
#   sommerfeld        2.30 -> 2.30      (N=400: 1.73 -> 1.73)
#
# The Sommerfeld floor is structural under `out=` and is called out in
# `_fold_ground_block`: `c2*img - rem` has to stay associated, so the image
# triple must exist WHOLE before the fold can start. Banding the fill over
# observers is what would fold it, and that is momwire#356's option 2.
#
# 0.6 sits 2.6x above the folded number and 2.0x below the allocating one —
# far more headroom than the 1.3x momwire#347 asks for, because this fix goes
# from two triples to one rather than trimming a transient.
_GC5_BAR = 0.6


@pytest.mark.memgate
@pytest.mark.skipif(not _HAVE_ACCEL, reason="C++ accelerator not built")
@pytest.mark.parametrize("n", [300, 400])
def test_gc5_the_grounded_accelerated_fold_holds_no_triple(monkeypatch, n):
    """Tracemalloc gate on `_fold_ground_block` itself (momwire#356).

    The BLOCK and not the whole assembly, deliberately. With the extended
    kernel on, `_assemble_Z`'s peak at these sizes used to be the near
    correction's per-pair working set — 5.14 MB per pair, fixed against N,
    41 MB at `_PAIR_BLOCK` 8 and 2.63 GB at the 512 that shipped — and it
    buried a triple either way: the whole-assembly number read 4.10 -> 4.15
    at N = 300, i.e. noise. That is G-D8c's finding repeated, and it is why
    the subject is measured on its own here.

    momwire#383 has since byte-budgeted that block (G-D9), so the shrink
    below now buys less than it did: `_near_block` already returns one pair
    under the extended kernel at this deck's rule, and `_PAIR_BLOCK` = 8 is
    a ceiling it is under. The monkeypatch stays because the ceiling is
    what this gate means to pin — that no future block size can bury its
    subject — and because it is what the number quoted below was taken at.

    What is left inside the bar is not the fill. momwire#358 took the
    eligibility mask and its build temporary out of it — 1.18 -> 0.23 -> 0.176
    triples at N = 300, 0.23 -> 0.175 at N = 400 — and G-C6 below measures
    what remains without the near correction in the way: 0.009 triples, i.e.
    nothing of matrix shape at all. The 0.176 this gate sees is
    `_apply_near_correction`'s own N^2 pair search (`_near_pairs`), which is
    neither the fill nor #356's or #358's subject.
    """
    import tracemalloc

    from momwire import sinusoidal_galerkin as _sg

    monkeypatch.setattr(_sg, "_PAIR_BLOCK", 8)

    sim = _gd8_bend(n, ground_z=0.0)
    geom = sim._build_geometry()
    N = geom["n_segs"]
    assert N == n
    ctx = sim._test_context(geom, sim._basis_coefs(geom, sim.k), sim.k)
    triple = 3 * 16 * ctx["w_entry"].shape[0] * N
    contribs = sim._tested_contribs(geom, sim.k, ctx, _plain_projection)

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        sim._fold_ground_block(geom, sim.k, ctx, contribs)
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < _GC5_BAR * triple, (
        f"N={n}: the PEC image block peaked {peak / 1e6:.2f} MB = "
        f"{peak / triple:.2f} triples (one triple = {triple / 1e6:.2f} MB), "
        f"bar {_GC5_BAR} — the fused fill is allocating its own again"
    )


# ===========================================================================
# G-C6 — the eligibility payload is not of matrix shape (momwire#358)
# ===========================================================================
# The EK twin took its eligibility as an (n_obs, N) bool mask — n_obs = N*nq
# rows, so 8 bytes per matrix cell of RESIDENT input, 11.5 MB at N = 1200 —
# built in Python as `(gi == gj) & (gi >= 0)`, which materializes the `==`
# result first and so peaks at two of them. It now takes the group labels
# themselves, one per test segment and one per source segment, and scores the
# same predicate per pair inside the sweep.
#
# G-C5 above cannot see the whole of that: with the near correction on, its
# number is dominated by `_near_pairs`' own N^2 search. So this gate measures
# the grounded fill with the near correction OFF — G-C1's isolation, for
# G-C1's reason — where what is left IS the fill's own working set. Measured
# on the same `_gd8_bend` decks, main (61e22da) -> this branch:
#
#   N=300   1.50 MB = 0.1157 triples  ->  0.11 MB = 0.0087 triples
#   N=400   2.63 MB = 0.1144 triples  ->  0.15 MB = 0.0065 triples
#
# The 1.39 MB and 2.45 MB that went are 2 x 8 x N^2 to the byte: the mask and
# the `==` temporary under it. What is left is O(N) — the `ascontiguousarray`
# copies of the observer tables and the delta rule — so it FALLS as a fraction
# of the triple with N, which is the shape of the claim.
#
# 0.03 sits 3.4x above the folded number and 3.9x below main's, and unlike
# G-C5's bar it is red on main: it is a gate on this change specifically, so
# it is pinned against the value main actually measured rather than against a
# round number. (Red on main is checkable here even though the new C++
# signature is not, because this is a residency measurement of main's own
# code, taken with main's own .so.)
_GC6_BAR = 0.03


@pytest.mark.memgate
@pytest.mark.skipif(not _HAVE_ACCEL, reason="C++ accelerator not built")
@pytest.mark.parametrize("n", [300, 400])
def test_gc6_the_fused_ek_fill_holds_nothing_of_matrix_shape(monkeypatch, n):
    """Tracemalloc gate on the grounded EK fill alone (momwire#358).

    Near correction off, so the subject is the fused fill and its payload and
    not `_apply_near_correction`'s pair search — the same isolation
    `_far_blocks` makes for G-C1, and the reason G-C5's number stalls at 0.176
    while this one reads 0.009.
    """
    import tracemalloc

    from momwire import sinusoidal_galerkin as _sg

    monkeypatch.setattr(_sg, "_PAIR_BLOCK", 8)

    sim = _gd8_bend(n, ground_z=0.0)
    sim.near_correction = False
    geom = sim._build_geometry()
    N = geom["n_segs"]
    assert N == n
    ctx = sim._test_context(geom, sim._basis_coefs(geom, sim.k), sim.k)
    triple = 3 * 16 * ctx["w_entry"].shape[0] * N
    contribs = sim._tested_contribs(geom, sim.k, ctx, _plain_projection)

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        sim._fold_ground_block(geom, sim.k, ctx, contribs)
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < _GC6_BAR * triple, (
        f"N={n}: the PEC image fill peaked {peak / 1e6:.3f} MB = "
        f"{peak / triple:.4f} triples (one triple = {triple / 1e6:.2f} MB), "
        f"bar {_GC6_BAR} — something of matrix shape is back in the payload"
    )


# ===========================================================================
# G-S — the extended kernel over the SOMMERFELD ground (momwire#287)
# ===========================================================================
#
# #246 unit B narrowed #233's blanket refusal down to one combination —
# `extended_kernel=True` with `ground_model="sommerfeld"` — and refused it
# loudly, because that ground is served as TWO evaluators (NEC's eqs 143-147:
# the C2-scaled exact image plus a smooth interpolated ground-wave remainder)
# and only the first of them is an Eqs 76-79 build the delta rides through.
# Extending the image while the remainder stayed reduced would have been a
# silent mixture of two kernels inside one ground model.
#
# The refusal is now gone, and the mixture is not: the image half carries the
# delta, the remainder stays REDUCED, and G-S1 measures what that costs
# instead of asserting that it is small. That is the same answer
# `BSplineSolver` reached in #269 (its Gate 18), reached again here under this
# fill's own test quadrature rather than inherited — #246's G-B1 found the
# Galerkin EK shift runs LARGER than the point-matched one at thin Δ/a because
# test integrals sweep the segment-end region collocation never samples, and
# the same mechanism could plausibly have enlarged the mixture's visible cost.
# Measured, it does not: the numbers land within a factor of 1.2 of the
# B-spline family's on every deck.
#
#   G-S1  the extended remainder built outright, and the cost of not
#         shipping it — per deck, per ground, converged in the ring count,
#         and O(a²) at the contact where the analytic estimate degenerates;
#   G-S2  δZ = Z(EK on) − Z(EK off) over Sommerfeld inside #249 §7's 25 %
#         cross-basis bar against `BSplineSolver` and against the
#         point-matched solver, fat decks clear of and in contact with the
#         plane, with the deck's own PEC mismatch as the control;
#   G-S3  ‖G−Gᵀ‖/‖G‖ with EK on over Sommerfeld, with #246's asymmetric-mask
#         falsifying contrast;
#   G-S4  a → 0 collapse on a Sommerfeld deck, and the EK-ON half of G-B4's
#         EK-off bit-identity statement;
#   G-S5  the image half really is extended — mutating only the mirrored
#         pair mask moves the Sommerfeld answer.

from momwire import _sommerfeld as _sm  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402

_S_EPS = {"soil": (13.0, 0.005), "sea": (81.0, 5.0)}
_S_H = LAM_NEC / 4
_S_NS = 21


def _s_decks():
    """G-S1's decks. Three are `test_extended_kernel_bspline.py`'s Gate-18
    decks verbatim, so the two families' numbers are comparable line by line;
    the fourth is not, and the substitution is deliberate.

    #269's fourth deck is a BENT fat wire above the plane, and when these
    gates landed this solver could not use it: the Galerkin EK fill was wrong
    on any multi-edge wire whose nodes split, growing as the wire thinned
    instead of collapsing (momwire#299, diagnosed as the end-bracket residue
    and fixed by the NODE predicate G-D gates). `slanted` is the bent deck's
    FIRST ARM alone: the same fat radius, the same tilt (so p̂ rotates pair by
    pair and min r₁ still sits just above 2h), no bend.

    The substitution STAYS, and that is now a choice rather than a
    constraint. With #299 in, the bent deck's EK shift agrees with
    `BSplineSolver`'s to 11.7 % over PEC and 10.3 % over Sommerfeld soil
    (+11.52 − 4.97j against +11.94 − 6.50j, and +13.04 − 2.41j against
    +13.33 − 3.81j) — inside #249 §7's 25 % bar, so the deck is usable
    again. Restoring it would re-baseline every measured number in G-S1's
    table and G-S2's, which belong to momwire#287 and are not this arc's to
    move; the bent deck is gated instead by
    `test_gd7_the_bent_deck_of_287_is_usable_again` below, on both grounds.
    """
    return {
        # A quarter-wave monopole STANDING IN the plane — NEC's IND = 0
        # ground-contact branch, and the deck where the (a/2h)² estimate
        # degenerates because r₁ → 0.
        "mono_contact": dict(
            wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, _S_H]])],
            n_per_edge_per_wire=[[_S_NS]],
            nsegs=_S_NS,
            wire_radius=0.02,
            feed_arclength=(_S_H / _S_NS) * 0.5,
            ground_z=0.0,
        ),
        # The same wire lifted clear of the plane: the image is a real
        # distance away and both ends are ordinary IND = 1.
        "mono_lifted": dict(
            wires=[np.array([[0.0, 0.0, 0.4], [0.0, 0.0, 2.9]])],
            n_per_edge_per_wire=[[_S_NS]],
            nsegs=_S_NS,
            wire_radius=0.02,
            feed_arclength=(2.5 / _S_NS) * 10.5,
            ground_z=0.0,
        ),
        # Horizontal at height: the image is PARALLEL, not coaxial, so the
        # image block itself stays reduced and what G-S1 measures on this deck
        # is the remainder alone.
        "horizontal": dict(
            wires=[np.array([[-2.5, 0.0, 1.5], [2.5, 0.0, 1.5]])],
            n_per_edge_per_wire=[[_S_NS]],
            nsegs=_S_NS,
            wire_radius=0.05,
            feed_arclength=(5.0 / _S_NS) * 10.5,
            ground_z=0.0,
        ),
        # Fat (Δ/a = 4.9) and tilted: neither coaxial with its image nor
        # parallel to the plane, and close enough to it that min r₁ = 0.627
        # against 2h = 0.600.
        "slanted": dict(
            wires=[np.array([[0.0, 0.0, 0.3], [2.0, 0.0, 1.5]])],
            n_per_edge_per_wire=[[6]],
            nsegs=6,
            wire_radius=0.08,
            feed_arclength=(float(np.hypot(2.0, 1.2)) / 6.0) * 3.5,
            ground_z=0.0,
        ),
    }


_S_DECKS = _s_decks()


def _s_solve(deck, eps, cls=SinusoidalGalerkinSolver, ek=True, **over):
    kw = dict(
        _S_DECKS[deck],
        wavelength=LAM_NEC,
        extended_kernel=ek,
        ground_eps=_S_EPS[eps],
        ground_model="sommerfeld",
    )
    kw.update(over)
    if cls is BSplineSolver:
        kw.update(degree=2, feed_model="segment")
    z, _ = cls(**kw).compute_impedance()
    return complex(z)


# ---------------------------------------------------------------------------
# G-S1 — the Sommerfeld remainder stays reduced, and that is MEASURED
# ---------------------------------------------------------------------------
#
# THE ARITHMETIC. EK is the azimuthal average of the source over a tube of
# radius a: an O((a/R)²) correction, precisely |fac − 1| ≈ (a²/2R²)|1 + jkR|
# for a ≪ R. Every term in the remainder has the ground REFLECTION as its
# source, so its R is the image distance r₁ = √(ρ² + (z+z')²) ≥ 2h for a wire
# at height h, and the un-applied correction is O((a/2h)²). Measured min r₁
# over this fill's remainder quadrature confirms the floor is tight
# (mono_lifted 0.816 against 2h = 0.800; horizontal 3.000 against 3.000;
# slanted 0.627 against 0.600) — and says nothing at all at a CONTACT, where
# r₁ → 0 and only the SMOOTHNESS of the remainder bounds it. That is the
# whole point of NEC's decomposition (the C2 image absorbs the singular part),
# and it is why this gate measures rather than estimates.
#
# THE MEASUREMENT. Build the extended remainder outright — the same
# `_sommerfeld.remainder_field_proj` the shipped code calls, evaluated on a
# ring of `n_phi` source points at radius a about each source axis point and
# averaged, which IS the tube average EK applies — and re-solve. The cost of
# shipping the reduced one is the difference in the solved Z:
#
#   deck            a       min r1   2h      |ΔZ|/|Z| soil   sea        EK shift
#   mono_contact    0.020   0.0158   0.000   3.498e-03   4.490e-03   5.3e-3
#   mono_lifted     0.020   0.8158   0.800   9.503e-07   9.250e-08   5.2e-3
#   horizontal      0.050   3.0000   3.000   3.957e-05   4.202e-06   2.1e-2
#   slanted         0.080   0.6273   0.600   1.024e-05   1.331e-06   2.4e-2
#
# (the last column is that deck's own EK shift over soil, |Z_on − Z_off|/|Z|,
# i.e. what the EXTENDED half of the same ground model is worth). Converged in
# n_phi — 4, 8 and 16 agree to every digit shown — and O(a²) as the expansion
# says: on mono_contact/soil, a = 0.04 / 0.02 / 0.01 / 0.005 gives
# 8.506e-3 / 3.498e-3 / 1.170e-3 / 3.291e-4, ratios 2.4 / 3.0 / 3.6 → 4.
#
# HOW TO READ IT. Clear of the plane the un-modelled term is ≤ 4.0e-5
# relative — three orders below the EK shift the image blocks DO carry on
# those same decks, and two below this fill's own cross-basis Z gap. AT GROUND
# CONTACT it is 3.5-4.5e-3, which is 66 % (soil) / 55 % (sea) of that deck's
# own EK shift: the one place the mixture is visible. It is still an order
# below the cross-basis Z gap there (3.6 % against `BSplineSolver`), and
# refl-coef is invalid at a contact (#153) so Sommerfeld is the only model
# available — refusing it would cost the consumer more than 0.4 % of |Z|.
#
# AGAINST THE B-SPLINE PRECEDENT (#269 Gate 18, which measured 3.03e-3 /
# 3.83e-3 at contact, 9.60e-7 / 9.30e-8 lifted, 3.94e-5 / 4.15e-6
# horizontal): every cell lands within a factor of 1.2. #246's G-B1 warning —
# that Galerkin test integrals sweep the segment-end region and can enlarge an
# EK-scale effect — does not reach this one, because the remainder's spatial
# scale is r₁ ≥ 2h and not the segment end.

_GS1_TOL = {
    "mono_contact": 8e-3,
    "mono_lifted": 3e-6,
    "horizontal": 1e-4,
    "slanted": 3e-5,
}
_GS1_NPHI = 8


def _tube_averaged_proj(a, n_phi=_GS1_NPHI):
    """`_sommerfeld.remainder_field_proj` with the SOURCE points spread onto
    a ring of radius `a` about the source axis and averaged — NEC Eq 89's tube
    average, applied to the smooth remainder field the shipped code leaves on
    the axis. `test_extended_kernel_bspline.py`'s Gate-18 helper, transferred
    verbatim: the Galerkin fill consumes the SAME evaluator through
    `_tested_sommerfeld_remainder`, only with the test-quadrature points as
    its observer set.
    """
    original = _sm.remainder_field_proj

    def ring(obs, t_obs, src, t_src, gz, k, grid, cancel_flag=0):
        ref = np.tile(np.array([0.0, 0.0, 1.0]), (t_src.shape[0], 1))
        ref[np.abs(t_src[:, 2]) > 0.9] = np.array([1.0, 0.0, 0.0])
        e1 = np.cross(t_src, ref)
        e1 /= np.linalg.norm(e1, axis=1)[:, None]
        e2 = np.cross(t_src, e1)
        total = None
        for i in range(n_phi):
            phi = 2.0 * np.pi * i / n_phi
            s = src + a * (np.cos(phi) * e1 + np.sin(phi) * e2)
            v = original(obs, t_obs, s, t_src, gz, k, grid, cancel_flag)
            total = v if total is None else total + v
        return total / n_phi

    return original, ring


def _gs1_cost(monkeypatch, deck, eps, n_phi=_GS1_NPHI, radius=None):
    """|ΔZ|/|Z| between shipping the reduced remainder and the extended one."""
    over = {} if radius is None else dict(wire_radius=radius)
    a = float(radius if radius is not None else _S_DECKS[deck]["wire_radius"])
    z_axis = _s_solve(deck, eps, **over)
    original, ring = _tube_averaged_proj(a, n_phi)
    assert original is not ring
    monkeypatch.setattr(_sm, "remainder_field_proj", ring)
    z_tube = _s_solve(deck, eps, **over)
    monkeypatch.setattr(_sm, "remainder_field_proj", original)
    return abs(z_tube - z_axis) / abs(z_axis)


@pytest.mark.parametrize("eps", list(_S_EPS))
@pytest.mark.parametrize("deck", list(_S_DECKS))
def test_gs1_reduced_sommerfeld_remainder_is_negligible(monkeypatch, deck, eps):
    rel = _gs1_cost(monkeypatch, deck, eps)
    assert rel <= _GS1_TOL[deck], (
        f"{deck} / {eps}: extending the Sommerfeld remainder moves Z by "
        f"{rel:.3e} > {_GS1_TOL[deck]:.3e} — the reduced-remainder mixture is "
        f"no longer negligible on this deck"
    )


@pytest.mark.parametrize("deck", ["mono_contact", "slanted"])
def test_gs1_the_ring_count_is_converged(monkeypatch, deck):
    """The tube average is a quadrature too, and G-S1's numbers are only a
    measurement if it has converged. 4 and 8 nodes agree to 4e-3 relative on
    the contact deck (identically, in fact) and 3.5e-3 on the slanted one."""
    coarse = _gs1_cost(monkeypatch, deck, "soil", n_phi=4)
    fine = _gs1_cost(monkeypatch, deck, "soil", n_phi=8)
    assert abs(fine - coarse) <= 0.01 * fine, (deck, coarse, fine)


def test_gs1_the_contact_cost_is_o_a2(monkeypatch):
    """The one deck where the analytic O((a/2h)²) estimate says nothing, made
    to say it anyway: the measured cost still falls like a². Ratios 2.4 / 3.0 /
    3.6 over a = 0.04 → 0.005, approaching 4 from below as the expansion's
    higher orders drop out."""
    prev = None
    for a in (0.04, 0.02, 0.01, 0.005):
        rel = _gs1_cost(monkeypatch, "mono_contact", "soil", radius=a)
        if prev is not None:
            assert 2.0 < prev / rel < 4.5, (a, prev, rel)
        prev = rel


def test_gs1_the_tube_average_is_a_real_perturbation(monkeypatch):
    """The control. With the ring installed at a radius the solve can
    actually feel, G-S1's numbers must MOVE — otherwise the gate would pass on
    a no-op monkeypatch (a route change that stopped calling
    `remainder_field_proj` at all, say) and prove nothing. Measured 1.2e-1 at
    25x the real radius."""
    rel = _gs1_cost(monkeypatch, "mono_contact", "soil")
    assert rel < 1e-2, rel
    z_axis = _s_solve("mono_contact", "soil")
    original, ring = _tube_averaged_proj(0.5)  # 25x the deck's radius
    monkeypatch.setattr(_sm, "remainder_field_proj", ring)
    z_fat = _s_solve("mono_contact", "soil")
    monkeypatch.setattr(_sm, "remainder_field_proj", original)
    assert abs(z_fat - z_axis) / abs(z_axis) > 1e-2, "the ring is not reaching"


# ---------------------------------------------------------------------------
# G-S2 — the cross-basis shift bar, over the Sommerfeld ground
# ---------------------------------------------------------------------------
#
# #249 §7's 25 % bar on δZ = Z(EK on) − Z(EK off), applied over Sommerfeld and
# scored against the two solvers that already serve this combination:
# `BSplineSolver` (since #269) and the point-matched `SinusoidalSolver` (since
# #259). The SHIFT and not Z, because at any one mesh the absolute
# basis-to-basis gap is the size of the EK shift itself, so only the shift
# measures the kernel.
#
# The decks are FAT (Δ/a ≈ 1.07), because that is where the extended kernel is
# a first-order term rather than a rounding difference — the G16 decks above
# are at Δ/a ≥ 2.4, where δZ is 0.6 % of Z and every ratio taken on it is
# denominator noise. One stands IN the plane, one is lifted clear of it.
#
# Measured (δ_gal against δ_bsp and δ_sin, |Δδ|/|δ|):
#
#   deck          ground       δ_gal              vs bsp   vs sin
#   fat lifted    PEC          +4.62 − 156.16j    0.1473   0.0278
#   fat lifted    somm soil    +4.56 − 157.07j    0.1477   0.0267
#   fat lifted    somm sea     +4.67 − 156.21j    0.1474   0.0277
#   fat contact   PEC          −4.19 −   1.60j    0.0016   0.0947
#   fat contact   somm sea     −4.26 −   1.36j    0.0212   0.0986
#   fat contact   somm soil    −1.68 −   0.11j    0.5635*  0.4122*
#
# Read the lifted rows as a COLUMN: the deck's cross-basis EK mismatch is a
# property of the deck (14.7 % against bspline — G11's class — and 2.7 %
# against the point-matched fill), and adding the Sommerfeld ground does not
# move it, to 4e-4 absolute. That is the stronger statement and
# `test_gs2_the_sommerfeld_ground_does_not_move_the_cross_basis_mismatch`
# makes it directly.
#
# (*) THE CONTACT ROW OVER SOIL is the one pairing whose ratio is not a usable
# statistic, for two separate reasons, and both are gated below rather than
# waved at:
#
#   - against the point-matched solver it is denominator noise, exactly as
#     `_G16_ABSOLUTE` is in the B-spline file. Over average soil this ground
#     nearly cancels the shift's real part, so |δ| falls from 4.37 under PEC
#     to 1.41 and the ratio inflates while the shift VECTORS stay 0.58 apart
#     against the PEC control's own 0.41. Scored on the numerator it passes.
#   - against `BSplineSolver` it is not noise, and it is not about EK at all:
#     the two families disagree by 9.8 % on Z ITSELF, EK-OFF, at a fat contact
#     over soil (54.49 + 10.74j against 59.75 + 9.35j), and by 22.5 % over the
#     reflection-coefficient ground. That is the #151-vs-#282 contact story —
#     a ground-connected basis against a subtracted contact charge — and it
#     predates this arc on both finite grounds. A shift ratio taken across a
#     10 % disagreement about the answer measures the disagreement.
#     `test_gs2_the_bspline_bar_is_scored_where_the_bases_agree_on_z` pins
#     both halves of that, so the exclusion cannot rot into a silent skip.

_S2_DECKS = {
    "fat contact": dict(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.4]])],
        n_per_edge_per_wire=[[15]],
        nsegs=15,
        wire_radius=0.15,
        feed_arclength=0.08,
        ground_z=0.0,
    ),
    "fat lifted": dict(
        wires=[np.array([[0.0, 0.0, 0.6], [0.0, 0.0, 3.0]])],
        n_per_edge_per_wire=[[15]],
        nsegs=15,
        wire_radius=0.15,
        feed_arclength=0.08,
        ground_z=0.0,
    ),
}

_S2_GROUNDS = {
    "PEC": {},
    "somm soil": dict(ground_eps=(13.0, 0.005), ground_model="sommerfeld"),
    "somm sea": dict(ground_eps=(81.0, 5.0), ground_model="sommerfeld"),
}


@functools.lru_cache(maxsize=None)
def _s2_z(deck, ground, cls, ek):
    kw = dict(_S2_DECKS[deck], wavelength=LAM_NEC, extended_kernel=ek)
    kw.update(_S2_GROUNDS[ground])
    if cls is BSplineSolver:
        kw.update(degree=2, feed_model="segment")
    z, _ = cls(**kw).compute_impedance()
    return complex(z)


def _s2_shift(deck, ground, cls):
    return _s2_z(deck, ground, cls, True) - _s2_z(deck, ground, cls, False)


def _s2_mismatch(deck, ground, other):
    d_gal = _s2_shift(deck, ground, SinusoidalGalerkinSolver)
    d_oth = _s2_shift(deck, ground, other)
    return abs(d_gal - d_oth) / abs(d_oth), abs(d_gal - d_oth)


_S2_SOMMERFELD = [g for g in _S2_GROUNDS if g != "PEC"]


@pytest.mark.parametrize(
    "other", [BSplineSolver, SinusoidalSolver], ids=["bspline", "point-matched"]
)
@pytest.mark.parametrize("ground", _S2_SOMMERFELD)
def test_gs2_lifted_fat_wire_shift_is_within_the_cross_basis_bar(ground, other):
    ratio, _ = _s2_mismatch("fat lifted", ground, other)
    assert ratio < _SHIFT_BAR, f"fat lifted / {ground} / {other.__name__}: {ratio:.4f}"


@pytest.mark.parametrize(
    "other", [BSplineSolver, SinusoidalSolver], ids=["bspline", "point-matched"]
)
@pytest.mark.parametrize("deck", list(_S2_DECKS))
def test_gs2_the_sommerfeld_ground_does_not_move_the_cross_basis_mismatch(deck, other):
    """The control, and the stronger of the two statements: whatever the
    deck's own cross-basis EK mismatch is, adding sea water under it must not
    change it. Measured worst 0.0011 absolute on the lifted deck.

    The contact deck is scored on the NUMERATOR — the shift vectors' distance
    — for the reason the section comment gives: over a finite ground its |δ|
    shrinks and the ratio stops being a statistic. Sea water, whose ε̃ puts
    this ground near the PEC limit, is the pairing where both metrics agree.
    """
    if deck == "fat lifted":
        base = _s2_mismatch(deck, "PEC", other)[0]
        for ground in _S2_SOMMERFELD:
            got = _s2_mismatch(deck, ground, other)[0]
            assert abs(got - base) < 0.01, (deck, ground, base, got)
        return
    base = _s2_mismatch(deck, "PEC", other)[1]
    got = _s2_mismatch(deck, "somm sea", other)[1]
    assert abs(got - base) < 0.2, (deck, base, got)


@pytest.mark.parametrize(
    "other", [BSplineSolver, SinusoidalSolver], ids=["bspline", "point-matched"]
)
def test_gs2_contact_shift_over_sea_is_within_the_cross_basis_bar(other):
    """Sea water is the finite ground on which every basis still agrees about
    the contact deck's Z (0.4 % EK-off against bspline), so the ratio bar is a
    statement about the kernel there. Measured 0.021 / 0.099."""
    ratio, _ = _s2_mismatch("fat contact", "somm sea", other)
    assert ratio < _SHIFT_BAR, f"fat contact / somm sea / {other.__name__}: {ratio:.4f}"


def test_gs2_contact_shift_over_soil_is_scored_on_the_numerator():
    """The `_G16_ABSOLUTE` idiom: where |δ| collapses the ratio is denominator
    noise, so score the shift VECTORS' distance and compare it to the same
    deck's PEC control on the same metric. Measured 0.583 against 0.414."""
    _, num = _s2_mismatch("fat contact", "somm soil", SinusoidalSolver)
    _, ctrl = _s2_mismatch("fat contact", "PEC", SinusoidalSolver)
    assert num < 2.0 * ctrl, f"|Δδ| {num:.3f} against the PEC control's {ctrl:.3f}"
    # And the denominator really did collapse — otherwise the ratio would have
    # been usable and this test would be excusing nothing.
    assert abs(_s2_shift("fat contact", "somm soil", SinusoidalSolver)) < 0.5 * abs(
        _s2_shift("fat contact", "PEC", SinusoidalSolver)
    )


def test_gs2_the_bspline_bar_is_scored_where_the_bases_agree_on_z():
    """Why `fat contact / somm soil` is not scored against `BSplineSolver` at
    all, pinned as a measurement so that it stays an informed exclusion rather
    than an unexplained gap.

    A shift ratio between two bases is only about the kernel if the two agree
    about the answer. On this deck EK-OFF they agree to 0.5 % under PEC and to
    0.4 % under sea water — and to 9.8 % under average soil, where a
    ground-connected B-spline basis (#151) and a subtracted contact charge
    (#282) genuinely part company. That gap is EK-OFF, i.e. nothing this arc
    touched, and it is what the 0.56 shift ratio inherits.
    """

    def gap(ground):
        zg = _s2_z("fat contact", ground, SinusoidalGalerkinSolver, False)
        zb = _s2_z("fat contact", ground, BSplineSolver, False)
        return abs(zg - zb) / abs(zg)

    pec = gap("PEC")
    assert pec < 0.01, pec
    assert gap("somm sea") < 2.0 * pec, gap("somm sea")
    assert gap("somm soil") > 5.0 * pec, gap("somm soil")


# ---------------------------------------------------------------------------
# G-S3 — reciprocity over the Sommerfeld ground, with EK on
# ---------------------------------------------------------------------------
def test_gs3_symmetry_survives_the_extended_kernel_over_sommerfeld():
    """G-B2's statement on the one ground it was not allowed to make it on.

    On the fat contact monopole the EK-ON ratio sits BELOW the reduced fill's
    over both soils — 6.6e-8 against 9.5e-8 (soil), 1.1e-7 against 1.6e-7
    (sea) — because on this ground neither number is the EK fill's: the floor
    is the remainder block's own SOURCE rule (`n_qp_sommerfeld`), which is why
    G-B2's "refine `n_qp_test` and it falls" contrast is parametrized without
    this ground and `test_g4_sommerfeld_symmetry_near_the_plane_is_source_
    quadrature_limited` owns the identification.

    Measured with #282's contact-charge correction off, for the reason G-B2
    gives: that correction is source-side only and makes the fill
    non-self-adjoint by construction on a deck that touches the plane.
    """
    for name in ("sommerfeld", "sommerfeld sea"):
        kw = (
            _GROUND_KW["sommerfeld"]
            if name == "sommerfeld"
            else dict(ground_z=0.0, ground_eps=(81.0, 5.0), ground_model="sommerfeld")
        )
        with _without_the_282_contact_correction():
            red = _sym_ratio(_monopole(**kw))
            ext = _sym_ratio(_monopole(extended_kernel=True, **kw))
        assert ext < 2.0 * red, f"{name}: EK {ext:.2e} against reduced {red:.2e}"
        assert ext < 1e-6, f"{name}: {ext:.2e}"


def test_gs3_an_asymmetric_mask_breaks_reciprocity_over_sommerfeld_too(monkeypatch):
    """The falsifying contrast, on this ground: extend only the `j > i` pairs
    — the caricature of NEC's per-source-END decision — and ‖G−Gᵀ‖/‖G‖ jumps
    from 6.6e-8 to 6.6e-1, seven orders, AND stops responding to the test rule
    (6.63e-1 at `n_qp_test` 8 and 16 alike), because the asymmetry is now in
    what is integrated rather than in how well.

    On the numpy backend for G-B2's momwire#358 reason: an upper-triangular
    mutation is not a thing the label payload the fused fill now takes can
    say.
    """
    monkeypatch.setattr(_sg, "_HAVE_GALERKIN_FAR_FILL", False)
    original = SinusoidalGalerkinSolver._ek_pairs

    def upper_only(self, geom, m_idx, n_idx, mirror, n_panels=1):
        pairs = original(self, geom, m_idx, n_idx, mirror, n_panels)
        return pairs._replace(eligible=pairs.eligible & (n_idx > m_idx))

    kw = _GROUND_KW["sommerfeld"]
    with _without_the_282_contact_correction():
        honest = _sym_ratio(_monopole(extended_kernel=True, **kw))
        try:
            SinusoidalGalerkinSolver._ek_pairs = upper_only
            coarse = _sym_ratio(_monopole(extended_kernel=True, **kw))
            fine = _sym_ratio(
                _monopole(extended_kernel=True, n_qp_test=16, n_qp_near=16, **kw)
            )
        finally:
            SinusoidalGalerkinSolver._ek_pairs = original
    assert coarse > 1e-3, coarse
    assert coarse > 1e4 * honest, (coarse, honest)
    assert fine > 0.5 * coarse, f"{fine:.2e} vs {coarse:.2e}: not structural"


# ---------------------------------------------------------------------------
# G-S4 — the armor, extended to the Sommerfeld decks
# ---------------------------------------------------------------------------
def _s4_fill(radius, extended_kernel=False):
    sim = SinusoidalGalerkinSolver(
        wires=[np.array([[0.0, 0.0, 0.6], [0.0, 0.0, 3.0]])],
        n_per_edge_per_wire=[[15]],
        wavelength=LAM_NEC,
        wire_radius=radius,
        nsegs=15,
        feed_arclength=0.08,
        extended_kernel=extended_kernel,
        **_GROUND_KW["sommerfeld"],
    )
    return sim._assemble_Z(sim._build_geometry(), sim.k)[0]


def test_gs4_the_sommerfeld_fill_collapses_onto_the_reduced_one():
    """G-B3 over the Sommerfeld ground. ‖G_ek − G_red‖/‖G_red‖ on a lifted
    monopole over average soil:

        a = 0.05    1.04e-1
        a = 0.005   1.68e-3
        a = 5e-4    4.98e-5
        a = 5e-5    3.21e-6

    Monotone, and four and a half decades of collapse for three decades of
    radius. Not a clean a² law at the thin end, for the reason G-B3 gives:
    reduced-plus-delta leaves ~ε·(H/ρ)² of the delta's peak behind, and the
    remainder — which does not collapse at all, being the same in both fills —
    is subtracted out of this difference exactly.
    """
    prev = None
    for radius in (0.05, 0.005, 5e-4, 5e-5):
        rel = np.linalg.norm(
            _s4_fill(radius, True) - _s4_fill(radius)
        ) / np.linalg.norm(_s4_fill(radius))
        assert prev is None or rel < 0.1 * prev, (radius, rel, prev)
        prev = rel
    assert prev < 1e-5, prev


def test_gs4_ek_on_over_sommerfeld_enters_the_ek_code(ek_call_counts):
    """The other half of G-B4. That file's `sommerfeld ground` deck already
    proves EK OFF is bit-identical to the default and enters no EK code at
    all; before #287 there was no EK-ON statement to make on it, because the
    constructor refused. Now there is, and it is the control that keeps the
    EK-off counter gate from passing vacuously on this ground.
    """
    kw = dict(_G4_DECKS["sommerfeld ground"], wavelength=LAM_NEC)
    z_off, _ = SinusoidalGalerkinSolver(**kw).compute_impedance()
    z_on, _ = SinusoidalGalerkinSolver(**kw, extended_kernel=True).compute_impedance()
    for attr, n in ek_call_counts.items():
        # Two entry points a deck can honestly leave alone, both for the same
        # reason: they run only where a node SPLITS (momwire#299) and this
        # deck is a straight wire. `_ek_end_bracket_fields` evaluates the
        # bracket, and since momwire#355 `_ek_bracket_block` is only reached
        # for a source block that HAS a bad end. `_ek_bracket_plans` is what
        # asks the question now, and it is counted above, so the route is
        # covered either way.
        if attr in ("_ek_end_bracket_fields", "_ek_bracket_block"):
            continue
        assert n > 0, f"{attr} never called with EK on over the Sommerfeld ground"
    assert z_on != z_off
    assert abs(z_on - z_off) < 0.1 * abs(z_off), (z_on, z_off)


# ---------------------------------------------------------------------------
# G-S5 — the IMAGE half of this ground really is extended
# ---------------------------------------------------------------------------
def test_gs5_the_sommerfeld_image_block_carries_its_own_delta():
    """A mutation gate. Everything above measures what the REDUCED half
    costs; this one shows there is an extended half to compare it to.

    Neuter the pair mask on the MIRRORED source only — the free-space block
    keeps its delta, the image block loses it — and the Sommerfeld answer must
    move. It is a fat monopole standing in the plane, so every real/image pair
    is coaxial and equal-radius through the mirror (NEC's IND = 0 branch) and
    the whole image block is eligible. Measured 1.9e-2 of |Z|, against the
    3.5e-3 the reduced remainder is worth on a comparable deck: the mixture is
    a correction to an effect that is really there, not the whole of it.

    Both payload builders again (momwire#358): the Sommerfeld image block is
    plainly projected, so its far half is the fused fill reading
    `_ek_far_labels` and its near half is `_ek_pairs`. On the label side the
    neutering is spelled in the labels' own vocabulary — the observer half set
    to the never-extend value −1 — which is the same empty eligible set the
    zeroed mask gives.
    """
    orig_pairs = SinusoidalGalerkinSolver._ek_pairs
    orig_labels = SinusoidalGalerkinSolver._ek_far_labels

    def no_image_pairs(self, geom, m_idx, n_idx, mirror, n_panels=1):
        pairs = orig_pairs(self, geom, m_idx, n_idx, mirror, n_panels)
        if not mirror:
            return pairs
        return pairs._replace(eligible=np.zeros_like(pairs.eligible))

    def no_image_labels(self, geom, mirror, n_panels=1):
        lab = orig_labels(self, geom, mirror, n_panels)
        if not mirror:
            return lab
        return lab._replace(group_obs=np.full_like(lab.group_obs, -1))

    kw = _GROUND_KW["sommerfeld"]
    z_full, _ = _monopole(extended_kernel=True, **kw).compute_impedance()
    try:
        SinusoidalGalerkinSolver._ek_pairs = no_image_pairs
        SinusoidalGalerkinSolver._ek_far_labels = no_image_labels
        z_flat, _ = _monopole(extended_kernel=True, **kw).compute_impedance()
    finally:
        SinusoidalGalerkinSolver._ek_pairs = orig_pairs
        SinusoidalGalerkinSolver._ek_far_labels = orig_labels
    assert abs(z_flat - z_full) > 1e-3 * abs(z_full), (z_full, z_flat)


# ===========================================================================
# G-D — the end brackets are gated by a NODE predicate (momwire#299)
# ===========================================================================
#
# #246 gated the whole EK delta by a PAIR rule (`_ek_pairs`: same axis line,
# same radius, symmetric by construction). The delta's END BRACKET — the
# boundary term of its integration by parts in ξ, `_ek_end_bracket_fields` —
# is not a pair object: it is O(1/a) per source end, and the two brackets
# meeting at an interior node cancel only if BOTH sides are extended. At a
# bend, a radius step or a K ≥ 3 junction the pair rule's eligible set stops AT
# the node, one bracket survives uncancelled, and the fill diverges as the wire
# thins. Measured on main at 5a1e363, δZ = Z(EK on) − Z(EK off):
#
#   deck                         a = 0.02          a = 0.002
#   L (2 x 8 segs, λ = 10)       −21.5 − 240.5j    −24.1 − 526.7j
#   collinear radius step 1:2    − 5.3 − 257.0j    −54.4 − 554.2j
#   T junction, K = 3            − 0.9 −   6.6j    − 1.6 −  16.1j
#   `_G4_DECKS` vee (LAM_NEC)    −25.5 − 195.7j    −36.9 − 431.3j
#
# against `BSplineSolver`'s −0.035 − 0.657j → −0.002 − 0.041j on the L. The
# straight dipole was always healthy, because no node on it splits.
#
# The fix gates the brackets per SOURCE END by a NODE predicate
# (`_ek_reduced_ends`: extend at node P iff every segment meeting there shares
# one axis line and one radius — NEC's IND = 0 scored on the node, not on the
# pair), and lands it as a post-fill correction assembled through
# `_ek_bracket_correction_tested`. The gates:
#
#   G-D1  a → 0 collapse on every node kind, tracking `BSplineSolver`;
#   G-D2  the predicate itself, and the decks it must leave alone;
#   G-D3  a deck with no split node is bit-identical without the correction;
#   G-D4  the correction's DIVERGENT half is symmetric, which is what makes
#         halving it with its transpose (G-D5's reason) free;
#   G-D5  reciprocity is unmoved on the decks the correction fires on;
#   G-D6  the falsifier: without the correction these decks still diverge.

_GD_LAM = 10.0
_GD_A = (0.02, 0.01, 0.005, 0.002)
# 0.02 → 0.002 is 3.3 halvings, so a bar of 0.55 per halving is ~0.19 over the
# ladder; measured worst is 0.452 (L, 0.01 → 0.005).
_GD_RATIO_BAR = 0.55


def _gd_decks(radius):
    """One deck per NODE KIND the predicate has to decide, all λ = 10, all in
    free space, all 16-20 segments — the design note's own decks, so the
    numbers in these docstrings are its numbers.
    """
    arm = [np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]])]
    return {
        # No split node anywhere: the control, and the deck whose numbers must
        # not move at all.
        "straight": dict(
            wires=[np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]])],
            n_per_edge_per_wire=[[16]],
            nsegs=16,
            wire_radius=radius,
        ),
        # A right-angle bend: two arms, labels split at the corner.
        "L": dict(
            wires=[np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 2.0, 0.0]])],
            n_per_edge_per_wire=[[8, 8]],
            nsegs=16,
            wire_radius=radius,
        ),
        # A 60° bend — the same node kind at a shallower angle.
        "vee": dict(
            wires=[
                np.array(
                    [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 2.0 + np.sqrt(3.0), 0.0]]
                )
            ],
            n_per_edge_per_wire=[[8, 8]],
            nsegs=16,
            wire_radius=radius,
        ),
        # PERFECTLY COLLINEAR, and it still splits: the radii differ 1:2, so
        # the equal-radius half of the label rule separates the two arms. This
        # is the deck an "extend everything" cheat passes the L on and fails.
        "radius step": dict(
            wires=arm + [np.array([[0.0, 2.0, 0.0], [0.0, 4.0, 0.0]])],
            n_per_edge_per_wire=[[8], [8]],
            nsegs=16,
            wire_radius=[radius, 2.0 * radius],
            junctions=[[(0, "end"), (1, "start")]],
        ),
        # K = 3, all radii equal: two of the three members ARE collinear, so
        # the label test alone would pass the node and the topological K ≥ 3
        # test is what fails it.
        "T": dict(
            wires=arm
            + [
                np.array([[0.0, 2.0, 0.0], [0.0, 4.0, 0.0]]),
                np.array([[0.0, 2.0, 0.0], [1.0, 2.0, 0.0]]),
            ],
            n_per_edge_per_wire=[[8], [8], [4]],
            nsegs=20,
            wire_radius=radius,
            junctions=[[(0, "end"), (1, "start"), (2, "start")]],
        ),
    }


def _gd_shift(name, radius, cls=SinusoidalGalerkinSolver):
    kw = dict(_gd_decks(radius)[name], feed_arclength=1.0, wavelength=_GD_LAM)
    if cls is BSplineSolver:
        kw.update(degree=2, feed_model="segment")
    off, _ = cls(**kw).compute_impedance()
    on, _ = cls(**kw, extended_kernel=True).compute_impedance()
    return complex(on) - complex(off)


# ---------------------------------------------------------------------------
# G-D1 — a → 0 on every node kind
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(_gd_decks(0.02)))
def test_gd1_the_fill_collapses_on_every_node_kind(name):
    """G-B3's idiom off the straight dipole and onto the four decks whose
    nodes split. δZ over a = 0.02 / 0.01 / 0.005 / 0.002:

        straight      −0.143 − 1.097j … −0.007 − 0.079j   (unmoved)
        L             −0.071 − 1.040j … −0.004 − 0.075j
        vee           −0.154 − 1.069j … −0.008 − 0.078j
        radius step   −1.072 − 3.535j … −0.036 − 0.213j
        T             −0.140 − 1.039j … −0.007 − 0.075j

    Monotone, worst ratio 0.452 per halving (0.425 on the radius step), and
    every deck now lands within a factor of two of the straight dipole it
    used to be three orders of magnitude away from.
    """
    prev = None
    for radius in _GD_A:
        d = abs(_gd_shift(name, radius))
        if prev is not None:
            step = 0.5 if radius != 0.002 else 0.4  # the last rung is 2.5x
            assert d < prev, (name, radius, d, prev)
            assert d / prev <= _GD_RATIO_BAR * (step / 0.5), (name, radius, d / prev)
        prev = d


@pytest.mark.parametrize("name", ["L", "vee", "radius step", "T"])
def test_gd1_the_collapse_tracks_the_bspline_family(name):
    """Order, not value: the collapse factor over the whole ladder against
    `BSplineSolver`'s on the same deck — 13.8x against 16.0x on the L
    (1.16), 13.7/15.8 vee, 16.6/17.4 radius step, 13.8/16.0 T. The two bases
    disagree about the SIZE of the EK shift by #249 §7's cross-basis margin;
    they must not disagree about its ORDER in a.
    """
    fat, thin = _GD_A[0], _GD_A[-1]
    gal = abs(_gd_shift(name, fat)) / abs(_gd_shift(name, thin))
    bsp = abs(_gd_shift(name, fat, BSplineSolver)) / abs(
        _gd_shift(name, thin, BSplineSolver)
    )
    assert 0.5 < gal / bsp < 2.0, f"{name}: galerkin {gal:.1f}x vs bspline {bsp:.1f}x"


def test_gd1_the_vee_deck_gains_its_ek_on_gate():
    """momwire#299's acceptance sketch, on `_G4_DECKS`' own vee — until now
    checked EK-OFF only, which is how #246 shipped the defect.

        a        δZ galerkin        δZ bspline        |ratio|
        0.020    +0.399 − 32.137j   +0.530 − 35.177j   0.914
        0.010    +0.153 − 16.590j   +0.210 − 18.085j   0.917
        0.005    +0.060 −  8.364j   +0.085 −  9.092j   0.920
        0.002    +0.019 −  3.338j   +0.027 −  3.617j   0.923

    On main the same column reads −25.5 − 195.7j … −36.9 − 431.3j: growing,
    and 6x the size of the answer.
    """
    kw = dict(_G4_DECKS["vee"], wavelength=LAM_NEC)
    prev = None
    for radius in _GD_A:
        deck = dict(kw, wire_radius=radius)
        off, _ = SinusoidalGalerkinSolver(**deck).compute_impedance()
        on, _ = SinusoidalGalerkinSolver(
            **deck, extended_kernel=True
        ).compute_impedance()
        b_off, _ = BSplineSolver(
            **deck, degree=2, feed_model="segment"
        ).compute_impedance()
        b_on, _ = BSplineSolver(
            **deck, degree=2, feed_model="segment", extended_kernel=True
        ).compute_impedance()
        d, db = complex(on - off), complex(b_on - b_off)
        assert abs(d - db) < _SHIFT_BAR * abs(db), (radius, d, db)
        if prev is not None:
            assert abs(d) < 0.55 * prev, (radius, abs(d), prev)
        prev = abs(d)


# ---------------------------------------------------------------------------
# G-D2 — the predicate
# ---------------------------------------------------------------------------
def _bad_ends(sim, mirror=False):
    geom = sim._build_geometry()
    lo, hi = sim._ek_reduced_ends(geom, mirror)
    return geom, lo, hi


@pytest.mark.parametrize(
    "name,n_bad", [("straight", 0), ("L", 2), ("vee", 2), ("radius step", 2), ("T", 3)]
)
def test_gd2_the_node_predicate_marks_exactly_the_split_nodes(name, n_bad):
    """One marked end per segment meeting the split node, and nothing else: a
    bend and a radius step have two members, the T has three, and the straight
    wire — whose only nodes are its own free ends and its collinear
    equal-radius interior — has none. Free ends are never marked, which is
    what keeps every straight deck's numbers where they were.
    """
    sim = SinusoidalGalerkinSolver(
        **dict(_gd_decks(0.02)[name], feed_arclength=1.0, wavelength=_GD_LAM),
        extended_kernel=True,
    )
    geom, lo, hi = _bad_ends(sim)
    assert int(lo.sum() + hi.sum()) == n_bad, (name, lo, hi)
    # And the ends that ARE marked all sit at the same point in space.
    hh = 0.5 * np.asarray(geom["seg_h"])[:, None]
    ends = np.concatenate(
        [
            geom["seg_centers"][lo] - hh[lo] * geom["seg_tangents"][lo],
            geom["seg_centers"][hi] + hh[hi] * geom["seg_tangents"][hi],
        ]
    )
    if ends.shape[0]:
        assert np.allclose(ends, ends[0], atol=1e-12), (name, ends)


@pytest.mark.parametrize("name", list(_G4_DECKS))
def test_gd2_the_served_grounds_have_no_split_node(name):
    """Every deck G-B4 and G-C pin is a straight wire — over PEC, over
    refl-coef, over Sommerfeld, and the vee. Only the vee has a split node, on
    the REAL geometry and on the MIRRORED one alike (it has no ground), so the
    ground decks' EK-on numbers cannot move under this correction and the vee's
    must. A ground contact is a one-segment node and stays extended, which is
    #292's and #287's arithmetic left alone."""
    sim = SinusoidalGalerkinSolver(
        **dict(_G4_DECKS[name], wavelength=LAM_NEC), extended_kernel=True
    )
    _, lo, hi = _bad_ends(sim)
    want = name == "vee"
    assert bool(lo.any() or hi.any()) is want, (name, lo, hi)
    if sim.ground_z is not None:
        _, lo_m, hi_m = _bad_ends(sim, mirror=True)
        assert not (lo_m.any() or hi_m.any()), (name, lo_m, hi_m)


# ---------------------------------------------------------------------------
# G-D3 — no split node, no correction, to the bit
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _without_the_299_bracket_correction():
    cls = SinusoidalGalerkinSolver
    orig = cls._ek_bracket_correction_tested
    cls._ek_bracket_correction_tested = lambda self, G, geom, k, ctx: None
    try:
        yield
    finally:
        cls._ek_bracket_correction_tested = orig


@pytest.mark.parametrize("name", [n for n in _G4_DECKS if n != "vee"])
def test_gd3_a_deck_without_a_split_node_is_bit_identical(name):
    """The other half of G-D2's structural claim, measured on Z rather than on
    the predicate: with the extended kernel ON, removing the correction
    entirely changes nothing on a deck whose nodes all pass — not to a
    tolerance, to the bit. That is what says #246's straight-wire, ground and
    Sommerfeld numbers are the ones this arc inherited."""
    kw = dict(_G4_DECKS[name], wavelength=LAM_NEC)
    z_on, c_on = SinusoidalGalerkinSolver(
        **kw, extended_kernel=True
    ).compute_impedance()
    with _without_the_299_bracket_correction():
        z_off, c_off = SinusoidalGalerkinSolver(
            **kw, extended_kernel=True
        ).compute_impedance()
    assert z_on == z_off, f"{name}: {z_on!r} vs {z_off!r}"
    assert np.array_equal(c_on, c_off)


def test_gd3_the_straight_dipole_is_where_it_was():
    """The pinned value, so a future change to the correction's pair set
    cannot quietly reach a deck it has no business on: the 16-segment straight
    wire of `_gd_decks` at a = 0.02 shifts by −0.1434 − 1.0967j, the number
    main gives."""
    d = _gd_shift("straight", 0.02)
    assert abs(d - (-0.14337 - 1.09666j)) < 5e-5, d


# ---------------------------------------------------------------------------
# G-D4 — the correction's divergent half is symmetric
# ---------------------------------------------------------------------------
def _bracket_matrix(radius, name="L"):
    """The free-space block's correction C, before symmetrization."""
    sim = SinusoidalGalerkinSolver(
        **dict(_gd_decks(radius)[name], feed_arclength=1.0, wavelength=_GD_LAM),
        extended_kernel=True,
    )
    geom = sim._build_geometry()
    ctx = sim._test_context(geom, sim._basis_coefs(geom, sim.k), sim.k)
    N = ctx["N"]
    # The whole-triple spelling on purpose: one band over every test segment,
    # every source column retained, and the fill's own scatter product. What
    # this gate measures is C's structure, not how it is assembled.
    (plan,) = sim._ek_bracket_plans(
        geom, ctx, [(_plain_projection, None, None, False, 1.0)]
    )
    every = np.arange(N)
    plan = plan._replace(cols=every, col_of=every)
    corr = tuple(
        np.zeros((ctx["w_entry"].shape[0], N), dtype=np.complex128) for _ in range(3)
    )
    sim._ek_bracket_block(geom, sim.k, ctx, corr, plan, 0, N)
    return np.asarray(sim._scatter_coef_product(ctx, corr))


def test_gd4_the_bracket_correction_diverges_symmetrically():
    """The licence for `G −= ½(C + Cᵀ)`. The cap is a SOURCE-sided boundary
    term, so C itself is not symmetric — but its divergence is, because the
    cap is κ_P·f_i(P)·f_j(P), a rank-one outer product in the two sides'
    current at the node. Measured on the L:

        a        ‖C‖        a·‖C‖       ‖C−Cᵀ‖/‖C‖
        0.0200   9.320e−2   1.864e−3    8.374e−3
        0.0100   1.856e−1   1.856e−3    2.166e−3
        0.0050   3.708e−1   1.854e−3    5.522e−4
        0.0020   9.268e−1   1.854e−3    8.961e−5
        0.0010   1.853e+0   1.853e−3    2.253e−5

    ‖C‖ is exactly 1/a (a·‖C‖ constant to 0.6 % over a decade and a half) and
    the asymmetry falls like a² (0.26 per halving), so halving C with its
    transpose removes the same O(1/a) term and averages only an O(a²)
    remainder.
    """
    scaled, asym = [], []
    for radius in (0.02, 0.01, 0.005, 0.002, 0.001):
        C = _bracket_matrix(radius)
        norm = np.linalg.norm(C)
        scaled.append(radius * norm)
        asym.append(np.linalg.norm(C - C.T) / norm)
    assert max(scaled) / min(scaled) < 1.02, scaled
    for prev, cur in zip(asym, asym[1:]):
        assert cur < 0.35 * prev, asym
    assert asym[-1] < 1e-4, asym


# ---------------------------------------------------------------------------
# G-D5 — reciprocity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(_gd_decks(0.02)))
@pytest.mark.parametrize("radius", [0.02, 0.002])
def test_gd5_reciprocity_is_unmoved_by_the_bracket_correction(name, radius):
    """G-B2's statement on the decks the correction actually fires on.
    ‖G−Gᵀ‖/‖G‖ with EK on against the reduced fill's, at a = 0.02 / 0.002:

        straight      1.34e−12 / 6.09e−13     2.32e−12 / 2.31e−12
        L             1.04e−10 / 6.16e−11     4.85e−11 / 5.09e−11
        vee           3.72e−11 / 6.50e−12     4.27e−11 / 4.03e−11
        radius step   1.14e−01 / 1.15e−01     5.85e−02 / 5.85e−02
        T             4.17e−10 / 1.64e−10     3.10e−10 / 2.09e−10

    — the free-space floor everywhere except the radius-step deck, whose 1e−1
    is its MIXED-RADIUS reduced fill's own and moves by 1 % under the extended
    kernel. The correction as literally spelled — the source-sided bracket
    subtracted where the node rule says, without the transpose average — gives
    1.6e−3 on the L and 2.9e−3 on the T instead, seven orders worse than the
    reduced fill and the reason `_ek_bracket_correction_tested` halves C with
    its transpose.
    """
    kw = dict(_gd_decks(radius)[name], feed_arclength=1.0, wavelength=_GD_LAM)
    red = _sym_ratio(SinusoidalGalerkinSolver(**kw))
    ext = _sym_ratio(SinusoidalGalerkinSolver(**kw, extended_kernel=True))
    assert ext < 8.0 * red, f"{name} a={radius}: {ext:.2e} vs reduced {red:.2e}"
    # And in absolute terms: still at the fill's own floor, not merely close
    # to a reduced number that happens to be large.
    assert ext < 1e-8 or name == "radius step", f"{name} a={radius}: {ext:.2e}"


# ---------------------------------------------------------------------------
# G-D6 — the falsifier
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["L", "radius step", "T"])
def test_gd6_without_the_correction_these_decks_still_diverge(name):
    """The control for all of the above: every gate here would pass vacuously
    on a correction that never fired, so this removes it and demands the
    defect back. δZ must GROW from a = 0.02 to a = 0.002, by 2x on the L and
    the T and 2.2x on the radius step — the numbers from main."""
    with _without_the_299_bracket_correction():
        fat = abs(_gd_shift(name, 0.02))
        thin = abs(_gd_shift(name, 0.002))
    assert thin > 1.5 * fat, f"{name}: {fat:.3f} → {thin:.3f}"


# ---------------------------------------------------------------------------
# G-D7 — #287's bent deck, usable again
# ---------------------------------------------------------------------------
def test_gd7_the_bent_deck_of_287_is_usable_again():
    """`_s_decks()` substituted `slanted` for #269's Gate-18 BENT deck because
    this fill was wrong on it. Measured with the node gate in, δZ = Z(EK on) −
    Z(EK off) on that very deck:

        ground        galerkin           bspline            point-matched
        PEC           +11.52 −  4.97j    +11.94 −  6.50j    +12.81 −  5.71j
        somm soil     +13.04 −  2.41j    +13.33 −  3.81j    +14.42 −  2.68j

    — 11.7 % / 10.3 % from the B-spline family and 10.6 % / 9.6 % from the
    point-matched one, inside #249 §7's 25 % cross-basis bar. On main the same
    deck reads +50.86 + 215.97j and −12.70 + 225.97j: a shift 30x too big,
    with the wrong sign on the reactance.
    """
    bend = np.array([[0.0, 0.0, 0.3], [2.0, 0.0, 1.5], [3.6, 1.1, 1.5]])
    h_bend = float(np.linalg.norm(bend[1] - bend[0])) / 6
    deck = dict(
        wires=[bend],
        n_per_edge_per_wire=[[6, 5]],
        nsegs=11,
        wire_radius=0.08,
        feed_arclength=h_bend * 3.5,
        ground_z=0.0,
        wavelength=6.0,
    )
    grounds = ({}, dict(ground_eps=_S_EPS["soil"], ground_model="sommerfeld"))
    for ground in grounds:
        shifts = {}
        for tag, cls, extra in (
            ("gal", SinusoidalGalerkinSolver, {}),
            ("bsp", BSplineSolver, dict(degree=2, feed_model="segment")),
            ("pm", SinusoidalSolver, {}),
        ):
            off, _ = cls(**deck, **ground, **extra).compute_impedance()
            on, _ = cls(
                **deck, **ground, **extra, extended_kernel=True
            ).compute_impedance()
            shifts[tag] = complex(on) - complex(off)
        for other in ("bsp", "pm"):
            rel = abs(shifts["gal"] - shifts[other]) / abs(shifts[other])
            assert rel < _SHIFT_BAR, (ground, other, rel, shifts)


# ---------------------------------------------------------------------------
# G-D8 — the correction streams: no triple of its own (momwire#355)
# ---------------------------------------------------------------------------
# `_ek_bracket_correction_tested` used to build a whole (nnz, N) triple of its
# own — a second copy of the largest object in the fill — and it built it while
# `_assemble_Z` still held the fill's. Two things take that away without
# moving a float:
#
#   * the correction can only ever write the SOURCE columns that have a bad end
#     (`_ek_reduced_ends`), so its triple is allocated over those alone; every
#     column dropped was identically zero, and `x + 0` is `x` in float64 under
#     any association;
#   * what is left is banded over TEST segments and folded into the scatter
#     band by band, so nothing of matrix size is ever whole. Test segments are
#     the axis that keeps this bit-exact: a matrix cell's writers are its own
#     test segment's entries, so a cell is finished inside one band.
#
# G-D8a is the arithmetic gate for both — widen the columns back to all N, or
# band the correction one test segment at a time, and G must not move by a
# bit. G-D8b and G-D8c are the residency ones.


def _gd8_bend(n, **ground):
    """Two straight arms meeting at a right angle 4 m over the plane: ONE
    split node, so `cols` is 2 of N and the column narrowing is what pays."""
    half = 0.962 * _GD8_WL / 4
    ys = np.linspace(0.0, half, n // 2 + 1)
    xs = np.linspace(0.0, half, n - n // 2 + 1)[1:]
    pts = [[0.0, y, 4.0] for y in ys] + [[x, half, 4.0] for x in xs]
    return SinusoidalGalerkinSolver(
        wires=[np.array(pts)],
        n_per_edge_per_wire=[[1] * n],
        nsegs=n,
        wavelength=_GD8_WL,
        extended_kernel=True,
        **ground,
    )


def _gd8_zigzag(n, **ground):
    """Every node a bend: `cols` is EVERY source column, so the narrowing buys
    nothing and the banding is the whole of the fix. This is the deck the
    streaming exists for."""
    half = 0.962 * _GD8_WL / 4
    ys = np.linspace(-half, half, n + 1)
    xs = 0.02 * _GD8_WL * (np.arange(n + 1) % 2)
    return SinusoidalGalerkinSolver(
        wires=[np.column_stack([xs, ys, np.full_like(ys, 4.0)])],
        n_per_edge_per_wire=[[1] * n],
        nsegs=n,
        wavelength=_GD8_WL,
        extended_kernel=True,
        **ground,
    )


_GD8_WL = 22.0
_GD8_DECKS = {"bend": _gd8_bend, "zigzag": _gd8_zigzag}
_GD8_GROUNDS = {
    "free": {},
    "pec": {"ground_z": 0.0},
    "refl": {"ground_z": 0.0, "ground_eps": (13.0, 0.005)},
}


@contextlib.contextmanager
def _gd8_every_column():
    """Run the correction over ALL N source columns instead of the bad-end
    ones — the pre-#355 column set, reached through the shipped dispatch."""
    from momwire import sinusoidal_galerkin as _sg

    orig = _sg.SinusoidalGalerkinSolver._ek_bracket_plans

    def wide(self, geom, ctx, blocks):
        cols = np.arange(ctx["N"])
        return [p._replace(cols=cols) for p in orig(self, geom, ctx, blocks)]

    _sg.SinusoidalGalerkinSolver._ek_bracket_plans = wide
    try:
        yield
    finally:
        _sg.SinusoidalGalerkinSolver._ek_bracket_plans = orig


@pytest.mark.parametrize("name", list(_gd_decks(0.02)))
@pytest.mark.parametrize("ground", list(_GD8_GROUNDS))
def test_gd8a_narrowing_and_banding_move_the_matrix_by_nothing(
    monkeypatch, name, ground
):
    """Exact equality, not a tolerance, on all five node kinds and all three
    grounds — the two narrowings are index bookkeeping and the banding sums
    each cell in the order it always did, so anything at all here is a bug.

    Three spellings are compared against the shipped one: every column
    retained (which is the whole (nnz, N) triple back), one test segment per
    band (the finest streaming the band rule allows), and both at once — which
    together ARE the pre-#355 arithmetic, differing from it only in that the
    scatter is reached band by band.
    """
    from momwire import sinusoidal_galerkin as _sg

    kw = dict(_gd_decks(0.02)[name], feed_arclength=1.0, wavelength=_GD_LAM)
    if ground != "free":
        # Lift the deck off the plane: these are free-space fixtures.
        kw["wires"] = [np.asarray(w) + np.array([0.0, 0.0, 1.0]) for w in kw["wires"]]
        kw.update(_GD8_GROUNDS[ground])
    sim = SinusoidalGalerkinSolver(**kw, extended_kernel=True)
    geom = sim._build_geometry()
    ref, _ = sim._assemble_Z(geom, sim.k)

    for wide in (False, True):
        for band in (False, True):
            if not wide and not band:
                continue
            monkeypatch.setattr(_sg, "_EK_BRACKET_BAND_BYTES", 1 if band else 1 << 25)
            with _gd8_every_column() if wide else contextlib.nullcontext():
                got, _ = SinusoidalGalerkinSolver(
                    **kw, extended_kernel=True
                )._assemble_Z(geom, sim.k)
            assert np.array_equal(ref, got), (
                f"{name}/{ground} wide={wide} band={band}: the bracket moved G "
                f"by {np.abs(ref - got).max():.3e}"
            )


# Peak transient of the correction ALONE, in triples of 3 x 16 x nnz x N, at
# the N = 300 decks above with the band budget shrunk to one test segment.
# Measured, before (main) -> after:
#
#   bend    free 1.45 -> 0.34   pec 1.45 -> 0.34   refl 1.95 -> 0.84
#   zigzag  free 1.65 -> 0.68   pec 1.65 -> 0.68   refl 2.15 -> 1.18
#
# The refl-coef column carries ~0.5 triple of specular/Fresnel tables in both
# spellings — a per-segment-pair cache, not a fill object — so its bar is
# offset by exactly that. The bend deck's floor is the (n_basis, 2) scatter,
# i.e. nothing; the zigzag's is the (n_basis, N) one, which is a third of a
# triple and is the smallest a symmetrized C can be assembled from.
_GD8_BAR = {
    ("bend", "free"): 0.75,
    ("bend", "pec"): 0.75,
    ("bend", "refl"): 1.30,
    ("zigzag", "free"): 1.10,
    ("zigzag", "pec"): 1.10,
    ("zigzag", "refl"): 1.60,
}


@pytest.mark.memgate
@pytest.mark.parametrize("deck", list(_GD8_DECKS))
@pytest.mark.parametrize("ground", list(_GD8_GROUNDS))
def test_gd8b_the_bracket_correction_holds_no_triple(monkeypatch, deck, ground):
    """Tracemalloc gate on `_ek_bracket_correction_tested` itself (#355).

    Budget is in TRIPLES — 3 x 16 x nnz x N, the fill's own structural unit —
    because a second one of those is what the correction used to build. At
    N = 300 nnz is 898 and a triple is 12.93 MB. Every bar in `_GD8_BAR` sits
    at least 1.3x above the streamed number and at least 1.3x below the
    whole-triple one, which is the headroom momwire#347 taught this suite to
    leave: a gate that passed locally at 1.06x its bar failed on the GitHub
    runner at 1.11x.

    The correction is called on its own rather than through `_assemble_Z` so
    that what is measured is the correction and not the fill — the extended
    kernel's near correction holds a per-pair working set several times this
    on the same deck, which is `_PAIR_BLOCK`'s business and not this gate's.
    G is a zero matrix of the right shape: the correction only subtracts into
    it, and its size is charged to the caller either way.

    `_EK_BRACKET_BAND_BYTES` is shrunk to one test segment's worth, the same
    lever `test_grounded_fill_holds_no_parallel_ground_triple` pulls on
    `_FILL_WORKSPACE_BYTES`: at N = 300 the shipped 32 MB budget is not
    binding, and what this gate is for is the behaviour at the sizes where it
    is. `_PAIR_BLOCK` is shrunk for the same reason it is there — the pair
    scratch is a fixed working set that would otherwise bury the triples.
    """
    import tracemalloc

    from momwire import sinusoidal_galerkin as _sg

    monkeypatch.setattr(_sg, "_EK_BRACKET_BAND_BYTES", 1)
    monkeypatch.setattr(_sg, "_PAIR_BLOCK", 8)

    n = 300
    sim = _GD8_DECKS[deck](n, **_GD8_GROUNDS[ground])
    geom = sim._build_geometry()
    N = geom["n_segs"]
    assert N == n
    ctx = sim._test_context(geom, sim._basis_coefs(geom, sim.k), sim.k)
    triple = 3 * 16 * ctx["w_entry"].shape[0] * N
    G = np.zeros((N, N), dtype=np.complex128)

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        sim._ek_bracket_correction_tested(G, geom, sim.k, ctx)
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    bar = _GD8_BAR[(deck, ground)]
    assert peak < bar * triple, (
        f"{deck}/{ground}: the end-bracket correction peaked "
        f"{peak / 1e6:.2f} MB = {peak / triple:.2f} triples (one triple = "
        f"{triple / 1e6:.2f} MB), bar {bar} — it is building a triple again"
    )


@pytest.mark.memgate
def test_gd8c_the_free_space_ek_assembly_holds_one_fill_not_two(monkeypatch):
    """The whole `_assemble_Z` peak, which is what momwire#355 was opened on.

    Free space only, and that is the finding rather than a convenience: with a
    ground the peak belongs to the FILL — the fused C++ image block allocates
    its own triple to fold, and the refl-coef ground its specular tables — so
    the grounded assembly reads 2.47 -> 2.09 (PEC) and 2.98 -> 2.82 (refl-coef)
    here, differences too small for any bar to sit inside. Free space is where
    the correction is the peak, and it moves 2.47 -> 1.36 triples: the
    correction's own triple, plus the fill's, which `_assemble_Z` used to hold
    across it and now drops.

    1.8 sits 1.32x above the streamed number and 1.37x below the whole-triple
    one — the momwire#347 headroom, on the tightest of these three gates.

    The issue's own headline number, ~4.1 triples in every mode, is NOT this
    and is not the correction: at the `_PAIR_BLOCK` of 512 that shipped when
    momwire#355 landed, the extended kernel's near correction held 5.14 MB
    per pair in the block — 2.63 GB — and that fixed working set, not the
    matrix, is what the original measurement saw. momwire#383 has since sized
    that block to a byte budget (G-D9), and the free-space assembly's whole
    transient at the SHIPPED configuration is now 17.6 MB against 2.63 GB.
    Shrinking the pair block to 1 here is therefore belt-and-braces rather
    than the only way to see the triples; it stays because a ceiling this
    gate pins itself against cannot be inherited from another issue's
    constant.
    """
    import tracemalloc

    from momwire import sinusoidal_galerkin as _sg

    monkeypatch.setattr(_sg, "_FILL_WORKSPACE_BYTES", 1)
    monkeypatch.setattr(_sg, "_EK_BRACKET_BAND_BYTES", 1)
    monkeypatch.setattr(_sg, "_PAIR_BLOCK", 1)

    n = 300
    sim = _gd8_bend(n)
    geom = sim._build_geometry()
    N = geom["n_segs"]
    nnz = sim._test_context(geom, sim._basis_coefs(geom, sim.k), sim.k)[
        "w_entry"
    ].shape[0]
    triple = 3 * 16 * nnz * N

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        G, _ = sim._assemble_Z(geom, sim.k)
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    transient = peak - G.nbytes
    assert transient < 1.8 * triple, (
        f"the free-space EK assembly peaked {transient / 1e6:.2f} MB above G "
        f"= {transient / triple:.2f} contribution triples (one triple = "
        f"{triple / 1e6:.2f} MB) — the bracket's own triple is back, or the "
        f"fill's is being held across it"
    )


# ===========================================================================
# G-D9 — the near correction's working set is byte-budgeted (momwire#383)
# ===========================================================================
# G-D8c above recorded a transient it could not account for: ~4.1 triples on
# every deck and every ground, unmoved by the bracket streaming and unmoved by
# the matrix. It was `_apply_near_correction`, and it was not of matrix shape
# at all — a FIXED working set of one pair block, riding every extended-kernel
# Galerkin assembly with the near correction on.
#
# Per statement, with tracemalloc, on the #355 bend deck at N = 300 (the
# endpoint-graded rule there is G = 96 nodes, the delta rule n_d = 128):
#
#   reduced path   16·G·(12.5·nq_c + 65)  = 254 KB/pair — the Eqs 76-79
#                  tables and the (G, nq_c) source-quadrature scratch under
#                  `int_G0` and `_folded_cos_fields`;
#   extended path  + ~25 live (G, n_d) complex arrays inside
#                  `_folded_ek_delta_fields` — t, cosh_t, R, zeta, xi, w, r2,
#                  x, x2, x3, x4, a1…a4, inv2, phase, base, g1…g4, t_c, t_z,
#                  l_z, l_r, kxi and the two shapes — 4.88 MB/pair, 20x the
#                  whole reduced cost.
#
# Measured slope, block 1 -> 32: 5.14 MB/pair extended, 0.25 MB/pair reduced,
# flat in N. At the shipped `_PAIR_BLOCK` of 512 that is 2.63 GB and 112 MB.
# `_NEAR_WORKSPACE_BYTES` (8 MB) now sizes the block instead: one pair under
# the extended kernel at this rule, 32 under the reduced one.
#
#   G-D9a  the block size moves no float;
#   G-D9b  the budget is honoured at the SHIPPED configuration — no gate in
#          this suite has been able to assert that before, because the
#          shipped configuration was the one nobody could afford to trace.
#
# The reuse the issue also asked about is not available and does not need to
# be: those ~25 arrays are one vectorized call's live temporaries, not a
# per-pair allocation inside a Python loop, so there is nothing to hoist out
# of a loop that does not exist. Sizing the call is the whole of the fix.


def _gd9_bend(n, ek, **kw):
    """The G-D8 bend on a FAT wire (Δ/a ~ 5), which is the near correction's
    own worst case — the extended-minus-reduced delta is largest there — and
    which coarsens the endpoint-graded rule enough that even a 512-pair block
    is affordable to allocate in a test."""
    half = 0.962 * _GD8_WL / 4
    ys = np.linspace(0.0, half, n // 2 + 1)
    xs = np.linspace(0.0, half, n - n // 2 + 1)[1:]
    pts = [[0.0, y, 4.0] for y in ys] + [[x, half, 4.0] for x in xs]
    return SinusoidalGalerkinSolver(
        wires=[np.array(pts)],
        n_per_edge_per_wire=[[1] * n],
        nsegs=n,
        wavelength=_GD8_WL,
        wire_radius=0.02,
        extended_kernel=ek,
        feeds=[(0, 0, 1.0)],
        **kw,
    )


def _gd9_monopole(n, ek, **kw):
    """A quarter-wave monopole standing ON the plane: the near-pair-rich
    geometry of momwire#356's sweep, where the contact node puts a segment
    next to its own image and the near set is at its densest."""
    z = np.linspace(0.0, _GD8_WL / 4, n + 1)
    pts = np.column_stack([np.zeros(n + 1), np.zeros(n + 1), z])
    return SinusoidalGalerkinSolver(
        wires=[pts],
        n_per_edge_per_wire=[[1] * n],
        nsegs=n,
        wavelength=_GD8_WL,
        wire_radius=0.02,
        extended_kernel=ek,
        feeds=[(0, 0, 1.0)],
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        **kw,
    )


_GD9_DECKS = {"bend": _gd9_bend, "monopole": _gd9_monopole}


@pytest.mark.parametrize("ek", [True, False])
@pytest.mark.parametrize("deck", list(_GD9_DECKS))
def test_gd9a_the_near_block_size_moves_no_float(monkeypatch, deck, ek):
    """G and Z are the same to the BIT however the near pairs are blocked.

    The arithmetic reason: each near pair's contribution is computed from its
    own (test segment, source segment) geometry and ASSIGNED into its own
    cells — `_apply_near_correction`'s docstring states the ownership, and no
    accumulator crosses pairs inside a block — so blockmates never reach each
    other's floats. Blocking is a scheduling decision, and this pins it.

    Four block sizes: 1, 8, whatever `_near_block` computes for the deck
    (2-4 here, and 1 on a thin wire) and the pre-momwire#383 flat 512, which
    on these decks covers every pair in one call.

    This carried ONE caveat until momwire#392, recorded here because the
    caveat was true of main as well: numpy elides a dead temporary on the
    RIGHT of a complex product into an in-place multiply once it passes
    256 KB, and the in-place loop does not round like the out-of-place one,
    so `_field_components_bcast`'s `G0_e * (bracket)` moved `Erho_sin` and
    `Ez_sin` in the last bits with the batch shape. A (P, G) table crosses
    256 KB at P = 128 for G = 128, so on a thin-wire deck the SHIPPED
    512-pair block was on the far side of that boundary and the small blocks
    every residency gate here monkeypatches were not: measured 8.2e-20 of
    ‖G‖ between them at N = 80. #383 could only contain that (G-D9c, the
    budgeted block being always on the small-block side); #392 named the
    operands, and G-D9d pins the kernel's shape independence outright. The
    512 was never a plateau this test could have pinned, and there is now no
    block size at which it is not the plateau.
    """
    ref_G = ref_Z = None
    for blk in (1, 8, None, 512):
        if blk is not None:
            monkeypatch.setattr(_sg, "_near_block", lambda *a, _b=blk: _b)
        else:
            monkeypatch.undo()
        sim = _GD9_DECKS[deck](24, ek)
        G, _seg_view = sim._assemble_Z(sim._build_geometry(), sim.k)
        Z = sim.compute_y_matrix()
        if ref_G is None:
            ref_G, ref_Z = G, Z
            continue
        assert np.array_equal(G, ref_G), (
            f"{deck}/ek={ek}: blocking the near pairs {blk} at a time moved G "
            f"by {np.max(np.abs(G - ref_G)):.3e} — the block size is reaching "
            f"the arithmetic"
        )
        assert np.array_equal(Z, ref_Z), f"{deck}/ek={ek}: Y moved at blk={blk}"


@pytest.mark.parametrize(
    "nq_graded,n_qp_const,ek",
    [
        (g, c, e)
        for g in (16, 32, 96, 144, 512)
        for c in (4, 8, 16)
        for e in (True, False)
    ],
)
def test_gd9c_the_budgeted_block_stays_under_numpys_threshold(
    nq_graded, n_qp_const, ek
):
    """The budgeted block's (P, G) field tables are under 256 KB, always.

    That is the boundary G-D9a's caveat recorded — above it numpy elides a
    dead temporary into an in-place complex multiply that rounds differently,
    and the near correction's rounding would depend on how the pairs were cut
    into calls. momwire#392 removed the dependence at its source, so this is
    no longer what makes the correction's arithmetic safe (G-D9d is), and it
    is kept for what it still says: the budget cannot put the near
    correction's tables anywhere near a size where anything about the
    evaluation strategy changes. The budget puts every block on the same side
    of it by construction:
    P·G·16 ≤ `_NEAR_WORKSPACE_BYTES`/(13·nq_c + 66) ≤ 8 MB/118 = 71 KB, and
    the `_PAIR_BLOCK` cap can only lower it. Structural, so it is checked
    structurally rather than on a deck.
    """
    blk = _sg._near_block(nq_graded, n_qp_const, ek)
    assert blk >= 1
    assert blk * nq_graded * 16 < 256 * 1024, (
        f"G={nq_graded}, nq_c={n_qp_const}, ek={ek}: the budgeted block of "
        f"{blk} pairs makes a {blk * nq_graded * 16 / 1024:.0f} KB field "
        f"table, over numpy's 256 KB loop boundary"
    )


# The bar for G-D9b, in units of the budget the block is sized to. Measured on
# the N = 200 bend, all three grounds: 5.25 MB extended (0.63 of the 8 MB
# budget, the block being one pair whose set does not fill it) and 8.11 MB
# reduced (0.97, a 32-pair block sized to fill it). 1.4 sits 1.44x above the
# larger of those — the momwire#347 headroom — and 224x below the 2.63 GB the
# shipped 512-pair block held before momwire#383 (512 pairs x 5.14 MB, which
# is what this measurement reads on main).
_GD9_BAR = 1.4


@pytest.mark.memgate
@pytest.mark.parametrize("ek", [True, False])
@pytest.mark.parametrize("ground", list(_GD8_GROUNDS))
def test_gd9b_the_near_correction_holds_one_budget(ek, ground):
    """Tracemalloc gate on `_apply_near_correction` at the SHIPPED block.

    Nothing is monkeypatched — that is the point of this gate and the thing
    G-D8b, G-D8c, G-C5 and G-C6 could not do. Each of them shrinks
    `_PAIR_BLOCK` to 8 or 1 to see past this working set, and each says so;
    what none of them measures is the configuration that actually ships. At
    `_PAIR_BLOCK = 512` with the extended kernel on, that configuration held
    2.63 GB — fixed, independent of N, on a 16 GB machine.

    The correction is called on its own, as G-D8b calls the bracket: the
    subject is a per-pair working set and not the fill, and the destination
    triple is the caller's either way.
    """
    import tracemalloc

    n = 200
    sim = _gd8_bend(n, **_GD8_GROUNDS[ground])
    sim.extended_kernel = ek
    geom = sim._build_geometry()
    N = geom["n_segs"]
    assert N == n
    ctx = sim._test_context(geom, sim._basis_coefs(geom, sim.k), sim.k)
    nnz = ctx["w_entry"].shape[0]
    contribs = tuple(np.zeros((nnz, N), dtype=np.complex128) for _ in range(3))

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        sim._apply_near_correction(
            geom,
            sim.k,
            ctx,
            contribs,
            _plain_projection,
            geom["seg_centers"],
            geom["seg_tangents"],
            False,
        )
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    budget = _sg._NEAR_WORKSPACE_BYTES
    assert peak - base < _GD9_BAR * budget, (
        f"ek={ek}/{ground}: the near correction peaked "
        f"{(peak - base) / 1e6:.2f} MB = {(peak - base) / budget:.2f} of its "
        f"{budget / 1e6:.2f} MB budget, bar {_GD9_BAR} — the pair block is "
        f"not being sized to the budget"
    )


@pytest.mark.memgate
def test_gd9b_has_teeth():
    """G-D9b's companion: widen the budget and the same measurement blows
    through the bar, so what it is watching is live.

    16x the budget, which is a 16-pair block on this deck and ~84 MB — the
    smallest multiple that makes the point without allocating the 2.63 GB the
    unbudgeted block would.
    """
    import tracemalloc

    n = 200
    sim = _gd8_bend(n)
    geom = sim._build_geometry()
    ctx = sim._test_context(geom, sim._basis_coefs(geom, sim.k), sim.k)
    nnz = ctx["w_entry"].shape[0]
    contribs = tuple(np.zeros((nnz, n), dtype=np.complex128) for _ in range(3))
    args = (
        geom,
        sim.k,
        ctx,
        contribs,
        _plain_projection,
        geom["seg_centers"],
        geom["seg_tangents"],
        False,
    )
    budget = _sg._NEAR_WORKSPACE_BYTES
    real = _sg._near_block
    try:
        _sg._near_block = lambda g, c, e: 16 * real(g, c, e)
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            base = tracemalloc.get_traced_memory()[0]
            sim._apply_near_correction(*args)
            _cur, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    finally:
        _sg._near_block = real

    assert peak - base > _GD9_BAR * budget, (
        f"a 16x block peaked only {(peak - base) / 1e6:.2f} MB — G-D9b's bar "
        f"is not measuring the pair block at all"
    )


# ---------------------------------------------------------------------------
# G-D9d: the field kernel is a function of its PAIRS, not of its batch shape
# (momwire#392). This is the caveat G-D9a carried and G-D9c could only
# contain: numpy elides a dead temporary that is the RIGHT operand of a
# complex product into an in-place multiply once it passes 256 KB
# (`NPY_MIN_ELIDE_BYTES`), and for complex128 the in-place loop does not round
# like the out-of-place one. `_field_components_bcast` spelled its endpoint
# brackets as `G0_e * (…)` and its prefactor as `pref_rho * (…)`, so a caller
# whose tables crossed 256 KB got different last bits for the same pair —
# making the near correction's block size, the far fill's block size and even
# the collocation solver's N reach the arithmetic. The fix names those
# operands; these gates pin the property rather than the spelling.
# ---------------------------------------------------------------------------

_GD9D_WL = 22.0


def _gd9d_pairs(n_pairs, n_obs, delta, a):
    """`n_pairs` source segments along z, each with `n_obs` observers ON it.

    Deliberately the near correction's own shape — the observer inside the
    source segment's span, one radius off the axis — because that is where a
    fill spends its most important pairs. The observers are offset by a
    per-pair factor so no two pairs share a float and a difference cannot
    hide behind a repeat.
    """
    hh = 0.5 * delta
    src_c = np.zeros((n_pairs, 1, 3))
    src_c[:, 0, 2] = np.arange(n_pairs) * delta
    src_t = np.zeros((n_pairs, 1, 3))
    src_t[:, 0, 2] = 1.0
    grade = np.linspace(-0.97, 0.97, n_obs)[None, :]
    off = hh * grade * (1.0 + 1e-3 * np.arange(n_pairs)[:, None])
    obs_c = np.zeros((n_pairs, n_obs, 3))
    obs_c[:, :, 2] = src_c[:, :, 2] + off
    obs_c[:, :, 0] = a
    obs_t = np.zeros((n_pairs, n_obs, 3))
    obs_t[:, :, 2] = 1.0
    return dict(
        obs_c=obs_c,
        obs_t=obs_t,
        a=a,
        src_c=src_c,
        src_t=src_t,
        src_hh=np.full((n_pairs, 1), hh),
    )


def _gd9d_sim(**kw):
    z = np.linspace(-_GD9D_WL / 4, _GD9D_WL / 4, 3)
    return SinusoidalSolver(
        wires=[np.column_stack([np.zeros(3), np.zeros(3), z])],
        n_per_edge_per_wire=[[1, 1]],
        nsegs=2,
        wavelength=_GD9D_WL,
        wire_radius=0.001,
        **kw,
    )


# (label, cos_shape, ek payload builder). All four combinations the dispatch
# accepts: the two reduced shapes, EKSCX's per-end gating on the literal
# shape, and the folded reduced-plus-delta pair rule.
_GD9D_PAYLOADS = {
    "reduced-cos": ("cos", None),
    "reduced-folded": ("cos-1", None),
    "ekscx": ("cos", "ekscx"),
    "reduced+delta": ("cos-1", "pairs"),
}


@pytest.mark.parametrize("payload", list(_GD9D_PAYLOADS))
def test_gd9d_the_field_kernel_ignores_its_batch_shape(payload):
    """The same pairs, cut into blocks of 1 … 256, give the same tables — bit
    for bit, on every shape and every extended-kernel payload.

    256 pairs x 128 observers is a 512 KB table, twice numpy's threshold, so
    the unnamed spelling fails this at blocks 128 and 256 and passes at every
    smaller one. Nothing about a pair's field depends on its blockmates —
    each is computed from its own (observer, source) geometry — so any
    difference here is the evaluation strategy reading the temporary's size.

    What this ASSERTS is host-independent; what would make it FAIL is not.
    Whether the elided in-place complex multiply rounds differently from the
    out-of-place one is a matter of which SIMD loop numpy dispatches to, and
    that follows the CPU: it differs on the machine momwire#392 was found on
    and does not on this repo's CI runner, where the unnamed spelling passes
    this gate as well. `test_gd9d_has_teeth` probes for it and skips rather
    than fails where it is absent. The named spellings are required either
    way — a value that depends on the host's vector width is worse than one
    that depends on the block size, not better.
    """
    a = 0.001
    n_pairs, n_obs = 256, 128
    cos_shape, kind = _GD9D_PAYLOADS[payload]
    sim = _gd9d_sim(extended_kernel=kind is not None)
    geo = _gd9d_pairs(n_pairs, n_obs, _GD9D_WL / 80.0, a)
    k = 2.0 * np.pi / _GD9D_WL
    assert n_pairs * n_obs * 16 > 256 * 1024

    def ek_for(count):
        if kind is None:
            return None
        on = np.ones((count, 1), dtype=bool)
        if kind == "ekscx":
            return (a, on, on)
        return _EKPairs(a, on, n_panels=4)

    def tables(block):
        out = {}
        for p0 in range(0, n_pairs, block):
            p1 = min(p0 + block, n_pairs)
            cm = sim._field_components_bcast(
                k,
                obs_c=geo["obs_c"][p0:p1],
                obs_t=geo["obs_t"][p0:p1],
                a=geo["a"],
                src_c=geo["src_c"][p0:p1],
                src_t=geo["src_t"][p0:p1],
                src_hh=geo["src_hh"][p0:p1],
                cos_shape=cos_shape,
                ek=ek_for(p1 - p0),
            )
            for key, val in cm.items():
                out.setdefault(key, []).append(np.asarray(val))
        return {key: np.concatenate(v, axis=0) for key, v in out.items()}

    ref = tables(1)
    scale = np.max(np.abs(ref["Ez_const"])) + np.max(np.abs(ref["Erho_const"]))
    assert scale > 0.0
    for block in (8, 64, 128, 256):
        got = tables(block)
        for key, want in ref.items():
            if want.dtype != np.complex128:
                continue
            assert np.array_equal(got[key], want), (
                f"{payload}: {key} moved by "
                f"{np.max(np.abs(got[key] - want)) / scale:.2e} of the const "
                f"field between a 1-pair and a {block}-pair call — the batch "
                f"shape is reaching the arithmetic (momwire#392)"
            )


def _gd9d_elide_spellings(n):
    """`G0 * (bracket)` against the same product with the bracket named.

    The forbidden spelling and its fix, on `n` synthetic complex entries and
    nothing else — the whole mechanism in four lines.
    """
    rng = np.random.default_rng(392)
    g0 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    u = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    elided = g0 * (u + v)
    named = u + v
    named = g0 * named
    return elided, named


def test_gd9d_has_teeth():
    """Whether G-D9d can fail on THIS host, reported rather than assumed.

    The two spellings differ only in whether numpy elides the temporary into
    an in-place multiply, so any difference between them is a difference
    between two SIMD loops — and which loop runs follows the CPU. On the
    machine momwire#392 was found on they differ above 256 KB and agree
    below it, which is what gives G-D9d its bite; on this repo's CI runner
    they agree at both sizes, and G-D9d passes there with or without the fix.

    So this SKIPS where the mechanism is absent instead of failing: the
    absence is a fact about the host, not a regression. What it still
    asserts unconditionally is the half that must hold everywhere — that the
    two spellings agree BELOW the threshold. If that ever broke, the named
    spellings would be changing the arithmetic rather than pinning the
    evaluation strategy, and momwire#392's central claim (that the fix lands
    on the value every small batch already produced) would be false.
    """
    small = _gd9d_elide_spellings(1000)  # 16 KB
    assert np.array_equal(*small), (
        "the elided and named spellings differ BELOW numpy's threshold — the "
        "fix would then be a change of arithmetic, not of evaluation strategy"
    )
    big = _gd9d_elide_spellings(40_000)  # 625 KB
    if np.array_equal(*big):
        pytest.skip(
            "this host's numpy rounds the elided and the named complex "
            "product identically at 625 KB, so G-D9d cannot fail here. That "
            "is not a reason to relax the named spellings in "
            "`_field_components_bcast` / `_ek_end_gxx`: on the hosts where "
            "the two DO differ, the same deck at the same N disagrees "
            "between machines and between block sizes without them"
        )
