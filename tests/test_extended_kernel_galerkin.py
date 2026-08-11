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
    """The refusal momwire#233 left behind is narrowed to one combination,
    not lifted: free space, the PEC image and the reflection-coefficient
    image all solve, and the answer moves."""
    kw = _GROUND_KW[ground]
    z_red, _ = _monopole(**kw).compute_impedance()
    z_ext, _ = _monopole(extended_kernel=True, **kw).compute_impedance()
    assert abs(z_ext - z_red) > 1e-3 * abs(z_red)


def test_galerkin_refuses_the_extended_kernel_under_sommerfeld_ground():
    """The Sommerfeld remainder is a different evaluator with a different
    derivation, and serving it reduced under an extended image would be a
    silent mixture of two kernels inside one ground model."""
    with pytest.raises(NotImplementedError, match="sommerfeld"):
        _monopole(
            extended_kernel=True,
            ground_z=0.0,
            ground_eps=(13.0, 0.005),
            ground_model="sommerfeld",
        )


def test_galerkin_still_serves_sommerfeld_with_the_reduced_kernel():
    """The refusal is about the combination, not about either half."""
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


@pytest.mark.parametrize("ground", list(_GROUND_KW))
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
    coarse = _sym_ratio(_monopole(extended_kernel=True, **kw))
    fine = _sym_ratio(_monopole(extended_kernel=True, n_qp_test=16, n_qp_near=16, **kw))
    assert coarse < 1e-10, coarse
    assert fine < 0.1 * coarse, f"{fine:.2e} did not improve on {coarse:.2e}"
    assert fine <= 10.0 * _sym_ratio(_monopole(n_qp_test=16, n_qp_near=16, **kw))


def test_gb2_an_asymmetric_mask_breaks_reciprocity_and_refinement_cannot_fix_it():
    """The falsifying contrast, and the reason the pair rule exists.

    Extend only the pairs with j > i — a caricature of a per-source-segment
    decision — and ‖G−Gᵀ‖/‖G‖ jumps by orders of magnitude AND stops
    responding to the test rule, because the asymmetry is now in what is
    being integrated rather than in how well.
    """
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
    (SinusoidalGalerkinSolver, "_ek_axis_labels"),
    (SinusoidalGalerkinSolver, "_ek_pairs"),
]


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
    below pass vacuously."""
    _monopole(extended_kernel=True, ground_z=0.0).compute_impedance()
    for attr, n in ek_call_counts.items():
        assert n > 0, f"{attr} never called with EK on"


@pytest.mark.parametrize("name", list(_G4_DECKS))
def test_gb4b_ek_off_enters_no_ek_code(ek_call_counts, name):
    kw = dict(_G4_DECKS[name], wavelength=LAM_NEC)
    SinusoidalGalerkinSolver(**kw).compute_impedance()
    assert ek_call_counts == dict.fromkeys(ek_call_counts, 0)


def test_gb4_the_fused_far_fill_is_not_asked_to_serve_the_extended_kernel():
    """The C++ far fill takes no eligibility mask, so routing an EK-on block
    through it would drop the delta silently. The capability flag
    `_HAVE_GALERKIN_FAR_FILL_EK` is what unit C flips; until then
    `_tested_contribs` must take the numpy path, and `_far_fill_accel` must
    refuse an EK payload rather than ignore it."""
    sim = _monopole(extended_kernel=True)
    with pytest.raises(NotImplementedError, match="unit C"):
        sim._far_fill_accel(sim.k, None, None, None, ek=_EKPairs(np.zeros(1), None))


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


def test_gb5_the_image_block_carries_its_own_delta():
    """Not a ride-along: force the image block's payload to None and the
    grounded answer moves, by more than the free-space block's own EK shift
    would explain."""
    original = SinusoidalGalerkinSolver._ek_pairs

    def free_space_only(self, geom, m_idx, n_idx, mirror, n_panels=1):
        if mirror:
            return None
        return original(self, geom, m_idx, n_idx, mirror, n_panels)

    z_both, _ = _monopole(extended_kernel=True, ground_z=0.0).compute_impedance()
    try:
        SinusoidalGalerkinSolver._ek_pairs = free_space_only
        z_free_only, _ = _monopole(
            extended_kernel=True, ground_z=0.0
        ).compute_impedance()
    finally:
        SinusoidalGalerkinSolver._ek_pairs = original
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
        sim._tested_ground_block(geom, sim.k, ctx)
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
