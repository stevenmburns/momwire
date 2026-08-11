"""The extended thin-wire kernel for the Galerkin family (momwire#246).

Unit A: the delta kernel and the seam. `SinusoidalSolver
._folded_ek_delta_fields` computes the EK-minus-reduced FIELD correction by
direct Gauss-Legendre quadrature of a smooth kernel, and
`_field_components_bcast` dispatches on the EK payload's type so that the
folded (`cos_shape="cos-1"`) and point-matched (`"cos"`) contracts can no
longer be silently confused.

The gates here are, in order:

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
"""

import numpy as np
import pytest

from momwire.sinusoidal import _EKPairs, _N_QP_EK_DELTA, SinusoidalSolver

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
Z_OVER_DELTA = (1.0, 2.0, 10.0)

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

# G-A4: N = 16 against N = 48. Measured 2.7e-15 — the same size as G-A1's
# number and for the same reason: with truncation 1e-20 away, what separates
# two node counts is the rounding of two different sums, and the longer sum
# carries more of it. (The design's E1 reached 1e-15 with 8-12 nodes.)
NODE_CONVERGENCE = 1e-14


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
    """(H, z, a) over the box. On an EK-eligible pair the observer is coaxial
    with the source and carries the same radius, so ρ_eval == a exactly."""
    for da in DELTA_OVER_A:
        for kh in KH:
            H = kh / K
            seg = 2.0 * H
            a = seg / da
            for zf in Z_OVER_DELTA:
                yield H, zf * seg, a


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


def _ld_delta(eta, H, z, rho, a, shape, n=512):
    """(ΔE_z, ΔE_ρ) of one source shape, at 80 bits and N nodes.

    Independent of the production routine in precision, in node count and in
    spelling: the four u-derivatives of the reduced kernel are written out
    from their own algebra (each Aₙ as an explicit polynomial in kR with the
    j's resolved by hand into real and imaginary parts) rather than through
    the reverse-Bessel recursion, and the operator brackets are assembled in
    a different grouping. What it shares with the production routine is the
    scheme — which is the point: this measures the float64 ARITHMETIC, in the
    idiom of `test_sinusoidal_galerkin.py`'s 60-digit `cos kξ − 1` reference.
    """
    gx, gw = _ld_leggauss(n)
    kL = np.longdouble(K)
    HL, zL, rL, aL = (np.longdouble(v) for v in (H, z, rho, a))
    xi = HL * gx
    w = HL * gw
    zeta = zL - xi
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
    worst = 0.0
    worst_same_nodes = 0.0
    for H, z, a in _probes():
        scale = _const_scale(_coaxial_tables(sim, H, z, a))
        got = sim._folded_ek_delta_fields(
            K,
            np.array([H]),
            np.array([z]),
            np.array([a]),
            np.array([a]),
            cos_shape=cos_shape,
        )
        for shape in ("const", "sin", "cos"):
            ref_shape = shape
            if shape == "cos" and cos_shape == "cos-1":
                ref_shape = "folded"
            for n, bound in ((512, "worst"), (_N_QP_EK_DELTA, "same")):
                ez, er = _ld_delta(sim.eta, H, z, a, a, ref_shape, n=n)
                err = max(
                    abs(got[f"Ez_{shape}"][0] - ez) / scale,
                    abs(got[f"Erho_{shape}"][0] - er) / scale,
                )
                if bound == "worst":
                    worst = max(worst, err)
                else:
                    worst_same_nodes = max(worst_same_nodes, err)
    assert worst < DELTA_VS_LONGDOUBLE, worst
    assert worst_same_nodes < DELTA_ARITHMETIC, worst_same_nodes


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
    worst = dict.fromkeys(
        ("Ez_const", "Erho_const", "Ez_sin", "Erho_sin", "Ez_cos", "Erho_cos"), 0.0
    )
    for H, z, a in _probes():
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
        )
        for key in worst:
            got = red[key][0, 0] + delta[key][0, 0]
            worst[key] = max(worst[key], abs(got - ekscx[key][0, 0]) / scale)
    assert max(worst.values()) < DELTA_VS_EKSCX, worst
    # E_z carries no near-cancelling endpoint group in either route, so it is
    # the digit-for-digit statement; E_ρ is held to EKSCX's own noise above.
    assert max(worst[k] for k in ("Ez_const", "Ez_sin", "Ez_cos")) < 1e-14, worst


# ---------------------------------------------------------------------------
# G-A3 — the a → 0 collapse
# ---------------------------------------------------------------------------
def test_delta_is_exactly_zero_at_zero_radius():
    """Structural, not asymptotic: the whole correction is linear in a², and
    that factor is applied as one multiplication at the end, so a = 0 gives
    IEEE zeros rather than something that rounds to them."""
    sim = _solver()
    H, z, a = next(iter(_probes()))
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
    worst = 0.0
    for H, z, a in _probes():
        scale = _const_scale(_coaxial_tables(sim, H, z, a))
        args = (K, np.array([H]), np.array([z]), np.array([a]), np.array([a]))
        lo = sim._folded_ek_delta_fields(*args)
        hi = sim._folded_ek_delta_fields(*args, n_gl=48)
        for key in lo:
            worst = max(worst, abs(lo[key][0] - hi[key][0]) / scale)
    assert worst < NODE_CONVERGENCE, worst


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------
def test_seam_serves_the_folded_delta_through_field_components():
    """`_EKPairs` + `cos_shape="cos-1"` returns the #205 folded reduced tables
    plus the delta, and nothing else moves: `td`, the ρ̂ projection and the
    geometry columns are EFLD's, computed before the two kernels part."""
    sim = _solver()
    H, z, a = next(iter(_probes()))
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
    H, z, a = next(iter(_probes()))
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
    H, z, a = next(iter(_probes()))
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
