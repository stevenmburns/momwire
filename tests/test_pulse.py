"""`PulseSolver`: the formulation, the ladder, and the scheme's own floor.

momwire#416's probe row. The gates here are deliberately NOT "beats
bspline" — pulse plus point matching is the slowest-converging scheme
momwire ships. They are:

1. the assembled matrix IS the row the module docstring derives, checked
   against a brute-force fill written from that docstring alone;
2. the free-space ladder approaches `BSplineSolver` MONOTONICALLY, at the
   measured level pinned below;
3. the scheme's accuracy is governed by Δ/a and not by Δ/λ — it is at its
   best at Δ/a ≈ 1 and comes apart below it, which is momwire#248's house
   rule arriving from the other side;
4. junctions need no basis work at all (roundoff-identical split), and the
   current balance they do NOT enforce decays at O(h);
5. reciprocity: point matching does not give a symmetric Z, so the
   asymmetry is MEASURED and attributed rather than asserted away.

The ground gates live in `tests/test_pulse_ground.py` and the refusals in
`tests/test_pulse_capabilities.py`.
"""

import numpy as np
import pytest

from momwire import BSplineSolver, CancelToken, PulseSolver, SolveAborted

# 14 MHz, the momwire#416 brief's ladder deck: a half-wave dipole at 0.95
# resonance shortening on a 20 mm-radius (fat, but perfectly real —
# aluminium tube) conductor. The radius is chosen so the N = 512 rung lands
# at Δ/a ≈ 1, the floor momwire#248 sets and the exact place this scheme is
# sharpest.
WAVELENGTH = 299.792458 / 14.0
DIP_LEN = 0.5 * WAVELENGTH * 0.95
DIP_RAD = 0.02
LADDER = (32, 64, 128, 256, 512)
# The in-house reference. 400 segments at degree 2 sits within 0.12 Ω of the
# 600-segment answer (measured), which is a tenth of the gap the ladder's
# last rung is pinned at.
REF_NSEGS = 400


def _dipole(z=0.0):
    return np.array([[0.0, -DIP_LEN / 2, z], [0.0, DIP_LEN / 2, z]])


def _z_pulse(n, radius=DIP_RAD, **kw):
    z, _ = PulseSolver(
        wires=[_dipole()],
        nsegs=n,
        wire_radius=radius,
        wavelength=WAVELENGTH,
        **kw,
    ).compute_impedance()
    return complex(z)


def _z_reference(radius=DIP_RAD, **kw):
    z, _ = BSplineSolver(
        wires=[_dipole()],
        n_per_edge_per_wire=[[REF_NSEGS]],
        wire_radius=radius,
        wavelength=WAVELENGTH,
        degree=2,
        **kw,
    ).compute_impedance()
    return complex(np.atleast_1d(z)[0])


# --------------------------------------------------------------------------
# 1. the matrix is the documented row
# --------------------------------------------------------------------------


def _reference_fill(sim):
    """The module docstring's row, written out entry by entry.

    Deliberately shares nothing with the solver: every kernel evaluation is
    a fresh `exp(-jkR)/(4πR)`, the vector term's segment integral is a
    single flat 64-point Gauss-Legendre rule (no static/remainder split),
    and the charge term is spelled as its four named endpoint pairs. The
    deck below is fat enough (h/a = 2) that the flat rule is converged to
    machine precision on the self term, so this is an independent
    statement of the formula and not a re-run of the implementation.
    """
    geom = sim._build_geometry()
    seg_l, seg_r = geom["seg_l"], geom["seg_r"]
    tang, h = geom["tangents"], geom["h_per_seg"]
    n, a2 = h.size, sim.wire_radius**2
    k, omega = sim.k, sim.omega
    cent = 0.5 * (seg_l + seg_r)
    xg, wg = np.polynomial.legendre.leggauss(64)

    def g(p, q):
        R = np.sqrt(np.sum((p - q) ** 2) + a2)
        return np.exp(-1j * k * R) / (4.0 * np.pi * R)

    Z = np.empty((n, n), dtype=np.complex128)
    for m in range(n):
        for j in range(n):
            tau = 0.5 * h[j] * (1.0 + xg)
            M0 = float(0.5 * h[j]) * sum(
                wg[i] * g(cent[m], seg_l[j] + tau[i] * tang[j]) for i in range(64)
            )
            ddg = (
                g(seg_r[m], seg_r[j])
                - g(seg_r[m], seg_l[j])
                - g(seg_l[m], seg_r[j])
                + g(seg_l[m], seg_l[j])
            )
            Z[m, j] = 1j * omega * sim.mu * h[m] * float(
                tang[m] @ tang[j]
            ) * M0 + ddg / (1j * omega * sim.eps)
    return Z


def test_the_matrix_is_the_row_the_docstring_derives():
    """`_assemble_Z` == the brute-force fill, on a bent two-wire deck.

    A bend and a second wire so the tangent dot product, the per-row h_m
    and the four-point stencil's cross terms are all exercised; 1e-12
    relative, which is the closed-form static moment against a converged
    flat quadrature and nothing else.
    """
    sim = PulseSolver(
        wires=[
            np.array([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0], [0.6, 0.5, 0.3]]),
            np.array([[0.0, 0.4, 0.9], [0.7, 0.4, 0.9]]),
        ],
        n_per_edge_per_wire=[[3, 2], [3]],
        wire_radius=0.1,
        wavelength=8.0,
        n_qp_source=32,
    )
    geom = sim._build_geometry()
    Z = sim._assemble_Z(geom, sim.k)
    ref = _reference_fill(sim)
    assert np.abs(Z - ref).max() / np.abs(ref).max() < 1e-12


def test_the_charge_term_is_the_endpoint_stencil_of_the_continuity_charge():
    """Sanity on the charge model itself, from continuity rather than from
    the assembled matrix: the potential a single pulse raises at a distant
    point equals that of ±I/(jω) at its two ends, computed by hand.
    """
    sim = PulseSolver(
        wires=[np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])],
        nsegs=4,
        wire_radius=0.05,
        wavelength=6.0,
    )
    geom = sim._build_geometry()
    obs = np.array([[0.3, 2.0, 1.0]])
    k, a2 = sim.k, sim.wire_radius**2

    def g(p, q):
        R = np.sqrt(np.sum((p - q) ** 2) + a2)
        return np.exp(-1j * k * R) / (4.0 * np.pi * R)

    n = 2
    by_hand = g(obs[0], geom["seg_r"][n]) - g(obs[0], geom["seg_l"][n])
    stacked = sim._point_g(obs, np.vstack([geom["seg_l"], geom["seg_r"]]), k)
    from_solver = stacked[0, geom["h_per_seg"].size + n] - stacked[0, n]
    assert abs(from_solver - by_hand) < 1e-13 * abs(by_hand)


# --------------------------------------------------------------------------
# 2. the free-space ladder
# --------------------------------------------------------------------------


def test_free_space_ladder_approaches_the_reference_monotonically():
    """N = 32…512 against the converged `BSplineSolver`, measured 2026-08-18:

    |    N | Δ/a  | Z (Ω)               | |Z − Z_ref| |
    |------|------|---------------------|-------------|
    |   32 | 15.9 |  54.202 − 416.353j  |    417.8    |
    |   64 |  7.9 |  60.202 − 155.668j  |    157.2    |
    |  128 |  4.0 |  67.682 −  40.672j  |     42.0    |
    |  256 |  2.0 |  71.919 −   3.575j  |      4.6    |
    |  512 |  1.0 |  72.734 +   1.101j  |      0.38   |
    | ref  |      |  72.445 +   1.125j  |             |

    Every rung improves on the one before, by better than 2.5× — and the
    ratio GROWS (2.7, 3.7, 9.1, 12.2) because two error mechanisms retire
    together as the mesh refines: the O(Δ/λ) discretization every scheme
    pays, and the point-charge concentration error that is this scheme's
    own (see the Δ/a test below).

    The pins are the honest ones: a factor-2.5 improvement per rung, and
    0.6 Ω at the last — 1.6× the 0.38 Ω measured, room for the ~5 %
    allocator/BLAS variance the house sees on CI without room to hide a
    regression that would show as a percent of a 72 Ω impedance.
    """
    ref = _z_reference()
    errs = [abs(_z_pulse(n) - ref) for n in LADDER]
    for lo, hi, n_lo, n_hi in zip(errs[1:], errs[:-1], LADDER[1:], LADDER[:-1]):
        assert lo < hi / 2.5, f"N={n_hi}→{n_lo}: {hi:.3g} → {lo:.3g} is not halving"
    assert errs[-1] < 0.6, f"N={LADDER[-1]}: {errs[-1]:.3g} Ω from the reference"


def test_the_scheme_is_governed_by_delta_over_a_not_delta_over_lambda():
    """The floor, stated as a measurement rather than as a caveat.

    This basis's charge is two POINT charges observed at their own
    location, regularised only by the reduced kernel's a². The
    concentration error that costs is a function of Δ/a — so refining a
    mesh PAST Δ/a ≈ 1 does not merely stop helping, it reverses, and
    momwire#248's "never validate below Δ/a ≈ 1" arrives here as a
    property of the scheme rather than as a comparison convention.

    On a 50 mm-radius conductor (measured 2026-08-18): Δ/a = 3.2 → 5.0 Ω
    from the reference, Δ/a = 1.6 → 1.5 Ω, and then Δ/a = 0.4 → 55 Ω, a
    36× regression from a mesh four times finer.
    """
    fat = 0.05
    ref = _z_reference(radius=fat)
    err = {n: abs(_z_pulse(n, radius=fat) - ref) for n in (64, 128, 512)}
    assert err[128] < err[64], "above the floor, refining must still help"
    assert err[512] > 10 * err[128], (
        "below Δ/a ≈ 1 the point-charge model must come apart — it did not, "
        f"so the floor this scheme documents has moved: {err}"
    )


# --------------------------------------------------------------------------
# 3. junctions: nothing to do, and what that costs
# --------------------------------------------------------------------------


def test_a_colinear_split_is_the_same_antenna_to_roundoff():
    """One wire of 2N segments == two wires of N meeting at the midpoint.

    The sharpest possible statement that this basis needs no junction
    machinery: the two spellings produce the same segment set, and because
    coincident endpoint charges superpose by arithmetic alone the two Z
    matrices are the same matrix. Measured 3e-15 relative — LU roundoff,
    not agreement.
    """
    p0, p1 = _dipole()[0], _dipole()[1]
    mid = 0.5 * (p0 + p1)
    n = 48
    kw = dict(wire_radius=DIP_RAD, wavelength=WAVELENGTH, feeds=[(0, DIP_LEN / 4, 1j)])
    z1, c1 = PulseSolver(
        wires=[np.array([p0, p1])], n_per_edge_per_wire=[[2 * n]], **kw
    ).compute_impedance()
    z2, c2 = PulseSolver(
        wires=[np.array([p0, mid]), np.array([mid, p1])],
        n_per_edge_per_wire=[[n], [n]],
        **kw,
    ).compute_impedance()
    assert abs(z1 - z2) / abs(z1) < 1e-13
    assert np.abs(c1 - c2).max() < 1e-13 * np.abs(c1).max()


def test_junction_current_balance_decays_at_first_order():
    """What the basis does NOT enforce, measured.

    Three wires leaving one node. The pulse basis has no continuity
    constraint, so Σ I out of the node is not zero — it is jω times the
    point charge sitting there, a real O(h) discretization artefact. It
    must vanish with the mesh, and it does, halving per rung: measured
    5.2e-2, 2.3e-2, 9.8e-3, 4.2e-3 of the largest branch current at
    N = 8, 16, 32, 64.

    Pinned as "each rung is at most 60 % of the last" — comfortably above
    the measured 0.42-0.45 ratio, and far below the 1.0 that a scheme with
    a stalled or absent balance would show.
    """
    node = np.array([0.0, 0.0, 2.0])
    wires = [
        np.array([node, node + [0.0, 0.0, 2.5]]),
        np.array([node, node + [2.0, 0.0, 0.0]]),
        np.array([node, node + [-1.4, 0.9, 0.0]]),
    ]
    prev = None
    for n in (8, 16, 32, 64):
        sim = PulseSolver(
            wires=wires,
            nsegs=n,
            wire_radius=DIP_RAD,
            wavelength=WAVELENGTH,
            feeds=[(0, 1.2, 1.0 + 0j)],
        )
        _z, coeffs = sim.compute_impedance()
        off = sim._build_geometry()["seg_offsets"]
        # Every wire STARTS at the node, so each first segment's current
        # flows away from it and the imbalance is the plain sum.
        first = [coeffs[off[w]] for w in range(3)]
        rel = abs(sum(first)) / max(abs(c) for c in first)
        if prev is not None:
            assert rel < 0.6 * prev, f"N={n}: balance {prev:.3g} → {rel:.3g}"
        prev = rel
    assert prev < 0.01


# --------------------------------------------------------------------------
# 4. reciprocity
# --------------------------------------------------------------------------


def test_reciprocity_error_is_the_vector_term_alone():
    """Point matching does not give a symmetric Z. Which half is guilty is
    not a matter of opinion, and this measures it instead of bounding it.

    The charge term ΔΔg is EXACTLY symmetric — g is symmetric and the m↔n
    swap transposes the four-point stencil — so it comes back at 1e-16.
    The vector term h_m M0[c_m, n] is the midpoint rule on one side of a
    pairing Galerkin testing would integrate on both, and it is where the
    whole asymmetry lives: measured 2.2e-5, 4.2e-6, 7.4e-7 relative at
    N = 12, 24, 48 per wire on an asymmetric bent-plus-parasitic deck,
    decaying faster than O(h²).

    Nothing is symmetrised. The formulation is what it is; this test is the
    record of what it costs.
    """
    wires = [
        np.array([[0.0, -3.0, 1.0], [0.0, 2.0, 1.0], [1.5, 2.0, 2.2]]),
        np.array([[0.7, -2.0, 0.4], [0.7, 3.1, 0.4]]),
    ]
    prev = None
    for n in (12, 24, 48):
        sim = PulseSolver(
            wires=wires, nsegs=n, wire_radius=DIP_RAD, wavelength=WAVELENGTH
        )
        geom = sim._build_geometry()
        t, h = geom["tangents"], geom["h_per_seg"]
        cent = 0.5 * (geom["seg_l"] + geom["seg_r"])
        T1 = (h[:, None] * (t @ t.T)) * sim._seg_M0(cent, geom, sim.k)
        T2 = sim._charge_stencil(geom, geom, sim.k)
        assert np.abs(T2 - T2.T).max() / np.abs(T2).max() < 1e-14, (
            "the charge term is symmetric by construction — if this fires, "
            "the four-point stencil has stopped being the stencil"
        )
        a1 = np.abs(T1 - T1.T).max() / np.abs(T1).max()
        assert 1e-9 < a1 < 1e-3, f"N={n}: vector-term asymmetry {a1:.3g}"
        if prev is not None:
            assert a1 < prev / 3.0, f"N={n}: asymmetry {prev:.3g} → {a1:.3g}"
        prev = a1


# --------------------------------------------------------------------------
# 5. feeds, ports and the readout
# --------------------------------------------------------------------------


def test_the_feed_snaps_to_a_segment_centroid():
    """A pulse row is a segment, so a delta gap lands on a segment — the
    opposite snap from the knot-centred tent solvers. Asking for a feed a
    third of the way along must land on the segment holding that arc.
    """
    sim = PulseSolver(
        wires=[_dipole()],
        nsegs=9,
        wire_radius=DIP_RAD,
        wavelength=WAVELENGTH,
        feed_arclength=DIP_LEN / 3.0,
    )
    geom = sim._build_geometry()
    (idx,) = sim._feed_basis_indices(geom)
    arc = geom["per_wire"][0]["arc_at_knot"]
    assert arc[idx] <= DIP_LEN / 3.0 <= arc[idx + 1]


def test_two_port_admittance_and_the_single_solve_agree():
    """`compute_y_matrix` is the same fill: driving port 0 with 1 V and
    port 1 with 0 V reproduces column 0 of Y, and the reciprocity gap in
    Y_01 vs Y_10 is the formulation's, not a bookkeeping slip — bounded
    here at 1 %, measured well under it.
    """
    wires = [_dipole(z=0.0), _dipole(z=1.7)]
    kw = dict(
        wires=wires,
        nsegs=48,
        wire_radius=DIP_RAD,
        wavelength=WAVELENGTH,
        feeds=[(0, None, 1.0 + 0j), (1, None, 0.0 + 0j)],
    )
    Y = PulseSolver(**kw).compute_y_matrix()
    _z, coeffs = PulseSolver(**kw).compute_impedance()
    geom = PulseSolver(**kw)._build_geometry()
    idx = PulseSolver(**kw)._feed_basis_indices(geom)
    assert np.allclose(coeffs[idx], Y[:, 0], rtol=1e-10)
    assert abs(Y[0, 1] - Y[1, 0]) < 0.01 * abs(Y[0, 1])


def test_currents_at_knots_reads_the_staircase_honestly():
    """Interior knots are the mean of their two pulses; a free wire END
    reports its terminal pulse and is NOT zeroed.

    Reporting 0 at the tip would hide this basis's largest artefact from
    every field consumer. It is a real number, not a rounding one: 5.4 %
    of the peak current at N = 64 (measured), which is why the tip pin
    below is 10 % and not a roundoff bar.
    """
    sim = PulseSolver(
        wires=[_dipole()], nsegs=64, wire_radius=DIP_RAD, wavelength=WAVELENGTH
    )
    _z, coeffs = sim.compute_impedance()
    (knots,) = sim.currents_at_knots(coeffs)
    assert knots.size == 65
    assert np.allclose(knots[1:-1], 0.5 * (coeffs[:-1] + coeffs[1:]))
    assert knots[0] == coeffs[0] and knots[-1] == coeffs[-1]
    assert abs(knots[0]) < 0.1 * np.abs(coeffs).max()


def test_element_currents_rides_on_the_mixin_unchanged():
    """`_ElementCurrents` needs `wires_polylines`, the normalised
    `n_per_edge_per_wire` and a `currents_at_knots` — this row supplies all
    three and writes no readout of its own. The per-wire continuity steps
    must telescope to zero, which is the mixin's own invariant.
    """
    sim = PulseSolver(
        wires=[_dipole()], nsegs=32, wire_radius=DIP_RAD, wavelength=WAVELENGTH
    )
    _z, coeffs = sim.compute_impedance()
    mid, moment, nodes, delta = sim.element_currents(coeffs)
    assert mid.shape == (32, 3) and moment.shape == (32, 3)
    assert nodes.shape == (33, 3) and delta.shape == (33,)
    assert abs(delta.sum()) < 1e-18
    mid3, _m, _n, _d = sim.element_currents(coeffs, subdiv=3)
    assert mid3.shape == (96, 3)


def test_cancel_token_aborts_the_fill():
    """`_Cancelable` consumed unchanged: a token tripped before the solve
    raises at the first checkpoint instead of running the fill."""
    token = CancelToken()
    sim = PulseSolver(
        wires=[_dipole()],
        nsegs=64,
        wire_radius=DIP_RAD,
        wavelength=WAVELENGTH,
        cancel=token,
    )
    token.cancel()
    with pytest.raises(SolveAborted):
        sim.compute_impedance()
