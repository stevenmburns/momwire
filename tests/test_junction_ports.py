"""Junction ports (#172): voltage-drivable ports at junction nodes.

A junction port promotes a junction group to a network port: its KCL
closure row leaves the constraint set (the grounded-junction #151 move) and
becomes the port's Galerkin drive/readout vector — voltage V excites
``v += V·A_p`` and the port current reads ``I_p = A_p·coeffs``, the same
reciprocity pairing as delta-gap feeds. This is the end-attachment
mechanism for stevenmburns/antennaknobs#579 (a network element terminal on a
conductor END), whose saddle-system math was validated first as an oracle
built on the solver's exposed pieces (V_port = −λ; paired-drive convergence
to a bridged-gap reference at 1.1 %/0.7 % after gap-size extrapolation).

Oracles here:
  * driving a split dipole through two tip junction ports reproduces the
    same structure driven by a real bridged delta-gap feed, converging
    linearly as the split gap shrinks (Richardson-extrapolated difference
    < 1.5 %);
  * a 1-entry junction group (now legal) with no port role reproduces the
    free-end solution exactly;
  * mixed gap+junction-port Y is symmetric and consistent with
    compute_impedance readouts; swept paths agree with per-k solves.

The final section runs the SAME oracle against `SinusoidalGalerkinSolver`
(momwire#182 M5) and records why the sinusoidal family does not get junction
ports even with Galerkin test rows in hand — see that section's header.
"""

import numpy as np
import pytest
import scipy.linalg

from momwire import (
    BSplineSolver,
    HMatrixSolver,
    SinusoidalGalerkinSolver,
    SinusoidalSolver,
)

WAVELENGTH = 10.0
L_DIP = 0.48 * WAVELENGTH


def _dipole_solver(n=41, **kw):
    wire = [np.array([(0.0, -L_DIP / 2, 0.0), (0.0, L_DIP / 2, 0.0)])]
    return BSplineSolver(
        wires=wire,
        n_per_edge_per_wire=[[n]],
        feeds=[(0, L_DIP / 2, 1.0 + 0j)],
        wavelength=WAVELENGTH,
        **kw,
    )


def _split_wires(delta, whisker, n_half):
    """Two dipole halves with tip whiskers hosting the tip junctions."""
    wires = [
        np.array([(0.0, -L_DIP / 2, 0.0), (0.0, -delta / 2, 0.0)]),
        np.array([(0.0, -delta / 2, 0.0), (whisker, -delta / 2, 0.0)]),
        np.array([(0.0, delta / 2, 0.0), (0.0, L_DIP / 2, 0.0)]),
        np.array([(0.0, delta / 2, 0.0), (whisker, delta / 2, 0.0)]),
    ]
    npe = [[n_half], [1], [n_half], [1]]
    junctions = [
        [(0, "end"), (1, "start")],
        [(2, "start"), (3, "start")],
    ]
    return wires, npe, junctions


def _bridged_z(delta, whisker, n_half, cls=BSplineSolver, **kw):
    """Reference: the same split structure with a real bridge wire across
    the gap carrying a classic delta-gap feed.

    `cls` lets the M5 section run the identical oracle on a different solver
    — the reference must be built by the SAME scheme as the port solve, or
    the comparison measures the schemes' basis gap rather than the port."""
    wires, npe, junctions = _split_wires(delta, whisker, n_half)
    wires.append(np.array([(0.0, -delta / 2, 0.0), (0.0, delta / 2, 0.0)]))
    npe.append([1])
    junctions[0].append((4, "start"))
    junctions[1].append((4, "end"))
    s = cls(
        wires=wires,
        n_per_edge_per_wire=npe,
        feeds=[(4, delta / 2, 1.0 + 0j)],
        junctions=junctions,
        wavelength=WAVELENGTH,
        **kw,
    )
    z, _ = s.compute_impedance()
    return complex(np.atleast_1d(z)[0])


def _port_pair_solver(delta, whisker, n_half, volts=(0j, 0j), cls=BSplineSolver, **kw):
    wires, npe, junctions = _split_wires(delta, whisker, n_half)
    return cls(
        wires=wires,
        n_per_edge_per_wire=npe,
        feeds=[(0, L_DIP / 4, 0.0 + 0j)],  # dummy 0 V gap feed
        junctions=junctions,
        junction_ports=[(0, volts[0]), (1, volts[1])],
        wavelength=WAVELENGTH,
        **kw,
    )


def _port_pair_zdiff(delta, whisker, n_half):
    """Differential impedance between the two tip ports, from the mixed
    Y matrix: invert the junction-port sub-block of Z = the 2x2 obtained by
    eliminating the (shorted, passive) dummy gap feed — i.e. take Y over
    [feed, p1, p2], and use the p-block of inv on the port sub-system with
    the feed held at 0 V (row/col simply excluded: V_feed = 0)."""
    s = _port_pair_solver(delta, whisker, n_half)
    Y = s.compute_y_matrix()
    assert Y.shape == (3, 3)
    Zp = np.linalg.inv(Y[1:, 1:])  # feed shorted: V_f = 0 drops its column
    return complex(Zp[0, 0] - Zp[0, 1] - Zp[1, 0] + Zp[1, 1])


def test_free_end_equals_one_entry_junction():
    """A 1-entry junction group with b = 0 is a free end enforced through
    the Lagrange row instead of the basis end-condition: same physics, so
    the classic dipole impedance must be reproduced to solver precision."""
    z_free, _ = _dipole_solver().compute_impedance()
    s = _dipole_solver(junctions=[[(0, "start")]])
    z_j, _ = s.compute_impedance()
    z_free = complex(np.atleast_1d(z_free)[0])
    z_j = complex(np.atleast_1d(z_j)[0])
    assert abs(z_j - z_free) / abs(z_free) < 1e-9


def test_paired_ports_converge_to_bridged_reference():
    """Drive the split dipole through its tip junction ports; the
    difference from the bridged-gap reference is the physical bridge metal,
    linear in the split gap delta — Richardson-extrapolate to delta -> 0 and
    demand < 1.5 % (the antennaknobs#579 oracle bound)."""
    n_half, whisker = 20, 0.01
    diffs = {}
    for delta in (0.04, 0.02):
        z_ref = _bridged_z(delta, whisker, n_half)
        z_prt = _port_pair_zdiff(delta, whisker, n_half)
        diffs[delta] = (z_prt - z_ref, z_ref)
    extrap = 2 * diffs[0.02][0] - diffs[0.04][0]
    assert abs(extrap) / abs(diffs[0.02][1]) < 0.015


def test_mixed_y_symmetric_and_reciprocal():
    s = _port_pair_solver(0.04, 0.01, 20)
    Y = s.compute_y_matrix()
    assert Y.shape == (3, 3)
    np.testing.assert_allclose(Y, Y.T, rtol=1e-8)


def test_compute_impedance_readout_consistent_with_y():
    """Voltage-driving the two tip ports (+0.5 / −0.5) through
    compute_impedance must agree with currents predicted by the Y matrix
    for the same port voltage vector."""
    volts = np.array([0.0 + 0j, 0.5 + 0j, -0.5 + 0j])
    s = _port_pair_solver(0.04, 0.01, 20, volts=(volts[1], volts[2]))
    z_per, coeffs = s.compute_impedance()
    z_per = np.atleast_1d(z_per)
    assert z_per.shape == (3,)
    # per-port currents from the impedance readout; the dummy gap feed is
    # driven at 0 V (its z entry is 0/I), so only the junction ports compare
    i_imp = volts[1:] / z_per[1:]
    Y = _port_pair_solver(0.04, 0.01, 20).compute_y_matrix()
    i_y = Y @ volts
    np.testing.assert_allclose(i_imp, i_y[1:], rtol=1e-8)


def test_swept_matches_per_k():
    s = _port_pair_solver(0.04, 0.01, 12)
    k0 = s.k
    ks = np.array([0.9 * k0, k0, 1.1 * k0])
    Y_swept = s.compute_y_matrix_swept(ks)
    for i, kk in enumerate(ks):
        s2 = _port_pair_solver(0.04, 0.01, 12)
        s2.k = float(kk)
        s2.omega = s2.k * s2.c
        s2.wavelength = s2.c / (s2.omega / (2 * np.pi))
        np.testing.assert_allclose(Y_swept[i], s2.compute_y_matrix(), rtol=1e-6)


def test_impedance_swept_matches_per_k_with_junction_ports():
    """The batched impedance sweep agrees with per-k compute_impedance when
    junction ports are present — with a gap feed and (issue #175) entirely
    feed-less, where the batched path used to crash on a 1-D hstack operand."""

    def per_k(make, ks):
        out = []
        for kk in ks:
            s = make()
            s.k = float(kk)
            s.omega = s.k * s.c
            s.wavelength = s.c / (s.omega / (2 * np.pi))
            z, _ = s.compute_impedance()
            out.append(np.atleast_1d(z))
        return np.array(out)

    def make_mixed():
        return _port_pair_solver(0.04, 0.01, 12, volts=(0.5 + 0j, -0.5 + 0j))

    def make_feedless():
        wires, npe, junctions = _split_wires(0.04, 0.01, 12)
        return BSplineSolver(
            wires=wires,
            n_per_edge_per_wire=npe,
            feeds=[],
            junctions=junctions,
            junction_ports=[(0, 0.5 + 0j), (1, -0.5 + 0j)],
            wavelength=WAVELENGTH,
        )

    k0 = make_mixed().k
    ks = np.array([0.9 * k0, k0, 1.1 * k0])
    for make in (make_mixed, make_feedless):
        assert make()._swept_batched_available()  # the path under test
        np.testing.assert_allclose(
            make().compute_impedance_swept(ks), per_k(make, ks), rtol=1e-6
        )


def test_zero_gap_feeds_allowed_with_junction_ports():
    """A solve driven entirely through junction ports needs no gap feed:
    feeds=[] is legal when junction_ports exist, and the Y matrix covers
    exactly the junction ports."""
    wires, npe, junctions = _split_wires(0.04, 0.01, 20)
    s = BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=npe,
        feeds=[],
        junctions=junctions,
        junction_ports=[0, 1],
        wavelength=WAVELENGTH,
    )
    Y = s.compute_y_matrix()
    assert Y.shape == (2, 2)
    Zp = np.linalg.inv(Y)
    z_diff = complex(Zp[0, 0] - Zp[0, 1] - Zp[1, 0] + Zp[1, 1])
    # same structure as _port_pair_zdiff (whose dummy feed is a shorted
    # gap = plain wire): the two must agree to solver precision
    z_ref = _port_pair_zdiff(0.04, 0.01, 20)
    assert abs(z_diff - z_ref) / abs(z_ref) < 1e-6


def test_grounded_junction_port_rejected():
    """A junction whose node touches the ground plane already has its KCL
    row dropped (current exits via the image, #151) — declaring it a port
    too is ambiguous and must be rejected."""
    h = 0.24 * WAVELENGTH
    wires = [
        np.array([(0.0, 0.0, 0.0), (0.0, 0.0, h)]),  # monopole
        np.array([(0.0, 0.0, 0.0), (0.01, 0.0, 0.01)]),  # slanted whisker
    ]
    s = BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=[[21], [1]],
        feeds=[(0, h / 2, 1.0 + 0j)],
        junctions=[[(0, "start"), (1, "start")]],
        junction_ports=[0],
        wavelength=WAVELENGTH,
        ground_z=0.0,
    )
    with pytest.raises(ValueError, match="grounded"):
        s.compute_impedance()


def test_validation_and_unsupported_solvers():
    wires, npe, junctions = _split_wires(0.04, 0.01, 8)
    with pytest.raises(ValueError, match="out of range"):
        BSplineSolver(
            wires=wires, n_per_edge_per_wire=npe,
            feeds=[(0, 1.0, 1.0 + 0j)], junctions=junctions,
            junction_ports=[7], wavelength=WAVELENGTH,
        )  # fmt: skip
    with pytest.raises(ValueError, match="twice"):
        BSplineSolver(
            wires=wires, n_per_edge_per_wire=npe,
            feeds=[(0, 1.0, 1.0 + 0j)], junctions=junctions,
            junction_ports=[0, 0], wavelength=WAVELENGTH,
        )  # fmt: skip
    # The iterative solvers took junction ports in #234; the refusal that
    # used to live here is gone (tests/test_junction_ports_iterative.py).
    HMatrixSolver(
        wires=wires, n_per_edge_per_wire=npe,
        feeds=[(0, 1.0, 1.0 + 0j)], junctions=junctions,
        junction_ports=[0], wavelength=WAVELENGTH,
    )  # fmt: skip
    with pytest.raises(NotImplementedError, match="outside its span.*bridge"):
        SinusoidalSolver(
            wires=wires, n_per_edge_per_wire=npe,
            feeds=[(0, 1.0, 1.0 + 0j)], junctions=junctions,
            junction_ports=[0], wavelength=WAVELENGTH,
        )  # fmt: skip


def test_enrichment_dense_fallback_allows_junction_ports():
    """Issue #176: use_singular_enrichment sends every solve down the dense
    BSplineSolver path, which fully supports junction ports — so the
    iterative solvers must accept the combination instead of rejecting it
    at construction. The results are the dense solver's own, verbatim."""
    from momwire import ArrayBlockSolver

    wires, npe, junctions = _split_wires(0.04, 0.01, 8)
    kw = dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        feeds=[(0, L_DIP / 4, 0.0 + 0j)],
        junctions=junctions,
        junction_ports=[(0, 0.5 + 0j), (1, -0.5 + 0j)],
        wavelength=WAVELENGTH,
        use_singular_enrichment=True,
    )
    y_ref = BSplineSolver(**kw).compute_y_matrix()
    z_ref, _ = BSplineSolver(**kw).compute_impedance()
    for solver in (HMatrixSolver, ArrayBlockSolver):
        s = solver(**kw)
        np.testing.assert_allclose(s.compute_y_matrix(), y_ref, rtol=1e-12)
        z, _ = solver(**kw).compute_impedance()
        np.testing.assert_allclose(z, z_ref, rtol=1e-12)


# ===========================================================================
# momwire#182 M5 — the same oracle on SinusoidalGalerkinSolver, and why the
# sinusoidal family still does not get junction ports
# ===========================================================================
#
# #177 blamed the point-matched solver's refusal on two things: the port
# vector is outside the basis span (true, and reconfirmed constructively
# below), and a *reciprocal* port would need segment-integrated test rows
# that the point-collocation field stack cannot provide. The Galerkin solver
# HAS those rows, so M5 built the port exactly as #177 specified — port basis
# column `g_p = (1/P_J)·Σ_m ext_m`, its own tested row, dual drive/readout —
# and ran it against this file's `_bridged_z` oracle.
#
# The construction is sound: unit inflow to 6e-15, matrix symmetric to 8e-12,
# port block of Y symmetric to 1.7e-14. The PHYSICS is not, for a reason #177
# did not anticipate and which no mesh work can reach:
#
#   a current with unit net inflow at a node TERMINATES there, so it deposits
#   a point charge q = I/(jω) at the node; the Eqs 78-79 endpoint terms —
#   the same width-`a` feature M2's graded rule exists to resolve — charge it
#   its own self-energy, regularized at the WIRE RADIUS. Z_pp comes out at
#   1/(jω·4πε·a), i.e. the self-capacitance of a sphere of radius a, and the
#   KCL-clean span cannot cancel it because no ordinary basis carries net
#   charge at a junction. That is precisely the KCL identity #177 found.
#
# `BSplineSolver` escapes this because its ports are a MIXED-POTENTIAL
# construct — the node potential is a Lagrange multiplier on a KCL row, i.e.
# the integration-by-parts boundary term held OUTSIDE the reaction integral.
# The sinusoidal family is deliberately field-based (Eqs 76-79 merge both
# potentials), so that term lives inside the kernel and cannot be separated.
#
# These tests pin the measurement, because the measurement is the only thing
# that stops the construction being re-proposed.
#
# SUPERSEDED, NOT DELETED (M5b formulation (b), last section of this file):
# M5's blanket solve refusal is gone — holding the node's lumped charge
# OUTSIDE the reaction integral turns this same port basis into one that
# reproduces `BSplineSolver` to 3e-5. Everything below still measures
# `_assemble_Z`, i.e. M5's reaction form verbatim, and every number in it
# still reproduces. What changed is only which matrix the SOLVES use
# (`_assemble_Z_ported`). Read this section as the record of why the naive
# construction is wrong, which is exactly what makes the correction legible.

EPS0 = 8.8541878188e-12


def _sg_port_system(s):
    """(geom, G, seg_view, drive columns) for a port-carrying Galerkin solve.

    Goes through the assembly directly: the public solve entry points refuse
    when junction ports are present, which is the point of this section."""
    geom = s._build_geometry()
    G, seg_view = s._assemble_Z(geom, s.k)
    return geom, G, seg_view, s._drive_columns(geom, seg_view, s.k)


def _sg_port_y(s):
    geom, G, seg_view, U = _sg_port_system(s)
    alphas = scipy.linalg.solve(G, U)
    return np.stack(
        [s._port_currents(alphas[:, j], geom, seg_view, U) for j in range(s.n_ports)],
        axis=1,
    )


def _sg_net_inflow(s, geom, seg_view, j_idx, n_basis):
    """Net current flowing INTO junction `j_idx` from outside, per basis.

    Each member contributes ±I(node) with the sign that makes 'into the node'
    positive: +I at a member whose natural end-2 is at the node, −I at one
    whose end-1 is."""
    inflow = np.zeros(n_basis, dtype=np.complex128)
    for m, sgn in s._junction_members(geom, j_idx):
        h = float(geom["seg_h"][m])
        a, b = seg_view["starts"][m], seg_view["starts"][m + 1]
        val = (
            seg_view["sigma"][a:b] * seg_view["A"][a:b]
            + seg_view["B"][a:b] * np.sin(s.k * 0.5 * h * sgn)
            + seg_view["sigma"][a:b] * seg_view["C"][a:b] * np.cos(s.k * 0.5 * h)
        )
        np.add.at(inflow, seg_view["jbasis"][a:b], sgn * val)
    return inflow


def test_sg_port_basis_is_exactly_the_vector_the_span_lacks():
    """#177's span claim, reconfirmed constructively rather than by inference.

    Every ordinary sinusoidal basis has net inflow 0 at a junction node as an
    algebraic identity (measured max 6.0e-15 here over all 42 of them), and
    the M5 port basis has exactly 1 — so it IS the missing direction, and the
    two port bases do not contaminate each other's node.
    """
    s = _port_pair_solver(0.04, 0.01, 20, cls=SinusoidalGalerkinSolver)
    geom = s._build_geometry()
    seg_view = s._basis_coefs(geom, s.k)
    n_segs = geom["n_segs"]
    n_basis = n_segs + 2
    for p in range(2):
        inflow = _sg_net_inflow(s, geom, seg_view, p, n_basis)
        assert np.abs(inflow[:n_segs]).max() < 1e-12, (
            "an ordinary basis has nonzero net inflow at a junction — #177's "
            f"KCL identity is broken: {np.abs(inflow[:n_segs]).max():.3e}"
        )
        assert abs(inflow[n_segs + p] - 1.0) < 1e-12
        assert abs(inflow[n_segs + 1 - p]) < 1e-12


def test_sg_port_rows_keep_the_galerkin_matrix_and_y_symmetric():
    """The port row is a genuine Galerkin row, not a bolted-on constraint:
    the (N+P, N+P) matrix stays at the G1 symmetry floor (8.3e-12 measured)
    and the port sub-block of Y — whose drive and readout are the same
    vector — is symmetric to 1.7e-14.

    So nothing that follows can be blamed on a broken fill.
    """
    s = _port_pair_solver(0.04, 0.01, 20, cls=SinusoidalGalerkinSolver)
    geom, G, _seg_view, _U = _sg_port_system(s)
    assert G.shape == (geom["n_segs"] + 2, geom["n_segs"] + 2)
    assert np.linalg.norm(G - G.T) / np.linalg.norm(G) < 1e-10
    Y = _sg_port_y(s)
    assert Y.shape == (3, 3)
    port_block = Y[1:, 1:]
    asym = np.linalg.norm(port_block - port_block.T) / np.linalg.norm(port_block)
    assert asym < 1e-10, f"port-block Y asymmetry {asym:.3e}"


@pytest.mark.parametrize("radius", [0.0002, 0.0005])
def test_sg_port_self_term_is_the_node_charge_priced_at_the_wire_radius(radius):
    """THE blocker, as a number.

    A unit-inflow port basis terminates its current at the node, so it leaves
    a point charge q = 1/(jω) there, and the field kernel prices that charge's
    self-energy at the thin-wire radius. The prediction is the reactance of a
    sphere of radius `a`:

        Z_pp ≈ 1/(jω · 4πε · a)

    Measured Z_pp/prediction: 0.965 at a=2e-4, 0.930 at a=5e-4 (n_half=20) —
    a model with no fitted constant, agreeing to <7 %.

    And the KCL-clean span cannot take it away: the Schur complement of the
    whole N-dimensional ordinary basis removes 0.8 % / 2.0 % of it. That is
    the same KCL identity again — no ordinary basis carries net charge at a
    junction, so none of them has anything to cancel with.
    """
    s = _port_pair_solver(
        0.04, 0.01, 20, cls=SinusoidalGalerkinSolver, wire_radius=radius
    )
    geom, G, _seg_view, _U = _sg_port_system(s)
    n = geom["n_segs"]
    predicted = 1.0 / (s.omega * 4.0 * np.pi * EPS0 * radius)
    ratio = G[n, n].imag / predicted
    assert 0.85 < ratio < 1.05, (
        f"Z_pp/(1/(w*4*pi*eps*a)) = {ratio:.3f}: the port self-term is no "
        "longer the node point charge at the wire radius"
    )
    schur = G[n:, n:] - G[n:, :n] @ np.linalg.solve(G[:n, :n], G[:n, n:])
    removed = 1.0 - schur[0, 0].imag / G[n, n].imag
    assert removed < 0.05, (
        f"the ordinary span now removes {removed:.1%} of the port self-term "
        "— re-check whether the obstruction still holds"
    )


def test_sg_port_self_term_is_set_by_the_radius_not_the_mesh():
    """The other half of the obstruction: it does not converge away.

    Quadrupling the mesh (n_half 10 → 40) moves Z_pp by 9 % (91713 → 83890 Ω
    at a=5e-4), while a *tenfold* change in the WIRE RADIUS moves it by the
    full decade. A discretization error would do the opposite.
    """
    zs = {}
    for n_half in (10, 20, 40):
        s = _port_pair_solver(0.04, 0.01, n_half, cls=SinusoidalGalerkinSolver)
        _geom, G, _sv, _U = _sg_port_system(s)
        zs[n_half] = G[-2, -2].imag
    mesh_swing = abs(zs[40] - zs[10]) / zs[10]
    assert mesh_swing < 0.25, f"mesh swing {mesh_swing:.1%} — recheck the story"

    s_fat = _port_pair_solver(
        0.04, 0.01, 20, cls=SinusoidalGalerkinSolver, wire_radius=0.005
    )
    _geom, G_fat, _sv, _U = _sg_port_system(s_fat)
    radius_swing = zs[20] / G_fat[-2, -2].imag
    assert radius_swing > 5.0, (
        f"a 10x radius change moved Z_pp by only {radius_swing:.1f}x — the "
        "self-term is no longer radius-set"
    )


@pytest.mark.parametrize("n_half", [10, 20, 40])
def test_sg_paired_ports_miss_the_bridged_reference_at_every_mesh(n_half):
    """G5's oracle, run and failed — recorded, not weakened.

    The B-spline ports pass this at 1.5 % after gap extrapolation. The
    sinusoidal-Galerkin ports land 2508× / 2434× / 2298× away at n_half =
    10 / 20 / 40 — the differential impedance reads ~-1.7e5 j (the two node
    self-capacitances in series) instead of ~70 − 15j, and refining the mesh
    4× improves it by 8 %.

    Both sides use the SAME solver, so this is not a basis comparison.
    """
    z_ref = _bridged_z(0.04, 0.01, n_half, cls=SinusoidalGalerkinSolver)
    Y = _sg_port_y(_port_pair_solver(0.04, 0.01, n_half, cls=SinusoidalGalerkinSolver))
    Zp = np.linalg.inv(Y[1:, 1:])
    z_port = complex(Zp[0, 0] - Zp[0, 1] - Zp[1, 0] + Zp[1, 1])
    rel = abs(z_port - z_ref) / abs(z_ref)
    assert rel > 100.0, (
        f"the sinusoidal-Galerkin junction port now lands within {rel:.3g} of "
        "the bridged reference — the M5 obstruction may have been lifted, "
        "which would be news; re-derive before relaxing this test"
    )
    assert abs(z_port.imag) > 1e4, z_port


def test_sg_junction_port_reaction_form_is_still_what_the_solves_reject():
    """M5's blanket solve refusal was SUPERSEDED by M5b formulation (b), and
    this test is what keeps the supersession honest.

    Everything above still measures `_assemble_Z` — M5's reaction-form
    construction, verbatim, still 2400× from its oracle. What changed is that
    the solves no longer use that matrix: they go through
    `_assemble_Z_ported`, which holds the node's lumped charge outside the
    reaction integral. So the two must differ, and differ by exactly the
    blocker: the port self-term drops from ~8.9e4 j to ~5.0e3 j (94 %), and
    that is the whole of the difference between a refused solve and a green
    one.
    """
    s = _port_pair_solver(0.04, 0.01, 20, cls=SinusoidalGalerkinSolver)
    geom = s._build_geometry()
    G_raw, _sv = s._assemble_Z(geom, s.k)
    G_solve, _sv2 = s._assemble_Z_ported(geom, s.k)
    n = geom["n_segs"]
    assert G_raw[n, n].imag > 5e4, G_raw[n, n]
    assert G_solve[n, n].imag < 1e4, G_solve[n, n]
    assert G_solve[n, n].imag / G_raw[n, n].imag < 0.1
    # the ordinary block is untouched: only the port rows/columns move
    np.testing.assert_array_equal(G_raw[:n, :n], G_solve[:n, :n])
    # and the correction is symmetric by construction, so the fill's own
    # reciprocity residual is exactly as good as it was
    raw = np.linalg.norm(G_raw - G_raw.T)
    cor = np.linalg.norm(G_solve - G_solve.T)
    assert abs(cor - raw) / raw < 1e-6, (raw, cor)


@pytest.mark.parametrize("cls", [SinusoidalSolver, SinusoidalGalerkinSolver])
def test_free_end_equals_one_entry_junction_sinusoidal(cls):
    """G5's third clause, and the one that IS green on this family.

    A 1-entry junction group emits no neighbour entries at all, so the member
    end keeps the basis's own free-end branch — the sinusoidal analogue of
    B-spline's "enforced through the Lagrange row instead". Both sinusoidal
    solvers reproduce the plain dipole BIT-EXACTLY (relative difference 0.0,
    not merely small), which is a stronger statement than the B-spline
    version of this test can make.
    """
    wl = 22.0
    hd = 0.962 * wl / 4
    dip = [np.array([(0.0, -hd, 0.0), (0.0, hd, 0.0)])]
    base = dict(wires=dip, n_per_edge_per_wire=[[41]], nsegs=41, wavelength=wl)
    z_free = complex(np.atleast_1d(cls(**base).compute_impedance()[0])[0])
    z_junc = complex(
        np.atleast_1d(
            cls(**base, junctions=[[(0, "start")], [(0, "end")]]).compute_impedance()[0]
        )[0]
    )
    assert z_junc == z_free


def test_sg_junction_port_validation_rules():
    """#172's validation rules hold on the Galerkin solver too — they are
    checked at construction, before the solve-time refusal, so a malformed
    port never reaches the interesting error."""
    wires, npe, junctions = _split_wires(0.04, 0.01, 8)
    common = dict(
        wires=wires, n_per_edge_per_wire=npe, feeds=[(0, 1.0, 1.0 + 0j)],
        junctions=junctions, wavelength=WAVELENGTH,
    )  # fmt: skip
    with pytest.raises(ValueError, match="out of range"):
        SinusoidalGalerkinSolver(**common, junction_ports=[7])
    with pytest.raises(ValueError, match="twice"):
        SinusoidalGalerkinSolver(**common, junction_ports=[0, 0])
    # plain ints mean voltage 0, and feeds=[] is legal with ports present
    s = SinusoidalGalerkinSolver(
        wires=wires, n_per_edge_per_wire=npe, feeds=[], junctions=junctions,
        junction_ports=[0, 1], wavelength=WAVELENGTH,
    )  # fmt: skip
    assert s.junction_ports == [(0, 0j), (1, 0j)]
    assert s.n_ports == 2


def test_sg_grounded_junction_port_rejected():
    """A junction node in the ground plane is connected to its own image
    instead of to its partners (#151), so it emits no neighbour entries and
    its voltage is pinned — declaring it a port is ambiguous, exactly as on
    BSplineSolver. Raised from the assembly, so it beats the M5 refusal."""
    h = 0.24 * WAVELENGTH
    s = SinusoidalGalerkinSolver(
        wires=[
            np.array([(0.0, 0.0, 0.0), (0.0, 0.0, h)]),
            np.array([(0.0, 0.0, 0.0), (0.01, 0.0, 0.01)]),
        ],
        n_per_edge_per_wire=[[21], [1]],
        feeds=[(0, h / 2, 1.0 + 0j)],
        junctions=[[(0, "start"), (1, "start")]],
        junction_ports=[0],
        wavelength=WAVELENGTH,
        ground_z=0.0,
    )
    with pytest.raises(ValueError, match="grounded"):
        s._assemble_Z(s._build_geometry(), s.k)


# ===========================================================================
# momwire#182 M5b — node ports: the SAME oracle, formulation (a)
# ===========================================================================
#
# M5's refutation is constructive: it forbids exactly ONE thing, a port basis
# that TERMINATES current at a node (depositing q = I/jω that the field kernel
# prices at the wire radius). Formulation (a) avoids it by never terminating
# anything — the port's two terminals are JOINED into one junction, so current
# flows THROUGH the node under the ordinary KCL-identical span, and the drive
# is a zero-width delta-gap EMF sitting exactly at the node:
#
#     b_i = -V·f_i(node),      I_port = -b·α / V
#
# The point-matched solver cannot do this and the reason is #177's own: a node
# source samples to an identically zero RHS at segment centres. Galerkin test
# functions have nonzero node values, so the same source has a well-defined
# excitation here. That makes the node port a capability of the TESTING, which
# is what the fourth basis × testing cell exists to expose.
#
# What it is NOT: a one-terminal net-inflow port. `PortAtEnd` in antennaknobs
# resolves every one of `wire.sterba_bl`'s 16 ports to a ONE-member junction
# at a dangling conductor end, with the return path at a different node
# (0.04 m away differentially, λ/2 away in common mode) — there is nothing to
# bipartition and the node genuinely must accept net current. Formulation (a)
# rejects that topology at construction rather than answering it wrongly. See
# the M5b PR body for the scoping and `test_sg_node_port_rejects_the_one_
# terminal_topology` below for the guard.

M5B_WHISKER = 0.01


def _m5b_bridged_z(delta, whisker, n_half, cls=SinusoidalGalerkinSolver, **kw):
    """Reference: a dipole split at its centre and RE-JOINED by a physical
    1-segment bridge wire of length `delta` carrying a classic delta-gap feed
    — the workaround NEC-2 itself requires, and the delta → 0 limit of the
    node port below.

    Optional cross whiskers hold two extra wire-ends at each tip so the node
    port's junction is K=4 rather than K=2. They point in OPPOSITE directions
    (∓x) because the delta → 0 limit has to stay a legal geometry: the
    original `_split_wires` whiskers both point +x, so joining the tips would
    place two identical overlapping wires on top of each other.
    """
    wires = [
        np.array([(0.0, -L_DIP / 2, 0.0), (0.0, -delta / 2, 0.0)]),
        np.array([(0.0, -delta / 2, 0.0), (0.0, delta / 2, 0.0)]),
        np.array([(0.0, delta / 2, 0.0), (0.0, L_DIP / 2, 0.0)]),
    ]
    npe = [[n_half], [1], [n_half]]
    junctions = [[(0, "end"), (1, "start")], [(1, "end"), (2, "start")]]
    if whisker:
        wires += [
            np.array([(0.0, -delta / 2, 0.0), (-whisker, -delta / 2, 0.0)]),
            np.array([(0.0, delta / 2, 0.0), (whisker, delta / 2, 0.0)]),
        ]
        npe += [[1], [1]]
        junctions[0].append((3, "start"))
        junctions[1].append((4, "start"))
    s = cls(
        wires=wires, n_per_edge_per_wire=npe,
        feeds=[(1, delta / 2, 1.0 + 0j)], junctions=junctions,
        wavelength=WAVELENGTH, **kw,
    )  # fmt: skip
    return complex(np.atleast_1d(s.compute_impedance()[0])[0])


def _m5b_node_port_solver(whisker, n_half, volts=1.0 + 0j, feed_v=0.0 + 0j, **kw):
    """The SAME structure at delta = 0: the two dipole halves meet at one
    junction and the port is an EMF across it.

    Carries a dummy gap feed because `SinusoidalSolver` requires at least one
    (`feeds=[]` is B-spline-only). At 0 V a delta gap is a plain wire — it
    touches the RHS and nothing else — which
    `test_sg_node_port_zero_volts_reproduces_the_plain_solve` pins.
    """
    wires = [
        np.array([(0.0, -L_DIP / 2, 0.0), (0.0, 0.0, 0.0)]),
        np.array([(0.0, 0.0, 0.0), (0.0, L_DIP / 2, 0.0)]),
    ]
    npe = [[n_half], [n_half]]
    junctions = [[(0, "end"), (1, "start")]]
    side_a = (0,)
    if whisker:
        wires += [
            np.array([(0.0, 0.0, 0.0), (-whisker, 0.0, 0.0)]),
            np.array([(0.0, 0.0, 0.0), (whisker, 0.0, 0.0)]),
        ]
        npe += [[1], [1]]
        junctions[0] += [(2, "start"), (3, "start")]
        side_a = (0, 2)
    return SinusoidalGalerkinSolver(
        wires=wires, n_per_edge_per_wire=npe,
        feeds=[(0, L_DIP / 4, feed_v)], junctions=junctions,
        node_ports=[(0, side_a, volts)], wavelength=WAVELENGTH, **kw,
    )  # fmt: skip


def _m5b_node_port_z(whisker, n_half, **kw):
    z, _a = _m5b_node_port_solver(whisker, n_half, **kw).compute_impedance()
    return complex(np.atleast_1d(z)[1])  # [dummy gap feed, node port]


def _m5b_reference_family(whisker, n_half, deltas=(0.08, 0.04, 0.02)):
    """The bridged reference extrapolated to delta → 0, as a FAMILY.

    M3's constraint applies: there is no single "converged reference" here
    either, so the verdict is the one that survives every defensible choice —
    linear-in-delta Richardson from each consecutive pair of a 4×-spanning
    delta ladder.
    """
    vals = [_m5b_bridged_z(d, whisker, n_half) for d in deltas]
    return [2 * vals[i + 1] - vals[i] for i in range(len(deltas) - 1)]


# --- G5b clause 1: the `_bridged_z` oracle -------------------------------

# Worst-case |Z_node − Z_ref| / |Z_ref| over the reference family, measured
# 2026-07-30. Pinned slightly ABOVE each measurement (these are error bounds,
# so the floor goes on the other side from a payoff ratio).
M5B_ORACLE_MEASURED = {
    (0.0, 10): 0.005572, (0.0, 20): 0.003179, (0.0, 40): 0.001807,
    (0.01, 10): 0.005217, (0.01, 20): 0.002766, (0.01, 40): 0.001341,
}  # fmt: skip
M5B_ORACLE_PINNED = {key: 1.15 * v for key, v in M5B_ORACLE_MEASURED.items()}


@pytest.mark.parametrize("whisker", [0.0, M5B_WHISKER])
@pytest.mark.parametrize("n_half", [10, 20, 40])
def test_sg_node_port_matches_the_bridged_reference(whisker, n_half):
    """G5b clause 1 — GREEN, at 3.6× to 11× inside the gate.

    The port drive is a zero-width gap at the joined node; the reference is
    the same structure with a real bridge wire of length delta carrying a
    real delta-gap feed, extrapolated to delta → 0. Worst case over the
    reference family:

        K=2 (no whiskers): 0.557 % / 0.318 % / 0.181 % at n_half 10/20/40
        K=4 (whiskers):    0.522 % / 0.277 % / 0.134 %

    against the 1.5 % the B-spline junction ports passed at. Both sides are
    built by the SAME solver, so this is not a basis comparison; and the
    difference DECAYS with mesh, which is what says the residue is
    discretization rather than a modelling gap.
    """
    z = _m5b_node_port_z(whisker, n_half)
    refs = _m5b_reference_family(whisker, n_half)
    worst = max(abs(z - r) / abs(r) for r in refs)
    assert worst < 0.015, f"G5b clause 1 RED: {worst:.4%} from the reference"
    assert worst < M5B_ORACLE_PINNED[(whisker, n_half)], (
        f"node-port oracle drifted: {worst:.4%} vs "
        f"{M5B_ORACLE_MEASURED[(whisker, n_half)]:.4%} measured at M5b"
    )


@pytest.mark.parametrize("whisker", [0.0, M5B_WHISKER])
def test_sg_node_port_oracle_improves_with_mesh(whisker):
    """The residue in the clause above is discretization, not a modelling
    gap: it falls monotonically (~0.57× per mesh doubling, i.e. a little
    slower than O(h)) rather than settling on a floor. A modelling gap — the
    shape M5's node charge had — would be mesh-independent."""
    errs = []
    for n_half in (10, 20, 40):
        z = _m5b_node_port_z(whisker, n_half)
        refs = _m5b_reference_family(whisker, n_half)
        errs.append(max(abs(z - r) / abs(r) for r in refs))
    assert errs[1] < errs[0] and errs[2] < errs[1], errs
    assert errs[2] / errs[0] < 0.45, errs


@pytest.mark.parametrize("whisker", [0.0, M5B_WHISKER])
def test_sg_node_port_reference_family_is_well_posed(whisker):
    """The verdict must not depend on which extrapolation you pick.

    Richardson from (0.08, 0.04) and from (0.04, 0.02) agree to 0.013–0.016 %
    (K=2) / 0.039–0.041 % (K=4) — 10× to 100× below the measured difference
    they are the reference for, and ~40× to 100× below the 1.5 % gate. So the
    delta → 0 idealization the node port makes is absorbed by the
    extrapolation, exactly as M3's constraint requires it to be checked
    rather than assumed.
    """
    for n_half in (10, 20, 40):
        refs = _m5b_reference_family(whisker, n_half)
        spread = abs(refs[1] - refs[0]) / abs(refs[1])
        assert spread < 0.001, f"n_half={n_half}: reference spread {spread:.4%}"


# --- the structural claims formulation (a) rests on ----------------------


def test_sg_node_port_cuts_a_kcl_identical_span_and_deposits_no_charge():
    """The mechanism, as two numbers.

    M5 forbids a basis that terminates current at a node. A node port
    terminates nothing: the drive vector is a CUT of the ordinary span, whose
    net inflow summed over ALL the junction's members is the identically-zero
    KCL residual (#177's identity, 4.0e-13 relative here) — while the cut
    itself over half the members is O(1). So there is no node charge to price
    at the wire radius, and M5's Z_pp ≈ 1/(jω·4πε·a) has nothing to attach to.

    Also asserted: no basis column is added at all (α stays length N), which
    is the concrete difference from the M5 construction.
    """
    s = _m5b_node_port_solver(M5B_WHISKER, 20)
    geom = s._build_geometry()
    seg_view = s._basis_coefs(geom, s.k)
    cut = s._node_cut_vectors(geom, seg_view, s.k)[:, 0]
    s_all = _m5b_node_port_solver(M5B_WHISKER, 20)
    s_all.node_ports = [(0, (0, 1, 2), 1.0 + 0j)]  # 3 of 4 members
    partial = s_all._node_cut_vectors(geom, seg_view, s.k)[:, 0]
    # cut(all four members) = cut(0,2) + cut(1,3); build it as cut + complement
    s_c = _m5b_node_port_solver(M5B_WHISKER, 20)
    s_c.node_ports = [(0, (1, 3), 1.0 + 0j)]
    kcl = cut + s_c._node_cut_vectors(geom, seg_view, s.k)[:, 0]
    scale = np.abs(cut).max()
    assert scale > 1e-3, scale
    assert np.abs(kcl).max() / scale < 1e-10, (
        f"KCL residual {np.abs(kcl).max() / scale:.2e} — the node port would "
        "be depositing charge, which is exactly what M5 refuted"
    )
    # An UNEVEN split is a port too, and its cut is the exact negative of the
    # complement's — the KCL identity again, now without the 2-2 symmetry that
    # could hide a sign bug. (Its magnitude is 1.1e-4, three decades below the
    # 2-2 cut, because the complement here is a lone 0.01-long free-ended
    # whisker: a nearly-open port, which is the right answer for one.)
    s_w = _m5b_node_port_solver(M5B_WHISKER, 20)
    s_w.node_ports = [(0, (3,), 1.0 + 0j)]
    lone = s_w._node_cut_vectors(geom, seg_view, s.k)[:, 0]
    assert np.abs(lone).max() > 1e-6
    assert np.abs(partial + lone).max() / np.abs(lone).max() < 1e-10
    _z, alpha = s.compute_impedance()
    assert alpha.shape == (geom["n_segs"],)


@pytest.mark.parametrize("whisker", [0.0, M5B_WHISKER])
def test_sg_node_port_is_invariant_under_swapping_the_gap_sides(whisker):
    """Which side of the gap is called "+" is a convention, and the port
    impedance may not know about it: swapping `side_a` for its complement
    negates the cut vector, and Z is quadratic in it. Agreement to 2.5e-16 —
    a structural check that the drive and readout use the SAME vector."""
    s = _m5b_node_port_solver(whisker, 20)
    j_idx, side_a, volts = s.node_ports[0]
    rest = tuple(m for m in range(len(s.junctions[j_idx])) if m not in side_a)
    z_a = complex(np.atleast_1d(s.compute_impedance()[0])[1])
    s2 = _m5b_node_port_solver(whisker, 20)
    s2.node_ports = [(j_idx, rest, volts)]
    z_b = complex(np.atleast_1d(s2.compute_impedance()[0])[1])
    assert abs(z_a - z_b) / abs(z_a) < 1e-13


# --- G5b clause 2: mixed gap + port Y symmetry ---------------------------


def _m5b_mixed_y_solver(feed_readout):
    """A dipole cut into three, gap-fed on the first third and node-ported at
    BOTH interior nodes — a genuine 3-port with a gap feed in it."""
    ys = [-L_DIP / 2, -L_DIP / 6, L_DIP / 6, L_DIP / 2]
    wires = [np.array([(0.0, ys[i], 0.0), (0.0, ys[i + 1], 0.0)]) for i in range(3)]
    return SinusoidalGalerkinSolver(
        wires=wires, n_per_edge_per_wire=[[15], [15], [15]],
        feeds=[(0, L_DIP / 8, 1.0 + 0j)],
        junctions=[[(0, "end"), (1, "start")], [(1, "end"), (2, "start")]],
        node_ports=[(0, (0,), 0.5 + 0j), (1, (0,), -0.5 + 0j)],
        wavelength=WAVELENGTH, feed_readout=feed_readout,
    )  # fmt: skip


@pytest.mark.parametrize("feed_readout", ["centre", "variational"])
def test_sg_node_port_y_block_is_symmetric_in_both_readouts(feed_readout):
    """G5b clause 2, the half that is unconditionally green.

    A node port's readout is its drive vector, always — the through-current
    functional has no centre-vs-average ambiguity for the delta-gap feed to
    contaminate, so it costs none of the M3 payoff. The node-port block of Y
    is symmetric to 1.3e-13 under BOTH `feed_readout` settings.
    """
    Y = _m5b_mixed_y_solver(feed_readout).compute_y_matrix()
    assert Y.shape == (3, 3)
    block = Y[1:, 1:]
    asym = np.linalg.norm(block - block.T) / np.linalg.norm(block)
    assert asym < 1e-11, f"node-port Y block asymmetry {asym:.3e}"


def test_sg_node_port_full_mixed_y_symmetry_inherits_the_m5_amendment():
    """G5b clause 2, the half that carries M5's amendment unchanged.

    The FULL Y also has the gap feed's rows in it, and the default
    `feed_readout="centre"` is not the Galerkin drive's dual — so the mixed Y
    is symmetric only to the O(h) gap between the two feed functionals
    (1.06e-4 here), exactly as M5 recorded for the delta-gap-only Y.
    `feed_readout="variational"` restores it to 7.3e-13, i.e. inside the 1e-10
    clause, at the M5-measured cost to the M3 payoff.

    The amendment is the feed's, not the node port's: the block above is
    machine-symmetric either way.
    """
    asym = {}
    for ro in ("centre", "variational"):
        Y = _m5b_mixed_y_solver(ro).compute_y_matrix()
        asym[ro] = np.linalg.norm(Y - Y.T) / np.linalg.norm(Y)
    assert asym["variational"] < 1e-10, asym
    assert 1e-5 < asym["centre"] < 1e-3, asym
    assert asym["variational"] < 1e-6 * asym["centre"], asym


# --- G5b clause 3: the degenerate case reproduces the plain solve --------


def test_sg_node_port_zero_volts_reproduces_the_plain_solve():
    """G5b clause 3 for the new construction — BIT-exact, not merely close.

    A node port at 0 V is a short across the node, i.e. an ordinary junction.
    It adds no basis column and its drive column is scaled by V, so the whole
    system is untouched: a gap-fed solve with a 0 V node port declared must
    return the identical float to one without it. That is also what makes the
    dummy 0 V gap feed in `_m5b_node_port_solver` free.
    """
    s = _m5b_node_port_solver(0.0, 20, volts=0j, feed_v=1.0 + 0j)
    z_with = complex(np.atleast_1d(s.compute_impedance()[0])[0])
    plain = SinusoidalGalerkinSolver(
        wires=[
            np.array([(0.0, -L_DIP / 2, 0.0), (0.0, 0.0, 0.0)]),
            np.array([(0.0, 0.0, 0.0), (0.0, L_DIP / 2, 0.0)]),
        ],
        n_per_edge_per_wire=[[20], [20]],
        feeds=[(0, L_DIP / 4, 1.0 + 0j)],
        junctions=[[(0, "end"), (1, "start")]],
        wavelength=WAVELENGTH,
    )
    z_plain = complex(np.atleast_1d(plain.compute_impedance()[0])[0])
    assert z_with == z_plain


# --- the rest of #172's contract, on node ports --------------------------


def test_sg_node_port_readout_consistent_with_y_and_swept():
    """`compute_impedance`, `compute_y_matrix` and both swept paths share one
    drive/readout pair (M5's prerequisite fix), so I = Y·V must reproduce the
    impedance readout and the batched sweep must reproduce the per-k solves.
    The node port's drive column is k-DEPENDENT (the cut vector evaluates the
    basis shapes, which move with k), so the sweeps rebuild it per step."""
    s = _m5b_mixed_y_solver("centre")
    V = s._port_voltages()
    z_per = np.atleast_1d(s.compute_impedance()[0])
    Y = _m5b_mixed_y_solver("centre").compute_y_matrix()
    np.testing.assert_allclose((V / z_per)[1:], (Y @ V)[1:], rtol=1e-9)

    ks = np.array([0.9 * s.k, s.k, 1.1 * s.k])
    Y_swept = _m5b_mixed_y_solver("centre").compute_y_matrix_swept(ks)
    z_swept = _m5b_mixed_y_solver("centre").compute_impedance_swept(ks)
    for i, kk in enumerate(ks):
        s2 = _m5b_mixed_y_solver("centre")
        s2._set_k(float(kk))
        np.testing.assert_allclose(Y_swept[i], s2.compute_y_matrix(), rtol=1e-8)
        s3 = _m5b_mixed_y_solver("centre")
        s3._set_k(float(kk))
        np.testing.assert_allclose(
            z_swept[i], np.atleast_1d(s3.compute_impedance()[0]), rtol=1e-8
        )


def test_sg_node_port_rejects_the_one_terminal_topology():
    """The scoping result, as a guard.

    antennaknobs' `PortAtEnd` resolves to a ONE-member junction at a lone
    conductor end and needs genuine net inflow there — the current leaves via
    an unmodelled lead whose return is at a different node. A node port is an
    EMF ACROSS a node and has nothing to bipartition with one member, so it
    must refuse rather than silently model an open circuit (which is exactly
    the measured failure mode of every gap-family substitute in
    antennaknobs#608). That topology is still `junction_ports`' job, and
    `junction_ports` is still refused here.
    """
    wires = [np.array([(0.0, -L_DIP / 2, 0.0), (0.0, 0.0, 0.0)])]
    with pytest.raises(ValueError, match="at least two wire-ends"):
        SinusoidalGalerkinSolver(
            wires=wires, n_per_edge_per_wire=[[8]],
            feeds=[(0, L_DIP / 4, 1.0 + 0j)], junctions=[[(0, "end")]],
            node_ports=[(0, (0,), 1.0 + 0j)], wavelength=WAVELENGTH,
        )  # fmt: skip


def test_sg_node_port_needs_galerkin_testing():
    """#177's observation, read forwards: a source at a NODE point-samples to
    an identically zero RHS, because a node is never a segment centre. So the
    point-matched solver has no node ports and does not accept the keyword —
    while the Galerkin drive column for the same source is O(1) (asserted in
    `test_sg_node_port_cuts_a_kcl_identical_span_and_deposits_no_charge`).
    This is the one capability difference between the two sinusoidal cells
    that is a TESTING property rather than a basis property."""
    wires = [
        np.array([(0.0, -L_DIP / 2, 0.0), (0.0, 0.0, 0.0)]),
        np.array([(0.0, 0.0, 0.0), (0.0, L_DIP / 2, 0.0)]),
    ]
    with pytest.raises(TypeError):
        SinusoidalSolver(
            wires=wires, n_per_edge_per_wire=[[8], [8]],
            feeds=[(0, L_DIP / 4, 1.0 + 0j)],
            junctions=[[(0, "end"), (1, "start")]],
            node_ports=[(0, (0,), 1.0 + 0j)], wavelength=WAVELENGTH,
        )  # fmt: skip


def test_sg_node_port_validation_rules():
    """Malformed node ports are refused at construction, before any solve."""
    wires = [
        np.array([(0.0, -L_DIP / 2, 0.0), (0.0, 0.0, 0.0)]),
        np.array([(0.0, 0.0, 0.0), (0.0, L_DIP / 2, 0.0)]),
    ]
    common = dict(
        wires=wires, n_per_edge_per_wire=[[8], [8]],
        feeds=[(0, L_DIP / 4, 1.0 + 0j)],
        junctions=[[(0, "end"), (1, "start")]], wavelength=WAVELENGTH,
    )  # fmt: skip
    for ports, msg in (
        ([(3, (0,))], "out of range"),
        ([(0, (0,)), (0, (1,))], "listed twice"),
        ([(0, ())], "PROPER subset"),
        ([(0, (0, 1))], "PROPER subset"),
        ([(0, (0, 0))], "repeated member"),
        ([(0, (5,))], "member index out of range"),
        ([0], "must be"),
    ):
        with pytest.raises(ValueError, match=msg):
            SinusoidalGalerkinSolver(**common, node_ports=ports)
    with pytest.raises(ValueError, match="both a junction_port and a node_port"):
        SinusoidalGalerkinSolver(
            **common, junction_ports=[0], node_ports=[(0, (0,))]
        )  # fmt: skip
    # the 2-tuple form means voltage 0
    s = SinusoidalGalerkinSolver(**common, node_ports=[(0, (0,))])
    assert s.node_ports == [(0, (0,), 0j)]
    assert s.n_ports == 2


def test_sg_grounded_node_port_rejected():
    """A junction node lying in the ground plane carries current into its own
    image (#151) rather than closing on its partners, so there is no
    through-current for an EMF to drive — refused from the assembly, for the
    same reason a grounded junction port is."""
    h = 0.24 * WAVELENGTH
    s = SinusoidalGalerkinSolver(
        wires=[
            np.array([(0.0, 0.0, 0.0), (0.0, 0.0, h)]),
            np.array([(0.0, 0.0, 0.0), (0.01, 0.0, 0.01)]),
        ],
        n_per_edge_per_wire=[[21], [1]],
        feeds=[(0, h / 2, 1.0 + 0j)],
        junctions=[[(0, "start"), (1, "start")]],
        node_ports=[(0, (0,), 1.0 + 0j)],
        wavelength=WAVELENGTH,
        ground_z=0.0,
    )
    with pytest.raises(ValueError, match="grounded and a node port"):
        s.compute_impedance()


# ===========================================================================
# momwire#182 M5b — formulation (b): the mixed-potential port row
# ===========================================================================
#
# Formulation (a) above is a two-terminal object and cannot serve the payoff:
# antennaknobs' `PortAtEnd` needs a ONE-terminal net-inflow port at a
# one-member junction. That is exactly the construction M5 refuted — so M5's
# mechanism had to be dissolved, not avoided.
#
# It dissolves on one observation, measured rather than argued:
# `BSplineSolver`'s own one-terminal port impedance is −1.87 − 35.05j where
# the node self-capacitance 1/(w.4pi.eps.a) would be 9.5e4. It carries NONE of
# it. That is the physical content of its Lagrange-multiplier port: the
# current reaching the node leaves through an ideal UNMODELLED lead, so
# nothing accumulates there and there is no node charge to price. M5's port
# basis, by contrast, terminates its current in vacuum.
#
# So formulation (b) redefines the port basis's CHARGE to be its line charge
# only and removes the lumped node charge from the source, symmetrically:
#
#     G'[i,p] = G'[p,i] = G[i,p] - D[i,p]
#     G'[p,q]           = G[p,q] - D[p,q] - D[q,p] + S[p,q]
#
# D is every basis tested against the node charge's field (the same test
# quadrature, graded); S is the lumped-lumped term put back after the double
# subtraction. Both constants come from the solver's own kernel — the
# point-charge field IS the Eqs 78-79 endpoint term — so nothing is fitted.
#
# `_assemble_Z` still returns M5's reaction-form matrix and the M5 section
# above still measures it. Only `_assemble_Z_ported`, which the solves use,
# is new.


def _sg_zdiff_b(delta, whisker, n_half, **kw):
    """Differential impedance across the two tip ports, through the PUBLIC
    API — this is the formulation-(b) analogue of `_port_pair_zdiff`."""
    Y = _port_pair_solver(
        delta, whisker, n_half, cls=SinusoidalGalerkinSolver, **kw
    ).compute_y_matrix()
    Zp = np.linalg.inv(Y[1:, 1:])
    return complex(Zp[0, 0] - Zp[0, 1] - Zp[1, 0] + Zp[1, 1])


def _sg_oracle_extrap(n_half, deltas=(0.04, 0.02), whisker=0.01):
    """The B-spline oracle's own procedure, verbatim: the port-minus-bridge
    difference is the physical bridge metal, linear in the split gap, so
    Richardson-extrapolate it to delta -> 0 and normalize by |Z_ref|."""
    diffs = {}
    for delta in deltas:
        z_ref = _bridged_z(delta, whisker, n_half, cls=SinusoidalGalerkinSolver)
        diffs[delta] = (_sg_zdiff_b(delta, whisker, n_half) - z_ref, z_ref)
    ext = 2 * diffs[deltas[1]][0] - diffs[deltas[0]][0]
    return abs(ext) / abs(diffs[deltas[1]][1])


@pytest.mark.parametrize("n_half", [10, 20, 40])
def test_sg_junction_port_meets_the_bridged_oracle(n_half):
    """G5b clause 1 for formulation (b) — GREEN, at 1.429 % against 1.5 %.

    Same oracle, same procedure, same delta pair (0.04, 0.02) the B-spline
    ports were accepted on. Measured 1.4290 / 1.4289 / 1.4280 % at n_half
    10 / 20 / 40 — mesh-INDEPENDENT, which is the tell that what is left is
    the oracle's own extrapolation residue rather than anything about the
    port (see the next test).

    The margin is thin and honestly so: B-spline's own number on this oracle
    is 0.73 / 1.08 / 1.28 / 1.39 % at n_half 10 / 20 / 40 / 80 — it CLIMBS
    with mesh toward the same place. Both formulations are converging on the
    same ~1.4 %, so 1.5 % is a statement about the oracle's delta ladder, not
    a discriminator between them.
    """
    rel = _sg_oracle_extrap(n_half)
    assert rel < 0.015, f"G5b clause 1 RED for formulation (b): {rel:.4%}"
    assert 0.012 < rel < 0.0146, f"drifted from the M5b-measured 1.429 %: {rel:.4%}"


def test_sg_junction_port_oracle_residue_is_the_extrapolation_not_the_port():
    """Why 1.43 % is not the port's error.

    The oracle extrapolates the port-minus-bridge difference LINEARLY in the
    split gap. Halve the ladder and the residue halves — 2.876 % from
    (0.08, 0.04), 1.429 % from (0.04, 0.02), 0.700 % from (0.02, 0.01) — so
    the difference is not linear in delta and the two-point Richardson leaves
    an O(delta) tail. `BSplineSolver` does exactly the same thing on the same
    geometry (2.542 / 1.075 / 0.370 %), which is what identifies the tail as
    the oracle's rather than either formulation's.
    """
    rels = [
        _sg_oracle_extrap(20, deltas=d)
        for d in ((0.08, 0.04), (0.04, 0.02), (0.02, 0.01))
    ]
    assert rels[1] < 0.55 * rels[0], rels
    assert rels[2] < 0.55 * rels[1], rels


@pytest.mark.parametrize("delta", [0.04, 0.02])
def test_sg_junction_port_reproduces_the_bspline_port(delta):
    """The strongest evidence formulation (b) is right, and it is not the
    oracle.

    `BSplineSolver`'s junction port is a Lagrange multiplier on a KCL
    constraint row in a spline basis; this one is a node-charge term held
    outside a reaction integral in a sinusoidal basis. Nothing is shared —
    not the basis, not the testing, not the port algebra. Their full 2×2 port
    Y matrices agree ENTRYWISE to 3.4e-5 / 4.2e-5, and the individual
    one-terminal Z11 to 4e-5 (−1.8697 − 35.0471j vs −1.8704 − 35.0469j) —
    including the gauge-dependent self terms, not merely the differential
    combination.
    """
    Yb = _port_pair_solver(delta, 0.01, 20).compute_y_matrix()[1:, 1:]
    Yg = _port_pair_solver(
        delta, 0.01, 20, cls=SinusoidalGalerkinSolver
    ).compute_y_matrix()[1:, 1:]
    rel = np.abs(Yg - Yb) / np.abs(Yb)
    assert rel.max() < 2e-4, rel
    assert abs(np.linalg.inv(Yg)[0, 0] - np.linalg.inv(Yb)[0, 0]) < 5e-3


def test_sg_junction_port_serves_the_one_member_portatend_topology():
    """The payoff clause: the topology `PortAtEnd` actually uses.

    Every one of `wire.sterba_bl`'s 16 ports resolves to a ONE-member
    junction at a lone conductor end — genuine net inflow, no through-current
    and nothing to bipartition, so formulation (a) refuses it by design. This
    is what formulation (b) buys: on a split dipole driven through two such
    one-member ports, the sinusoidal-Galerkin Y matches `BSplineSolver`'s
    entrywise to 3.9e-6.
    """
    wires = [
        np.array([(0.0, -L_DIP / 2, 0.0), (0.0, -0.02, 0.0)]),
        np.array([(0.0, 0.02, 0.0), (0.0, L_DIP / 2, 0.0)]),
    ]
    common = dict(
        wires=wires, n_per_edge_per_wire=[[20], [20]],
        junctions=[[(0, "end")], [(1, "start")]],
        junction_ports=[0, 1], wavelength=WAVELENGTH,
    )  # fmt: skip
    Yb = BSplineSolver(feeds=[], **common).compute_y_matrix()
    Yg = SinusoidalGalerkinSolver(
        feeds=[(0, L_DIP / 4, 0j)], **common
    ).compute_y_matrix()[1:, 1:]
    rel = np.abs(Yg - Yb) / np.abs(Yb)
    assert rel.max() < 5e-5, rel


@pytest.mark.parametrize("radius", [0.0002, 0.0005, 0.002])
def test_sg_junction_port_correction_removes_the_1_over_a_blocker(radius):
    """The M5 blocker law, and its removal, on the same matrix entry.

    `_assemble_Z` still puts the node's self-capacitance in the port diagonal
    at 0.965 / 0.930 / 0.824 × 1/(w.4pi.eps.a) — M5's measurement, unchanged
    and still pinned above. `_assemble_Z_ported` takes it out: the same entry
    reads 5897 / 4991 / 3617 j across a decade of radius, a 1.63× swing where
    the raw one swings 11.7×. The law is gone, not merely reduced.
    """
    s = _port_pair_solver(
        0.04, 0.01, 20, cls=SinusoidalGalerkinSolver, wire_radius=radius
    )
    geom = s._build_geometry()
    n = geom["n_segs"]
    raw = s._assemble_Z(geom, s.k)[0][n, n].imag
    cor = s._assemble_Z_ported(geom, s.k)[0][n, n].imag
    predicted = 1.0 / (s.omega * 4.0 * np.pi * EPS0 * radius)
    assert 0.75 < raw / predicted < 1.05, raw / predicted
    assert cor / raw < 0.2, (raw, cor)
    assert 2e3 < cor < 8e3, cor


def test_sg_junction_port_pair_block_regularization_is_load_bearing():
    """`_node_charge_pair_block` regularizes the node separation as
    sqrt(d² + a²), not d, and that is not cosmetic.

    The columns are subtracted twice, so the lumped-lumped term must go back
    in at exactly the separation the columns' own integration by parts
    produced — which is the regularized one, because the point-charge field
    is -grad of a potential in the regularized R. Use the bare d instead and
    a residue against `BSplineSolver` appears that runs as a²/d³: it grows
    8× when the gap halves and 100× over a decade of radius. Reproduced here
    by patching the block, so the choice cannot be silently undone.
    """
    kw = dict(cls=SinusoidalGalerkinSolver, wire_radius=0.002)
    good = _sg_zdiff_b(0.04, 0.01, 20, wire_radius=0.002)
    s_bad = _port_pair_solver(0.04, 0.01, 20, **kw)
    orig = s_bad._node_charge_pair_block

    def unregularized(geom, k):
        nodes = np.array(
            [s_bad._junction_node_position(geom, j) for j, _v in s_bad.junction_ports]
        )
        d = np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=-1)
        np.fill_diagonal(d, float(s_bad._uniform_radius))
        return 1j * s_bad.eta * np.exp(-1j * k * d) / (4.0 * np.pi * k * d)

    s_bad._node_charge_pair_block = unregularized
    Y = s_bad.compute_y_matrix()
    Zp = np.linalg.inv(Y[1:, 1:])
    bad = complex(Zp[0, 0] - Zp[0, 1] - Zp[1, 0] + Zp[1, 1])
    assert orig is not None
    # B-spline reference at the SAME fat radius, where the a²/d³ residue bites
    Yb = _port_pair_solver(0.04, 0.01, 20, wire_radius=0.002).compute_y_matrix()
    Zb = np.linalg.inv(Yb[1:, 1:])
    ref = complex(Zb[0, 0] - Zb[0, 1] - Zb[1, 0] + Zb[1, 1])
    assert abs(good - ref) / abs(ref) < 1e-3, (good, ref)
    assert abs(bad - ref) / abs(ref) > 0.02, (bad, ref)


def test_sg_junction_port_node_charge_quadrature_is_converged():
    """`n_qp_node` is a converged setting, not a tuned one: the port
    impedance moves 2.8e-5 from 8 panels-per-end to 12, 4.0e-9 from 12 to 16,
    and 1e-10 beyond — so the default 16 sits two decades inside the floor,
    and the answer is a property of the formulation rather than of the rule.
    """
    zs = {q: _sg_zdiff_b(0.04, 0.01, 20, n_qp_node=q) for q in (12, 16, 24)}
    assert abs(zs[16] - zs[12]) / abs(zs[16]) < 1e-7
    assert abs(zs[24] - zs[16]) / abs(zs[16]) < 1e-9


@pytest.mark.parametrize("feed_readout", ["centre", "variational"])
def test_sg_junction_port_y_block_symmetric(feed_readout):
    """G5b clause 2 for formulation (b), the half that is green.

    Drive and readout are still the same vector and the correction is
    symmetric by construction, so the junction-port block of Y is symmetric
    to 4.2e-12 under either readout — inside the 1e-10 clause.
    """
    Y = _port_pair_solver(
        0.04, 0.01, 20, cls=SinusoidalGalerkinSolver, feed_readout=feed_readout
    ).compute_y_matrix()
    block = Y[1:, 1:]
    asym = np.linalg.norm(block - block.T) / np.linalg.norm(block)
    assert asym < 1e-10, f"port block asymmetry {asym:.3e}"


def test_sg_junction_port_full_y_symmetry_is_the_fills_own_floor():
    """G5b clause 2 for formulation (b), the half that carries an AMENDMENT
    M5 did not need — stated with its attribution measured.

    The full mixed Y is symmetric to 3.7e-8 under `feed_readout="variational"`
    (1.4e-4 under the default `"centre"`, which is M5's gap-feed amendment
    unchanged). 3.7e-8 misses the clause's 1e-10, and the reason is not the
    port: symmetrising the assembled matrix by hand drops the Y asymmetry to
    1.4e-16, so ALL of it is the fill's own reciprocity residual
    (‖G−Gᵀ‖/‖G‖ = 8.3e-12, M1/M2's floor) amplified through the port solve.

    M5 did not see this because its port diagonal was the 8.9e4 j node
    self-capacitance, which made the port solve trivially well conditioned
    while being physically wrong. Removing it removes that masking:
    cond(G) falls 9.7e9 -> 5.9e8 and the fill's honest error becomes visible.
    Refining the SOURCE quadrature helps only slowly (3.7e-8 -> 2.5e-8 at 4×
    `n_qp_const`), which is what says it is a fill-accuracy statement rather
    than a port defect. Recorded, not papered over by symmetrising the fill.
    """
    s = _port_pair_solver(
        0.04, 0.01, 20, cls=SinusoidalGalerkinSolver, feed_readout="variational"
    )
    geom = s._build_geometry()
    G, seg_view = s._assemble_Z_ported(geom, s.k)
    U = s._drive_columns(geom, seg_view, s.k)

    def y_from(M):
        alphas = scipy.linalg.solve(M, U)
        return np.stack(
            [
                s._port_currents(alphas[:, j], geom, seg_view, U)
                for j in range(s.n_ports)
            ],
            axis=1,
        )

    def asym(Y):
        return np.linalg.norm(Y - Y.T) / np.linalg.norm(Y)

    a_built = asym(y_from(G))
    a_sym = asym(y_from(0.5 * (G + G.T)))
    assert 1e-9 < a_built < 1e-6, a_built
    assert a_sym < 1e-13, a_sym
    assert a_sym < 1e-6 * a_built, (a_built, a_sym)


def test_sg_junction_port_impedance_and_sweeps_agree():
    """One drive/readout pair across all four entry points, with the ported
    assembly in the loop."""

    def make():
        return _port_pair_solver(
            0.04, 0.01, 12, cls=SinusoidalGalerkinSolver, volts=(0.5 + 0j, -0.5 + 0j)
        )

    s = make()
    V = s._port_voltages()
    z_per = np.atleast_1d(s.compute_impedance()[0])
    Y = make().compute_y_matrix()
    # port 0 is the shorted dummy gap feed, whose z entry is 0/I
    np.testing.assert_allclose(V[1:] / z_per[1:], (Y @ V)[1:], rtol=1e-9)

    ks = np.array([0.9 * s.k, s.k, 1.1 * s.k])
    Y_swept = make().compute_y_matrix_swept(ks)
    z_swept = make().compute_impedance_swept(ks)
    for i, kk in enumerate(ks):
        s2 = make()
        s2._set_k(float(kk))
        np.testing.assert_allclose(Y_swept[i], s2.compute_y_matrix(), rtol=1e-8)
        s3 = make()
        s3._set_k(float(kk))
        np.testing.assert_allclose(
            z_swept[i], np.atleast_1d(s3.compute_impedance()[0]), rtol=1e-8
        )


def test_sg_junction_port_finite_ground_and_mixed_radius_still_refuse():
    """What formulation (b) does NOT cover, refused rather than approximated.

    #191 narrowed this to the FINITE grounds. The PEC image of the removed
    lumped charge is a point charge at the mirrored node, so the same
    correction takes it out (section below) and `ground_z` alone now solves.
    The reflection-coefficient and Sommerfeld images are not point charges —
    the reflection is angle-dependent, and Sommerfeld's is not an image at
    all — so the removed term has no closed mirror there and part of the M5
    blocker would survive. Under mixed radii the kernel is not reciprocal at
    all (M2) and the regularization radius at a node whose members disagree
    about `a` is ambiguous. Both raise instead of returning a plausible
    number.
    """
    wires, npe, junctions = _split_wires(0.04, 0.01, 8)
    common = dict(
        wires=[w + np.array([0.0, 0.0, 3.0]) for w in wires],
        n_per_edge_per_wire=npe, feeds=[(0, L_DIP / 4, 0j)],
        junctions=junctions, junction_ports=[0, 1], wavelength=WAVELENGTH,
    )  # fmt: skip
    # PEC solves, and the ports read as ports rather than as the node's own
    # self-capacitance (Y through the same public entry point that refused).
    Y = SinusoidalGalerkinSolver(**common, ground_z=0.0).compute_y_matrix()
    assert np.all(np.isfinite(Y[1:, 1:]))
    with pytest.raises(NotImplementedError, match="over a FINITE ground"):
        SinusoidalGalerkinSolver(
            **common, ground_z=0.0, ground_eps=(13.0, 0.005)
        ).compute_impedance()
    with pytest.raises(NotImplementedError, match="over a FINITE ground"):
        SinusoidalGalerkinSolver(
            **common, ground_z=0.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld"
        ).compute_impedance()
    with pytest.raises(NotImplementedError, match="mixed per-wire radii"):
        SinusoidalGalerkinSolver(
            **common, wire_radius=[1e-3, 2e-3, 1e-3, 2e-3]
        ).compute_impedance()


def test_sg_junction_port_and_bspline_converge_on_the_same_oracle_residue():
    """The claim the 1.5 % margin rests on, as a measurement.

    Formulation (b)'s oracle miss is flat in the mesh (1.4290 / 1.4289 /
    1.4280 / 1.4268 % at n_half 10 / 20 / 40 / 80) while `BSplineSolver`'s
    CLIMBS toward it (0.73 / 1.08 / 1.28 / 1.39 %). Neither is converging to
    zero — they are converging to each other, which is what identifies the
    residue as the oracle's linear-in-delta extrapolation rather than either
    port formulation's error. So the honest reading of "under 1.5 %" here is
    "indistinguishable from the reference implementation", not "1.5 %
    accurate".
    """
    ours, theirs = {}, {}
    for n_half in (40, 80):
        ours[n_half] = _sg_oracle_extrap(n_half)
        diffs = {}
        for delta in (0.04, 0.02):
            z_ref = _bridged_z(delta, 0.01, n_half)
            diffs[delta] = (_port_pair_zdiff(delta, 0.01, n_half) - z_ref, z_ref)
        ext = 2 * diffs[0.02][0] - diffs[0.04][0]
        theirs[n_half] = abs(ext) / abs(diffs[0.02][1])
    assert ours[80] < ours[40], ours  # flat-to-falling
    assert theirs[80] > theirs[40], theirs  # climbing
    assert abs(ours[80] - theirs[80]) < abs(ours[40] - theirs[40]), (ours, theirs)
    assert abs(ours[80] - theirs[80]) < 0.005, (ours, theirs)


# ======================================================================
# #191 — formulation (b) over a PEC ground
# ======================================================================
# M5b removed the port basis's lumped NODE charge from the free-space
# source only, so any `ground_z` refused: the image of that charge was left
# in, restoring a fraction of the very term the correction exists to take
# out.
#
# Under PEC the fix needs no new object. `_assemble_Z` builds the ground as
# the free-space field of MIRRORED sources and subtracts it once, G = A - B,
# and the image of a point charge is a point charge at the mirrored node. So
# the same D/S correction runs on B at the mirrored separation and enters G
# with the OPPOSITE sign:
#
#     A' = A - D     - D^T     + S        (free space, as before)
#     B' = B - D_img - D_img^T + S_img    (mirrored nodes)
#     G' = A' - B'
#
# The sign is derived, not fitted, and the tests below pin it: dropping the
# image half costs 15 % against `BSplineSolver` and flipping it costs 38 %,
# where the derived sign lands at the free-space agreement floor.
#
# Fresnel/Sommerfeld stay refused — their "image" of a point charge is not a
# point charge (test above). #151's grounded-junction rejection is untouched:
# a node IN the plane is pinned by its own image and cannot be a port.

PEC_H = 3.0  # node height over the plane — well clear of #151's grounded case


def _elevated(wires, h=PEC_H):
    return [w + np.array([0.0, 0.0, h]) for w in wires]


def _pec_two_member_solver(
    cls, delta=0.04, whisker=0.01, n_half=20, ground_z=0.0, **kw
):
    """M5b's two-member oracle topology, lifted over a PEC plane at z = 0."""
    wires, npe, junctions = _split_wires(delta, whisker, n_half)
    return cls(
        wires=_elevated(wires), n_per_edge_per_wire=npe,
        feeds=[(0, L_DIP / 4, 0j)], junctions=junctions,
        junction_ports=[0, 1], wavelength=WAVELENGTH, ground_z=ground_z, **kw,
    )  # fmt: skip


def _pec_one_member_solver(cls, n=20, ground_z=0.0, **kw):
    """The `PortAtEnd` topology — two ONE-member junctions — over PEC."""
    wires = [
        np.array([(0.0, -L_DIP / 2, 0.0), (0.0, -0.02, 0.0)]),
        np.array([(0.0, 0.02, 0.0), (0.0, L_DIP / 2, 0.0)]),
    ]
    return cls(
        wires=_elevated(wires), n_per_edge_per_wire=[[n], [n]],
        feeds=[(0, L_DIP / 4, 0j)], junctions=[[(0, "end")], [(1, "start")]],
        junction_ports=[0, 1], wavelength=WAVELENGTH, ground_z=ground_z, **kw,
    )  # fmt: skip


def _port_y(solver):
    """The junction-port sub-block of Y, with the dummy gap feed dropped."""
    return solver.compute_y_matrix()[1:, 1:]


@pytest.mark.parametrize(
    "make", [_pec_two_member_solver, _pec_one_member_solver], ids=["two", "one"]
)
def test_sg_junction_port_over_pec_reproduces_the_bspline_port(make):
    """The #191 gate: M5b's entrywise-vs-`BSplineSolver` check, over PEC.

    Nothing is shared between the two port formulations (Lagrange multiplier
    in a spline basis vs a node charge held outside a sinusoidal reaction
    integral) and now nothing is shared about the ground either — B-spline
    images its charge and current expansions, this one images the whole
    field. The full 2x2 port Y still agrees entrywise to 8.5e-5 (two-member)
    and 4.7e-6 (one-member `PortAtEnd`), against 5.6e-5 / 3.9e-6 for the same
    geometries in free space: the PEC block costs a factor under 1.6, i.e.
    the ground rides at the free-space agreement floor.

    What is left is discretization, not formulation: the two-member gap
    halves with the mesh (1.9e-4 / 8.5e-5 / 4.6e-5 at n_half 10 / 20 / 40)
    and is quadrature-converged (unmoved past `n_qp_node` 12, 1e-9).

    The ground is load-bearing in the comparison, not decorative — it moves
    the port Y by 33 % from the free-space answer.
    """
    Yb = _port_y(make(BSplineSolver))
    Yg = _port_y(make(SinusoidalGalerkinSolver))
    rel = np.abs(Yg - Yb) / np.abs(Yb)
    assert rel.max() < 2e-4, rel

    free = _port_y(make(BSplineSolver, ground_z=None))
    moved = np.abs(free - Yb).max() / np.abs(Yb).max()
    assert moved > 0.05, moved


class _ImageCorrectionScaled(SinusoidalGalerkinSolver):
    """`_assemble_Z_ported` with #191's image half rescaled — 0 undoes it
    (M5b's free-space-only correction over a ground), -1 flips its sign."""

    def __init__(self, *, image_factor=0.0, **kw):
        super().__init__(**kw)
        self.image_factor = float(image_factor)

    def _assemble_Z_ported(self, geom, k):
        G, seg_view = super()._assemble_Z_ported(geom, k)
        N = geom["n_segs"]
        w = 1.0 - self.image_factor
        D = self._node_charge_columns(
            geom, seg_view, k, nodes=self._port_node_positions(geom, mirror=True)
        )
        G = G.copy()
        G[:, N:] -= w * D
        G[N:, :] -= w * D.T
        G[N:, N:] += w * self._node_charge_image_pair_block(geom, k)
        return G, seg_view


@pytest.mark.parametrize(
    "make", [_pec_two_member_solver, _pec_one_member_solver], ids=["two", "one"]
)
def test_sg_junction_port_pec_image_term_and_its_sign_are_load_bearing(make):
    """The derivation, as a measurement: the image block is SUBTRACTED, so
    removing its node charge ADDS D_img back.

    Against `BSplineSolver` over the same PEC plane, the derived sign lands
    at 1e-5 (two-member 8.5e-5); leaving the image charge in — what M5b's
    refusal existed to prevent — misses by 15 %, and the opposite sign,
    double-counting instead of removing, by 38 %. Three decades between the
    derived answer and either neighbouring choice, so the sign is pinned by
    the oracle and not by inspection of the code.
    """
    Yb = _port_y(make(BSplineSolver))

    def miss(cls, **kw):
        return (np.abs(_port_y(make(cls, **kw)) - Yb) / np.abs(Yb)).max()

    good = miss(SinusoidalGalerkinSolver)
    none = miss(_ImageCorrectionScaled)
    flipped = miss(_ImageCorrectionScaled, image_factor=-1.0)
    assert good < 2e-4, good
    assert none > 100 * good, (none, good)
    assert flipped > 100 * good, (flipped, good)


@pytest.mark.parametrize("feed_readout", ["centre", "variational"])
def test_sg_junction_port_over_pec_y_block_symmetric(feed_readout):
    """The port block stays symmetric at the free-space floor over PEC.

    Both halves of the correction are applied to row and column alike and
    the mirrored separation is itself symmetric (reflection is an isometry:
    |node_p - M.node_q| = |M.node_p - node_q|), so the image half adds no
    asymmetry of its own — measured 6.3e-13 over PEC against 2.5e-12 in free
    space, both inside G5b's 1e-10 clause.
    """
    Y = _port_y(
        _pec_two_member_solver(SinusoidalGalerkinSolver, feed_readout=feed_readout)
    )
    asym = np.linalg.norm(Y - Y.T) / np.linalg.norm(Y)
    assert asym < 1e-10, f"port block asymmetry over PEC {asym:.3e}"
