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
    with pytest.raises(NotImplementedError):
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
    # Without enrichment the iterative path is real and still guards.
    with pytest.raises(NotImplementedError):
        HMatrixSolver(**{**kw, "use_singular_enrichment": False})


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


def test_sg_junction_port_solves_refuse():
    """The construction is kept and measured; the numbers are not shipped.
    Every solve entry point refuses so no caller can pick up a port
    impedance that is off by three orders of magnitude."""
    s = _port_pair_solver(0.04, 0.01, 8, cls=SinusoidalGalerkinSolver)
    ks = np.array([s.k])
    for call in (
        s.compute_impedance,
        s.compute_y_matrix,
        lambda: s.compute_impedance_swept(ks),
        lambda: s.compute_y_matrix_swept(ks),
    ):
        with pytest.raises(NotImplementedError, match="point charge.*bridge"):
            call()


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
