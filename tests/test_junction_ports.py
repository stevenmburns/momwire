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
"""

import numpy as np
import pytest

from momwire import BSplineSolver, HMatrixSolver, SinusoidalSolver

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


def _bridged_z(delta, whisker, n_half):
    """Reference: the same split structure with a real bridge wire across
    the gap carrying a classic delta-gap feed."""
    wires, npe, junctions = _split_wires(delta, whisker, n_half)
    wires.append(np.array([(0.0, -delta / 2, 0.0), (0.0, delta / 2, 0.0)]))
    npe.append([1])
    junctions[0].append((4, "start"))
    junctions[1].append((4, "end"))
    s = BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=npe,
        feeds=[(4, delta / 2, 1.0 + 0j)],
        junctions=junctions,
        wavelength=WAVELENGTH,
    )
    z, _ = s.compute_impedance()
    return complex(np.atleast_1d(z)[0])


def _port_pair_solver(delta, whisker, n_half, volts=(0j, 0j)):
    wires, npe, junctions = _split_wires(delta, whisker, n_half)
    return BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=npe,
        feeds=[(0, L_DIP / 4, 0.0 + 0j)],  # dummy 0 V gap feed
        junctions=junctions,
        junction_ports=[(0, volts[0]), (1, volts[1])],
        wavelength=WAVELENGTH,
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
    with pytest.raises(NotImplementedError):
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
