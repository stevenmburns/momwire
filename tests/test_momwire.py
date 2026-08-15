import pytest
import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

from momwire.bspline import BSplineSolver
from momwire.sinusoidal import SinusoidalSolver

import numpy as np


@pytest.mark.parametrize("nsegs", [20, 40, 80])
def test_d1_two_wire_yagi_smoke(nsegs):
    # Driver + 1.05x reflector at 1 halfdriver spacing — the classic 2-element
    # Yagi case. Mutual coupling pushes the driver Z away from bare-dipole
    # 69.6 - j18.2 toward roughly 77 + j6.
    hd = 0.962 * 22 / 4  # matches the solver defaults
    sp = hd
    driver = np.array([[0.0, -hd, 0.0], [0.0, hd, 0.0]])
    refl = np.array([[-sp, -1.05 * hd, 0.0], [-sp, 1.05 * hd, 0.0]])
    z, c = BSplineSolver(
        wires=[driver, refl],
        n_per_edge_per_wire=[[nsegs], [nsegs]],
        nsegs=nsegs,
        degree=1,
    ).compute_impedance()
    assert c.shape == (2 * (nsegs - 1),)
    assert np.isfinite(z.real) and np.isfinite(z.imag)
    assert np.isfinite(c).all()
    assert 65.0 < z.real < 85.0
    assert -10.0 < z.imag < 25.0


@pytest.mark.parametrize("nsegs", [20, 40, 80])
def test_d1_collinear_polyline(nsegs):
    # A "bent" wire whose polyline anchors happen to be collinear should give
    # nearly the same answer as a single-edge straight wire: the only path
    # difference is that cross-edge pairs go through quadrature instead of
    # the analytic formula.
    L = 2 * 0.962 * 22 / 4
    straight = np.array([[0.0, 0.0, 0.0], [0.0, L, 0.0]])
    polyline = np.array([[0.0, 0.0, 0.0], [0.0, L / 2, 0.0], [0.0, L, 0.0]])
    z_straight, _ = BSplineSolver(
        wires=[straight], n_per_edge_per_wire=[[nsegs]], nsegs=nsegs, degree=1
    ).compute_impedance()
    # Use n_qp_off=8 so the artificial cross-edge quadrature at the fake
    # corner has the same precision as the analytic same-edge path.
    z_bent, _ = BSplineSolver(
        wires=[polyline],
        n_per_edge_per_wire=[[nsegs // 2, nsegs // 2]],
        nsegs=nsegs,
        degree=1,
        n_qp_pair=8,
    ).compute_impedance()
    assert abs(z_bent - z_straight) < 0.2


def test_d1_v_dipole_smoke():
    # 30-deg V-dipole: arms bent away from the y-axis in the y-z plane.
    L = 2 * 0.962 * 22 / 4
    half = L / 2
    alpha = np.radians(30)
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)
    polyline = np.array(
        [
            [0.0, -half * cos_a, -half * sin_a],
            [0.0, 0.0, 0.0],
            [0.0, +half * cos_a, -half * sin_a],
        ]
    )
    z, c = BSplineSolver(
        wires=[polyline], n_per_edge_per_wire=[[40, 40]], nsegs=80, degree=1
    ).compute_impedance()
    assert c.shape == (79,)
    assert np.isfinite(z.real) and np.isfinite(z.imag)
    assert np.isfinite(c).all()
    # Bending lowers R and pushes X more negative compared to straight (69.6 - j18.5).
    assert 30.0 < z.real < 65.0
    assert z.imag < -25.0


def test_d1_moxon_smoke():
    # Approximate moxon at 28.57 MHz with the antenna_designer default
    # parameters. Sanity-check R/X land in plausible bands and currents
    # come out finite.
    C_LIGHT = 299_792_458.0
    freq_mhz = 28.57
    wavelength = C_LIGHT / (freq_mhz * 1e6)
    halfdriver = 0.962 * wavelength / 4
    aspect_ratio = 0.3646
    tipspacer_factor = 0.0773
    t0_factor = 0.4078
    long_ = 2 * halfdriver / (1 + 2 * aspect_ratio * t0_factor)
    short_ = aspect_ratio * long_
    tipspacer = short_ * tipspacer_factor
    t0 = short_ * t0_factor
    eps = 0.05

    def rx(p):
        return (-p[0], p[1], p[2])

    def ry(p):
        return (p[0], -p[1], p[2])

    S = (short_ / 2, eps, 0.0)
    A = (S[0], long_ / 2, 0.0)
    B = (A[0] - t0, A[1], 0.0)
    Cc = (B[0] - tipspacer, B[1], 0.0)
    D = rx(A)
    E = ry(D)
    F = ry(Cc)
    G = ry(B)
    H = ry(A)
    T = ry(S)

    driver = np.array([G, H, T, S, A, B], dtype=float)
    reflector = np.array([Cc, D, E, F], dtype=float)

    sim = BSplineSolver(
        wires=[driver, reflector],
        n_per_edge_per_wire=[[8, 21, 1, 21, 8], [8, 21, 8]],
        feed_wire_index=0,
        nsegs=40,
        wavelength=wavelength,
        halfdriver_factor=0.962,
        degree=1,
    )
    z, c = sim.compute_impedance()
    assert np.isfinite(z.real) and np.isfinite(z.imag)
    assert np.isfinite(c).all()
    # Moxons are nominally tuned for ~50 Ω at resonance; with the canonical
    # antenna_designer factors and 28.57 MHz design freq we see ~70 + j10
    # which is a reasonable working point (the canonical design is tuned
    # for a slightly different free-space target than ours).
    assert 40.0 < z.real < 110.0
    assert -30.0 < z.imag < 40.0


def test_d1_hexbeam_smoke():
    # Single-band hexbeam at 28.47 MHz with the antenna_designer default
    # factors (halfdriver=2.82m, tipspacer=0.1312, t0=0.1243). Hexbeams
    # are tuned for ~50 Ω.
    import math

    C_LIGHT = 299_792_458.0
    freq_mhz = 28.47
    wavelength = C_LIGHT / (freq_mhz * 1e6)
    halfdriver = 2.82
    tipspacer_factor = 0.1312
    t0_factor = 0.1243
    radius = halfdriver / (2 - t0_factor - tipspacer_factor)
    tipspacer = radius * tipspacer_factor
    t0 = radius * t0_factor
    t1 = radius - tipspacer - t0
    eps = 0.05
    cos30 = math.sqrt(3) / 2
    sin30 = 0.5

    def rx(p):
        return (-p[0], p[1], p[2])

    def ry(p):
        return (p[0], -p[1], p[2])

    A = (radius * cos30, radius * sin30, 0.0)
    B = (A[0] - t1 * cos30, A[1] + t1 * sin30, 0.0)
    D = (0.0, radius, 0.0)
    Cc = (D[0] + t0 * cos30, D[1] - t0 * sin30, 0.0)
    E = rx(A)
    F = ry(E)
    G = ry(D)
    H = ry(Cc)
    I_ = ry(B)
    J = ry(A)
    S = (eps * cos30, eps * sin30, 0.0)
    T = ry(S)

    driver = np.array([I_, J, T, S, A, B], dtype=float)
    reflector = np.array([Cc, D, E, F, G, H], dtype=float)

    sim = BSplineSolver(
        wires=[driver, reflector],
        n_per_edge_per_wire=[[15, 21, 1, 21, 15], [3, 21, 21, 21, 3]],
        feed_wire_index=0,
        nsegs=40,
        wavelength=wavelength,
        halfdriver_factor=1.071,
        degree=1,
    )
    z, c = sim.compute_impedance()
    assert np.isfinite(z.real) and np.isfinite(z.imag)
    assert np.isfinite(c).all()
    # Hexbeam at the canonical free-space design point lands near 50+j20.
    assert 30.0 < z.real < 75.0
    assert -10.0 < z.imag < 45.0


# ---- Junctions (K wires meeting at a node) ----


@pytest.mark.parametrize("degree,tol", [(1, 1e-9), (2, 1e-4)])
def test_k2_junction_equivalent_to_single_polyline(degree, tol):
    """A K=2 junction at a kink is mathematically equivalent to a single
    polyline with that kink as an interior knot. For the tent basis (d=1)
    the equivalence is EXACT — the Lagrange-augmented KCL constraint
    reduces the two directional bases to one effective DOF identical to
    the interior tent — so d=1 pins roundoff. For d=2 the split slightly
    changes the basis space (a single clamped spline carries more
    smoothness through the kink knot than two clamped splines joined by
    directional bases + KCL), so agreement is only near-exact (measured
    ~3.6e-6 Ω at this mesh); the loose gate guards the junction
    formulation without overclaiming equivalence.
    """
    # Bent dipole, kink at (0, 0, -2), feed mid-arm (NOT at the kink).
    pl_single = np.array([[0.0, -5.0, 0.0], [0.0, 0.0, -2.0], [0.0, 5.0, 0.0]])
    sim_single = BSplineSolver(
        wires=[pl_single],
        n_per_edge_per_wire=[[15, 15]],
        feed_wire_index=0,
        feed_arclength=2.5,
        wavelength=22,
        nsegs=15,
        wire_radius=0.0005,
        degree=degree,
    )
    z_single, _ = sim_single.compute_impedance()

    # Same geometry split into 2 wires joined by a K=2 junction at the kink.
    pl0 = np.array([[0.0, -5.0, 0.0], [0.0, 0.0, -2.0]])
    pl1 = np.array([[0.0, 0.0, -2.0], [0.0, 5.0, 0.0]])
    sim_junction = BSplineSolver(
        wires=[pl0, pl1],
        n_per_edge_per_wire=[[15], [15]],
        feed_wire_index=0,
        feed_arclength=2.5,
        wavelength=22,
        nsegs=15,
        wire_radius=0.0005,
        junctions=[[(0, "end"), (1, "start")]],
        degree=degree,
    )
    z_junction, _ = sim_junction.compute_impedance()
    assert abs(z_junction - z_single) < tol, (
        f"K=2 junction Z={z_junction}, single-polyline Z={z_single}"
    )


def test_sinusoidal_y_matrix_with_junctions_single_feed():
    """SinusoidalSolver handles junctions structurally (via the N±(i) neighbour
    topology with σ=±1 signs) rather than via a Lagrange-augmented KCL, so
    compute_y_matrix doesn't need a separate junction path. Lock that in:
    Y[0,0] = 1/Z on a K=2-junction bent dipole."""
    pl0 = np.array([[0.0, -5.0, 0.0], [0.0, 0.0, -2.0]])
    pl1 = np.array([[0.0, 0.0, -2.0], [0.0, 5.0, 0.0]])
    common = dict(
        wires=[pl0, pl1],
        n_per_edge_per_wire=[[15], [15]],
        feed_wire_index=0,
        feed_arclength=2.5,
        wavelength=22,
        wire_radius=0.0005,
        junctions=[[(0, "end"), (1, "start")]],
    )
    z, _ = SinusoidalSolver(**common).compute_impedance()
    Y = SinusoidalSolver(**common).compute_y_matrix()
    assert Y.shape == (1, 1)
    assert abs(Y[0, 0] - 1.0 / z) < 1e-9, f"Y[0,0]={Y[0, 0]}, 1/Z={1.0 / z}"


def test_sinusoidal_y_matrix_with_junctions_multi_feed():
    """Two feeds (one per wire) sharing a K=2 junction. Y should be near-
    symmetric (sinusoidal's MoM isn't quite as tight as the tent basis's at
    fixed segmentation, so use a looser tolerance) and match the N-solve
    reference."""
    pl0 = np.array([[0.0, -5.0, 0.0], [0.0, 0.0, -2.0]])
    pl1 = np.array([[0.0, 0.0, -2.0], [0.0, 5.0, 0.0]])
    common = dict(
        wires=[pl0, pl1],
        n_per_edge_per_wire=[[15], [15]],
        feeds=[(0, 2.5, 1 + 0j), (1, 2.5, 1 + 0j)],
        wavelength=22,
        wire_radius=0.0005,
        junctions=[[(0, "end"), (1, "start")]],
    )

    Y = SinusoidalSolver(**common).compute_y_matrix()
    assert Y.shape == (2, 2)
    assert abs(Y[0, 1] - Y[1, 0]) < 1e-6, "Y not symmetric within MoM tolerance"

    Y_ref = np.zeros((2, 2), dtype=np.complex128)
    for j in range(2):
        feeds_j = [
            (w, arc, 1.0 + 0j if k == j else 0.0 + 0j)
            for k, (w, arc, _) in enumerate(common["feeds"])
        ]
        ref_kwargs = {**common, "feeds": feeds_j}
        sim_j = SinusoidalSolver(**ref_kwargs)
        _z, alpha = sim_j.compute_impedance()
        geom = sim_j._build_geometry()
        _G, seg_view = sim_j._assemble_Z(geom, sim_j.k)
        Y_ref[:, j] = [
            sim_j._feed_segment_current(alpha, seg_view, fi) for fi in geom["feed_segs"]
        ]
    assert np.allclose(Y, Y_ref, atol=1e-10), f"Y - Y_ref:\n{Y - Y_ref}"


def test_sinusoidal_y_matrix_swept_with_junctions_matches_per_freq():
    pl0 = np.array([[0.0, -5.0, 0.0], [0.0, 0.0, -2.0]])
    pl1 = np.array([[0.0, 0.0, -2.0], [0.0, 5.0, 0.0]])
    common = dict(
        wires=[pl0, pl1],
        n_per_edge_per_wire=[[15], [15]],
        feeds=[(0, 2.5, 1 + 0j), (1, 2.5, 1 + 0j)],
        wire_radius=0.0005,
        junctions=[[(0, "end"), (1, "start")]],
    )
    C_LIGHT = 299_792_458.0
    freqs_mhz = np.array([10.0, 14.0, 20.0])
    k_array = 2 * np.pi * freqs_mhz * 1e6 / C_LIGHT

    sim_swept = SinusoidalSolver(wavelength=22, **common)
    Y_swept = sim_swept.compute_y_matrix_swept(k_array)
    assert Y_swept.shape == (3, 2, 2)

    for i, f in enumerate(freqs_mhz):
        sim_f = SinusoidalSolver(wavelength=C_LIGHT / (f * 1e6), **common)
        Y_f = sim_f.compute_y_matrix()
        assert np.allclose(Y_swept[i], Y_f, atol=1e-10), (
            f"f={f}: swept Y differs from per-k Y"
        )


@pytest.mark.parametrize("degree", [1, 2])
def test_bspline_y_matrix_with_junctions_single_feed(degree):
    """compute_y_matrix on a K=2-junction antenna with a single feed should
    return [[1/Z]] where Z is what compute_impedance reports. Bspline uses
    the Lagrange-augmented KCL identically to the d=1 tent basis but with the
    Galerkin reciprocity Y = B^T X readout. Covers both d=1 (tent-equivalent)
    and d=2 (the default quadratic) bases."""
    pl0 = np.array([[0.0, -5.0, 0.0], [0.0, 0.0, -2.0]])
    pl1 = np.array([[0.0, 0.0, -2.0], [0.0, 5.0, 0.0]])
    common = dict(
        wires=[pl0, pl1],
        n_per_edge_per_wire=[[15], [15]],
        feed_wire_index=0,
        feed_arclength=2.5,
        wavelength=22,
        wire_radius=0.0005,
        junctions=[[(0, "end"), (1, "start")]],
        degree=degree,
    )
    z, _ = BSplineSolver(**common).compute_impedance()
    Y = BSplineSolver(**common).compute_y_matrix()
    assert Y.shape == (1, 1)
    assert abs(Y[0, 0] - 1.0 / z) < 1e-10, (
        f"d={degree}: Y[0,0]={Y[0, 0]}, 1/Z={1.0 / z}"
    )


@pytest.mark.parametrize("degree", [1, 2])
def test_bspline_y_matrix_with_junctions_multi_feed(degree):
    """Two feeds on a K=2-junction antenna. Y should be symmetric and match
    the N-independent-solves reference at both d=1 and d=2."""
    pl0 = np.array([[0.0, -5.0, 0.0], [0.0, 0.0, -2.0]])
    pl1 = np.array([[0.0, 0.0, -2.0], [0.0, 5.0, 0.0]])
    common = dict(
        wires=[pl0, pl1],
        n_per_edge_per_wire=[[15], [15]],
        feeds=[(0, 2.5, 1 + 0j), (1, 2.5, 1 + 0j)],
        wavelength=22,
        wire_radius=0.0005,
        junctions=[[(0, "end"), (1, "start")]],
        degree=degree,
    )

    Y = BSplineSolver(**common).compute_y_matrix()
    assert Y.shape == (2, 2)
    assert abs(Y[0, 1] - Y[1, 0]) < 1e-10, f"d={degree}: Y not symmetric (reciprocity)"

    # Reference: N independent solves, Y[:, j] = B^T @ coeffs_j.
    Y_ref = np.zeros((2, 2), dtype=np.complex128)
    for j in range(2):
        feeds_j = [
            (w, arc, 1.0 + 0j if k == j else 0.0 + 0j)
            for k, (w, arc, _) in enumerate(common["feeds"])
        ]
        sim_j = BSplineSolver(**{**common, "feeds": feeds_j})
        _z, coeffs = sim_j.compute_impedance()
        geom = sim_j._build_geometry()
        supp_seg, polys, _kcl, wk, wbg = sim_j._build_basis_polynomials(geom)
        n_bt = supp_seg.shape[0]
        for i, (w_i, arc_i, _v) in enumerate(common["feeds"]):
            arc_at_knot = geom["per_wire"][w_i]["arc_at_knot"]
            s_f = arc_i if arc_i is not None else arc_at_knot[-1] / 2.0
            v_i = sim_j._build_source_vector(geom, wk, wbg, n_bt, wi=w_i, s_f=s_f)
            Y_ref[i, j] = v_i @ coeffs
    assert np.allclose(Y, Y_ref, atol=1e-10), f"d={degree}: Y - Y_ref:\n{Y - Y_ref}"


@pytest.mark.parametrize("degree", [1, 2])
def test_bspline_y_matrix_swept_with_junctions_matches_per_freq(degree):
    pl0 = np.array([[0.0, -5.0, 0.0], [0.0, 0.0, -2.0]])
    pl1 = np.array([[0.0, 0.0, -2.0], [0.0, 5.0, 0.0]])
    common = dict(
        wires=[pl0, pl1],
        n_per_edge_per_wire=[[15], [15]],
        feeds=[(0, 2.5, 1 + 0j), (1, 2.5, 1 + 0j)],
        wire_radius=0.0005,
        junctions=[[(0, "end"), (1, "start")]],
        degree=degree,
    )
    C_LIGHT = 299_792_458.0
    freqs_mhz = np.array([10.0, 14.0, 20.0])
    k_array = 2 * np.pi * freqs_mhz * 1e6 / C_LIGHT

    sim_swept = BSplineSolver(wavelength=22, **common)
    Y_swept = sim_swept.compute_y_matrix_swept(k_array)
    assert Y_swept.shape == (3, 2, 2)

    for i, f in enumerate(freqs_mhz):
        sim_f = BSplineSolver(wavelength=C_LIGHT / (f * 1e6), **common)
        Y_f = sim_f.compute_y_matrix()
        assert np.allclose(Y_swept[i], Y_f, atol=1e-10), (
            f"d={degree}, f={f}: swept Y differs from per-k Y"
        )


def test_d1_hentenna_smoke():
    """Single-band hentenna at 28.47 MHz with the antenna_designer params_50
    factors. Geometry is a tall narrow rectangular loop with a horizontal
    cross-bar near the bottom; the feed sits in a small gap (T,S) at the
    middle of the cross-bar. K=3 junctions at B (right of cross-bar) and D
    (left of cross-bar) where the cross-bar half meets the upper and lower
    loop perimeters; K=2 junctions at S and T where the cross-bar halves
    meet the feed wire.

        C----------------------------A
        |                            |
        |                            |
        D------------T--S------------B
        |                            |
        |                            |
        E----------------------------F
    """
    C_LIGHT = 299_792_458.0
    freq_mhz = 28.47
    wavelength = C_LIGHT / (freq_mhz * 1e6)
    # antenna_designer hentenna params_50 (tuned for ~50 Ω feed).
    width_factor = 0.1378
    top_height_factor = 0.5081
    mid_height_factor = 0.1094
    eps = 0.05

    half_w = wavelength * width_factor / 2
    z_mid = wavelength * (mid_height_factor - top_height_factor)
    z_bot = -wavelength * top_height_factor

    A = (0.0, half_w, 0.0)
    B = (0.0, half_w, z_mid)
    F = (0.0, half_w, z_bot)
    S = (0.0, eps, z_mid)
    C = (0.0, -half_w, 0.0)
    D = (0.0, -half_w, z_mid)
    E = (0.0, -half_w, z_bot)
    T = (0.0, -eps, z_mid)

    N = 21
    Nfeed = 3
    wires = [
        np.array([T, S], dtype=float),  # 0: feed gap
        np.array([S, B], dtype=float),  # 1: right half of cross-bar
        np.array(
            [B, A, C, D], dtype=float
        ),  # 2: upper rectangle (right-up-top-down to D)
        np.array([T, D], dtype=float),  # 3: left half of cross-bar
        np.array([D, E, F, B], dtype=float),  # 4: lower rectangle (down-bottom-up to B)
    ]
    n_per_edge_per_wire = [[Nfeed], [N], [N, N, N], [N], [N, N, N]]
    junctions = [
        [(0, "end"), (1, "start")],  # at S
        [(0, "start"), (3, "start")],  # at T
        [(1, "end"), (2, "start"), (4, "end")],  # at B (K=3)
        [(2, "end"), (3, "end"), (4, "start")],  # at D (K=3)
    ]

    sim = BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=n_per_edge_per_wire,
        feed_wire_index=0,
        feed_arclength=eps,
        wavelength=wavelength,
        nsegs=N,
        wire_radius=0.0005,
        junctions=junctions,
        degree=1,
    )
    z, c = sim.compute_impedance()
    assert np.isfinite(z.real) and np.isfinite(z.imag)
    assert np.isfinite(c).all()
    # Hentenna params_50 is tuned for ~50 Ω at 28.47 MHz in NEC2; the
    # tent (d=1) Galerkin basis lands within a similar window. Use the same
    # generous bands as the moxon/hexbeam smoke tests.
    assert 25.0 < z.real < 110.0, f"R={z.real} out of plausible 50Ω-tuned range"
    assert -40.0 < z.imag < 60.0, f"X={z.imag} out of plausible 50Ω-tuned range"


def test_bspline_cpp_assemble_z_matches_numpy():
    """The C++ assemble_Z_bspline accelerator must agree with the numpy
    reference path bit-exactly (modulo floating-point reduction order ~1e-12
    relative). Run by toggling the dispatch flag in momwire.bspline.
    """
    import momwire.bspline as bmod
    from momwire.bspline import BSplineSolver

    if not bmod._HAVE_BSPLINE_ASSEMBLE_ACCEL:
        pytest.skip("Z assembly accelerator not built")

    L = 2 * 0.962 * 22 / 4
    wires = [np.array([[0.0, -L / 2, 0.0], [0.0, L / 2, 0.0]])]
    sim = BSplineSolver(wires=wires, n_per_edge_per_wire=[[21]], nsegs=21, degree=2)

    # Driver impedance via C++ path
    z_cpp, _ = sim.compute_impedance()

    # Force numpy fallback by flipping the module flag
    saved = bmod._HAVE_BSPLINE_ASSEMBLE_ACCEL
    try:
        bmod._HAVE_BSPLINE_ASSEMBLE_ACCEL = False
        # Re-build sim to invalidate any cached Z (compute_impedance is stateless re Z)
        z_np, _ = BSplineSolver(
            wires=wires, n_per_edge_per_wire=[[21]], nsegs=21, degree=2
        ).compute_impedance()
    finally:
        bmod._HAVE_BSPLINE_ASSEMBLE_ACCEL = saved

    rel = abs(z_cpp - z_np) / abs(z_np)
    assert rel < 1e-12, f"C++ vs numpy Z assembly disagreement: rel diff {rel}"


def test_bspline_cpp_kernel_matches_numpy():
    """The C++ B-spline moment-integral and static-moments accelerators must
    agree with the pure-numpy reference to ~1e-9 relative (the floating-point
    reduction-order error is below GL quadrature precision and the closed-form
    arithmetic precision).
    """
    from momwire._bspline_kernels import (
        _seg_seg_full_moments_offedge,
        _seg_seg_static_moments,
        _HAVE_BSPLINE_ACCEL,
        _HAVE_BSPLINE_STATIC_ACCEL,
    )

    if not _HAVE_BSPLINE_ACCEL or not _HAVE_BSPLINE_STATIC_ACCEL:
        pytest.skip("C++ accelerators not built")

    # Force the numpy reference paths via monkeypatching the module-level flags
    import momwire._bspline_kernels as kmod

    N = 12
    seg = np.linspace(0, 6.0, N + 1)
    seg_l = np.column_stack([np.zeros(N), seg[:-1], np.zeros(N)])
    seg_r = np.column_stack([np.zeros(N), seg[1:], np.zeros(N)])

    J_cpp = _seg_seg_full_moments_offedge(seg_l, seg_r, seg_l, seg_r, 0.0005, 0.3, 2, 4)
    S_cpp = _seg_seg_static_moments(seg, 0.0005, max_d=2)

    # Force numpy paths
    saved_full = kmod._HAVE_BSPLINE_ACCEL
    saved_static = kmod._HAVE_BSPLINE_STATIC_ACCEL
    try:
        kmod._HAVE_BSPLINE_ACCEL = False
        kmod._HAVE_BSPLINE_STATIC_ACCEL = False
        J_np = _seg_seg_full_moments_offedge(
            seg_l, seg_r, seg_l, seg_r, 0.0005, 0.3, 2, 4
        )
        S_np = _seg_seg_static_moments(seg, 0.0005, max_d=2)
    finally:
        kmod._HAVE_BSPLINE_ACCEL = saved_full
        kmod._HAVE_BSPLINE_STATIC_ACCEL = saved_static

    rel_J = np.max(np.abs(J_cpp - J_np) / (np.abs(J_np) + 1e-30))
    rel_S = np.max(np.abs(S_cpp - S_np) / (np.abs(S_np) + 1e-30))
    # Both paths evaluate sympy-derived closed forms but through different
    # math libraries (numpy / libm), so float-precision differences in
    # arcsinh / sqrt cause ~1e-9 relative deviation. Well below the
    # GL-quadrature precision and the antenna-Z noise floor.
    assert rel_J < 1e-8, f"J kernel rel diff {rel_J}"
    assert rel_S < 1e-7, f"Static kernel rel diff {rel_S}"


def test_bspline_cpp_degree1_matches_numpy():
    """Exercise the C++ degree=1 (max_d=1) template instantiations of the
    B-spline accelerators -- the moment kernel<1>, the static moments at
    max_d=1, and assemble_Z_bspline_kernel<1> -- which the degree=2 tests
    never reach. Each must agree with the pure-numpy reference path.
    """
    import momwire.bspline as bmod
    from momwire.bspline import BSplineSolver
    import momwire._bspline_kernels as kmod
    from momwire._bspline_kernels import (
        _seg_seg_full_moments_offedge,
        _seg_seg_static_moments,
    )

    if not (
        kmod._HAVE_BSPLINE_ACCEL
        and kmod._HAVE_BSPLINE_STATIC_ACCEL
        and bmod._HAVE_BSPLINE_ASSEMBLE_ACCEL
    ):
        pytest.skip("B-spline C++ accelerators not built")

    # --- moment kernel<1> + static moments at max_d=1 (off-edge + same-edge) --
    N = 12
    seg = np.linspace(0.0, 6.0, N + 1)
    seg_l = np.column_stack([np.zeros(N), seg[:-1], np.zeros(N)])
    seg_r = np.column_stack([np.zeros(N), seg[1:], np.zeros(N)])

    J_cpp = _seg_seg_full_moments_offedge(seg_l, seg_r, seg_l, seg_r, 0.0005, 0.3, 1, 4)
    S_cpp = _seg_seg_static_moments(seg, 0.0005, max_d=1)
    # max_d=1 -> (d+1, d+1) = (2, 2) leading dims
    assert J_cpp.shape[:2] == (2, 2)
    assert S_cpp.shape[:2] == (2, 2)

    saved_full = kmod._HAVE_BSPLINE_ACCEL
    saved_static = kmod._HAVE_BSPLINE_STATIC_ACCEL
    try:
        kmod._HAVE_BSPLINE_ACCEL = False
        kmod._HAVE_BSPLINE_STATIC_ACCEL = False
        J_np = _seg_seg_full_moments_offedge(
            seg_l, seg_r, seg_l, seg_r, 0.0005, 0.3, 1, 4
        )
        S_np = _seg_seg_static_moments(seg, 0.0005, max_d=1)
    finally:
        kmod._HAVE_BSPLINE_ACCEL = saved_full
        kmod._HAVE_BSPLINE_STATIC_ACCEL = saved_static

    rel_J = np.max(np.abs(J_cpp - J_np) / (np.abs(J_np) + 1e-30))
    rel_S = np.max(np.abs(S_cpp - S_np) / (np.abs(S_np) + 1e-30))
    assert rel_J < 1e-8, f"degree=1 J kernel rel diff {rel_J}"
    assert rel_S < 1e-7, f"degree=1 static kernel rel diff {rel_S}"

    # --- assemble_Z_bspline_kernel<1> via a full degree=1 dipole solve --------
    L = 2 * 0.962 * 22 / 4
    wires = [np.array([[0.0, -L / 2, 0.0], [0.0, L / 2, 0.0]])]
    z_cpp, _ = BSplineSolver(
        wires=wires, n_per_edge_per_wire=[[21]], nsegs=21, degree=1
    ).compute_impedance()

    saved_asm = bmod._HAVE_BSPLINE_ASSEMBLE_ACCEL
    try:
        bmod._HAVE_BSPLINE_ASSEMBLE_ACCEL = False
        z_np, _ = BSplineSolver(
            wires=wires, n_per_edge_per_wire=[[21]], nsegs=21, degree=1
        ).compute_impedance()
    finally:
        bmod._HAVE_BSPLINE_ASSEMBLE_ACCEL = saved_asm

    rel_z = abs(z_cpp - z_np) / abs(z_np)
    assert rel_z < 1e-12, f"degree=1 Z assembly C++ vs numpy rel diff {rel_z}"


@pytest.mark.parametrize("degree,nsegs", [(2, 21), (2, 81)])
def test_bspline_dipole_converges_to_nec(degree, nsegs):
    """BSplineSolver degree-2 (quadratic) on the default half-wave dipole.
    With higher-order bases and analytic singularity subtraction we expect
    rapid convergence to the NEC reference 69.64 - j18.21; even N=21 should
    be within ~1 Ω.
    """
    L = 2 * 0.962 * 22 / 4
    wires = [np.array([[0.0, -L / 2, 0.0], [0.0, L / 2, 0.0]])]
    sim = BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=[[nsegs]],
        nsegs=nsegs,
        degree=degree,
    )
    z, coeffs = sim.compute_impedance()
    assert np.isfinite(z.real) and np.isfinite(z.imag)
    assert np.isfinite(coeffs).all()
    # Half-wave dipole NEC reference: 69.64 - j18.21
    assert abs(z.real - 69.64) < 1.0, f"R={z.real}"
    assert abs(z.imag - (-18.21)) < 1.0, f"X={z.imag}"


def test_bspline_d2_dipole_smoothed_source():
    """Source smoothing via `feed_smoothing_factor` (α) replaces the delta
    gap with a cos² bump of width w = α·h_feed_segment, integrated against
    each basis. On the dipole this removes the source-localized current
    singularity that otherwise caps the integrated-impedance convergence
    at O(1/N) regardless of basis degree.

    Two checks:
      1. Pin α=4 at n=81 to the recorded productized value.
      2. Fit R(N) = R_inf + C/N^p over n ∈ {21, 41, 81} with α=2 and assert
         the rate clearly lifts above the delta-gap baseline (empirically
         the baseline runs ~1.20; α=2 lifts to ~1.55).
    """
    L = 2 * 0.962 * 22 / 4
    wires = [np.array([[0.0, -L / 2, 0.0], [0.0, L / 2, 0.0]])]

    def sweep(alpha, ns):
        Zs = []
        for n in ns:
            z, _ = BSplineSolver(
                wires=wires,
                n_per_edge_per_wire=[[n]],
                nsegs=n,
                degree=2,
                feed_smoothing_factor=alpha,
            ).compute_impedance()
            Zs.append(z)
        return Zs

    # 1. Pin α=4 at n=81 (smoothing-on converged value; ±0.1 Ω R, ±0.5 Ω X
    #    leaves headroom for compiler/platform jitter while staying tighter
    #    than the gap to the delta-gap baseline at the same n).
    z = sweep(4.0, [81])[0]
    assert abs(z.real - 69.78) < 0.1, f"α=4 n=81 R={z.real}, expected ≈69.78"
    assert abs(z.imag - (-18.02)) < 0.5, f"α=4 n=81 X={z.imag}, expected ≈-18.02"

    # 2. R-rate lift at α=2 vs delta-gap baseline.
    ns = [21, 41, 81]
    Rs_delta = [zz.real for zz in sweep(None, ns)]
    Rs_a2 = [zz.real for zz in sweep(2.0, ns)]

    def rate(vals):
        d12 = vals[0] - vals[1]
        d23 = vals[1] - vals[2]
        assert d12 * d23 > 0, f"R differences sign-flipped (noise floor): {vals}"
        return np.log(abs(d12 / d23)) / np.log(ns[1] / ns[0])

    p_delta = rate(Rs_delta)
    p_a2 = rate(Rs_a2)
    # Floor of 1.4 has margin vs the empirical α=2 rate of ~1.55; the
    # +0.2 lift assertion catches a silent regression where α=2 happens
    # to land near 69.78 without actually improving the rate.
    assert p_a2 > 1.4, (
        f"α=2 R rate p={p_a2:.2f} below the 1.4 floor (delta p={p_delta:.2f})"
    )
    assert p_a2 > p_delta + 0.2, (
        f"α=2 did not clearly lift R rate vs delta-gap: {p_delta:.2f} → {p_a2:.2f}"
    )


# ---------------------------------------------------------------------------
# feed_model — NEC's segment-wide gap on the B-spline oracle (momwire#216)
# ---------------------------------------------------------------------------


def _segment_gap_kwargs(n, degree=2, **extra):
    hd = 0.962 * 22 / 4
    return dict(
        wires=[np.array([[0.0, 0.0, -hd], [0.0, 0.0, hd]])],
        n_per_edge_per_wire=[[n]],
        feed_wire_index=0,
        feed_arclength=hd,
        wavelength=22,
        wire_radius=0.0005,
        degree=degree,
        **extra,
    )


def test_bspline_feed_model_validation_and_guard():
    """The option's surface: the siblings' validation idiom, plus the guard
    that `"segment"` and `feed_smoothing_factor` may not be combined — both
    replace the point drive with a spread one, so composing them would silently
    integrate a cos² bump against a gap-average and mean nothing."""
    with pytest.raises(ValueError, match="feed_model must be"):
        BSplineSolver(**_segment_gap_kwargs(21, feed_model="bogus"))
    with pytest.raises(ValueError, match="mutually exclusive"):
        BSplineSolver(
            **_segment_gap_kwargs(21, feed_model="segment", feed_smoothing_factor=2.0)
        )
    # The default is the point gap, named.
    assert BSplineSolver(**_segment_gap_kwargs(21)).feed_model == "point"


@pytest.mark.parametrize("degree", [1, 2])
def test_bspline_point_feed_model_is_bit_identical_to_the_default(degree):
    """`feed_model="point"` must be the untouched pre-#216 drive — bit for
    bit, not to a tolerance, because every pinned B-spline constant in the
    tree is a point-gap reading."""
    kw = _segment_gap_kwargs(41, degree=degree)
    z_default, c_default = BSplineSolver(**kw).compute_impedance()
    z_named, c_named = BSplineSolver(**kw, feed_model="point").compute_impedance()
    assert complex(z_named) == complex(z_default)
    assert np.array_equal(c_named, c_default)


@pytest.mark.parametrize("degree", [1, 2])
def test_bspline_segment_gap_drive_is_the_exact_cell_average(degree):
    """The `"segment"` drive is v_m = (1/Δ)∫_cell Φ_m ds, and the existing
    `n_qp_source` Gauss rule integrates it EXACTLY: each Φ_m is a polynomial
    of degree d on one knot span, and a 16-node rule is exact through degree
    31. Checked against an independent 4000-node composite Simpson evaluation
    of the same integral (no shared quadrature), to ~1e-14.

    Also checked: the drive sums to 1 over the full basis partition-of-unity
    on an interior cell (∫(1/Δ)Σ_m Φ_m = 1), which is the statement that the
    source carries exactly V volts across the gap — NEC's Eq 187 convention.
    """
    kw = _segment_gap_kwargs(21, degree=degree)
    sim = BSplineSolver(**kw, feed_model="segment")
    geom = sim._build_geometry()
    supp_seg, _p, _kcl, wk, wbg = sim._build_basis_polynomials(geom)
    n_bt = supp_seg.shape[0]
    s_f = kw["feed_arclength"]
    v = sim._build_source_vector(geom, wk, wbg, n_bt, wi=0, s_f=s_f)

    from scipy.interpolate import BSpline as _BSpline

    arc = geom["per_wire"][0]["arc_at_knot"]
    seg = int(np.searchsorted(arc, s_f, side="right")) - 1
    s_lo, s_hi = float(arc[seg]), float(arc[seg + 1])
    # Independent quadrature: composite Simpson on 4001 samples of the cell.
    m = 4000
    xs = np.linspace(s_lo, s_hi, m + 1)
    w = np.ones(m + 1)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    w *= (s_hi - s_lo) / (3.0 * m)
    DM = _BSpline.design_matrix(xs, wk[0], sim.degree).toarray()
    ref_full = (DM * w[:, None]).sum(axis=0) / (s_hi - s_lo)

    kept, local_to_global = wbg[0]
    ref = np.zeros(n_bt, dtype=np.complex128)
    for kept_idx, (j, _kind, _ji, _ep) in enumerate(kept):
        ref[local_to_global[kept_idx]] = ref_full[j]
    assert np.abs(v - ref).max() < 1e-14, np.abs(v - ref).max()
    # Partition of unity over the cell: the gap carries exactly V.
    assert abs(ref_full.sum() - 1.0) < 1e-14, ref_full.sum()
    # ... and (at d=2) it is genuinely NOT the point drive. At d=1 it is —
    # see `test_bspline_segment_gap_is_the_point_gap_on_the_tent_basis`.
    v_pt = BSplineSolver(**kw)._build_source_vector(geom, wk, wbg, n_bt, wi=0, s_f=s_f)
    if degree == 2:
        assert np.abs(v - v_pt).max() > 1e-3, np.abs(v - v_pt).max()


def test_bspline_segment_gap_is_the_point_gap_on_the_tent_basis():
    """A degeneracy worth pinning rather than tripping over (momwire#216): on
    d=1 with the feed at its cell's CENTRE, `feed_model` is an exact no-op.

    Each tent is linear on a knot span, and the cell average of a linear
    function is its value at the cell midpoint — so (1/Δ)∫_cell Φ_m = Φ_m(s_f)
    identically, to roundoff. The two feed models are only distinguishable by
    a basis with curvature inside the cell (d≥2) or by a feed that is not
    centred in it, both checked below.
    """
    # Centred feed (odd segment count puts the wire midpoint at a cell centre).
    kw = _segment_gap_kwargs(21, degree=1)
    sim = BSplineSolver(**kw, feed_model="segment")
    geom = sim._build_geometry()
    supp_seg, _p, _kcl, wk, wbg = sim._build_basis_polynomials(geom)
    n_bt = supp_seg.shape[0]
    s_f = kw["feed_arclength"]
    v_seg = sim._build_source_vector(geom, wk, wbg, n_bt, wi=0, s_f=s_f)
    v_pt = BSplineSolver(**kw)._build_source_vector(geom, wk, wbg, n_bt, wi=0, s_f=s_f)
    assert np.abs(v_seg - v_pt).max() < 1e-15, np.abs(v_seg - v_pt).max()

    # Off-centre in its cell: the drives separate, so this is the centred-feed
    # coincidence and not a dead option.
    h = float(geom["per_wire"][0]["h_per_seg"][0])
    off = s_f + 0.25 * h
    v_seg_off = sim._build_source_vector(geom, wk, wbg, n_bt, wi=0, s_f=off)
    v_pt_off = BSplineSolver(**kw)._build_source_vector(
        geom, wk, wbg, n_bt, wi=0, s_f=off
    )
    assert np.abs(v_seg_off - v_pt_off).max() > 0.2, np.abs(v_seg_off - v_pt_off).max()


@pytest.mark.parametrize("degree", [1, 2])
def test_bspline_segment_gap_reaches_the_y_path_and_multi_feed(degree):
    """All four solve paths share `_build_source_vector`, so the option is
    honored everywhere by construction; the two that a caller actually reaches
    (impedance and Y) are checked, on a two-feed model so multi-feed is covered
    too. Y stays reciprocal — the drive vector is still the readout vector.

    Port 1 sits off its cell's centre, so it separates the two feed models at
    both degrees; port 0 is centred, which at d=1 makes it a no-op (pinned by
    `test_bspline_segment_gap_is_the_point_gap_on_the_tent_basis`), so the
    all-entries check is d=2 only.
    """
    hd = 0.962 * 22 / 4
    kw = _segment_gap_kwargs(31, degree=degree)
    kw.pop("feed_wire_index")
    kw.pop("feed_arclength")
    kw["feeds"] = [(0, hd, 1 + 0j), (0, hd * 0.5, 0 + 0j)]
    Y_seg = BSplineSolver(**kw, feed_model="segment").compute_y_matrix()
    Y_pt = BSplineSolver(**kw).compute_y_matrix()
    assert Y_seg.shape == (2, 2)
    assert abs(Y_seg[0, 1] - Y_seg[1, 0]) < 1e-10 * abs(Y_seg).max()
    # The option reached the Y path — at the off-centre port under either
    # degree, and at every entry once the basis has curvature in the cell.
    moved = [
        (i, j)
        for i in range(2)
        for j in range(2)
        if abs(Y_seg[i, j] - Y_pt[i, j]) > 1e-6 * abs(Y_pt[i, j])
    ]
    assert (1, 1) in moved, (degree, Y_seg, Y_pt)
    if degree == 2:
        assert len(moved) == 4, (degree, moved)
    # ... and the impedance path likewise, on the same two-feed model.
    z_seg, _ = BSplineSolver(**kw, feed_model="segment").compute_impedance()
    z_pt, _ = BSplineSolver(**kw).compute_impedance()
    dz = abs(complex(np.atleast_1d(z_seg)[0]) - complex(np.atleast_1d(z_pt)[0]))
    if degree == 2:
        assert dz > 1e-6, dz
    else:
        assert dz < 1e-9, dz


def test_bspline_d2_hentenna_arbitrates_against_d1():
    """Degree-2 B-spline on the hentenna converges to the SAME value as the
    tent basis (within ~0.1 Ω), independently arbitrating the
    NEXT_STEPS items 13/14 question: the tent basis is NOT converged-to-the-
    wrong-place; NEC's three-term basis is the outlier that drifts super-log.
    (d=1 reproduces the retired TriangularSolver to roundoff on this
    knot-fed mesh — see tests/test_tent_parity.py.)

    Two independent basis families (degree-1 tent, degree-2 quadratic) land
    on the same impedance at the canonical n=21 hentenna sweep point.
    """
    C_LIGHT = 299_792_458.0
    freq_mhz = 28.47
    wavelength = C_LIGHT / (freq_mhz * 1e6)
    width_factor = 0.1378
    top_height_factor = 0.5081
    mid_height_factor = 0.1094
    eps_feed = 0.05
    half_w = wavelength * width_factor / 2
    z_mid = wavelength * (mid_height_factor - top_height_factor)
    z_bot = -wavelength * top_height_factor
    A = (0.0, half_w, 0.0)
    B_ = (0.0, half_w, z_mid)
    F = (0.0, half_w, z_bot)
    S = (0.0, eps_feed, z_mid)
    C_ = (0.0, -half_w, 0.0)
    D = (0.0, -half_w, z_mid)
    E_ = (0.0, -half_w, z_bot)
    T = (0.0, -eps_feed, z_mid)
    wires = [
        np.array([T, S], dtype=float),
        np.array([S, B_], dtype=float),
        np.array([B_, A, C_, D], dtype=float),
        np.array([T, D], dtype=float),
        np.array([D, E_, F, B_], dtype=float),
    ]
    junctions = [
        [(0, "end"), (1, "start")],
        [(0, "start"), (3, "start")],
        [(1, "end"), (2, "start"), (4, "end")],
        [(2, "end"), (3, "end"), (4, "start")],
    ]
    n = 21
    # tent and bspline both want EVEN nfeed (interior knot at z=0).
    nfeed = 2
    npe = [[nfeed], [n], [n, n, n], [n], [n, n, n]]
    z_tri, _ = BSplineSolver(
        degree=1,
        wires=wires,
        n_per_edge_per_wire=npe,
        feed_wire_index=0,
        feed_arclength=eps_feed,
        wavelength=wavelength,
        wire_radius=0.0005,
        nsegs=n,
        junctions=junctions,
    ).compute_impedance()
    z_b2, _ = BSplineSolver(
        degree=2,
        wires=wires,
        n_per_edge_per_wire=npe,
        feed_wire_index=0,
        feed_arclength=eps_feed,
        wavelength=wavelength,
        wire_radius=0.0005,
        nsegs=n,
        junctions=junctions,
    ).compute_impedance()
    # Tent basis (d=1, == retired TriangularSolver to roundoff) at n=21:
    # 43.158 + j38.027
    # B-spline d=2 at n=21: 43.066 + j38.849
    # The two converge to different small-N transients but agree at the
    # asymptote (~43.05 R, ~38.85 X at n=161) — they're independent
    # basis families that BOTH reject the NEC super-log drift.
    assert abs(z_tri.real - 43.16) < 0.1, f"tri R={z_tri.real}"
    assert abs(z_tri.imag - 38.03) < 0.1, f"tri X={z_tri.imag}"
    assert abs(z_b2.real - 43.07) < 0.1, f"bsp d=2 R={z_b2.real}"
    assert abs(z_b2.imag - 38.85) < 0.1, f"bsp d=2 X={z_b2.imag}"
    # Cross-basis disagreement bound. Post-PR-#51 the n=15..161 sweep
    # (scripts/compare_hentenna_solvers.py) pins the asymptote at 43.05 +
    # j38.84 for BOTH bases. At n=21 specifically tri and b2 differ by
    # ~0.09 Ω on R (b2 is essentially converged; tri still tightening
    # from its degree-1 O(1/N²) slope) and by ~0.82 Ω on X (where tri
    # has a larger small-N transient). Tighten R to 0.15 Ω; keep X at
    # 0.9 Ω — both leave ~15-30% headroom over the actual gap and would
    # fail loudly if either basis silently drifts off the arbitration
    # asymptote.
    assert abs(z_tri.real - z_b2.real) < 0.15, (
        f"basis disagreement on R: tri={z_tri.real}, bsp={z_b2.real}"
    )
    assert abs(z_tri.imag - z_b2.imag) < 0.9, (
        f"basis disagreement on X: tri={z_tri.imag}, bsp={z_b2.imag}"
    )


def test_bspline_d2_hentenna_singular_enrichment():
    """Singular basis enrichment at K≥3 junctions flips the hentenna R-rate
    from O(1/N) to ~O(1/N^(d+1)). Pin the converged R/X at n=81 and assert
    that the fitted convergence rate p in Z(N) = Z_inf + C/N^p satisfies
    p > 2.5 on R — both checks catch silent regressions.

    Reference values (productized C++ path, 2026-06):
      n=21  → 42.8574 + j44.2296
      n=41  → 43.0766 + j39.2185
      n=81  → 43.0858 + j38.9038
      n=161 → 43.0845 + j38.8749
    Rate fit on R over the four points gives p ≈ 2.74.
    """
    pytest.importorskip("momwire._accelerators")
    C_LIGHT = 299_792_458.0
    freq_mhz = 28.47
    wavelength = C_LIGHT / (freq_mhz * 1e6)
    width_factor = 0.1378
    top_height_factor = 0.5081
    mid_height_factor = 0.1094
    eps_feed = 0.05
    half_w = wavelength * width_factor / 2
    z_mid = wavelength * (mid_height_factor - top_height_factor)
    z_bot = -wavelength * top_height_factor
    A = (0.0, half_w, 0.0)
    B_ = (0.0, half_w, z_mid)
    F = (0.0, half_w, z_bot)
    S = (0.0, eps_feed, z_mid)
    C_ = (0.0, -half_w, 0.0)
    D = (0.0, -half_w, z_mid)
    E_ = (0.0, -half_w, z_bot)
    T = (0.0, -eps_feed, z_mid)
    wires = [
        np.array([T, S], dtype=float),
        np.array([S, B_], dtype=float),
        np.array([B_, A, C_, D], dtype=float),
        np.array([T, D], dtype=float),
        np.array([D, E_, F, B_], dtype=float),
    ]
    junctions = [
        [(0, "end"), (1, "start")],
        [(0, "start"), (3, "start")],
        [(1, "end"), (2, "start"), (4, "end")],
        [(2, "end"), (3, "end"), (4, "start")],
    ]

    nfeed = 3
    ns = [21, 41, 81]
    Rs = []
    Xs = []
    for n in ns:
        npe = [[nfeed], [n], [n, n, n], [n], [n, n, n]]
        z, _ = BSplineSolver(
            degree=2,
            wires=wires,
            n_per_edge_per_wire=npe,
            feed_wire_index=0,
            feed_arclength=eps_feed,
            wavelength=wavelength,
            wire_radius=0.0005,
            nsegs=n,
            junctions=junctions,
            use_singular_enrichment=True,
        ).compute_impedance()
        Rs.append(z.real)
        Xs.append(z.imag)

    # Pin n=81 to the recorded productized value (5e-3 Ω tolerance leaves
    # headroom for compiler/platform jitter but catches any constant-offset
    # drift much smaller than the prototype's value-spread to non-enriched).
    assert abs(Rs[2] - 43.0866) < 5e-3, f"R(n=81)={Rs[2]}, expected ≈43.0866"
    assert abs(Xs[2] - 38.8740) < 5e-3, f"X(n=81)={Xs[2]}, expected ≈38.8740"

    # Fit Z = Z_inf + C/N^p on the X component over n in {21, 41, 81}.
    # (X used to be R, but with the enrichment-orig sign fix the R
    # convergence is fast enough that R hits the few-mΩ noise floor
    # between N=41 and N=81, making the sign-based rate estimator
    # unreliable. X still has a few tens of mΩ of headroom at these N
    # and converges monotonically.) Three-point Richardson-style:
    #   p ≈ log( (X(N1) - X(N2)) / (X(N2) - X(N3)) ) / log(N2/N1)
    # with N1 < N2 < N3. The same leading constant cancels.
    dX_12 = Xs[0] - Xs[1]
    dX_23 = Xs[1] - Xs[2]
    assert dX_12 * dX_23 > 0, (
        f"X differences sign-flipped — noise floor reached too early; Xs={Xs}"
    )
    p = np.log(abs(dX_12 / dX_23)) / np.log(ns[1] / ns[0])
    assert p > 2.5, f"X convergence rate p={p:.2f} below the 2.5 floor (Xs={Xs})"


def test_bspline_d2_hentenna_enrichment_stable_variant():
    """Pin the stable-XFEM hentenna asymptote and verify that:
      (a) at n=81 it lands within ~0.005 Ω of the raw variant — the two
          variants must converge to the same Z, only with different
          small-N transients;
      (b) at d=1 the stable variant is bit-exact to raw — the BC-preserving
          polynomial bubble subspace is empty for d=1 (P_1 ∩ {p(0)=p(1)=0}
          = {0}), so projection coeffs are all zero and Φ_sing_stable
          identically equals Φ_sing. This pins the "d=1 enrichment is a
          no-op" symmetry between variants.

    Reference values (productized C++ path, this branch, n=81):
      stable: 43.0864 + j38.8757
      raw   : 43.0866 + j38.8740
      diff  : ~0.0002 Ω R, ~0.0017 Ω X
    """
    C_LIGHT = 299_792_458.0
    freq_mhz = 28.47
    wavelength = C_LIGHT / (freq_mhz * 1e6)
    width_factor = 0.1378
    top_height_factor = 0.5081
    mid_height_factor = 0.1094
    eps_feed = 0.05
    half_w = wavelength * width_factor / 2
    z_mid = wavelength * (mid_height_factor - top_height_factor)
    z_bot = -wavelength * top_height_factor
    A = (0.0, half_w, 0.0)
    B_ = (0.0, half_w, z_mid)
    F = (0.0, half_w, z_bot)
    S = (0.0, eps_feed, z_mid)
    C_ = (0.0, -half_w, 0.0)
    D = (0.0, -half_w, z_mid)
    E_ = (0.0, -half_w, z_bot)
    T = (0.0, -eps_feed, z_mid)
    wires = [
        np.array([T, S], dtype=float),
        np.array([S, B_], dtype=float),
        np.array([B_, A, C_, D], dtype=float),
        np.array([T, D], dtype=float),
        np.array([D, E_, F, B_], dtype=float),
    ]
    junctions = [
        [(0, "end"), (1, "start")],
        [(0, "start"), (3, "start")],
        [(1, "end"), (2, "start"), (4, "end")],
        [(2, "end"), (3, "end"), (4, "start")],
    ]
    n = 81
    npe = [[3], [n], [n, n, n], [n], [n, n, n]]
    common = dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        feed_wire_index=0,
        feed_arclength=eps_feed,
        wavelength=wavelength,
        wire_radius=0.0005,
        nsegs=n,
        junctions=junctions,
        use_singular_enrichment=True,
    )
    z_raw, _ = BSplineSolver(
        degree=2, enrichment_variant="raw", **common
    ).compute_impedance()
    z_stb, _ = BSplineSolver(
        degree=2, enrichment_variant="stable", **common
    ).compute_impedance()
    # (a) Both variants converge to the same Z; pin stable's value and the
    # raw-stable agreement so any future projection-coefficient drift fails
    # loudly. The tighter check is the raw-stable diff: 0.005 Ω covers
    # rounding noise but catches any sign / shape regression.
    assert abs(z_stb.real - 43.0864) < 5e-3
    assert abs(z_stb.imag - 38.8757) < 5e-3
    assert abs(z_raw - z_stb) < 5e-3

    # (b) At d=1 the bubble subspace is empty → stable = raw bit-exact.
    z1_raw, _ = BSplineSolver(
        degree=1, enrichment_variant="raw", **common
    ).compute_impedance()
    z1_stb, _ = BSplineSolver(
        degree=1, enrichment_variant="stable", **common
    ).compute_impedance()
    assert z1_raw == z1_stb


def test_bspline_d2_hentenna_enrichment_tikhonov_variant():
    """Pin the two limit cases of the tikhonov variant:
      (a) λ=0 reduces to raw bit-exact (the penalty disappears);
      (b) λ→∞ reduces to use_singular_enrichment=False (penalty
          dominates and α_enr → 0).

    These limits are what makes tikhonov a knob rather than a separate
    variant; if either limit drifts, the dimensionless scaling (λ·s with
    s = mean |diag(Z_ee)|) is wrong and the knob loses its meaning.
    """
    C_LIGHT = 299_792_458.0
    freq_mhz = 28.47
    wavelength = C_LIGHT / (freq_mhz * 1e6)
    width_factor = 0.1378
    top_height_factor = 0.5081
    mid_height_factor = 0.1094
    eps_feed = 0.05
    half_w = wavelength * width_factor / 2
    z_mid = wavelength * (mid_height_factor - top_height_factor)
    z_bot = -wavelength * top_height_factor
    A = (0.0, half_w, 0.0)
    B_ = (0.0, half_w, z_mid)
    F = (0.0, half_w, z_bot)
    S = (0.0, eps_feed, z_mid)
    C_ = (0.0, -half_w, 0.0)
    D = (0.0, -half_w, z_mid)
    E_ = (0.0, -half_w, z_bot)
    T = (0.0, -eps_feed, z_mid)
    wires = [
        np.array([T, S], dtype=float),
        np.array([S, B_], dtype=float),
        np.array([B_, A, C_, D], dtype=float),
        np.array([T, D], dtype=float),
        np.array([D, E_, F, B_], dtype=float),
    ]
    junctions = [
        [(0, "end"), (1, "start")],
        [(0, "start"), (3, "start")],
        [(1, "end"), (2, "start"), (4, "end")],
        [(2, "end"), (3, "end"), (4, "start")],
    ]
    n = 21  # small enough to see the tikhonov effect; UI default
    npe = [[3], [n], [n, n, n], [n], [n, n, n]]
    common = dict(
        degree=2,
        wires=wires,
        n_per_edge_per_wire=npe,
        feed_wire_index=0,
        feed_arclength=eps_feed,
        wavelength=wavelength,
        wire_radius=0.0005,
        nsegs=n,
        junctions=junctions,
    )
    z_raw, _ = BSplineSolver(
        **common, use_singular_enrichment=True, enrichment_variant="raw"
    ).compute_impedance()
    z_tik_zero, _ = BSplineSolver(
        **common,
        use_singular_enrichment=True,
        enrichment_variant="tikhonov",
        tikhonov_lambda=0.0,
    ).compute_impedance()
    z_off, _ = BSplineSolver(
        **common, use_singular_enrichment=False
    ).compute_impedance()
    z_tik_big, _ = BSplineSolver(
        **common,
        use_singular_enrichment=True,
        enrichment_variant="tikhonov",
        tikhonov_lambda=1e6,
    ).compute_impedance()
    # (a) λ=0 → raw bit-exact
    assert z_raw == z_tik_zero
    # (b) λ→∞ → no-enrichment to ~1e-6 relative
    assert abs(z_tik_big - z_off) / abs(z_off) < 1e-6


def test_bspline_enrichment_auto_two_pass_selects_correctly():
    """The 'auto' variant runs a two-pass solve: pass 1 without enrichment
    measures tap_ratio = min(|I_wire|)/max(|I_wire|) at each K≥enrichment_min_k
    junction; pass 2 applies raw enrichment only at junctions where
    tap_ratio > auto_tap_ratio_threshold.

    Two canonical geometries pin the per-junction decision:
      (a) **Hentenna** — dominant-pair K=3 (tap_ratio ≈ 0.16): auto must
          select no junctions and the result must equal no-enrichment
          bit-exact (the +0.25 Ω X small-N transient that raw introduces
          is the regression this is meant to prevent).
      (b) **Y-fixture** — balanced 3-way K=3 (tap_ratio ≈ 0.50): auto
          must select the K=3 junction (index 1; the K=2 at index 0 is
          correctly excluded by enrichment_min_k=3) and the result must
          equal raw bit-exact (preserves the legitimate cusp).
    """
    C_LIGHT = 299_792_458.0
    freq_mhz = 28.47
    wavelength = C_LIGHT / (freq_mhz * 1e6)
    eps_feed = 0.05

    # --- (a) Hentenna ---
    width_factor = 0.1378
    top_height_factor = 0.5081
    mid_height_factor = 0.1094
    half_w = wavelength * width_factor / 2
    z_mid = wavelength * (mid_height_factor - top_height_factor)
    z_bot = -wavelength * top_height_factor
    A = (0.0, half_w, 0.0)
    B_ = (0.0, half_w, z_mid)
    F = (0.0, half_w, z_bot)
    S = (0.0, eps_feed, z_mid)
    C_ = (0.0, -half_w, 0.0)
    D = (0.0, -half_w, z_mid)
    E_ = (0.0, -half_w, z_bot)
    T = (0.0, -eps_feed, z_mid)
    h_wires = [
        np.array([T, S], dtype=float),
        np.array([S, B_], dtype=float),
        np.array([B_, A, C_, D], dtype=float),
        np.array([T, D], dtype=float),
        np.array([D, E_, F, B_], dtype=float),
    ]
    h_juncs = [
        [(0, "end"), (1, "start")],
        [(0, "start"), (3, "start")],
        [(1, "end"), (2, "start"), (4, "end")],
        [(2, "end"), (3, "end"), (4, "start")],
    ]
    n = 21
    h_kw = dict(
        degree=2,
        wires=h_wires,
        n_per_edge_per_wire=[[3], [n], [n, n, n], [n], [n, n, n]],
        feed_wire_index=0,
        feed_arclength=eps_feed,
        wavelength=wavelength,
        wire_radius=0.0005,
        nsegs=n,
        junctions=h_juncs,
    )
    z_h_off, _ = BSplineSolver(
        **h_kw, use_singular_enrichment=False
    ).compute_impedance()
    sim_h_auto = BSplineSolver(
        **h_kw, use_singular_enrichment=True, enrichment_variant="auto"
    )
    z_h_auto, _ = sim_h_auto.compute_impedance()
    assert sim_h_auto._auto_active_junctions == []
    assert z_h_auto == z_h_off

    # --- (b) Y-fixture ---
    L = wavelength / 4.0
    T_ = (-eps_feed, 0.0, 0.0)
    S_ = (+eps_feed, 0.0, 0.0)
    a1 = (T_[0] - L, 0.0, 0.0)
    c60 = float(np.cos(np.pi / 3.0))
    s60 = float(np.sin(np.pi / 3.0))
    a2 = (S_[0] + L * c60, +L * s60, 0.0)
    a3 = (S_[0] + L * c60, -L * s60, 0.0)
    y_wires = [
        np.array([T_, S_], dtype=float),
        np.array([T_, a1], dtype=float),
        np.array([S_, a2], dtype=float),
        np.array([S_, a3], dtype=float),
    ]
    y_juncs = [
        [(0, "start"), (1, "start")],  # K=2 at T (skipped by enrichment_min_k=3)
        [(0, "end"), (2, "start"), (3, "start")],  # K=3 at S (the probe junction)
    ]
    n_y = 41
    y_kw = dict(
        degree=2,
        wires=y_wires,
        n_per_edge_per_wire=[[2], [n_y], [n_y], [n_y]],
        feed_wire_index=0,
        feed_arclength=eps_feed,
        wavelength=wavelength,
        wire_radius=0.0005,
        nsegs=n_y,
        junctions=y_juncs,
        enrichment_min_k=3,
    )
    z_y_raw, _ = BSplineSolver(
        **y_kw, use_singular_enrichment=True, enrichment_variant="raw"
    ).compute_impedance()
    sim_y_auto = BSplineSolver(
        **y_kw, use_singular_enrichment=True, enrichment_variant="auto"
    )
    z_y_auto, _ = sim_y_auto.compute_impedance()
    assert sim_y_auto._auto_active_junctions == [1]
    assert z_y_auto == z_y_raw


def test_bspline_assemble_z_enrich_cpp_matches_numpy():
    """The C++ `assemble_Z_enrich` accelerator must agree with the
    pure-numpy reference across all four enrichment variants. The
    reference path is what runs on platforms where the Pybind11 extension
    isn't built (Windows — setup.py skips it because the GCC-only
    `-fopenmp` / `-mavx2` / `-lmvec` flags don't link under MSVC).

    Sweep:
      * Two geometries — hentenna (small-tap K=3, 2 enrichable junctions)
        and Y-fixture (balanced 3-way K=3, 1 enrichable junction). Together
        they exercise the multi-junction Z_ee block, the L-R symmetric
        enrichment pair (hentenna), and the auto-variant's per-junction
        filter (Y-fixture's K=2 junction must be excluded).
      * All four variants — raw / stable / tikhonov / auto. raw and stable
        differ in the kernel's `proj_coeffs` argument; tikhonov uses the
        raw kernel call but adds λ·I in Python; auto routes through raw
        with a junction filter and may take the pass-1-only path (which
        skips the kernel entirely — that path agrees trivially but is
        still worth exercising as a "no kernel call" case).

    1e-12 relative is well below GL quadrature precision and lets the
    two implementations differ only in floating-point reduction order
    (numpy's pairwise vs C++'s sequential).
    """
    import momwire.bspline as bmod

    # If the C++ extension wasn't built (Windows wheel, or local build
    # without pybind11), there's no C++ kernel to compare against. The
    # numpy reference is the only path; the parity sweep below would
    # NameError trying to call into the missing _acc.
    if not bmod._HAVE_ENRICH_ACCEL:
        pytest.skip("C++ assemble_Z_enrich not available — numpy-only build")

    C_LIGHT = 299_792_458.0
    freq_mhz = 28.47
    wavelength = C_LIGHT / (freq_mhz * 1e6)
    eps_feed = 0.05

    # Hentenna
    width_factor = 0.1378
    top_height_factor = 0.5081
    mid_height_factor = 0.1094
    half_w = wavelength * width_factor / 2
    z_mid = wavelength * (mid_height_factor - top_height_factor)
    z_bot = -wavelength * top_height_factor
    A = (0.0, half_w, 0.0)
    B_ = (0.0, half_w, z_mid)
    F = (0.0, half_w, z_bot)
    S = (0.0, eps_feed, z_mid)
    C_ = (0.0, -half_w, 0.0)
    D = (0.0, -half_w, z_mid)
    E_ = (0.0, -half_w, z_bot)
    T = (0.0, -eps_feed, z_mid)
    h_kw = dict(
        degree=2,
        wires=[
            np.array([T, S], dtype=float),
            np.array([S, B_], dtype=float),
            np.array([B_, A, C_, D], dtype=float),
            np.array([T, D], dtype=float),
            np.array([D, E_, F, B_], dtype=float),
        ],
        n_per_edge_per_wire=[[3], [21], [21, 21, 21], [21], [21, 21, 21]],
        feed_wire_index=0,
        feed_arclength=eps_feed,
        wavelength=wavelength,
        wire_radius=0.0005,
        nsegs=21,
        junctions=[
            [(0, "end"), (1, "start")],
            [(0, "start"), (3, "start")],
            [(1, "end"), (2, "start"), (4, "end")],
            [(2, "end"), (3, "end"), (4, "start")],
        ],
    )

    # Y-fixture
    L = wavelength / 4.0
    T_ = (-eps_feed, 0.0, 0.0)
    S_ = (+eps_feed, 0.0, 0.0)
    a1 = (T_[0] - L, 0.0, 0.0)
    c60 = float(np.cos(np.pi / 3.0))
    s60 = float(np.sin(np.pi / 3.0))
    a2 = (S_[0] + L * c60, +L * s60, 0.0)
    a3 = (S_[0] + L * c60, -L * s60, 0.0)
    y_kw = dict(
        degree=2,
        wires=[
            np.array([T_, S_], dtype=float),
            np.array([T_, a1], dtype=float),
            np.array([S_, a2], dtype=float),
            np.array([S_, a3], dtype=float),
        ],
        n_per_edge_per_wire=[[2], [41], [41], [41]],
        feed_wire_index=0,
        feed_arclength=eps_feed,
        wavelength=wavelength,
        wire_radius=0.0005,
        nsegs=41,
        junctions=[
            [(0, "start"), (1, "start")],
            [(0, "end"), (2, "start"), (3, "start")],
        ],
        enrichment_min_k=3,
    )

    def run(kw, variant):
        z, _ = BSplineSolver(
            **kw,
            use_singular_enrichment=True,
            enrichment_variant=variant,
            tikhonov_lambda=0.1,
        ).compute_impedance()
        return z

    for label, kw in [("hentenna", h_kw), ("y-fixture", y_kw)]:
        for variant in ("raw", "stable", "tikhonov", "auto"):
            saved = bmod._HAVE_ENRICH_ACCEL
            try:
                bmod._HAVE_ENRICH_ACCEL = True
                z_cpp = run(kw, variant)
                bmod._HAVE_ENRICH_ACCEL = False
                z_np = run(kw, variant)
            finally:
                bmod._HAVE_ENRICH_ACCEL = saved
            rel = abs(z_cpp - z_np) / abs(z_cpp)
            assert rel < 1e-12, (
                f"{label} variant={variant}: C++ vs numpy enrich kernel "
                f"disagreement rel={rel:.2e} (cpp={z_cpp}, np={z_np})"
            )


def test_bspline_enrichment_assemble_holds_no_n_squared_tangent_table():
    """`_enrichment_Z_assemble` must not build the free-space `(N, N)`
    tangent-dot table (issue #334). `assemble_Z_enrich` (the C++ kernel and
    its numpy twin) only ever reads that table at the handful of
    (spec_seg[e], n) pairs the enrichment DOFs touch — a table scaling as
    N² was pure transient, alive even when `ground_z is None`, and rebuilt
    per k in an enrichment sweep.

    Arithmetic for the threshold, at the N = 6,413 segments this scaled-up
    hentenna geometry builds (tracemalloc sees numpy's data allocations;
    the accelerator's own C++-side buffers are outside its reach, so this
    measures the Python-side transient that used to be `tangents @
    tangents.T`):

      * the retired `td_all` table alone was `8 * N**2` = 328.9 MB.
      * measured peak here (this test, tracemalloc): ~1.2 MB — the (N, 3)
        tangent table passed straight through, plus the small enrichment-
        side (n_enrich, n_qp) precompute arrays.

    10 MB therefore sits ~8x above the measured peak and ~33x below the
    retired N² table.
    """
    import tracemalloc

    C_LIGHT = 299_792_458.0
    freq_mhz = 28.47
    wavelength = C_LIGHT / (freq_mhz * 1e6)
    eps_feed = 0.05
    width_factor = 0.1378
    top_height_factor = 0.5081
    mid_height_factor = 0.1094
    half_w = wavelength * width_factor / 2
    z_mid = wavelength * (mid_height_factor - top_height_factor)
    z_bot = -wavelength * top_height_factor
    A = (0.0, half_w, 0.0)
    B_ = (0.0, half_w, z_mid)
    F = (0.0, half_w, z_bot)
    S = (0.0, eps_feed, z_mid)
    C_ = (0.0, -half_w, 0.0)
    D = (0.0, -half_w, z_mid)
    E_ = (0.0, -half_w, z_bot)
    T = (0.0, -eps_feed, z_mid)

    n = 800  # per heavily-subdivided edge — N ~ 6,400 total
    sim = BSplineSolver(
        degree=2,
        wires=[
            np.array([T, S], dtype=float),
            np.array([S, B_], dtype=float),
            np.array([B_, A, C_, D], dtype=float),
            np.array([T, D], dtype=float),
            np.array([D, E_, F, B_], dtype=float),
        ],
        n_per_edge_per_wire=[[3], [n], [n, n, n], [n], [n, n, n]],
        feed_wire_index=0,
        feed_arclength=eps_feed,
        wavelength=wavelength,
        wire_radius=0.0005,
        nsegs=n,
        junctions=[
            [(0, "end"), (1, "start")],
            [(0, "start"), (3, "start")],
            [(1, "end"), (2, "start"), (4, "end")],
            [(2, "end"), (3, "end"), (4, "start")],
        ],
        use_singular_enrichment=True,
        enrichment_variant="raw",
    )
    geom = sim._build_geometry()
    supp_seg, polys, _kcl_A, _wire_knots, _wire_basis_global = (
        sim._build_basis_polynomials(geom)
    )
    n_segs = geom["n_segs_total"]

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        enrich = sim._enrichment_Z_assemble(geom, supp_seg, polys)
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert enrich is not None and enrich["n_enrich"] > 0
    assert peak < 10_000_000, (
        f"_enrichment_Z_assemble peaked {peak / 1e6:.2f} MB at N={n_segs} "
        f"(8 * N**2 = {8 * n_segs**2 / 1e6:.1f} MB) — the free-space "
        "tangent-dot table is dense again"
    )


def test_bspline_hentenna_enrichment_left_right_symmetry():
    """Hentenna is mirror-symmetric about y=0, so the BSpline+enrichment solve
    must produce mirror-symmetric per-knot currents on the upper and lower
    polylines. A sign bug in the enrichment-basis derivative — where
    `dΦ_sing/du_arc` was computed without the chain-rule sign flip for
    "end"-orientation bases (junction at the segment's right endpoint) —
    used to break this exact symmetry: junction-D current magnitudes were
    several percent off from their junction-B mirrors, with the residual
    only converging to zero as N→∞. Caught here at modest N so any
    re-introduction of the bug fails loudly.
    """
    pytest.importorskip("momwire._accelerators")
    design_freq_mhz = 28.47
    C_LIGHT = 299_792_458.0
    wavelength = C_LIGHT / (design_freq_mhz * 1e6)
    width_factor = 0.1378
    top_height_factor = 0.5081
    mid_height_factor = 0.1094
    half_w = wavelength * width_factor / 2.0
    eps_feed = 0.05
    z_mid = wavelength * (mid_height_factor - top_height_factor)
    z_bot = -wavelength * top_height_factor
    A = (0.0, half_w, 0.0)
    B_ = (0.0, half_w, z_mid)
    F = (0.0, half_w, z_bot)
    S = (0.0, eps_feed, z_mid)
    C_ = (0.0, -half_w, 0.0)
    D = (0.0, -half_w, z_mid)
    E_ = (0.0, -half_w, z_bot)
    T = (0.0, -eps_feed, z_mid)
    wires = [
        np.array([T, S], dtype=float),
        np.array([S, B_], dtype=float),
        np.array([B_, A, C_, D], dtype=float),
        np.array([T, D], dtype=float),
        np.array([D, E_, F, B_], dtype=float),
    ]
    junctions = [
        [(0, "end"), (1, "start")],
        [(0, "start"), (3, "start")],
        [(1, "end"), (2, "start"), (4, "end")],
        [(2, "end"), (3, "end"), (4, "start")],
    ]

    n = 21
    npe = [[4], [n], [n, n, n], [n], [n, n, n]]
    sim = BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=npe,
        feed_wire_index=0,
        feed_arclength=eps_feed,
        wavelength=wavelength,
        wire_radius=0.0005,
        nsegs=n,
        degree=2,
        junctions=junctions,
        use_singular_enrichment=True,
    )
    _, coeffs = sim.compute_impedance()
    currents = sim.currents_at_knots(coeffs)

    # The K=3 junction directional polynomial bases sit on the polyline
    # endpoint knots of `upper` and `lower`. By antenna L-R mirror, the
    # current magnitudes at D (left) must equal those at B (right) to
    # machine precision: ratio == 1 within ~1e-6.
    upper = currents[2]
    lower = currents[4]
    assert abs(abs(upper[0]) - abs(upper[-1])) / abs(upper[0]) < 1e-6, (
        f"upper |I| at B={abs(upper[0]):.6f} vs D={abs(upper[-1]):.6f} "
        "(L/R mirror should be exact)"
    )
    assert abs(abs(lower[0]) - abs(lower[-1])) / abs(lower[0]) < 1e-6, (
        f"lower |I| at D={abs(lower[0]):.6f} vs B={abs(lower[-1]):.6f} "
        "(L/R mirror should be exact)"
    )

    # Interior polynomial bases on `upper` near the two K=3 junctions
    # should also mirror. The pre-fix bug surfaced as ~5% asymmetry on the
    # interior basis adjacent to the directional one; cap at 0.01% here
    # so any re-introduction of the orig=1 sign flip fails immediately.
    for k in range(1, 6):
        ratio = abs(upper[k]) / abs(upper[-1 - k])
        assert abs(ratio - 1.0) < 1e-4, (
            f"upper knot pair (k={k}, n-1-{k}) ratio={ratio:.6f} — L/R asymmetry"
        )


def test_sinusoidal_field_tensor_cpp_matches_numpy():
    """The C++ `sinusoidal_field_tensor` accelerator and the pure-numpy
    reference path must produce bit-equivalent (Phi_const, Phi_sin, Phi_cos)
    on representative geometries. Anything looser than ~1e-13 relative
    indicates the C++ kernel diverged from the formula in sinusoidal.py.
    """
    import momwire.sinusoidal as sin_mod

    if not sin_mod._HAVE_FIELD_TENSOR:
        pytest.skip("C++ accelerator not built")

    # Bent two-edge polyline so the (m, n) loop sees varying tangents.
    wires = [np.array([[0.0, 0.0, -5.0], [0.5, 0.0, 0.0], [0.0, 0.0, 5.0]])]
    for n in (15, 41):
        sim = SinusoidalSolver(
            wires=wires,
            n_per_edge_per_wire=[[n, n]],
            wavelength=22.0,
            wire_radius=5e-4,
            nsegs=n,
        )
        geom = sim._build_geometry()
        # C++ path
        Pc_cpp, Ps_cpp, Pco_cpp = sim._field_tensor(geom, sim.k)
        # Numpy reference path
        sin_mod._HAVE_FIELD_TENSOR = False
        try:
            Pc_np, Ps_np, Pco_np = sim._field_tensor(geom, sim.k)
        finally:
            sin_mod._HAVE_FIELD_TENSOR = True
        for name, a_cpp, a_np in [
            ("Phi_const", Pc_cpp, Pc_np),
            ("Phi_sin", Ps_cpp, Ps_np),
            ("Phi_cos", Pco_cpp, Pco_np),
        ]:
            denom = max(np.max(np.abs(a_np)), 1e-30)
            rel = np.max(np.abs(a_cpp - a_np)) / denom
            assert rel < 1e-13, f"n={n} {name} max rel diff = {rel:.3e}"


def test_sinusoidal_hentenna_left_right_symmetry():
    """Hentenna is mirror-symmetric about y=0, so SinusoidalSolver's per-knot
    currents on the cross-bar halves (wires 1 and 3) and on the upper /
    lower rectangles (wires 2 and 4) must be mirror-symmetric to machine
    precision.

    Pre-fix bug: `currents_at_knots` multiplied the whole (A + B·sin +
    C·cos) basis evaluation by σ instead of using the (σA, B, σC) effective
    coefficients that `_assemble_Z` already uses. At σ=−1 junction
    neighbours (K=2 junctions where both wires "start" or both "end" at
    the node, plus the σ=−1 entries at K=3 junctions) this added a
    spurious 2·B·sin(k·s) term, surfacing as asymmetric kinks at the
    junction-adjacent knots. Caught here at modest N so any re-introduction
    fails loudly.
    """
    C_LIGHT = 299_792_458.0
    freq_mhz = 28.47
    wavelength = C_LIGHT / (freq_mhz * 1e6)
    width_factor = 0.1378
    top_height_factor = 0.5081
    mid_height_factor = 0.1094
    eps = 0.05
    half_w = wavelength * width_factor / 2
    z_mid = wavelength * (mid_height_factor - top_height_factor)
    z_bot = -wavelength * top_height_factor
    A = (0.0, half_w, 0.0)
    B_ = (0.0, half_w, z_mid)
    F = (0.0, half_w, z_bot)
    S = (0.0, eps, z_mid)
    C_ = (0.0, -half_w, 0.0)
    D = (0.0, -half_w, z_mid)
    E_ = (0.0, -half_w, z_bot)
    T = (0.0, -eps, z_mid)
    wires = [
        np.array([T, S], dtype=float),
        np.array([S, B_], dtype=float),
        np.array([B_, A, C_, D], dtype=float),
        np.array([T, D], dtype=float),
        np.array([D, E_, F, B_], dtype=float),
    ]
    junctions = [
        [(0, "end"), (1, "start")],
        [(0, "start"), (3, "start")],
        [(1, "end"), (2, "start"), (4, "end")],
        [(2, "end"), (3, "end"), (4, "start")],
    ]
    n = 21
    sim = SinusoidalSolver(
        wires=wires,
        n_per_edge_per_wire=[[3], [n], [n, n, n], [n], [n, n, n]],
        feed_wire_index=0,
        feed_arclength=eps,
        wavelength=wavelength,
        wire_radius=0.0005,
        nsegs=n,
        junctions=junctions,
    )
    _, alpha = sim.compute_impedance()
    knots = sim.currents_at_knots(alpha)

    # Cross-bar halves wire 1 (S→B) vs wire 3 (T→D): same arc-direction
    # mirror, so |I_w1[i]| should equal |I_w3[i]| for all i.
    w1 = np.abs(knots[1])
    w3 = np.abs(knots[3])
    max_dev = float(np.max(np.abs(w1 - w3)))
    assert max_dev < 1e-12, (
        f"cross-bar halves max |Δ|I|| = {max_dev:.2e} — L/R asymmetry"
    )

    # Upper rectangle wire 2 (B→A→C→D) is its own mirror traversed
    # backwards: |I_w2[i]| should equal |I_w2[-1-i]| for all i.
    w2 = np.abs(knots[2])
    max_dev = float(np.max(np.abs(w2 - w2[::-1])))
    assert max_dev < 1e-12, (
        f"upper rectangle max self-reversed |Δ|I|| = {max_dev:.2e} — L/R asymmetry"
    )

    # Lower rectangle wire 4 (D→E→F→B): same self-mirror story.
    w4 = np.abs(knots[4])
    max_dev = float(np.max(np.abs(w4 - w4[::-1])))
    assert max_dev < 1e-12, (
        f"lower rectangle max self-reversed |Δ|I|| = {max_dev:.2e} — L/R asymmetry"
    )


@pytest.mark.parametrize(
    "model_cls,kwargs",
    [
        (SinusoidalSolver, {}),
        (BSplineSolver, {"degree": 1}),
        (BSplineSolver, {"degree": 2}),
    ],
)
def test_currents_at_knots_s_array_matches_default_at_knots(model_cls, kwargs):
    """`currents_at_knots(coeffs, s_array=[knot_arcs_per_wire])` must agree
    bit-for-bit with the default `currents_at_knots(coeffs)` at the mesh
    knots, for every basis family. Guards against future basis-evaluation
    drift between the two paths.
    """
    wires = [np.array([[0.0, 0.0, -0.24], [0.0, 0.0, 0.24]])]
    nsegs = 21
    sim = model_cls(
        wires=wires,
        n_per_edge_per_wire=[[nsegs]],
        wavelength=1.0,
        wire_radius=1e-3,
        nsegs=nsegs,
        **kwargs,
    )
    _, coeffs = sim.compute_impedance()
    knot_default = sim.currents_at_knots(coeffs)

    geom = sim._build_geometry()
    if isinstance(sim, SinusoidalSolver):
        first = geom["wire_first"][0]
        last = geom["wire_last"][0]
        wire_h = geom["seg_h"][first : last + 1]
        arc_at_knot = np.concatenate([[0.0], np.cumsum(wire_h)])
    else:
        arc_at_knot = geom["per_wire"][0]["arc_at_knot"]

    knot_via_s = sim.currents_at_knots(coeffs, s_array=[arc_at_knot])
    np.testing.assert_allclose(knot_via_s[0], knot_default[0], rtol=0, atol=1e-12)

    # Sampling at quarter-points should pass through the knot values exactly
    # at every even-indexed sample (where samples = knots interleaved with
    # midpoints).
    mid_arc = 0.5 * (arc_at_knot[:-1] + arc_at_knot[1:])
    sample_arc = np.empty(2 * mid_arc.shape[0] + 1)
    sample_arc[0::2] = arc_at_knot
    sample_arc[1::2] = mid_arc
    sample_I = sim.currents_at_knots(coeffs, s_array=[sample_arc])[0]
    np.testing.assert_allclose(sample_I[0::2], knot_default[0], rtol=0, atol=1e-12)
    # Mid-segment samples should be finite and on roughly the right scale.
    assert np.isfinite(sample_I).all()


@pytest.mark.parametrize("nsegs", [21, 41, 101])
def test_sinusoidal_dipole_matches_nec2(nsegs):
    """SinusoidalSolver implements NEC2's three-term basis (Eqs 43-64 of the
    LLNL theory manual). On a straight dipole it should match PyNEC/nec2c
    to <0.1 Ohm — the only differences are floating-point and quadrature.
    """
    wires = [np.array([[0.0, 0.0, -5.291], [0.0, 0.0, 5.291]], dtype=float)]
    sim = SinusoidalSolver(
        wires=wires,
        n_per_edge_per_wire=[[nsegs]],
        wavelength=22.0,
        wire_radius=0.0005,
        nsegs=nsegs,
    )
    z, _ = sim.compute_impedance()
    # NEC2 reference at this geometry (docs/convergence_analysis.md):
    # 69.69 - j18.67 at N=21, 69.64 - j18.21 at N=101.
    assert 69.5 < z.real < 69.8, f"R={z.real}"
    assert -19.0 < z.imag < -17.5, f"X={z.imag}"


def test_sinusoidal_hentenna_reproduces_pynec():
    """Sinusoidal-basis momwire reproduces PyNEC's hentenna numbers to
    ~0.05 Ohm. Validates the K=2/K=3 junction-basis path against the
    NEXT_STEPS.md item 13 PyNEC reference.
    """
    C_LIGHT = 299_792_458.0
    freq_mhz = 28.47
    wavelength = C_LIGHT / (freq_mhz * 1e6)
    width_factor = 0.1378
    top_height_factor = 0.5081
    mid_height_factor = 0.1094
    eps = 0.05
    half_w = wavelength * width_factor / 2
    z_mid = wavelength * (mid_height_factor - top_height_factor)
    z_bot = -wavelength * top_height_factor
    A = (0.0, half_w, 0.0)
    B_ = (0.0, half_w, z_mid)
    F = (0.0, half_w, z_bot)
    S = (0.0, eps, z_mid)
    C_ = (0.0, -half_w, 0.0)
    D = (0.0, -half_w, z_mid)
    E_ = (0.0, -half_w, z_bot)
    T = (0.0, -eps, z_mid)
    wires = [
        np.array([T, S], dtype=float),
        np.array([S, B_], dtype=float),
        np.array([B_, A, C_, D], dtype=float),
        np.array([T, D], dtype=float),
        np.array([D, E_, F, B_], dtype=float),
    ]
    junctions = [
        [(0, "end"), (1, "start")],
        [(0, "start"), (3, "start")],
        [(1, "end"), (2, "start"), (4, "end")],
        [(2, "end"), (3, "end"), (4, "start")],
    ]
    n = 21
    # Sinusoidal basis: ODD n_feed parity → delta-gap segment centred at z=0.
    nfeed = 3
    sim = SinusoidalSolver(
        wires=wires,
        n_per_edge_per_wire=[[nfeed], [n], [n, n, n], [n], [n, n, n]],
        feed_wire_index=0,
        feed_arclength=eps,
        wavelength=wavelength,
        wire_radius=0.0005,
        nsegs=n,
        junctions=junctions,
    )
    z, _ = sim.compute_impedance()
    # PyNEC reference at n=21 (NEXT_STEPS.md item 13): 45.604 - j4.604.
    assert abs(z.real - 45.604) < 0.1, f"R={z.real}"
    assert abs(z.imag - (-4.604)) < 0.1, f"X={z.imag}"


def _fandipole_two_band_sim(N, wavelength, solver_cls=BSplineSolver, **solver_kwargs):
    """Helper: build the same K=3 two-band fan dipole used in the smoke test.
    Returned simulator has junctions=[S, T] each connecting 3 wire ends.

    `solver_cls`/`solver_kwargs` let callers build the same geometry with a
    different basis (e.g. BSplineSolver(degree=...)).
    """
    import math

    band_lengths = [10.2551, 5.2691]
    slope = 0.5
    cone_radius = 0.12
    t0 = cone_radius * math.sqrt(2.0)
    eps = 0.01
    Zc = 1.0 / math.sqrt(1.0 + slope**2)
    Zs = slope * Zc
    S = (0.0, eps, 0.0)
    T = (0.0, -eps, 0.0)
    C = (S[0], S[1] + t0 * Zc, S[2] - t0 * Zs)
    lst = [
        (math.cos(math.pi * i / 180), math.sin(math.pi * i / 180))
        for i in range(36, 360, 72)
    ][:2]
    A_pos = [
        (
            C[0] + cone_radius * x,
            C[1] + cone_radius * y * Zs,
            C[2] + cone_radius * y * Zc,
        )
        for (x, y) in lst
    ]
    ls = [
        band_lengths[i] / 2 - math.sqrt(sum((s - a) ** 2 for s, a in zip(S, A_pos[i])))
        for i in range(2)
    ]
    B_pos = [(a[0], a[1] + l * Zc, a[2] - l * Zs) for l, a in zip(ls, A_pos)]
    A_neg = [(a[0], -a[1], a[2]) for a in A_pos]
    B_neg = [(b[0], -b[1], b[2]) for b in B_pos]

    wires = [np.array([T, S], dtype=float)]
    n_per_edge = [[2]]
    for i in range(2):
        wires.append(np.array([S, A_pos[i], B_pos[i]], dtype=float))
        n_per_edge.append([N, N])
    for i in range(2):
        wires.append(np.array([T, A_neg[i], B_neg[i]], dtype=float))
        n_per_edge.append([N, N])
    junctions = [
        [(0, "end"), (1, "start"), (2, "start")],
        [(0, "start"), (3, "start"), (4, "start")],
    ]
    return solver_cls(
        wires=wires,
        n_per_edge_per_wire=n_per_edge,
        feed_wire_index=0,
        feed_arclength=eps,
        wavelength=wavelength,
        nsegs=N,
        wire_radius=0.0005,
        junctions=junctions,
        **solver_kwargs,
    )


@pytest.mark.parametrize("degree", [1, 2])
def test_bspline_fandipole_swept_matches_per_freq(degree):
    """BSpline batched swept-k path (hoisted static + reg-geometry, reg
    moments batched over k) must agree with per-frequency solves to
    roundoff on a K=3-junction fan dipole. Guards the same-edge swept
    optimization against the untouched single-k path.
    """
    C_LIGHT = 299_792_458.0
    freqs_mhz = np.array([12.0, 14.3, 21.0, 28.47])
    k_array = 2 * np.pi * freqs_mhz * 1e6 / C_LIGHT
    sim_sweep = _fandipole_two_band_sim(
        N=11, wavelength=22.0, solver_cls=BSplineSolver, degree=degree
    )
    z_swept = sim_sweep.compute_impedance_swept(k_array)
    for f, zs in zip(freqs_mhz, z_swept):
        sim_f = _fandipole_two_band_sim(
            N=11,
            wavelength=C_LIGHT / (f * 1e6),
            solver_cls=BSplineSolver,
            degree=degree,
        )
        z_f, _ = sim_f.compute_impedance()
        # Batched-over-k reg einsum changes the floating-point reduction order,
        # so this is roundoff-equal (~1e-12 relative), not bit-identical.
        assert abs(zs - z_f) <= 1e-9 * abs(z_f), (
            f"d={degree}, f={f} MHz: swept={zs}, single={z_f}"
        )


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("ground_z", [None, 0.0])
def test_bspline_swept_fully_batched_matches_per_freq(degree, ground_z):
    """The fully batched swept fast path (batched off-edge moments + batched
    C++ assemble + stacked LAPACK solve) must agree with per-frequency
    solves to roundoff. Covers free space and the PEC image ground — the
    two cases the fast path claims; junction/enrichment/finite-ground
    cases fall back and are covered elsewhere.
    """
    hd = 0.962 * 22 / 4
    z_off = 7.0 if ground_z is not None else 0.0
    wires = (
        [np.array([[-hd, 0.0, z_off], [hd, 0.0, z_off]])]
        if ground_z is not None
        else [np.array([[0.0, 0.0, -hd], [0.0, 0.0, hd]])]
    )
    kwargs = dict(wires=wires, nsegs=24, degree=degree)
    if ground_z is not None:
        kwargs["ground_z"] = ground_z

    C_LIGHT = 299_792_458.0
    k0 = 2 * np.pi / 22
    k_array = np.linspace(0.9 * k0, 1.1 * k0, 5)

    sim = BSplineSolver(**kwargs)
    z_swept = sim.compute_impedance_swept(k_array)
    for kk, zs in zip(k_array, z_swept):
        sim_f = BSplineSolver(wavelength=2 * np.pi / kk, **kwargs)
        z_f, _ = sim_f.compute_impedance()
        assert abs(zs - z_f) <= 1e-9 * abs(z_f), (
            f"d={degree}, ground={ground_z}, k={kk}: swept={zs}, single={z_f}"
        )
    _ = C_LIGHT  # (kept for symmetry with sibling tests)


def test_bspline_swept_fully_batched_grounded_bent_deck_matches_per_freq():
    """The fully batched swept fast path's PEC-image branch must agree with
    per-frequency solves on a BENT deck, not the straight horizontal wire
    `test_bspline_swept_fully_batched_matches_per_freq` uses for its
    `ground_z=0.0` case. A straight horizontal wire has t_z == 0 on every
    segment, so the image mirror M = diag(1, 1, -1) is the identity on
    every tangent — a dropped or mis-signed mirror on the image tangent
    table is then numerically invisible (the #323 trap, from the same side
    as #333's mirrored tangent table). `_grounded_window_kw`'s inverted-V
    gives every segment a z-component, so the mirror has to be applied
    correctly for this to pass.
    """
    kw = _grounded_window_kw()
    sim = BSplineSolver(**kw)
    assert sim._swept_batched_available()

    k0 = 2 * np.pi / kw["wavelength"]
    k_array = np.linspace(0.94 * k0, 1.06 * k0, 6)
    z_swept = sim.compute_impedance_swept(k_array)
    for kk, zs in zip(k_array, z_swept):
        sim_f = BSplineSolver(**{**kw, "wavelength": 2 * np.pi / kk})
        z_f, _ = sim_f.compute_impedance()
        assert abs(zs - z_f) <= 1e-9 * abs(z_f), f"k={kk}: swept={zs}, single={z_f}"


def test_bspline_swept_fully_batched_multifeed_matches_per_freq():
    """Multi-feed variant of the batched fast path: per-feed driving-point
    vector must match per-frequency solves."""
    hd = 0.962 * 22 / 4
    wires = [
        np.array([[0.0, 0.0, -hd], [0.0, 0.0, hd]]),
        np.array([[3.0, 0.0, -hd], [3.0, 0.0, hd]]),
    ]
    feeds = [(0, None, 1.0), (1, None, 1.0 + 0.5j)]
    k0 = 2 * np.pi / 22
    k_array = np.linspace(0.95 * k0, 1.05 * k0, 3)
    sim = BSplineSolver(wires=wires, nsegs=20, degree=2, feeds=feeds)
    z_swept = sim.compute_impedance_swept(k_array)  # (n_k, n_feeds)
    for kk, zs in zip(k_array, z_swept):
        sim_f = BSplineSolver(
            wires=wires, nsegs=20, degree=2, feeds=feeds, wavelength=2 * np.pi / kk
        )
        z_f, _ = sim_f.compute_impedance()
        assert np.allclose(zs, z_f, rtol=1e-9, atol=0), f"k={kk}: {zs} vs {z_f}"


def test_bspline_swept_batched_tiny_memory_budget_matches():
    """`swept_mem_mb` caps the batched sweep's transient memory by
    shrinking the k-chunk. Correctness must be budget-independent — even
    the degenerate chunk=1 case (budget below one k's tensors) must match
    the default. Guards the chunking arithmetic and the per-chunk
    same-edge reg-moment slices.
    """
    hd = 0.962 * 22 / 4
    wires = [np.array([[0.0, 0.0, -hd], [0.0, 0.0, hd]])]
    k0 = 2 * np.pi / 22
    k_array = np.linspace(0.9 * k0, 1.1 * k0, 5)

    z_default = BSplineSolver(wires=wires, nsegs=24, degree=2).compute_impedance_swept(
        k_array
    )
    z_tiny = BSplineSolver(
        wires=wires, nsegs=24, degree=2, swept_mem_mb=1
    ).compute_impedance_swept(k_array)
    assert np.allclose(z_default, z_tiny, rtol=1e-11, atol=0)


# ---- PEC ground (image method) ----


def _h_dipole(L, h):
    return np.array([[0.0, -L / 2, h], [0.0, L / 2, h]])


# ---- PEC ground on BSplineSolver (image method) ----


def test_bspline_ground_none_matches_free_space_bit_exact():
    L = 2 * 0.962 * 22 / 4
    poly = _h_dipole(L, 0.0)
    z_no, _ = BSplineSolver(
        wires=[poly], n_per_edge_per_wire=[[40]], nsegs=40, degree=2
    ).compute_impedance()
    z_none, _ = BSplineSolver(
        wires=[poly],
        n_per_edge_per_wire=[[40]],
        nsegs=40,
        degree=2,
        ground_z=None,
    ).compute_impedance()
    assert z_no == z_none


def test_bspline_ground_horizontal_dipole_at_height_recovers_free_space():
    L = 2 * 0.962 * 22 / 4
    N = 30
    z_free, _ = BSplineSolver(
        wires=[_h_dipole(L, 0.0)], n_per_edge_per_wire=[[N]], nsegs=N, degree=2
    ).compute_impedance()
    z_high, _ = BSplineSolver(
        wires=[_h_dipole(L, 100.0)],  # ~5 wavelengths up
        n_per_edge_per_wire=[[N]],
        nsegs=N,
        degree=2,
        ground_z=0.0,
    ).compute_impedance()
    assert abs(z_high.real - z_free.real) < 2.0
    assert abs(z_high.imag - z_free.imag) < 3.0


def test_bspline_ground_horizontal_dipole_at_zero_height_shorts_out():
    L = 2 * 0.962 * 22 / 4
    z_lo, _ = BSplineSolver(
        wires=[_h_dipole(L, 0.01)],
        n_per_edge_per_wire=[[40]],
        nsegs=40,
        degree=2,
        ground_z=0.0,
    ).compute_impedance()
    assert abs(z_lo.real) < 0.5


def test_bspline_d2_ground_agrees_with_d1_at_moderate_height():
    # Tent-basis (O(1/N) R-rate) and B-spline d=2 (basis-limited) converge
    # to the same Z as N→∞ but disagree by their respective truncation
    # errors at finite N. R is the physically meaningful number for ground
    # effects (image symmetry doubles it near h=λ/4); X has higher
    # basis-sensitivity at this N. Tolerances reflect that.
    L = 2 * 0.962 * 22 / 4
    N = 40
    h = 7.0
    z_tri, _ = BSplineSolver(
        wires=[_h_dipole(L, h)],
        n_per_edge_per_wire=[[N]],
        nsegs=N,
        degree=1,
        ground_z=0.0,
    ).compute_impedance()
    z_bsp, _ = BSplineSolver(
        wires=[_h_dipole(L, h)],
        n_per_edge_per_wire=[[N]],
        nsegs=N,
        degree=2,
        ground_z=0.0,
    ).compute_impedance()
    assert abs(z_bsp.real - z_tri.real) < 0.03 * abs(z_tri.real)
    assert abs(z_bsp.imag - z_tri.imag) < 0.10 * max(abs(z_tri.imag), 5.0)


def test_bspline_ground_swept_matches_single_freq():
    L = 2 * 0.962 * 22 / 4
    N = 30
    h = 5.0
    sim = BSplineSolver(
        wires=[_h_dipole(L, h)],
        n_per_edge_per_wire=[[N]],
        nsegs=N,
        degree=2,
        ground_z=0.0,
    )
    z_single, _ = sim.compute_impedance()
    z_swept = sim.compute_impedance_swept(np.array([sim.k]))[0]
    assert abs(z_single - z_swept) < 1e-9


# ---- PEC ground on SinusoidalSolver (NEC three-term basis, image method) ----


def test_sinusoidal_ground_none_matches_free_space_bit_exact():
    L = 2 * 0.962 * 22 / 4
    poly = _h_dipole(L, 0.0)
    z_no, _ = SinusoidalSolver(
        wires=[poly], n_per_edge_per_wire=[[40]], nsegs=40
    ).compute_impedance()
    z_none, _ = SinusoidalSolver(
        wires=[poly], n_per_edge_per_wire=[[40]], nsegs=40, ground_z=None
    ).compute_impedance()
    assert z_no == z_none


def test_sinusoidal_ground_horizontal_dipole_at_height_recovers_free_space():
    L = 2 * 0.962 * 22 / 4
    N = 30
    z_free, _ = SinusoidalSolver(
        wires=[_h_dipole(L, 0.0)], n_per_edge_per_wire=[[N]], nsegs=N
    ).compute_impedance()
    z_high, _ = SinusoidalSolver(
        wires=[_h_dipole(L, 100.0)],
        n_per_edge_per_wire=[[N]],
        nsegs=N,
        ground_z=0.0,
    ).compute_impedance()
    assert abs(z_high.real - z_free.real) < 2.0
    assert abs(z_high.imag - z_free.imag) < 3.0


def test_sinusoidal_ground_horizontal_dipole_at_zero_height_shorts_out():
    L = 2 * 0.962 * 22 / 4
    z_lo, _ = SinusoidalSolver(
        wires=[_h_dipole(L, 0.01)],
        n_per_edge_per_wire=[[40]],
        nsegs=40,
        ground_z=0.0,
    ).compute_impedance()
    assert abs(z_lo.real) < 0.5


def test_sinusoidal_ground_agrees_with_d1_at_moderate_height():
    L = 2 * 0.962 * 22 / 4
    N = 40
    h = 7.0
    z_tri, _ = BSplineSolver(
        wires=[_h_dipole(L, h)],
        n_per_edge_per_wire=[[N]],
        nsegs=N,
        degree=1,
        ground_z=0.0,
    ).compute_impedance()
    z_sin, _ = SinusoidalSolver(
        wires=[_h_dipole(L, h)],
        n_per_edge_per_wire=[[N]],
        nsegs=N,
        ground_z=0.0,
    ).compute_impedance()
    assert abs(z_sin.real - z_tri.real) < 0.03 * abs(z_tri.real)
    assert abs(z_sin.imag - z_tri.imag) < 0.10 * max(abs(z_tri.imag), 5.0)


def test_sinusoidal_ground_swept_matches_single_freq():
    L = 2 * 0.962 * 22 / 4
    N = 30
    h = 5.0
    sim = SinusoidalSolver(
        wires=[_h_dipole(L, h)],
        n_per_edge_per_wire=[[N]],
        nsegs=N,
        ground_z=0.0,
    )
    z_single, _ = sim.compute_impedance()
    z_swept = sim.compute_impedance_swept(np.array([sim.k]))[0]
    assert abs(z_single - z_swept) < 1e-9


def test_bspline_all_grounds_with_enrichment_construct():
    """Enrichment is supported over every ground model (#167 complete): PEC
    image (no ground_eps), fast finite (refl-coef with ground_eps), and
    Sommerfeld (ground_model='sommerfeld'). None raise; the functional
    grounded-enrichment tests below check each one's physics."""
    L = 2 * 0.962 * 22 / 4
    for extra in (
        {},
        {"ground_eps": (13.0, 0.005)},
        {"ground_eps": (13.0, 0.005), "ground_model": "sommerfeld"},
    ):
        BSplineSolver(
            wires=[_h_dipole(L, 5.0)],
            n_per_edge_per_wire=[[20]],
            nsegs=20,
            degree=2,
            ground_z=0.0,
            use_singular_enrichment=True,
            **extra,
        )


# ---------------------------------------------------------------------------
# Multi-feed support (delta-gap excitations with prescribed complex voltages)
# ---------------------------------------------------------------------------


def _two_dipoles(halfdriver, spacing):
    a = np.array([[0.0, -halfdriver, 0.0], [0.0, halfdriver, 0.0]])
    b = np.array([[spacing, -halfdriver, 0.0], [spacing, halfdriver, 0.0]])
    return [a, b]


# ---------------------------------------------------------------------------
# Same multi-feed checks against BSplineSolver and SinusoidalSolver.
# ---------------------------------------------------------------------------


def _mk_sim(cls, *, wires, n_per_edge_per_wire, nsegs, feeds=None, **extra):
    kw = dict(
        wires=wires,
        n_per_edge_per_wire=n_per_edge_per_wire,
        nsegs=nsegs,
    )
    if feeds is not None:
        kw["feeds"] = feeds
    if cls is BSplineSolver:
        kw["degree"] = 2
    kw.update(extra)
    return cls(**kw)


@pytest.mark.parametrize("cls", [BSplineSolver, SinusoidalSolver])
def test_multifeed_single_feed_via_feeds_kwarg_matches_legacy(cls):
    L = 2 * 0.962 * 22 / 4
    nsegs = 40
    wires = [np.array([[0.0, 0.0, 0.0], [0.0, L, 0.0]])]
    z_legacy, _ = _mk_sim(
        cls, wires=wires, n_per_edge_per_wire=[[nsegs]], nsegs=nsegs
    ).compute_impedance()
    z_new, _ = _mk_sim(
        cls,
        wires=wires,
        n_per_edge_per_wire=[[nsegs]],
        nsegs=nsegs,
        feeds=[(0, None, 1.0 + 0.0j)],
    ).compute_impedance()
    assert abs(z_new - z_legacy) < 1e-9


@pytest.mark.parametrize("cls", [BSplineSolver, SinusoidalSolver])
def test_multifeed_two_dipoles_in_phase(cls):
    hd = 0.962 * 22 / 4
    nsegs = 40
    wires = _two_dipoles(hd, spacing=2.0)
    sim = _mk_sim(
        cls,
        wires=wires,
        n_per_edge_per_wire=[[nsegs], [nsegs]],
        nsegs=nsegs,
        feeds=[(0, None, 1.0 + 0.0j), (1, None, 1.0 + 0.0j)],
    )
    z_per_feed, c = sim.compute_impedance()
    assert z_per_feed.shape == (2,)
    assert np.isfinite(c).all()
    assert abs(z_per_feed[0] - z_per_feed[1]) / abs(z_per_feed[0]) < 1e-4
    assert 30.0 < z_per_feed[0].real < 200.0


@pytest.mark.parametrize("cls", [BSplineSolver, SinusoidalSolver])
def test_multifeed_phase_shift_changes_driving_point(cls):
    hd = 0.962 * 22 / 4
    nsegs = 40
    wires = _two_dipoles(hd, spacing=2.0)
    z_inphase, _ = _mk_sim(
        cls,
        wires=wires,
        n_per_edge_per_wire=[[nsegs], [nsegs]],
        nsegs=nsegs,
        feeds=[(0, None, 1.0 + 0.0j), (1, None, 1.0 + 0.0j)],
    ).compute_impedance()
    z_anti, _ = _mk_sim(
        cls,
        wires=wires,
        n_per_edge_per_wire=[[nsegs], [nsegs]],
        nsegs=nsegs,
        feeds=[(0, None, 1.0 + 0.0j), (1, None, -1.0 + 0.0j)],
    ).compute_impedance()
    assert abs(z_inphase[0] - z_anti[0]) > 5.0


@pytest.mark.parametrize("cls", [BSplineSolver, SinusoidalSolver])
def test_multifeed_homogeneity_in_voltage(cls):
    # Driving-point impedance V/I is a ratio; scaling all voltages by a
    # common (complex) factor must leave Z_i unchanged. This is a
    # solver-agnostic linearity check that exercises the multi-feed RHS
    # plumbing without requiring an external port-Z reference.
    hd = 0.962 * 22 / 4
    nsegs = 40
    wires = _two_dipoles(hd, spacing=2.0)
    common = dict(wires=wires, n_per_edge_per_wire=[[nsegs], [nsegs]], nsegs=nsegs)
    V = np.array([1.0 + 0j, np.exp(1j * np.pi / 3)])
    z1, _ = _mk_sim(
        cls, **common, feeds=[(0, None, V[0]), (1, None, V[1])]
    ).compute_impedance()
    z2, _ = _mk_sim(
        cls,
        **common,
        feeds=[(0, None, 2.5 * V[0]), (1, None, 2.5 * V[1])],
    ).compute_impedance()
    assert np.allclose(z1, z2, rtol=1e-8, atol=1e-12)
    assert np.isfinite(z1).all()


@pytest.mark.parametrize("cls", [BSplineSolver, SinusoidalSolver])
def test_multifeed_swept_matches_single_k(cls):
    hd = 0.962 * 22 / 4
    nsegs = 30
    wires = _two_dipoles(hd, spacing=2.5)
    feeds = [(0, None, 1.0 + 0.0j), (1, None, np.exp(1j * np.pi / 4))]
    sim = _mk_sim(
        cls,
        wires=wires,
        n_per_edge_per_wire=[[nsegs], [nsegs]],
        nsegs=nsegs,
        feeds=feeds,
    )
    z_single, _ = sim.compute_impedance()
    z_swept = sim.compute_impedance_swept(np.array([sim.k]))
    assert z_swept.shape == (1, 2)
    assert np.allclose(z_swept[0], z_single, rtol=1e-8, atol=1e-12)


def test_d1_bowtiearray_1x2_phased():
    # Simplified "bowtie-array 1x2" stand-in: two V-shaped (kinked-dipole)
    # elements side-by-side, each driven with its own complex voltage.
    # The point of this test is the multi-feed plumbing on a non-trivial
    # multi-wire / kinked geometry, not bowtie geometric fidelity.
    hd = 0.962 * 22 / 4
    nsegs = 30
    bend = 0.3 * hd  # z-droop at the tip — gives each element a kink
    del_y = 2.0
    elem_tmpl = np.array(
        [
            [0.0, -hd, -bend],
            [0.0, 0.0, 0.0],
            [0.0, hd, -bend],
        ]
    )
    left = elem_tmpl + np.array([0.0, -del_y, 0.0])
    right = elem_tmpl + np.array([0.0, +del_y, 0.0])

    sim = BSplineSolver(
        wires=[left, right],
        n_per_edge_per_wire=[[nsegs, nsegs], [nsegs, nsegs]],
        nsegs=nsegs,
        degree=1,
        feeds=[(0, None, 1.0 + 0.0j), (1, None, np.exp(1j * np.pi / 2))],
    )
    z_per_feed, coeffs = sim.compute_impedance()
    assert z_per_feed.shape == (2,)
    assert np.isfinite(z_per_feed).all()
    assert np.isfinite(coeffs).all()
    # Per-feed Re(Z_i) can legitimately go negative when ports exchange
    # power through strong mutual coupling (the system as a whole still
    # radiates). Just sanity-check magnitudes stay bounded.
    assert abs(z_per_feed[0]) < 1000.0
    assert abs(z_per_feed[1]) < 1000.0


@pytest.mark.parametrize("degree", [1, 2])
def test_bspline_chunked_dense_z_matches_tensor_path(degree):
    """The chunked fill+assemble (`_compute_Z_dense_chunked`, issue #136)
    must reproduce the legacy build-the-tensor-then-assemble Z to
    reduction-order precision. Geometry mixes a junction, multiple edges,
    and a 2-segment wire (zero-weight padding wings), and swept_mem_mb=0
    forces 1-row chunks so every window boundary case is exercised —
    including the per-edge same-edge correction windows."""
    import momwire.bspline as bmod
    from momwire.bspline import BSplineSolver

    if not bmod._HAVE_BSPLINE_WINDOWED_ASSEMBLE_ACCEL:
        pytest.skip("windowed Z assembly accelerator not built")

    h = 0.962 * 22 / 4
    w0 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, h]])
    w1 = np.array([[0.0, 0.0, h], [0.0, h, h]])
    w2 = np.array([[0.0, 0.0, 0.0], [0.35, 0.0, 0.0]])
    junctions = [[(0, "end"), (1, "start")], [(0, "start"), (2, "start")]]
    sim = BSplineSolver(
        wires=[w0, w1, w2],
        degree=degree,
        n_per_edge_per_wire=[[13], [9], [2]],
        nsegs=13,
        wavelength=22.0,
        junctions=junctions,
        feed_wire_index=0,
    )
    geom = sim._build_geometry()
    supp, polys, _kcl, _wk, _wbg = sim._build_basis_polynomials(geom)

    sim.swept_mem_mb = 0  # chunk = max(1, 0) -> single-row windows
    Z_chunked = sim._compute_Z_dense_chunked(geom, sim.k, supp, polys)

    J = sim._build_J_blocks(geom, sim.k)
    Z_tensor = sim._assemble_Z(J, supp, polys, geom)

    rel = np.abs(Z_chunked - Z_tensor).max() / np.abs(Z_tensor).max()
    # 1e-10 margin, same policy as the grounded tests below: MSVC/AppleClang
    # reduction order lands these equalities at 1e-12..8e-12 (main wheels
    # runs 29345985527, 29350220998) — pin the algebra, not the compiler.
    assert rel < 1e-10, f"chunked vs tensor Z disagreement: rel {rel:.2e}"


def test_bspline_chunked_dense_impedance_matches_tensor_path():
    """End-to-end compute_impedance through a forced chunked path vs the
    tensor path (flag flipped off) — covers the
    same-edge prep sharing and everything downstream of Z."""
    import momwire.bspline as bmod
    from momwire.bspline import BSplineSolver

    if not bmod._HAVE_BSPLINE_WINDOWED_ASSEMBLE_ACCEL:
        pytest.skip("windowed Z assembly accelerator not built")

    L = 2 * 0.962 * 22 / 4
    wires = [np.array([[0.0, -L / 2, 0.0], [0.0, L / 2, 0.0]])]
    kw = {"wires": wires, "n_per_edge_per_wire": [[21]], "nsegs": 21, "degree": 2}
    chunked = BSplineSolver(**kw)
    chunked.swept_mem_mb = 0
    z_chunked, _ = chunked.compute_impedance()

    saved = bmod._HAVE_BSPLINE_WINDOWED_ASSEMBLE_ACCEL
    try:
        bmod._HAVE_BSPLINE_WINDOWED_ASSEMBLE_ACCEL = False
        z_tensor, _ = BSplineSolver(**kw).compute_impedance()
    finally:
        bmod._HAVE_BSPLINE_WINDOWED_ASSEMBLE_ACCEL = saved

    rel = abs(z_chunked - z_tensor) / abs(z_tensor)
    # 1e-10 margin — MSVC landed this at 1.02e-12 on the main wheels run
    # 29350220998; same cross-compiler policy as the grounded tests.
    assert rel < 1e-10, f"chunked vs tensor impedance disagreement: rel {rel}"


@pytest.mark.parametrize("extended_kernel", [False, True])
def test_bspline_chunked_dense_z_holds_no_n_squared_transient(extended_kernel):
    """The chunked fill must not carry any N²-scale float64 side table
    (issue #318). The tangent-dot matrix used to be built up front as
    `tangents @ tangents.T` and handed to the windowed assembler, so it
    stayed alive for the whole fill — 0.5x the dense Z's own size, added
    to peak RSS at exactly the worst moment. The assembler now takes the
    (n_segs, 3) tangent table and forms each pair's dot itself.

    Arithmetic for the threshold, at the N = 1200 basis functions this
    geometry builds (tracemalloc sees numpy's data allocations):

      * Z itself is 16 N² = 23.0 MB, subtracted off below.
      * the deleted td_all was 8 N² = 11.5 MB.
      * everything else the fill allocates is O(n_segs), not O(N²): one
        observer-row chunk (bounded by swept_mem_mb = 1) plus the
        per-edge same-edge quadrature prep — measured together at
        ~2.2 MB, so the old code peaked ~13.8 MB above Z.

    6 MB therefore sits ~2.7x above the transient the fixed code needs
    and ~2.3x below what the N × N table alone would have cost.

    The extended-kernel case runs the SAME budget, deliberately: EK
    routes the fill through the `..._ek` twins, and everything extra it
    touches is bounded by the chunk, not by N². The whole-mesh `_EK`
    spec is (n_segs,) group labels; `_ek_slice` hands each chunk a
    *slice* of them, and the C++ EK kernels take the labels rather than
    materialising any pair table. (On the numpy fallback `_ek_pair_mask`
    is (chunk_rows, n_segs) bool — still chunk-bounded, and 1/16th the
    moment chunk that sets the budget in the first place.) Measured at
    N = 1200: 2.23 MB with EK against 2.24 MB without, i.e. flat. A
    second N² table appearing only under EK is exactly what this case
    exists to catch.
    """
    import tracemalloc

    import momwire.bspline as bmod
    from momwire.bspline import BSplineSolver

    if not bmod._HAVE_BSPLINE_WINDOWED_ASSEMBLE_ACCEL:
        pytest.skip("windowed Z assembly accelerator not built")

    # One straight wire, split into many short edges: the same-edge
    # correction blocks stay small, keeping every non-N² transient well
    # clear of the threshold. One straight run of one radius is also a
    # single coaxial EK group, so the EK case extends every pair.
    n_edges, seg_per_edge = 150, 8
    L = 0.962 * 22 / 2
    ys = np.linspace(-L / 2, L / 2, n_edges + 1)
    wires = [np.column_stack([np.zeros_like(ys), ys, np.zeros_like(ys)])]
    sim = BSplineSolver(
        wires=wires,
        degree=2,
        n_per_edge_per_wire=[[seg_per_edge] * n_edges],
        nsegs=n_edges * seg_per_edge,
        wavelength=22.0,
        extended_kernel=extended_kernel,
    )
    geom = sim._build_geometry()
    if extended_kernel:
        # Guard the guard: a spec that labelled nothing eligible would
        # make this case a duplicate of the reduced one.
        assert sim._ek_spec(geom).group_i is not None
    supp, polys, _kcl, _wk, _wbg = sim._build_basis_polynomials(geom)
    assert supp.shape[0] == n_edges * seg_per_edge  # N = 1200

    sim.swept_mem_mb = 1
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        Z = sim._compute_Z_dense_chunked(geom, sim.k, supp, polys)
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    transient = peak - Z.nbytes
    assert transient < 6_000_000, (
        f"chunked fill peaked {transient / 1e6:.2f} MB above Z "
        f"(8 N**2 = {8 * Z.shape[0] ** 2 / 1e6:.2f} MB) — an N-squared "
        "transient is back"
    )


def test_bspline_chunked_dense_z_fill_transient_honors_swept_mem_mb_budget():
    """The chunked fill's own advertised budget must be honest: peak
    transient above Z should track `swept_mem_mb`, not roughly double it
    (issue #338).

    Before this fix, the observer-chunk loop rebound `J_chunk` to a new
    window each iteration WITHOUT dropping the old one first — `J_chunk =
    producer(...)` builds the new array before the assignment retires the
    old reference, so for one heartbeat the loop held both the just-used
    window and its replacement at once. Nothing in the row-byte budget
    arithmetic accounted for that: it sizes the chunk for ONE window, and
    the loop transiently held two. `_compute_Z_dense_chunked` now `del`s
    each window immediately after `_accumulate` consumes it, so only one
    is ever alive.

    At the N = 1200 geometry (150 edges x 8 segs) this file's other
    chunked-fill gates use, swept_mem_mb = 8 sizes a chunk = 48 rows, i.e.
    a ~8.29 MB window — comfortably above the O(N) same-edge-prep floor
    (~1.4 MB total across all 150 edges) other gates in this file measure,
    so the ratio below isolates the double-buffering effect rather than
    noise from that floor.

    Measured: 16.61 MB transient (1.980x budget) pre-fix, 8.33 MB
    (0.993x) post-fix — the acceptance bar from #338 is <= 1.1x. Margin:
    threshold is 1.1x budget, comfortably between the fixed value and the
    doubled pre-fix one.
    """
    import tracemalloc

    import momwire.bspline as bmod
    from momwire.bspline import BSplineSolver

    if not bmod._HAVE_BSPLINE_WINDOWED_ASSEMBLE_ACCEL:
        pytest.skip("windowed Z assembly accelerator not built")

    n_edges, seg_per_edge = 150, 8
    L = 0.962 * 22 / 2
    ys = np.linspace(-L / 2, L / 2, n_edges + 1)
    wires = [np.column_stack([np.zeros_like(ys), ys, np.zeros_like(ys)])]
    sim = BSplineSolver(
        wires=wires,
        degree=2,
        n_per_edge_per_wire=[[seg_per_edge] * n_edges],
        nsegs=n_edges * seg_per_edge,
        wavelength=22.0,
    )
    geom = sim._build_geometry()
    supp, polys, _kcl, _wk, _wbg = sim._build_basis_polynomials(geom)
    assert supp.shape[0] == n_edges * seg_per_edge  # N = 1200

    budget_mb = 8
    sim.swept_mem_mb = budget_mb
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        Z = sim._compute_Z_dense_chunked(geom, sim.k, supp, polys)
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    budget_bytes = budget_mb * 1024 * 1024
    transient = peak - Z.nbytes
    ratio = transient / budget_bytes
    assert ratio <= 1.1, (
        f"chunked fill transient {transient / 1e6:.2f} MB is {ratio:.2f}x "
        f"the {budget_mb} MB swept_mem_mb budget (want <= 1.1x) — the "
        "budget arithmetic (or the loop holding two windows at once) is "
        "dishonest again"
    )


@pytest.mark.parametrize(
    "ground_kwargs",
    [
        pytest.param({}, id="pec"),
        pytest.param({"ground_eps": (10.0, 0.002)}, id="refl-coef"),
        pytest.param(
            {"ground_eps": (10.0, 0.002), "ground_model": "sommerfeld"},
            id="sommerfeld",
        ),
    ],
)
def test_bspline_chunked_image_fill_holds_no_n_squared_transient(ground_kwargs):
    """The grounded chunked image fill must allocate nothing N²-scale
    beyond Z itself (issue #323). The accumulator used to be handed the
    same global (N, N) weight tables the tensor path builds and slice
    them per observer chunk, so TWO complex128 (N, N) tables stayed
    resident for the whole fill even though each chunk reads one row
    band. `_image_weight_window_fn` now produces the windows per chunk.

    Arithmetic for the threshold, at the N = 1200 basis functions this
    geometry builds (tracemalloc sees numpy's data allocations; Z is
    preallocated outside the traced region, so the peak below IS the
    transient):

      * the retired residency was 2 × 16 N² = 46.1 MB. One table alone
        is 23.0 MB, and even the float64 gemm precursor the PEC branch
        computes before its complex cast is 8 N² = 11.5 MB — that
        11.5 MB is the smallest N²-scale object that can appear here.
      * what the fixed code allocates is all chunk-bounded: the
        mirrored-source moment window (capped by swept_mem_mb = 1),
        the two (chunk, n_segs) complex128 weight windows, and — in
        the refl-coef case — the five (chunk, n_segs) float64 specular
        intermediates `specular_pair_tables` returns. At
        swept_mem_mb = 1 the chunk is 6 observer rows, so all of that
        is well under the moment window that sets the budget.

    Measured peaks: 2.39 MB (pec), 2.42 MB (refl-coef), 2.38 MB
    (sommerfeld) — flat across modes, as expected when nothing scales
    with N. 6 MB therefore sits ~2.5x above the worst mode and ~1.9x
    below the smallest N²-scale object that could come back. Restoring
    the retired residency in the PEC branch alone measures 48.2 MB.

    Unlike the grounded equality gates, which need bent decks so a
    mirror-sign or dyad error cannot cancel, this one measures
    ALLOCATION rather than algebra: a straight horizontal wire is fine
    here, and it keeps the same-edge-free image fill's per-chunk work
    uniform. It is lifted to z = 2.2 over ground_z = 0 because a deck
    lying in the ground plane is degenerate for image builds.

    The sommerfeld case gates the C2 exact-image term only — the smooth
    `_Z_sommerfeld_remainder` is a separate, separately-chunked term
    outside issue #323's scope and outside the call traced below.
    """
    import tracemalloc

    import momwire.bspline as bmod
    from momwire.bspline import BSplineSolver

    if not bmod._HAVE_BSPLINE_W_WINDOWED_ASSEMBLE_ACCEL:
        pytest.skip("weighted windowed Z assembly accelerator not built")

    n_edges, seg_per_edge = 150, 8
    L = 0.962 * 22 / 2
    ys = np.linspace(-L / 2, L / 2, n_edges + 1)
    wires = [np.column_stack([np.zeros_like(ys), ys, np.full_like(ys, 2.2)])]
    sim = BSplineSolver(
        wires=wires,
        degree=2,
        n_per_edge_per_wire=[[seg_per_edge] * n_edges],
        nsegs=n_edges * seg_per_edge,
        wavelength=22.0,
        ground_z=0.0,
        **ground_kwargs,
    )
    geom = sim._build_geometry()
    supp, polys, _kcl, _wk, _wbg = sim._build_basis_polynomials(geom)
    assert supp.shape[0] == n_edges * seg_per_edge  # N = 1200

    # Z is allocated (and zeroed) before tracing starts, so the peak the
    # image fill reports is its own transient with nothing subtracted.
    Z = np.zeros((supp.shape[0],) * 2, dtype=np.complex128, order="F")
    sim.swept_mem_mb = 1
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        # The window producer is built INSIDE the traced region on purpose:
        # a mode that went back to materialising its tables up front would
        # do it here, in the closure's body, and escape a hoisted call.
        sim._accumulate_Z_image_chunked(
            Z, geom, sim.k, supp, polys, sim._image_weight_window_fn(geom)
        )
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # Guard the guard: a fill that accumulated nothing would pass on
    # allocation while measuring nothing at all.
    assert np.any(Z != 0.0)
    assert peak < 6_000_000, (
        f"chunked image fill peaked {peak / 1e6:.2f} MB "
        f"(one complex weight table = {16 * Z.shape[0] ** 2 / 1e6:.2f} MB, "
        f"its float64 precursor = {8 * Z.shape[0] ** 2 / 1e6:.2f} MB) — an "
        "N-squared weight residency is back"
    )


def test_bspline_dense_dispatch_respects_memory_budget(monkeypatch):
    """Small tensors use the faster tensor path; oversized ones stay chunked."""
    import momwire.bspline as bmod
    from momwire.bspline import BSplineSolver

    if not bmod._HAVE_BSPLINE_WINDOWED_ASSEMBLE_ACCEL:
        pytest.skip("windowed Z assembly accelerator not built")

    L = 2 * 0.962 * 22 / 4
    wires = [np.array([[0.0, -L / 2, 0.0], [0.0, L / 2, 0.0]])]
    kw = {"wires": wires, "n_per_edge_per_wire": [[21]], "nsegs": 21, "degree": 2}

    def unexpected(*_args, **_kwargs):
        pytest.fail("unexpected dense assembly path")

    tensor = BSplineSolver(**kw)
    monkeypatch.setattr(tensor, "_compute_Z_dense_chunked", unexpected)
    tensor.compute_impedance()

    chunked = BSplineSolver(**kw)
    chunked.swept_mem_mb = 0
    monkeypatch.setattr(chunked, "_build_J_blocks", unexpected)
    chunked.compute_impedance()


def test_bspline_chunked_dense_y_matrix_matches_tensor_path():
    """compute_y_matrix through a forced chunked fill vs the tensor path
    (flag flipped off), on a genuinely 2x2 Y (a mutual term plus two self
    terms). Until issue #235 the Y entry point bypassed the issue-#136
    dispatch entirely and materialised the full moment tensor
    unconditionally — this pins the routed path to the legacy algebra."""
    import momwire.bspline as bmod
    from momwire.bspline import BSplineSolver

    if not bmod._HAVE_BSPLINE_WINDOWED_ASSEMBLE_ACCEL:
        pytest.skip("windowed Z assembly accelerator not built")

    L = 2 * 0.962 * 22 / 4
    wires = [
        np.array([[0.0, -L / 2, 0.0], [0.0, L / 2, 0.0]]),
        np.array([[3.0, -L / 2, 0.0], [3.0, L / 2, 0.0]]),
    ]
    kw = dict(
        wires=wires,
        n_per_edge_per_wire=[[21], [17]],
        nsegs=21,
        degree=2,
        wavelength=22.0,
        feeds=[(0, L / 2, 1.0), (1, L / 2, 0.0)],
    )
    chunked = BSplineSolver(**kw)
    chunked.swept_mem_mb = 0  # chunk = max(1, 0) -> single-row windows
    Y_chunked = chunked.compute_y_matrix()

    saved = bmod._HAVE_BSPLINE_WINDOWED_ASSEMBLE_ACCEL
    try:
        bmod._HAVE_BSPLINE_WINDOWED_ASSEMBLE_ACCEL = False
        Y_tensor = BSplineSolver(**kw).compute_y_matrix()
    finally:
        bmod._HAVE_BSPLINE_WINDOWED_ASSEMBLE_ACCEL = saved

    rel = np.abs(Y_chunked - Y_tensor).max() / np.abs(Y_tensor).max()
    # 1e-10 margin, same cross-compiler policy as the impedance twin above.
    assert rel < 1e-10, f"chunked vs tensor Y disagreement: rel {rel:.2e}"


def test_bspline_y_matrix_dispatch_respects_memory_budget(monkeypatch):
    """compute_y_matrix takes the tensor path when the moment tensor fits
    the budget and the chunked path when it doesn't. The second half is
    the disabled-path probe for the issue-#235 routing: before it,
    compute_y_matrix never called the chunked build at any budget."""
    import momwire.bspline as bmod
    from momwire.bspline import BSplineSolver

    if not bmod._HAVE_BSPLINE_WINDOWED_ASSEMBLE_ACCEL:
        pytest.skip("windowed Z assembly accelerator not built")

    L = 2 * 0.962 * 22 / 4
    wires = [np.array([[0.0, -L / 2, 0.0], [0.0, L / 2, 0.0]])]
    kw = {"wires": wires, "n_per_edge_per_wire": [[21]], "nsegs": 21, "degree": 2}

    def unexpected(*_args, **_kwargs):
        pytest.fail("unexpected dense assembly path")

    tensor = BSplineSolver(**kw)
    monkeypatch.setattr(tensor, "_compute_Z_dense_chunked", unexpected)
    tensor.compute_y_matrix()

    chunked = BSplineSolver(**kw)
    chunked.swept_mem_mb = 0
    monkeypatch.setattr(chunked, "_build_J_blocks", unexpected)
    chunked.compute_y_matrix()


def test_bspline_y_swept_fallback_respects_memory_budget(monkeypatch):
    """The swept twin of the dispatch probe above (issue #238).

    `compute_y_matrix_swept`'s fallback — the route taken when the batched
    swept accelerator is unavailable — used to call `_build_J_blocks` itself
    and materialise the full (d+1, d+1, N, N) moment tensor at ANY budget,
    the same hole #235 closed at single k. It now goes through
    `compute_port_solution`, hence `_compute_Z_operator`, hence the
    `swept_mem_mb` dispatch. Both halves are booby-trapped: the budgeted
    solver must never touch the tensor build, the default-budget one must
    never touch the chunked build, and the two must agree.
    """
    import momwire.bspline as bmod
    from momwire.bspline import BSplineSolver

    if not bmod._HAVE_BSPLINE_WINDOWED_ASSEMBLE_ACCEL:
        pytest.skip("windowed Z assembly accelerator not built")

    L = 2 * 0.962 * 22 / 4
    wires = [np.array([[0.0, -L / 2, 0.0], [0.0, L / 2, 0.0]])]
    kw = {
        "wires": wires,
        "n_per_edge_per_wire": [[21]],
        "nsegs": 21,
        "degree": 2,
        "wavelength": 22.0,
    }
    # Force the fallback: with the batched swept assembly available the sweep
    # never reaches the per-k route this test is about.
    monkeypatch.setattr(bmod, "_HAVE_BSPLINE_SWEPT_ASSEMBLE_ACCEL", False)

    def unexpected(*_args, **_kwargs):
        pytest.fail("unexpected dense assembly path")

    k0 = 2 * np.pi / 22.0
    k_array = np.linspace(0.94 * k0, 1.06 * k0, 4)

    budgeted = BSplineSolver(**kw)
    budgeted.swept_mem_mb = 0  # chunk = max(1, 0) -> single-row windows
    monkeypatch.setattr(budgeted, "_build_J_blocks", unexpected)
    Y_chunked = budgeted.compute_y_matrix_swept(k_array)

    tensor = BSplineSolver(**kw)
    monkeypatch.setattr(tensor, "_compute_Z_dense_chunked", unexpected)
    Y_tensor = tensor.compute_y_matrix_swept(k_array)

    rel = np.abs(Y_chunked - Y_tensor).max() / np.abs(Y_tensor).max()
    # 1e-10 margin, same cross-compiler policy as the single-k twins above.
    assert rel < 1e-10, f"chunked vs tensor swept Y disagreement: rel {rel:.2e}"


def _swept_fallback_kw():
    """One 21-segment dipole edge, the shared shape of the #263 probes."""
    L = 2 * 0.962 * 22 / 4
    wires = [np.array([[0.0, -L / 2, 0.0], [0.0, L / 2, 0.0]])]
    return {
        "wires": wires,
        "n_per_edge_per_wire": [[21]],
        "nsegs": 21,
        "degree": 2,
        "wavelength": 22.0,
    }


def test_bspline_swept_fallback_hoist_chunked_under_budget(monkeypatch):
    """The fallback sweeps' same-edge reg-moment hoist is chunked over k
    under `swept_mem_mb` (issue #263). Pre-#263 both fallback sweeps called
    `_seg_seg_reg_moments_from_geometry_swept(reg_geo, k_array)` over the
    WHOLE sweep — an O(n_k · nm² · ΣN_e²) transient the budget never saw
    (~2 GB at 600 segments × 40 k, measured). The spy trips on any hoist
    call whose k-chunk exceeds what the budget allows, so the pre-fix
    whole-sweep call fails here without needing a multi-GB allocation.
    """
    import momwire.bspline as bmod
    from momwire.bspline import BSplineSolver

    # Force the fallback: with the batched swept assembly available the sweep
    # never reaches the per-k route this test is about.
    monkeypatch.setattr(bmod, "_HAVE_BSPLINE_SWEPT_ASSEMBLE_ACCEL", False)

    kw = _swept_fallback_kw()
    n_k = 32
    k0 = 2 * np.pi / 22.0
    k_array = np.linspace(0.94 * k0, 1.06 * k0, n_k)

    budget_mb = 1
    s = BSplineSolver(**kw)
    s.swept_mem_mb = budget_mb
    # The helper's own chunk arithmetic: nm² ΣN_e² complex128 per k.
    prep = s._same_edge_prep(s._build_geometry())
    nm = s.degree + 1
    sum_ne2 = sum((sl.stop - sl.start) ** 2 for sl, _a, _arc, _radius in prep)
    max_chunk = max(1, (budget_mb << 20) // (nm * nm * sum_ne2 * 16))
    assert max_chunk < n_k, "probe must need more than one chunk"

    real = bmod._seg_seg_reg_moments_from_geometry_swept
    seen = []

    def spy(reg_geo, ks, *args, **kwargs):
        seen.append(len(ks))
        assert len(ks) <= max_chunk, (
            f"hoist over {len(ks)} k at once exceeds the {budget_mb} MB budget "
            f"(max chunk {max_chunk})"
        )
        return real(reg_geo, ks, *args, **kwargs)

    monkeypatch.setattr(bmod, "_seg_seg_reg_moments_from_geometry_swept", spy)
    s.compute_impedance_swept(k_array)
    assert len(seen) > 1 and sum(seen) == n_k

    seen.clear()
    s2 = BSplineSolver(**kw)
    s2.swept_mem_mb = budget_mb
    s2.compute_y_matrix_swept(k_array)
    assert len(seen) > 1 and sum(seen) == n_k


def test_bspline_swept_fallback_hoist_chunking_bit_identical(monkeypatch):
    """Chunking the fallback hoist is pure re-batching (issue #263): a
    multi-chunk sweep must match the single-chunk (pre-#263 whole-sweep)
    call to the last ulp, because each k's moment block is computed
    independently of its chunk mates. The 1 MB budget splits the 32-point
    sweep into two hoist chunks while the per-k (d+1,d+1,N,N) tensor
    (~63 KB here) still fits, so the #238 dispatch — a genuinely different
    assembly — stays on the same side for both solvers.
    """
    import momwire._bspline_kernels as kmod
    import momwire.bspline as bmod
    from momwire.bspline import BSplineSolver

    monkeypatch.setattr(bmod, "_HAVE_BSPLINE_SWEPT_ASSEMBLE_ACCEL", False)

    kw = _swept_fallback_kw()
    k0 = 2 * np.pi / 22.0
    k_array = np.linspace(0.94 * k0, 1.06 * k0, 32)

    def pair(entry):
        base = getattr(BSplineSolver(**kw), entry)(k_array)
        multi = BSplineSolver(**kw)
        multi.swept_mem_mb = 1
        return base, getattr(multi, entry)(k_array)

    z_base, z_multi = pair("compute_impedance_swept")
    assert np.array_equal(z_base, z_multi)
    y_base, y_multi = pair("compute_y_matrix_swept")
    assert np.array_equal(y_base, y_multi)

    # The numpy-einsum inner path (builds without the reg-swept C++ kernel)
    # must be per-k independent too.
    monkeypatch.setattr(kmod, "_HAVE_BSPLINE_REG_SWEPT_ACCEL", False)
    z_np_base, z_np_multi = pair("compute_impedance_swept")
    assert np.array_equal(z_np_base, z_np_multi)


def test_bspline_swept_batched_path_skips_fallback_hoist(monkeypatch):
    """Entry-count armor for the batched fast path (issue #263): when the
    batched swept accelerator serves, the fallback's chunked hoist helper
    must never run — `_swept_batched_z_chunks` owns its own per-chunk
    same-edge moments and is untouched by the #263 re-batching."""
    import momwire.bspline as bmod
    from momwire.bspline import BSplineSolver

    if not (
        bmod._HAVE_BSPLINE_SWEPT_ASSEMBLE_ACCEL
        and bmod._HAVE_BSPLINE_OFFEDGE_SWEPT_ACCEL
    ):
        pytest.skip("batched swept accelerators not built")

    def unexpected(*_args, **_kwargs):
        pytest.fail("batched swept path entered the fallback hoist")

    kw = _swept_fallback_kw()
    k0 = 2 * np.pi / 22.0
    k_array = np.linspace(0.94 * k0, 1.06 * k0, 4)

    s = BSplineSolver(**kw)
    assert s._swept_batched_available()
    monkeypatch.setattr(s, "_same_edge_prep_swept_chunks", unexpected)
    s.compute_impedance_swept(k_array)
    s2 = BSplineSolver(**kw)
    monkeypatch.setattr(s2, "_same_edge_prep_swept_chunks", unexpected)
    s2.compute_y_matrix_swept(k_array)


def test_bspline_swept_batched_z_chunks_holds_no_n_squared_tangent_table(monkeypatch):
    """`_swept_batched_z_chunks` must not hoist an (N, N) tangent-dot table
    for the whole sweep (issue #333, the unconverted swept-batched twin of
    #318). Pre-fix it built `td_free = tangents @ tangents.T` (8 N² bytes)
    unconditionally, plus `td_img = self._image_tangent_dot(tangents)`
    (another 8 N² bytes) when grounded — both alive across every chunk of
    the sweep, exactly like #318's `td_all` was alive across the whole
    dense fill. Post-fix the C++ kernel forms each pair's dot in-kernel
    from the (N, 3) tangent table (`assemble_Z_bspline_swept`'s new
    `tangents_row`/`tangents_col` args), so the only per-sweep tables are
    `tangents` itself and, when grounded, its O(N) mirrored copy.

    Isolating the setup cost (as opposed to the per-chunk J tensor, which
    is deliberately large and budget-governed — a different concern from
    this issue): a spy on `_seg_seg_full_moments_offedge_swept`, the first
    thing each chunk iteration allocates, snapshots tracemalloc's CURRENT
    (not peak) byte count the moment it's entered for chunk 0 — i.e.
    everything `_swept_batched_z_chunks` built before the per-chunk loop's
    first heavy allocation: `_same_edge_prep`'s per-edge `A_static` blocks
    (kept deliberately, O(ΣN_e²) — see issue #330), `tangents`, and the
    grounded mirrored copy — nothing O(N²) in segment count N.

    Geometry mirrors #318/#329's N=1200 shape (150 edges × 8 segs), grounded
    (PEC image, so BOTH retired tables would have been resident) so the
    dropped floor is the full 16 N²:

      * 16 N² = 23.04 MB is what the pre-fix code held here (measured:
        23.86 MB — the extra ~0.8 MB is the same-edge `A_static` blocks
        plus bookkeeping, present either way).
      * measured post-fix: 0.85 MB.

    3 MB sits ~3.5x above the measured 0.85 MB and ~6.6x below the 23.04 MB
    two-table floor — the same margin discipline as this file's other
    N²-transient gates (#318, #323, #329, #330).
    """
    import tracemalloc

    import momwire.bspline as bmod
    from momwire.bspline import BSplineSolver

    if not (
        bmod._HAVE_BSPLINE_SWEPT_ASSEMBLE_ACCEL
        and bmod._HAVE_BSPLINE_OFFEDGE_SWEPT_ACCEL
    ):
        pytest.skip("batched swept accelerators not built")

    n_edges, seg_per_edge = 150, 8
    L = 0.962 * 22 / 2
    ys = np.linspace(-L / 2, L / 2, n_edges + 1)
    wires = [np.column_stack([np.zeros_like(ys), ys, np.full_like(ys, 7.0)])]
    sim = BSplineSolver(
        wires=wires,
        degree=2,
        n_per_edge_per_wire=[[seg_per_edge] * n_edges],
        nsegs=n_edges * seg_per_edge,
        wavelength=22.0,
        ground_z=0.0,
    )
    assert sim._swept_batched_available()
    geom = sim._build_geometry()
    supp_seg, polys, _kcl, _wk, _wbg = sim._build_basis_polynomials(geom)
    n = supp_seg.shape[0]
    assert n == n_edges * seg_per_edge  # N = 1200

    k0 = 2 * np.pi / 22.0
    k_array = np.linspace(0.98 * k0, 1.02 * k0, 2)

    real_offedge = bmod._seg_seg_full_moments_offedge_swept
    captured = {}

    def spy(*args, **kwargs):
        cur, _peak = tracemalloc.get_traced_memory()
        captured.setdefault("setup_bytes", cur)
        return real_offedge(*args, **kwargs)

    monkeypatch.setattr(bmod, "_seg_seg_full_moments_offedge_swept", spy)

    tracemalloc.start()
    try:
        gen = sim._swept_batched_z_chunks(k_array, geom, supp_seg, polys)
        next(gen)
    finally:
        tracemalloc.stop()

    setup_bytes = captured["setup_bytes"]
    assert setup_bytes < 3_000_000, (
        f"_swept_batched_z_chunks setup peaked {setup_bytes / 1e6:.2f} MB "
        f"at N={n} (16*N**2 = {16 * n**2 / 1e6:.2f} MB is the retired "
        "two-table floor) — an N-squared tangent-dot table is back"
    )


def test_bspline_swept_same_edge_prep_holds_no_reg_table(monkeypatch):
    """`_same_edge_prep`'s return value — the list EVERY swept entry point
    (`compute_impedance_swept`, `compute_y_matrix_swept`,
    `compute_port_solution_swept`, on both the fallback and batched routes)
    holds resident for the WHOLE sweep — must not carry each edge's
    quadrature-squared reg-geometry table (issue #330). Pre-fix, each tuple
    was `(slice, A_static, reg_geometry)` where `reg_geometry["R"]` is an
    `(N_e·n_qp, N_e·n_qp)` float64 table: 128·N_e² bytes at the default
    `n_qp_pair=4`. Post-fix each tuple is `(slice, A_static, ed_arc, a_w)`
    — `ed_arc` is the (N_e+1,) arc array `geom` already owns (a reference,
    not a copy) and `a_w` a scalar, both O(N_e); the R table is rebuilt at
    consumption instead (`_same_edge_prep_swept_chunks` /
    `_swept_batched_z_chunks`) and immediately discarded, one edge at a
    time.

    Structural check (exact, not a fuzzy memory-profiler threshold): no
    tuple element is R-table-sized, `ed_arc` is exactly O(N_e), and the
    ONLY ΣN_e²-scale array retained per edge is `A_static` itself — kept
    deliberately, per the issue (72·N_e² bytes, small enough to ride the
    sweep).

    Numeric check: tracemalloc around the isolated `_same_edge_prep` call
    (captured via a spy so the sweep still runs for real, forced onto the
    fallback route — read `_swept_batched_available`'s guard, both routes
    called `_same_edge_prep` pre-fix, but the fallback is the simpler one
    to force). At n_edges=20, seg_per_edge=50 (ΣN_e² = 20·50² = 50,000):

      * measured peak is 72·ΣN_e² = 3.60 MB on the nose — exactly
        `A_static`'s own returned bytes, nothing else survives the call.
      * a re-introduced R table would add 128·ΣN_e² = 6.40 MB more.

    Those two coefficients (72 vs 128) are only 1.78x apart, tighter than
    this file's usual ~2x-above-measured / well-below-the-old-floor
    margin (issue #329's basis-build gate had an 11.5x floor to work
    with) — there just isn't a wider gap available here without loosening
    the threshold enough to hide a partial regression. 5.5 MB splits the
    difference: 1.53x above the measured 3.60 MB, 14% below the 6.40 MB
    an R table would add.
    """
    import tracemalloc

    import momwire.bspline as bmod
    from momwire.bspline import BSplineSolver

    # Force the fallback route (`_same_edge_prep_swept_chunks`); the
    # batched route (`_swept_batched_z_chunks`) held the same shape of
    # `prep` pre-fix and is covered by the bit-exactness gates elsewhere.
    monkeypatch.setattr(bmod, "_HAVE_BSPLINE_SWEPT_ASSEMBLE_ACCEL", False)

    n_edges, seg_per_edge = 20, 50
    L = 0.962 * 22 / 2 * 10
    ys = np.linspace(-L / 2, L / 2, n_edges + 1)
    wires = [np.column_stack([np.zeros_like(ys), ys, np.zeros_like(ys)])]
    sim = BSplineSolver(
        wires=wires,
        degree=2,
        n_per_edge_per_wire=[[seg_per_edge] * n_edges],
        nsegs=n_edges * seg_per_edge,
        wavelength=22.0,
    )
    assert not sim._swept_batched_available()

    real_prep = bmod.BSplineSolver._same_edge_prep
    captured = {}

    def spy(self, geom):
        tracemalloc.reset_peak()
        prep = real_prep(self, geom)
        _cur, peak = tracemalloc.get_traced_memory()
        captured["prep"] = prep
        captured["peak"] = peak
        return prep

    monkeypatch.setattr(bmod.BSplineSolver, "_same_edge_prep", spy)

    k0 = 2 * np.pi / 22.0
    k_array = np.linspace(0.94 * k0, 1.06 * k0, 4)

    tracemalloc.start()
    try:
        sim.compute_impedance_swept(k_array)
    finally:
        tracemalloc.stop()

    prep = captured["prep"]
    sum_ne2 = sum((sl.stop - sl.start) ** 2 for sl, _a, _arc, _r in prep)
    assert sum_ne2 == n_edges * seg_per_edge**2  # 50,000

    nm = sim.degree + 1
    for sl, A_st, ed_arc, a_w in prep:
        n_e = sl.stop - sl.start
        # A_static: kept deliberately, exactly (nm, nm, N_e, N_e) float64.
        assert A_st.nbytes == 8 * nm * nm * n_e * n_e
        # ed_arc: O(N_e), not O(N_e^2) — the R table is gone from the tuple.
        assert ed_arc.nbytes == 8 * (n_e + 1)
        assert isinstance(a_w, float)

    peak = captured["peak"]
    assert peak < 5_500_000, (
        f"_same_edge_prep peaked {peak / 1e6:.2f} MB "
        f"(72*sum_ne2 = {72 * sum_ne2 / 1e6:.2f} MB expected, "
        f"128*sum_ne2 = {128 * sum_ne2 / 1e6:.2f} MB is the retained-R "
        "regression) — a reg-geometry table is being retained again"
    )


def test_bspline_swept_fallback_multi_edge_matches_per_freq(monkeypatch):
    """The fallback route's rebuilt-per-chunk-per-edge reg geometry (issue
    #330) must still pick each edge's OWN `ed_arc` / `a_w` — a wrong-edge
    mistake in `_same_edge_prep_swept_chunks`'s rebuild (e.g. reusing edge
    0's arc array for every edge) is invisible on the single-edge
    geometries every other fallback test in this file uses, since there is
    only one edge to get right. This wire has FIVE edges of different
    lengths so a wrong-edge substitution changes the answer.

    Regression coverage: introducing exactly that mistake (hardcoding the
    first edge's `ed_arc` for the rebuild) passed the whole scoped suite
    but failed this comparison at ~1.7e-11 relative (three orders above
    the 1e-9 roundoff tolerance this test uses) — this test is what would
    have caught it. The tolerance itself (not `==`) matches this file's
    other swept-vs-per-k comparisons: the fallback route's per-k moments
    come from `_seg_seg_reg_moments_from_geometry_swept` (geometry
    precomputed once, moments batched over k) while `compute_impedance`'s
    direct single-k path calls `_seg_seg_reg_moments` (quadrature done in
    one shot) — same algebra, different floating-point reduction order.
    """
    import momwire.bspline as bmod
    from momwire.bspline import BSplineSolver

    monkeypatch.setattr(bmod, "_HAVE_BSPLINE_SWEPT_ASSEMBLE_ACCEL", False)

    # Five straight runs of different lengths on one wire (10, 8, 12, 6, 9
    # segments), each a distinct edge with its own arc array.
    seg_counts = [10, 8, 12, 6, 9]
    step = 22.0 / 100  # short enough segments to stay well inside one wavelength
    ys = [0.0]
    for n in seg_counts:
        ys.append(ys[-1] + n * step)
    wires = [np.array([[0.0, y, 0.0] for y in ys])]
    sim = BSplineSolver(
        wires=wires,
        degree=2,
        n_per_edge_per_wire=[seg_counts],
        nsegs=sum(seg_counts),
        wavelength=22.0,
    )
    assert not sim._swept_batched_available()

    k0 = 2 * np.pi / 22.0
    k_array = np.linspace(0.95 * k0, 1.05 * k0, 6)
    z_swept = sim.compute_impedance_swept(k_array)

    for kk, zs in zip(k_array, z_swept):
        sim_k = BSplineSolver(
            wires=wires,
            degree=2,
            n_per_edge_per_wire=[seg_counts],
            nsegs=sum(seg_counts),
            wavelength=2 * np.pi / kk,
        )
        z_k, _ = sim_k.compute_impedance()
        assert abs(zs - z_k) <= 1e-9 * abs(z_k), f"k={kk}: swept={zs}, per-k={z_k}"


# ---- Swept size dispatch on the accelerated solvers (issue #262) ------------
#
# Below `HMatrixSolver.SWEPT_DENSE_MAX_BASES` both swept entry points hand the
# sweep to the batched dense route instead of rebuilding the ACA operator per
# frequency. The gates: the two routes agree (it is a performance dispatch,
# not a physics change), BOTH entry points read the SAME predicate (#241 was
# exactly the defect of one solver answering swept Y and swept Z on two
# engines), and the ceiling stays inside the memory the dense route needs.


def _dispatch_kw(nsegs=21, **extra):
    """Two gap-fed dipoles, `nsegs` bases each — far below the dispatch
    ceiling either way, so the default construction dispatches and
    `swept_dense_max_bases=0` does not. Two ports so the Y gate has an
    off-diagonal to disagree about.

    The routing gates take the 21 (42 bases) default: cheap, and routing does
    not care about block structure. The equality gate takes 40 (80 bases),
    which is the smallest size on this geometry whose partition has any
    ADMISSIBLE blocks at all — below it every block is near/dense, the ACA
    tolerance never enters, and the two routes agree to 1e-13 for a reason
    that has nothing to do with the dispatch being right.
    """
    L = 2 * 0.962 * 22 / 4
    wires = [
        np.array([[0.0, -L / 2, 0.0], [0.0, L / 2, 0.0]]),
        np.array([[3.0, -L / 2, 0.0], [3.0, L / 2, 0.0]]),
    ]
    return dict(
        wires=wires,
        n_per_edge_per_wire=[[nsegs], [nsegs]],
        nsegs=nsegs,
        degree=2,
        wavelength=22.0,
        feeds=[(0, None, 1.0 + 0j), (1, None, 0.5 - 0.25j)],
        **extra,
    )


def _dispatch_ks():
    k0 = 2 * np.pi / 22.0
    return np.linspace(0.94 * k0, 1.06 * k0, 4)


@pytest.mark.parametrize(
    "ground_kw", [{}, {"ground_z": 0.0, "z_offset": 2.2}], ids=["free", "pec"]
)
def test_swept_dense_dispatch_agrees_with_the_accelerated_route(ground_kw):
    """The dispatched (dense) sweep and the forced-accelerated sweep are the
    same answer, Z and Y, in free space and over PEC ground.

    This is the gate that makes #262 a *policy* change: the two engines
    differ by the accelerator's own approximations (ACA `aca_tol`, GMRES
    `solve_tol`) and by nothing else, so the bound here is the iterative
    tolerance, not machine precision. Measured on this box at the default
    knobs: 1.0e-9 (Z) / 2.8e-9 (Y) free space, 6.3e-10 / 1.7e-9 over PEC on
    this 80-basis fixture, and 1.5e-7 on a 240- and an 800-basis array where
    the far blocks carry more of the matrix. The pin is loose enough for a
    cross-compiler BLAS and for a bigger model's share of ACA, and far
    tighter than any real routing mistake — a sweep on different geometry or
    a shuffled port order blows through 1e-4 by orders.
    """
    from momwire.hmatrix import HMatrixSolver

    z_off = ground_kw.pop("z_offset", 0.0)
    kw = _dispatch_kw(nsegs=40, **ground_kw)
    if z_off:
        kw["wires"] = [w + np.array([0.0, 0.0, z_off]) for w in kw["wires"]]
    ks = _dispatch_ks()

    dispatched = HMatrixSolver(**kw)
    assert dispatched._swept_prefers_dense(), "fixture must be under the ceiling"
    accelerated = HMatrixSolver(**kw, swept_dense_max_bases=0)
    assert not accelerated._swept_prefers_dense(), "escape hatch must pin accel"
    # The accelerated side must really be compressing something, or this gate
    # compares a dense solve with a dense solve and passes for free.
    assert accelerated.build_partition()["far"], "fixture has no admissible blocks"

    z_d = dispatched.compute_impedance_swept(ks)
    z_a = accelerated.compute_impedance_swept(ks)
    rel_z = np.abs(z_d - z_a).max() / np.abs(z_a).max()
    assert rel_z < 1e-4, f"dispatched vs accelerated swept Z: rel {rel_z:.3e}"

    y_d = HMatrixSolver(**kw).compute_y_matrix_swept(ks)
    y_a = HMatrixSolver(**kw, swept_dense_max_bases=0).compute_y_matrix_swept(ks)
    rel_y = np.abs(y_d - y_a).max() / np.abs(y_a).max()
    assert rel_y < 1e-4, f"dispatched vs accelerated swept Y: rel {rel_y:.3e}"


def test_swept_dense_dispatch_is_one_predicate_for_both_entry_points(monkeypatch):
    """Z and Y route off the SAME predicate — forced both ways.

    `_swept_prefers_dense` is monkeypatched True and then False, and each
    entry point is checked to land on the engine the predicate names. A
    mutation that hard-codes either route (an `if n_basis <= …` re-spelled
    inside `compute_impedance_swept`, a `_port_solutions_swept` that forgets
    to ask) fails one half of this: the forced predicate would no longer move
    that entry point, and its trap fires.
    """
    from momwire.hmatrix import HMatrixSolver

    kw = _dispatch_kw()
    ks = _dispatch_ks()

    def _trap(name):
        def go(*_a, **_kw):
            pytest.fail(f"swept route took the wrong engine: {name}")

        return go

    def _spy(solver, name):
        seen = []
        real = getattr(solver, name)

        def wrapper(*a, **kw_):
            seen.append(1)
            return real(*a, **kw_)

        monkeypatch.setattr(solver, name, wrapper)
        return seen

    # --- predicate True: both entry points must reach the batched dense route
    for entry, dense_marker, accel_entry in (
        (
            "compute_impedance_swept",
            "_compute_impedance_swept_batched",
            "compute_impedance",
        ),
        ("compute_y_matrix_swept", "_swept_batched_z_chunks", "compute_port_solution"),
    ):
        s = HMatrixSolver(**kw)
        monkeypatch.setattr(s, "_swept_prefers_dense", lambda: True)
        seen = _spy(s, dense_marker)
        monkeypatch.setattr(s, accel_entry, _trap(f"{entry} stayed accelerated"))
        getattr(s, entry)(ks)
        assert seen, f"{entry}: predicate True but the dense route never ran"

    # --- predicate False: both entry points must rebuild the operator per k
    for entry, dense_marker in (
        ("compute_impedance_swept", "_compute_impedance_swept_batched"),
        ("compute_y_matrix_swept", "_swept_batched_z_chunks"),
    ):
        s = HMatrixSolver(**kw)
        monkeypatch.setattr(s, "_swept_prefers_dense", lambda: False)
        monkeypatch.setattr(s, dense_marker, _trap(f"{entry} went dense"))
        built = _spy(s, "_build_operator")
        getattr(s, entry)(ks)
        assert len(built) == len(ks), f"{entry}: one accelerated operator per k"


def test_swept_dense_dispatch_ceiling_respects_the_dense_memory_it_implies():
    """Provenance gate for `SWEPT_DENSE_MAX_BASES` (#262).

    The constant is a measurement (the table lives in `_swept_prefers_dense`),
    but the CAP on it is arithmetic, and that part a test can hold:

    * a single-k dense Z at the ceiling is n²·16 B — must stay well inside
      what a laptop can spare;
    * the batched sweep's real peak is the (chunk, (d+1)², N, N) moment
      tensor, and `_swept_batched_z_chunks` floors `chunk` at 1, so ONE k's
      tensor at the ceiling must still fit the default `swept_mem_mb` (256 MB)
      at degree 2 — past that point the #263 budget stops being honoured and
      the dense peaks leave the few-hundred-MB range (measured: 919 MB at
      1,600 bases, 3.7 GB at 3,360);
    * #143's 12,682-basis whip — the model this accelerator exists for —
      must never dispatch, at any construction.
    """
    from momwire.hmatrix import HMatrixSolver

    cap = HMatrixSolver.SWEPT_DENSE_MAX_BASES
    assert cap == 1024, "the ceiling is a measured constant; re-measure to move it"

    dense_z_mb = cap * cap * 16 / (1 << 20)
    assert dense_z_mb < 32, f"single-k dense Z at the ceiling: {dense_z_mb:.1f} MB"

    default_budget_mb = BSplineSolver(**_dispatch_kw()).swept_mem_mb
    moment_mb = (2 + 1) ** 2 * cap * cap * 16 / (1 << 20)
    assert moment_mb <= default_budget_mb, (
        f"one k's d=2 moment tensor at the ceiling is {moment_mb:.0f} MB, over "
        f"the {default_budget_mb} MB default budget the chunk floor cannot honour"
    )

    whip = HMatrixSolver(**_dispatch_kw())
    whip._hm_context = {**whip._context(), "n_basis": 12682}
    assert not whip._swept_prefers_dense(), "#143's whip must never dispatch dense"


def test_swept_dense_dispatch_escape_hatch_and_array_block_opt_out():
    """The knob and the subclass opt-out, both directions.

    `swept_dense_max_bases=None` takes the class default, 0 pins the
    accelerated route at every size, and `ArrayBlockSolver` never dispatches
    at all — its wall clock against the dense sweep is set by array structure,
    not basis count, so the H-matrix threshold does not speak for it.
    """
    from momwire.array_block import ArrayBlockSolver
    from momwire.hmatrix import HMatrixSolver

    kw = _dispatch_kw()
    assert HMatrixSolver(**kw).swept_dense_max_bases == (
        HMatrixSolver.SWEPT_DENSE_MAX_BASES
    )
    assert HMatrixSolver(**kw, swept_dense_max_bases=0).swept_dense_max_bases == 0
    assert HMatrixSolver(**kw, swept_dense_max_bases=16).swept_dense_max_bases == 16
    # 16 is under this fixture's 42 bases, so the ceiling really is consulted —
    # the knob is a threshold, not just an on/off.
    assert not HMatrixSolver(**kw, swept_dense_max_bases=16)._swept_prefers_dense()
    assert HMatrixSolver(**kw, swept_dense_max_bases=42)._swept_prefers_dense()

    arr = ArrayBlockSolver(**kw)
    assert not arr._swept_dense_dispatch_ok()
    assert not arr._swept_prefers_dense(), "array sweeps stay on the array operator"


def _grounded_window_kw(**ground_kw):
    """The grounded chunked-image deck: a 17-seg INVERTED-V over ground.

    Bent, not the straight ŷ dipole it started as, because a straight one
    leaves t_z ≡ 0 — and then the M = diag(1, 1, −1) image mirror is the
    identity on every tangent, so dropping its sign in the image weights is
    numerically invisible and none of the gates below can see it (the #249
    trap from the other side, where an all-VERTICAL deck could not tell a
    joint mirror labelling from a free-space one). The apex lift gives every
    segment a z-component; the two legs give the refl-coef dyad a spread of
    specular angles. Height 2.2 m keeps the whole wire strictly above ground,
    which the Sommerfeld branch requires.
    """
    L = 2 * 0.962 * 22 / 4
    h = 2.2
    apex = np.array([[0.0, -L / 2, h], [0.0, 0.0, h + 0.6 * L], [0.0, L / 2, h]])
    return dict(
        wires=[apex],
        n_per_edge_per_wire=[[8, 9]],
        nsegs=17,
        degree=2,
        wavelength=22.0,
        ground_z=0.0,
        **ground_kw,
    )


@pytest.mark.parametrize(
    "ground_kw",
    [
        {},
        {"ground_eps": (10.0, 0.002)},
        {"ground_eps": (10.0, 0.002), "ground_model": "sommerfeld"},
    ],
    ids=["pec", "refl-coef", "sommerfeld"],
)
def test_bspline_chunked_ground_y_matrix_matches_tensor_path(ground_kw):
    """Grounded compute_y_matrix, forced chunked vs windowed-flags-off —
    the Y twin of the grounded impedance test below, pinning that the
    issue-#235 shared-operator routing preserves every ground branch
    (the flags-off side reproduces the pre-#235 Y code path exactly:
    image tensor + `_ground_finite_Z`)."""
    import momwire.bspline as bmod
    from momwire.bspline import BSplineSolver

    if not bmod._HAVE_BSPLINE_W_WINDOWED_ASSEMBLE_ACCEL:
        pytest.skip("weighted windowed Z assembly accelerator not built")

    kw = _grounded_window_kw(**ground_kw)
    chunked = BSplineSolver(**kw)
    chunked.swept_mem_mb = 0
    Y_chunked = chunked.compute_y_matrix()

    saved = (
        bmod._HAVE_BSPLINE_WINDOWED_ASSEMBLE_ACCEL,
        bmod._HAVE_BSPLINE_W_WINDOWED_ASSEMBLE_ACCEL,
    )
    try:
        bmod._HAVE_BSPLINE_WINDOWED_ASSEMBLE_ACCEL = False
        bmod._HAVE_BSPLINE_W_WINDOWED_ASSEMBLE_ACCEL = False
        Y_tensor = BSplineSolver(**kw).compute_y_matrix()
    finally:
        (
            bmod._HAVE_BSPLINE_WINDOWED_ASSEMBLE_ACCEL,
            bmod._HAVE_BSPLINE_W_WINDOWED_ASSEMBLE_ACCEL,
        ) = saved

    rel = np.abs(Y_chunked - Y_tensor).max() / np.abs(Y_tensor).max()
    # 1e-10 — same cross-compiler reduction-order policy as the grounded
    # impedance twin.
    assert rel < 1e-10, f"chunked vs tensor grounded Y disagreement: rel {rel}"


@pytest.mark.parametrize(
    "ground_kw",
    [
        {},
        {"ground_eps": (10.0, 0.002)},
        {"ground_eps": (10.0, 0.002), "ground_model": "sommerfeld"},
    ],
    ids=["pec", "refl-coef", "sommerfeld"],
)
def test_bspline_chunked_ground_matches_tensor_path(ground_kw):
    """Grounded compute_impedance through a forced chunked image path vs the
    image-tensor path (windowed flags flipped off)
    — PEC mirror-dot, Fresnel refl-coef tables, and the Sommerfeld
    constant-C2 exact image all route through the weighted windowed
    accumulator with scale = -1."""
    import momwire.bspline as bmod
    from momwire.bspline import BSplineSolver

    if not bmod._HAVE_BSPLINE_W_WINDOWED_ASSEMBLE_ACCEL:
        pytest.skip("weighted windowed Z assembly accelerator not built")

    kw = _grounded_window_kw(**ground_kw)
    chunked = BSplineSolver(**kw)
    chunked.swept_mem_mb = 0
    z_chunked, _ = chunked.compute_impedance()

    saved = (
        bmod._HAVE_BSPLINE_WINDOWED_ASSEMBLE_ACCEL,
        bmod._HAVE_BSPLINE_W_WINDOWED_ASSEMBLE_ACCEL,
    )
    try:
        bmod._HAVE_BSPLINE_WINDOWED_ASSEMBLE_ACCEL = False
        bmod._HAVE_BSPLINE_W_WINDOWED_ASSEMBLE_ACCEL = False
        z_tensor, _ = BSplineSolver(**kw).compute_impedance()
    finally:
        (
            bmod._HAVE_BSPLINE_WINDOWED_ASSEMBLE_ACCEL,
            bmod._HAVE_BSPLINE_W_WINDOWED_ASSEMBLE_ACCEL,
        ) = saved

    rel = abs(z_chunked - z_tensor) / abs(z_tensor)
    # 1e-10, not 1e-12: the weighted windowed kernel accumulates the image
    # contribution in complex sums whose rounding is compiler-dependent —
    # MSVC and AppleClang land at 2e-12..8e-12 vs gcc's <1e-12 (wheels run
    # 29345985527). Still 5+ orders below physical tolerance; this pins
    # the algebra, not the reduction order.
    assert rel < 1e-10, f"chunked vs tensor grounded Z disagreement: rel {rel}"


def _image_weight_dense_tables(sim, geom):
    """The (N, N) image weight tables the tensor/enrichment paths build —
    the reference `_image_weight_window_fn`'s windows must reproduce."""
    from momwire import _ground_refl

    if sim.ground_eps is None:
        w_A = sim._image_tangent_dot(geom["tangents"]).astype(np.complex128)
        return w_A, np.ones_like(w_A)
    if sim.ground_model == "sommerfeld":
        eps_t = _ground_refl.eps_tilde(sim.ground_eps, sim.omega, sim.eps)
        c2 = (eps_t - 1.0) / (eps_t + 1.0)
        w_A = c2 * sim._image_tangent_dot(geom["tangents"])
        return w_A, np.full_like(w_A, c2)
    return sim._image_refl_weights(sim._image_refl_prep(geom), sim.omega)


def _phi_mode_window_cases():
    """One case per ground mode, refl-coef split by `ground_phi_mode`."""
    from momwire import _ground_refl

    eps = (10.0, 0.002)
    return [
        ({}, "pec"),
        ({"ground_eps": eps, "ground_model": "sommerfeld"}, "sommerfeld"),
        *[
            ({"ground_eps": eps, "ground_phi_mode": m}, f"refl-{m}")
            for m in _ground_refl.PHI_MODES
        ],
    ]


@pytest.mark.parametrize(
    "ground_kw",
    [kw for kw, _ in _phi_mode_window_cases()],
    ids=[name for _, name in _phi_mode_window_cases()],
)
def test_bspline_image_weight_windows_match_the_dense_tables(ground_kw):
    """`_image_weight_window_fn` is row-local: window == dense rows (#323).

    The chunked image fill used to be handed the same global (N, N) w_A/w_Phi
    tables the tensor path builds and slice them per chunk — 2× the dense Z
    resident for weights only ever read one row-band at a time. The windows
    are now produced per chunk instead, so the equality that used to be free
    (it *was* a slice) is what needs pinning: each mode's per-chunk algebra
    must land on exactly the rows of its dense table, including the
    frequency-only `eps_tilde` lift out of the loop and the rectangular
    `specular_pair_tables` block the refl-coef mode uses in place of the
    square `_image_refl_prep` cache.

    Every `ground_phi_mode` is covered because two of the four
    (`image`, `normal`) make `phi_term_weights` return a SCALAR — the
    broadcast to window shape is a per-mode branch, not shared code.
    """
    from momwire.bspline import BSplineSolver

    sim = BSplineSolver(**_grounded_window_kw(**ground_kw))
    geom = sim._build_geometry()
    n_segs = geom["n_segs_total"]
    w_A_dense, w_Phi_dense = _image_weight_dense_tables(sim, geom)
    weights_fn = sim._image_weight_window_fn(geom)

    # Whole mesh, a 1-row window at the origin, an interior 1-row window
    # (i0 > 0 — the sliced-source alignment a row-local producer can get
    # wrong), and an interior band.
    for i0, i1 in [(0, n_segs), (0, 1), (n_segs // 2, n_segs // 2 + 1), (3, 9)]:
        w_A_win, w_Phi_win = weights_fn(i0, i1)
        for win, dense, name in (
            (w_A_win, w_A_dense, "w_A"),
            (w_Phi_win, w_Phi_dense, "w_Phi"),
        ):
            assert win.shape == (i1 - i0, n_segs), f"{name} window shape [{i0}, {i1})"
            assert win.dtype == np.complex128 and win.flags.c_contiguous, (
                f"{name} window must reach the assembler as C-contiguous "
                f"complex128, got {win.dtype} contiguous={win.flags.c_contiguous}"
            )
            np.testing.assert_allclose(
                win,
                dense[i0:i1],
                rtol=1e-13,
                atol=0.0,
                err_msg=f"{name} window [{i0}, {i1}) != dense rows",
            )


@pytest.mark.parametrize(
    "ground_kw",
    [
        {},
        {"ground_eps": (10.0, 0.002)},
        {"ground_eps": (10.0, 0.002), "ground_model": "sommerfeld"},
    ],
    ids=["pec", "refl-coef", "sommerfeld"],
)
def test_bspline_grounded_tensor_route_dispatch_is_call_gated(ground_kw, monkeypatch):
    """issue #273: `_build_J_image_blocks` (and `_image_Z_refl`/
    `_ground_finite_Z` under it) must fire exactly when
    `_HAVE_BSPLINE_W_WINDOWED_ASSEMBLE_ACCEL` is False and must never fire
    when it is True — the dispatch condition `_compute_Z_operator` actually
    branches on. #273's mutation-coverage hole was that on an accelerated
    build (this box) no test forced that condition, so a defect confined to
    the fallback chain was invisible to `compute_impedance()`; the two
    assertions below are the reachability tripwire that keeps that hole
    from reopening silently, and the rel-diff pins the fallback answers the
    accelerated route to the same tolerance the (pre-existing)
    `test_bspline_chunked_ground_matches_tensor_path` above already holds
    for the internal-consistency side of this."""
    import momwire.bspline as bmod
    from momwire.bspline import BSplineSolver

    if not bmod._HAVE_BSPLINE_W_WINDOWED_ASSEMBLE_ACCEL:
        pytest.skip("weighted windowed Z assembly accelerator not built")

    kw = _grounded_window_kw(**ground_kw)

    calls = []
    orig = BSplineSolver._build_J_image_blocks

    def spy(self, *a, **kw2):
        calls.append(1)
        return orig(self, *a, **kw2)

    monkeypatch.setattr(BSplineSolver, "_build_J_image_blocks", spy)

    # Accelerated route (the default on this box): the fallback must be
    # dark.
    z_accel, _ = BSplineSolver(**kw).compute_impedance()
    assert calls == [], (
        f"{ground_kw}: the accelerated route called _build_J_image_blocks "
        f"{len(calls)} time(s) — it must stay on _accumulate_Z_image_chunked"
    )

    # Forced fallback: the tensor route must fire exactly once.
    calls.clear()
    saved = (
        bmod._HAVE_BSPLINE_WINDOWED_ASSEMBLE_ACCEL,
        bmod._HAVE_BSPLINE_W_WINDOWED_ASSEMBLE_ACCEL,
    )
    try:
        bmod._HAVE_BSPLINE_WINDOWED_ASSEMBLE_ACCEL = False
        bmod._HAVE_BSPLINE_W_WINDOWED_ASSEMBLE_ACCEL = False
        z_forced, _ = BSplineSolver(**kw).compute_impedance()
    finally:
        (
            bmod._HAVE_BSPLINE_WINDOWED_ASSEMBLE_ACCEL,
            bmod._HAVE_BSPLINE_W_WINDOWED_ASSEMBLE_ACCEL,
        ) = saved
    assert calls == [1], (
        f"{ground_kw}: the forced fallback called _build_J_image_blocks "
        f"{len(calls)} time(s), want exactly 1"
    )

    rel = abs(z_forced - z_accel) / abs(z_accel)
    # Same 1e-10 cross-compiler reduction-order policy as the chunked-vs-
    # tensor pin above (this is that same comparison, plus the call-count
    # tripwire on top).
    assert rel < 1e-10, f"{ground_kw}: forced-fallback vs accelerated rel {rel:.2e}"


# --------------------------------------------------------------------------
# Enrichment through the Y-matrix path (issue #165)
# --------------------------------------------------------------------------


def _hentenna_enrich_kwargs(
    n=21,
    feeds=None,
    variant="raw",
    ground_clearance=None,
    ground_eps=None,
    ground_model=None,
):
    """The K=3-junction hentenna fixture (same geometry as the enrichment
    convergence tests above), packaged for the Y-matrix tests. `feeds`
    overrides the single default feed with explicit (wire, arc, V) tuples.

    `ground_clearance` (in wavelengths), when set, adds a ground plane that
    many below the antenna's lowest wire — i.e. `ground_z = z_bot -
    ground_clearance·λ` — exercising the enrichment image reaction (#167)
    while keeping every wire strictly above the plane. `ground_eps` (a
    complex ε̃ or (eps_r, sigma) tuple) selects the fast finite ground;
    None leaves the plane PEC."""
    C_LIGHT = 299_792_458.0
    freq_mhz = 28.47
    wavelength = C_LIGHT / (freq_mhz * 1e6)
    width_factor = 0.1378
    top_height_factor = 0.5081
    mid_height_factor = 0.1094
    eps_feed = 0.05
    half_w = wavelength * width_factor / 2
    z_mid = wavelength * (mid_height_factor - top_height_factor)
    z_bot = -wavelength * top_height_factor
    A = (0.0, half_w, 0.0)
    B_ = (0.0, half_w, z_mid)
    F = (0.0, half_w, z_bot)
    S = (0.0, eps_feed, z_mid)
    C_ = (0.0, -half_w, 0.0)
    D = (0.0, -half_w, z_mid)
    E_ = (0.0, -half_w, z_bot)
    T = (0.0, -eps_feed, z_mid)
    wires = [
        np.array([T, S], dtype=float),
        np.array([S, B_], dtype=float),
        np.array([B_, A, C_, D], dtype=float),
        np.array([T, D], dtype=float),
        np.array([D, E_, F, B_], dtype=float),
    ]
    junctions = [
        [(0, "end"), (1, "start")],
        [(0, "start"), (3, "start")],
        [(1, "end"), (2, "start"), (4, "end")],
        [(2, "end"), (3, "end"), (4, "start")],
    ]
    npe = [[3], [n], [n, n, n], [n], [n, n, n]]
    kw = dict(
        degree=2,
        wires=wires,
        n_per_edge_per_wire=npe,
        wavelength=wavelength,
        wire_radius=0.0005,
        nsegs=n,
        junctions=junctions,
        use_singular_enrichment=True,
        enrichment_variant=variant,
    )
    if feeds is None:
        kw.update(feed_wire_index=0, feed_arclength=eps_feed)
    else:
        kw["feeds"] = feeds
    if ground_clearance is not None:
        kw["ground_z"] = z_bot - ground_clearance * wavelength
    if ground_eps is not None:
        kw["ground_eps"] = ground_eps
    if ground_model is not None:
        kw["ground_model"] = ground_model
    return kw


@pytest.mark.parametrize(
    "ground_kw",
    [
        {},
        {"ground_eps": (13.0, 0.005)},
        {"ground_eps": (13.0, 0.005), "ground_model": "sommerfeld"},
    ],
    ids=["pec", "refl-coef", "sommerfeld"],
)
def test_bspline_enrichment_image_weight_blocks_match_the_dense_tables(ground_kw):
    """`_image_weight_enrich_blocks`'s row/col/ee sub-blocks are exactly the
    (N, n_enrich)-scale slices of the (N, N) dense weight tables the tensor
    path builds (issue #328). `_enrichment_Z_assemble`'s `ground_z is not
    None` branch used to build the full `w_A_all`/`w_Phi_all` tables
    `_image_weight_dense_tables` reproduces here, and only ever read the
    three sub-blocks pinned below out of them — the enrichment analog of
    `test_bspline_image_weight_windows_match_the_dense_tables` (#323), for
    the sub-block form instead of the row-window form.

    A git-stash harness run (this fixture, n=21, all three modes) compared
    `_enrichment_Z_assemble`'s full (Z_pe, Z_ep, Z_ee) output against the
    pre-#328 code with `np.array_equal` and found ZERO bit difference in
    every one of the nine matrices — this test is the permanent regression
    guard for that result once the pre-#328 code is gone.
    """
    from momwire import _ground_refl

    kw = _hentenna_enrich_kwargs(ground_clearance=0.25, **ground_kw)
    sim = BSplineSolver(**kw)
    geom = sim._build_geometry()
    specs = sim._enrichment_specs(geom)
    n_enrich = len(specs)
    assert n_enrich > 0  # guard the guard: this deck must carry enrichment DOFs
    seg_e_arr = np.fromiter((s[3] for s in specs), dtype=np.int64, count=n_enrich)

    w_A_dense, w_Phi_dense = _image_weight_dense_tables(sim, geom)
    eps_t = (
        _ground_refl.eps_tilde(sim.ground_eps, sim.omega, sim.eps)
        if sim.ground_eps is not None
        else None
    )
    w_A_row, w_Phi_row, w_A_col, w_Phi_col, w_A_ee, w_Phi_ee = (
        sim._image_weight_enrich_blocks(geom, seg_e_arr, eps_t=eps_t)
    )

    n_segs = geom["n_segs_total"]
    cases = [
        (
            "row",
            w_A_row,
            w_Phi_row,
            w_A_dense[seg_e_arr, :],
            w_Phi_dense[seg_e_arr, :],
            (n_enrich, n_segs),
        ),
        (
            "col",
            w_A_col,
            w_Phi_col,
            w_A_dense[:, seg_e_arr],
            w_Phi_dense[:, seg_e_arr],
            (n_segs, n_enrich),
        ),
        (
            "ee",
            w_A_ee,
            w_Phi_ee,
            w_A_dense[np.ix_(seg_e_arr, seg_e_arr)],
            w_Phi_dense[np.ix_(seg_e_arr, seg_e_arr)],
            (n_enrich, n_enrich),
        ),
    ]
    for name, w_A_blk, w_Phi_blk, w_A_ref, w_Phi_ref, shape in cases:
        for blk, ref, label in (
            (w_A_blk, w_A_ref, "w_A"),
            (w_Phi_blk, w_Phi_ref, "w_Phi"),
        ):
            assert blk.shape == shape, f"{name} {label} shape {blk.shape} != {shape}"
            assert blk.dtype == np.complex128 and blk.flags.c_contiguous, (
                f"{name} {label} must reach the assembler as C-contiguous "
                f"complex128, got {blk.dtype} contiguous={blk.flags.c_contiguous}"
            )
            np.testing.assert_allclose(
                blk,
                ref,
                rtol=1e-13,
                atol=0.0,
                err_msg=f"{name} {label} != dense sub-block",
            )


def _stub_sommerfeld_remainder_enrich(
    self, geom, supp_seg_poly, polys_poly, spec_seg, spec_origin, eps_t
):
    """Zero-returning stand-in for `_Q_sommerfeld_remainder_enrich`, used
    only to keep its field-projection array — a separate, already-addressed
    O(N) transient, not the (N, N) tables issue #328 is about, but big
    enough on its own (double-digit MB at the N below) to swamp a tight
    budget — out of the trace in
    `test_bspline_enrichment_image_fill_holds_no_n_squared_transient`. It
    runs AFTER `_enrichment_Z_assemble`'s ground weight-table branch, so
    stubbing it changes nothing about what that test actually gates."""
    n_poly = supp_seg_poly.shape[0]
    n_enrich = spec_seg.shape[0]
    z = np.complex128
    return (
        np.zeros((n_poly, n_enrich), dtype=z),
        np.zeros((n_enrich, n_poly), dtype=z),
        np.zeros((n_enrich, n_enrich), dtype=z),
    )


@pytest.mark.parametrize(
    "ground_kw",
    [
        {},
        {"ground_eps": (13.0, 0.005)},
        {"ground_eps": (13.0, 0.005), "ground_model": "sommerfeld"},
    ],
    ids=["pec", "refl-coef", "sommerfeld"],
)
def test_bspline_enrichment_image_fill_holds_no_n_squared_transient(
    ground_kw, monkeypatch
):
    """The grounded enrichment reaction must not allocate anything N²-scale
    (issue #328). `_enrichment_Z_assemble`'s `ground_z is not None` branch
    used to build the same global (N, N) `w_A_all`/`w_Phi_all` weight
    tables the tensor path builds — for all three ground modes — even
    though `_assemble_Z_enrich_image_numpy` only ever reads (N, n_enrich),
    (n_enrich, N) and (n_enrich, n_enrich) sub-blocks of them (n_enrich = a
    handful of enrichment DOFs, one per K≥`enrichment_min_k` junction).

    The traced call is `_enrichment_Z_assemble` itself — the real
    orchestration entry point, not an internal helper — so a regression
    that reintroduces a full table build inside it (the likeliest
    regression site: e.g. "simplifying" back to the tensor path's tables)
    is caught here, not just in the sub-block equality test above.

    The Sommerfeld smooth-remainder step (`_Q_sommerfeld_remainder_enrich`)
    is monkeypatched to `_stub_sommerfeld_remainder_enrich` for the
    duration of the trace: it runs AFTER the weight-table branch this issue
    is about, and its own field-projection array is a separate,
    already-addressed O(N) (not O(N²)) transient that would otherwise push
    every mode's peak well past any budget tight enough to catch the actual
    regression — the same kind of exclusion `_image_weight_window_fn`'s
    #323 gate makes for `_Z_sommerfeld_remainder`, done here via monkeypatch
    (rather than a narrower call) because the real entry point bundles both
    steps together and the point of this test is to trace that entry point.

    Arithmetic at the N = 803 basis functions this hentenna deck builds
    (n=100; n_enrich=6 K≥3 junctions; tracemalloc sees numpy's data
    allocations; one (N, N) complex128 table is 16 N² = 10.3 MB):

      * measured against the pre-#328 code (identical stub, single real
        call to `_enrichment_Z_assemble`): pec 21.13 MB, refl-coef
        62.19 MB, sommerfeld 26.34 MB.
      * measured against the fix: pec 0.81 MB, refl-coef 0.88 MB,
        sommerfeld 0.86 MB — flat across modes, as expected when nothing
        scales with N.

    6 MB sits ~7x above the fixed peak (worst mode) and ~3.5x below the
    smallest pre-fix failure (PEC — the cheapest per-pair weight, and so
    the tightest margin of the three modes).
    """
    import tracemalloc

    monkeypatch.setattr(
        BSplineSolver,
        "_Q_sommerfeld_remainder_enrich",
        _stub_sommerfeld_remainder_enrich,
    )

    kw = _hentenna_enrich_kwargs(n=100, ground_clearance=0.25, **ground_kw)
    sim = BSplineSolver(**kw)
    geom = sim._build_geometry()
    supp_seg, polys, _kcl, _wk, _wbg = sim._build_basis_polynomials(geom)
    n_segs = geom["n_segs_total"]

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        enrich = sim._enrichment_Z_assemble(geom, supp_seg, polys)
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # Guard the guard: a fill that computed nothing would pass on
    # allocation while measuring nothing at all.
    assert enrich is not None and np.any(enrich["Z_ee"] != 0.0)
    assert peak < 6_000_000, (
        f"enrichment image fill peaked {peak / 1e6:.2f} MB "
        f"(one (N, N) complex table = {16 * n_segs**2 / 1e6:.2f} MB) — an "
        "N-squared weight residency is back"
    )


@pytest.mark.parametrize("variant", ["raw", "stable", "tikhonov", "auto"])
def test_bspline_enrichment_y_matrix_matches_impedance(variant):
    """Single feed: 1/Y[0,0] must equal compute_impedance's Z essentially
    exactly — both paths solve the SAME augmented system with the same
    poly-block-restricted readout, so any gap is an implementation drift,
    not physics. For "auto", the pass-1 union over one port column is the
    same set compute_impedance selects, pinned via _auto_active_junctions."""
    z_imp, _ = BSplineSolver(
        **_hentenna_enrich_kwargs(variant=variant)
    ).compute_impedance()
    s = BSplineSolver(**_hentenna_enrich_kwargs(variant=variant))
    Y = s.compute_y_matrix()
    assert Y.shape == (1, 1)
    assert abs(1.0 / Y[0, 0] - z_imp) / abs(z_imp) < 1e-9, (1.0 / Y[0, 0], z_imp)
    if variant == "auto":
        s2 = BSplineSolver(**_hentenna_enrich_kwargs(variant=variant))
        s2.compute_impedance()
        assert s._auto_active_junctions == s2._auto_active_junctions


@pytest.mark.parametrize("variant", ["raw", "stable", "tikhonov", "auto"])
def test_bspline_enrichment_pec_ground_y_matrix_identity(variant):
    """The #165 oracle with a PEC ground plane (#167): 1/Y[0,0] must still
    equal compute_impedance's Z, now that both border the enrichment blocks
    with their ground-image reaction subtracted. If the image reflection were
    inconsistent between the impedance Schur solve and the Y-matrix port
    solve the identity would break — it is the tightest self-consistency
    check on the new blocks."""
    kw = _hentenna_enrich_kwargs(variant=variant, ground_clearance=0.25)
    z_imp, _ = BSplineSolver(**kw).compute_impedance()
    Y = BSplineSolver(**kw).compute_y_matrix()
    assert Y.shape == (1, 1)
    assert abs(1.0 / Y[0, 0] - z_imp) / abs(z_imp) < 1e-9, (1.0 / Y[0, 0], z_imp)


def test_bspline_enrichment_pec_ground_reciprocity():
    """Two ports over a PEC ground: Y stays symmetric (Y[0,1] == Y[1,0]).
    Galerkin reciprocity of the augmented system survives only if the image
    reaction blocks Z_pe and Z_ep are mutual transposes — a direct check that
    the enrichment↔image and image↔enrichment legs agree."""
    kw1 = _hentenna_enrich_kwargs(ground_clearance=0.25)
    w2 = np.array(kw1["wires"][2])
    arc2 = float(np.linalg.norm(np.diff(w2, axis=0), axis=1).sum()) / 2.0
    feeds = [(0, kw1["feed_arclength"], 1.0 + 0.0j), (2, arc2, 0.0 + 0.0j)]
    kw = _hentenna_enrich_kwargs(feeds=feeds, ground_clearance=0.25)
    Y = BSplineSolver(**kw).compute_y_matrix()
    assert Y.shape == (2, 2)
    assert abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 0]) < 1e-6


def test_bspline_enrichment_pec_far_plane_approaches_free_space():
    """Image-limit consistency (the eps→∞ analog for the PEC path): pushing
    the ground plane far below the antenna must drive the enriched grounded
    impedance back to the free-space enriched value as the image reaction
    (∝ 1/R) vanishes. Monotone in clearance, and within 1e-3 at 50 λ."""
    z_free, _ = BSplineSolver(**_hentenna_enrich_kwargs()).compute_impedance()
    z_near, _ = BSplineSolver(
        **_hentenna_enrich_kwargs(ground_clearance=5.0)
    ).compute_impedance()
    z_far, _ = BSplineSolver(
        **_hentenna_enrich_kwargs(ground_clearance=50.0)
    ).compute_impedance()
    assert abs(z_far - z_free) < abs(z_near - z_free)  # monotone toward free
    assert abs(z_far - z_free) / abs(z_free) < 1e-3


def test_bspline_d2_hentenna_enrichment_convergence_over_pec():
    """Acceptance gate (#167): the enrichment convergence-rate result holds
    over a PEC ground plane. Sweeping the mesh with a fixed ground 0.25 λ
    below the antenna, the fitted X-rate p in Z(N) = Z_inf + C/N^p must stay
    above 2.5 — the enrichment still restores high-order convergence at the
    K≥3 junctions; the smooth image reaction reintroduces no cusp."""
    ns = [21, 41, 81]
    Xs = []
    for n in ns:
        z, _ = BSplineSolver(
            **_hentenna_enrich_kwargs(n=n, ground_clearance=0.25)
        ).compute_impedance()
        Xs.append(z.imag)
    dX_12 = Xs[0] - Xs[1]
    dX_23 = Xs[1] - Xs[2]
    assert dX_12 * dX_23 > 0, f"X differences sign-flipped (noise floor); Xs={Xs}"
    p = np.log(abs(dX_12 / dX_23)) / np.log(ns[1] / ns[0])
    assert p > 2.5, f"X convergence rate over PEC p={p:.2f} below 2.5 (Xs={Xs})"


@pytest.mark.parametrize("variant", ["raw", "stable", "tikhonov", "auto"])
def test_bspline_enrichment_refl_coef_ground_y_matrix_identity(variant):
    """The #165 oracle over the fast finite (refl-coef) ground (#167 stage 2):
    1/Y[0,0] == compute_impedance with the enrichment blocks bordered by their
    Fresnel-weighted ground-image reaction. Same tightest self-consistency
    check as the PEC case, now with complex per-segment-pair weights."""
    kw = _hentenna_enrich_kwargs(
        variant=variant, ground_clearance=0.25, ground_eps=(13.0, 0.005)
    )
    z_imp, _ = BSplineSolver(**kw).compute_impedance()
    Y = BSplineSolver(**kw).compute_y_matrix()
    assert Y.shape == (1, 1)
    assert abs(1.0 / Y[0, 0] - z_imp) / abs(z_imp) < 1e-9, (1.0 / Y[0, 0], z_imp)


def test_bspline_enrichment_refl_coef_pec_limit():
    """ε̃ → ∞ collapses the fast finite ground to the PEC image (the Fresnel
    weights → the mirror tangent dot / unit charge weight). The enriched
    refl-coef impedance at ground_eps=1e16 must reproduce the PEC-image
    enriched impedance — the canonical ground-limit consistency check, now
    reachable for enrichment because stage 2 admits ground_eps."""
    z_pec, _ = BSplineSolver(
        **_hentenna_enrich_kwargs(ground_clearance=0.25)
    ).compute_impedance()
    z_lim, _ = BSplineSolver(
        **_hentenna_enrich_kwargs(ground_clearance=0.25, ground_eps=1e16)
    ).compute_impedance()
    assert abs(z_lim - z_pec) / abs(z_pec) < 1e-6


def test_bspline_enrichment_refl_coef_ground_reciprocity():
    """Two ports over the fast finite ground: Y stays symmetric
    (Y[0,1] == Y[1,0]). Galerkin reciprocity survives the complex Fresnel-
    weighted image blocks only if Z_pe and Z_ep remain mutual transposes."""
    kw1 = _hentenna_enrich_kwargs(ground_clearance=0.25, ground_eps=(13.0, 0.005))
    w2 = np.array(kw1["wires"][2])
    arc2 = float(np.linalg.norm(np.diff(w2, axis=0), axis=1).sum()) / 2.0
    feeds = [(0, kw1["feed_arclength"], 1.0 + 0.0j), (2, arc2, 0.0 + 0.0j)]
    kw = _hentenna_enrich_kwargs(
        feeds=feeds, ground_clearance=0.25, ground_eps=(13.0, 0.005)
    )
    Y = BSplineSolver(**kw).compute_y_matrix()
    assert Y.shape == (2, 2)
    assert abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 0]) < 1e-6


@pytest.mark.parametrize("variant", ["raw", "stable", "tikhonov", "auto"])
def test_bspline_enrichment_sommerfeld_ground_y_matrix_identity(variant):
    """The #165 oracle over the Sommerfeld ground (#167 stage 3): 1/Y[0,0] ==
    compute_impedance with the enrichment blocks bordered by their C2 exact-
    image reaction plus the smooth remainder reaction. Both solve paths must
    agree on the augmented Sommerfeld system."""
    kw = _hentenna_enrich_kwargs(
        variant=variant,
        ground_clearance=0.25,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
    )
    z_imp, _ = BSplineSolver(**kw).compute_impedance()
    Y = BSplineSolver(**kw).compute_y_matrix()
    assert Y.shape == (1, 1)
    assert abs(1.0 / Y[0, 0] - z_imp) / abs(z_imp) < 1e-9, (1.0 / Y[0, 0], z_imp)


def test_bspline_enrichment_sommerfeld_pec_limit():
    """ε̃ → ∞ collapses Sommerfeld to the PEC image: the exact-image weight
    C2 = (ε̃−1)/(ε̃+1) → 1 and the smooth remainder field → 0. The enriched
    Sommerfeld impedance at ground_eps=1e13 must reproduce the PEC-image
    enriched impedance — a direct check that BOTH new Sommerfeld pieces (the
    C2 image and the remainder reaction) vanish correctly in the limit."""
    z_pec, _ = BSplineSolver(
        **_hentenna_enrich_kwargs(ground_clearance=0.25)
    ).compute_impedance()
    z_lim, _ = BSplineSolver(
        **_hentenna_enrich_kwargs(
            ground_clearance=0.25, ground_eps=1e13, ground_model="sommerfeld"
        )
    ).compute_impedance()
    assert abs(z_lim - z_pec) / abs(z_pec) < 1e-6


def test_bspline_enrichment_sommerfeld_tracks_refl_coef():
    """Sanity bound on the remainder magnitude: over the same finite ground,
    the exact Sommerfeld and the fast refl-coef approximation must land close
    (both model the same physics, differing only by the remainder correction
    the refl-coef path drops). A few percent apart, not diverging — catches a
    remainder block that is mis-scaled or has the wrong sign."""
    kw = dict(ground_clearance=0.25, ground_eps=(13.0, 0.005))
    z_somm, _ = BSplineSolver(
        **_hentenna_enrich_kwargs(ground_model="sommerfeld", **kw)
    ).compute_impedance()
    z_refl, _ = BSplineSolver(**_hentenna_enrich_kwargs(**kw)).compute_impedance()
    assert abs(z_somm - z_refl) / abs(z_refl) < 0.05


def test_bspline_enrichment_sommerfeld_ground_reciprocity():
    """Two ports over the Sommerfeld ground: Y stays symmetric
    (Y[0,1] == Y[1,0]). Reciprocity survives only if the C2-image and
    remainder blocks each keep Z_pe and Z_ep mutual transposes."""
    kw1 = _hentenna_enrich_kwargs(
        ground_clearance=0.25, ground_eps=(13.0, 0.005), ground_model="sommerfeld"
    )
    w2 = np.array(kw1["wires"][2])
    arc2 = float(np.linalg.norm(np.diff(w2, axis=0), axis=1).sum()) / 2.0
    feeds = [(0, kw1["feed_arclength"], 1.0 + 0.0j), (2, arc2, 0.0 + 0.0j)]
    kw = _hentenna_enrich_kwargs(
        feeds=feeds,
        ground_clearance=0.25,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
    )
    Y = BSplineSolver(**kw).compute_y_matrix()
    assert Y.shape == (2, 2)
    assert abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 0]) < 1e-6


def test_bspline_enrichment_y_matrix_two_port():
    """Two ports (feed wire + a port on the far rail): Y is symmetric
    (Galerkin reciprocity survives the augmented system) and the [0,0]
    entry equals the driven-port current of a (1 V, 0 V) impedance solve."""
    kw1 = _hentenna_enrich_kwargs()
    # port 2 halfway along wire 2's three-edge polyline
    w2 = np.array(kw1["wires"][2])
    arc2 = float(np.linalg.norm(np.diff(w2, axis=0), axis=1).sum()) / 2.0
    feeds = [(0, kw1["feed_arclength"], 1.0 + 0.0j), (2, arc2, 0.0 + 0.0j)]
    kw = _hentenna_enrich_kwargs(feeds=feeds)
    Y = BSplineSolver(**kw).compute_y_matrix()
    assert Y.shape == (2, 2)
    assert abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 0]) < 1e-6
    z_per, _ = BSplineSolver(**kw).compute_impedance()
    # feeds = (1 V, 0 V): driving-point current at port 0 is exactly the
    # first Y column's diagonal entry.
    assert abs(1.0 / z_per[0] - Y[0, 0]) / abs(Y[0, 0]) < 1e-9


def test_bspline_enrichment_y_matrix_swept_matches_single_k():
    """The swept enrichment path is a per-k compute_y_matrix loop; its
    first entry at k_array[0] == self.k must reproduce the single-k Y, and
    self.k/omega/wavelength must be restored afterwards."""
    s = BSplineSolver(**_hentenna_enrich_kwargs())
    Y0 = s.compute_y_matrix()
    k0, om0, wl0 = s.k, s.omega, s.wavelength
    Ys = s.compute_y_matrix_swept(np.array([s.k, 1.02 * s.k]))
    assert Ys.shape == (2, 1, 1)
    assert abs(Ys[0, 0, 0] - Y0[0, 0]) / abs(Y0[0, 0]) < 1e-10
    assert (s.k, s.omega, s.wavelength) == (k0, om0, wl0)
    assert abs(Ys[1, 0, 0] - Y0[0, 0]) / abs(Y0[0, 0]) > 1e-4  # k moved, Y moved


def _reference_basis_polynomials_dense(sim, geom):
    """The pre-#329 `_build_basis_polynomials` body, verbatim.

    Kept here as the bit-exactness oracle for the banded rewrite: it
    densifies the design matrix with `.toarray()` and contracts the full
    (n_total_w, d+1, n_basis_w) block, so it exercises none of the CSR
    banding the solver now uses.
    """
    from scipy.interpolate import BSpline

    from momwire.bspline import _V_UNIT_INV

    d = sim.degree
    n_wings = d + 1
    n_poly = d + 1

    start_status, end_status = sim._wire_endpoint_status()

    all_supp_seg = []
    all_polys = []
    junction_dirs = {j: [] for j in range(len(sim.junctions))}

    m_global = 0
    for w_idx, pw in enumerate(geom["per_wire"]):
        arc = pw["arc_at_knot"]
        wire_arc = arc[-1]
        knots = np.concatenate([np.full(d, 0.0), arc.copy(), np.full(d, wire_arc)])
        n_basis_w = len(knots) - d - 1

        kept = []
        if start_status[w_idx] == "free":
            pass
        elif start_status[w_idx] == "ground":
            kept.append((0, "gnd", None, "start"))
        else:
            kept.append((0, "dir", start_status[w_idx], "start"))
        for j in range(1, n_basis_w - 1):
            kept.append((j, "int", None, None))
        if end_status[w_idx] == "free":
            pass
        elif end_status[w_idx] == "ground":
            kept.append((n_basis_w - 1, "gnd", None, "end"))
        else:
            kept.append((n_basis_w - 1, "dir", end_status[w_idx], "end"))

        seg_off = geom["seg_offsets"][w_idx]
        h_per_seg_w = pw["h_per_seg"]
        arc_at_knot_w = pw["arc_at_knot"]
        n_total_w = pw["n_total"]

        unit = np.linspace(0.0, 1.0, d + 1)
        u_local_per_seg = h_per_seg_w[:, None] * unit[None, :]
        u_global_per_seg = arc_at_knot_w[:-1, None] + u_local_per_seg
        u_flat = u_global_per_seg.reshape(-1)

        DM = BSpline.design_matrix(u_flat, knots, d).toarray()
        DM_seg = DM.reshape(n_total_w, d + 1, n_basis_w)

        V_unit_inv = _V_UNIT_INV[d]
        inv_h_powers = h_per_seg_w[:, None] ** (-np.arange(d + 1))
        poly_per_seg = np.einsum("ij,sjk->sik", V_unit_inv, DM_seg)
        poly_per_seg *= inv_h_powers[:, :, None]

        for _kept_idx, (j, kind, junc_idx, end_pos) in enumerate(kept):
            seg_lo = max(0, j - d)
            seg_hi = min(n_total_w, j + 1)
            n_actual = seg_hi - seg_lo

            supp_seg_m = np.zeros(n_wings, dtype=np.int64)
            polys_m = np.zeros((n_wings, n_poly), dtype=np.float64)
            supp_seg_m[:n_actual] = seg_off + np.arange(seg_lo, seg_hi)
            polys_m[:n_actual, :] = poly_per_seg[seg_lo:seg_hi, :, j]

            all_supp_seg.append(supp_seg_m)
            all_polys.append(polys_m)

            if kind == "dir":
                junction_dirs[junc_idx].append(
                    (m_global, +1.0 if end_pos == "start" else -1.0)
                )
            m_global += 1

    supp_seg = (
        np.stack(all_supp_seg, axis=0)
        if all_supp_seg
        else np.zeros((0, n_wings), dtype=np.int64)
    )
    polys = (
        np.stack(all_polys, axis=0)
        if all_polys
        else np.zeros((0, n_wings, n_poly), dtype=np.float64)
    )

    grounded = sim._grounded_junctions()
    kcl_rows = [j for j in range(len(sim.junctions)) if j not in grounded]
    kcl_A = np.zeros((len(kcl_rows), supp_seg.shape[0]), dtype=np.float64)
    for row, j_idx in enumerate(kcl_rows):
        for m_g, sign in junction_dirs[j_idx]:
            kcl_A[row, m_g] = sign
    return supp_seg, polys, kcl_A


def _basis_case_kwargs(case):
    """Geometry kwargs for the #329 bit-exactness matrix."""
    if case == "single":
        ys = np.linspace(-2.0, 2.0, 5)
        w = np.column_stack([np.zeros_like(ys), ys, np.zeros_like(ys)])
        return {"wires": [w], "n_per_edge_per_wire": [[4] * 4], "nsegs": 16}
    if case == "nonuniform":
        ys = np.linspace(-2.0, 2.0, 5)
        w = np.column_stack([np.zeros_like(ys), ys, np.zeros_like(ys)])
        # Unequal n_per_edge on unequal-length edges: h_per_seg varies
        # segment to segment, so inv_h_powers is not a single constant.
        return {"wires": [w], "n_per_edge_per_wire": [[3, 7, 2, 5]], "nsegs": 17}
    if case == "junction":
        w0 = np.array([[0.0, -2.0, 0.0], [0.0, 0.0, 0.0]])
        w1 = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        w2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]])
        return {
            "wires": [w0, w1, w2],
            "n_per_edge_per_wire": [[6], [5], [4]],
            "nsegs": 15,
            "junctions": [[(0, "end"), (1, "start"), (2, "start")]],
        }
    if case == "ground":
        # Wire 0 stands on the plane (ground end at z = 0); wire 1 joins it
        # above the plane, so the case carries a ground end AND a junction.
        w0 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
        w1 = np.array([[0.0, 0.0, 2.0], [1.3, 0.0, 2.0]])
        return {
            "wires": [w0, w1],
            "n_per_edge_per_wire": [[7], [5]],
            "nsegs": 12,
            "junctions": [[(0, "end"), (1, "start")]],
            "ground_z": 0.0,
        }
    raise AssertionError(case)


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("case", ["single", "nonuniform", "junction", "ground"])
def test_bspline_banded_basis_build_is_bit_exact(degree, case):
    """The banded basis build (#329) must reproduce the densified route
    bit for bit, not merely to tolerance.

    The rewrite reads `BSpline.design_matrix` out of its CSR instead of
    `.toarray()`, and the alignment it depends on is delicate: a sample
    sitting exactly on an interior knot is resolved into the NEXT span,
    so the d+1 rows of one segment do not share a column start. An
    off-by-one there produces polynomials that are plausible — smooth,
    right magnitude, wrong basis — and only an exact comparison catches
    it, hence `array_equal` rather than `allclose`.

    The matrix covers the paths where the alignment can differ: degree 1
    and 2 (band width 2 vs 3), a single wire (both ends dropped), a
    3-wire junction (directional bases kept at one shared knot), a
    grounded end plus junction (#151 kept end basis, dropped KCL row),
    and non-uniform n_per_edge (h_per_seg varying segment to segment, so
    the `inv_h_powers` scaling is not a shared constant).
    """
    import momwire.bspline as bmod

    kwargs = _basis_case_kwargs(case)
    sim = BSplineSolver(degree=degree, wavelength=8.0, **kwargs)
    geom = sim._build_geometry()
    ref_supp, ref_polys, ref_kcl = _reference_basis_polynomials_dense(sim, geom)

    bmod._BASIS_POLY_CACHE.clear()
    sim._cached_basis_polynomials = None
    supp, polys, kcl, _wk, _wbg = sim._build_basis_polynomials(geom)

    assert np.array_equal(supp, ref_supp)
    assert np.array_equal(kcl, ref_kcl)
    assert polys.shape == ref_polys.shape
    assert np.array_equal(polys, ref_polys), (
        "banded basis build diverged from the dense route; max |Δ| = "
        f"{np.max(np.abs(polys - ref_polys))}"
    )


def test_bspline_basis_build_holds_no_n_squared_transient():
    """`_build_basis_polynomials` must not densify the design matrix
    (issue #329). `BSpline.design_matrix(...).toarray()` used to expand a
    CSR with exactly d+1 nonzeros per row into a (N_w·(d+1), N_w+d)
    float64 block, and the einsum that follows materialised a second
    array of the same shape — both fully resident, both scaling as N_w²
    per WIRE, at 24-48 bytes per entry for d = 2.

    Arithmetic for the threshold, at the N = 1200 segments this geometry
    builds on ONE wire (tracemalloc sees numpy's data allocations):

      * the dense DM alone was 8 · N · (d+1) · (N+d) ≈ 8 N² (d+1) =
        34.6 MB at d = 2, and `poly_per_seg` was a second copy of it —
        the pair measured 70.2 MB here. Even the single-copy 8 N² floor
        is 11.5 MB.
      * the banded build allocates O(N · (d+1)²) throughout: the CSR's
        own data + indices, the (N, d+1, d+1) band, its einsum output,
        and the per-basis (d+1, d+1) rows that get stacked into `polys`
        — measured together at 1.29 MB, of which 0.38 MB is the returned
        result itself.

    3 MB therefore sits ~2.3x above what the fixed code needs and ~3.8x
    below the 8 N² single-copy floor the old route could not go under.

    The geometry must be FRESH per measurement: `_BASIS_POLY_CACHE` is a
    module-level FIFO keyed on geometry, so re-using a shape measures a
    cache hit instead of a build.
    """
    import tracemalloc

    import momwire.bspline as bmod

    n_edges, seg_per_edge = 150, 8
    ys = np.linspace(-5.113, 5.113, n_edges + 1)
    wires = [np.column_stack([np.zeros_like(ys), ys, np.zeros_like(ys)])]
    sim = BSplineSolver(
        wires=wires,
        degree=2,
        n_per_edge_per_wire=[[seg_per_edge] * n_edges],
        nsegs=n_edges * seg_per_edge,
        wavelength=22.0,
    )
    geom = sim._build_geometry()
    bmod._BASIS_POLY_CACHE.clear()
    sim._cached_basis_polynomials = None

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        supp, _polys, _kcl, _wk, _wbg = sim._build_basis_polynomials(geom)
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    n = supp.shape[0]
    assert n == n_edges * seg_per_edge  # N = 1200
    assert peak < 3_000_000, (
        f"basis build peaked {peak / 1e6:.2f} MB (8 N**2 = "
        f"{8 * n**2 / 1e6:.2f} MB) — the design matrix is dense again"
    )


# ----------------------------------------------------------------------
# Observer-banded point-matched fill (momwire#332 unit A)
# ----------------------------------------------------------------------

_SIN_BAND_GROUNDS = [
    pytest.param({}, id="free"),
    pytest.param({"ground_z": 0.0}, id="pec"),
    pytest.param({"ground_z": 0.0, "ground_eps": (13.0, 0.005)}, id="refl-coef"),
    pytest.param(
        {
            "ground_z": 0.0,
            "ground_eps": (13.0, 0.005),
            "ground_model": "sommerfeld",
        },
        id="sommerfeld",
    ),
]


def _sin_band_wires(mixed):
    """One elevated half-wave wire, or two side by side when the case wants
    two radii — mixed radii are what force `_radius_runs` to intersect its
    runs with the observer band instead of dispatching whole."""
    lam = 22.0
    ys = np.linspace(-0.962 * lam / 4, 0.962 * lam / 4, 2)

    def wire(x):
        return np.column_stack([np.full_like(ys, x), ys, np.full_like(ys, 4.0)])

    return [wire(0.0), wire(0.7)] if mixed else [wire(0.0)]


@pytest.mark.parametrize(
    "wire_radius",
    [pytest.param(0.0005, id="uniform"), pytest.param([0.0005, 0.002], id="mixed")],
)
@pytest.mark.parametrize("extended_kernel", [False, True])
@pytest.mark.parametrize("ground_kwargs", _SIN_BAND_GROUNDS)
def test_sinusoidal_banded_assembly_is_bit_equal(
    ground_kwargs, extended_kernel, wire_radius
):
    """`SinusoidalSolver._assemble_Z` must give the same matrix whatever the
    observer band is (issue #332). Banding moves residency, not arithmetic:
    each Z row is one Φ row reduced against M_{A,B,C} on the SOURCE axis, so
    nothing about the band can reach an output element's reduction order.
    Equality is exact, and this asserts exact — a tolerance would not be a
    gate here.

    Exact is the only useful gate for the EK cases in particular. EK-vs-
    reduced deltas are themselves O((a/R)²), so a band that mis-sliced an EK
    input would still land inside every physics tolerance in the suite while
    being wrong. Same reasoning for the reflection-coefficient case's (N, N)
    specular tables, which have to take the band on their observer axis
    alongside the observer centres.

    Both geometries sit above `_DENSE_ASSEMBLY_THRESHOLD` on purpose,
    asserted below: under it `_assemble_Z` clamps to a single band, because
    the reduction there is a dense zgemm whose k-blocking BLAS chooses from
    the operand shape — banding it reassociates (measured 8.6e-15 relative
    at N=25). Above it the reduction is CSC, which sums a column's ~3
    nonzeros in storage order regardless of row count.
    """
    import momwire.sinusoidal as smod

    mixed = not np.isscalar(wire_radius)
    kw = dict(
        wires=_sin_band_wires(mixed),
        nsegs=40 if mixed else 81,
        wavelength=22.0,
        wire_radius=wire_radius,
        extended_kernel=extended_kernel,
        **ground_kwargs,
    )

    whole = SinusoidalSolver(**kw)
    whole.swept_mem_mb = 1 << 20  # one band over every row: the pre-#332 fill
    geom = whole._build_geometry()
    N = geom["n_segs"]
    # Guard the guard: below the threshold the band is clamped to N and every
    # case here would be comparing the whole fill against itself.
    assert N >= smod._DENSE_ASSEMBLY_THRESHOLD
    if mixed:
        assert len(whole._radius_runs(geom)) == 2
    Z_whole, _ = whole._assemble_Z(geom, whole.k)

    # 7 rows leaves a ragged final band at both N (81 = 11·7 + 4, 80 = 11·7
    # + 3), which is where an off-by-one in the band bookkeeping would show;
    # 1 row is the degenerate end of the range.
    for rows_per_band in (1, 7):
        sim = SinusoidalSolver(**kw)
        # Ask for a row count through the solver's own budget arithmetic
        # rather than restating it here.
        sim.swept_mem_mb = rows_per_band * sim._fill_row_bytes(N) / (1 << 20)
        Z_banded, _ = sim._assemble_Z(sim._build_geometry(), sim.k)
        assert np.array_equal(Z_whole, Z_banded), (
            f"{rows_per_band}-row bands disagree with the whole fill: max rel "
            f"{np.abs(Z_banded - Z_whole).max() / np.abs(Z_whole).max():.3e}"
        )


@pytest.mark.parametrize(
    "ground_kwargs, swept_mem_mb, budget_mb",
    [
        pytest.param({}, 1, 6, id="free"),
        pytest.param({"ground_z": 0.0}, 1, 6, id="pec"),
        pytest.param(
            {"ground_z": 0.0, "ground_eps": (13.0, 0.005)}, 1, 6, id="refl-coef"
        ),
        pytest.param(
            {
                "ground_z": 0.0,
                "ground_eps": (13.0, 0.005),
                "ground_model": "sommerfeld",
            },
            2,
            18,
            id="sommerfeld",
            # ~14 s: the remainder's fixed per-band working set (#343) is
            # rebuilt 93 times at these band heights.
            marks=pytest.mark.slow,
        ),
    ],
)
def test_sinusoidal_assembly_holds_no_whole_matrix_field_tensor(
    ground_kwargs, swept_mem_mb, budget_mb
):
    """The point-matched fill must not hold any (N, N) field tensor (issue
    #332). It used to hold several: the free-space Φ triple plus the matmul
    temporaries, the image triple on top of that under a ground, and the
    Sommerfeld remainder triple plus the C2-scaled differences on top of
    that. `_assemble_Z` now sweeps observer bands bounded by `swept_mem_mb`.

    Arithmetic for the thresholds, at the N = 1200 segments this geometry
    builds (tracemalloc sees numpy's data allocations; Z is allocated inside
    the fill, so it is subtracted off below):

      * one complex128 (N, N) table is 16 N² = 23.04 MB; what a real
        regression restores is a Φ TRIPLE, 69.1 MB. 23.04 MB is the
        paranoid floor and both thresholds sit under it.
      * measured on the pre-#332 whole-tensor fill, above Z: 110.2 MB free
        space, 176.1 MB PEC and refl-coef, 242.1 MB sommerfeld.
      * measured on the banded fill, above Z: 1.28 MB free space, 1.41 MB
        PEC, 1.41 MB refl-coef (all at swept_mem_mb = 1), 12.39 MB
        sommerfeld (at 2 — see below).

    6 MB therefore sits ~4.3x above the worst free/image mode and ~3.8x
    below the paranoid floor.

    Sommerfeld gets 18 MB instead because the remainder evaluator carries a
    fixed working set for the interpolated grid dyad — issue #343's
    transient, measured flat at 10.07 / 10.15 / 10.31 MB for N = 300 / 600 /
    1200 and flat again across band heights of 1 to 64 rows. A constant is
    not an N² table, so it belongs under the threshold rather than in it,
    but it does eat the margin: 18 MB is 1.45x over the measurement, 1.28x
    under the paranoid floor and 3.8x under the triple.

    That same fixed working set is rebuilt per band, at ~0.074 s each, so
    the sommerfeld case runs at swept_mem_mb = 2 (13-row bands) rather than
    1 (6-row): 20 s instead of 30 s, for 1.1 MB more transient (11.29 →
    12.39 MB). Each case pins its own `swept_mem_mb` because the band is
    itself part of the measured transient — what the gate catches is
    anything that does NOT shrink with the band.

    `_image_refl_prep`'s specular tables are deliberately NOT in scope: five
    float64 (N, N) tables cached per geometry, retired by #332's unit B. The
    warm-up fill below is what moves them — and the Sommerfeld grid, and the
    basis coefficients — outside the traced region, so this gate measures
    the fill's own transient and nothing it merely reads.
    """
    import tracemalloc

    # 150 short edges on one straight wire: N = 1200 with one radius, so one
    # `_radius_runs` run and nothing per-segment large enough to compete with
    # the band that sets the budget. Same shape as the #318/#323 gates.
    n_edges, seg_per_edge = 150, 8
    lam = 22.0
    ys = np.linspace(-0.962 * lam / 4, 0.962 * lam / 4, n_edges + 1)
    wires = [np.column_stack([np.zeros_like(ys), ys, np.full_like(ys, 4.0)])]
    sim = SinusoidalSolver(
        wires=wires,
        n_per_edge_per_wire=[[seg_per_edge] * n_edges],
        nsegs=n_edges * seg_per_edge,
        wavelength=lam,
        **ground_kwargs,
    )
    geom = sim._build_geometry()
    assert geom["n_segs"] == n_edges * seg_per_edge  # N = 1200
    sim.swept_mem_mb = swept_mem_mb

    Z, _ = sim._assemble_Z(geom, sim.k)  # warm the caches the fill only reads
    del Z
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        Z, _ = sim._assemble_Z(geom, sim.k)
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    transient = peak - Z.nbytes
    assert transient < budget_mb * 1_000_000, (
        f"banded fill peaked {transient / 1e6:.2f} MB above Z (16 N**2 = "
        f"{16 * Z.shape[0] ** 2 / 1e6:.2f} MB) — a whole-matrix field tensor "
        "is back"
    )


def test_sinusoidal_refl_coef_specular_tables_are_not_cached():
    """`_image_refl_prep`'s five (N, N) specular tables must not survive a
    point-matched `ground_eps` fill, and must not appear even transiently
    during the FIRST fill (issue #332 unit B).

    This is deliberately a SEPARATE gate from
    `test_sinusoidal_assembly_holds_no_whole_matrix_field_tensor`: that one
    warms `_assemble_Z` once before starting tracemalloc, specifically so
    the warm-up (not the traced call) pays for whatever `_image_refl_prep`
    built and cached — which is exactly what would hide a regression here.
    Tracing the COLD first fill instead catches it directly: measured on
    the pre-unit-B build (`_image_refl_prep` caching a whole-geometry
    table `_field_tensor_image_refl` then row-sliced), this same N = 1200
    cold fill peaked at 117.89 MB above Z — 8 N² = 11.52 MB is one table's
    floor, and 5 × 8 N² = 57.6 MB is what the five cached tables alone cost
    (the rest is the ρ_v/ρ_h complex128 pair fresnel_rho computes from
    cos_th, also (N, N) but not cached — both are retired by producing
    per-band tables directly). Post-unit-B this same measurement is 2.20 MB.

    Budget is set well above the per-band build (a handful of MB, scaling
    with `swept_mem_mb`) and well below the 11.52 MB single-table floor, so
    it fails the instant any one whole-geometry specular table comes back
    — from either transient allocation or renewed residency.

    The direct instance-attribute check below is the second half of the
    gate and does not depend on tracemalloc's accounting at all: a plain
    `SinusoidalSolver` (not the Galerkin subclass, which legitimately still
    populates this cache — see `_image_refl_prep`'s docstring) must leave
    `_cached_image_refl_prep` at its `__init__` default of `None` after a
    refl-coef solve, because nothing in the point-matched path is meant to
    touch it anymore.
    """
    import tracemalloc

    n_edges, seg_per_edge = 150, 8
    lam = 22.0
    ys = np.linspace(-0.962 * lam / 4, 0.962 * lam / 4, n_edges + 1)
    wires = [np.column_stack([np.zeros_like(ys), ys, np.full_like(ys, 4.0)])]
    sim = SinusoidalSolver(
        wires=wires,
        n_per_edge_per_wire=[[seg_per_edge] * n_edges],
        nsegs=n_edges * seg_per_edge,
        wavelength=lam,
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
    )
    geom = sim._build_geometry()
    assert geom["n_segs"] == n_edges * seg_per_edge  # N = 1200
    sim.swept_mem_mb = 1

    assert sim._cached_image_refl_prep is None  # nothing built yet

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        Z, _ = sim._assemble_Z(geom, sim.k)  # the FIRST fill — no warm-up
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    transient = peak - Z.nbytes
    assert transient < 6_000_000, (
        f"cold refl-coef fill peaked {transient / 1e6:.2f} MB above Z "
        f"(one specular table = 8 N**2 = {8 * Z.shape[0] ** 2 / 1e6:.2f} MB) "
        "— a whole-geometry specular table is back"
    )
    assert sim._cached_image_refl_prep is None, (
        "the point-matched fill populated _cached_image_refl_prep — a "
        "whole-geometry specular table is resident again"
    )
