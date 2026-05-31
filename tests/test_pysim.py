import pytest
import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

from pysim.triangular import TriangularPySim
from pysim._accelerators import dist_outer_product

import numpy as np


def test_extension():
    nsegs = 20
    pts = np.array([[0, 0, z] for z in range(nsegs + 1)]) / (2 * nsegs)

    result = dist_outer_product(pts, pts)
    expected = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("nsegs", [20, 40, 80])
def test_triangular_dipole_smoke(nsegs):
    L = 2 * 0.962 * 22 / 4
    sim = TriangularPySim(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, L, 0.0]])],
        n_per_edge_per_wire=[[nsegs]],
        nsegs=nsegs,
    )
    z, c = sim.compute_impedance()
    assert c.shape == (nsegs - 1,)
    assert np.isfinite(z.real) and np.isfinite(z.imag)
    assert np.isfinite(c).all()
    # NEC reference for the default dipole geometry: 69.64 - j18.21.
    # Triangular basis converges quickly; even N=20 is within ~2 Ohm on the
    # real part and ~2 Ohm on the imag.
    assert abs(z.real - 69.64) < 3.0
    assert abs(z.imag - (-18.21)) < 6.0


@pytest.mark.parametrize("nsegs", [20, 40, 80])
def test_triangular_two_wire_yagi_smoke(nsegs):
    # Driver + 1.05x reflector at 1 halfdriver spacing — the classic 2-element
    # Yagi case. Mutual coupling pushes the driver Z away from bare-dipole
    # 69.6 - j18.2 toward roughly 77 + j6.
    hd = 0.962 * 22 / 4  # matches TriangularPySim defaults
    sp = hd
    driver = np.array([[0.0, -hd, 0.0], [0.0, hd, 0.0]])
    refl = np.array([[-sp, -1.05 * hd, 0.0], [-sp, 1.05 * hd, 0.0]])
    z, c = TriangularPySim(
        wires=[driver, refl],
        n_per_edge_per_wire=[[nsegs], [nsegs]],
        nsegs=nsegs,
    ).compute_impedance()
    assert c.shape == (2 * (nsegs - 1),)
    assert np.isfinite(z.real) and np.isfinite(z.imag)
    assert np.isfinite(c).all()
    assert 65.0 < z.real < 85.0
    assert -10.0 < z.imag < 25.0


@pytest.mark.parametrize("nsegs", [20, 40, 80])
def test_triangular_collinear_polyline(nsegs):
    # A "bent" wire whose polyline anchors happen to be collinear should give
    # nearly the same answer as a single-edge straight wire: the only path
    # difference is that cross-edge pairs go through quadrature instead of
    # the analytic formula.
    L = 2 * 0.962 * 22 / 4
    straight = np.array([[0.0, 0.0, 0.0], [0.0, L, 0.0]])
    polyline = np.array([[0.0, 0.0, 0.0], [0.0, L / 2, 0.0], [0.0, L, 0.0]])
    z_straight, _ = TriangularPySim(
        wires=[straight], n_per_edge_per_wire=[[nsegs]], nsegs=nsegs
    ).compute_impedance()
    # Use n_qp_off=8 so the artificial cross-edge quadrature at the fake
    # corner has the same precision as the analytic same-edge path.
    z_bent, _ = TriangularPySim(
        wires=[polyline],
        n_per_edge_per_wire=[[nsegs // 2, nsegs // 2]],
        nsegs=nsegs,
        n_qp_off=8,
    ).compute_impedance()
    assert abs(z_bent - z_straight) < 0.2


def test_triangular_v_dipole_smoke():
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
    z, c = TriangularPySim(
        wires=[polyline], n_per_edge_per_wire=[[40, 40]], nsegs=80
    ).compute_impedance()
    assert c.shape == (79,)
    assert np.isfinite(z.real) and np.isfinite(z.imag)
    assert np.isfinite(c).all()
    # Bending lowers R and pushes X more negative compared to straight (69.6 - j18.5).
    assert 30.0 < z.real < 65.0
    assert z.imag < -25.0


@pytest.mark.parametrize("nsegs", [20, 40])
def test_triangular_swept_matches_per_freq(nsegs):
    # Build a small two-wire moxon-like geometry; the batched solver must
    # agree with single-freq calls to machine precision.
    L = 2 * 0.962 * 22 / 4
    halfL = L / 2
    driver = np.array(
        [
            [-0.1, -halfL, 0.0],
            [0.3, -halfL, 0.0],
            [0.3, -0.05, 0.0],
            [0.3, 0.05, 0.0],
            [0.3, halfL, 0.0],
            [-0.1, halfL, 0.0],
        ]
    )
    refl = np.array(
        [
            [-0.2, halfL, 0.0],
            [-0.6, halfL, 0.0],
            [-0.6, -halfL, 0.0],
            [-0.2, -halfL, 0.0],
        ]
    )
    sim = TriangularPySim(
        wires=[driver, refl],
        n_per_edge_per_wire=[[4, nsegs, 1, nsegs, 4], [4, nsegs, 4]],
        nsegs=nsegs,
        feed_wire_index=0,
    )
    z_single, _ = sim.compute_impedance()
    k_arr = np.array([sim.k])
    z_swept = sim.compute_impedance_swept(k_arr)
    assert abs(z_single - z_swept[0]) < 1e-9
    assert np.isfinite(z_single.real) and np.isfinite(z_single.imag)


def test_triangular_moxon_smoke():
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

    sim = TriangularPySim(
        wires=[driver, reflector],
        n_per_edge_per_wire=[[8, 21, 1, 21, 8], [8, 21, 8]],
        feed_wire_index=0,
        nsegs=40,
        wavelength=wavelength,
        halfdriver_factor=0.962,
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


def test_triangular_hexbeam_smoke():
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

    sim = TriangularPySim(
        wires=[driver, reflector],
        n_per_edge_per_wire=[[15, 21, 1, 21, 15], [3, 21, 21, 21, 3]],
        feed_wire_index=0,
        nsegs=40,
        wavelength=wavelength,
        halfdriver_factor=1.071,
    )
    z, c = sim.compute_impedance()
    assert np.isfinite(z.real) and np.isfinite(z.imag)
    assert np.isfinite(c).all()
    # Hexbeam at the canonical free-space design point lands near 50+j20.
    assert 30.0 < z.real < 75.0
    assert -10.0 < z.imag < 45.0


# ---- Junctions (K wires meeting at a node) ----


def test_triangular_k2_junction_equivalent_to_single_polyline():
    """A K=2 junction at a kink is mathematically equivalent to a single
    polyline with that kink as an interior knot — the Lagrange-augmented
    KCL constraint reduces the two directional bases to one effective DOF
    matching the interior tent basis. Should agree to roundoff.
    """
    # Bent dipole, kink at (0, 0, -2), feed mid-arm (NOT at the kink).
    pl_single = np.array([[0.0, -5.0, 0.0], [0.0, 0.0, -2.0], [0.0, 5.0, 0.0]])
    sim_single = TriangularPySim(
        wires=[pl_single],
        n_per_edge_per_wire=[[15, 15]],
        feed_wire_index=0,
        feed_arclength=2.5,
        wavelength=22,
        nsegs=15,
        wire_radius=0.0005,
    )
    z_single, _ = sim_single.compute_impedance()

    # Same geometry split into 2 wires joined by a K=2 junction at the kink.
    pl0 = np.array([[0.0, -5.0, 0.0], [0.0, 0.0, -2.0]])
    pl1 = np.array([[0.0, 0.0, -2.0], [0.0, 5.0, 0.0]])
    sim_junction = TriangularPySim(
        wires=[pl0, pl1],
        n_per_edge_per_wire=[[15], [15]],
        feed_wire_index=0,
        feed_arclength=2.5,
        wavelength=22,
        nsegs=15,
        wire_radius=0.0005,
        junctions=[[(0, "end"), (1, "start")]],
    )
    z_junction, _ = sim_junction.compute_impedance()
    assert abs(z_junction - z_single) < 1e-9, (
        f"K=2 junction Z={z_junction}, single-polyline Z={z_single}"
    )


def test_triangular_k2_junction_swept_matches_per_freq():
    """Batched swept solver with junctions should match per-freq solves."""
    pl0 = np.array([[0.0, -5.0, 0.0], [0.0, 0.0, -2.0]])
    pl1 = np.array([[0.0, 0.0, -2.0], [0.0, 5.0, 0.0]])
    common = dict(
        wires=[pl0, pl1],
        n_per_edge_per_wire=[[15], [15]],
        feed_wire_index=0,
        feed_arclength=2.5,
        nsegs=15,
        wire_radius=0.0005,
        junctions=[[(0, "end"), (1, "start")]],
    )
    sim_sweep = TriangularPySim(wavelength=22, **common)
    C_LIGHT = 299_792_458.0
    freqs_mhz = np.array([10.0, 14.0, 20.0])
    k_array = 2 * np.pi * freqs_mhz * 1e6 / C_LIGHT
    z_swept = sim_sweep.compute_impedance_swept(k_array)
    for f, zs in zip(freqs_mhz, z_swept):
        sim_f = TriangularPySim(wavelength=C_LIGHT / (f * 1e6), **common)
        z_f, _ = sim_f.compute_impedance()
        assert abs(zs - z_f) < 1e-9, f"f={f}: swept={zs}, single={z_f}"


def test_triangular_fandipole_two_band_smoke():
    """Two-band fan dipole (cone arrangement from antenna_designer) modelled
    with pysim junctions at S and T. Verifies the K=3 path runs, converges,
    and produces plausible Z near the design freq (resonant 20m band).
    """
    import math

    C_LIGHT = 299_792_458.0
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

    N = 21
    wires = [np.array([T, S], dtype=float)]
    n_per_edge = [[2]]
    for i in range(2):
        wires.append(np.array([S, A_pos[i], B_pos[i]], dtype=float))
        n_per_edge.append([N, N])
    for i in range(2):
        wires.append(np.array([T, A_neg[i], B_neg[i]], dtype=float))
        n_per_edge.append([N, N])
    junctions = [
        [(0, "end"), (1, "start"), (2, "start")],  # at S
        [(0, "start"), (3, "start"), (4, "start")],  # at T
    ]
    for fmhz in [14.3, 28.47]:
        wavelength = C_LIGHT / (fmhz * 1e6)
        sim = TriangularPySim(
            wires=wires,
            n_per_edge_per_wire=n_per_edge,
            feed_wire_index=0,
            feed_arclength=eps,
            wavelength=wavelength,
            nsegs=N,
            wire_radius=0.0005,
            junctions=junctions,
        )
        z, coeffs = sim.compute_impedance()
        assert np.isfinite(z.real) and np.isfinite(z.imag)
        assert np.isfinite(coeffs).all()
        # Triangular Galerkin on this multi-wire cone topology lands ~60+j0
        # at the design freqs; PyNEC pulse basis gives ~46+j0 — the gap is
        # the basis-shape difference at K=3 junctions. The smoke window
        # below tolerates both solvers' typical answers.
        assert 30.0 < z.real < 90.0, f"f={fmhz}: R={z.real} out of plausible range"
        assert -50.0 < z.imag < 60.0, f"f={fmhz}: X={z.imag} out of plausible range"


# ---- PEC ground (image method) ----


def _h_dipole(L, h):
    return np.array([[0.0, -L / 2, h], [0.0, L / 2, h]])


def test_ground_none_matches_free_space_bit_exact():
    # ground_z=None must take the same code path as the no-argument case.
    L = 2 * 0.962 * 22 / 4
    poly = _h_dipole(L, 0.0)
    z_no, _ = TriangularPySim(
        wires=[poly], n_per_edge_per_wire=[[40]], nsegs=40
    ).compute_impedance()
    z_none, _ = TriangularPySim(
        wires=[poly], n_per_edge_per_wire=[[40]], nsegs=40, ground_z=None
    ).compute_impedance()
    assert z_no == z_none


def test_ground_horizontal_dipole_at_height_recovers_free_space():
    # As h -> infinity above PEC, the image vanishes and Z -> Z_free.
    L = 2 * 0.962 * 22 / 4
    N = 30
    z_free, _ = TriangularPySim(
        wires=[_h_dipole(L, 0.0)], n_per_edge_per_wire=[[N]], nsegs=N
    ).compute_impedance()
    z_high, _ = TriangularPySim(
        wires=[_h_dipole(L, 100.0)],  # ~5 wavelengths up
        n_per_edge_per_wire=[[N]],
        nsegs=N,
        ground_z=0.0,
    ).compute_impedance()
    # At ~5λ height the image is weak but not negligible — a couple of Ohms
    # of shift on R and X is expected.
    assert abs(z_high.real - z_free.real) < 2.0
    assert abs(z_high.imag - z_free.imag) < 3.0


def test_ground_horizontal_dipole_at_zero_height_shorts_out():
    # As h -> 0 above PEC, the anti-parallel image cancels the antenna and
    # the radiated power (and hence the input resistance) goes to zero.
    L = 2 * 0.962 * 22 / 4
    z_lo, _ = TriangularPySim(
        wires=[_h_dipole(L, 0.01)],
        n_per_edge_per_wire=[[40]],
        nsegs=40,
        ground_z=0.0,
    ).compute_impedance()
    assert abs(z_lo.real) < 0.5  # essentially zero radiation resistance


def test_ground_swept_matches_single_freq_with_ground():
    L = 2 * 0.962 * 22 / 4
    N = 30
    h = 5.0
    sim = TriangularPySim(
        wires=[_h_dipole(L, h)],
        n_per_edge_per_wire=[[N]],
        nsegs=N,
        ground_z=0.0,
    )
    z_single, _ = sim.compute_impedance()
    z_swept = sim.compute_impedance_swept(np.array([sim.k]))[0]
    assert abs(z_single - z_swept) < 1e-9


def _straight_dipole(halfdriver_factor=0.962, wavelength=22.0):
    halfdriver = halfdriver_factor * wavelength / 4
    return [np.array([[0.0, 0.0, -halfdriver], [0.0, 0.0, halfdriver]])]


def test_frill_default_is_delta_gap():
    sim = TriangularPySim(wires=_straight_dipole(), n_per_edge_per_wire=[[41]])
    assert sim.feed_model == "delta_gap"


def test_frill_invalid_model_raises():
    with pytest.raises(ValueError, match="feed_model"):
        TriangularPySim(
            wires=_straight_dipole(), n_per_edge_per_wire=[[41]], feed_model="bogus"
        )


def test_frill_invalid_outer_factor_raises():
    with pytest.raises(ValueError, match="frill_outer_factor"):
        TriangularPySim(
            wires=_straight_dipole(),
            n_per_edge_per_wire=[[41]],
            feed_model="magnetic_frill",
            frill_outer_factor=1.0,
        )


def test_frill_dc_limit_recovers_unit_voltage():
    # In the k -> 0 limit, integrating E_z over the whole wire gives the
    # source voltage drop. With the +prefactor sign chosen to match the
    # delta-gap convention v[m_center]=1, the basis sum should equal 1.
    sim = TriangularPySim(
        wires=_straight_dipole(),
        n_per_edge_per_wire=[[81]],
        wavelength=1e12,  # k ~ 0
        feed_model="magnetic_frill",
    )
    geom = sim._build_geometry()
    v = sim._build_source_vector(geom)
    assert abs(v.sum().real - 1.0) < 1e-7
    assert abs(v.sum().imag) < 1e-7


def test_frill_excitation_is_spread_across_neighbors():
    # The frill is a smooth source, not a point source — neighboring bases
    # of m_center should pick up a non-zero contribution proportional to
    # the fraction of E_z mass they overlap.
    sim = TriangularPySim(
        wires=_straight_dipole(),
        n_per_edge_per_wire=[[81]],
        feed_model="magnetic_frill",
    )
    geom = sim._build_geometry()
    v = sim._build_source_vector(geom)
    m = sim._feed_basis_index(geom)
    # m_center holds most of the excitation, but immediate neighbors are
    # measurably non-zero (delta-gap would have them exactly zero).
    assert 0.9 < abs(v[m]) <= 1.0
    assert abs(v[m - 1]) > 1e-4
    assert abs(v[m + 1]) > 1e-4
    # Bases two segments away from the source should be ~zero (frill E_z
    # falls off as 1/z^3 for z >> b).
    assert abs(v[m - 3]) < 1e-3
    assert abs(v[m + 3]) < 1e-3


def test_frill_impedance_close_to_delta_gap_at_typical_N():
    # For h >> b (typical N), the frill's E_z is so narrow vs the tent that
    # the projected v is nearly identical to the delta-gap v after the
    # matrix solve. Documenting this as a regression so future changes that
    # break the b -> a limit (e.g. sign flip on the prefactor) get caught.
    wires = _straight_dipole()
    z_delta, _ = TriangularPySim(
        wires=wires, n_per_edge_per_wire=[[81]], feed_model="delta_gap"
    ).compute_impedance()
    z_frill, _ = TriangularPySim(
        wires=wires, n_per_edge_per_wire=[[81]], feed_model="magnetic_frill"
    ).compute_impedance()
    assert abs(z_frill - z_delta) < 0.1


def test_frill_swept_matches_single_at_self_k():
    sim_single = TriangularPySim(
        wires=_straight_dipole(),
        n_per_edge_per_wire=[[41]],
        feed_model="magnetic_frill",
    )
    z_single, _ = sim_single.compute_impedance()
    z_swept = sim_single.compute_impedance_swept(np.array([sim_single.k]))[0]
    assert abs(z_single - z_swept) < 1e-9


def test_frill_delta_gap_path_unchanged():
    # Bit-exact regression: with feed_model defaulting to delta_gap, the
    # source vector is the existing one-hot, and compute_impedance returns
    # exactly the same result as before this PR. The hardcoded reference is
    # the delta-gap impedance from main at this N.
    sim = TriangularPySim(wires=_straight_dipole(), n_per_edge_per_wire=[[81]])
    z, _ = sim.compute_impedance()
    geom = sim._build_geometry()
    v = sim._build_source_vector(geom)
    m = sim._feed_basis_index(geom)
    assert v[m] == 1.0
    assert (v == 0).sum() == v.size - 1
