"""Per-wire radius on SinusoidalSolver (stevenmburns/momwire#147).

Covers the two mixed-radius conventions (see the "Per-wire radius" section
of docs/sinusoidal_basis_design.md):

* basis end-condition constants use each segment's own radius (nec2c TBF);
* the field kernel offsets onto the OBSERVER segment's surface
  (necpp EFLD: rh = sqrt(rho² + a_obs²)).

Scalar-regression contract: a uniform per-wire array is bit-identical to
the scalar it equals. Oracle contract: mixed-radius geometries agree with
PyNEC within the tolerances established for single-radius geometries
(uniform 0.5 mm dipole ~0.05 Ω, uniform 5 mm ~0.44 Ω at this length).
"""

import numpy as np
import pytest

from momwire import _wire_loading
from momwire.sinusoidal import SinusoidalSolver

C_LIGHT = 299_792_458.0
WL = 22.0
FREQ_MHZ = C_LIGHT / WL / 1e6


# ----------------------------------------------------------------------
# Geometry builders
# ----------------------------------------------------------------------


def _two_arm_dipole(radii, n):
    """Center-junction dipole, one wire per arm, fed on the top arm's
    first segment (mirrors the PyNEC deck's EX on tag 2 segment 1)."""
    L = 5.291
    return SinusoidalSolver(
        wires=[
            np.array([[0.0, 0.0, -L], [0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, L]]),
        ],
        n_per_edge_per_wire=[[n], [n]],
        junctions=[[(0, "end"), (1, "start")]],
        feed_wire_index=1,
        feed_arclength=0.0,
        wavelength=WL,
        wire_radius=radii,
        nsegs=n,
    )


def _groundplane(radii, n):
    """Vertical + 4 radials meeting at the origin (free space): a K=5
    junction where every member can carry its own radius."""
    Lv = 5.5
    O = (0.0, 0.0, 0.0)
    tips = [(Lv, 0.0, 0.0), (-Lv, 0.0, 0.0), (0.0, Lv, 0.0), (0.0, -Lv, 0.0)]
    wires = [np.array([O, (0.0, 0.0, Lv)])] + [np.array([O, t]) for t in tips]
    return SinusoidalSolver(
        wires=wires,
        n_per_edge_per_wire=[[n]] * 5,
        junctions=[[(w, "start") for w in range(5)]],
        feed_wire_index=0,
        feed_arclength=0.0,
        wavelength=WL,
        wire_radius=radii,
        nsegs=n,
    )


def _pynec_z(wires_nec, feed_tag):
    """Free-space single-frequency drive; wires_nec = (tag, n, p0, p1, rad)."""
    nec = pytest.importorskip("PyNEC")
    c = nec.nec_context()
    geo = c.get_geometry()
    for tag, n, p0, p1, rad in wires_nec:
        geo.wire(tag, n, *p0, *p1, rad, 1.0, 1.0)
    c.geometry_complete(0)
    c.gn_card(-1, 0, 0, 0, 0, 0, 0, 0)
    c.ex_card(0, feed_tag, 1, 0, 1.0, 0.0, 0, 0, 0, 0)
    c.fr_card(0, 1, FREQ_MHZ, 0)
    c.xq_card(0)
    return complex(c.get_input_parameters(0).get_impedance()[0])


# ----------------------------------------------------------------------
# Scalar regression: uniform array == scalar, bit for bit
# ----------------------------------------------------------------------


def test_uniform_array_bit_identical_to_scalar_dipole():
    z_s, alpha_s = _two_arm_dipole(0.0005, 21).compute_impedance()
    z_a, alpha_a = _two_arm_dipole([0.0005, 0.0005], 21).compute_impedance()
    assert z_s == z_a
    np.testing.assert_array_equal(alpha_s, alpha_a)


def test_uniform_array_bit_identical_to_scalar_junctions():
    z_s, alpha_s = _groundplane(0.0005, 9).compute_impedance()
    z_a, alpha_a = _groundplane([0.0005] * 5, 9).compute_impedance()
    assert z_s == z_a
    np.testing.assert_array_equal(alpha_s, alpha_a)


def test_uniform_array_takes_scalar_fast_path():
    sim = _two_arm_dipole([0.0005, 0.0005], 9)
    assert sim._uniform_radius == 0.0005
    assert _two_arm_dipole([0.0005, 0.0004], 9)._uniform_radius is None


def test_wire_radius_validation():
    with pytest.raises(ValueError, match="length-2"):
        _two_arm_dipole([0.0005] * 3, 9)
    with pytest.raises(ValueError, match="positive and finite"):
        _two_arm_dipole([0.0005, 0.0], 9)
    with pytest.raises(ValueError, match="positive and finite"):
        _two_arm_dipole([0.0005, -0.001], 9)
    with pytest.raises(ValueError, match="positive and finite"):
        _two_arm_dipole([0.0005, np.nan], 9)


# ----------------------------------------------------------------------
# Mixed-radius mechanics (no oracle needed)
# ----------------------------------------------------------------------


def test_mixed_radius_cpp_gate_falls_back_to_numpy(monkeypatch):
    """With the C++ field tensor nominally available, a mixed-radius solve
    must produce the identical result with it disabled — the uniform-only
    gate routes mixed radii to the numpy path either way."""
    import momwire.sinusoidal as sin_mod

    sim_on = _two_arm_dipole([0.005, 0.0005], 15)
    z_on, _ = sim_on.compute_impedance()
    monkeypatch.setattr(sin_mod, "_HAVE_FIELD_TENSOR", False)
    monkeypatch.setattr(sin_mod, "_HAVE_FIELD_TENSOR_REFL", False)
    z_off, _ = _two_arm_dipole([0.005, 0.0005], 15).compute_impedance()
    assert z_on == z_off


def test_mixed_radius_swept_matches_per_k():
    """The basis-coefs cache keys on (geom, k, radius-bytes): a swept solve
    over two wavenumbers must match two independent single-k solves."""
    sim = _two_arm_dipole([0.005, 0.0005], 15)
    k0 = sim.k
    ks = np.array([0.95 * k0, 1.05 * k0])
    z_swept = sim.compute_impedance_swept(ks)
    for kk, z_ref in zip(ks, z_swept):
        wl = 2.0 * np.pi / kk
        sim_k = _two_arm_dipole([0.005, 0.0005], 15)
        sim_k.k = float(kk)
        sim_k.omega = sim_k.k * sim_k.c
        sim_k.wavelength = wl
        z_k, _ = sim_k.compute_impedance()
        assert z_k == pytest.approx(z_ref, rel=1e-12)


def test_mixed_radius_loading_uses_each_wires_radius():
    """Skin-effect loading evaluates at each wire's own radius: the fatter
    wire must show the smaller per-unit-length loss resistance, matching
    wire_internal_impedance at that wire's radius exactly."""
    sim = SinusoidalSolver(
        wires=[
            np.array([[0.0, 0.0, -5.291], [0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.291]]),
        ],
        n_per_edge_per_wire=[[15], [15]],
        junctions=[[(0, "end"), (1, "start")]],
        feed_wire_index=1,
        feed_arclength=0.0,
        wavelength=WL,
        wire_radius=[0.005, 0.0005],
        nsegs=15,
        wire_conductivity=5.8e7,
    )
    zw = sim._loading_zw(sim.omega)
    assert zw[0].real < zw[1].real
    for w, rad in enumerate([0.005, 0.0005]):
        expected = _wire_loading.wire_internal_impedance(sim.omega, rad, 5.8e7)
        assert zw[w] == complex(expected)
    z, alpha = sim.compute_impedance()
    p_tot, p_wire = sim.wire_loss_power(alpha)
    assert p_tot > 0.0 and np.all(p_wire > 0.0)


def test_series_impedance_per_wire_radius_array():
    omega = 2.0 * np.pi * 14.0e6
    sigma = np.array([5.8e7, 5.8e7])
    z_arr = _wire_loading.series_impedance_per_wire(
        omega, np.array([0.005, 0.0005]), sigma, None, None
    )
    for w, rad in enumerate([0.005, 0.0005]):
        z_one = _wire_loading.series_impedance_per_wire(
            omega, rad, sigma[w : w + 1], None, None
        )
        assert z_arr[w] == z_one[0]
    with pytest.raises(ValueError, match="length-2"):
        _wire_loading.series_impedance_per_wire(
            omega, np.array([0.005, 0.0005, 0.001]), sigma, None, None
        )


# ----------------------------------------------------------------------
# Oracle: PyNEC parity on mixed-radius geometries
# ----------------------------------------------------------------------


@pytest.mark.parametrize("radii", [(0.005, 0.0005), (0.0005, 0.005)])
def test_mixed_radius_dipole_matches_pynec(radii):
    """Two-radius dipole (each arm its own radius), both orderings.
    Measured deltas ~0.2-0.3 Ω at N=21 and N=41 (stable under refinement);
    the uniform 5 mm baseline is ~0.44 Ω, so 0.5 Ω catches a convention
    regression (the refuted source-radius kernel gave 8-12 Ω at N=41)."""
    n = 41
    L = 5.291
    z_mw, _ = _two_arm_dipole(list(radii), n).compute_impedance()
    z_nec = _pynec_z(
        [
            (1, n, (0.0, 0.0, -L), (0.0, 0.0, 0.0), radii[0]),
            (2, n, (0.0, 0.0, 0.0), (0.0, 0.0, L), radii[1]),
        ],
        feed_tag=2,
    )
    assert abs(z_mw - z_nec) < 0.5, f"momwire={z_mw}, pynec={z_nec}"


def test_mixed_radius_groundplane_junction_matches_pynec():
    """Fat vertical + four thin radials meeting at a K=5 junction: the
    strongest test of the per-segment TBF constants at a mixed-radius
    junction. Measured delta ~0.21 Ω at N=31 (uniform baseline ~0.03 Ω)."""
    n = 31
    Lv = 5.5
    O = (0.0, 0.0, 0.0)
    tips = [(Lv, 0.0, 0.0), (-Lv, 0.0, 0.0), (0.0, Lv, 0.0), (0.0, -Lv, 0.0)]
    z_mw, _ = _groundplane([0.005] + [0.0005] * 4, n).compute_impedance()
    z_nec = _pynec_z(
        [(1, n, O, (0.0, 0.0, Lv), 0.005)]
        + [(2 + i, n, O, t, 0.0005) for i, t in enumerate(tips)],
        feed_tag=1,
    )
    assert abs(z_mw - z_nec) < 0.5, f"momwire={z_mw}, pynec={z_nec}"


# ----------------------------------------------------------------------
# BSplineSolver (Galerkin family)
# ----------------------------------------------------------------------

from momwire import BSplineSolver  # noqa: E402
from momwire.hmatrix import HMatrixSolver  # noqa: E402


def _two_arm_dipole_bsp(radii, n, **kw):
    """Same two-arm center-junction dipole; feed at the TOP ARM's midpoint
    (the bspline delta-gap lives at a knot — feeding at the junction knot
    is mirror-symmetric between the arms and can't tell the orderings
    apart, so mixed-radius tests keep the feed away from the step)."""
    L = 5.291
    return BSplineSolver(
        wires=[
            np.array([[0.0, 0.0, -L], [0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, L]]),
        ],
        n_per_edge_per_wire=[[n], [n]],
        junctions=[[(0, "end"), (1, "start")]],
        feed_wire_index=1,
        feed_arclength=None,
        wavelength=WL,
        wire_radius=radii,
        nsegs=n,
        **kw,
    )


def test_bspline_uniform_array_bit_identical_to_scalar():
    z_s, alpha_s = _two_arm_dipole_bsp(0.0005, 15).compute_impedance()
    z_a, alpha_a = _two_arm_dipole_bsp([0.0005, 0.0005], 15).compute_impedance()
    assert z_s == z_a
    np.testing.assert_array_equal(alpha_s, alpha_a)


def test_bspline_wire_radius_validation():
    with pytest.raises(ValueError, match="length-2"):
        _two_arm_dipole_bsp([0.0005] * 3, 9)
    with pytest.raises(ValueError, match="positive and finite"):
        _two_arm_dipole_bsp([0.0005, 0.0], 9)


def test_bspline_mixed_radius_orderings_differ():
    """With the feed on the top arm, swapping which arm is fat must change
    Z (this is what the per-observer-row radius buys; a radius-blind build
    returns identical values for both orderings)."""
    z_ab, _ = _two_arm_dipole_bsp([0.005, 0.0005], 21).compute_impedance()
    z_ba, _ = _two_arm_dipole_bsp([0.0005, 0.005], 21).compute_impedance()
    assert abs(z_ab - z_ba) > 1.0


def test_bspline_mixed_radius_degree_consistent():
    """The mixed-radius answer is a property of the formulation, not the
    basis degree: d=1 and d=2 agree to well under an ohm at N=41 (measured
    ~0.45 Ω, both converging to the same value; NEC-2 itself is
    NON-convergent at an in-line radius step — see the design-note
    section — so cross-degree consistency is the right oracle here)."""
    z1, _ = _two_arm_dipole_bsp([0.005, 0.0005], 41, degree=1).compute_impedance()
    z2, _ = _two_arm_dipole_bsp([0.005, 0.0005], 41, degree=2).compute_impedance()
    assert abs(z1 - z2) < 1.0, f"d1={z1}, d2={z2}"


def test_bspline_mixed_radius_swept_matches_single_k():
    sim = _two_arm_dipole_bsp([0.005, 0.0005], 15)
    k0 = sim.k
    z_single, _ = sim.compute_impedance()
    z_swept = _two_arm_dipole_bsp([0.005, 0.0005], 15).compute_impedance_swept(
        np.array([k0])
    )
    assert z_single == pytest.approx(z_swept[0], rel=1e-9)


def test_bspline_mixed_radius_groundplane_matches_pynec():
    """Fat vertical + thin radials (K=5 junction, no in-line radius step):
    NEC converges here, so direct parity applies. Measured ~0.51-0.56 Ω
    across N=15/31 (uniform fat-wire baseline ~0.73 Ω)."""
    pytest.importorskip("PyNEC")
    import PyNEC as nec

    n = 31
    Lv = 5.5
    O = (0.0, 0.0, 0.0)
    tips = [(Lv, 0.0, 0.0), (-Lv, 0.0, 0.0), (0.0, Lv, 0.0), (0.0, -Lv, 0.0)]
    wires = [np.array([O, (0.0, 0.0, Lv)])] + [
        np.array([O, t], dtype=float) for t in tips
    ]
    z_mw, _ = BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=[[n]] * 5,
        junctions=[[(w, "start") for w in range(5)]],
        feed_wire_index=0,
        feed_arclength=None,
        wavelength=WL,
        wire_radius=[0.005] + [0.0005] * 4,
        nsegs=n,
    ).compute_impedance()
    c = nec.nec_context()
    geo = c.get_geometry()
    geo.wire(1, n, *O, 0.0, 0.0, Lv, 0.005, 1.0, 1.0)
    for i, t in enumerate(tips):
        geo.wire(2 + i, n, *O, *t, 0.0005, 1.0, 1.0)
    c.geometry_complete(0)
    c.gn_card(-1, 0, 0, 0, 0, 0, 0, 0)
    c.ex_card(0, 1, (n + 1) // 2, 0, 1.0, 0.0, 0, 0, 0, 0)
    c.fr_card(0, 1, FREQ_MHZ, 0)
    c.xq_card(0)
    z_nec = complex(c.get_input_parameters(0).get_impedance()[0])
    assert abs(z_mw - z_nec) < 1.0, f"momwire={z_mw}, pynec={z_nec}"


def test_hmatrix_uniform_array_matches_scalar_dense():
    L = 5.291
    kw = dict(
        wires=[
            np.array([[0.0, 0.0, -L], [0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, L]]),
        ],
        n_per_edge_per_wire=[[15], [15]],
        junctions=[[(0, "end"), (1, "start")]],
        feed_wire_index=1,
        feed_arclength=None,
        wavelength=WL,
        nsegs=15,
    )
    z_h, _ = HMatrixSolver(wire_radius=[0.0005, 0.0005], **kw).compute_impedance()
    z_d, _ = BSplineSolver(wire_radius=0.0005, **kw).compute_impedance()
    assert abs(z_h - z_d) < 1e-6


# ----------------------------------------------------------------------
# HMatrixSolver: block fills under mixed per-wire radii
# ----------------------------------------------------------------------


def _dipole_kw(n, **extra):
    L = 5.291
    kw = dict(
        wires=[
            np.array([[0.0, 0.0, -L], [0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, L]]),
        ],
        n_per_edge_per_wire=[[n], [n]],
        junctions=[[(0, "end"), (1, "start")]],
        feed_wire_index=1,
        feed_arclength=None,
        wavelength=WL,
        nsegs=n,
    )
    kw.update(extra)
    return kw


def test_hmatrix_mixed_radius_matches_dense():
    """Mixed-radius H-matrix parity against the dense Galerkin build, on a
    mesh fine enough (leaf 8, N=41 per arm) that the partition really has
    admissible far blocks — otherwise the test never leaves zblock."""
    kw = _dipole_kw(41)
    sim = HMatrixSolver(wire_radius=[0.005, 0.0005], aca_leaf_size=8, **kw)
    H = sim.build_hmatrix()
    assert len(H.far) > 0, "partition degenerated to dense — test is vacuous"
    z_h, _ = sim.compute_impedance()
    z_d, _ = BSplineSolver(wire_radius=[0.005, 0.0005], **kw).compute_impedance()
    assert abs(z_h - z_d) / abs(z_d) < 1e-3, f"hmatrix={z_h}, dense={z_d}"


def test_hmatrix_mixed_radius_accel_gate(monkeypatch):
    """The grouped-row C++ block evaluators and the numpy zblock fallback
    must agree on a mixed-radius solve (identical when the accelerator is
    absent — both take the numpy path)."""
    kw = _dipole_kw(41)
    radii = [0.005, 0.0005]
    z_on, _ = HMatrixSolver(
        wire_radius=radii, aca_leaf_size=8, hmatrix_use_accel=True, **kw
    ).compute_impedance()
    z_off, _ = HMatrixSolver(
        wire_radius=radii, aca_leaf_size=8, hmatrix_use_accel=False, **kw
    ).compute_impedance()
    assert abs(z_on - z_off) / abs(z_off) < 1e-3, f"accel={z_on}, numpy={z_off}"


def test_hmatrix_mixed_radius_pec_ground_matches_dense():
    """PEC ground exercises the per-observer-row radius on the image block
    fills (`_zblock_image` and the mirrored C++ evaluators)."""
    kw = _dipole_kw(41, ground_z=-8.0)
    radii = [0.005, 0.0005]
    z_h, _ = HMatrixSolver(wire_radius=radii, aca_leaf_size=8, **kw).compute_impedance()
    z_d, _ = BSplineSolver(wire_radius=radii, **kw).compute_impedance()
    assert abs(z_h - z_d) / abs(z_d) < 1e-3, f"hmatrix={z_h}, dense={z_d}"


# ----------------------------------------------------------------------
# ArrayBlockSolver: shape classes, caches, and symmetry under mixed radii
# ----------------------------------------------------------------------

from momwire.array_block import (  # noqa: E402
    ArrayBlockSolver,
    element_groups,
    reset_array_caches,
)


def _dipole_line_kw(radii_per_elem, nsegs=16):
    """Four well-separated parallel dipoles, one wire each — element e gets
    `radii_per_elem[e]`."""
    half = 0.962 * WL / 4
    ys = [-9.0, -3.0, 3.0, 9.0][: len(radii_per_elem)]
    wires = [np.array([[0.0, y, -half], [0.0, y, half]]) for y in ys]
    return dict(
        wires=wires,
        degree=2,
        n_per_edge_per_wire=[[nsegs]] * len(wires),
        wavelength=WL,
        feeds=[(i, None, 1.0 + 0.0j) for i in range(len(wires))],
        wire_radius=list(radii_per_elem),
    )


def test_array_block_radius_refines_shape_classes():
    """Geometric translates with different radii must not share a shape
    class (their self-blocks differ), while same-radius translates still
    dedup — [fat, thin, fat, thin] is exactly two classes."""
    uni = ArrayBlockSolver(**_dipole_line_kw([0.0005] * 4))
    assert len(set(element_groups(uni).shape_of_elem.tolist())) == 1
    mixed = ArrayBlockSolver(**_dipole_line_kw([0.005, 0.0005, 0.005, 0.0005]))
    part = element_groups(mixed)
    assert len(set(part.shape_of_elem.tolist())) == 2
    assert part.shape_of_elem[0] == part.shape_of_elem[2]
    assert part.shape_of_elem[1] == part.shape_of_elem[3]


def test_array_block_mixed_radius_elements_match_dense():
    """[fat, thin, fat, thin] line: two shape classes with two members each,
    so the block path (not the degenerate H-matrix fallback) runs, with
    self-block and coupling reuse live. Y-matrix parity vs dense."""
    reset_array_caches()
    kw = _dipole_line_kw([0.005, 0.0005, 0.005, 0.0005])
    arr = ArrayBlockSolver(**kw)
    assert not arr._degenerate_partition()
    ya = arr.compute_y_matrix()
    # same-radius pairs still dedup: fewer ACA runs than the 12 ordered pairs
    assert arr._last_n_coupling_aca < 12
    yd = BSplineSolver(**kw).compute_y_matrix()
    assert np.abs(ya - yd).max() / np.abs(yd).max() < 1e-4


def test_array_block_self_block_cache_never_aliases_radii():
    """Two decks with identical geometry but different radii solved
    back-to-back share the module-scope self-block cache — the radius
    pattern is part of the content address, so the second solve must not
    inherit the first's blocks."""
    reset_array_caches()
    y_uni = ArrayBlockSolver(**_dipole_line_kw([0.0005] * 4)).compute_y_matrix()
    kw = _dipole_line_kw([0.005, 0.0005, 0.005, 0.0005])
    y_mix = ArrayBlockSolver(**kw).compute_y_matrix()
    assert np.abs(y_mix - y_uni).max() / np.abs(y_uni).max() > 1e-3
    yd = BSplineSolver(**kw).compute_y_matrix()
    assert np.abs(y_mix - yd).max() / np.abs(yd).max() < 1e-4


def _bent_pair_kw(radii_per_wire, nsegs=12, dy=20.0):
    """Two translated copies of an L element (vertical + horizontal wire,
    internal junction). `radii_per_wire` is per WIRE: 4 entries, wires
    (0, 1) = element 0, wires (2, 3) = element 1."""
    h = 0.962 * WL / 4
    wires, junctions, feeds = [], [], []
    for e in range(2):
        y = e * dy
        base = len(wires)
        wires += [
            np.array([[0.0, y, 0.0], [0.0, y, h]]),
            np.array([[0.0, y, h], [0.0, y + h, h]]),
        ]
        junctions.append([(base, "end"), (base + 1, "start")])
        feeds.append((base, None, 1.0 + 0.0j))
    return dict(
        wires=wires,
        degree=2,
        n_per_edge_per_wire=[[nsegs]] * 4,
        wavelength=WL,
        junctions=junctions,
        feeds=feeds,
        wire_radius=list(radii_per_wire),
    )


def test_array_block_internally_mixed_elements_gate_transpose_reuse():
    """Two identical elements whose wires mix radii INSIDE the element: the
    observer-row regularisation makes Z_ba != Z_ab^T, so the transposed-
    factor shortcut must not fire (2 ACA runs, not 1) and the answer must
    still match dense. The uniform control keeps the shortcut (1 run)."""
    reset_array_caches()
    uni = ArrayBlockSolver(**_bent_pair_kw([0.0005] * 4))
    uni.compute_y_matrix()
    assert uni._last_n_coupling_aca == 1  # transpose reuse intact

    reset_array_caches()
    kw = _bent_pair_kw([0.005, 0.0005, 0.005, 0.0005])
    arr = ArrayBlockSolver(**kw)
    assert not arr._degenerate_partition()
    ya = arr.compute_y_matrix()
    assert arr._last_n_coupling_aca == 2  # gate blocked the transposed reuse
    yd = BSplineSolver(**kw).compute_y_matrix()
    assert np.abs(ya - yd).max() / np.abs(yd).max() < 1e-4
