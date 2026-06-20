"""Tests for the hierarchical (H-matrix / ACA) B-spline MoM accelerator.

Phase 0: the on-demand block evaluator `HMatrixPySim.zblock(I, J)` must
reproduce, to machine precision, the corresponding sub-block of the exact
dense `BSplinePySim` impedance matrix — both the off-edge (far) path and the
same-edge analytic-overwrite (near) path.
"""

import numpy as np
import pytest

from pysim.hmatrix import HMatrixPySim


def _dense_Z(sim):
    """The exact dense bspline Z the evaluator must match."""
    geom = sim._build_geometry()
    supp_seg, polys, _kcl_A, _wk, _wbg = sim._build_basis_polynomials(geom)
    J = sim._build_J_blocks(geom, sim.k)
    return sim._assemble_Z(J, supp_seg, polys, geom)


def _dipole(degree, nsegs):
    half = 0.962 * 22 / 4
    wire = np.array([[0.0, 0.0, -half], [0.0, 0.0, half]])
    return HMatrixPySim(
        wires=[wire],
        degree=degree,
        n_per_edge_per_wire=[[nsegs]],
        nsegs=nsegs,
        wavelength=22.0,
    )


def _bent_wire_with_junction(degree, nsegs):
    """Two wires meeting at a right-angle junction — exercises multiple edges
    and a junction directional basis (the same-edge overwrite + KCL path)."""
    h = 0.962 * 22 / 4
    w0 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, h]])
    w1 = np.array([[0.0, 0.0, h], [0.0, h, h]])
    junctions = [[(0, "end"), (1, "start")]]
    return HMatrixPySim(
        wires=[w0, w1],
        degree=degree,
        n_per_edge_per_wire=[[nsegs], [nsegs]],
        nsegs=nsegs,
        wavelength=22.0,
        junctions=junctions,
        feed_wire_index=0,
    )


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize(
    "builder", [_dipole, _bent_wire_with_junction], ids=["dipole", "junction"]
)
def test_zblock_matches_dense_full(builder, degree):
    sim = builder(degree, 16)
    n = sim._context()["n_basis"]
    Z = _dense_Z(sim)
    full = sim.zblock(np.arange(n), np.arange(n))
    rel = np.abs(full - Z).max() / np.abs(Z).max()
    assert rel < 1e-12, f"full-matrix rel err {rel:.2e}"


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize(
    "builder", [_dipole, _bent_wire_with_junction], ids=["dipole", "junction"]
)
def test_zblock_matches_dense_random_subblocks(builder, degree):
    sim = builder(degree, 18)
    n = sim._context()["n_basis"]
    Z = _dense_Z(sim)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(12):
        szI = int(rng.integers(1, n // 2 + 1))
        szJ = int(rng.integers(1, n // 2 + 1))
        I = rng.choice(n, size=szI, replace=False)
        J = rng.choice(n, size=szJ, replace=False)
        ref = Z[np.ix_(I, J)]
        blk = sim.zblock(I, J)
        worst = max(worst, np.abs(blk - ref).max() / (np.abs(ref).max() + 1e-30))
    assert worst < 1e-12, f"worst sub-block rel err {worst:.2e}"


def _long_wire(degree, nsegs):
    half = 2.0 * 0.962 * 22 / 4
    wire = np.array([[0.0, 0.0, -half], [0.0, 0.0, half]])
    return HMatrixPySim(
        wires=[wire],
        degree=degree,
        n_per_edge_per_wire=[[nsegs]],
        nsegs=nsegs,
        wavelength=22.0,
    )


@pytest.mark.parametrize("eta", [1.0, 2.0])
@pytest.mark.parametrize("leaf_size", [16, 32])
def test_block_partition_is_exact_cover(eta, leaf_size):
    """The far + near leaf blocks must tile the n x n index product with no
    gaps and no overlaps."""
    sim = _long_wire(1, 200)
    part = sim.build_partition(eta=eta, leaf_size=leaf_size)
    n = sim._context()["n_basis"]
    cover = np.zeros((n, n), dtype=np.int32)
    for s, t in part["far"] + part["near"]:
        cover[np.ix_(s.indices, t.indices)] += 1
    assert cover.min() == 1 and cover.max() == 1
    st = part["stats"]
    assert st["covered"] == st["total"] == n * n


def test_block_partition_compresses_far_field():
    """A multi-wavelength wire should put most of the matrix area into
    admissible (far) blocks."""
    sim = _long_wire(1, 300)
    part = sim.build_partition(eta=1.0, leaf_size=32)
    assert part["stats"]["far_frac"] > 0.6


def test_far_blocks_have_no_same_edge_pairs():
    """Admissibility must spatially separate clusters, so no far block may
    contain a self-pair (a basis paired with itself)."""
    sim = _long_wire(1, 300)
    part = sim.build_partition(eta=1.0, leaf_size=32)
    for s, t in part["far"]:
        assert np.intersect1d(s.indices, t.indices).size == 0


@pytest.mark.parametrize("tol", [1e-3, 1e-5])
def test_hmatvec_matches_dense(tol):
    """H-matrix matvec must reproduce the dense Z @ x; ACA's block-local
    tolerance bounds the error well below 1."""
    sim = _long_wire(1, 250)
    Z = _dense_Z(sim)
    n = Z.shape[0]
    rng = np.random.default_rng(1)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    H = sim.build_hmatrix(tol=tol)
    rel = np.linalg.norm(H.matvec(x) - Z @ x) / np.linalg.norm(Z @ x)
    assert rel < 50 * tol


def test_hmatrix_to_dense_reconstruction():
    sim = _long_wire(1, 200)
    Z = _dense_Z(sim)
    H = sim.build_hmatrix(tol=1e-6)
    rel = np.abs(H.to_dense() - Z).max() / np.abs(Z).max()
    assert rel < 1e-3


def test_hmatrix_compresses_storage():
    """A multi-wavelength wire must store strictly fewer scalars than dense,
    and the far-field ranks must stay small."""
    sim = _long_wire(1, 400)
    H = sim.build_hmatrix(tol=1e-4)
    st = H.stats()
    assert st["compression"] < 0.7
    assert st["max_rank"] <= 32


def test_hmatrix_handles_junction_geometry():
    sim = _bent_wire_with_junction(1, 60)
    Z = _dense_Z(sim)
    n = Z.shape[0]
    rng = np.random.default_rng(2)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    H = sim.build_hmatrix(tol=1e-5)
    rel = np.linalg.norm(H.matvec(x) - Z @ x) / np.linalg.norm(Z @ x)
    assert rel < 1e-3


def test_zblock_off_edge_skips_same_edge_path():
    """A block between two well-separated single basis functions must contain
    no same-edge pairs, so it is computed purely off-edge — and still matches
    the dense reference."""
    sim = _dipole(2, 24)
    Z = _dense_Z(sim)
    # endpoints of the basis index range are far apart along the wire
    I = np.array([1])
    J = np.array([sim._context()["n_basis"] - 2])
    blk = sim.zblock(I, J)
    ref = Z[np.ix_(I, J)]
    assert np.abs(blk - ref).max() / (np.abs(ref).max() + 1e-30) < 1e-12
