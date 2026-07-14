"""Interior-anchor (mid-polyline) junctions — issue #138.

The load-bearing fact these tests pin: a polyline with an interior-anchor
attachment spans EXACTLY the same function space as the mesh shattered into
per-junction wires ({C^{d-1} through-spline} ⊕ {d node bases} = the split
spline pair, see `_build_basis_polynomials`), so every observable must match
the shattered form to roundoff — not to discretization tolerance.

A T-junction is the canonical case: through-wire P with a stub C attached at
P's middle anchor, vs P split into two wires A + B with a 3-endpoint junction.
"""

import numpy as np
import pytest

from momwire.bspline import BSplineSolver
from momwire.hmatrix import HMatrixSolver
from momwire.array_block import ArrayBlockSolver
from momwire.sinusoidal import SinusoidalSolver

WL = 22.0
L = 0.962 * WL / 4


def _t_junction_interior(nseg=12, degree=2, **kw):
    """Through-polyline P (anchor mid at the origin) + stub C attached there."""
    P = np.array([[-L, 0.0, 0.0], [0.0, 0.0, 0.0], [L, 0.0, 0.0]])
    C = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, L]])
    return dict(
        wires=[P, C],
        degree=degree,
        n_per_edge_per_wire=[[nseg, nseg], [nseg]],
        wavelength=WL,
        feeds=[(0, 0.5 * L, 1.0 + 0.0j)],
        junctions=[[(0, 1), (1, "start")]],
        **kw,
    )


def _t_junction_shattered(nseg=12, degree=2, **kw):
    """The same mesh with P pre-split at the node (the status quo form)."""
    A = np.array([[-L, 0.0, 0.0], [0.0, 0.0, 0.0]])
    B = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]])
    C = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, L]])
    return dict(
        wires=[A, B, C],
        degree=degree,
        n_per_edge_per_wire=[[nseg], [nseg], [nseg]],
        wavelength=WL,
        feeds=[(0, 0.5 * L, 1.0 + 0.0j)],
        junctions=[[(0, "end"), (1, "start"), (2, "start")]],
        **kw,
    )


def _z(sim):
    return np.atleast_1d(sim.compute_impedance()[0])[0]


@pytest.mark.parametrize("degree", [1, 2])
def test_t_junction_matches_shattered(degree):
    zi = _z(BSplineSolver(**_t_junction_interior(degree=degree)))
    zs = _z(BSplineSolver(**_t_junction_shattered(degree=degree)))
    assert abs(zi - zs) / abs(zs) < 1e-10


@pytest.mark.parametrize("degree", [1, 2])
def test_t_junction_basis_count_matches_shattered(degree):
    """No basis-count savings — and none are possible: the junction's d
    jump dofs are physics, not bookkeeping (a lone value-jump basis is
    inconsistent — the solve converges to a disconnected stub). What the
    interior form buys is polyline bookkeeping: no shattering pass, and
    ~an order of magnitude fewer wires on junction-heavy meshes."""
    si = BSplineSolver(**_t_junction_interior(degree=degree))
    ss = BSplineSolver(**_t_junction_shattered(degree=degree))
    ni = si._build_basis_polynomials(si._build_geometry())[0].shape[0]
    ns = ss._build_basis_polynomials(ss._build_geometry())[0].shape[0]
    assert ni == ns


def test_crossing_matches_shattered():
    """Two through-polylines crossing mid-wire (a wire-grid node): the KCL
    row ties the two node value-jumps with no endpoint basis at all."""
    P = np.array([[-L, 0.0, 0.0], [0.0, 0.0, 0.0], [L, 0.0, 0.0]])
    Q = np.array([[0.0, -L, 0.0], [0.0, 0.0, 0.0], [0.0, L, 0.0]])
    nseg = 10
    zi = _z(
        BSplineSolver(
            wires=[P, Q],
            degree=2,
            n_per_edge_per_wire=[[nseg, nseg], [nseg, nseg]],
            wavelength=WL,
            feeds=[(0, 0.5 * L, 1.0 + 0.0j)],
            junctions=[[(0, 1), (1, 1)]],
        )
    )
    halves = {
        "A": np.array([[-L, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        "B": np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]]),
        "D": np.array([[0.0, -L, 0.0], [0.0, 0.0, 0.0]]),
        "E": np.array([[0.0, 0.0, 0.0], [0.0, L, 0.0]]),
    }
    zs = _z(
        BSplineSolver(
            wires=list(halves.values()),
            degree=2,
            n_per_edge_per_wire=[[nseg]] * 4,
            wavelength=WL,
            feeds=[(0, 0.5 * L, 1.0 + 0.0j)],
            junctions=[[(0, "end"), (1, "start"), (2, "end"), (3, "start")]],
        )
    )
    assert abs(zi - zs) / abs(zs) < 1e-10


def test_two_anchors_on_one_wire_match_shattered():
    """Two stubs on one through-polyline — multiple node-basis sets coexist."""
    P = np.array([[-1.5 * L, 0, 0], [-0.5 * L, 0, 0], [0.5 * L, 0, 0], [1.5 * L, 0, 0]])
    C1 = np.array([[-0.5 * L, 0, 0], [-0.5 * L, 0, L]])
    C2 = np.array([[0.5 * L, 0, 0], [0.5 * L, 0, L]])
    nseg = 8
    zi = _z(
        BSplineSolver(
            wires=[P, C1, C2],
            degree=2,
            n_per_edge_per_wire=[[nseg] * 3, [nseg], [nseg]],
            wavelength=WL,
            feeds=[(0, 0.5 * L, 1.0 + 0.0j)],
            junctions=[[(0, 1), (1, "start")], [(0, 2), (2, "start")]],
        )
    )
    A = np.array([[-1.5 * L, 0, 0], [-0.5 * L, 0, 0]])
    B = np.array([[-0.5 * L, 0, 0], [0.5 * L, 0, 0]])
    D = np.array([[0.5 * L, 0, 0], [1.5 * L, 0, 0]])
    zs = _z(
        BSplineSolver(
            wires=[A, B, D, C1, C2],
            degree=2,
            n_per_edge_per_wire=[[nseg]] * 5,
            wavelength=WL,
            feeds=[(0, 0.5 * L, 1.0 + 0.0j)],
            junctions=[
                [(0, "end"), (1, "start"), (3, "start")],
                [(1, "end"), (2, "start"), (4, "start")],
            ],
        )
    )
    assert abs(zi - zs) / abs(zs) < 1e-10


def test_t_junction_ground_matches_shattered():
    """PEC ground: image blocks consume supp/polys generically, so the
    roundoff equivalence must survive."""
    zi = _z(BSplineSolver(**_t_junction_interior(ground_z=-0.5 * L)))
    zs = _z(BSplineSolver(**_t_junction_shattered(ground_z=-0.5 * L)))
    assert abs(zi - zs) / abs(zs) < 1e-10


def test_t_junction_smoothed_source_matches_shattered():
    zi = _z(BSplineSolver(**_t_junction_interior(feed_smoothing_factor=1.0)))
    zs = _z(BSplineSolver(**_t_junction_shattered(feed_smoothing_factor=1.0)))
    assert abs(zi - zs) / abs(zs) < 1e-10


def test_t_junction_lossy_matches_shattered():
    """Distributed wire loading builds its Gram from supp/polys — the node
    bases' overlaps ride along (the jump current flows in the wire metal)."""
    zi = _z(BSplineSolver(**_t_junction_interior(wire_conductivity=5.8e7)))
    zs = _z(BSplineSolver(**_t_junction_shattered(wire_conductivity=5.8e7)))
    assert abs(zi - zs) / abs(zs) < 1e-10


def test_t_junction_swept_matches_shattered():
    k0 = 2 * np.pi / WL
    k_arr = np.linspace(0.9 * k0, 1.1 * k0, 3)
    zi = BSplineSolver(**_t_junction_interior()).compute_impedance_swept(k_arr)
    zs = BSplineSolver(**_t_junction_shattered()).compute_impedance_swept(k_arr)
    assert np.max(np.abs(zi - zs) / np.abs(zs)) < 1e-10


def test_kcl_current_balance_at_node():
    """I_P(s*+) − I_P(s*−) must equal −I_C(0): the through-current sheds
    exactly the branch current. `currents_at_knots` reports the right limit
    AT the anchor; the left limit is sampled just before it."""
    sim = BSplineSolver(**_t_junction_interior())
    # Solve directly for the coefficient vector via the assembly pieces.
    geom = sim._build_geometry()
    supp, polys, kcl_A, wk, wbg = sim._build_basis_polynomials(geom)
    J = sim._build_J_blocks(geom, sim.k)
    Z = sim._assemble_Z(J, supp, polys, geom)
    v = sim._build_source_vector(geom, wk, wbg, supp.shape[0])
    coeffs = sim._solve_with_kcl(Z, v, kcl_A)[: supp.shape[0]]
    s_star = L  # arc position of P's middle anchor
    eps = 1e-9 * L
    IP = sim.currents_at_knots(
        coeffs, s_array=[np.array([s_star - eps, s_star]), np.array([0.0])]
    )
    jump = IP[0][1] - IP[0][0]
    I_C0 = IP[1][0]
    assert abs(jump + I_C0) < 1e-8 * max(1.0, abs(I_C0))
    assert abs(I_C0) > 1e-6  # the stub actually carries current


def test_accelerated_solvers_handle_interior_anchor():
    """H-matrix and array-block paths consume supp/polys/kcl_A generically;
    a modest tolerance vs dense covers their ACA/GMRES stages."""
    zd = _z(BSplineSolver(**_t_junction_interior(nseg=16)))
    zh = _z(HMatrixSolver(**_t_junction_interior(nseg=16)))
    za = _z(ArrayBlockSolver(**_t_junction_interior(nseg=16)))
    assert abs(zh - zd) / abs(zd) < 1e-3
    assert abs(za - zd) / abs(zd) < 1e-3


def test_anchor_endpoint_coercion():
    """Anchor 0 / M−1 are the endpoints — normalized to 'start'/'end'."""
    kw = _t_junction_interior()
    kw["junctions"] = [[(0, 0), (1, 1)]]  # wire 0 anchor 0, wire 1 anchor 1(=end)
    sim = BSplineSolver(**kw)
    assert sim.junctions == [[(0, "start"), (1, "end")]]


def test_anchor_validation():
    kw = _t_junction_interior()
    kw["junctions"] = [[(0, 7), (1, "start")]]  # wire 0 has 3 anchors
    with pytest.raises(ValueError, match="anchor 7 out of range"):
        BSplineSolver(**kw)


def test_sinusoidal_rejects_interior_anchor():
    kw = _t_junction_interior()
    kw.pop("degree")
    with pytest.raises(ValueError, match="interior-anchor"):
        SinusoidalSolver(**kw)


def test_enrichment_rejects_interior_anchor():
    kw = _t_junction_interior()
    kw["use_singular_enrichment"] = True
    with pytest.raises(NotImplementedError, match="interior-anchor"):
        BSplineSolver(**kw)


def test_array_of_t_elements_block_jacobi_path():
    """Two identical, well-separated T-elements: ArrayBlockSolver takes the
    real block path (repetition exists), so `_BlockJacobiAugPrecond` must
    claim each junction's KCL row — node bases included — inside one element."""

    def t_pair(cls):
        wires, junctions, feeds, npe = [], [], [], []
        for e, y in enumerate((-8.0, 8.0)):
            P = np.array([[-L, y, 0.0], [0.0, y, 0.0], [L, y, 0.0]])
            C = np.array([[0.0, y, 0.0], [0.0, y, L]])
            base = len(wires)
            wires += [P, C]
            npe += [[10, 10], [10]]
            junctions.append([(base, 1), (base + 1, "start")])
            feeds.append((base, 0.5 * L, 1.0 + 0.0j))
        return cls(
            wires=wires,
            degree=2,
            n_per_edge_per_wire=npe,
            wavelength=WL,
            feeds=feeds,
            junctions=junctions,
        )

    ya = t_pair(ArrayBlockSolver).compute_y_matrix()
    yd = t_pair(BSplineSolver).compute_y_matrix()
    assert np.abs(ya - yd).max() / np.abs(yd).max() < 1e-3
