"""`element_currents()` — the public field-readout hook (#251).

The fine-knot resample plus the continuity differencing used to live in each
consumer, written against `currents_at_knots` and the solver's geometry
attributes. Moving it in means momwire, not the consumer, now owns the
contract, so what these gates pin is the contract rather than the arithmetic:

1. The SEMANTIC ORACLE. `oracle_current_elements` below is antennaknobs'
   `NECPortal.current_elements` transcribed line for line, with only the
   attribute names changed (`engine._polylines` → `wires_polylines`,
   `engine._edge_segments` → `n_per_edge_per_wire`). `element_currents` must
   equal it BIT FOR BIT — the point of the move is that the consumer can delete
   its copy and see no change, not that a numerically equivalent rewrite
   arrived.
2. The four arrays' shapes and dtypes, on every family.
3. Telescoping: a wire's own delta slice sums to zero. To round-off, not
   exactly — `delta[j] = e[j-1] - e[j]` rounds each difference before the sum
   cancels it, so the residual is ulp-scale against `max|delta|` (measured at
   ~1e-16 relative here) and a bit-equality assert would be a lie.
4. `subdiv=k` refines rather than replaces: its knots contain the `subdiv=1`
   knots at stride k. To tolerance — `linspace(p0, p1, n*k+1)[j*k]` and
   `linspace(p0, p1, n+1)[j]` are the same point by different arithmetic.
5. A `compute_port_solution().coeffs` COLUMN is a valid input (#232), on every
   family — that pairing is the reason both APIs exist.
"""

import numpy as np
import pytest

from momwire import (
    ArrayBlockSolver,
    BSplineSolver,
    HMatrixSolver,
    SinusoidalGalerkinSolver,
    SinusoidalSolver,
)

WL = 10.0
L = 0.48 * WL

SPLINE = [BSplineSolver, HMatrixSolver, ArrayBlockSolver]
SEGMENT = [SinusoidalSolver, SinusoidalGalerkinSolver]
ALL_CLASSES = SPLINE + SEGMENT

# The accelerated operators approximate the SOLVE, not the readout; tighten
# them so the coeffs the readout is handed are the dense ones to solve_tol.
TIGHT = dict(aca_tol=1e-10, solve_tol=1e-11)


def _kw(cls, kw):
    return {**kw, **TIGHT} if cls in (HMatrixSolver, ArrayBlockSolver) else kw


def bent_tee(**extra):
    """Two wires meeting at the origin, one of them bent — so the walk has to
    cross an EDGE boundary inside a wire as well as a WIRE boundary at a
    junction, which are the two places the knot bookkeeping can go wrong."""
    wires = [
        np.array([(0.0, 0.0, 0.0), (0.0, 0.0, L / 2), (L / 4, 0.0, L / 2)]),
        np.array([(0.0, 0.0, 0.0), (0.0, 0.0, -L / 2)]),
    ]
    return dict(
        wires=wires,
        n_per_edge_per_wire=[[4, 3], [5]],
        wavelength=WL,
        wire_radius=1e-3,
        junctions=[[(0, "start"), (1, "start")]],
        feeds=[(0, L / 4, 1.0 + 0j)],
        **extra,
    )


def oracle_current_elements(solver, coeffs, subdiv=1):
    """antennaknobs `NECPortal.current_elements`, verbatim except for the two
    geometry attribute names. Do not "clean up" — its value is that it is a
    copy of the caller, not a second opinion."""
    fine = []
    arcs = []
    for w_idx, polyline in enumerate(solver.wires_polylines):
        parts = []
        for i, n_e in enumerate(solver.n_per_edge_per_wire[w_idx]):
            seg = np.linspace(polyline[i], polyline[i + 1], n_e * subdiv + 1)
            parts.append(seg if i == 0 else seg[1:])
        knots = np.vstack(parts)
        fine.append(knots)
        step = np.linalg.norm(knots[1:] - knots[:-1], axis=1)
        arcs.append(np.concatenate([[0.0], np.cumsum(step)]))
    currents = solver.currents_at_knots(coeffs, None if subdiv == 1 else arcs)
    mids, moments, nodes, deltas = [], [], [], []
    for knots, cur in zip(fine, currents, strict=True):
        cur = np.asarray(cur)
        element = 0.5 * (cur[1:] + cur[:-1])
        mids.append(0.5 * (knots[1:] + knots[:-1]))
        moments.append(element[:, None] * (knots[1:] - knots[:-1]))
        nodes.append(knots)
        zero = np.zeros(1, dtype=np.complex128)
        deltas.append(np.concatenate([zero, element]) - np.concatenate([element, zero]))
    return (
        np.concatenate(mids, axis=0),
        np.concatenate(moments, axis=0),
        np.concatenate(nodes, axis=0),
        np.concatenate(deltas, axis=0),
    )


def wire_knot_counts(solver, subdiv):
    """Knots per wire — the split a consumer needs to slice the concatenated
    return per wire. `n_elem + 1` per wire, elements being every mesh segment
    subdivided `subdiv` ways."""
    return [sum(npe) * subdiv + 1 for npe in solver.n_per_edge_per_wire]


def element_endpoints(per_wire):
    """Index into the concatenated `nodes` of each element's two end knots.
    Elements never span a wire boundary, so the pairs are consecutive within a
    wire's slice and the last knot of each wire opens no element."""
    left, right, off = [], [], 0
    for n in per_wire:
        idx = np.arange(off, off + n - 1)
        left.append(idx)
        right.append(idx + 1)
        off += n
    return np.concatenate(left), np.concatenate(right)


@pytest.fixture(scope="module")
def solved():
    """One port solution per family on `bent_tee`, built once — the readout is
    what is under test, and re-solving per gate buys nothing."""
    out = {}
    for cls in ALL_CLASSES:
        solver = cls(**_kw(cls, bent_tee()))
        out[cls] = (solver, solver.compute_port_solution())
    return out


# -- gate 1: the semantic oracle -------------------------------------------


@pytest.mark.parametrize("cls", ALL_CLASSES)
@pytest.mark.parametrize("subdiv", [1, 3])
def test_matches_consumer_reference_bit_for_bit(solved, cls, subdiv):
    solver, sol = solved[cls]
    coeffs = sol.coeffs[:, 0]
    got = solver.element_currents(coeffs, subdiv=subdiv)
    want = oracle_current_elements(solver, coeffs, subdiv=subdiv)
    for name, g, w in zip(("mid", "moment", "nodes", "delta"), got, want, strict=True):
        assert np.array_equal(g, w), f"{name} differs from the consumer reference"


# -- gate 2: the four arrays -----------------------------------------------


@pytest.mark.parametrize("cls", ALL_CLASSES)
@pytest.mark.parametrize("subdiv", [1, 3])
def test_shapes_and_dtypes(solved, cls, subdiv):
    solver, sol = solved[cls]
    mid, moment, nodes, delta = solver.element_currents(sol.coeffs[:, 0], subdiv=subdiv)

    per_wire = wire_knot_counts(solver, subdiv)
    n_knots = sum(per_wire)
    n_elem = n_knots - len(per_wire)  # one fewer element than knot, per wire

    assert mid.shape == (n_elem, 3)
    assert moment.shape == (n_elem, 3)
    assert nodes.shape == (n_knots, 3)
    assert delta.shape == (n_knots,)
    assert np.isrealobj(mid) and np.isrealobj(nodes)
    assert moment.dtype == np.complex128
    assert delta.dtype == np.complex128
    # Midpoints really are the element midpoints of the returned knots, wire
    # boundaries excluded — the one relation tying `mid` to `nodes`.
    left, right = element_endpoints(per_wire)
    assert np.allclose(mid, 0.5 * (nodes[left] + nodes[right]), rtol=0, atol=1e-12)


@pytest.mark.parametrize("cls", ALL_CLASSES)
@pytest.mark.parametrize("subdiv", [1, 3])
def test_delta_telescopes_per_wire(solved, cls, subdiv):
    solver, sol = solved[cls]
    _, _, _, delta = solver.element_currents(sol.coeffs[:, 0], subdiv=subdiv)
    scale = np.abs(delta).max()
    assert scale > 0
    off = 0
    for n in wire_knot_counts(solver, subdiv):
        # Nothing enters or leaves a wire except through its own two ends, and
        # those ends carry their adjacent element's full current — so the wire's
        # steps must cancel. Round-off only: the differences are formed before
        # the sum cancels them.
        assert abs(delta[off : off + n].sum()) < 1e-12 * scale
        off += n
    assert off == delta.shape[0]


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_subdiv_refines_the_subdiv_1_mesh(solved, cls):
    solver, sol = solved[cls]
    k = 3
    _, _, coarse, _ = solver.element_currents(sol.coeffs[:, 0])
    _, _, fine, _ = solver.element_currents(sol.coeffs[:, 0], subdiv=k)

    c_off = f_off = 0
    for n_coarse, n_fine in zip(
        wire_knot_counts(solver, 1), wire_knot_counts(solver, k), strict=True
    ):
        assert n_fine == k * (n_coarse - 1) + 1
        sub = fine[f_off : f_off + n_fine : k]
        assert np.allclose(sub, coarse[c_off : c_off + n_coarse], rtol=0, atol=1e-9)
        c_off += n_coarse
        f_off += n_fine


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_moment_is_current_times_element_vector(solved, cls):
    """`moment` is I·dl and nothing else: it must be parallel to the element
    and scale with the element's length, which is what a far-field sum assumes
    when it multiplies by a phase and adds."""
    solver, sol = solved[cls]
    mid, moment, nodes, _ = solver.element_currents(sol.coeffs[:, 0])
    left, right = element_endpoints(wire_knot_counts(solver, 1))
    dl = nodes[right] - nodes[left]
    unit = dl / np.linalg.norm(dl, axis=1, keepdims=True)
    # The component of `moment` off the element direction is exactly zero.
    perp = moment - (moment * unit).sum(axis=1)[:, None] * unit
    assert np.abs(perp).max() < 1e-14 * np.abs(moment).max()
    assert mid.shape == moment.shape


# -- gate 3: the #232 pairing and the input contract ------------------------


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_port_solution_columns_are_valid_input(solved, cls):
    """Every column of a `PortSolution` reads out, and the readout is linear in
    the column — so a consumer may superpose EX sets before OR after it."""
    solver, sol = solved[cls]
    assert sol.n_ports >= 1
    v = np.array([0.3 - 0.7j] * sol.n_ports)
    combined = sol.coeffs @ v

    _, moment_sum, _, delta_sum = solver.element_currents(combined)
    acc_m = np.zeros_like(moment_sum)
    acc_d = np.zeros_like(delta_sum)
    for j in range(sol.n_ports):
        _, m, _, d = solver.element_currents(sol.coeffs[:, j])
        acc_m += v[j] * m
        acc_d += v[j] * d
    scale = np.abs(moment_sum).max()
    assert np.abs(moment_sum - acc_m).max() < 1e-12 * scale
    assert np.abs(delta_sum - acc_d).max() < 1e-12 * np.abs(delta_sum).max()


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_geometry_only_arrays_do_not_depend_on_the_coefficients(solved, cls):
    """`mid` and `nodes` are mesh, not solution — a caller that keeps them
    across excitations is entitled to."""
    solver, sol = solved[cls]
    a_mid, _, a_nodes, _ = solver.element_currents(sol.coeffs[:, 0])
    b_mid, _, b_nodes, _ = solver.element_currents(np.zeros_like(sol.coeffs[:, 0]))
    assert np.array_equal(a_mid, b_mid)
    assert np.array_equal(a_nodes, b_nodes)


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_refuses_a_coefficient_matrix_and_a_zero_subdiv(solved, cls):
    solver, sol = solved[cls]
    with pytest.raises(ValueError, match="ONE solution vector"):
        solver.element_currents(sol.coeffs)
    with pytest.raises(ValueError, match="subdiv must be >= 1"):
        solver.element_currents(sol.coeffs[:, 0], subdiv=0)


# -- the mesh the walk uses is the RESOLVED one -----------------------------


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_auto_per_edge_counts_resolve_to_nsegs(cls):
    """`n_per_edge_per_wire=None` (and a bare int) is normalised in `__init__`
    to one explicit count per edge, and it is that resolved table the readout
    walks — momwire has no wavelength-driven auto-mesh, so `None` means
    `nsegs` on every edge. Pinning it here is what lets `element_currents`
    read `self.n_per_edge_per_wire` instead of re-deriving the mesh."""
    wires = [
        np.array([(0.0, 0.0, 0.0), (0.0, 0.0, L / 2), (L / 4, 0.0, L / 2)]),
        np.array([(0.0, 0.0, 0.0), (0.0, 0.0, -L / 2)]),
    ]
    solver = cls(
        **_kw(
            cls,
            dict(
                wires=wires,
                n_per_edge_per_wire=None,
                nsegs=4,
                wavelength=WL,
                wire_radius=1e-3,
                junctions=[[(0, "start"), (1, "start")]],
                feeds=[(0, L / 4, 1.0 + 0j)],
            ),
        )
    )
    assert solver.n_per_edge_per_wire == [[4, 4], [4]]
    sol = solver.compute_port_solution()
    _, _, nodes, _ = solver.element_currents(sol.coeffs[:, 0], subdiv=2)
    assert nodes.shape[0] == (8 + 8 + 1) + (8 + 1)
    got = solver.element_currents(sol.coeffs[:, 0], subdiv=2)
    want = oracle_current_elements(solver, sol.coeffs[:, 0], subdiv=2)
    assert all(np.array_equal(g, w) for g, w in zip(got, want, strict=True))


def test_every_public_family_shares_one_implementation():
    """One implementation, reached by inheritance from the two classes that
    define `currents_at_knots` — not five copies that can drift."""
    impls = {cls.element_currents for cls in ALL_CLASSES}
    assert len(impls) == 1
