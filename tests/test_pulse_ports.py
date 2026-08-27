"""The pulse family's portal surface: ports, sweeps, and the charge it reports.

momwire#564. Before it, `PulseSolver` and `HarringtonSolver` had neither
`compute_port_solution` nor `_port_count` nor `current_slopes`, and that is
the whole of why the family had no portal surface — `momwire.portal` calls the
first and third unconditionally, so momwire#559's roster entry raised
`AttributeError` on every deck. An escaping exception is not a refusal: it
takes the daemon down mid-frame while the host waits on a sentinel that never
arrives.

Two of the three are now here. The third is deliberately still absent on
`PulseSolver` and that absence is load-bearing — see the charge section below.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import BSplineSolver, PulseSolver
from momwire.harrington import HarringtonSolver

WAVELENGTH = 299792458.0 / 14e6
_DIPOLE = [np.array([[0.0, -5.0, 10.0], [0.0, 5.0, 10.0]])]
_PAIR = [
    np.array([[0.0, -5.0, 10.0], [0.0, 5.0, 10.0]]),
    np.array([[3.0, -5.0, 10.0], [3.0, 5.0, 10.0]]),
]
FAMILY = (PulseSolver, HarringtonSolver)


def _solver(cls, n=21, wires=None, **kw):
    return cls(
        wires=wires or _DIPOLE,
        nsegs=n,
        wire_radius=0.001,
        wavelength=WAVELENGTH,
        **kw,
    )


# --------------------------------------------------------------------------
# ports


@pytest.mark.parametrize("cls", FAMILY)
def test_the_y_matrix_is_read_off_the_port_solution(cls):
    """The house rule for a family that serves ports: one entry point.

    `compute_y_matrix` is `compute_port_solution().y`, so the two cannot drift
    — which is the drift momwire#232 closed everywhere else and this family
    was outside of.  Equality is BIT exact rather than approximate because
    there is only one expression being evaluated, not two agreeing ones.
    """
    sol = _solver(cls).compute_port_solution()
    assert np.array_equal(sol.y, _solver(cls).compute_y_matrix())


@pytest.mark.parametrize("cls", FAMILY)
def test_the_port_readout_is_the_y_matrix_itself(cls):
    """`port_currents is y` — and on this family that is not a convention.

    A pulse coefficient IS the segment current in amperes, so the port readout
    is a row selection out of `coeffs` rather than a basis evaluation, and
    there is no second expression of it that could disagree.
    """
    sol = _solver(cls).compute_port_solution()
    assert sol.port_currents is sol.y


@pytest.mark.parametrize("cls", FAMILY)
def test_one_fill_answers_every_port_and_the_columns_come_back(cls):
    """The point of `PortSolution`: the drive columns are not dropped.

    Two ports, so the shapes are distinguishable — `coeffs` is (n_dof,
    n_ports) and `y` is (n_ports, n_ports); a family that returned the same
    array for both would pass a one-port check.
    """
    solver = _solver(cls, n=11, wires=_PAIR, feeds=[(0, None, 1.0), (1, None, 0.0)])
    sol = solver.compute_port_solution()

    assert sol.n_ports == 2 == solver._port_count()
    assert sol.y.shape == (2, 2)
    assert sol.coeffs.shape == (22, 2)  # 11 segments per wire, one dof each
    # Column j IS the solution for a 1 V drive at port j: its entry at port
    # j's own row is that port's self-admittance.
    for j in range(2):
        assert sol.coeffs[:, j][np.argmax(np.abs(sol.coeffs[:, j]))] != 0


@pytest.mark.parametrize("cls", FAMILY)
def test_the_port_count_needs_no_solve(cls):
    """`_port_count` is asked before any fill, so an empty sweep still gets a
    shape.  This family has no junction ports and no node gaps to append —
    both are refused by presence at construction — so the feed list is it."""
    solver = _solver(cls, n=11, wires=_PAIR, feeds=[(0, None, 1.0), (1, None, 0.0)])
    assert solver._port_count() == 2
    assert solver.z is None  # nothing was filled to answer that


# --------------------------------------------------------------------------
# sweeps


@pytest.mark.parametrize("cls", FAMILY)
def test_a_swept_port_solve_is_the_stacked_single_k_one(cls):
    """momwire#252's contract, and the reason the per-k core is the single-k
    entry point: with no batched assembly to preserve there is nothing for the
    swept path to do differently, so the two agree to the ULP rather than to a
    tolerance.  A tolerance here would hide a second copy of the port algebra,
    which is exactly what #252 was about.
    """
    ks = np.array([2 * np.pi * f / 299792458.0 for f in (13e6, 14e6, 15e6)])
    swept = _solver(cls).compute_port_solution_swept(ks)
    assert len(swept) == 3
    for kk, got in zip(ks, swept):
        one = _solver(cls)
        one.wavelength = 2 * np.pi / kk
        one.k = float(kk)
        one.omega = one.k * one.c
        assert np.array_equal(got.y, one.compute_port_solution().y)


@pytest.mark.parametrize("cls", FAMILY)
def test_a_sweep_puts_the_frequency_triple_back(cls):
    """`_k_restored`'s whole job.  A sweep that leaves `k` where it stepped to
    makes the NEXT call on the instance answer a frequency nobody asked for.
    """
    solver = _solver(cls)
    before = (solver.k, solver.wavelength, solver.omega)
    solver.compute_y_matrix_swept(np.array([2 * np.pi * 13e6 / 299792458.0]))
    assert (solver.k, solver.wavelength, solver.omega) == before


# --------------------------------------------------------------------------
# the charge, which is where the two rows part company (momwire#611 step 4)


def test_only_the_row_with_a_charge_DENSITY_reports_one():
    """The pair, as a presence check — and the presence follows the physics.

    Both rows carry the same piecewise-constant current, so `dI/ds` is a delta
    comb for both.  `HarringtonSolver` spreads each node's charge over that
    node's dual cell, which turns the comb into a real piecewise-constant
    density; `PulseSolver` leaves it as two POINT charges per segment and has
    no density anywhere.  So one has the method and the other must not.

    Absence rather than a raising stub because both seams gate on presence:
    `eznec._serve` checks `hasattr` and writes a printout refusal naming the
    basis, which a stub would defeat.
    """
    assert hasattr(HarringtonSolver, "current_slopes")
    assert not hasattr(PulseSolver, "current_slopes")
    assert issubclass(HarringtonSolver, PulseSolver)


def test_the_reported_slope_is_the_cell_the_matrix_was_filled_with():
    """Read off `_node_map`, not re-derived — so the density reported is the
    one `_charge_stencil` integrated over, and a change to the cell rule
    cannot move one without moving the other.

    Rebuilt here by hand from the parent's own charge convention (`Q_R =
    +I_n/(jw)` at the arc-h end, `Q_L = -I_n/(jw)` at the arc-0 end) so the
    assertion is against the FORMULATION rather than against the code.
    """
    solver = _solver(HarringtonSolver, n=11)
    _z, coeffs = solver.compute_impedance()
    geom = solver._build_geometry()
    _ln, _rn, pieces, cell_of_piece = solver._node_map(geom)

    lengths = np.zeros(int(cell_of_piece.max()) + 1)
    np.add.at(lengths, cell_of_piece, pieces["h_per_seg"])

    # One wire, so cell j is knot j and the sum is the two segments meeting
    # there; a free end has one segment and a HALF-length cell.
    by_hand = np.empty(len(lengths), dtype=np.complex128)
    by_hand[0] = coeffs[0] / lengths[0]
    by_hand[-1] = -coeffs[-1] / lengths[-1]
    for j in range(1, len(lengths) - 1):
        by_hand[j] = (coeffs[j] - coeffs[j - 1]) / lengths[j]

    assert np.array_equal(np.asarray(solver.current_slopes(coeffs)[0]), by_hand)


def test_the_charge_over_a_free_wire_sums_to_nothing():
    """Conservation, which is the one property no sampling convention can
    fudge: the current is zero at both free ends, so the integral of `dI/ds`
    over the wire is zero, and the wire's total charge with it.

    Exact rather than approximate — it is a telescoping sum over the cells.
    """
    solver = _solver(HarringtonSolver, n=21)
    _z, coeffs = solver.compute_impedance()
    geom = solver._build_geometry()
    _ln, _rn, pieces, cell_of_piece = solver._node_map(geom)
    lengths = np.zeros(int(cell_of_piece.max()) + 1)
    np.add.at(lengths, cell_of_piece, pieces["h_per_seg"])

    total = np.sum(np.asarray(solver.current_slopes(coeffs)[0]) * lengths)
    assert abs(total) < 1e-18


def test_a_symmetric_antenna_reports_a_symmetric_charge():
    """What the interpolated reading buys, and why it is not a lookup.

    The charge cells are centred on KNOTS and bounded by segment CENTRES, so
    this family's charge grid is staggered half a segment from its current
    grid — and the portal's charge column samples at segment centres, i.e. on
    every cell boundary at once.  Taking one side there would tilt the whole
    column half a cell and destroy this symmetry; interpolating between cell
    centres lands on the mean of the two, which is symmetric.

    Also pinned at the FEED, which is where a one-sided reading is worst: the
    two flanking cells are equal and opposite there, so the answer is zero,
    and a one-sided read returns the full discontinuity instead (measured at
    0.23 of peak while this was a float-equality tie-break).
    """
    solver = _solver(HarringtonSolver, n=41)
    _z, coeffs = solver.compute_impedance()
    arc = solver._build_geometry()["per_wire"][0]["arc_at_knot"]
    centres = 0.5 * (arc[:-1] + arc[1:])

    got = np.asarray(solver.current_slopes(coeffs, [centres])[0])
    assert np.max(np.abs(got + got[::-1])) < 1e-12 * np.max(np.abs(got))
    assert abs(solver.current_slopes(coeffs, [np.array([5.0])])[0][0]) < 1e-12


def test_sampling_at_a_knot_is_the_cell_itself():
    """The other end of the same rule: a knot is a cell's MIDDLE, so asking
    for it explicitly must give what the default (no `s_array`) gives.  If
    interpolation moved it, the default and the sampled forms would be two
    different readings of one quantity.
    """
    solver = _solver(HarringtonSolver, n=21)
    _z, coeffs = solver.compute_impedance()
    arc = solver._build_geometry()["per_wire"][0]["arc_at_knot"]
    assert np.allclose(
        np.asarray(solver.current_slopes(coeffs, [arc])[0]),
        np.asarray(solver.current_slopes(coeffs)[0]),
        rtol=0,
        atol=0,
    )


@pytest.mark.slow
def test_the_reported_charge_converges_onto_the_converged_basis():
    """That the reading is RIGHT and not merely self-consistent.

    Shape only — normalised by its own peak — because this row's impedance is
    still converging at these meshes and folding that in would measure the
    wrong thing.  Against a converged degree-2 B-spline the deviation halves
    per mesh doubling, which is the O(1/N) `HarringtonSolver`'s own docstring
    claims for this scheme:

        N =  51   27.0 %
        N = 101   12.4 %
        N = 201    5.8 %
        N = 401    2.8 %

    Monotone is the assertion; the digits are the record.  A sampling rule
    that took one side at a cell boundary is NOT monotone here — it read
    27.1 / 15.8 / 5.8 / 23.5, and the tail is what found the bug.
    """
    xs = np.linspace(1.0, 9.0, 17)

    def shape(cls, n, **kw):
        solver = _solver(cls, n=n, **kw)
        _z, coeffs = solver.compute_impedance()
        got = np.asarray(solver.current_slopes(coeffs, [xs])[0])
        return got / np.max(np.abs(got))

    ref = shape(BSplineSolver, 201, degree=2)
    devs = [
        np.max(np.abs(shape(HarringtonSolver, n) - ref)) for n in (51, 101, 201, 401)
    ]
    assert devs == sorted(devs, reverse=True), devs
    assert devs[-1] < 0.04
