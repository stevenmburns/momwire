"""SG's solve factors G in place — momwire#1224.

`compute_impedance` and `compute_port_solution` called
`scipy.linalg.solve(G, ·)`, which leaves G alone and so hands LAPACK an
F-ordered COPY of it: one extra (n_basis, n_basis) at the solve. They now go
through `_solve_in_place` (`lu_factor(overwrite_a=True)` + `lu_solve`), and
the fill keeps G F-ordered so that LAPACK can take it as it is.

Gates:

  * `_solve_in_place` is `scipy.linalg.solve` to the bit on real SG
    matrices (the same getrf on the same matrix, then getrs), for one and
    several right-hand sides;
  * it factors the caller's storage — no copy — when G is F-ordered;
  * the G each solve receives IS F-ordered on the sparse-product path, for a
    free-space deck, the crossing hub (`_add_crossing_blocks`) and a
    junction-port deck (`_assemble_Z_ported`'s copy): the tripwire for the
    memory property, which nothing else would notice losing;
  * an exactly singular G still raises `LinAlgError`, as `solve` did,
    instead of the inf/NaN `lu_factor` alone would return.

End to end, Z/currents and `compute_port_solution`'s Y and coeffs were
measured `np.array_equal` to 96f1ec7 on 11 decks (hub16 x1/x2/x4, invl
x1/x4, the above-ground array x1/x4, three detached mixed decks and a
junction-port pair).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.linalg
from test_crossing_serve_524 import hub_deck
from test_junction_ports import _port_pair_solver

from momwire import sinusoidal_galerkin as sg
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

C0 = 299792458.0
WL7 = C0 / 7e6


def _free_dipole():
    return dict(
        wires=[np.array([(0.0, -10.0, 0.0), (0.0, 10.0, 0.0)])],
        n_per_edge_per_wire=[[81]],
        feeds=[(0, 10.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=1e-3,
    )


DECKS = {
    "free-dipole": lambda: SinusoidalGalerkinSolver(**_free_dipole()),
    "junction-ports": lambda: _port_pair_solver(
        0.02, 0.01, 40, volts=(1 + 0j, 0.3 + 0j), cls=SinusoidalGalerkinSolver
    ),
    "hub16-crossing": lambda: SinusoidalGalerkinSolver(**hub_deck(n_radials=16)),
}


def _solve_inputs(s, monkeypatch):
    """The (G, rhs) pairs `compute_impedance` and `compute_port_solution`
    hand the solve, copied as they arrive, and the solutions they got.

    The spy is SCOPED to the two solves: left installed, every later call to
    `sg._solve_in_place` in the test appends to the very list the test is
    iterating over, and the loop never ends (it did, and took 24 GB before a
    MemoryError)."""
    seen = []
    solve = sg._solve_in_place

    def spy(G, rhs):
        entry = [G.copy(order="K"), np.array(rhs, copy=True)]
        x = solve(G, rhs)
        entry.append(x)
        seen.append(entry)
        return x

    with monkeypatch.context() as m:
        m.setattr(sg, "_solve_in_place", spy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s.compute_impedance()
            s.compute_port_solution()
    assert sg._solve_in_place is solve
    assert len(seen) == 2
    return tuple(seen)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(n, marks=pytest.mark.slow) if n == "hub16-crossing" else n
        for n in DECKS
    ],
)
def test_the_solve_is_scipy_solve_to_the_bit_and_in_place(name, monkeypatch):
    for G, rhs, x in _solve_inputs(DECKS[name](), monkeypatch):
        # The memory property: the fill handed the solve an F-ordered G.
        assert G.flags.f_contiguous, f"{name}: G reached the solve C-ordered"
        # Bit-identity with what the solve used to be.
        assert np.array_equal(x, scipy.linalg.solve(G, rhs))
        # And it factored the caller's storage rather than a copy.
        work = G.copy(order="F")
        lu, _ = scipy.linalg.lu_factor(work, overwrite_a=True)
        assert np.shares_memory(lu, work)
        again = G.copy(order="F")
        assert np.array_equal(sg._solve_in_place(again, rhs), x)
        assert not np.array_equal(again, G)  # G now holds its factors


def test_a_c_ordered_matrix_is_still_solved_correctly():
    rng = np.random.default_rng(1224)
    A = rng.standard_normal((70, 70)) + 1j * rng.standard_normal((70, 70))
    b = rng.standard_normal(70) + 0j
    ref = scipy.linalg.solve(A, b)
    work = np.ascontiguousarray(A)
    assert np.array_equal(sg._solve_in_place(work, b), ref)
    assert np.array_equal(work, A)  # copied, so left alone


def test_an_exactly_singular_matrix_still_raises():
    A = np.ones((4, 4), dtype=np.complex128, order="F")
    with pytest.raises(np.linalg.LinAlgError):
        scipy.linalg.solve(A.copy(), np.ones(4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", scipy.linalg.LinAlgWarning)
        with pytest.raises(np.linalg.LinAlgError):
            sg._solve_in_place(A, np.ones(4))
