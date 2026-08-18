"""RazorSolver's public multi-port solve result (#429 rank-9).

`compute_port_solution` / `compute_port_solution_swept` close the sharing
audit's (#429) rank-9 item: razor onto `_SweptPortSolutions`, the same
`PortSolution` contract `_port_solution.py` documents for every other
family. Razor has no batched C++ accelerator (`docs/razor-solver.md` "No
C++ accelerator"), so it takes `SinusoidalSolver`'s spelling rather than
`BSplineSolver`'s: the swept generator is a bare per-k loop over
`compute_port_solution`, not a second, chunked code path, which is what
makes the swept gate below a BIT gate rather than a reassociation-tolerance
one.

The gates here are the same properties `tests/test_port_solution.py` pins
for the B-spline / sinusoidal families, read against razor's own idiom:

1. `compute_y_matrix()` IS `compute_port_solution().y` — bit for bit, in
   free space, under all three served grounds, and in both quadrature lanes
   (`nec5_quadrature`).
2. The columns really are solutions — `_assemble_Z(geom, k) @ coeffs[:, j]`
   reproduces port j's unit right-hand side.
3. `coeffs @ V` reproduces what `compute_impedance` solves for the
   configured voltages — the superposition property, and the tightest
   available statement of port ORDER: razor's ports are `feeds`, in order,
   with nothing after them (`junction_ports` and `node_gaps` are both
   refused at construction).
4. One fill, one factorisation, regardless of port count.
5. `compute_port_solution_swept` == a per-k `compute_port_solution` loop,
   bit for bit, over a ground whose ε̃ moves with ω — the house swept gate
   (`tests/test_razor_refl_coef_ground.py`'s
   `test_swept_refl_coef_matches_the_per_k_solves`), port-solution edition.

The portal end-to-end gate (`--basis razor` / `--basis razor-nec5` driving a
live deck) lives in `tests/test_portal.py`, beside the rest of the
`--basis` battery.
"""

import numpy as np
import pytest
import scipy.linalg

import momwire.razor as razor_mod
from momwire import PortSolution, RazorSolver

WL = 10.0
L = 0.48 * WL


def _dipole(**extra):
    return dict(
        wires=[np.array([(0.0, 0.0, -L / 2), (0.0, 0.0, L / 2)])],
        n_per_edge_per_wire=[[24]],
        wavelength=WL,
        wire_radius=1e-3,
        **extra,
    )


def _elevated_dipole(**extra):
    """Clear of the z=0 plane, so all three served grounds are legal here."""
    return dict(
        wires=[np.array([(0.0, 0.0, 3.0), (0.0, 0.0, 3.0 + L)])],
        n_per_edge_per_wire=[[24]],
        wavelength=WL,
        wire_radius=1e-3,
        **extra,
    )


def _two_feed(**extra):
    """Two parallel dipoles, each gap-fed — a genuine 2-port Y, no junctions."""
    wires = [
        np.array([(0.0, 0.0, -L / 2), (0.0, 0.0, L / 2)]),
        np.array([(1.5, 0.0, -L / 2), (1.5, 0.0, L / 2)]),
    ]
    return dict(
        wires=wires,
        n_per_edge_per_wire=[[20], [20]],
        wavelength=WL,
        wire_radius=1e-3,
        feeds=[(0, None, 1.0 + 0.0j), (1, None, 0.5 - 0.25j)],
        **extra,
    )


# free space, and the three grounds this class serves (module docstring
# "Scope"), each legal on the elevated dipole above.
GROUNDS = {
    "free": {},
    "pec": dict(ground_z=0.0),
    "refl-coef": dict(ground_z=0.0, ground_eps=13.0 - 0.005j),
    "sommerfeld": dict(
        ground_z=0.0, ground_eps=13.0 - 0.005j, ground_model="sommerfeld"
    ),
}


# ---- Gate 1: compute_y_matrix() IS compute_port_solution().y ---------------


@pytest.mark.parametrize("ground", sorted(GROUNDS), ids=lambda g: g)
@pytest.mark.parametrize("lane", [False, True], ids=["gl", "nec5"])
def test_y_matrix_is_the_port_solutions_y(ground, lane):
    """Bit for bit, on the same deck, across every served ground and both
    quadrature lanes: `compute_y_matrix` is implemented as
    `compute_port_solution().y`, so a divergence here would mean the two
    entry points stopped describing the same solve."""
    kw = _elevated_dipole(nec5_quadrature=lane, **GROUNDS[ground])
    sol = RazorSolver(**kw).compute_port_solution()
    y = RazorSolver(**kw).compute_y_matrix()
    assert isinstance(sol, PortSolution)
    assert sol.n_ports == 1
    assert sol.y.shape == y.shape == (1, 1)
    assert sol.port_currents is sol.y
    assert sol.y.tobytes() == y.tobytes()


def test_port_solution_reproduces_the_original_fill_algebra():
    """Reassembles Z and the unit-drive columns independently — through the
    same `_assemble_Z` / `_feed_basis_indices` pair the pre-#429-rank-9
    `compute_y_matrix` body called inline — and checks `compute_port_solution`
    reproduces it bit for bit. This is the "no reassociation" half of the
    consistency gate: the new entry point did not just land on the same
    ANSWER, it runs the identical fill and the identical
    `scipy.linalg.solve` call in the identical order the code it replaced
    did (see `docs/design/solver-architecture.md` §6.11)."""
    kw = _dipole()
    solver = RazorSolver(**kw)
    geom = solver._build_geometry()
    Z = solver._assemble_Z(geom, solver.k)
    idx = solver._feed_basis_indices(geom)
    B = np.zeros((geom["n_basis_total"], len(idx)), dtype=np.complex128)
    for j, m_j in enumerate(idx):
        B[m_j, j] = 1.0
    y_manual = scipy.linalg.solve(Z, B)[idx, :]

    sol = RazorSolver(**kw).compute_port_solution()
    assert sol.y.tobytes() == y_manual.tobytes()


# ---- Gate 2: the columns are solutions --------------------------------------


def test_columns_solve_their_own_port():
    """Column j of `coeffs` satisfies `Z @ coeffs[:, j] == B[:, j]` against
    the operator `_assemble_Z` builds independently — a fresh reassembly
    through the family's own public routine, not a copy of
    `compute_port_solution`'s internals."""
    kw = _two_feed()
    solver = RazorSolver(**kw)
    sol = solver.compute_port_solution()
    geom = solver._build_geometry()
    Z = solver._assemble_Z(geom, solver.k)
    idx = solver._feed_basis_indices(geom)
    B = np.zeros((geom["n_basis_total"], len(idx)), dtype=np.complex128)
    for j, m_j in enumerate(idx):
        B[m_j, j] = 1.0
    resid = Z @ sol.coeffs - B
    rel = np.abs(resid).max() / np.abs(B).max()
    assert rel < 1e-11, f"port-column residual {rel:.3e}"


# ---- Gate 3: column j is the unit drive of port j, feeds in order ----------


def test_columns_superpose_to_the_configured_excitation():
    """`coeffs @ V` reproduces the coefficient vector `compute_impedance`
    solves for directly — the superposition property the API exists for,
    and the tightest available statement of the port order: swap the two
    feeds' voltages and this fails."""
    kw = _two_feed()
    sol = RazorSolver(**kw).compute_port_solution()
    _z, coeffs = RazorSolver(**kw).compute_impedance()
    V = np.array([v for _, _, v in kw["feeds"]], dtype=np.complex128)
    recon = sol.coeffs @ V
    rel = np.abs(recon - coeffs).max() / np.abs(coeffs).max()
    assert rel < 1e-9, f"superposition mismatch {rel:.3e}"


def test_ports_are_feeds_in_order():
    """The two-feed fixture's Y is 2x2 in `feeds` order — pinned against a
    per-feed driving-point impedance with the OTHER feed shorted (V=0),
    which is exactly `1 / Y[i, i]` by the admittance matrix's own
    definition. Swapping the two feed indices in `kw["feeds"]` would swap
    which check passes, which is the port-order statement."""
    kw = _two_feed()
    sol = RazorSolver(**kw).compute_port_solution()
    assert sol.y.shape == (2, 2)
    for i in range(2):
        solo = list(kw["feeds"])
        other = 1 - i
        solo[other] = (solo[other][0], solo[other][1], 0.0j)
        z_solo, _c = RazorSolver(**{**kw, "feeds": solo}).compute_impedance()
        want = z_solo[i]
        got = 1.0 / sol.y[i, i]
        assert abs(got - want) < 1e-9 * abs(want), (i, got, want)


# ---- Gate 4: one fill, one factorisation ------------------------------------


def test_one_fill_and_one_factorisation_per_call(monkeypatch):
    """N ports cost one fill and one `scipy.linalg.solve`, not N — the
    contract SimNEC's daemon (behind the portal) is built on."""
    kw = _two_feed()
    solver = RazorSolver(**kw)

    fills = []
    orig_fill = razor_mod.RazorSolver._assemble_Z_from_prepared

    def spy_fill(self, *a, **kw2):
        fills.append(1)
        return orig_fill(self, *a, **kw2)

    monkeypatch.setattr(razor_mod.RazorSolver, "_assemble_Z_from_prepared", spy_fill)

    solves = []
    orig_solve = razor_mod.scipy.linalg.solve

    def spy_solve(*a, **kw2):
        solves.append(1)
        return orig_solve(*a, **kw2)

    monkeypatch.setattr(razor_mod.scipy.linalg, "solve", spy_solve)

    solver.compute_port_solution()
    assert len(fills) == 1, f"_assemble_Z_from_prepared ran {len(fills)}x"
    assert len(solves) == 1, f"scipy.linalg.solve ran {len(solves)}x"


# ---- basis handle and export -------------------------------------------------


def test_basis_handle_is_opaque_but_present():
    """`basis` is a handle, not an interface: the contract is only that it
    is there, that it is the same object for every port of ONE solution,
    and that a fresh solve makes a fresh one."""
    kw = _dipole()
    solver = RazorSolver(**kw)
    first = solver.compute_port_solution()
    second = solver.compute_port_solution()
    assert first.basis is not None
    assert first.basis is not second.basis
    assert type(first.basis) is type(second.basis)


# ---- Gate 5: the swept entry point is the per-k core, stacked (#252) ------


def test_port_solution_swept_matches_the_per_k_solves_bit_for_bit():
    """`compute_port_solution_swept` over a refl-coef ground built as an
    `(eps_r, sigma)` tuple == solving each k alone, bit for bit — the
    ω-boundary gate (`test_razor_refl_coef_ground.py`'s
    `test_swept_refl_coef_matches_the_per_k_solves`, port-solution
    edition). ε̃ = εr − jσ/(ωε₀) MOVES with the wavenumber here, so a build
    that hoisted the Fresnel weights into the k-independent prepare half —
    the mistake unit 4's docstring warns against — would return the
    sweep's nominal-ε̃ answer at every k and fail at the two off-design
    rungs. Razor has no batched fast path, so this is a bit gate rather
    than a reassociation-tolerance one: `_port_solutions_swept` IS a loop
    over `compute_port_solution`."""
    kw = _elevated_dipole(ground_z=0.0, ground_eps=(13.0, 0.005))
    solver = RazorSolver(**kw)
    ks = solver.k * np.array([0.93, 1.0, 1.07])
    swept = solver.compute_port_solution_swept(ks)
    assert len(swept) == 3

    saved = (solver.k, solver.wavelength, solver.omega)
    try:
        for i, kk in enumerate(ks):
            solver.k = float(kk)
            solver.omega = solver.k * solver.c
            solver.wavelength = solver.c / (solver.omega / (2 * np.pi))
            one = solver.compute_port_solution()
            assert swept[i].y.tobytes() == one.y.tobytes(), f"k index {i}: y"
            assert swept[i].coeffs.tobytes() == one.coeffs.tobytes(), (
                f"k index {i}: coeffs"
            )
    finally:
        solver.k, solver.wavelength, solver.omega = saved


def test_y_matrix_swept_is_the_swept_port_solutions_y():
    """`compute_y_matrix_swept` (now `_SweptPortSolutions`'s generic
    implementation, not razor's own hand-rolled one) is
    `compute_port_solution_swept`'s `y` field and nothing else."""
    kw = _dipole()
    ks = RazorSolver(**kw).k * np.array([0.9, 1.0, 1.1])
    y_swept = RazorSolver(**kw).compute_y_matrix_swept(ks)
    sols = RazorSolver(**kw).compute_port_solution_swept(ks)
    assert y_swept.shape == (3, 1, 1)
    for i, sol in enumerate(sols):
        assert y_swept[i].tobytes() == sol.y.tobytes()


def test_swept_impedance_still_matches_the_per_k_solves():
    """`compute_impedance_swept` is untouched by this unit (it solves a
    superposed single right-hand side, not one column per port) — pinned so
    a future edit near `compute_port_solution` can't quietly move it."""
    kw = _dipole()
    solver = RazorSolver(**kw)
    ks = solver.k * np.array([0.92, 1.0, 1.08])
    swept = solver.compute_impedance_swept(ks)

    saved = (solver.k, solver.wavelength, solver.omega)
    try:
        for i, kk in enumerate(ks):
            solver.k = float(kk)
            solver.omega = solver.k * solver.c
            solver.wavelength = solver.c / (solver.omega / (2 * np.pi))
            one, _coeffs = solver.compute_impedance()
            assert swept[i] == one, f"k index {i}"
    finally:
        solver.k, solver.wavelength, solver.omega = saved
