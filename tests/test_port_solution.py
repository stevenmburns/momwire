"""`compute_port_solution()` — the public multi-port solve result (#232).

Every family already computed the per-port solution columns and threw them
away, returning only Y. The public API hands them back. The gates here are the
four properties an external consumer (SimNEC's NEC portal: one geometry, N
`EX` sets, its own field evaluation afterwards) is entitled to rely on:

1. `compute_y_matrix()` IS `compute_port_solution().y` — bit for bit, on every
   family, under grounds, junction ports, enrichment and the extended kernel.
   (Identity against the *pre-#232* implementation was verified out of band
   with a hex-float battery across the same matrix; what a test in the tree can
   pin forever is that the two entry points cannot drift from each other.)
2. The columns really are solutions — the family's own operator applied to
   column j reproduces port j's right-hand side (up to the Lagrange multipliers
   of the junction constraints, which the columns satisfy).
3. Column j is the UNIT drive of port j, in port order [feeds…, junction
   ports…]: `coeffs @ V` reproduces what `compute_impedance` solves for the
   configured voltages, which is the superposition the whole API exists for.
4. One fill, one factorisation, regardless of port count.

Plus the refusals: where a family will not serve a configuration,
`compute_port_solution` refuses in exactly the same way `compute_y_matrix`
does.
"""

import numpy as np
import pytest

import momwire
from momwire import (
    ArrayBlockSolver,
    BSplineSolver,
    PortSolution,
    SinusoidalGalerkinSolver,
    SinusoidalSolver,
    HMatrixSolver,
)

WL = 10.0
L = 0.48 * WL

SPLINE = [BSplineSolver, HMatrixSolver, ArrayBlockSolver]
SEGMENT = [SinusoidalSolver, SinusoidalGalerkinSolver]
ALL_CLASSES = SPLINE + SEGMENT

# The accelerated operator's own approximations sit above the port algebra, so
# the iterative gates tighten both knobs and compare near the solve tolerance.
TIGHT = dict(aca_tol=1e-10, solve_tol=1e-11)


def _tight(cls, kw):
    return {**kw, **TIGHT} if cls in (HMatrixSolver, ArrayBlockSolver) else kw


def plain_multifeed(**extra):
    """Two parallel dipoles, each gap fed — a genuine 2-port Y, no junctions."""
    wires = [
        np.array([(0.0, 0.0, -L / 2), (0.0, 0.0, L / 2)]),
        np.array([(1.5, 0.0, -L / 2), (1.5, 0.0, L / 2)]),
    ]
    return dict(
        wires=wires,
        n_per_edge_per_wire=[[21], [21]],
        wavelength=WL,
        wire_radius=1e-3,
        feeds=[(0, None, 1.0 + 0j), (1, None, 0.5 - 0.25j)],
        **extra,
    )


def single_feed(**extra):
    """One dipole, one port — the port-count contrast for the fill counters."""
    wires = [np.array([(0.0, 0.0, -L / 2), (0.0, 0.0, L / 2)])]
    return dict(
        wires=wires,
        n_per_edge_per_wire=[[21]],
        wavelength=WL,
        wire_radius=1e-3,
        feeds=[(0, None, 1.0 + 0j)],
        **extra,
    )


def junction_ports(**extra):
    """Split dipole with tip whiskers: one gap feed plus two junction ports at
    the inner nodes (#172's paired-drive geometry, shrunk). Mixed feed/port
    readout, so it pins the port ORDER as well as the algebra."""
    delta, whisker = 0.04, 0.01
    wires = [
        np.array([(0.0, -L / 2, 0.0), (0.0, -delta / 2, 0.0)]),
        np.array([(0.0, -delta / 2, 0.0), (whisker, -delta / 2, 0.0)]),
        np.array([(0.0, delta / 2, 0.0), (0.0, L / 2, 0.0)]),
        np.array([(0.0, delta / 2, 0.0), (whisker, delta / 2, 0.0)]),
    ]
    return dict(
        wires=wires,
        n_per_edge_per_wire=[[20], [1], [20], [1]],
        wavelength=WL,
        wire_radius=1e-3,
        junctions=[[(0, "end"), (1, "start")], [(2, "start"), (3, "start")]],
        feeds=[(0, L / 4, 0.25 + 0j)],
        junction_ports=[(0, 0.5 + 0j), (1, -0.5 + 0j)],
        **extra,
    )


def pec_ground(**extra):
    """Two verticals over a PEC ground plane, both gap fed."""
    h = 0.24 * WL
    wires = [
        np.array([(0.0, 0.0, 0.05), (0.0, 0.0, 0.05 + h)]),
        np.array([(1.2, 0.0, 0.05), (1.2, 0.0, 0.05 + h)]),
    ]
    return dict(
        wires=wires,
        n_per_edge_per_wire=[[15], [15]],
        wavelength=WL,
        wire_radius=1e-3,
        ground_z=0.0,
        feeds=[(0, None, 1.0 + 0j), (1, None, 1.0 + 0j)],
        **extra,
    )


def enrichment(**extra):
    """Three-way junction with singular enrichment on, two gap feeds. The
    spline families only — and the accelerated ones fall back to the dense
    path here, which is the `super().compute_port_solution()` branch."""
    arm = 0.2 * WL
    wires = [
        np.array([(0.0, 0.0, 0.0), (arm, 0.0, 0.0)]),
        np.array([(0.0, 0.0, 0.0), (-arm, 0.0, 0.0)]),
        np.array([(0.0, 0.0, 0.0), (0.0, 0.0, arm)]),
    ]
    return dict(
        wires=wires,
        n_per_edge_per_wire=[[12], [12], [12]],
        wavelength=WL,
        wire_radius=1e-3,
        junctions=[[(0, "start"), (1, "start"), (2, "start")]],
        feeds=[(0, arm / 2, 1.0 + 0j), (1, arm / 2, 0.25 + 0j)],
        use_singular_enrichment=True,
        **extra,
    )


# Which fixture each family can be handed. The refusals get their own tests.
CASES = []
for _cls in SPLINE:
    CASES += [
        (_cls, f) for f in (plain_multifeed, junction_ports, pec_ground, enrichment)
    ]
CASES += [
    (SinusoidalSolver, plain_multifeed),
    (SinusoidalSolver, pec_ground),
    (SinusoidalSolver, lambda **e: plain_multifeed(extended_kernel=True, **e)),
    (SinusoidalGalerkinSolver, plain_multifeed),
    (SinusoidalGalerkinSolver, junction_ports),
    (SinusoidalGalerkinSolver, pec_ground),
]


def _case_id(case):
    cls, fx = case
    name = getattr(fx, "__name__", "lambda")
    return f"{cls.__name__}-{'extended_kernel' if name == '<lambda>' else name}"


def _port_voltages(kw):
    return np.array(
        [v for _w, _s, v in kw["feeds"]]
        + [v for _j, v in kw.get("junction_ports") or []],
        dtype=np.complex128,
    )


# ---- Gate 1: compute_y_matrix IS compute_port_solution().y -----------------


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_y_matrix_is_the_port_solutions_y(case):
    """Bit for bit, on the same solver instance, across grounds, junction
    ports, singular enrichment and the extended kernel. `compute_y_matrix` is
    implemented as `compute_port_solution().y`, so a divergence here would mean
    a solve stopped being reproducible — either way the portal's single fill
    and the library's Y would be describing different antennas."""
    cls, fixture = case
    kw = _tight(cls, fixture())
    solver = cls(**kw)
    sol = solver.compute_port_solution()
    y = solver.compute_y_matrix()
    assert isinstance(sol, PortSolution)
    assert sol.y.shape == y.shape == (len(_port_voltages(kw)),) * 2
    assert sol.n_ports == y.shape[0]
    assert sol.y.tobytes() == y.tobytes()


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_port_currents_is_the_y_matrix(case):
    """Not "equal to machine precision" — the SAME array. The field exists so
    a consumer can assert the identity rather than assume it, which is only
    worth anything if the implementation makes it structural."""
    cls, fixture = case
    sol = cls(**_tight(cls, fixture())).compute_port_solution()
    assert sol.port_currents is sol.y


# ---- Gate 2: the columns are solutions -------------------------------------


def _constrained_residual(matvec, X, B, A):
    """max_j ‖Z·x_j + Aᵀλ_j − b_j‖/‖b_j‖ minimised over λ, and max_j ‖A·x_j‖.

    With junction constraints the columns solve the SADDLE system, so the raw
    residual is not zero — it is exactly the Lagrange term. Projecting it out
    with a least squares against Aᵀ is the honest statement of "column j solves
    port j's problem"; `A·x_j` then says the constraints are actually met.
    """
    res, cons = 0.0, 0.0
    for j in range(X.shape[1]):
        r = matvec(X[:, j]) - B[:, j]
        if A is not None and A.shape[0]:
            lam, *_ = np.linalg.lstsq(A.T.astype(np.complex128), -r, rcond=None)
            r = r + A.T @ lam
            cons = max(cons, float(np.abs(A @ X[:, j]).max()))
        res = max(res, float(np.linalg.norm(r) / np.linalg.norm(B[:, j])))
    return res, cons


def _sinusoidal_system(solver):
    geom = solver._build_geometry()
    G, _seg_view = solver._assemble_Z(geom, solver.k)
    B = np.zeros((geom["n_segs"], len(geom["feed_segs"])), dtype=np.complex128)
    for j, fi in enumerate(geom["feed_segs"]):
        B[fi, j] = -1.0 / geom["seg_h"][fi]
    return G, B, None


def _galerkin_system(solver):
    geom = solver._build_geometry()
    G, seg_view = solver._assemble_Z_ported(geom, solver.k)
    return G, solver._drive_columns(geom, seg_view, solver.k), None


def _spline_system(solver):
    """The dense B-spline operator, its port columns and the constraint rows —
    rebuilt here through the solver's own assembly, not copied from it."""
    geom = solver._build_geometry()
    supp_seg, polys, kcl_A, wire_knots, wire_basis_global = (
        solver._build_basis_polynomials(geom)
    )
    n = supp_seg.shape[0]
    Z = solver._compute_Z_operator(geom, supp_seg, polys)
    B = np.zeros((n, len(solver.feeds)), dtype=np.complex128)
    for j, (w_i, arc_i, _v) in enumerate(solver.feeds):
        arc_at_knot = geom["per_wire"][w_i]["arc_at_knot"]
        B[:, j] = solver._build_source_vector(
            geom,
            wire_knots,
            wire_basis_global,
            n,
            wi=w_i,
            s_f=arc_i if arc_i is not None else arc_at_knot[-1] / 2.0,
        )
    kcl_con, port_A, _port_V = solver._split_kcl_ports(kcl_A)
    B = np.hstack([B, port_A.T.astype(np.complex128)])
    return Z, B, kcl_con


RESIDUAL_CASES = [
    (SinusoidalSolver, plain_multifeed, _sinusoidal_system, 1e-12),
    (SinusoidalSolver, pec_ground, _sinusoidal_system, 1e-12),
    (SinusoidalGalerkinSolver, plain_multifeed, _galerkin_system, 1e-12),
    # The ported Galerkin system carries the port rows at O(1) alongside field
    # entries orders of magnitude smaller, so its conditioning — not the port
    # algebra — sets the floor here.
    (SinusoidalGalerkinSolver, junction_ports, _galerkin_system, 1e-9),
    (BSplineSolver, plain_multifeed, _spline_system, 1e-12),
    (BSplineSolver, junction_ports, _spline_system, 1e-12),
    (BSplineSolver, pec_ground, _spline_system, 1e-12),
]


@pytest.mark.parametrize(
    "cls,fixture,system,tol",
    RESIDUAL_CASES,
    ids=lambda v: getattr(v, "__name__", None) or str(v),
)
def test_columns_solve_their_own_port(cls, fixture, system, tol):
    """Dense families: reassemble the operator and the port right-hand sides
    through the family's own routines and check every returned column against
    them. Enrichment is excluded on purpose — reproducing the augmented block
    system here would be a copy of the implementation, not an independent
    check of it."""
    kw = fixture()
    sol = cls(**kw).compute_port_solution()
    Z, B, A = system(cls(**kw))
    X = sol.coeffs
    assert X.shape == (Z.shape[0], B.shape[1])
    res, cons = _constrained_residual(lambda v: Z @ v, X, B, A)
    assert res < tol, f"port-column residual {res:.3e}"
    assert cons < 1e-10, f"KCL constraint residual {cons:.3e}"


@pytest.mark.parametrize(
    "cls", [HMatrixSolver, ArrayBlockSolver], ids=lambda c: c.__name__
)
def test_iterative_columns_solve_their_own_port(cls):
    """Iterative families: the columns come out of a block GMRES, so they solve
    the system to `solve_tol` rather than to machine precision — checked against
    the operator the solver itself built, applied by its own matvec."""
    kw = _tight(cls, plain_multifeed())
    solver = cls(**kw)
    sol = solver.compute_port_solution()
    ctx = solver._context()
    geom, n = ctx["geom"], ctx["n_basis"]
    H = solver._build_operator()
    B = np.zeros((n, len(solver.feeds)), dtype=np.complex128)
    for j, (w_i, arc_i, _v) in enumerate(solver.feeds):
        arc_at_knot = geom["per_wire"][w_i]["arc_at_knot"]
        B[:, j] = solver._build_source_vector(
            geom,
            ctx["wire_knots"],
            ctx["wire_basis_global"],
            n,
            wi=w_i,
            s_f=arc_i if arc_i is not None else arc_at_knot[-1] / 2.0,
        )
    res, _ = _constrained_residual(H.matvec, sol.coeffs, B, None)
    assert res < 10 * solver.solve_tol, f"port-column residual {res:.3e}"


# ---- Gate 3: column j is the unit drive of port j, feeds first --------------


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_columns_superpose_to_the_configured_excitation(case):
    """`coeffs @ V` — the columns weighted by the configured port voltages in
    [feeds…, junction ports…] order — reproduces the coefficient vector
    `compute_impedance` solves for directly. That is simultaneously the
    superposition property the API is FOR (any later excitation costs a matmul,
    not a fill) and the tightest available statement of the port ORDER: swap
    two ports and this fails."""
    cls, fixture = case
    kw = _tight(cls, fixture())
    sol = cls(**kw).compute_port_solution()
    _z, coeffs = cls(**kw).compute_impedance()
    recon = sol.coeffs @ _port_voltages(kw)
    assert recon.shape == coeffs.shape
    rel = np.abs(recon - coeffs).max() / np.abs(coeffs).max()
    assert rel < 1e-8, f"superposition mismatch {rel:.3e}"


def test_ports_are_feeds_then_junction_ports():
    """The mixed fixture's Y is (1 gap feed + 2 junction ports)², feeds first.
    Pinned against the driving-point impedances `compute_impedance` reports in
    its own documented order: Z_port = V / (Y·V) entry by entry."""
    kw = junction_ports()
    sol = BSplineSolver(**kw).compute_port_solution()
    z_ref = np.atleast_1d(BSplineSolver(**kw).compute_impedance()[0])
    assert sol.y.shape == (3, 3)
    v = _port_voltages(kw)
    z = v / (sol.y @ v)
    assert np.abs(z - z_ref).max() / np.abs(z_ref).max() < 1e-10


# ---- Gate 4: one fill, one factorisation, whatever the port count ----------


def _count(monkeypatch, obj, name):
    calls = []
    original = getattr(obj, name)

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(obj, name, counting)
    return calls


FILL_AND_SOLVE = {
    BSplineSolver: ("_compute_Z_operator", "_solve_with_kcl_ports"),
    HMatrixSolver: ("_build_operator", "_solve_hmatrix"),
    ArrayBlockSolver: ("_build_operator", "_solve_hmatrix"),
    SinusoidalSolver: ("_assemble_Z", "compute_port_solution"),
    SinusoidalGalerkinSolver: ("_assemble_Z_ported", "compute_port_solution"),
}


@pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
@pytest.mark.parametrize(
    "fixture", [single_feed, plain_multifeed], ids=["1port", "2port"]
)
def test_one_fill_and_one_factorisation_per_call(monkeypatch, cls, fixture, request):
    """The contract SimNEC's daemon is built on: N ports cost N
    back-substitutions, not N fills. Counted on the family's own fill and solve
    entry points, at one port and at two, so a per-port refill would show up as
    a count that tracks the port count."""
    fill_name, solve_name = FILL_AND_SOLVE[cls]
    solver = cls(**_tight(cls, fixture()))
    fills = _count(monkeypatch, solver, fill_name)
    # The segment families factor through scipy directly, so the solve counter
    # goes on the module-level entry point they call.
    module = type(solver).__module__
    solves = (
        _count(monkeypatch, __import__(module, fromlist=["x"]).scipy.linalg, "solve")
        if cls in SEGMENT
        else _count(monkeypatch, solver, solve_name)
    )
    solver.compute_port_solution()
    assert len(fills) == 1, f"{fill_name} ran {len(fills)}×"
    assert len(solves) == 1, f"solve ran {len(solves)}×"


# ---- The refusals ----------------------------------------------------------


def test_galerkin_refuses_junction_ports_over_finite_ground_identically():
    """Where a family will not serve a configuration, the new entry point
    refuses at the same moment, with the same type and the same message — a
    consumer that switched to `compute_port_solution` sees no change in what
    comes back."""
    kw = junction_ports()
    kw["wires"] = [w + np.array([0.0, 0.0, 0.6]) for w in kw["wires"]]
    kw["ground_z"], kw["ground_eps"] = 0.0, 13.0
    with pytest.raises(NotImplementedError) as old:
        SinusoidalGalerkinSolver(**kw).compute_y_matrix()
    with pytest.raises(NotImplementedError) as new:
        SinusoidalGalerkinSolver(**kw).compute_port_solution()
    assert str(new.value) == str(old.value)


def test_point_matched_solver_still_refuses_junction_ports_at_construction():
    """#177's span rule bites before either entry point exists, so there is no
    `compute_port_solution` behaviour to define — pinned so a future port
    solution on this family can't quietly acquire one."""
    with pytest.raises(NotImplementedError, match="junction_ports are not supported"):
        SinusoidalSolver(**junction_ports())


def test_basis_handle_is_opaque_but_present():
    """`basis` is a handle, not an interface: the contract is only that it is
    there, that it is the same object for every port of ONE solution, and that
    a fresh solve makes a fresh one."""
    kw = plain_multifeed()
    solver = BSplineSolver(**kw)
    first = solver.compute_port_solution()
    second = solver.compute_port_solution()
    assert first.basis is not None
    assert first.basis is not second.basis
    assert type(first.basis) is type(second.basis)


def test_port_solution_is_exported_and_frozen():
    assert momwire.PortSolution is PortSolution
    assert "PortSolution" in momwire.__all__
    sol = BSplineSolver(**single_feed()).compute_port_solution()
    with pytest.raises(Exception):
        sol.y = None
