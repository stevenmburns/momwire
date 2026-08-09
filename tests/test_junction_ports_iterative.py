"""Junction ports on the iterative solvers (#234).

#172 gave the dense `BSplineSolver` voltage-drivable junction ports: the
ported junction's KCL closure row leaves the constraint set, multiplies its
voltage into the RHS, and doubles as the port's current readout. The
accelerated paths threaded the whole `kcl_A` into the augmented saddle system
and refused `junction_ports` outright. They now take the same split — the
partition happens *before* the solve, so the augmented GMRES is handed a
shrunken constraint matrix and an already-driven RHS and needs no port notion
of its own.

The end-attachment fixtures here mirror `test_junction_ports.py`'s split
dipole with tip whiskers (antennaknobs#579's geometry) rather than importing
any design: two half-dipoles, each ending in a short whisker whose shared node
is the port.
"""

import numpy as np
import pytest

from momwire import ArrayBlockSolver, BSplineSolver, HMatrixSolver

WAVELENGTH = 10.0
L_DIP = 0.48 * WAVELENGTH

ITERATIVE = [HMatrixSolver, ArrayBlockSolver]

# The operator's own approximations (ACA truncation) sit above the constraint
# algebra under test, so the end-port gates tighten both knobs and compare at
# the iterative tolerance instead of the default 1e-4 ACA floor.
TIGHT = dict(aca_tol=1e-9, solve_tol=1e-10)


def _end_port_deck(n=40, n_copies=3, dx=3.0, delta=0.04, whisker=0.01, **extra):
    """`n_copies` split dipoles, each driven through the pair of junction
    ports at its inner tips (±0.5 V) — the paired-drive oracle of #172, laid
    out along x so the accelerators have far blocks to compress. One 0 V gap
    feed keeps a mixed feed/port readout in play."""
    wires, npe, junctions, ports = [], [], [], []
    for c in range(n_copies):
        x = c * dx
        b = len(wires)
        wires += [
            np.array([(x, -L_DIP / 2, 0.0), (x, -delta / 2, 0.0)]),
            np.array([(x, -delta / 2, 0.0), (x + whisker, -delta / 2, 0.0)]),
            np.array([(x, delta / 2, 0.0), (x, L_DIP / 2, 0.0)]),
            np.array([(x, delta / 2, 0.0), (x + whisker, delta / 2, 0.0)]),
        ]
        npe += [[n], [1], [n], [1]]
        junctions += [
            [(b, "end"), (b + 1, "start")],
            [(b + 2, "start"), (b + 3, "start")],
        ]
        ports += [(len(junctions) - 2, 0.5 + 0j), (len(junctions) - 1, -0.5 + 0j)]
    return dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        junctions=junctions,
        wavelength=WAVELENGTH,
        feeds=[(0, L_DIP / 4, 0.0 + 0j)],
        junction_ports=ports,
        **extra,
    )


def _lattice_end_fed_deck(px=3, py=2, s=15.4, nseg=8, whisker=0.05, **extra):
    """`px`×`py` lattice of identical L-shaped elements, each gap-fed at the
    base and carrying a junction port at the far END (a whisker node). Per
    element: one elbow KCL row that stays a constraint, one tip row that
    becomes a port — the mixed partition the Floquet preconditioner has to
    survive."""
    h = 0.962 * 22.0 / 4
    wires, npe, junctions, feeds, ports = [], [], [], [], []
    for i in range(px):
        for j in range(py):
            dx, dy = i * s, j * s
            b = len(wires)
            wires += [
                np.array([[dx, dy, 0.0], [dx, dy, h]]),
                np.array([[dx, dy, h], [dx, dy + h, h]]),
                np.array([[dx, dy + h, h], [dx + whisker, dy + h, h]]),
            ]
            npe += [[nseg], [nseg], [1]]
            junctions += [
                [(b, "end"), (b + 1, "start")],  # elbow: stays a constraint
                [(b + 1, "end"), (b + 2, "start")],  # end attachment: port
            ]
            feeds.append((b, None, 1.0 + 0j))
            ports.append((len(junctions) - 1, 1.0 + 0j))
    return dict(
        wires=wires,
        degree=2,
        n_per_edge_per_wire=npe,
        wavelength=22.0,
        junctions=junctions,
        feeds=feeds,
        junction_ports=ports,
        **extra,
    )


def _rel(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(b).max())


# ---- Gate 1: the end-port fixture matches dense on both accelerators --------


@pytest.mark.parametrize("cls", ITERATIVE, ids=lambda c: c.__name__)
def test_end_ports_match_dense_on_the_iterative_solvers(cls):
    """Y and the per-port impedance readout reproduce the dense answer, and
    the GMRES really ran (a zero-iteration solve would only prove the
    preconditioner is exact at this size, not that the ports rode the
    Krylov operator)."""
    ref = _end_port_deck()
    y_ref = BSplineSolver(**ref).compute_y_matrix()
    z_ref = np.atleast_1d(BSplineSolver(**ref).compute_impedance()[0])
    assert y_ref.shape == (7, 7)  # 1 gap feed + 6 junction ports

    s_y = cls(**_end_port_deck(**TIGHT))
    y = s_y.compute_y_matrix()
    s_z = cls(**_end_port_deck(**TIGHT))
    z = np.atleast_1d(s_z.compute_impedance()[0])

    assert s_y._last_solve_iters[0] >= 1
    assert s_z._last_solve_iters[0] >= 1
    assert y.shape == y_ref.shape and z.shape == z_ref.shape
    assert _rel(y, y_ref) < 1e-7
    # The 0 V gap feed's z entry is 0/I on both sides; compare the ports.
    assert _rel(z[1:], z_ref[1:]) < 1e-7
    # Reciprocity survives the split on the iterative path too.
    assert _rel(y, y.T) < 1e-9


@pytest.mark.parametrize("cls", ITERATIVE, ids=lambda c: c.__name__)
def test_end_port_sweep_matches_dense(cls):
    """`compute_impedance_swept` sized its output from the gap-feed count
    alone; junction ports widen the readout the same way they do on the
    dense sweep."""
    ref = _end_port_deck(n=16, n_copies=2)
    k0 = BSplineSolver(**ref).k
    ks = np.array([0.95 * k0, k0])
    z_ref = BSplineSolver(**ref).compute_impedance_swept(ks)
    z = cls(**_end_port_deck(n=16, n_copies=2, **TIGHT)).compute_impedance_swept(ks)
    assert z.shape == z_ref.shape == (2, 5)
    assert _rel(z[:, 1:], z_ref[:, 1:]) < 1e-6


# ---- Gate 2: end-fed elements on a lattice keep the FFT path ---------------


def test_end_fed_lattice_holds_the_fft_path():
    """`require_lattice_fft=True` on a lattice of end-fed elements: junction
    bases are wire-local, so the port rows leave the constraint set inside
    their own element and the block-Toeplitz coupling representation — which
    never sees `kcl_A` at all — is untouched. The per-element constraint
    blocks stay identical across elements, so the Floquet preconditioner
    still engages rather than degrading to block-Jacobi."""
    from momwire.array_block import LatticeArrayBlock, _LatticeFloquetAugPrecond

    ref = _lattice_end_fed_deck()
    y_ref = BSplineSolver(**ref).compute_y_matrix()
    assert y_ref.shape == (12, 12)  # 6 gap feeds + 6 end ports

    s = ArrayBlockSolver(
        lattice_fft=True, require_lattice_fft=True, **_lattice_end_fed_deck()
    )
    y = s.compute_y_matrix()
    assert isinstance(s._hmatrix, LatticeArrayBlock)
    assert isinstance(s._hmatrix._factored.precond, _LatticeFloquetAugPrecond)
    # One elbow row per element survives as a constraint; the six tip rows left.
    assert s._hmatrix._factored.nc == 6
    assert _rel(y, y_ref) < 1e-7


# ---- Gate 3: the #176 enrichment combination is still legal ----------------


@pytest.mark.parametrize("cls", ITERATIVE, ids=lambda c: c.__name__)
@pytest.mark.parametrize("enrich", [False, True])
def test_constructor_accepts_junction_ports(cls, enrich):
    """The refusal is deleted, not rerouted: both the real iterative path
    (enrichment off) and the #176 dense-fallback path (enrichment on)
    construct and solve."""
    kw = _end_port_deck(n=8, n_copies=1, use_singular_enrichment=enrich)
    ref = BSplineSolver(**kw).compute_y_matrix()
    y = cls(**kw).compute_y_matrix()
    assert y.shape == (3, 3)
    assert _rel(y, ref) < 1e-6


# ---- Gate 4 (adversarial): with the split disabled, nothing matches --------


def _disable_port_split(solver):
    """Restore the pre-#234 contract on one instance: `kcl_A` goes into the
    augmented system whole, so the ported junctions' rows are still enforced
    to zero outflow. Drive and readout vectors are left intact, which isolates
    the constraint/port partition as the thing under test."""
    real = solver._split_kcl_ports
    solver._split_kcl_ports = lambda kcl_A: (kcl_A,) + real(kcl_A)[1:]
    return solver


@pytest.mark.parametrize("cls", ITERATIVE, ids=lambda c: c.__name__)
def test_disabled_port_partition_does_not_match_dense(cls):
    """Pins that the port rows genuinely flow through the iterative solve:
    leave them in the constraint set and every port current is driven to
    zero, so the port block of Y collapses."""
    kw = _end_port_deck(n=16, n_copies=1, **TIGHT)
    y_ref = BSplineSolver(**_end_port_deck(n=16, n_copies=1)).compute_y_matrix()
    y_bad = _disable_port_split(cls(**kw)).compute_y_matrix()
    assert y_bad.shape == y_ref.shape
    assert _rel(y_bad, y_ref) > 0.9
    assert np.abs(y_bad[1:, 1:]).max() < 1e-9 * np.abs(y_ref[1:, 1:]).max()


# ---- The factorisation cache must key on the constraint rows ---------------


def test_operator_cache_reuse_across_differing_port_sets():
    """`ArrayBlockSolver`'s operator cache is keyed on geometry + k: junction
    ports change neither, so the *same* cached operator is handed two
    different constraint sets. The augmented factorisation cached on it has
    to notice, or the second solve silently answers the first one's system."""
    common = dict(
        wires=_end_port_deck(n=12, n_copies=2)["wires"],
        n_per_edge_per_wire=_end_port_deck(n=12, n_copies=2)["n_per_edge_per_wire"],
        junctions=_end_port_deck(n=12, n_copies=2)["junctions"],
        wavelength=WAVELENGTH,
        feeds=[(0, L_DIP / 4, 1.0 + 0j)],
    )
    ported = dict(common, junction_ports=[(0, 0.5 + 0j)], **TIGHT)
    plain = dict(common, **TIGHT)
    y_ported_ref = BSplineSolver(**dict(common, junction_ports=[(0, 0.5 + 0j)]))
    y_ported_ref = y_ported_ref.compute_y_matrix()
    y_plain_ref = BSplineSolver(**common).compute_y_matrix()

    y_ported = ArrayBlockSolver(**ported).compute_y_matrix()
    y_plain = ArrayBlockSolver(**plain).compute_y_matrix()  # cache hit, nc differs
    assert y_ported.shape == (2, 2) and y_plain.shape == (1, 1)
    assert _rel(y_ported, y_ported_ref) < 1e-7
    assert _rel(y_plain, y_plain_ref) < 1e-7
