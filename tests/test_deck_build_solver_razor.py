"""``momwire.deck.build_solver`` learns ``RazorSolver`` (momwire#432).

PR #431 gave `RazorSolver` its own loading kwargs — `wire_conductivity` /
`insulation_radius` / `insulation_eps_r` verbatim from the siblings, and
`lumped_loads=[(wire, arclength, Z)]` as its own spelling of a lumped load,
because this formulation refuses the siblings' deck-level port algebra over
a zero-volt gap. This module is the follow-up that PR filed: `build_solver`
learns two roster entries (`"razor"`, and `"razor-nec5"` for the identified
quadrature, momwire#316) and the LD-card translation into razor's native
kwargs.

Gates, in the order the issue asked for:

1. **Deck-built == direct-built, exact.** A battery of eight decks — free
   dipole, `LD 4` mid-element, `LD 5` copper, a `GN 1` base-fed contact
   monopole, a `GN 1` elevated inverted-V, both finite grounds, a
   mixed-radius two-wire deck — each checked against an INDEPENDENTLY
   constructed `RazorSolver` (the geometry and kwargs re-derived by hand in
   this file, not by calling `_lumped_loads` or any other translation
   helper) to LU roundoff.
2. **Route equivalence.** On a loaded dipole the port-algebra route serves
   (`bspline`, stamped by this file's own `_stamp` — the same
   `V_gap = (1 + Z·Y)^-1 V_source` the portal and PR #431's Thevenin gate
   use), razor's translated answer converges to it with N. Measured, not
   asserted at a guessed rate: pinned at the measured level with margin.
3. **Refusals.** A deck needing what razor refuses reaches razor's own
   refusal message through `build_solver`, not a bare `KeyError` /
   `TypeError` — contact over a finite ground, and a `node_gaps` deck (built
   on the model directly, since the `nec2` dialect never emits one).
4. **Existing roster untouched.** No test here touches another basis's own
   construction; `test_every_basis_builds_the_same_model` in
   `test_deck_build_solver.py` is extended (not rewritten) to count the new
   family.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire.deck import BASES, build_solver, parse
from momwire.deck.model import DeckModel, DeckWire
from momwire.razor import RazorSolver

C_LIGHT = 299_792_458.0

DIPOLE = """CM a dipole
CE
GW 1 9 -2.5 0. 0. 2.5 0. 0. 1.E-3
GE 0
EX 0 1 5 0 1. 0.
FR 0 1 0 0 30.
XQ
NX
"""


def _wl(freq_mhz: float) -> float:
    return C_LIGHT / (freq_mhz * 1e6)


def _assert_close(a: complex, b: complex, rel: float = 1e-12) -> None:
    assert abs(a - b) <= rel * abs(b), f"{a!r} vs {b!r}: {abs(a - b) / abs(b):.3g} rel"


# ---------------------------------------------------------------------------
# the roster entries
# ---------------------------------------------------------------------------


def test_razor_and_razor_nec5_are_in_the_roster():
    solver_class, kwargs = BASES["razor"]
    assert solver_class is RazorSolver
    assert dict(kwargs) == {}
    solver_class, kwargs = BASES["razor-nec5"]
    assert solver_class is RazorSolver
    assert dict(kwargs) == {"nec5_quadrature": True}


def test_razor_nec5_binds_the_identified_quadrature():
    model = parse(DIPOLE)
    plain = build_solver(model, basis="razor")
    n5q = build_solver(model, basis="razor-nec5")
    assert plain.solver.nec5_quadrature is False
    assert n5q.solver.nec5_quadrature is True
    # Two different quadrature rules on the same deck; not required to
    # agree, only both required to solve.
    z_plain, _ = plain.solver.compute_impedance()
    z_n5q, _ = n5q.solver.compute_impedance()
    assert z_plain != z_n5q


# ---------------------------------------------------------------------------
# gate 1: deck-built == direct-built, exact
# ---------------------------------------------------------------------------


def test_a_free_dipole():
    model = parse(DIPOLE)
    built = build_solver(model, basis="razor")
    direct = RazorSolver(
        wires=[np.array([[-2.5, 0.0, 0.0], [2.5, 0.0, 0.0]])],
        nsegs=9,
        wire_radius=1.0e-3,
        wavelength=_wl(30.0),
        feeds=[(0, 2.5, 1 + 0j)],
    )
    za, _ = built.solver.compute_impedance()
    zb, _ = direct.compute_impedance()
    _assert_close(za, zb)


def test_ld4_mid_element():
    text = DIPOLE.replace("EX 0 1 5", "LD 4 1 3 3 50. 10.\nEX 0 1 5")
    model = parse(text)
    built = build_solver(model, basis="razor")
    # Segment 3 of 9's centre, NEC's own rule: (k - 1/2) L / NS.
    load_arc = (3 - 0.5) * 5.0 / 9
    direct = RazorSolver(
        wires=[np.array([[-2.5, 0.0, 0.0], [2.5, 0.0, 0.0]])],
        nsegs=9,
        wire_radius=1.0e-3,
        wavelength=_wl(30.0),
        feeds=[(0, 2.5, 1 + 0j)],
        lumped_loads=[(0, load_arc, 50 + 10j)],
    )
    za, _ = built.solver.compute_impedance()
    zb, _ = direct.compute_impedance()
    _assert_close(za, zb)
    # And the load moved the answer — the gate is not vacuously true.
    unloaded, _ = build_solver(parse(DIPOLE), basis="razor").solver.compute_impedance()
    assert abs(za - unloaded) > 1.0


def test_ld5_copper():
    text = DIPOLE.replace("EX 0 1 5", "LD 5 1 0 0 5.8E7\nEX 0 1 5")
    model = parse(text)
    built = build_solver(model, basis="razor")
    direct = RazorSolver(
        wires=[np.array([[-2.5, 0.0, 0.0], [2.5, 0.0, 0.0]])],
        nsegs=9,
        wire_radius=1.0e-3,
        wavelength=_wl(30.0),
        feeds=[(0, 2.5, 1 + 0j)],
        wire_conductivity=np.array([5.8e7]),
    )
    za, _ = built.solver.compute_impedance()
    zb, _ = direct.compute_impedance()
    _assert_close(za, zb)


def test_gn1_monopole_base_fed_contact():
    text = """CM base fed monopole over PEC
CE
GW 1 20 0. 0. 0. 0. 0. 5. 1.E-3
GE -1
GN 1
EX 0 1 1 0 1. 0.
FR 0 1 0 0 14.1
XQ
NX
"""
    model = parse(text)
    assert model.ground == "pec"
    built = build_solver(model, basis="razor")
    feed_arc = (1 - 0.5) * 5.0 / 20
    direct = RazorSolver(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])],
        nsegs=20,
        wire_radius=1.0e-3,
        wavelength=_wl(14.1),
        ground_z=0.0,
        feeds=[(0, feed_arc, 1 + 0j)],
    )
    za, _ = built.solver.compute_impedance()
    zb, _ = direct.compute_impedance()
    _assert_close(za, zb)


def test_gn1_elevated_invvee():
    text = """CM elevated inverted-v
CE
GW 1 10 -2.0 0. 6.0 0. 0. 8.0 1.E-3
GW 2 10 0. 0. 8.0 2.0 0. 6.0 1.E-3
GE 0
GN 1
EX 0 2 1 0 1. 0.
FR 0 1 0 0 20.
XQ
NX
"""
    model = parse(text)
    built = build_solver(model, basis="razor")
    leg = float(np.hypot(2.0, 2.0))
    feed_arc = (1 - 0.5) * leg / 10
    direct = RazorSolver(
        wires=[
            np.array([[-2.0, 0.0, 6.0], [0.0, 0.0, 8.0]]),
            np.array([[0.0, 0.0, 8.0], [2.0, 0.0, 6.0]]),
        ],
        nsegs=10,
        wire_radius=1.0e-3,
        wavelength=_wl(20.0),
        ground_z=0.0,
        feeds=[(1, feed_arc, 1 + 0j)],
    )
    za, _ = built.solver.compute_impedance()
    zb, _ = direct.compute_impedance()
    _assert_close(za, zb)


_FINITE_GROUND_DIPOLE = """CM elevated dipole over a finite ground
CE
GW 1 9 -2.5 0. 5. 2.5 0. 5. 1.E-3
GE 0
GN {code} 0 0 0 13. 0.005
EX 0 1 5 0 1. 0.
FR 0 1 0 0 14.1
XQ
NX
"""


def test_finite_reflection_coefficient_ground():
    model = parse(_FINITE_GROUND_DIPOLE.format(code=0))
    assert model.ground == ("finite-fast", 13.0, 0.005)
    built = build_solver(model, basis="razor")
    direct = RazorSolver(
        wires=[np.array([[-2.5, 0.0, 5.0], [2.5, 0.0, 5.0]])],
        nsegs=9,
        wire_radius=1.0e-3,
        wavelength=_wl(14.1),
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        feeds=[(0, 2.5, 1 + 0j)],
    )
    za, _ = built.solver.compute_impedance()
    zb, _ = direct.compute_impedance()
    _assert_close(za, zb)


def test_sommerfeld_ground():
    model = parse(_FINITE_GROUND_DIPOLE.format(code=2))
    assert model.ground == ("finite", 13.0, 0.005)
    built = build_solver(model, basis="razor")
    direct = RazorSolver(
        wires=[np.array([[-2.5, 0.0, 5.0], [2.5, 0.0, 5.0]])],
        nsegs=9,
        wire_radius=1.0e-3,
        wavelength=_wl(14.1),
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
        feeds=[(0, 2.5, 1 + 0j)],
    )
    za, _ = built.solver.compute_impedance()
    zb, _ = direct.compute_impedance()
    _assert_close(za, zb)


def test_mixed_radius_two_wire():
    text = """CM mixed radius dipole
CE
GW 1 10 0. 0. 0. 2.5 0. 0. 5.E-3
GW 2 10 0. 0. 0. -2.5 0. 0. 5.E-4
GE 0
EX 0 1 1 0 1. 0.
FR 0 1 0 0 30.
XQ
NX
"""
    model = parse(text)
    built = build_solver(model, basis="razor")
    feed_arc = (1 - 0.5) * 2.5 / 10
    direct = RazorSolver(
        wires=[
            np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [-2.5, 0.0, 0.0]]),
        ],
        nsegs=10,
        wire_radius=[5.0e-3, 5.0e-4],
        wavelength=_wl(30.0),
        feeds=[(0, feed_arc, 1 + 0j)],
    )
    za, _ = built.solver.compute_impedance()
    zb, _ = direct.compute_impedance()
    _assert_close(za, zb)


# ---------------------------------------------------------------------------
# gate 2: route equivalence — converges to the port-algebra (bspline) route
# ---------------------------------------------------------------------------


def _seg_at(nsegs: int, length: float, arclength: float) -> int:
    """1-based NEC segment whose centre is nearest ``arclength``."""
    seg = round(arclength * nsegs / length + 0.5)
    return max(1, min(nsegs, seg))


def _stamp(built) -> complex:
    """The port-algebra route's own drive-point impedance: the Thevenin
    reduction PR #431's gate and the portal's `DeckSolver.solve_group` both
    use, written independently here rather than imported from either."""
    plan = built.ports
    y = built.solver.compute_port_solution().y
    n = plan.n_ports
    freq_hz = built.frequency_mhz * 1e6
    z_load = np.zeros(n, dtype=np.complex128)
    for port, spec in plan.loaded_ports():
        z_load[port] = spec.impedance(freq_hz)
    v_source = np.zeros(n, dtype=np.complex128)
    (feed_port,) = plan.feed_ports
    v_source[feed_port] = 1 + 0j
    system = np.eye(n, dtype=np.complex128) + z_load[:, None] * y
    v_gap = np.linalg.solve(system, v_source)
    i_port = y @ v_gap
    return v_source[feed_port] / i_port[feed_port]


_LOAD_LEN = 5.0
_LOAD_Z = 40.0 - 120.0j
_LOAD_ARC = 1.0  # a fixed physical position, off the driven port


def _loaded_dipole_deck(nsegs: int) -> str:
    load_seg = _seg_at(nsegs, _LOAD_LEN, _LOAD_ARC)
    feed_seg = _seg_at(nsegs, _LOAD_LEN, _LOAD_LEN / 2)
    if load_seg == feed_seg:
        load_seg = max(1, load_seg - 1)
    return f"""CM loaded dipole N={nsegs}
CE
GW 1 {nsegs} -2.5 0. 0. 2.5 0. 0. 1.E-3
GE 0
LD 4 1 {load_seg} {load_seg} {_LOAD_Z.real} {_LOAD_Z.imag}
EX 0 1 {feed_seg} 0 1. 0.
FR 0 1 0 0 28.8
XQ
NX
"""


@pytest.mark.slow
def test_the_translation_converges_to_the_port_algebra_route():
    """Measured, not derived: both routes carry their own O(1/N) walk (razor
    is razor-blade tested, bspline is Galerkin), so the gap is not expected
    to fall monotonically step to step (momwire#309's own "pair-walk" is the
    same behaviour on the other cross-formulation gates in this repo) — what
    is gated is that it is clearly SMALLER at the ladder's far end than at
    its start, and the far end is pinned at its measured level with a
    healthy margin."""
    ns = (21, 41, 81, 161, 321)
    gaps = []
    for nsegs in ns:
        model = parse(_loaded_dipole_deck(nsegs))
        z_razor, _ = build_solver(model, basis="razor").solver.compute_impedance()
        z_bspline = _stamp(build_solver(model, basis="bspline"))
        gaps.append(abs(z_razor - z_bspline))

    # Measured: [3.35, 2.16, 0.72, 1.86, 0.30] — a clear ladder-scale drop
    # (~11x) from the start to the far end, despite the mid-ladder blip both
    # solvers' own independent O(1/N) walks are expected to produce.
    assert gaps[-1] < min(gaps[0], gaps[1]) / 3
    assert gaps[-1] < 1.0  # measured 0.30 Ω


# ---------------------------------------------------------------------------
# gate 3: refusals reach razor's own message through build_solver
# ---------------------------------------------------------------------------


def test_contact_over_finite_ground_refuses():
    text = """CM base fed over a finite ground
CE
GW 1 20 0. 0. 0. 0. 0. 5. 1.E-3
GE -1
GN 2 0 0 0 13. 0.005
EX 0 1 1 0 1. 0.
FR 0 1 0 0 14.1
XQ
NX
"""
    model = parse(text)
    with pytest.raises(
        NotImplementedError, match="ground CONTACT over a finite ground"
    ):
        build_solver(model, basis="razor")


def test_a_node_gaps_deck_refuses():
    # The nec2 dialect never emits `node_gaps`; build the model directly, as
    # a NEC-5 dialect's edge-source translation would.
    model = DeckModel(
        wires=(
            DeckWire(
                vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                radius=1e-3,
                edge_elements=(4,),
            ),
        ),
        node_gaps=((0, 0, 1 + 0j),),
    )
    with pytest.raises(NotImplementedError, match="node_gaps"):
        build_solver(model, basis="razor", frequency_mhz=30.0)


@pytest.mark.parametrize("basis", ["razor", "razor-nec5"])
def test_extended_kernel_card_reaches_the_solver(basis):
    """An `EK` card arms the extended kernel on this class too (momwire#436).

    It used to be refused. The taper study identified the NEC-5 binary as
    extended-kernel EVERYWHERE, which made the refusal a statement about the
    reference that was false, so razor takes the house kwarg now and
    `build_solver` needed no razor-specific line to deliver it — the same
    `kwargs["extended_kernel"] = True` that serves every sibling.
    """
    model = parse(DIPOLE.replace("EX 0 1 5", "EK\nEX 0 1 5"))
    built = build_solver(model, basis=basis)
    assert built.extended_kernel is True
    assert built.solver.extended_kernel is True
    # and it is not merely stored: the kernel moves the answer.
    reduced = build_solver(model, basis=basis, extended_kernel=False)
    assert reduced.solver.extended_kernel is False
    z_ek, _ = built.solver.compute_impedance()
    z_red, _ = reduced.solver.compute_impedance()
    assert z_ek != z_red


def test_a_deck_without_ek_stays_reduced():
    model = parse(DIPOLE)
    built = build_solver(model, basis="razor")
    assert built.extended_kernel is False
    assert built.solver.extended_kernel is False


def test_the_refusals_are_not_bare_key_or_type_errors():
    """The issue's own wording: a `KeyError`/`TypeError` here would mean the
    translation dropped the card silently instead of naming razor's reason."""
    model = DeckModel(
        wires=(
            DeckWire(
                vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                radius=1e-3,
                edge_elements=(4,),
            ),
        ),
        node_gaps=((0, 0, 1 + 0j),),
    )
    try:
        build_solver(model, basis="razor", frequency_mhz=30.0)
    except NotImplementedError:
        pass
    else:
        pytest.fail("expected a refusal")
