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
   Run twice — once on an `LD 4` ladder of this file's own, and once on the
   portal fixture `dipole_load_ld0`'s geometry, which is where momwire#588's
   measured bar against nec2c comes from.
3. **Refusals.** A deck needing what razor refuses reaches razor's own
   refusal message through `build_solver`, not a bare `KeyError` /
   `TypeError` — contact over a finite ground, and a `node_gaps` deck (built
   on the model directly, since the `nec2` dialect never emits one).
4. **Existing roster untouched.** No test here touches another basis's own
   construction; `test_every_basis_builds_the_same_model` in
   `test_deck_build_solver.py` is extended (not rewritten) to count the new
   family.
5. **Port spaces** (momwire#588). Razor is the one family whose port count is
   not the deck's, so `build_solver` renumbers the plan onto the rows it
   actually built. The gate is that the two plans and the bridge between them
   agree with the matrix; the portal's end of it is in `test_portal.py`.
"""

from __future__ import annotations

import pathlib

import numpy as np

import pytest

from momwire.deck import BASES, build_solver, parse
from momwire.deck.model import DeckModel, DeckWire
from momwire.razor import RazorSolver

C_LIGHT = 299_792_458.0

LOADED_DIPOLE = """CE dipole with series RLC load
GW 1 9 0. 0. -2.5 0. 0. 2.5 0.001
GE 0
LD 0 1 3 3 50. 1.e-6 0.
EX 0 1 5 0 1.
FR 0 1 0 0 30. 0
XQ
NX
"""

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


def test_razor_2p_is_the_current_spelling_and_razor_nec5_an_alias():
    """`razor-2p` names the RULE; `razor-nec5` is the deprecated spelling.

    They must stay indistinguishable — same class, same kwargs — because the
    old name shipped in the named entry points and in antennaknobs' CLI
    roster, and an install that still says `razor-nec5` must keep getting
    exactly what it got before.

    The rename is about the claim the name makes, not about any host
    mis-reading it: EZNEC takes engine class and path as explicit fields and
    infers nothing from the name, and SimNEC sniffs nec2c before nec5 so
    `momwire-nec2c-razor-nec5` always classified correctly.
    """
    two_p, kw_2p = BASES["razor-2p"]
    alias, kw_alias = BASES["razor-nec5"]
    assert two_p is alias is RazorSolver
    assert dict(kw_2p) == dict(kw_alias) == {"nec5_quadrature": True}
    # And both remain distinct from the Gauss-Legendre lane.
    assert dict(BASES["razor"][1]) == {}


def test_every_razor_lane_has_an_entry_point_and_a_banner_suffix():
    """A roster name a filename-only host cannot select is not really served.

    `_BANNER_SUFFIXES` already fails at import if a roster name is missing
    (which is how `razor-2p` announced itself when it was added), so this
    covers the other half: the console script, and the client's own copy of
    the roster that `test_the_clients_basis_roster_is_the_engines_own` binds.
    """
    import tomllib

    from momwire.portal._portal import _BANNER_SUFFIXES, basis_from_program_name
    from momwire_nec2c_client import BASIS_NAMES

    root = pathlib.Path(__file__).resolve().parent.parent
    scripts = tomllib.loads((root / "pyproject.toml").read_text())["project"]["scripts"]
    for name in ("razor", "razor-2p", "razor-nec5"):
        assert name in _BANNER_SUFFIXES, name
        assert name in BASIS_NAMES, name
        script = f"momwire-nec2c-{name}"
        assert script in scripts, f"{script} missing from [project.scripts]"
        assert basis_from_program_name(script, "nec2c-") == name
    # The suffixes must be distinct, or a printout cannot say which ran.
    subset = {n: _BANNER_SUFFIXES[n] for n in ("razor", "razor-2p", "razor-nec5")}
    assert len(set(subset.values())) == 3, subset


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
GE 1
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


def _portal_fixture_deck(nsegs: int) -> str:
    """`tests/fixtures/nec_portal/dipole_load_ld0.deck` at N segments.

    Same wire, same frequency, same load, and the load pinned to the PHYSICAL
    position segment 3 of 9 put it at rather than to a segment number, so the
    ladder refines one antenna instead of walking a load along it.
    """
    load_arc = (3 - 0.5) * 5.0 / 9.0
    load_seg = max(1, min(nsegs, round(load_arc * nsegs / 5.0 + 0.5)))
    feed_seg = (nsegs + 1) // 2
    if load_seg == feed_seg:
        load_seg = max(1, load_seg - 1)
    return f"""CE dipole with series RLC load N={nsegs}
GW 1 {nsegs} 0. 0. -2.5 0. 0. 2.5 0.001
GE 0
LD 0 1 {load_seg} {load_seg} 50. 1.e-6 0.
EX 0 1 {feed_seg} 0 1.
FR 0 1 0 0 30. 0
XQ
NX
"""


@pytest.mark.integration
@pytest.mark.slow
def test_the_two_routes_converge_on_the_fixtures_own_geometry():
    """The evidence behind momwire#588's measured bar.

    The gate above uses an `LD 4` of 40 - 120j; the portal fixture
    `dipole_load_ld0` uses an `LD 0` of 50 Ω + j188.5 at 30 MHz, which is a
    much harder load — comparable to the antenna's own reactance and a third
    of the way along it — and the fixture meshes it at NINE segments. There
    the two routes are 57 Ω apart, and razor reads 22.9 % from the committed
    nec2c capture where bspline reads 1.79 %.

    This is what says that spread is the mesh. Refine the same antenna and
    the two routes converge on EACH OTHER, and the number they converge to
    (~160 + 200j) is about 8 % from the oracle's own nine-segment answer of
    144.06 + 188.89j. So at N = 9 all three are in different places and the
    two that share a refinement path agree; nothing here is a defect in the
    load-only service, which `test_portal.py` gates exactly instead.
    """
    ns = (9, 19, 41, 81, 161, 321)
    gaps = []
    for nsegs in ns:
        model = parse(_portal_fixture_deck(nsegs))
        z_razor, _ = build_solver(model, basis="razor").solver.compute_impedance()
        z_bspline = _stamp(build_solver(model, basis="bspline"))
        gaps.append(abs(z_razor - z_bspline))

    # Measured 2026-08-24: [56.94, 19.46, 14.04, 5.16, 2.44, 1.16] — a ~49x
    # fall, monotone from N=19 on, and the far end is 0.7 % of |Z|.
    assert gaps[0] > 20.0  # the fixture's own mesh IS the coarse end
    assert gaps[-1] < gaps[0] / 20.0
    assert gaps[-1] < 3.0  # measured 1.16 Ω


# ---------------------------------------------------------------------------
# gate 3: refusals reach razor's own message through build_solver
# ---------------------------------------------------------------------------


def test_contact_over_a_finite_ground_now_builds_through_the_deck_front_end():
    """momwire#624 inverted this test, and the deck route is worth its own.

    A `GN 2` deck reaches razor through `build_solver`, which is a different
    path from the constructor call `test_razor_ground_contact.py` exercises:
    the dialect has to translate the ground, discover the grounded end and
    hand razor a `ground_eps` — and while the refusal stood, none of that was
    ever reached on this basis. So the assert flips rather than the test
    being deleted: what used to raise now builds and solves.
    """
    text = """CM base fed over a finite ground
CE
GW 1 20 0. 0. 0. 0. 0. 5. 1.E-3
GE 1
GN 2 0 0 0 13. 0.005
EX 0 1 1 0 1. 0.
FR 0 1 0 0 14.1
XQ
NX
"""
    model = parse(text)
    built = build_solver(model, basis="razor")
    z, _ = built.solver.compute_impedance()
    z = complex(np.atleast_1d(z)[0])
    assert np.isfinite(z.real) and np.isfinite(z.imag)
    assert z.real > 0.0, f"a passive base-fed vertical came back as {z}"


def test_contact_under_refl_coef_still_refuses_through_the_deck_front_end():
    """D3's row, which momwire#624 did NOT lift, on the same deck.

    `GN 2` is the Sommerfeld spelling and builds above; the reflection-
    coefficient ground at zero clearance is refused on every trunk because
    the MODEL fails there, and razor now reaches that refusal through
    `_ground_spec`'s shared sentence rather than a razor-owned copy. Same
    geometry, same soil, one card different — which is what makes this pair
    a boundary rather than two unrelated assertions.
    """
    model = parse(
        """CM base fed over a finite ground
CE
GW 1 20 0. 0. 0. 0. 0. 5. 1.E-3
GE 1
GN 0 0 0 0 13. 0.005
EX 0 1 1 0 1. 0.
FR 0 1 0 0 14.1
XQ
NX
"""
    )
    with pytest.raises(NotImplementedError, match="momwire#282"):
        build_solver(model, basis="razor")


def test_a_node_gap_on_a_free_end_refuses_by_naming_the_geometry():
    """momwire#603 U4 turned this from a FAMILY refusal into a SITE one.

    Razor served no node gap at all when this test was written, so a lone
    wire's start refused with "node gaps are not supported". It serves them
    now, and this deck is still wrong — the named end is nobody's junction,
    so there is no through-current path for a series EMF to sit in. The
    refusal says that instead, which is the thing the caller can act on.
    """
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
    with pytest.raises(ValueError, match="not a member of any junction group"):
        build_solver(model, basis="razor", frequency_mhz=30.0)


@pytest.mark.parametrize("basis", ["razor", "razor-nec5"])
def test_extended_kernel_card_reaches_the_solver(basis):
    """An `EK` card arms the extended kernel on this class too (momwire#398 D1).

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
    except (NotImplementedError, ValueError):
        pass
    else:
        pytest.fail("expected a refusal")


def test_the_plan_a_built_solver_hands_back_is_in_the_solvers_own_port_space():
    """momwire#588's whole mechanism, in one deck.

    `PortPlan` says indices into `sites` ARE solver port indices, and
    `prepare_mesh` cannot make that true: it builds one plan per structure,
    before a basis has been chosen, and a `_NATIVE_LOADING` basis gives a
    load-only site no row of its own. So the same integer meant two things,
    which is momwire#439's `IndexError` and #588's dimension mismatch.

    `build_solver` now closes the gap where it is visible — it knows both the
    plan and the matrix it just built — and hands back three things: the plan
    in the solver's space, the plan in the deck's, and the bridge between
    them, built from the same flags the `feeds` list was filtered by so the
    two cannot drift.
    """
    model = parse(LOADED_DIPOLE)

    # Every other family: one port per site, in order, and the two plans are
    # the SAME OBJECT rather than an equal copy — nothing was renumbered.
    plain = build_solver(model, basis="bspline")
    assert plain.ports is plain.deck_ports
    assert plain.site_to_solver_port == tuple(range(len(plain.ports.sites)))
    assert len(plain.solver.feeds) == len(plain.ports.sites)

    # Razor: the load-only site has no row of its own, so the two plans part.
    razor = build_solver(model, basis="razor")
    assert razor.site_to_solver_port == (None, 0)
    assert razor.deck_ports.n_ports == 2
    assert razor.ports.n_ports == 1 == len(razor.solver.feeds)
    # The solver-space plan addresses the matrix: one site, the fed one, and
    # a feed index that is a row of `y` rather than a row of the deck.
    (site,) = razor.ports.sites
    assert site.feed is not None and site.load is None
    assert razor.ports.feed_ports == (0,)
    assert razor.ports.n_ports == razor.solver.compute_port_solution().y.shape[0]
    # The load is not lost — it is on the fill, which is why it has no port,
    # and `load_ports` says so in the one way an int cannot.
    assert razor.ports.load_ports == (None,)
    assert razor.ports.loaded_ports() == ()
    assert razor.solver.lumped_loads
    # The deck's own plan is untouched, and still counts the site the cards
    # cut: that is what a message about the DECK has to say.
    assert razor.deck_ports.load_ports == (0,)
    assert [s.load for s in razor.deck_ports.sites].count(None) == 1
