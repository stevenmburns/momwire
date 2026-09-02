"""``momwire.deck.build_solver`` — the model on a solver.

The equivalence side of this unit lives in ``test_deck_nec2_corpus.py``, which
measures the whole chain against antennaknobs over the reference corpus.  This
module measures the pieces that corpus cannot reach: a closed loop, a tee, a
radius change mid-chain, every basis in the roster, and the port plan's own
shape.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from momwire.deck import (
    BASES,
    NEC2_BASES,
    DeckModel,
    Environment,
    LoadSpec,
    build_solver,
    parse,
    prepare_mesh,
)
from momwire.deck._polylines import to_polylines
from momwire.deck._solver import _ground, _sites, _wire_loading
from momwire.deck.model import DeckWire, ExecuteGroup, WireMaterial

DIPOLE = """CM a dipole
CE
GW 1 9 -2.5 0. 0. 2.5 0. 0. 1.E-3
GE 0
EX 0 1 5 0 1. 0.
FR 0 1 0 0 30.
XQ
NX
"""

# A closed square loop, one wire per side, driven mid-side.  Every node has
# degree 2, so the structure is a PURE CYCLE and only the cut opens it.
LOOP = """CM a square loop
CE
GW 1 5 0. 0. 0. 1. 0. 0. 1.E-3
GW 2 5 1. 0. 0. 1. 1. 0. 1.E-3
GW 3 5 1. 1. 0. 0. 1. 0. 1.E-3
GW 4 5 0. 1. 0. 0. 0. 0. 1.E-3
GE 0
EX 0 1 3 0 1. 0.
FR 0 1 0 0 60.
XQ
NX
"""

# A tee: wire 2's end lands on wire 1's interior segment boundary, so the
# dialect shatters wire 1 and the crossing becomes a shared END.
TEE = """CM a tee
CE
GW 1 4 0. 0. 0. 1. 0. 0. 1.E-3
GW 2 4 0.5 0. 0. 0.5 1. 0. 1.E-3
GE 0
EX 0 2 2 0 1. 0.
FR 0 1 0 0 100.
XQ
NX
"""


def mesh_of(text: str):
    """The deck's mesh, over its own union port set."""
    model = parse(text)
    sites, _feeds, _loads = _sites(model)
    return model, to_polylines(
        model, tuple((site.wire, site.arclength) for site in sites)
    )


# ---------------------------------------------------------------------------
# the basis roster
# ---------------------------------------------------------------------------


def test_every_nec2_basis_builds_the_same_model():
    """Six solver families construct from one deck, under the seven names
    the nec2 roster takes. Razor's two names are in ``BASES`` for the NEC-5
    dialect and refused here (momwire#821, the test after this one)."""
    model = parse(DIPOLE)
    families = set()
    for name, (solver_class, _kwargs) in NEC2_BASES.items():
        built = build_solver(model, basis=name)
        assert isinstance(built.solver, solver_class)
        assert built.basis == name
        families.add(solver_class)
    assert len(families) == 6
    assert set(BASES) - set(NEC2_BASES) == {"razor-2p", "razor-nec5"}


def test_the_nec2_roster_is_the_centre_feeds_column_of_the_matrix():
    """`NEC2_BASES` is derived, not listed: exactly the `BASES` entries whose
    class declares it lands a gap on the segment-centre grid this dialect
    names. A new family that snaps to knots drops out of this front door by
    declaring so, and one that lands centres joins it the same way."""
    for name, (solver_class, _kwargs) in BASES.items():
        assert (name in NEC2_BASES) is bool(solver_class.capabilities.centre_feeds)


@pytest.mark.parametrize("basis", ["razor-2p", "razor-nec5"])
def test_a_centre_snapping_basis_refuses_a_fed_deck_by_name(basis):
    """The `centre_feeds` gate momwire#673 declared and momwire#821 landed.

    Every parsed nec2 deck carries an `EX`, so this refuses razor for every
    deck the dialect can produce. The message ends with the family's own
    declared sentence — `tests/test_refusals_are_declared.py`'s contract —
    and names the feed that tripped it, so the caller can see it was the
    deck's own `EX` and not a parser artefact.
    """
    from momwire.razor import RazorSolver

    with pytest.raises(ValueError) as excinfo:
        build_solver(parse(DIPOLE), basis=basis)
    message = str(excinfo.value)
    assert message.endswith(RazorSolver.capabilities.refusal("centre_feeds"))
    assert f"basis {basis!r} does not place a gap there" in message
    assert "names a segment CENTRE" in message


def test_the_default_basis_is_the_degree_2_bspline():
    built = build_solver(parse(DIPOLE))
    assert built.basis == "bspline"
    assert isinstance(built.solver, BASES["bspline"][0])


def test_an_unknown_basis_refuses_by_name():
    with pytest.raises(ValueError) as excinfo:
        build_solver(parse(DIPOLE), basis="bs2")
    message = str(excinfo.value)
    assert "unknown basis 'bs2'" in message
    # The refusal is the roster's own source, so a new family cannot be added
    # without the message learning its name.
    for name in BASES:
        assert repr(name) in message


# ---------------------------------------------------------------------------
# chaining, junctions, cycles
# ---------------------------------------------------------------------------


def test_a_straight_dipole_is_one_polyline_with_no_junctions():
    _model, mesh = mesh_of(DIPOLE)
    assert len(mesh.polylines) == 1
    # Fed at the middle element of an odd count: already the edge's midpoint,
    # so no split — one edge of nine elements.
    assert mesh.edge_elements == ((9,),)
    assert mesh.junctions == ()
    assert mesh.ports == ((0, 2.5),)


def test_an_off_centre_feed_gets_its_element_as_an_edge():
    """The port's position is then half of ONE edge rather than a sum."""
    off_centre = DIPOLE.replace("EX 0 1 5", "EX 0 1 3")
    _model, mesh = mesh_of(off_centre)
    assert mesh.edge_elements == ((2, 1, 6),)
    assert len(mesh.polylines) == 1
    assert mesh.polylines[0].shape == (4, 3)
    polyline, arclength = mesh.ports[0]
    # Element 3 of nine over a 5 m wire: centre at 2.5/9 * 5 m from the end.
    assert arclength == pytest.approx(5.0 * 2.5 / 9.0)


def test_two_ports_on_one_edge_both_split():
    """The midpoint exception is for a LONE port; a second one on the same
    edge means the first has to be cut out anyway."""
    two = DIPOLE.replace("EX 0 1 5 0 1. 0.", "EX 0 1 5 0 1. 0.\nEX 0 1 7 0 1. 0.")
    _model, mesh = mesh_of(two)
    assert mesh.edge_elements == ((4, 1, 1, 1, 2),)


def test_a_tee_junctions_at_the_shared_node():
    model, mesh = mesh_of(TEE)
    # The dialect shattered wire 1 in two; wire 2 is whole but fed off centre.
    assert len(model.wires) == 3
    assert len(mesh.polylines) == 3
    assert len(mesh.junctions) == 1
    members = mesh.junctions[0]
    assert len(members) == 3
    assert {end for _wire, end in members} <= {"start", "end"}
    # Every member is a real polyline end sitting on the tee node.
    tee_node = np.array([0.5, 0.0, 0.0])
    for wire, end in members:
        node = mesh.polylines[wire][0 if end == "start" else -1]
        assert np.allclose(node, tee_node)


def test_a_pure_cycle_is_cut_at_the_fed_edge():
    _model, mesh = mesh_of(LOOP)
    # Two polylines: the fed side, and the long way round.  The feed is at
    # the middle element of an odd count, so the midpoint exception keeps the
    # side whole and the whole side IS the cut edge.
    assert len(mesh.polylines) == 2
    assert mesh.edge_elements == ((5,), (5, 5, 5))
    assert sum(sum(counts) for counts in mesh.edge_elements) == 20
    # Both cut nodes carry one end of each polyline, so KCL closes the loop.
    assert len(mesh.junctions) == 2
    for members in mesh.junctions:
        assert sorted(wire for wire, _end in members) == [0, 1]
    polyline, arclength = mesh.ports[0]
    assert polyline == 0
    assert arclength == pytest.approx(0.5)


def test_an_off_centre_feed_cuts_a_cycle_at_its_element():
    """The split runs BEFORE the cut, so the cut edge is the port's element
    and the loop opens exactly where the deck drives it."""
    _model, mesh = mesh_of(LOOP.replace("EX 0 1 3", "EX 0 1 2"))
    assert len(mesh.polylines) == 2
    assert mesh.edge_elements[0] == (1,)
    assert sum(sum(counts) for counts in mesh.edge_elements) == 20
    assert len(mesh.junctions) == 2
    polyline, arclength = mesh.ports[0]
    assert polyline == 0
    assert arclength == pytest.approx(0.1)


def test_a_parasitic_cycle_is_cut_at_its_lowest_numbered_edge():
    """A loop with no port of its own still has to be opened."""
    text = LOOP.replace("EX 0 1 3 0 1. 0.", "EX 0 1 3 0 1. 0.").replace(
        "GE 0", "GW 5 3 0. 0. 1. 1. 0. 1. 1.E-3\nGE 0"
    )
    # Drive the added straight wire instead, leaving the square parasitic.
    text = text.replace("EX 0 1 3", "EX 0 5 2")
    _model, mesh = mesh_of(text)
    # The straight wire, plus the square opened into two polylines.
    assert len(mesh.polylines) == 3
    assert len(mesh.junctions) == 2


def test_a_radius_change_ends_a_polyline():
    """A solver takes one radius per wire, so a change mid-chain has to
    become a two-member junction rather than a silent average."""
    model = DeckModel(
        wires=(
            DeckWire(
                vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                radius=1e-3,
                edge_elements=(3,),
            ),
            DeckWire(
                vertices=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
                radius=2e-3,
                edge_elements=(3,),
            ),
        ),
        feeds=((0, 0.5, 1 + 0j),),
        groups=(ExecuteGroup(frequencies=(100.0,), voltages=(1 + 0j,)),),
    )
    mesh = to_polylines(model, ((0, 0.5),))
    assert len(mesh.polylines) == 2
    assert mesh.radii == (1e-3, 2e-3)
    assert len(mesh.junctions) == 1
    assert sorted(mesh.junctions[0]) == [(0, "end"), (1, "start")]


def test_a_material_change_ends_a_polyline_too():
    copper = WireMaterial(conductivity=5.8e7)
    model = DeckModel(
        wires=(
            DeckWire(
                vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                radius=1e-3,
                edge_elements=(3,),
                material=copper,
            ),
            DeckWire(
                vertices=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
                radius=1e-3,
                edge_elements=(3,),
            ),
        ),
        feeds=((0, 0.5, 1 + 0j),),
    )
    mesh = to_polylines(model, ((0, 0.5),))
    assert len(mesh.polylines) == 2
    assert mesh.materials == (copper, None)


def test_the_arclength_remap_is_measured_along_the_chained_polyline():
    """A port on the SECOND wire of a chain is positioned from the chain's
    start, not its own wire's."""
    model = DeckModel(
        wires=(
            DeckWire(
                vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                radius=1e-3,
                edge_elements=(2,),
            ),
            DeckWire(
                vertices=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
                radius=1e-3,
                edge_elements=(3,),
            ),
        ),
        feeds=((1, 0.5, 1 + 0j),),
    )
    mesh = to_polylines(model, ((1, 0.5),))
    assert len(mesh.polylines) == 1
    polyline, arclength = mesh.ports[0]
    assert polyline == 0
    # 1 m of the first wire, then the middle of the second.
    assert arclength == pytest.approx(1.5)
    assert math.isclose(float(mesh.polylines[0][0][0]), 0.0)


# ---------------------------------------------------------------------------
# the port plan
# ---------------------------------------------------------------------------


def test_the_plan_indexes_solver_ports_by_model_feed():
    model = parse(DIPOLE)
    built = build_solver(model)
    plan = built.ports
    assert plan.n_ports == 1
    assert plan.feed_ports == (0,)
    assert plan.load_ports == ()
    assert plan.sites[0].feed == 0
    assert plan.sites[0].load is None
    assert plan.voltages == ((1 + 0j,),)
    assert plan.n_ports == built.solver.compute_port_solution().y.shape[0]


def test_a_load_becomes_a_port_carrying_its_spec():
    text = DIPOLE.replace("EX 0 1 5", "LD 4 1 3 3 50. 10.\nEX 0 1 5")
    model = parse(text)
    built = build_solver(model)
    plan = built.ports
    assert plan.n_ports == 2
    assert len(plan.load_ports) == 1
    loaded = plan.loaded_ports()
    assert len(loaded) == 1
    port, spec = loaded[0]
    assert port == plan.load_ports[0]
    assert spec == LoadSpec("fixed", r=50.0, x=10.0)
    # The load port is undriven in every group — a shorted gap until a
    # consumer stamps it.
    for drive in plan.voltages:
        assert drive is not None
        assert drive[port] == 0j


def test_a_load_on_a_driven_segment_shares_its_port():
    text = DIPOLE.replace("EX 0 1 5", "LD 4 1 5 5 50. 0.\nEX 0 1 5")
    plan = build_solver(parse(text)).ports
    assert plan.n_ports == 1
    assert plan.feed_ports == plan.load_ports == (0,)
    site = plan.sites[0]
    assert site.feed == 0 and site.load == 0
    assert site.load_spec == LoadSpec("fixed", r=50.0, x=0.0)


def test_every_group_gets_a_drive_vector_over_one_port_set():
    """Two execute groups driving two different segments: one geometry, one
    port set, two voltage vectors — and a group that ran nothing is None."""
    text = """CM two groups
CE
GW 1 9 -2.5 0. 0. 2.5 0. 0. 1.E-3
GW 2 9 -2.5 1. 0. 2.5 1. 0. 1.E-3
GE 0
FR 0 1 0 0 30.
EX 0 1 5 0 1. 0.
XQ
EX 0 2 5 0 2. 0.
XQ
XQ
NX
"""
    model = parse(text)
    plan = build_solver(model).ports
    assert plan.n_ports == 2
    assert len(plan.voltages) == len(model.groups) == 3
    assert plan.voltages[2] is None
    first, second = plan.voltages[0], plan.voltages[1]
    assert first is not None and second is not None
    assert set(first) == {1 + 0j, 0j}
    assert set(second) == {2 + 0j, 0j}
    # Different ports drive in the two groups.
    assert first.index(1 + 0j) != second.index(2 + 0j)


def test_the_built_solver_records_the_choices_it_froze():
    model = parse(DIPOLE)
    built = build_solver(model, frequency_mhz=14.0, extended_kernel=True)
    assert built.frequency_mhz == 14.0
    assert built.extended_kernel is True
    assert built.wavelength == pytest.approx(299_792_458.0 / 14e6)
    assert built.group == 0


def test_a_group_that_ran_nothing_cannot_be_selected():
    text = DIPOLE.replace("XQ\nNX", "XQ\nXQ\nNX")
    model = parse(text)
    assert model.groups[1] is None
    with pytest.raises(ValueError, match="ran nothing"):
        build_solver(model, group=1)


# ---------------------------------------------------------------------------
# ground and material kwargs
# ---------------------------------------------------------------------------


def test_free_space_passes_no_plane():
    """``ground_z=None`` is "no plane"; 0.0 would be a plane at the origin."""
    assert _ground(Environment()) == {"ground_z": None}


def test_pec_ground_passes_the_plane_and_no_constants():
    assert _ground(Environment(ground="pec")) == {"ground_z": 0.0}


def test_the_reflection_coefficient_ground_names_no_model():
    """refl-coef is every solver's default, so naming it would be a second
    spelling of the same solve."""
    environment = Environment(ground=("finite-fast", 13.0, 0.005))
    assert _ground(environment) == {"ground_z": 0.0, "ground_eps": (13.0, 0.005)}


def test_the_sommerfeld_ground_names_its_model():
    environment = Environment(ground=("finite", 13.0, 0.005))
    assert _ground(environment) == {
        "ground_z": 0.0,
        "ground_eps": (13.0, 0.005),
        "ground_model": "sommerfeld",
    }


def test_an_unrecognised_ground_refuses():
    with pytest.raises(ValueError, match="unrecognised ground spec"):
        _ground(Environment(ground=("swamp", 1.0, 1.0)))


def test_a_gn0_deck_that_touches_the_plane_refuses_at_build():
    """The most-imported deck of its class — a ground-mounted vertical with
    `GN 0` — refuses BY DESIGN since momwire#282 stage 1 (2026-08-18).

    The front end passes `GN 0` through as the solvers' default ground and
    the solver's own constructor is what refuses, so this is a plumbing
    check rather than a second refusal: the deck path must not have some
    way around the withdrawal. It is gated because
    `site/src/content/docs/reference/deck-grammar-nec2.md` tells importers
    this in so many words, and a documented refusal that does not fire is
    worse than an undocumented one.
    """
    deck = (
        "CM ground-mounted vertical over GN 0\nCE\n"
        "GW 1 21 0. 0. 0. 0. 0. 5.35 5.E-3\n"
        "GE 1\nGN 0 0 0 0 13. .005\n"
        "EX 0 1 1 0 1. 0.\nFR 0 1 0 0 14.\nXQ\nEN\n"
    )
    with pytest.raises(NotImplementedError, match="ground CONTACT under"):
        build_solver(parse(deck))

    # `GN 2` — the Sommerfeld solution — is the way through, and it builds.
    somm = deck.replace("GN 0 0 0 0 13. .005", "GN 2 0 0 0 13. .005")
    assert build_solver(parse(somm)) is not None


def test_bare_wire_passes_no_loading_kwargs():
    assert _wire_loading((None, None)) == {}


def test_conductivity_and_insulation_travel_as_per_wire_arrays():
    materials = (
        WireMaterial(conductivity=5.8e7),
        None,
        WireMaterial(insulation_radius=2e-3, insulation_eps_r=2.3),
    )
    kwargs = _wire_loading(materials)
    assert set(kwargs) == {
        "wire_conductivity",
        "insulation_radius",
        "insulation_eps_r",
    }
    # NaN is momwire's "not this wire", so an absent value is a hole in the
    # array rather than a zero (which would be a perfect insulator / a short).
    assert kwargs["wire_conductivity"][0] == 5.8e7
    assert np.isnan(kwargs["wire_conductivity"][1:]).all()
    assert np.isnan(kwargs["insulation_radius"][:2]).all()
    assert kwargs["insulation_eps_r"][2] == 2.3


def test_a_ground_deck_builds_with_the_plane_in_place():
    text = DIPOLE.replace(
        "GW 1 9 -2.5 0. 0. 2.5 0. 0. 1.E-3", "GW 1 9 -2.5 0. 5. 2.5 0. 5. 1.E-3"
    ).replace("EX 0 1 5", "GN 2 0 0 0 13. 0.005\nEX 0 1 5")
    model = parse(text)
    assert model.ground == ("finite", 13.0, 0.005)
    built = build_solver(model)
    assert built.solver.compute_port_solution().y.shape == (1, 1)


# ---------------------------------------------------------------------------
# the environment: the group's, not the deck's
# ---------------------------------------------------------------------------

# A wire well clear of the plane, run once in free space and once over perfect
# ground.  ``GN`` arms, so the two execute cards are two real runs over two
# different half-spaces (spec ``#arming``).
GN_REARM = """CM gn between executes
CE
GW 1 9 0. 0. 2.0 0. 0. 7.0 0.001
GE -1
EX 0 1 5 0 1.
FR 0 1 0 0 14.1
XQ
GN 1
XQ
NX
"""


def test_a_solver_is_built_over_the_selected_groups_environment():
    """The environment is an operating-point choice like the frequency and
    the kernel, and ``group=`` selects all three together."""
    model = parse(GN_REARM)
    assert build_solver(model, group=0).solver.ground_z is None  # free space
    assert build_solver(model, group=1).solver.ground_z == 0.0  # the plane


def test_the_environment_override_beats_the_groups():
    """What a caller sweeping one structure over several grounds needs, and
    the shape the portal uses."""
    model = parse(GN_REARM)
    built = build_solver(model, group=1, environment=Environment())
    assert built.solver.ground_z is None


def test_a_model_with_no_group_falls_back_to_the_deck_environment():
    model = DeckModel(
        wires=(
            DeckWire(
                vertices=((-0.5, 0.0, 5.0), (0.5, 0.0, 5.0)),
                radius=1e-3,
                edge_elements=(5,),
            ),
        ),
        feeds=((0, 0.5, 1 + 0j),),
        ground="pec",
    )
    assert model.groups == ()
    assert build_solver(model, frequency_mhz=14.0).solver.ground_z == 0.0


# ---------------------------------------------------------------------------
# the prepared mesh: translate once, fill many times
# ---------------------------------------------------------------------------


def test_a_prepared_mesh_hands_every_solver_the_same_arrays():
    """The reuse is by identity, not by equality: two solvers built from one
    handle share the coordinate arrays rather than each rounding the same
    walk for itself."""
    model = parse(DIPOLE)
    handle = prepare_mesh(model)
    a = build_solver(model, mesh=handle, frequency_mhz=14.0)
    b = build_solver(model, mesh=handle, frequency_mhz=21.0)
    assert a.ports is b.ports is handle.ports
    for left, right in zip(a.solver.wires_polylines, b.solver.wires_polylines):
        assert left is right
        assert np.shares_memory(left, right)
    # And the same arrays the handle carries, not copies of them.
    for solver_array, mesh_array in zip(
        a.solver.wires_polylines, handle.mesh.polylines
    ):
        assert np.shares_memory(solver_array, mesh_array)


@pytest.mark.parametrize("deck", [DIPOLE, LOOP, TEE])
def test_a_prepared_solve_is_bit_equal_to_an_unprepared_one(deck):
    """The handle is a saved computation, not a different one."""
    model = parse(deck)
    plain = build_solver(model).solver.compute_port_solution().y
    prepared = (
        build_solver(model, mesh=prepare_mesh(model)).solver.compute_port_solution().y
    )
    assert np.array_equal(plain, prepared)


def test_a_prepared_mesh_carries_the_plan_the_unprepared_path_builds():
    model = parse(TEE)
    assert prepare_mesh(model).ports == build_solver(model).ports


def test_a_handle_prepared_for_another_model_refuses():
    """A silent wrong answer is the alternative: the handle names polylines
    and port rows, and another structure's are different integers."""
    handle = prepare_mesh(parse(DIPOLE))
    with pytest.raises(ValueError, match="different model"):
        build_solver(parse(TEE), mesh=handle)


def test_an_equal_model_replays_a_handle():
    """Equality, not identity: a model reparsed from the same text describes
    the same structure, and the handle is about the structure."""
    handle = prepare_mesh(parse(DIPOLE))
    twin = parse(DIPOLE)
    assert twin is not handle.model
    assert build_solver(twin, mesh=handle).ports is handle.ports


# ---------------------------------------------------------------------------
# node gaps (no nec2 deck emits one; the seam is real anyway)
# ---------------------------------------------------------------------------


def test_a_node_gap_becomes_a_trailing_port_at_a_wire_end():
    model = DeckModel(
        wires=(
            DeckWire(
                vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                radius=1e-3,
                edge_elements=(3,),
            ),
            DeckWire(
                vertices=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
                radius=1e-3,
                edge_elements=(3,),
            ),
        ),
        feeds=((0, 0.5, 1 + 0j),),
        node_gaps=((1, 0, 1 + 0j),),
        groups=(ExecuteGroup(frequencies=(100.0,), voltages=(1 + 0j,)),),
    )
    plan = build_solver(model).ports
    assert plan.node_gap_ports == (1,)
    assert plan.n_ports == 2
    # The gap forced its knot to be a polyline end, so the chain broke there.
    mesh = to_polylines(model, ((0, 0.5),))
    assert len(mesh.polylines) == 2
    assert mesh.node_gap_members == ((1, "start"),)
