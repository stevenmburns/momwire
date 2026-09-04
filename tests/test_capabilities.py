"""The capability registry (momwire#396): declarations vs. reality.

`Capabilities` and `.refusal()` are covered directly here (§ "Unit rows"),
and every shipping solver's declared row is cross-checked against its
ACTUAL refusal surface: for every declared False/absent cell with a
constructor- or solve-level refusal, a tiny deck exercising that cell must
both raise AND have `capabilities.refusal(cell)` say why; for every True
cell, the constructor must at least ACCEPT the kwarg. This is the
definition-of-done test the issue asks for — declarations are cross-checked
against code, not trusted on their own.
"""

import numpy as np
import pytest

from momwire._capabilities import AXIS_VALUES, DERIVED_AXES, axes_for
from momwire.deck._solver import BASES
from momwire.harrington import HarringtonSolver
from momwire.pulse import PulseSolver

from momwire import (
    ArrayBlockSolver,
    BSplineSolver,
    Capabilities,
    HMatrixSolver,
    RazorSolver,
    SinusoidalGalerkinSolver,
    SinusoidalSolver,
)

# Every class carrying a compositional row, for section 9. Kept beside the
# imports rather than re-derived per test so adding a solver is one edit.
_AXIS_CLASSES = (
    BSplineSolver,
    HMatrixSolver,
    ArrayBlockSolver,
    SinusoidalSolver,
    SinusoidalGalerkinSolver,
    RazorSolver,
    HarringtonSolver,
    PulseSolver,
)

WAVELENGTH = 10.0


def _wire(z=0.0, n=2):
    """One straight two-anchor wire, `n` segments, at height `z`."""
    return [np.array([(0.0, 0.0, z), (0.0, 1.0, z)])], [[n]]


def _junction_pair(z=0.0, n=2):
    """Two wires meeting end-to-start at one point, for junction_ports /
    node_gaps decks."""
    wires = [
        np.array([(0.0, 0.0, z), (1.0, 0.0, z)]),
        np.array([(1.0, 0.0, z), (1.0, 1.0, z)]),
    ]
    npe = [[n], [n]]
    junctions = [[(0, "end"), (1, "start")]]
    return wires, npe, junctions


# --------------------------------------------------------------------------
# 1. Capabilities / refusal() unit rows
# --------------------------------------------------------------------------

_FAKE = Capabilities(
    grounds=frozenset({"pec"}),
    wire_loading=True,
    extended_kernel=False,
    junction_ports=False,
    node_gaps=False,
    knot_feeds=True,
    per_wire_radius=True,
    singular_enrichment=False,
    refusals={
        "extended_kernel": "no EK here",
        "junction_ports+finite_ground": "combo reason",
    },
)


def test_served_boolean_returns_none():
    assert _FAKE.refusal("wire_loading") is None


def test_served_ground_returns_none():
    assert _FAKE.refusal("pec") is None


def test_absent_ground_uses_generated_default():
    msg = _FAKE.refusal("refl-coef")
    assert msg is not None and "refl-coef" in msg


def test_false_boolean_with_dict_entry_returns_reason():
    assert _FAKE.refusal("extended_kernel") == "no EK here"


def test_false_boolean_without_dict_entry_generates_default():
    msg = _FAKE.refusal("node_gaps")
    assert msg == "node_gaps is not supported by this solver"


def test_combination_key_hits_regardless_of_argument_order():
    assert _FAKE.refusal("junction_ports", "finite_ground") == "combo reason"
    assert _FAKE.refusal("finite_ground", "junction_ports") == "combo reason"


def test_combination_falls_back_to_first_refused_single_cell():
    # No "extended_kernel+node_gaps" entry: falls through to the first
    # single-cell check (extended_kernel, which has its own dict entry).
    assert _FAKE.refusal("extended_kernel", "node_gaps") == "no EK here"


def test_unmatched_condition_token_is_served():
    """A condition token ("finite_ground", "mixed_radii") is not a
    capability: it carries meaning only through a declared "a+b" key, so on
    a row WITHOUT that key it must read as served — else every solver that
    serves the combination would spuriously refuse it. This is the call
    shape the antennaknobs consumer uses, so the real case is pinned too:
    BSplineSolver serves junction ports over a finite ground and declares
    no such key."""
    assert _FAKE.refusal("wire_loading", "finite_ground") is None
    assert _FAKE.refusal("mixed_radii") is None
    assert BSplineSolver.capabilities.refusal("junction_ports", "finite_ground") is None
    assert (
        SinusoidalGalerkinSolver.capabilities.refusal("junction_ports", "finite_ground")
        is not None
    )


def test_capabilities_importable_from_top_level():
    import momwire

    assert momwire.Capabilities is Capabilities
    assert "Capabilities" in momwire.__all__


# --------------------------------------------------------------------------
# 2. RazorSolver — the PEC ground is served; the rest of the row is refused
# --------------------------------------------------------------------------


def test_razor_pec_ground_is_served():
    """momwire#398 unit 2: the razor-grounds pilot's first capability.

    `RazorSolver` shipped refusing every ground; `PotentialGround` gave it
    the PEC image without razor writing a mirror of its own, so the declared
    row gained `"pec"` and the constructor accepts `ground_z`. Both halves
    are checked together — a served declaration with a refusing constructor
    is exactly the drift this file exists to catch.
    """
    c = RazorSolver.capabilities
    assert "pec" in c.grounds
    assert c.refusal("pec") is None
    assert "pec" not in c.refusals

    wires, npe = _wire(z=1.0)
    sim = RazorSolver(
        wires=wires, n_per_edge_per_wire=npe, wavelength=WAVELENGTH, ground_z=0.0
    )
    assert sim.ground_z == 0.0
    z, _ = sim.compute_impedance()
    assert np.isfinite(z.real) and np.isfinite(z.imag)


def test_razor_refl_coef_ground_is_served():
    """momwire#398 unit 4: the second row the shared layer handed razor.

    Same two halves as the PEC row above — the declaration and the
    constructor together — plus the two knobs that come with it
    (`ground_phi_mode`, and `ground_model` at its served value). The
    physics lives in `tests/test_razor_refl_coef_ground.py`.
    """
    c = RazorSolver.capabilities
    assert "refl-coef" in c.grounds
    assert c.refusal("refl-coef") is None
    assert "refl-coef" not in c.refusals

    wires, npe = _wire(z=1.0)
    sim = RazorSolver(
        wires=wires,
        n_per_edge_per_wire=npe,
        wavelength=WAVELENGTH,
        ground_z=0.0,
        ground_eps=10 - 1j,
        ground_phi_mode="rho_v",
        ground_model="refl-coef",
    )
    assert sim.ground_eps == 10 - 1j
    assert sim.ground_phi_mode == "rho_v"
    z, _ = sim.compute_impedance()
    assert np.isfinite(z.real) and np.isfinite(z.imag)


def test_razor_sommerfeld_ground_is_served():
    """momwire#398 unit 5: the third row, and the first COMPOSING ground on
    this solver.

    Same two halves as the rows above. The extra check is the one that says
    the declaration is not a rename of the refl-coef row: the composing
    ground has to reach a DIFFERENT matrix, because C2 image + remainder is
    not any weighting of the Fresnel image. `ground_phi_mode` is accepted
    and unread over it, exactly as in `BSplineSolver`. The physics lives in
    `tests/test_razor_sommerfeld_ground.py`.
    """
    c = RazorSolver.capabilities
    assert "sommerfeld" in c.grounds
    assert c.refusal("sommerfeld") is None
    assert "sommerfeld" not in c.refusals

    wires, npe = _wire(z=1.0)
    base = dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        wavelength=WAVELENGTH,
        ground_z=0.0,
        ground_eps=10 - 1j,
    )
    sim = RazorSolver(**base, ground_model="sommerfeld")
    assert sim.ground_model == "sommerfeld"
    z, _ = sim.compute_impedance()
    assert np.isfinite(z.real) and np.isfinite(z.imag)
    z_refl, _ = RazorSolver(**base).compute_impedance()
    assert abs(z - z_refl) > 1e-6

    # ...and the permittivity is not optional: sommerfeld is the exact
    # ground OF something, so there is nothing to be exact about without it.
    with pytest.raises(ValueError, match="requires ground_eps"):
        RazorSolver(
            wires=wires,
            n_per_edge_per_wire=npe,
            wavelength=WAVELENGTH,
            ground_z=0.0,
            ground_model="sommerfeld",
        )


def test_razor_ground_contact_is_served_at_a_wire_end():
    """Ground CONTACT landed in unit 3, and the geometry refusals moved.

    `contact` became a declared axis in momwire#792 and razor's row
    reads True, so the two halves are checkable together again; what this
    still pins on its own is the BOUNDARY of the cell, which no declaration
    carries: a wire END in the plane is served, a wire lying IN it or dipping
    BELOW it is not. `tests/test_razor_ground_contact.py` carries the physics,
    and `tests/test_refusals_are_declared.py` gates the third geometry against
    the `buried` cell this deck's third case exercises.
    """
    assert RazorSolver.capabilities.contact
    assert RazorSolver.capabilities.refusal("contact") is None
    vertical = [np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)])]
    sim = RazorSolver(
        wires=vertical,
        n_per_edge_per_wire=[[4]],
        wavelength=WAVELENGTH,
        ground_z=0.0,
        feed_arclength=0.0,
    )
    z, _ = sim.compute_impedance()
    assert np.isfinite(z.real) and np.isfinite(z.imag)

    wires, npe = _wire(z=0.0)
    with pytest.raises(ValueError, match="edge lying in the ground plane"):
        RazorSolver(
            wires=wires, n_per_edge_per_wire=npe, wavelength=WAVELENGTH, ground_z=0.0
        )
    with pytest.raises(ValueError, match="runs below the ground plane"):
        RazorSolver(
            wires=wires, n_per_edge_per_wire=npe, wavelength=WAVELENGTH, ground_z=1.0
        )


@pytest.mark.parametrize(
    "cell,kwarg",
    [
        ("junction_ports", {"junction_ports": [0]}),
    ],
)
def test_razor_out_of_scope_kwargs_refuse(cell, kwarg):
    wires, npe = _wire()
    reason = RazorSolver.capabilities.refusal(cell)
    assert reason
    with pytest.raises(NotImplementedError):
        RazorSolver(
            wires=wires, n_per_edge_per_wire=npe, wavelength=WAVELENGTH, **kwarg
        )


def test_razor_per_wire_radius_accepted():
    """momwire#147 filled this cell: `wire_radius` takes the siblings'
    per-wire sequence, gated against the binary's own mixed-radius `GW`
    decks (`tests/test_razor_mixed_radius.py`)."""
    assert RazorSolver.capabilities.per_wire_radius
    assert RazorSolver.capabilities.refusal("per_wire_radius") is None
    wires = [
        np.array([(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]),
        np.array([(1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]),
    ]
    RazorSolver(
        wires=wires,
        n_per_edge_per_wire=[[4], [4]],
        wire_radius=[0.001, 0.0015],
        wavelength=WAVELENGTH,
    )


def test_razor_wire_loading_accepted():
    """momwire#427 moved this cell out of the TypeError list below."""
    wires, npe = _wire()
    RazorSolver(
        wires=wires,
        n_per_edge_per_wire=npe,
        wavelength=WAVELENGTH,
        wire_conductivity=1e7,
        lumped_loads=[(0, None, 50.0 + 0.0j)],
    )


@pytest.mark.parametrize(
    "cell,kwarg",
    [
        ("singular_enrichment", {"use_singular_enrichment": True}),
    ],
)
def test_razor_unsupported_kwargs_are_typeerrors(cell, kwarg):
    # Not in razor's _OUT_OF_SCOPE dict at all, so it's a caller-typo
    # TypeError rather than a NotImplementedError — but capabilities still
    # declares the cell refused, with a generated default reason.
    wires, npe = _wire()
    assert RazorSolver.capabilities.refusal(cell)
    with pytest.raises(TypeError):
        RazorSolver(
            wires=wires, n_per_edge_per_wire=npe, wavelength=WAVELENGTH, **kwarg
        )


def test_razor_served_row_is_every_ground_plus_loading_radii_and_the_kernel():
    """PEC (unit 2), refl-coef (unit 4) and sommerfeld (unit 5) — the whole
    ground column, for wires clear of the plane — plus wire loading
    (momwire#427), per-wire radii (momwire#147) and the EXTENDED KERNEL
    (momwire#398 D1, filled when the taper study identified the NEC-5 binary as
    extended-kernel everywhere). Every other cell is still refused, and the
    one combination that is refused inside the served column (contact over a
    finite ground, momwire#282) says so through its own key rather than by
    removing a ground."""
    c = RazorSolver.capabilities
    assert c.grounds == frozenset({"pec", "refl-coef", "sommerfeld"})
    assert c.wire_loading
    assert c.per_wire_radius
    assert c.extended_kernel
    assert c.refusal("wire_loading") is None
    assert c.refusal("per_wire_radius") is None
    assert c.refusal("extended_kernel") is None
    # momwire#624: contact over the SOMMERFELD ground is served, so the
    # `contact+finite_ground` and `contact+sommerfeld` keys are ABSENT rather
    # than answering prose — an absent key is this roster's spelling for "not
    # a refusal here", which is why it is asserted as None and not merely
    # left unmentioned.
    assert c.refusal("contact", "finite_ground") is None
    assert c.refusal("contact", "sommerfeld") is None
    # What remains is D3's row, refused across the whole tree because the
    # MODEL fails at zero clearance, and it is `_ground_spec`'s one sentence
    # rather than a razor-owned copy of it.
    assert c.refusal("contact", "refl-coef") == c.refusals["contact+refl-coef"]
    assert "momwire#282" in c.refusal("contact", "refl-coef")
    assert c.refusal("contact", "refl-coef") == BSplineSolver.capabilities.refusal(
        "contact", "refl-coef"
    )
    assert not any(
        [
            c.junction_ports,
            c.singular_enrichment,
        ]
    )
    # momwire#603 U4 filled this cell: the K-1 through-current tents were
    # always built, only the port that drives one was missing.
    assert c.node_gaps
    assert c.refusal("node_gaps") is None


# --------------------------------------------------------------------------
# 3. SinusoidalSolver — free-space/grounds/loading/EK/per-wire-radius served;
#    junction_ports and node_gaps refused
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ground_z", [0.5])
def test_sinusoidal_grounds_served(ground_z):
    wires, npe = _wire(z=ground_z)
    for kw in (
        {"ground_z": 0.0},
        {"ground_z": 0.0, "ground_eps": 10 - 1j},
        {"ground_z": 0.0, "ground_eps": 10 - 1j, "ground_model": "sommerfeld"},
    ):
        SinusoidalSolver(
            wires=wires, n_per_edge_per_wire=npe, wavelength=WAVELENGTH, **kw
        )


def test_sinusoidal_wire_loading_accepted():
    wires, npe = _wire()
    SinusoidalSolver(
        wires=wires,
        n_per_edge_per_wire=npe,
        wavelength=WAVELENGTH,
        wire_conductivity=1e7,
    )


def test_sinusoidal_extended_kernel_accepted():
    wires, npe = _wire()
    SinusoidalSolver(
        wires=wires,
        n_per_edge_per_wire=npe,
        wavelength=WAVELENGTH,
        extended_kernel=True,
    )


def test_sinusoidal_per_wire_radius_accepted():
    wires, npe = _wire()
    SinusoidalSolver(
        wires=wires, n_per_edge_per_wire=npe, wavelength=WAVELENGTH, wire_radius=[0.001]
    )


def test_sinusoidal_junction_ports_refused():
    wires, npe, junctions = _junction_pair()
    reason = SinusoidalSolver.capabilities.refusal("junction_ports")
    assert reason
    with pytest.raises(NotImplementedError):
        SinusoidalSolver(
            wires=wires,
            n_per_edge_per_wire=npe,
            junctions=junctions,
            junction_ports=[0],
            feeds=[],
            wavelength=WAVELENGTH,
        )


def test_sinusoidal_node_gaps_refused_no_such_kwarg():
    # node_gaps isn't even in SinusoidalSolver's signature — declaring it
    # False (not "unimplemented in a way that raises NotImplementedError")
    # is correct; passing it is a bare TypeError.
    wires, npe = _wire()
    reason = SinusoidalSolver.capabilities.refusal("node_gaps")
    assert reason
    with pytest.raises(TypeError):
        SinusoidalSolver(
            wires=wires,
            n_per_edge_per_wire=npe,
            wavelength=WAVELENGTH,
            node_gaps=[(0, "end", 1.0 + 0j)],
        )


def test_sinusoidal_singular_enrichment_no_such_kwarg():
    wires, npe = _wire()
    reason = SinusoidalSolver.capabilities.refusal("singular_enrichment")
    assert reason
    with pytest.raises(TypeError):
        SinusoidalSolver(
            wires=wires,
            n_per_edge_per_wire=npe,
            wavelength=WAVELENGTH,
            use_singular_enrichment=True,
        )


# --------------------------------------------------------------------------
# 4. SinusoidalGalerkinSolver — junction_ports/node_gaps served, but
#    refused in combination with a finite ground or mixed radii (at SOLVE
#    time — construction is lazy about the check)
# --------------------------------------------------------------------------


def test_sg_junction_ports_and_node_gaps_accepted_alone():
    wires, npe, junctions = _junction_pair()
    SinusoidalGalerkinSolver(
        wires=wires,
        n_per_edge_per_wire=npe,
        junctions=junctions,
        junction_ports=[0],
        feeds=[],
        wavelength=WAVELENGTH,
    )
    SinusoidalGalerkinSolver(
        wires=wires,
        n_per_edge_per_wire=npe,
        junctions=junctions,
        node_gaps=[(0, "end", 1.0 + 0j)],
        feeds=[],
        wavelength=WAVELENGTH,
    )


def test_sg_junction_ports_over_finite_ground_refused_at_solve():
    reason = SinusoidalGalerkinSolver.capabilities.refusal(
        "junction_ports", "finite_ground"
    )
    assert reason
    wires, npe, junctions = _junction_pair(z=0.5)
    s = SinusoidalGalerkinSolver(
        wires=wires,
        n_per_edge_per_wire=npe,
        junctions=junctions,
        junction_ports=[0],
        feeds=[],
        wavelength=WAVELENGTH,
        ground_z=0.0,
        ground_eps=10 - 1j,
    )
    with pytest.raises(NotImplementedError):
        s.compute_impedance()


def test_sg_junction_ports_with_mixed_radii_refused_at_solve():
    reason = SinusoidalGalerkinSolver.capabilities.refusal(
        "junction_ports", "mixed_radii"
    )
    assert reason
    wires, npe, junctions = _junction_pair()
    s = SinusoidalGalerkinSolver(
        wires=wires,
        n_per_edge_per_wire=npe,
        junctions=junctions,
        junction_ports=[0],
        feeds=[],
        wavelength=WAVELENGTH,
        wire_radius=[0.001, 0.0015],
    )
    with pytest.raises(NotImplementedError):
        s.compute_impedance()


def test_sg_junction_ports_over_pec_ground_is_still_served():
    """The narrowed refusal (#191): PEC alone is fine — only a FINITE
    ground raises. Guards against over-declaring the combination."""
    wires, npe, junctions = _junction_pair(z=0.5)
    s = SinusoidalGalerkinSolver(
        wires=wires,
        n_per_edge_per_wire=npe,
        junctions=junctions,
        junction_ports=[(0, 1.0 + 0j)],
        feeds=[],
        wavelength=WAVELENGTH,
        ground_z=0.0,
    )
    s.compute_impedance()  # must not raise


def test_sg_wire_loading_accepted():
    wires, npe = _wire()
    SinusoidalGalerkinSolver(
        wires=wires,
        n_per_edge_per_wire=npe,
        wavelength=WAVELENGTH,
        wire_conductivity=1e7,
    )


def test_sg_grounds_served():
    wires, npe = _wire(z=0.5)
    SinusoidalGalerkinSolver(
        wires=wires,
        n_per_edge_per_wire=npe,
        wavelength=WAVELENGTH,
        ground_z=0.0,
        ground_eps=10 - 1j,
        ground_model="sommerfeld",
    )


def test_sg_extended_kernel_accepted():
    wires, npe = _wire()
    SinusoidalGalerkinSolver(
        wires=wires,
        n_per_edge_per_wire=npe,
        wavelength=WAVELENGTH,
        extended_kernel=True,
    )


def test_sg_knot_feeds_carries_no_refusal_because_it_serves_the_cell():
    """momwire#685. The declaration comment used to claim this class overrode
    `_KNOT_FEEDS_REFUSAL` with its own "not yet plumbed" prose. It does not,
    and cannot: `knot_feeds` has been True here since momwire#648, and
    `Capabilities.refusal` returns a reason only for a cell the solver does
    NOT serve. So the base's prose is unreachable from this class.

    Pinned because the failure mode is a reader believing the comment and
    "restoring" a refusal that would take this family back off the NEC-5
    seam, which gates on exactly this cell."""
    assert SinusoidalGalerkinSolver.capabilities.knot_feeds is True
    assert SinusoidalGalerkinSolver.capabilities.refusal("knot_feeds") is None
    assert "knot_feeds" not in SinusoidalGalerkinSolver.capabilities.refusals
    # The base is the family that snaps, and it keeps the prose.
    assert SinusoidalSolver.capabilities.knot_feeds is False
    assert SinusoidalSolver.capabilities.refusal("knot_feeds")


# --------------------------------------------------------------------------
# 5. BSplineSolver — everything served, three `use_singular_enrichment`
#    combination refusals (all at CONSTRUCTION)
# --------------------------------------------------------------------------


def test_bspline_grounds_served():
    wires, npe = _wire(z=0.5)
    for kw in (
        {"ground_z": 0.0},
        {"ground_z": 0.0, "ground_eps": 10 - 1j},
        {"ground_z": 0.0, "ground_eps": 10 - 1j, "ground_model": "sommerfeld"},
    ):
        BSplineSolver(wires=wires, n_per_edge_per_wire=npe, wavelength=WAVELENGTH, **kw)


def test_bspline_wire_loading_accepted():
    wires, npe = _wire()
    BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=npe,
        wavelength=WAVELENGTH,
        wire_conductivity=1e7,
    )


def test_bspline_extended_kernel_accepted():
    wires, npe = _wire()
    BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=npe,
        wavelength=WAVELENGTH,
        extended_kernel=True,
    )


def test_bspline_per_wire_radius_accepted():
    wires, npe = _wire()
    BSplineSolver(
        wires=wires, n_per_edge_per_wire=npe, wavelength=WAVELENGTH, wire_radius=[0.001]
    )


def test_bspline_singular_enrichment_accepted():
    wires, npe = _wire()
    BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=npe,
        wavelength=WAVELENGTH,
        use_singular_enrichment=True,
    )


def test_bspline_junction_ports_and_node_gaps_accepted():
    wires, npe, junctions = _junction_pair()
    BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=npe,
        junctions=junctions,
        junction_ports=[0],
        feeds=[],
        wavelength=WAVELENGTH,
    )
    BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=npe,
        junctions=junctions,
        node_gaps=[(0, "end", 1.0 + 0j)],
        feeds=[],
        wavelength=WAVELENGTH,
    )


@pytest.mark.parametrize(
    "cells,kwarg",
    [
        (
            ("wire_loading", "singular_enrichment"),
            {"use_singular_enrichment": True, "wire_conductivity": 1e7},
        ),
        (
            ("extended_kernel", "singular_enrichment"),
            {"use_singular_enrichment": True, "extended_kernel": True},
        ),
        (
            ("per_wire_radius", "singular_enrichment"),
            {"use_singular_enrichment": True, "wire_radius": [0.001, 0.0015]},
        ),
    ],
)
def test_bspline_enrichment_combinations_refused(cells, kwarg):
    reason = BSplineSolver.capabilities.refusal(*cells)
    assert reason
    wires = [
        np.array([(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]),
        np.array([(1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]),
    ]
    npe = [[2], [2]]
    with pytest.raises(NotImplementedError):
        BSplineSolver(
            wires=wires, n_per_edge_per_wire=npe, wavelength=WAVELENGTH, **kwarg
        )


# --------------------------------------------------------------------------
# 6. HMatrixSolver / ArrayBlockSolver take BSplineSolver's row in every cell
#    but ONE. The survey that put them on the parent's row whole found no
#    user-facing difference — enrichment just forces their dense-path
#    fallback rather than being refused — and it missed `buried`, which IS
#    refused here and by name (momwire#553 U5).
# --------------------------------------------------------------------------


def test_hmatrix_differs_from_bspline_in_the_buried_cell_and_the_solve_axis():
    """The accelerators exist for decks the dense fill cannot hold, so a
    silent dense fallback on a buried array is an out-of-memory rather than a
    slow answer — `_refuse_buried_fast_operator` says so. Inheriting the
    parent's row told a consumer to send exactly that deck.

    TWO cells now, not one. This test was `..._in_the_buried_cell_alone`
    until antennaknobs#1006 added the compositional axes, and the rename is
    the point rather than bookkeeping: `buried` is what this solver SERVES
    differently, `solve_strategy` is what it IS differently, and a row that
    describes both has two cells here where a row describing only the first
    had one. The physics is still required to be identical — that is what
    makes the pair worth having — so everything except those two is pinned
    equal below.
    """
    c, b = HMatrixSolver.capabilities, BSplineSolver.capabilities
    assert b.buried and not c.buried
    assert c.axes["solve_strategy"] == ("aca",)
    assert b.axes["solve_strategy"] == ("dense",)
    # ...and NOTHING else moves: same basis, same testing, same kernel.
    assert {k: v for k, v in c.axes.items() if k != "solve_strategy"} == {
        k: v for k, v in b.axes.items() if k != "solve_strategy"
    }
    assert c._replace(buried=True, refusals=b.refusals, axes=b.axes) == b
    assert c.refusal("buried") == c.refusals["buried"]
    assert "the fast operator has no per-segment medium" in c.refusal("buried")
    # ...and everything the parent's row said around it is still said: the
    # `_replace(refusals=...)` REPLACE trap is what lost the galerkin row a
    # cell, so the merge is pinned rather than trusted.
    assert set(b.refusals) < set(c.refusals)
    assert all(c.refusals[k] == v for k, v in b.refusals.items())


def test_array_block_takes_the_hmatrix_row_but_for_its_own_solve_strategy():
    """Same guard, same served axes, a different structural decomposition.

    This test asserted `is HMatrixSolver.capabilities` — one shared object,
    no override — and was RIGHT while a row described only what a solver
    serves: these two serve identically. It became wrong the moment the row
    also described what a solver is MADE OF, because ArrayBlockSolver then
    reported its assembly as ACA, which is the one thing it is not
    (antennaknobs#1006). The class had carried no row of its own at all.

    So: still the parent's row in every served cell, and its own value on
    the axis the class exists for.
    """
    a, h = ArrayBlockSolver.capabilities, HMatrixSolver.capabilities
    assert a is not h
    assert a.axes["solve_strategy"] == ("element-block",)
    assert h.axes["solve_strategy"] == ("aca",)
    assert a._replace(axes=h.axes) == h  # every other cell, including refusals


# --------------------------------------------------------------------------
# 7. knot_feeds — the one axis that fails SILENTLY, so it is MEASURED
# --------------------------------------------------------------------------
#
# Every other cell in this module is cross-checked against a raise: ask for
# the thing, catch the refusal, match it to `capabilities.refusal(cell)`.
# `knot_feeds` has no raise behind it (momwire#611).  A family that resolves
# a `feeds` arclength to the nearest segment CENTRE does not complain — it
# answers half a segment away — so a declaration checked against "does it
# raise" would be checked against nothing at all.
#
# The probe is formulation-independent, which is what lets one loop cover
# nine roster entries and two off-roster families that share no basis, no
# testing and no port model.  Drive the centre KNOT of an even-segment
# dipole: the structure is symmetric about that knot, so |I| comes out
# symmetric about it if and only if the gap landed there.  Read through
# `currents_at_knots`, which every family has.
#
# Measured separation is twelve orders of magnitude — the honest families sit
# at ~1e-14 (round-off) and the snapping ones at ~2e-2 to ~1e-1 — so the two
# thresholds below are nowhere near each other and nothing sits between them.
#
# This is also where both knot_feeds refusals' ROUTE is executed (momwire#604
# class (2)): each of them sends the caller at "BSplineSolver or RazorSolver",
# and the True branch below is the demonstration that those two land a gap
# where it was asked for. `test_refusals_name_working_routes.py` scans RAISED
# messages and these are surfaced through `capabilities.refusal()` instead, so
# the check belongs here, with the axis it is about.

_KNOT_FED_WIRE = [(0.0, -0.25, 0.0), (0.0, 0.25, 0.0)]
_KNOT_FED_NSEGS = 12  # EVEN, so the wire's midpoint IS an interior knot
_KNOT_FED_ARC = 0.25  # that knot, metres from the first anchor


def _knot_feed_asymmetry(solver_class, **kwargs):
    """max||I| − |I| reversed| / max|I| for a dipole driven at its centre knot."""
    solver = solver_class(
        wires=[_KNOT_FED_WIRE],
        nsegs=_KNOT_FED_NSEGS,
        wavelength=WAVELENGTH,
        wire_radius=5e-4,
        feeds=[(0, _KNOT_FED_ARC, 1.0 + 0.0j)],
        **kwargs,
    )
    _z, alpha = solver.compute_impedance()
    mag = np.abs(np.asarray(solver.currents_at_knots(alpha)[0]))
    return float(np.max(np.abs(mag - mag[::-1])) / np.max(mag))


@pytest.mark.parametrize("basis", sorted(BASES))
def test_knot_feeds_declaration_matches_where_the_gap_actually_lands(basis):
    solver_class, basis_kwargs = BASES[basis]
    asymmetry = _knot_feed_asymmetry(solver_class, **basis_kwargs)
    if solver_class.capabilities.knot_feeds:
        assert asymmetry < 1e-10, (basis, asymmetry)
    else:
        # Not merely "some asymmetry": the gap is a WHOLE half-segment off,
        # which on this mesh is a percent-level distortion of the current.
        assert asymmetry > 1e-3, (basis, asymmetry)


def test_sg_knot_feeds_describes_the_point_gap_the_roster_can_build():
    """momwire#686. `Capabilities` is a CLASS attribute, so every cell is a
    statement about the class — but this one is only true per INSTANCE.

    `knot_feeds=True` holds under `feed_model="point"` (the default since
    momwire#654), where `feed_xi` carries the remainder. Under
    `feed_model="segment"` the same class snaps to a segment centre, and the
    class attribute still says True. Measured here rather than argued, on §7's
    own probe, so the gap is on the record as a number.

    It is not reachable through the seam, and that is the invariant this
    pins: `serve(deck, *, basis)` takes a basis NAME and no solver kwargs, and
    momwire#654 collapsed the roster to ONE Galerkin entry binding no
    `feed_model`. A future roster edit re-adding a `"segment"` spelling would
    hand `eznec/_serve.py`'s `knot_feeds` gate an instance the cell does not
    describe — the exact failure momwire#611 exists to prevent."""
    cls, kwargs = BASES["sinusoidal-galerkin"]
    assert cls.capabilities.knot_feeds is True

    # (1) the roster cannot construct the snapping instance
    assert "feed_model" not in kwargs, (
        "a roster entry binding feed_model would let the NEC-5 seam build an "
        "instance `knot_feeds=True` does not describe (momwire#686)"
    )
    assert not any("feed_model" in kw for _c, kw in BASES.values())

    # (2) and the instance it does not describe really does snap
    point = _knot_feed_asymmetry(cls, feed_model="point")
    segment = _knot_feed_asymmetry(cls, feed_model="segment")
    assert point < 1e-10, point
    assert segment > 1e-3, segment


# --------------------------------------------------------------------------
# 7b. centre_feeds — the same probe with one number changed (momwire#673)
# --------------------------------------------------------------------------
#
# `centre_feeds` fails silently for the same reason `knot_feeds` does, so it
# is measured the same way: the mirror of the probe above, with the segment
# count made ODD.
#
# That one change is what moves the target grid.  This wire is 0.5 m and the
# feed arclength is its midpoint; at 12 segments that midpoint IS an interior
# knot, and at 13 it is the CENTRE of the middle segment (0.25 / (0.5/13) =
# 6.5) and no knot is within half a cell of it.  So the same symmetry
# argument reads the other grid: |I| is symmetric about the midpoint if and
# only if the gap landed at the centre it was named.
#
# Measured, and the two probes are exact complements — every family misses
# exactly one of the two grids:
#
#     family                    odd (centre)   even (knot)
#     bspline, bspline-d1       ~1e-14         ~1e-14      lands both, snaps never
#     hmatrix, arrayblock       ~1e-14         ~1e-14
#     sinusoidal-galerkin       ~4e-15         ~4e-15
#     sinusoidal                ~1e-12         1.8e-01     snaps to the centre
#     pulse, harrington         ~2e-15         1.7e-01
#     razor-2p, razor-nec5      1.8e-01        2e-16       snaps to the knot
#
# The bars are the knot probe's, unchanged, because the separation is the
# same twelve orders: nothing measured sits between 1e-10 and 1e-3.

_CENTRE_FED_NSEGS = 13  # ODD, so the wire's midpoint is a segment CENTRE


def _centre_feed_asymmetry(solver_class, **kwargs):
    """§7's probe on an ODD mesh, where the midpoint is a segment centre."""
    solver = solver_class(
        wires=[_KNOT_FED_WIRE],
        nsegs=_CENTRE_FED_NSEGS,
        wavelength=WAVELENGTH,
        wire_radius=5e-4,
        feeds=[(0, _KNOT_FED_ARC, 1.0 + 0.0j)],
        **kwargs,
    )
    _z, alpha = solver.compute_impedance()
    mag = np.abs(np.asarray(solver.currents_at_knots(alpha)[0]))
    return float(np.max(np.abs(mag - mag[::-1])) / np.max(mag))


@pytest.mark.parametrize("basis", sorted(BASES))
def test_centre_feeds_declaration_matches_where_the_gap_actually_lands(basis):
    solver_class, basis_kwargs = BASES[basis]
    asymmetry = _centre_feed_asymmetry(solver_class, **basis_kwargs)
    if solver_class.capabilities.centre_feeds:
        assert asymmetry < 1e-10, (basis, asymmetry)
    else:
        # A whole half-segment off, same as the knot probe's failing side.
        assert asymmetry > 1e-3, (basis, asymmetry)


@pytest.mark.parametrize("basis", sorted(BASES))
def test_the_two_feed_axes_are_complements_not_duplicates(basis):
    """Every family lands at least one of the two grids, and the one family
    that snaps on each side is not the same family.

    This is the claim that makes `centre_feeds` a separate axis rather than
    `not knot_feeds`: `bspline` and `sinusoidal-galerkin` are True on BOTH,
    because they never snap at all and land wherever they were named. A
    single axis could not say that.
    """
    solver_class, basis_kwargs = BASES[basis]
    caps = solver_class.capabilities
    assert caps.knot_feeds or caps.centre_feeds, (
        f"{basis} declares it lands on neither grid, which would mean a gap "
        "it can never place where asked"
    )
    if caps.knot_feeds and caps.centre_feeds:
        # The strong claim: it really does land both, so it snaps to neither.
        assert _knot_feed_asymmetry(solver_class, **basis_kwargs) < 1e-10
        assert _centre_feed_asymmetry(solver_class, **basis_kwargs) < 1e-10


def test_razor_is_the_only_family_that_refuses_a_centre_gap():
    """The roster-wide shape of momwire#673, pinned so a new row that snaps
    to knots cannot arrive without either declaring it or failing here."""
    snapping = {
        basis
        for basis, (cls, _kw) in BASES.items()
        if not cls.capabilities.centre_feeds
    }
    assert snapping == {"razor-2p", "razor-nec5"}, snapping
    for basis in snapping:
        cls, _kw = BASES[basis]
        assert cls.capabilities.refusal("centre_feeds"), (
            f"{basis} refuses a centre gap without a declared reason"
        )


@pytest.mark.parametrize("solver_class", [PulseSolver, HarringtonSolver])
def test_the_off_roster_families_declare_the_snap_they_document(solver_class):
    """`PulseSolver._feed_basis_indices` says it in prose — "a delta gap lands
    on a segment and not on a knot, the opposite snap from the tent-basis
    solvers" — and `HarringtonSolver` inherits it.  Neither is in `BASES`, so
    neither can reach a seam today; the row is still checked, because the
    reason to declare a capability is that someone will read it before the
    roster changes rather than after.
    """
    assert not solver_class.capabilities.knot_feeds
    assert _knot_feed_asymmetry(solver_class) > 1e-3


def test_the_TESTING_is_what_separates_the_two_sinusoidal_families_here():
    """Same basis, same mesh, opposite `knot_feeds` — and the axis is testing.

    `SinusoidalGalerkinSolver` serves a knot gap because its pairing
    ⟨f_i, E_app⟩ collapses a delta at any point to the drive column −V·f_i(s₀)
    (momwire#648 made it evaluate there rather than at the segment centre it
    used to snap to). `SinusoidalSolver` cannot, ever: the match points ARE
    the segment centres, so a delta at a knot point-samples to nothing in
    every row and the RHS is the unexcited problem (momwire#177).

    So the refusal survives on exactly one of the two, and its sentence has to
    name the right failure — a reader told "δ·δ is undefined" (which is
    `feed_model="point"`'s problem, #212, a different thing) goes looking for
    a regularization that #212 §17 already ran to its dead end. Zero rows are
    not a regularization problem; they are the absence of an excitation, which
    is also why this one is permanent.
    """
    point = SinusoidalSolver.capabilities.refusal("knot_feeds")
    assert point
    assert "point-samples to nothing" in point and "#177" in point
    assert "undefined" not in point
    # The Galerkin family and every tent-basis family now say nothing at all,
    # which is what None means.
    for cls in (SinusoidalGalerkinSolver, BSplineSolver, RazorSolver):
        assert cls.capabilities.refusal("knot_feeds") is None, cls


# --------------------------------------------------------------------------
# 9. The compositional axes (antennaknobs#1006). The generated matrix's
#    `--check` already catches a row edit that skips regeneration; these
#    catch the three things regeneration would happily render.
# --------------------------------------------------------------------------


def test_every_declared_axis_and_value_is_in_the_vocabulary():
    """A typo renders as a column value rather than failing.

    `AXIS_VALUES` is data with no validator behind it — deliberately, the
    module contract forbids validation machinery — so this is where a row
    declaring `"galerkin "` or `"bspline3"` is caught. Without it the matrix
    would publish the typo and a panel would offer it as a choice.
    """
    for cls in _AXIS_CLASSES:
        for axis, values in cls.capabilities.axes.items():
            assert axis in AXIS_VALUES, f"{cls.__name__}: unknown axis {axis!r}"
            assert values, f"{cls.__name__}: {axis} declared empty"
            for v in values:
                assert v in AXIS_VALUES[axis], (
                    f"{cls.__name__}: {axis}={v!r} is not in the vocabulary "
                    f"{AXIS_VALUES[axis]}"
                )


def test_no_row_restates_a_derived_axis():
    """`ground_model` and `wire_position` come from `grounds` / `buried` /
    `contact` through `axes_for`, and a row declaring one would be a second
    source of truth that `axes_for` would then silently overwrite — the
    worst combination, because the row would read as authoritative and have
    no effect."""
    for cls in _AXIS_CLASSES:
        for axis in DERIVED_AXES:
            assert axis not in cls.capabilities.axes, (
                f"{cls.__name__} declares {axis}, which is derived"
            )


def test_axes_for_returns_declared_union_derived():
    """The single derivation, checked on a real row and on the empty one."""
    got = axes_for(BSplineSolver.capabilities)
    assert got["basis"] == frozenset({"bspline-1", "bspline-2"})
    assert got["ground_model"] == frozenset({"free", "pec", "refl-coef", "sommerfeld"})
    assert got["wire_position"] == frozenset({"above", "contact", "buried"})

    # The ~200-line prototype row of the design doc's §0.2: no declared axes
    # at all, but the derived pair still answers, because "free" and "above"
    # are universal. A consumer must be able to tell "not described" from
    # "described as nothing", and this is that distinction.
    empty = Capabilities(
        grounds=frozenset(),
        wire_loading=False,
        extended_kernel=False,
        junction_ports=False,
        node_gaps=False,
        per_wire_radius=False,
        singular_enrichment=False,
        refusals={},
    )
    bare = axes_for(empty)
    assert set(bare) == set(DERIVED_AXES)
    assert bare["ground_model"] == frozenset({"free"})
    assert bare["wire_position"] == frozenset({"above"})


def test_the_pairs_the_axes_exist_to_explain_differ_where_they_should():
    """The claim antennaknobs#1006 opens with, as a gate — and the one place
    it turns out to be an approximation.

    Two of the three pairs differ in exactly the one axis the pair exists to
    isolate. The sinusoidal pair differs in TWO, and that is not a
    mis-declaration: `SinusoidalSolver` REFUSES `feed_model="point"`
    (`_reject_point_feed_model`, momwire#212 — a zero-width gap sitting on a
    match point is undefined for point-matched collocation) where the
    Galerkin sibling serves it. So the second difference is FORCED by the
    first rather than independent of it.

    That is worth a gate rather than a footnote, because it is the first
    measured coupling in the product space: the reachable values of
    `feed_model` depend on `testing`, so a panel cannot treat the axes as
    freely combinable controls. #1006 asks "which cells are reachable, and
    why not the rest" — this is one answer, with a named reason.
    """
    for a, b, expected in (
        (BSplineSolver, HMatrixSolver, {"solve_strategy"}),
        (PulseSolver, HarringtonSolver, {"charge_support"}),
        # Coupled, see the docstring — NOT {"testing"}.
        (SinusoidalSolver, SinusoidalGalerkinSolver, {"testing", "feed_model"}),
    ):
        differ = {
            k
            for k in set(a.capabilities.axes) | set(b.capabilities.axes)
            if a.capabilities.axes.get(k) != b.capabilities.axes.get(k)
        }
        assert differ == expected, (
            f"{a.__name__} vs {b.__name__} differ in {sorted(differ)}, "
            f"expected {sorted(expected)}"
        )
