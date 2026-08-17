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

from momwire import (
    ArrayBlockSolver,
    BSplineSolver,
    Capabilities,
    HMatrixSolver,
    RazorSolver,
    SinusoidalGalerkinSolver,
    SinusoidalSolver,
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
# 2. RazorSolver — refuses the whole row
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cell,kwarg",
    [
        ("pec", {"ground_z": 0.0}),
        ("refl-coef", {"ground_eps": 10 - 1j}),
        ("sommerfeld", {"ground_model": "sommerfeld"}),
        ("junction_ports", {"junction_ports": [0]}),
        ("node_gaps", {"node_gaps": [(0, "end", 1.0 + 0j)]}),
        ("extended_kernel", {"extended_kernel": True}),
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


def test_razor_per_wire_radius_refuses():
    wires, npe = _wire()
    reason = RazorSolver.capabilities.refusal("per_wire_radius")
    assert reason
    with pytest.raises(NotImplementedError):
        RazorSolver(
            wires=wires,
            n_per_edge_per_wire=npe,
            wire_radius=[0.001, 0.0015],
            wavelength=WAVELENGTH,
        )


@pytest.mark.parametrize(
    "cell,kwarg",
    [
        ("wire_loading", {"wire_conductivity": 1e7}),
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


def test_razor_served_row_is_empty():
    c = RazorSolver.capabilities
    assert c.grounds == frozenset()
    assert not any(
        [
            c.wire_loading,
            c.extended_kernel,
            c.junction_ports,
            c.node_gaps,
            c.per_wire_radius,
            c.singular_enrichment,
        ]
    )


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
# 6. HMatrixSolver / ArrayBlockSolver inherit BSplineSolver's row unchanged
#    (the survey found no user-facing capability difference: enrichment
#    just forces their dense-path fallback rather than being refused)
# --------------------------------------------------------------------------


def test_hmatrix_inherits_bspline_capabilities():
    assert HMatrixSolver.capabilities is BSplineSolver.capabilities


def test_array_block_inherits_bspline_capabilities():
    assert ArrayBlockSolver.capabilities is BSplineSolver.capabilities
