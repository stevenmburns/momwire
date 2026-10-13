"""A junction port at a CROSSING node is refused for its own reason --
momwire#849.

Both Galerkin trunks refused it with the ground CONTACT's sentence: "a
grounded node's voltage is pinned by the ground image" on bspline, and on
Sin-Galerkin the finite-ground sentence that ends "Use BSplineSolver" --
which refuses the same port. A crossing node has no contact image: the
crossing fill carries the current across it. What is unserved there is a
potential reference (momwire#813), and what serves the node is a series
source, `node_gaps` (momwire#1285). The refusal now says both.

Wording only: nothing that was served or refused changes sides. A CONTACT
node keeps its own sentence, word for word.
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from momwire import _below_interface  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.sinusoidal_galerkin import (  # noqa: E402
    _JUNCTION_PORTS_FINITE_GROUND_REFUSAL,
    SinusoidalGalerkinSolver,
)
from test_crossing_serve_524 import A_WIRE, SOIL_A, WL7, crossing_deck  # noqa: E402

FAMILIES = [
    (BSplineSolver, ValueError),
    (SinusoidalGalerkinSolver, NotImplementedError),
]


def _crossing(**drive):
    """One rod below the plane, one monopole above, joined at z = 0."""
    deck = crossing_deck(1)
    deck.pop("feeds")
    deck.update(drive)
    return deck


def _contact(**ground):
    """Two wires above the plane meeting IN it, nothing buried: a ground
    contact, where the image reason is the right one."""
    return dict(
        wires=[
            np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 5.0)]),
            np.array([(0.0, 0.0, 0.0), (3.0, 0.0, 4.0)]),
        ],
        n_per_edge_per_wire=[[8], [8]],
        junctions=[[(0, "start"), (1, "start")]],
        junction_ports=[(0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=A_WIRE,
        ground_z=0.0,
        **ground,
    )


def _refusal(cls, exc, deck):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(exc) as caught:
            cls(**deck).compute_impedance()
    return str(caught.value)


@pytest.mark.parametrize("cls, exc", FAMILIES)
def test_a_crossing_node_is_refused_for_what_is_unserved_there(cls, exc):
    said = _refusal(cls, exc, _crossing(junction_ports=[(0, 1 + 0j)]))
    assert said == _below_interface.crossing_junction_port_refusal(0)
    # what is actually unserved, and what serves the case
    assert "crossing node" in said and "potential reference" in said
    assert "node_gaps" in said and "momwire#1285" in said
    # not the contact's reason, and not a pointer at another refusal
    for stale in ("ground image", "pinned", "ground stake", "Use BSplineSolver"):
        assert stale not in said


@pytest.mark.parametrize("cls, exc", FAMILIES)
def test_the_series_source_it_points_at_is_served(cls, exc):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z, _currents = cls(
            **_crossing(node_gaps=[(1, "start", 1 + 0j)])
        ).compute_impedance()
    assert np.all(np.isfinite(z)) and np.all(np.real(z) > 0.0)


def test_a_contact_node_keeps_the_image_sentence_on_bspline():
    said = _refusal(
        BSplineSolver,
        ValueError,
        _contact(ground_eps=SOIL_A, ground_model="sommerfeld"),
    )
    assert said == (
        "junction 0 is both grounded and a junction port — a grounded node's "
        "voltage is pinned by the ground image, so it cannot also be a driven port"
    )


def test_a_contact_node_keeps_its_sentences_on_sin_galerkin():
    # over a finite ground: the family's finite-ground sentence, unchanged
    said = _refusal(
        SinusoidalGalerkinSolver,
        NotImplementedError,
        _contact(ground_eps=SOIL_A, ground_model="sommerfeld"),
    )
    assert said == _JUNCTION_PORTS_FINITE_GROUND_REFUSAL
    # over PEC: the node meets its own image (#151)
    said = _refusal(SinusoidalGalerkinSolver, ValueError, _contact())
    assert "connected to its own image" in said and "crossing" not in said
