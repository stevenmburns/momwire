"""A deck that lists the same wire twice refuses by name — momwire#1042.

qantenna/airplane.nec carries GW 116 and GW 117 with identical endpoints in
opposite directions. Each basis on one has an identical, sign-flipped twin on
the other, so the operator is singular at any mesh: NEC-5's column-order fill
reports `Singular matrix`, the row-order engines pivot on rounding residue and
answer differently build to build, and BSplineSolver through the portal
answered 89.103-75.551j. The duplicate is refused at construction on every
family, naming both wires.

What must NOT refuse: a bundle that shares a segment with other wires (a
radial screen's coincident rises, momwire#524's fan widening, which
BSplineSolver serves and razor refuses segment by segment, #846), and two
merely CLOSE wires.
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from momwire._wire_spec import refuse_duplicated_wires  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.pulse import PulseSolver  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402
from momwire.sinusoidal import SinusoidalSolver  # noqa: E402
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver  # noqa: E402
from test_crossing_serve_524 import fan_rise_deck  # noqa: E402

FAMILIES = [
    PulseSolver,
    SinusoidalSolver,
    SinusoidalGalerkinSolver,
    RazorSolver,
    BSplineSolver,
]

# airplane.nec's GW 115-119, a closed corner of the wire grid with GW 116
# and GW 117 the same span run both ways. Free space, as the deck is.
A = (5.42580, -0.844875, 2.80628)
B = (5.41486, -3.08670, 2.80627)
C = (6.32782, -3.08619, 2.80627)
D = (6.28484, -4.66354, 2.80627)
E = (6.30227, -0.883203, 2.80563)


def _airplane_corner(dup=True):
    wires = [(E, C), (A, B), (C, B), (B, C), (D, B), (C, D)]
    if not dup:
        del wires[3]
    return dict(
        wires=[np.array(w) for w in wires],
        n_per_edge_per_wire=[[3] for _ in wires],
        wavelength=299792458.0 / 7.5e6,
        wire_radius=0.01,
        feeds=[(0, None, 1 + 0j)],
    )


@pytest.mark.parametrize("cls", FAMILIES, ids=lambda c: c.__name__)
def test_the_same_wire_twice_refuses_at_construction(cls):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError) as exc:
            cls(**_airplane_corner())
    msg = str(exc.value)
    assert msg.startswith("wires 2 and 3 are the same wire listed twice")
    assert "in reverse order" in msg
    assert "momwire#1042" in msg


def test_the_deck_without_the_copy_constructs():
    """The refusal is keyed on the copy: drop GW 117 and the corner builds."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        BSplineSolver(**_airplane_corner(dup=False))


def test_a_copy_in_the_same_direction_refuses_without_reverse():
    with pytest.raises(ValueError) as exc:
        refuse_duplicated_wires([np.array([A, B]), np.array([A, B])])
    assert "reverse" not in str(exc.value)


def test_a_bent_copy_refuses_and_a_partial_overlap_does_not():
    bent = np.array([A, B, C])
    with pytest.raises(ValueError):
        refuse_duplicated_wires([bent, bent[::-1].copy()])
    # Shares the A-B span only: a bundle-shaped geometry, not a duplicate.
    refuse_duplicated_wires([bent, np.array([A, B, D])])


def test_close_is_not_coincident():
    """Exact to the nanometre, like razor's bundle check: 1 um apart serves."""
    shifted = np.array([A, B]) + np.array([0.0, 0.0, 1e-6])
    refuse_duplicated_wires([np.array([A, B]), shifted])


@pytest.mark.parametrize("cls", [BSplineSolver, RazorSolver], ids=lambda c: c.__name__)
def test_the_bundle_still_constructs(cls):
    """momwire#524's fan: four radials whose rises coincide. Whole wires
    differ, so this check stays out of the way (razor refuses it at the
    SOLVE, by its own bundle sentence, #846)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cls(
            **{
                k: v
                for k, v in fan_rise_deck().items()
                if k not in ("ground_z", "ground_eps", "ground_model")
            }
        )
