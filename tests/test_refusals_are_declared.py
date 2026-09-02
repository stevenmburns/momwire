"""No constructor may refuse what the matrix would not have predicted (#396).

`Capabilities` DESCRIBES; every constructor and solve-time raise stays the
authoritative check. That division only works in one direction, though: a row
that claims more than the code serves is a lie a consumer discovers by
crashing. `tests/test_capabilities.py` closes the half that asks "does the
declared refusal actually raise". This file closes the other half, which is
the one that had holes in it — **does every raise have a declaration behind
it**.

The probe is deliberately geometric. For each solver class, a handful of tiny
decks spanning the geometry axes — free space, a wire clear of the plane, a
wire END in the plane (ground CONTACT), a wire wholly BELOW it, and a wire
CROSSING it mid-span — under each of the three grounds. For every one of them
exactly one of two things must be true:

* `capabilities.refusal(*cells)` is None and the deck constructs AND solves;
* it is a sentence, the deck raises, and the raised message ENDS WITH that
  sentence.

Three contradictions were found by writing it, all of them refusals raised
outside the matrix: `PulseSolver` (and `HarringtonSolver`, which shares the
check) refused ground contact under every ground while its row was silent
about contact; `HMatrixSolver` / `ArrayBlockSolver` refused buried decks while
inheriting a `BSplineSolver` row that claimed to serve them; and
`RazorSolver`'s whole buried refusal surface — its own no-buried-fill
sentence, the shared crossing and no-lower-medium sentences — appeared nowhere
in its row.

ENDS WITH, not equals
---------------------
A raise names the offending wire and the numbers that made it offending
("wire 0 lies wholly below the ground plane (min z = -1 < ground_z = 0), and
…"); a matrix cell cannot, because it is written before any deck exists. So
the tree's shape is a per-deck PREAMBLE followed by the constant reason, and
the constant reason is what the row declares — which is also what makes the
recorded reasons quotable in `docs/capability-matrix.md` rather than
rendering as "wire <wire> … z runs 0 to 0". The equality that matters is
between the row and the reason; the preamble is the deck talking.

What is deliberately NOT in scope
---------------------------------
Two refusals in the tree are kwarg-shaped rather than geometry-shaped, and
they stay out of the matrix rather than out of honesty:

* `SinusoidalSolver(feed_model="point")` (momwire#212). `feed_model` is not
  an axis any row declares and momwire#654 removed the second spelling from
  the roster, so no consumer picks it by name — there is nothing for a
  capability cell to answer. It is a bad argument value, which is what a
  `NotImplementedError` naming the working spelling is for.
* a `node_gaps` entry at a GROUNDED junction (`bspline.py`, momwire#151).
  This one IS expressible as a combination key, and declaring it would still
  be wrong: it depends on WHICH junction of an otherwise-served deck the
  caller addressed, and a class-level cell saying "BSplineSolver refuses node
  gaps over a ground" is false — a gap at an ungrounded junction over the
  same ground is served. A row describes the class; this is a property of the
  deck.

Both are covered where they belong, by `tests/test_capabilities.py` and
`tests/test_node_gaps.py`.
"""

import numpy as np
import pytest

from momwire.harrington import HarringtonSolver
from momwire.pulse import PulseSolver

from momwire import (
    ArrayBlockSolver,
    BSplineSolver,
    HMatrixSolver,
    RazorSolver,
    SinusoidalGalerkinSolver,
    SinusoidalSolver,
)

WL = 10.0
N_PER_EDGE = [[4]]

CLASSES = (
    BSplineSolver,
    HMatrixSolver,
    ArrayBlockSolver,
    SinusoidalSolver,
    SinusoidalGalerkinSolver,
    RazorSolver,
    PulseSolver,
    HarringtonSolver,
)

# The three grounds the matrix spells, in the kwargs each solver reads. PEC is
# not a `ground_model` value — it is `ground_z` with no `ground_eps`, which is
# why `grounds` has three tokens and the constructor has two knobs.
GROUNDS = {
    "pec": {"ground_z": 0.0},
    "refl-coef": {
        "ground_z": 0.0,
        "ground_eps": 13.0 - 1.0j,
        "ground_model": "refl-coef",
    },
    "sommerfeld": {
        "ground_z": 0.0,
        "ground_eps": 13.0 - 1.0j,
        "ground_model": "sommerfeld",
    },
}


def _horizontal(z):
    return [np.array([(0.0, -1.0, z), (0.0, 1.0, z)])]


def _vertical(z0, z1):
    return [np.array([(0.0, 0.0, z0), (0.0, 0.0, z1)])]


# (probe name, wires, ground token or None, the cells the deck exercises).
#
# `crossing` is asked as ("buried", "crossing") under one ground only: the
# mid-span reading happens before any solver looks at what is under the plane,
# so the answer does not vary with the ground column and paying for three
# would buy nothing.
_PROBES = [("free space", _horizontal(1.0), None, ())]
for _g in GROUNDS:
    _PROBES += [
        (f"clear of the plane / {_g}", _horizontal(1.0), _g, (_g,)),
        (f"contact / {_g}", _vertical(0.0, 2.0), _g, ("contact", _g)),
        (f"buried / {_g}", _horizontal(-1.0), _g, ("buried", _g)),
    ]
_PROBES.append(
    (
        "crossing / sommerfeld",
        _vertical(-1.0, 2.0),
        "sommerfeld",
        ("buried", "crossing"),
    )
)

_CASES = [(cls, *probe) for cls in CLASSES for probe in _PROBES]
_IDS = [f"{cls.__name__}-{name}" for cls, name, *_ in _CASES]


@pytest.mark.parametrize("cls,name,wires,ground,cells", _CASES, ids=_IDS)
def test_no_solver_refuses_what_its_row_did_not_predict(
    cls, name, wires, ground, cells
):
    expected = cls.capabilities.refusal(*cells)
    kwargs = dict(
        wires=wires,
        n_per_edge_per_wire=N_PER_EDGE,
        wavelength=WL,
        wire_radius=1e-3,
        **(GROUNDS[ground] if ground else {}),
    )

    if expected is None:
        # Solved, not merely constructed: `HMatrixSolver`'s buried refusal is
        # in `build_hmatrix`, so a construction-only probe would have called
        # its contradiction served.
        z, _ = cls(**kwargs).compute_impedance()
        assert np.isfinite(z.real) and np.isfinite(z.imag)
        return

    with pytest.raises((ValueError, NotImplementedError)) as exc:
        cls(**kwargs).compute_impedance()
    raised = str(exc.value)
    assert raised.endswith(expected), (
        f"{cls.__name__} refuses the {name!r} deck with a message its row did "
        f"not predict.\n\n  raised:   ...{raised[-220:]!r}\n\n  declared "
        f"({', '.join(cells)}): ...{expected[-220:]!r}"
    )


def test_the_probe_set_exercises_both_outcomes_on_every_class():
    """A guard on the guard: a probe set that refused everything, or served
    everything, would pass the test above while measuring nothing."""
    for cls in CLASSES:
        answers = {
            cls.capabilities.refusal(*cells) is None for _n, _w, _g, cells in _PROBES
        }
        assert answers == {True, False}, cls.__name__
