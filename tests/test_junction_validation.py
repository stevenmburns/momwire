"""Junction groups must name geometrically coincident wire-ends (momwire#522).

The #518 postmortem: a half-re-indexed ``junctions=`` list silently welded
wire ends 60-100 m apart and left the one wire whose membership was lost
electrically absent, and the resulting clean first-order convergence to a
limit 2 % off was filed as a solver bias.  These gates make that spec a loud
construction-time refusal.  The must-refuse fixture IS the ladder's exact
junction list; the must-serve twin is the same model with the indices fixed.

Tolerance contract: refusal fires only beyond max(1e-5 m, 1e-3 of the
shortest terminal segment among the group's members), so the deck front's
1e-6 m node grid (deck/_polylines._NODE_EPS quantization, worst case
~1.8e-6 m of Euclidean fuzz between fused ends) can never produce one.
"""

import numpy as np
import pytest

from momwire.bspline import BSplineSolver
from momwire.sinusoidal import SinusoidalSolver

WAVELENGTH = 299.8 / 0.0005  # Roy's coupled-loop model: 500 Hz, metres
RADIUS = 0.005

# The ladder's split-wire geometry: wire 1 of the deck cut at the z=20 gap
# knot, the loop and the stub following, each a 2-anchor polyline.
SPLIT_WIRES = [
    [(20, -40, 300), (20, -40, 20)],
    [(20, -40, 20), (20, -40, 0)],
    [(40, -40, 0), (40, 40, 0)],
    [(40, 40, 0), (-40, 40, 0)],
    [(-40, 40, 0), (-40, -40, 0)],
    [(-40, -40, 0), (20, -40, 0)],
    [(20, -40, 0), (40, -40, 0)],
]
SPLIT_NSEG = [[14], [1], [4], [4], [4], [3], [1]]

# ladder.py's actual list: entries 1 and 2 carry pre-split wire indices, so
# they name ends 60-100 m apart, and wire 6 (the stub) is in no group.
BUGGY_JUNCTIONS = [
    [(0, "end"), (1, "start")],
    [(1, "end"), (5, "start"), (4, "end")],
    [(5, "end"), (2, "start")],
    [(2, "end"), (3, "start")],
    [(3, "end"), (4, "start")],
]

CORRECTED_JUNCTIONS = [
    [(0, "end"), (1, "start")],
    [(1, "end"), (5, "end"), (6, "start")],
    [(6, "end"), (2, "start")],
    [(2, "end"), (3, "start")],
    [(3, "end"), (4, "start")],
    [(4, "end"), (5, "start")],
]


def _bspline(junctions, **kw):
    return BSplineSolver(
        wires=[np.array(w, dtype=float) for w in SPLIT_WIRES],
        n_per_edge_per_wire=SPLIT_NSEG,
        feeds=[],
        node_gaps=[(0, "end", 0j)],
        junctions=junctions,
        degree=kw.pop("degree", 2),
        wire_radius=RADIUS,
        wavelength=WAVELENGTH,
        **kw,
    )


def test_bspline_refuses_the_518_ladder_spec():
    """The postmortem's must-refuse fixture: the exact buggy junction list."""
    with pytest.raises(ValueError, match=r"junction 1:.*do not coincide"):
        _bspline(BUGGY_JUNCTIONS)


def test_bspline_error_names_the_offending_ends_and_distance():
    with pytest.raises(ValueError) as exc:
        _bspline(BUGGY_JUNCTIONS)
    msg = str(exc.value)
    assert "wire 5 start" in msg
    assert "wire 1 end" in msg
    assert "momwire#522" in msg


def test_bspline_serves_the_corrected_spec():
    """The must-serve twin: same model, indices fixed."""
    for degree in (1, 2):
        _bspline(CORRECTED_JUNCTIONS, degree=degree)


def test_sinusoidal_refuses_and_serves_the_same_pair():
    def build(junctions):
        return SinusoidalSolver(
            wires=[np.array(w, dtype=float) for w in SPLIT_WIRES],
            n_per_edge_per_wire=SPLIT_NSEG,
            feeds=[(0, 270.0, 0j)],
            junctions=junctions,
            wire_radius=RADIUS,
            wavelength=WAVELENGTH,
        )

    with pytest.raises(ValueError, match=r"junction 1:.*do not coincide"):
        build(BUGGY_JUNCTIONS)
    build(CORRECTED_JUNCTIONS)


def test_deck_grid_fuzz_is_accepted():
    """Ends the deck front fused onto its 1e-6 m grid may differ by up to
    ~1.8e-6 m Euclidean; the validator must never refuse them."""
    fuzz = np.array([1e-6, -1e-6, 1e-6])
    wires = [
        np.array([(0.0, 0.0, 10.0), (0.0, 0.0, 0.0)]),
        np.array([np.array([0.0, 0.0, 0.0]) + fuzz, (10.0, 0.0, 0.0)]),
    ]
    BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=[[4], [4]],
        feeds=[(0, 5.0, 1.0 + 0j)],
        junctions=[[(0, "end"), (1, "start")]],
        degree=1,
        wire_radius=1e-3,
        wavelength=100.0,
    )


def test_fine_mesh_keeps_the_absolute_floor():
    """1e-3 of a millimetre-scale terminal segment would undercut the deck
    grid; the 1e-5 m floor must hold there (accept 2e-6, refuse 1e-3)."""

    def build(gap):
        wires = [
            np.array([(0.0, 0.0, 0.01), (0.0, 0.0, 0.0)]),
            np.array([(0.0, gap, 0.0), (0.01, 0.0, 0.0)]),
        ]
        return BSplineSolver(
            wires=wires,
            n_per_edge_per_wire=[[10], [10]],  # 1 mm segments
            feeds=[(0, 0.005, 1.0 + 0j)],
            junctions=[[(0, "end"), (1, "start")]],
            degree=1,
            wire_radius=1e-4,
            wavelength=10.0,
        )

    build(2e-6)
    with pytest.raises(ValueError, match="do not coincide"):
        build(1e-3)


def test_legal_shapes_still_construct():
    """1-entry groups (issue #172) and a closed loop's own two ends."""
    loop = np.array(
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0)], dtype=float
    )
    BSplineSolver(
        wires=[loop],
        n_per_edge_per_wire=[[2, 2, 2, 2]],
        feeds=[(0, 0.5, 1.0 + 0j)],
        junctions=[[(0, "start"), (0, "end")]],
        degree=1,
        wire_radius=1e-3,
        wavelength=100.0,
    )
    stick = np.array([(0, 0, 0), (0, 0, 1.0)])
    BSplineSolver(
        wires=[stick],
        n_per_edge_per_wire=[[4]],
        feeds=[(0, 0.5, 1.0 + 0j)],
        junctions=[[(0, "start")]],  # 1-entry group: trivially coincident
        degree=1,
        wire_radius=1e-3,
        wavelength=100.0,
    )
