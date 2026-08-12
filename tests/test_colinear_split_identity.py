"""The colinear-split feed identity (#300): a straight wire fed at L/4
and the same wire spelled as a colinear 3-wire split — with the fed
segment as its own 1-segment bridge wire junctioned at both ends — are
electrically the same antenna, so the two spellings must agree within
small mesh terms.

History: antennaknobs' #872 phase-1 study read every momwire basis ~15 Ω
off in X on the bridge spelling and filed it as this solver defect
("the feed-local basis never enriches"). The momwire-level repro
dissolved it: the study's momwire rows solved the adapter-default 0.5 mm
wire against 1 mm reference decks, and at an off-center feed the radius
moves X while barely touching R. At a consistent radius the spellings
agree at ANY radius — which is exactly what this module pins, at both
radii the confusion straddled, so a real bridge-idiom defect can never
hide behind a radius story again (nor vice versa).

Geometry: L = 5.2 m at 27 MHz (wavelength ≈ 11.1 m), free space, feed at
L/4 — asymmetric so dZ/ds ≠ 0 and feed-model errors are visible.
"""

import numpy as np
import pytest

from momwire import BSplineSolver

L = 5.2
WAVELENGTH = 299.792458 / 27.0
N = 64  # pitch L/N; identity gaps measured 0.4-1.1 Ohm here, march-term sized


def _z_single(radius, degree):
    s = BSplineSolver(
        wires=[np.array([(0.0, 0.0, 0.0), (0.0, 0.0, L)])],
        n_per_edge_per_wire=[[N]],
        feeds=[(0, L / 4, 1.0 + 0j)],
        wavelength=WAVELENGTH,
        wire_radius=radius,
        degree=degree,
    )
    z, _ = s.compute_impedance()
    return complex(np.atleast_1d(z)[0])


def _z_bridge(radius, degree, bridge_nseg=1):
    d = L / N
    a, b = L / 4 - d / 2, L / 4 + d / 2
    s = BSplineSolver(
        wires=[
            np.array([(0.0, 0.0, 0.0), (0.0, 0.0, a)]),
            np.array([(0.0, 0.0, a), (0.0, 0.0, b)]),
            np.array([(0.0, 0.0, b), (0.0, 0.0, L)]),
        ],
        n_per_edge_per_wire=[[round(a / d)], [bridge_nseg], [round((L - b) / d)]],
        feeds=[(1, d / 2, 1.0 + 0j)],
        junctions=[[(0, "end"), (1, "start")], [(1, "end"), (2, "start")]],
        wavelength=WAVELENGTH,
        wire_radius=radius,
        degree=degree,
    )
    z, _ = s.compute_impedance()
    return complex(np.atleast_1d(z)[0])


@pytest.mark.parametrize("radius", [1e-3, 5e-4])
@pytest.mark.parametrize("degree", [1, 2])
def test_one_segment_bridge_matches_single_wire(radius, degree):
    """The #300 configuration itself: gap port on a 1-segment wire whose
    both ends are junction knots. The historical false reading was 15 Ω
    off in X; the identity holds to ~1 Ω at this mesh."""
    zs = _z_single(radius, degree)
    zb = _z_bridge(radius, degree)
    assert abs(zb - zs) < 2.0, f"single {zs:.3f} vs bridge {zb:.3f}"


def test_enriched_bridge_stays_on_the_identity():
    """#300's open question: enriching the feed-local basis (3-segment
    bridge) must not move the reading — measured < 0.6 Ω at this mesh."""
    zs = _z_single(1e-3, 2)
    zb3 = _z_bridge(1e-3, 2, bridge_nseg=3)
    assert abs(zb3 - zs) < 2.0, f"single {zs:.3f} vs 3-seg bridge {zb3:.3f}"


def test_both_spellings_track_the_radius_together():
    """The radius term that masqueraded as the bridge defect: halving the
    radius moves X by ~15 Ω on BOTH spellings equally. Pins the moved
    distance (the anomaly's size) and the spellings' agreement after the
    move, so neither a radius regression nor a bridge regression can
    cancel the other."""
    dz_single = _z_single(5e-4, 2) - _z_single(1e-3, 2)
    dz_bridge = _z_bridge(5e-4, 2) - _z_bridge(1e-3, 2)
    assert -17.0 < dz_single.imag < -13.0
    assert abs(dz_bridge - dz_single) < 1.0
