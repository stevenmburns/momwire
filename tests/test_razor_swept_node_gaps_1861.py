"""The swept Y counts razor's node gaps (#1861).

`RazorSolver._port_count` returned only the gap feeds, though razor serves
node gaps (#603). `compute_y_matrix_swept` sizes its block from it, so a
design whose ports are node gaps got an empty block (one gap: each 1x1 Y
broadcast into a 0x0 slot, silently) or a broadcast error (two gaps). The
single-point Y never read the count and was always right, so it is the
oracle here: the swept Y must be it, per k, bit for bit and at its shape.
"""

import numpy as np
import pytest

from momwire import BSplineSolver, RazorSolver, SinusoidalGalerkinSolver

WL = 299.792458 / 27.0
L = 5.2


def _three_arm(n_gaps):
    """A straight wire in three pieces; node gaps at the first one or both
    junctions, and no gap feeds."""
    a, b = L / 4, L / 2
    wires = [
        np.array([(0.0, 0.0, 0.0), (0.0, 0.0, a)]),
        np.array([(0.0, 0.0, a), (0.0, 0.0, b)]),
        np.array([(0.0, 0.0, b), (0.0, 0.0, L)]),
    ]
    gaps = [(0, "end", 1.0 + 0j), (1, "end", 1.0 + 0j)][:n_gaps]
    return dict(
        wires=wires,
        n_per_edge_per_wire=[[12], [12], [24]],
        junctions=[[(0, "end"), (1, "start")], [(1, "end"), (2, "start")]],
        feeds=[],
        node_gaps=gaps,
        wavelength=WL,
        wire_radius=1e-3,
    )


@pytest.mark.parametrize("n_gaps", [1, 2])
def test_swept_y_is_the_per_k_y_with_node_gaps(n_gaps):
    kw = _three_arm(n_gaps)
    k0 = RazorSolver(**kw).k
    ks = k0 * np.array([0.9, 1.0, 1.1])
    swept = RazorSolver(**kw).compute_y_matrix_swept(ks)
    assert swept.shape == (3, n_gaps, n_gaps)
    for i, k in enumerate(ks):
        one = RazorSolver(**{**kw, "wavelength": 2 * np.pi / k}).compute_y_matrix()
        assert one.shape == (n_gaps, n_gaps)
        np.testing.assert_allclose(swept[i], one, rtol=1e-12, atol=0)


@pytest.mark.parametrize("cls", [BSplineSolver, SinusoidalGalerkinSolver, RazorSolver])
@pytest.mark.parametrize("n_gaps", [1, 2])
def test_every_family_counts_the_ports_its_y_has(cls, n_gaps):
    """The count the swept path sizes by and the ports the single-point Y
    returns are two statements of one fact; #1861 was the two drifting
    apart in one family. Held for every family that serves node gaps."""
    s = cls(**_three_arm(n_gaps))
    assert s._port_count() == s.compute_y_matrix().shape[0] == n_gaps
