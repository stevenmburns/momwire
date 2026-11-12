"""`momwire._mesh_walk` against the per-edge loops it replaces, bit for bit."""

import numpy as np
import pytest

from momwire import _mesh_walk as MW
from momwire._vec3 import norm3


def _bits(a):
    a = np.ascontiguousarray(a)
    return a.view(np.uint8).tobytes(), a.shape, a.dtype


def _mesh(rng, n_wires=60):
    """Wires of 1-4 edges: random, axis-aligned (a zero step component, which
    takes numpy linspace's other branch), tiny and long, negative coordinates."""
    polylines, per_edge = [], []
    for w in range(n_wires):
        n_edges = int(rng.integers(1, 5))
        pts = [rng.normal(size=3) * 10.0 ** rng.integers(-4, 3)]
        for _ in range(n_edges):
            step = rng.normal(size=3)
            kind = w % 4
            if kind == 1:
                step[rng.integers(0, 3)] = 0.0
            elif kind == 2:
                step[:2] = 0.0
            elif kind == 3:
                step *= 1e-7
            pts.append(pts[-1] + step)
        polylines.append(np.array(pts))
        per_edge.append([int(rng.integers(1, 30)) for _ in range(n_edges)])
    return polylines, per_edge


def _ref_knots(polylines, per_edge, subdiv):
    out = []
    for polyline, npe in zip(polylines, per_edge):
        parts = []
        for i, n_e in enumerate(npe):
            seg = np.linspace(polyline[i], polyline[i + 1], n_e * subdiv + 1)
            parts.append(seg if i == 0 else seg[1:])
        out.append(np.vstack(parts))
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("subdiv", [1, 3])
def test_wire_knots_and_arcs_are_the_loops_bits(seed, subdiv):
    polylines, per_edge = _mesh(np.random.default_rng(seed))
    ref = _ref_knots(polylines, per_edge, subdiv)
    knots, offs = MW.wire_knots(polylines, per_edge, subdiv)
    assert offs.tolist() == np.cumsum([0] + [k.shape[0] for k in ref]).tolist()
    assert _bits(knots) == _bits(np.concatenate(ref))
    arcs = MW.wire_arcs(knots, offs)
    ref_arcs = [
        np.concatenate([[0.0], np.cumsum(norm3(np.diff(k, axis=0)))]) for k in ref
    ]
    assert _bits(arcs) == _bits(np.concatenate(ref_arcs))


def test_segmented_cumsum_is_each_slices_cumsum():
    rng = np.random.default_rng(7)
    lengths = rng.integers(0, 40, size=200)
    offs = np.concatenate([[0], np.cumsum(lengths)])
    x = rng.normal(size=int(offs[-1])) * 10.0 ** rng.integers(-8, 8, size=int(offs[-1]))
    got = MW.segmented_cumsum(x, offs)
    ref = np.concatenate([np.cumsum(x[a:b]) for a, b in zip(offs[:-1], offs[1:])])
    assert _bits(got) == _bits(ref)
