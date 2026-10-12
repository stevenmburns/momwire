"""The warm path's vectorised glue answers what the per-element code did.

momwire#1445 cut the fixed, per-call overhead of a small resident solve by
replacing Python loops over nodes and segments with whole-array numpy. Each
rewrite here claims BIT-identity with the loop it replaced, not closeness, so
each is held to that loop -- kept below as the reference, verbatim -- as
uint64, on random wires and on the values most likely to split a rounding
(exact half-tolerance coordinates, negative zero, the authored ends).
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire.deck._nec5 import Nec5Wire
from momwire.eznec import _serve


def _reference_points(wire: Nec5Wire) -> list[tuple[float, float, float]]:
    """``node_points`` as it stood before momwire#1445."""
    n = wire.segment_count
    a = np.asarray(wire.end1, dtype=float)
    b = np.asarray(wire.end2, dtype=float)
    points = [tuple(a + (b - a) * (k / n)) for k in range(n + 1)]
    points[0] = wire.end1
    points[-1] = wire.end2
    return points


def _bits(points) -> np.ndarray:
    return np.asarray(points, dtype=np.float64).view(np.uint64)


def _wires(seed: int, count: int) -> list[Nec5Wire]:
    rng = np.random.default_rng(seed)
    wires = []
    for tag in range(1, count + 1):
        end1 = tuple(float(v) for v in rng.uniform(-50.0, 50.0, 3))
        end2 = tuple(float(v) for v in rng.uniform(-50.0, 50.0, 3))
        wires.append(
            Nec5Wire(
                tag=tag,
                segment_count=int(rng.integers(1, 500)),
                end1=end1,
                end2=end2,
                radius=0.001,
            )
        )
    return wires


EDGE_WIRES = [
    # half-tolerance multiples: where round-half-even is the whole question
    Nec5Wire(
        tag=1,
        segment_count=4,
        end1=(0.0, 0.0, 0.0),
        end2=(2.5e-6, -1.5e-6, 4e-6),
        radius=0.001,
    ),
    # negative zero at both ends and through the middle
    Nec5Wire(
        tag=2,
        segment_count=3,
        end1=(-0.0, -0.0, 1.0),
        end2=(-0.0, 0.0, -1.0),
        radius=0.001,
    ),
    # a single segment: the grid is the two authored ends and nothing else
    Nec5Wire(
        tag=3, segment_count=1, end1=(1.0, 2.0, 3.0), end2=(0.1, 0.2, 0.3), radius=0.001
    ),
    # a = b + tiny: (b - a) * 1.0 + a is not always bitwise b
    Nec5Wire(
        tag=4,
        segment_count=7,
        end1=(0.1, 0.7, 1e-300),
        end2=(0.3, -0.7, 9.9e3),
        radius=0.001,
    ),
    # far enough out that int64 keys would overflow: the per-node fallback
    Nec5Wire(
        tag=5,
        segment_count=5,
        end1=(1e14, 0.0, 0.0),
        end2=(-1e14, 1.0, 2.0),
        radius=0.001,
    ),
]


@pytest.mark.parametrize(
    "wire", EDGE_WIRES + _wires(1445, 40), ids=lambda w: f"tag{w.tag}"
)
def test_node_points_are_the_per_node_loop_bit_for_bit(wire):
    got = _serve.node_points(wire)
    want = _reference_points(wire)
    assert len(got) == len(want)
    np.testing.assert_array_equal(_bits(got), _bits(want))
    # the element types too: the authored ends stay authored, the interior
    # stays numpy, exactly as before
    assert got[0] is wire.end1 and got[-1] is wire.end2
    assert [type(c) for p in got for c in p] == [type(c) for p in want for c in p]


@pytest.mark.parametrize(
    "wire", EDGE_WIRES + _wires(1446, 40), ids=lambda w: f"tag{w.tag}"
)
def test_node_keys_are_node_key_on_every_point(wire):
    points, keys = _serve._wire_nodes(wire)
    assert keys == [_serve._node_key(point) for point in _reference_points(wire)]
    assert all(type(c) is int for key in keys for c in key)
    np.testing.assert_array_equal(_bits(points), _bits(_reference_points(wire)))
