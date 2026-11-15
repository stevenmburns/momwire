"""momwire#1943 lever 5: the mesh's knot walk, once per structure.

The nec2 portal walks the same mesh at every frequency: `element_currents`
(the moments behind every RP table) and the segment match each call
`_mesh_walk.wire_knots`. On a prepared mesh's frozen polylines the walk is
now held by identity (`_identity_memo`); the gates are one walk across a
four-frequency sweep, the arrays read-only (a reader that writes into them
fails loudly), and the element currents BIT-equal to a sweep that walks at
every step.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _identity_memo as IM
from momwire import _mesh_walk
from momwire.dropin.nec2c import _portal

from test_sector_map_once_1929 import DECK


def _elements():
    deck = _portal.parse_deck(DECK)
    solver = _portal.DeckSolver(deck, basis="sinusoidal")
    (group,) = deck.groups
    out = []
    for f in group.freqs_mhz:
        r = solver.solve_group(group, f)
        out.append(solver.current_elements(r))
    return out


def test_one_walk_across_the_sweep_and_the_same_bits():
    _mesh_walk._KNOTS_MEMO.clear()
    misses = _mesh_walk._KNOTS_MEMO.misses
    hits = _mesh_walk._KNOTS_MEMO.hits
    memoised = _elements()
    assert _mesh_walk._KNOTS_MEMO.misses - misses == 1
    # Four element walks and the one segment match, all on one structure.
    assert _mesh_walk._KNOTS_MEMO.hits - hits >= 4
    nodes = [e[2] for e in memoised]
    assert all(n is nodes[0] for n in nodes)
    assert not nodes[0].flags.writeable
    with pytest.raises(ValueError):
        nodes[0][0, 0] = 1.0

    _mesh_walk._KNOTS_MEMO.clear()
    size = _mesh_walk._KNOTS_MEMO._size
    _mesh_walk._KNOTS_MEMO._size = 0
    try:
        fresh = _elements()
    finally:
        _mesh_walk._KNOTS_MEMO._size = size
    for a, b in zip(memoised, fresh, strict=True):
        for x, y in zip(a, b, strict=True):
            assert x.dtype == y.dtype and x.tobytes() == y.tobytes()


def test_writable_polylines_walk_every_time():
    pls = [np.array([[0.0, 0, 0], [0, 0, 1.0], [0, 1.0, 1.0]])]
    hits = _mesh_walk._KNOTS_MEMO.hits
    a, _ = _mesh_walk.wire_knots(pls, [[2, 3]])
    b, _ = _mesh_walk.wire_knots(pls, [[2, 3]])
    assert a is not b and a.flags.writeable
    assert _mesh_walk._KNOTS_MEMO.hits == hits
    IM.freeze(pls)
    c, _ = _mesh_walk.wire_knots(pls, [[2, 3]])
    d, _ = _mesh_walk.wire_knots(pls, [[2, 3]])
    assert c is d and c.tobytes() == a.tobytes()
    e, _ = _mesh_walk.wire_knots(pls, [[3, 2]])  # other counts, other walk
    assert e is not c
