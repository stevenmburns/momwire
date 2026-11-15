"""momwire#1943 lever 2: the segment-to-element search, row-blocked.

`DeckSolver._element_match_uncached` matched every NEC segment against every
momwire element through one N x M x 3 broadcast -- about 420 MB of temporary
on the ELT whip, once per launch. It now takes the segments a block of rows
at a time with each row's arithmetic unchanged. The gate is bit-equality of
`(nearest, dirs)` with the whole-structure broadcast, kept below verbatim, at
blocks of one row, an odd block and the default, on decks that exercise the
alignment rule (crossed wires sharing a midpoint), a reversed walk, transforms
and a ring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from momwire import _mesh_walk
from momwire._vec3 import norm3
from momwire.dropin.nec2c import _portal

FIXTURES = Path(__file__).parent / "fixtures" / "nec_portal"
DECKS = (
    "dipole_rp_crossed_quadrature",
    "apex_pq_reversed_walk",
    "dipole_gr_rotated_ring",
    "dipole_gx_reflected_pair",
    "catalog_beams_moxon",
    "catalog_loops_delta_loop",
)


def _whole_broadcast(deck_solver, solver):
    """The search as it was before #1943, verbatim."""
    knots, offs = _mesh_walk.wire_knots(
        solver.wires_polylines, solver.n_per_edge_per_wire
    )
    lo, hi = _mesh_walk.interior_pairs(offs)
    mids = 0.5 * (knots[hi] + knots[lo])
    dirs = knots[hi] - knots[lo]
    wanted = np.array([s.centre for s in deck_solver.segments])
    wanted_dir = np.array([s.direction for s in deck_solver.segments])
    d2 = ((wanted[:, None, :] - mids[None, :, :]) ** 2).sum(axis=2)
    unit_e = dirs / np.maximum(norm3(dirs)[:, None], 1e-30)
    unit_s = wanted_dir / np.maximum(norm3(wanted_dir)[:, None], 1e-30)
    aligned = np.abs(unit_s @ unit_e.T) >= 0.5
    cost = np.where(aligned, d2, np.inf)
    fallback = ~np.isfinite(cost).any(axis=1)
    cost[fallback] = d2[fallback]
    nearest = np.argmin(cost, axis=1)
    return nearest, dirs


@pytest.mark.parametrize("name", DECKS)
def test_the_blocked_search_is_the_broadcast_bit_for_bit(name):
    path = FIXTURES / f"{name}.deck"
    if not path.exists():
        pytest.fail(f"fixture {name} moved: the gate would measure nothing")
    deck = _portal.parse_deck(path.read_text())
    deck_solver = _portal.DeckSolver(deck, basis="sinusoidal")
    solver = next(iter(deck_solver._cache.values()))["solver"]
    want_nearest, want_dirs = _whole_broadcast(deck_solver, solver)
    for rows in (1, 7, None):
        nearest, dirs = deck_solver._element_match_uncached(solver, rows=rows)
        assert nearest.dtype == want_nearest.dtype
        assert nearest.tobytes() == want_nearest.tobytes()
        assert dirs.tobytes() == want_dirs.tobytes()


def test_the_crossed_pair_is_told_apart_by_direction():
    """The deck the alignment rule exists for: two wires crossing at a shared
    segment midpoint name two different elements, at every block size."""
    deck = _portal.parse_deck(
        (FIXTURES / "dipole_rp_crossed_quadrature.deck").read_text()
    )
    deck_solver = _portal.DeckSolver(deck, basis="sinusoidal")
    solver = next(iter(deck_solver._cache.values()))["solver"]
    for rows in (1, 7, None):
        nearest, _ = deck_solver._element_match_uncached(solver, rows=rows)
        assert len(set(nearest.tolist())) == deck_solver.n_segments
