"""The graded near cells' blocks on the shared pool — perf item 9
(momwire#1290).

`_near_cells` collects every near pair's graded values once per source block
for the banded fill, block by block (`_near_block`'s byte budget). Collected,
a block writes nothing and its cells are a function of its pairs alone, so
`_apply_near_correction` runs the blocks on the shared pool and appends them
in block order, as many at once as a band's budget holds blocks (the band
scratch is not live yet while the near cells are collected, so the fill's
peak does not move); `_NEAR_THREADS = False` is the serial loop. The blocks
are the serial loop's, so the claim is equality of bytes:
of every `_NearCells` a solve collects, and of the matrix handed to the
solve, Z and the currents — free, above-ground and buried, the threaded
collections counted.

The near cells are the banded fill's (decks of 60 segments and up), and
these decks are small, so the near-pair byte budget is shrunk to cut their
pairs into several blocks — the seam the residency gates shrink too — and
the band budget set to hold four of them.

The red control moves the first collection's values by one ulp each and
requires the matrix gate to see it.
"""

from __future__ import annotations

import concurrent.futures

import numpy as np
import pytest
from test_sg_cplx_far_fill_simd_1224 import _buried_dipole, _hub4
from test_sg_real_far_fill_staged_1290 import _array, _vee

from momwire import sinusoidal_galerkin as sg
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver


def _direct_sg(*args, **kw):
    """This module's subject is the DIRECT-field fill's own machinery, so
    every solver here names it (`SinusoidalGalerkinSolver`'s default fill is
    the mixed-potential one since momwire#1354, gated in
    `test_sinusoidal_mp_1354.py`)."""
    from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver as _cls

    kw.setdefault("fill", "direct")
    return _cls(*args, **kw)


def _banded(make, scale):
    """`make()`'s deck with every wire's segment count times `scale`: the
    near cells are collected by the banded fill alone, which takes decks of
    60 segments and up."""
    d = make()
    d["n_per_edge_per_wire"] = [
        [n * scale for n in e] for e in d["n_per_edge_per_wire"]
    ]
    return d


DECKS = {
    "vee-free": lambda: _banded(_vee, 2),
    "array-above": lambda: _array(True, n=21),
    "buried-dipole": lambda: _banded(_buried_dipole, 2),
    "hub4": lambda: _banded(_hub4, 2),
}


# The originals, taken once: `_solve` patches over them each call, and a
# second call in one test must not wrap the first call's spies.
_SOLVE = sg._solve_in_place
_NEAR = SinusoidalGalerkinSolver._near_cells

# A four-wide pool of the test's own: an xdist worker pins OMP_NUM_THREADS=1
# (tests/conftest.py), which would size the shared pool at one worker and
# leave the pooled path unexercised.
_POOL4 = concurrent.futures.ThreadPoolExecutor(max_workers=4)


class _CountingPool:
    """`_piece_pool()`'s stand-in: four workers, submissions counted."""

    _max_workers = 4

    def __init__(self, seen):
        self._seen = seen

    def submit(self, fn, *a):
        self._seen["threaded"] += 1
        return _POOL4.submit(fn, *a)


@pytest.fixture(autouse=True)
def _several_blocks(monkeypatch):
    monkeypatch.setattr(sg, "_NEAR_WORKSPACE_BYTES", 1 << 18)
    # Room for four blocks at once: the pool's concurrency is the band
    # budget's worth of near blocks.
    monkeypatch.setattr(
        SinusoidalGalerkinSolver, "_band_budget_bytes", lambda self, n: 4 << 18
    )


def _solve(make, monkeypatch, perturb=False):
    """(G, Z, currents, every collected _NearCells, threaded collections)."""
    seen = {"cells": [], "threaded": 0}
    solve, near = _SOLVE, _NEAR

    def spy_solve(G, rhs):
        seen.setdefault("G", np.array(G, copy=True))
        return solve(G, rhs)

    def spy_near(self, *a, **kw):
        cells = near(self, *a, **kw)
        if perturb and not seen["cells"] and cells.vals[0].size:
            # Every value of the first collection, by one ulp: a single cell
            # can round away against its destination (it did on macOS).
            v = cells.vals[0]
            v.real = np.nextafter(v.real, np.inf)
        seen["cells"].append(cells)
        return cells

    def spy_pool():
        return _CountingPool(seen)

    monkeypatch.setattr(sg, "_solve_in_place", spy_solve)
    monkeypatch.setattr(SinusoidalGalerkinSolver, "_near_cells", spy_near)
    monkeypatch.setattr(sg, "_piece_pool", spy_pool)
    Z, cur = _direct_sg(**make()).compute_impedance()
    return seen["G"], np.atleast_1d(Z), np.asarray(cur), seen["cells"], seen["threaded"]


def _cell_bytes(cells):
    return [
        (c.entry.tobytes(), c.seg.tobytes(), tuple(v.tobytes() for v in c.vals))
        for c in cells
    ]


@pytest.mark.parametrize("name", sorted(DECKS))
def test_threaded_near_cells_equal_the_serial_loop_to_the_byte(name, monkeypatch):
    G1, Z1, I1, cells1, threaded = _solve(DECKS[name], monkeypatch)
    assert threaded > 0, f"{name}: no near-cell collection ran on the pool"
    monkeypatch.setattr(sg, "_NEAR_THREADS", False)
    G0, Z0, I0, cells0, threaded0 = _solve(DECKS[name], monkeypatch)
    assert threaded0 == 0
    assert _cell_bytes(cells1) == _cell_bytes(cells0)
    assert G1.tobytes() == G0.tobytes()
    assert Z1.tobytes() == Z0.tobytes()
    assert I1.tobytes() == I0.tobytes()


def test_a_band_budget_under_two_blocks_collects_serially(monkeypatch):
    """The pooled blocks' scratch has to fit in a band's budget, which is free
    while the near cells are collected; with room for one block they run on
    this thread and the fill's peak cannot move."""
    monkeypatch.setattr(
        SinusoidalGalerkinSolver, "_band_budget_bytes", lambda self, n: (1 << 18) + 1
    )
    _, _, _, cells, threaded = _solve(DECKS["vee-free"], monkeypatch)
    assert cells and threaded == 0


def test_red_control_one_ulp_in_the_near_cells_is_seen(monkeypatch):
    G0, _, _, _, _ = _solve(DECKS["vee-free"], monkeypatch)
    G1, _, _, _, _ = _solve(DECKS["vee-free"], monkeypatch, perturb=True)
    assert G1.tobytes() != G0.tobytes()
