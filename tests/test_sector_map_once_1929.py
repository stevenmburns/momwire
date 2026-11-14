"""momwire#1929 item 2: the sector map is geometry, so a frequency sweep
reads it once.

The nec2c launchers construct one solver per FR step (`DeckSolver.at`), and
each construction under `rotational_symmetry=True` re-ran
`_rotational_symmetry.sector_map`'s orbit search -- 2.2 s of the sym4 ELT
whip's 13 steps under cProfile. `sector_map` is now memoised on a digest of
exactly what it reads (`_SectorInputs`) and the tolerance, never the
wavenumber.

The gates: one computation across a multi-frequency FR run through the
portal's own seam (the solver per step, as the launcher builds it), with the
port answers BIT-identical to a run that computes the map at every step; and
a different structure, tolerance or port set is a different key.
"""

from __future__ import annotations

import numpy as np
import pytest

import momwire._rotational_symmetry as RS
import momwire.sinusoidal as S
from momwire.dropin.nec2c import _portal

# A ground-plane vertical: the mast on the axis, fed at its base segment, and
# four drooping radials -- four rotated copies about the mast, so the sector
# route serves it with N = 4.
DECK = """CM ground-plane vertical, four radials
CE
GW 1 11 0 0 0 0 0 0.5 0.002
GW 2 7 0 0 0 0.45 0 -0.15 0.002
GW 3 7 0 0 0 0 0.45 -0.15 0.002
GW 4 7 0 0 0 -0.45 0 -0.15 0.002
GW 5 7 0 0 0 0 -0.45 -0.15 0.002
GE 0
EX 0 1 1 0 1 0
FR 0 4 0 0 130 10
XQ
EN
"""


@pytest.fixture
def sector_route(monkeypatch):
    """`rotational_symmetry=True` on every SinusoidalSolver the portal
    builds -- the ladder harness's `mw_sector.py`, which is how the route is
    reached from a launcher today (the launcher has no switch for it)."""
    real = S.SinusoidalSolver.__init__

    def init(self, *a, **k):
        k["rotational_symmetry"] = True
        real(self, *a, **k)

    monkeypatch.setattr(S.SinusoidalSolver, "__init__", init)
    RS._SECTOR_MAP_CACHE.clear()
    yield
    RS._SECTOR_MAP_CACHE.clear()


class _Counter:
    def __init__(self, real):
        self.real = real
        self.calls = 0

    def __call__(self, *a, **k):
        self.calls += 1
        return self.real(*a, **k)


def _sweep():
    """Every FR step of the deck's one group, through the portal's own seam:
    `(Y, v_gap, i_port, segment currents)` per frequency, and the solvers."""
    deck = _portal.parse_deck(DECK)
    solver = _portal.DeckSolver(deck, basis="sinusoidal")
    (group,) = deck.groups
    out, solvers = [], []
    for f in group.freqs_mhz:
        r = solver.solve_group(group, f)
        out.append((solver.at(f)["Y"], r.v_gap, r.i_port, r.segment_currents))
        solvers.append(r.solver)
    return out, solvers


def test_the_sector_map_is_computed_once_across_the_fr_loop(sector_route, monkeypatch):
    counter = _Counter(RS._sector_map)
    monkeypatch.setattr(RS, "_sector_map", counter)
    cached, solvers = _sweep()
    assert len(cached) == 4
    assert len({id(s) for s in solvers}) == 4  # a solver per frequency...
    assert all(s._rotational_map is not None for s in solvers)
    assert counter.calls == 1  # ...and one map between them
    assert all(s._rotational_map is solvers[0]._rotational_map for s in solvers)
    assert solvers[0]._rotational_map.n_sectors == 4

    # The same sweep computing the map at every step (a cache that holds
    # nothing): the same map, and the same answers to the bit.
    RS._SECTOR_MAP_CACHE.clear()
    monkeypatch.setattr(RS, "_SECTOR_MAP_CACHE_SIZE", 0)
    fresh, fresh_solvers = _sweep()
    assert counter.calls == 1 + 4
    assert fresh_solvers[0]._rotational_map == solvers[0]._rotational_map
    for a, b in zip(cached, fresh, strict=True):
        for x, y in zip(a, b, strict=True):
            x, y = np.asarray(x), np.asarray(y)
            assert x.dtype == y.dtype and x.shape == y.shape
            assert x.tobytes() == y.tobytes()


def test_the_key_is_the_structure_and_the_tolerance_not_the_frequency(
    sector_route, monkeypatch
):
    counter = _Counter(RS._sector_map)
    monkeypatch.setattr(RS, "_sector_map", counter)
    deck = _portal.parse_deck(DECK)
    solver = _portal.DeckSolver(deck, basis="sinusoidal")
    low, high = solver.at(130.0)["solver"], solver.at(160.0)["solver"]
    assert low.k != high.k
    assert counter.calls == 1
    assert RS.sector_map(high) is low._rotational_map
    assert counter.calls == 1
    # Another tolerance is another reading.
    RS.sector_map(high, tol_rel=2 * RS.TOL_REL)
    assert counter.calls == 2
    # Another port set is another key: a feed moved one segment up the mast.
    moved = _portal.DeckSolver(
        _portal.parse_deck(DECK.replace("EX 0 1 1 0 1 0", "EX 0 1 2 0 1 0")),
        basis="sinusoidal",
    )
    moved.at(130.0)
    assert counter.calls == 3
    # Another structure is another key: one radial a millimetre longer, which
    # breaks the symmetry and refuses -- a refusal is never served from the
    # cache, so the solver that is symmetric still reads its own map.
    broken = DECK.replace("0 -0.45 -0.15", "0 -0.451 -0.15")
    with pytest.raises(RS.RotationalSymmetryRefused):
        _portal.DeckSolver(_portal.parse_deck(broken), basis="sinusoidal").at(130.0)
    assert counter.calls == 4
    assert RS.sector_map(low) is low._rotational_map
    assert counter.calls == 4


def test_the_inputs_are_all_the_map_reads():
    """`_SectorInputs` carries a fixed set of fields and nothing else, so a
    condition that starts reading another solver attribute fails loudly
    (AttributeError) instead of reading something the digest does not
    cover."""
    inputs = object.__new__(RS._SectorInputs)
    with pytest.raises(AttributeError):
        inputs.k = 1.0  # no slot for the wavenumber, or anything else
    assert "k" not in RS._SectorInputs.__slots__
    assert "wavelength" not in RS._SectorInputs.__slots__
