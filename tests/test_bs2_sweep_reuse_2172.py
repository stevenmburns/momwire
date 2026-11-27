"""bs2's frequency sweeps read the `pair_reuse` plan (momwire#2172).

The batched sweep (`_swept_batched_z_chunks`) evaluates every pair's moments
over a chunk of k and reads no plan, so a sweep with reuse on used to be the
reuse-off sweep. With a served plan the sweep now runs frequency by frequency
(`_swept_reads_plan`), each k's fill copying the classes' values, and the plan,
k-independent, is built once for the sweep:

- the reuse fill runs at every k and the batched route at none, counted;
- the sweep moves Zin inside `pair_reuse`'s declared drift, and agrees with
  the reuse-on single-frequency solves;
- reuse off keeps the batched route and its bits;
- where the batched route still runs with reuse on (the census declined, or a
  solver whose per-k fill reads no plan), `pair_reuse_declined` says why.
"""

from __future__ import annotations

import numpy as np
import pytest

import momwire.bspline as bs
from momwire import _features
from momwire.bspline import BSplineSolver
from momwire.hmatrix import HMatrixSolver

from _reuse_seam import bs2

pytestmark = pytest.mark.reach(4)

LAM = 21.26
#: The same-edge Toeplitz form is held on on both sides (`_reuse_seam`), so
#: the two differ by the census.
DRIFT = _features.family("pair_reuse", "bspline").drift
PEC = {"ground_z": 0.0}
GROUNDS = {"free": {}, "pec": PEC}


@pytest.fixture
def small_decks():
    """The decks below are under the small-deck threshold; the census is what
    is under test, so it runs on them."""
    acc = bs._acc
    if acc is None or not hasattr(acc, "bspline_reuse_plan"):
        pytest.skip("bs2's pair-reuse census is not built")
    acc.bspline_reuse_set_small_deck_segs(0)
    yield
    acc.bspline_reuse_set_small_deck_segs(-1)


@pytest.fixture
def routes(monkeypatch):
    """What a sweep ran: batched chunks, the frequencies whose class values
    were evaluated (an image fill asks again for the values its k holds),
    censuses."""
    seen = {"batched": 0, "values": 0, "census": 0}
    ks = set()

    def count(name, key):
        real = getattr(BSplineSolver, name)

        def spy(self, *a, **kw):
            if key == "values":
                ks.add(float(a[2]))
                seen[key] = len(ks)
            else:
                seen[key] += 1
            return real(self, *a, **kw)

        monkeypatch.setattr(BSplineSolver, name, spy)

    count("_swept_batched_z_chunks_filled", "batched")
    count("_reuse_values", "values")
    count("_reuse_plan_built", "census")
    return seen


def _three(reuse, n=13, cls=BSplineSolver, lam=LAM, **kw):
    """Three identical parallel dipoles, fed on the first."""
    wires = [np.array([[3.0 * i, -4.0, 10.0], [3.0 * i, 4.0, 10.0]]) for i in range(3)]
    kw.setdefault("toeplitz", True)
    return bs2(
        cls=cls,
        wires=wires,
        n_per_edge_per_wire=[[n] for _ in wires],
        feed_wire_index=0,
        wavelength=lam,
        wire_radius=0.002,
        pair_reuse=reuse,
        **kw,
    )


K = 2 * np.pi / LAM * np.array([0.97, 1.0, 1.03, 1.06])


def _port_sweep(s):
    return np.stack([np.asarray(p.y) for p in s.compute_port_solution_swept(K)])


@pytest.mark.parametrize("ground", sorted(GROUNDS))
def test_a_sweep_reads_the_plan_at_every_frequency(ground, small_decks, routes):
    s = _three(True, **GROUNDS[ground])
    assert s._swept_batched_available()
    s.compute_impedance_swept(K)
    assert routes == {"batched": 0, "values": K.size, "census": 1}
    assert s.pair_reuse_declined is None
    assert s.pair_reuse_served["copied_pairs"] > 0


def test_the_port_sweep_reads_it_too(small_decks, routes):
    s = _three(True)
    _port_sweep(s)
    assert routes == {"batched": 0, "values": K.size, "census": 1}
    assert s.pair_reuse_declined is None


@pytest.mark.parametrize("ground", sorted(GROUNDS))
def test_the_sweep_stays_inside_the_declared_drift(ground, small_decks):
    on = _three(True, **GROUNDS[ground]).compute_impedance_swept(K)
    off = _three(False, **GROUNDS[ground]).compute_impedance_swept(K)
    assert np.max(np.abs(on - off) / np.abs(off)) <= DRIFT
    y_on, y_off = _port_sweep(_three(True)), _port_sweep(_three(False))
    assert np.max(np.abs(y_on - y_off)) / np.max(np.abs(y_off)) <= DRIFT


def test_the_sweep_is_the_single_frequency_solves(small_decks):
    """Each k of the sweep is the reuse-on solve at that k, inside the drift
    (the sweep shares its same-edge moments, a reassociation)."""
    z = _three(True).compute_impedance_swept(K)
    one = np.array([_three(True, lam=2 * np.pi / k).compute_impedance()[0] for k in K])
    assert np.max(np.abs(z - one) / np.abs(one)) <= DRIFT


@pytest.mark.parametrize("ground", sorted(GROUNDS))
def test_reuse_off_keeps_the_batched_route_and_its_bits(ground, small_decks, routes):
    s = _three(False, **GROUNDS[ground])
    z = s.compute_impedance_swept(K)
    assert routes["batched"] > 0 and routes["values"] == 0 and routes["census"] == 0
    assert s.pair_reuse_declined is None
    assert np.array_equal(
        z, _three(False, **GROUNDS[ground])._compute_impedance_swept_batched(K)
    )


def test_a_declined_census_keeps_the_batched_route_and_says_why(routes):
    """Under the small-deck threshold the census declines; the batched route
    runs with reuse on, and the census's reason is what is named."""
    s = _three(True)
    z = s.compute_impedance_swept(K)
    assert routes["batched"] > 0 and routes["values"] == 0
    assert s.pair_reuse_declined.startswith("too few segments to repay the plan")
    assert np.array_equal(z, _three(False).compute_impedance_swept(K))
    _port_sweep(s)
    assert s.pair_reuse_declined.startswith("too few segments to repay the plan")


def test_a_solver_whose_fill_reads_no_plan_keeps_the_batched_route(small_decks, routes):
    """The H-matrix hands a small sweep to the dense base, whose per-k fill
    would be its own operator: the batched route runs, and says so."""
    s = _three(True, cls=HMatrixSolver)
    assert s._swept_prefers_dense()
    z = s.compute_impedance_swept(K)
    assert routes["batched"] > 0 and routes["census"] == 0
    assert s.pair_reuse_declined == "the batched sweep does not read the hint"
    assert np.array_equal(
        z, _three(False, cls=HMatrixSolver).compute_impedance_swept(K)
    )
