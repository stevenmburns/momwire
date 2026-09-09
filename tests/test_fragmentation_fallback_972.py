"""The H-matrix route steps aside when the cluster tree fragments (momwire#972).

`verticals.elt_whip` — 4,067 separate wires, 12,405 bases — exceeded 600 s on
both accelerators against 83 s dense. Profiled, 55 % of that is ACA fetching
one row or column at a time through the off-edge kernel, across 18,354 far
blocks. The blocks are not small (mean 56², LARGER than the decks that win);
there are simply 107x as many as on the next deck.

So the predicate is the COUNT, and it is measured rather than chosen:

    deck                     far blocks   n_basis   far/basis
    verticals.elt_whip           18,354    12,405      1.48    <- times out
    arrays.bowtie4x4                984     1,440      0.68
    arrays.yagiarray                316       660      0.48
    every other catalog deck      <=278                <=0.48

`_FRAG_FAR_PER_BASIS = 1.0` clears the runner-up by 2.2x and
`_FRAG_MIN_FAR_BLOCKS = 2000` by 18.7x; both must hold, so a small deck can
never trip it.

AND THE FALLBACK IS SIZED BEFORE IT IS TAKEN. `_hmatrix_unsupported` records
why a silent dense route is the wrong kindness on a deck this class exists to
serve — it turns a slow answer into an out-of-memory one — and a fragmented
deck is exactly the shape that can be too big for dense. So the dense matrix
is measured first and the route only changes when it fits.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import momwire.hmatrix as _hm
from momwire import ArrayBlockSolver, BSplineSolver, HMatrixSolver
from momwire.array_block import ArrayBlockNoRepeats
from momwire.hmatrix import HMatrixFragmented

LAM = 8.0


def _wires(n, span=6.0):
    """`n` short collinear wires — the elt_whip shape in miniature: one
    physical run chopped into many separate GW-style pieces, which is what
    fragments a cluster tree."""
    zs = np.linspace(-span / 2, span / 2, n + 1)
    return [np.array([(0.0, 0.0, zs[i]), (0.0, 0.0, zs[i + 1])]) for i in range(n)]


def _sim(cls, n_wires, **kw):
    w = _wires(n_wires)
    return cls(
        wires=w,
        n_per_edge_per_wire=[[2]] * len(w),
        wavelength=LAM,
        wire_radius=0.005,
        feeds=[(len(w) // 2, 0.0, 1 + 0j)],
        degree=2,
        **kw,
    )


# ---------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------


def test_an_ordinary_deck_does_not_trip_it():
    """Delete-the-line for the threshold. A predicate that catches a deck the
    accelerators serve well is worse than no predicate — it would take the
    whole class off its own winning cases."""
    sim = _sim(HMatrixSolver, 4)
    assert sim._fragmentation() is None
    assert not sim._prefers_dense_for_fragmentation()


def test_both_clauses_are_load_bearing(monkeypatch):
    """Ratio AND count. Either alone catches decks it should not: the ratio
    alone would fire on any small fragmented toy, the count alone on any large
    deck that clusters perfectly."""
    sim = _sim(HMatrixSolver, 40)
    part = sim.build_partition()
    far, n_basis = len(part["far"]), int(part["root"].size)
    # Neither clause is met as shipped.
    assert not sim._prefers_dense_for_fragmentation()
    # Ratio alone: drop the floor and it still must not fire unless the ratio
    # is genuinely over.
    monkeypatch.setattr(_hm, "_FRAG_MIN_FAR_BLOCKS", 1)
    ratio = far / n_basis
    sim2 = _sim(HMatrixSolver, 40)
    assert bool(sim2._prefers_dense_for_fragmentation()) == (
        ratio > _hm._FRAG_FAR_PER_BASIS
    ), (far, n_basis, ratio)


def test_a_fragmented_deck_falls_back_and_says_so(monkeypatch):
    """The thresholds lowered to this toy's scale, so the ROUTE CHANGE is
    exercised without a 12,405-basis deck in the lane."""
    sim = _sim(HMatrixSolver, 40)
    part = sim.build_partition()
    far, n_basis = len(part["far"]), int(part["root"].size)
    assert far >= 4, "this toy has too few far blocks to exercise anything"
    monkeypatch.setattr(_hm, "_FRAG_MIN_FAR_BLOCKS", 1)
    monkeypatch.setattr(_hm, "_FRAG_FAR_PER_BASIS", far / n_basis / 2.0)
    fresh = _sim(HMatrixSolver, 40)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        z = complex(fresh.compute_impedance()[0])
    frag = [w for w in caught if issubclass(w.category, HMatrixFragmented)]
    assert len(frag) == 1, [str(w.message) for w in caught]
    assert "took the DENSE route" in str(frag[0].message)
    # And the answer IS the dense one, which is what the sentence claims.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z_dense = complex(_sim(BSplineSolver, 40).compute_impedance()[0])
    assert z == z_dense, (z, z_dense)


def test_it_refuses_to_fall_back_into_an_out_of_memory(monkeypatch):
    """The guard `_hmatrix_unsupported`'s docstring asks for: a fragmented
    deck is exactly the shape that can be too big for the dense fallback, and
    trading a slow answer for an OOM is not an improvement. With the budget
    set below this deck's dense matrix, the route must NOT change — and the
    advisory must say a different thing."""
    sim = _sim(HMatrixSolver, 40)
    part = sim.build_partition()
    far, n_basis = len(part["far"]), int(part["root"].size)
    monkeypatch.setattr(_hm, "_FRAG_MIN_FAR_BLOCKS", 1)
    monkeypatch.setattr(_hm, "_FRAG_FAR_PER_BASIS", far / n_basis / 2.0)
    monkeypatch.setattr(_hm, "_FRAG_DENSE_MAX_GB", 1e-12)
    fresh = _sim(HMatrixSolver, 40)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert fresh._prefers_dense_for_fragmentation() is False
    frag = [w for w in caught if issubclass(w.category, HMatrixFragmented)]
    assert len(frag) == 1
    msg = str(frag[0].message)
    assert "H-matrix route ran anyway" in msg, msg
    assert "over the" in msg and "GB this fallback allows" in msg, msg


def test_the_dense_budget_default_excludes_a_deck_dense_cannot_hold():
    """The shipped `_FRAG_DENSE_MAX_GB`, gated on arithmetic.

    Every other test here patches the budget to force one branch or the other,
    so the DEFAULT was untested — raising it to 1e9 (i.e. always fall back,
    guard gone) passed the whole file. Exercising it with a real deck needs
    ~19,400 bases, larger than anything in the catalog, so the sizing is a
    function and this checks the number against it.
    """
    from momwire.hmatrix import _dense_matrix_gb

    # elt_whip, the deck the fallback exists for: 2.46 GB, comfortably inside.
    assert _dense_matrix_gb(12405) == pytest.approx(2.46, rel=0.02)
    assert _dense_matrix_gb(12405) < _hm._FRAG_DENSE_MAX_GB

    # Twice that deck is 9.9 GB and must NOT be swapped onto the dense route:
    # the whole point of the guard is that a fragmented deck can be too big
    # for the fallback, and trading slow for out-of-memory is not a fix.
    assert _dense_matrix_gb(24810) > _hm._FRAG_DENSE_MAX_GB
    # And the budget is a real bound, not effectively infinite.
    assert 1.0 < _hm._FRAG_DENSE_MAX_GB < 64.0


def test_the_advisory_is_raised_once():
    """Four call sites consult the predicate; one sentence is enough."""
    sim = _sim(HMatrixSolver, 4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(4):
            sim._prefers_dense_for_fragmentation()
    assert sum(issubclass(w.category, HMatrixFragmented) for w in caught) == 0


def test_the_advisory_is_discoverable_the_way_antennaknobs_finds_them():
    assert HMatrixFragmented.__module__.split(".")[0] == "momwire"
    assert issubclass(HMatrixFragmented, UserWarning)


# ---------------------------------------------------------------------
# Composition with the no-repeats advisory
# ---------------------------------------------------------------------


def test_arrayblock_says_both_things_and_only_one_changes_the_route(monkeypatch):
    """The composition #972 asks for.

    On a fragmented single structure BOTH are true — nothing to block, and the
    tree fragmented — and they read as one story in that order. The base
    class's check routes straight to dense, which meant `array_partition()`
    was never called and the no-repeats sentence never fired; ArrayBlock
    overrides the check to build the partition first for exactly that reason.
    """
    sim = _sim(ArrayBlockSolver, 40)
    part = sim.build_partition()
    far, n_basis = len(part["far"]), int(part["root"].size)
    monkeypatch.setattr(_hm, "_FRAG_MIN_FAR_BLOCKS", 1)
    monkeypatch.setattr(_hm, "_FRAG_FAR_PER_BASIS", far / n_basis / 2.0)
    fresh = _sim(ArrayBlockSolver, 40)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fresh.compute_impedance()
    kinds = [w.category for w in caught]
    assert sum(k is ArrayBlockNoRepeats for k in kinds) == 1, [
        str(w.message) for w in caught
    ]
    assert sum(k is HMatrixFragmented for k in kinds) == 1
    # Order: why it is on the H-matrix path at all, then why even that was
    # abandoned.
    first = next(i for i, k in enumerate(kinds) if k is ArrayBlockNoRepeats)
    second = next(i for i, k in enumerate(kinds) if k is HMatrixFragmented)
    assert first < second, kinds


# ---------------------------------------------------------------------
# The catalog: nothing the accelerators serve may trip this
# ---------------------------------------------------------------------


# The whole-catalog census that used to close this module --
# `test_no_catalog_deck_but_the_timing_out_one_trips_the_threshold` -- moved to
# antennaknobs (antennaknobs#1299, PR #1318). Its subject was THAT catalog: a
# pkgutil walk of `antennaknobs.designs` asserting a property over every
# design, with this module's predicate as the instrument. It could not be
# fixture-driven the way momwire#988 fixed this repo's own behavioural tests,
# because the catalog IS the measurement.
#
# It ran nowhere from here. `importorskip("antennaknobs")` meant no CI lane
# executed it (momwire#988), AND `@pytest.mark.slow` plus this repo's default
# `addopts` deselect meant a local run did not even report it as skipped --
# `--collect-only` showed it as deselected, which `-rs` does not print. In
# antennaknobs both packages are installed and it runs on every PR.
