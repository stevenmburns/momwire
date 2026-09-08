"""ArrayBlock says when it has nothing to block (momwire#972).

`ArrayBlockSolver` exists to reuse one dense self-block across repeated
translated elements. On a structure with no repeats there is nothing to reuse,
and it quietly ran the inherited H-matrix path under its own name: the answer
was right, the speedup the user chose the class for was absent, and nothing
said so.

IT ADVISES RATHER THAN REFUSES, and the measurement is why. Across the
antennaknobs catalog — 103 designs — only 27 carry a repeated shape. A refusal
would have taken working capability off 72 decks, including the whole Yagi
class, where "no repeated elements" is true precisely BECAUSE the elements are
deliberately different lengths (`beams.owa_yagi`: 4 elements, 4 shapes). The
defect is that the user cannot see the gap, not that the answer is wrong, so
the remedy is a sentence rather than a stop.

Gates:

- G-972-1  a lattice emits NOTHING. Delete-the-line for the predicate: an
           advisory on a real array would be a false alarm on the one class
           this solver is for.
- G-972-2  a no-repeat structure emits EXACTLY ONE, with the counts in it.
- G-972-3  once per solver, not once per `array_partition()` call — the fill,
           the preconditioner and the swept path all reach it.
- G-972-4  the answer is unchanged and is HMatrixSolver's, which is what the
           advisory claims.
- G-972-5  the class is discoverable by `__module__` root, which is how
           antennaknobs records momwire advisories (it records by root, not by
           a class list, so a new advisory arrives without an edit there).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from momwire import ArrayBlockSolver, HMatrixSolver
from momwire.array_block import ArrayBlockNoRepeats

LAM = 8.0
RADIUS = 0.01


def _dipole(x, half):
    return np.array([(x, 0.0, -half), (x, 0.0, half)])


def _solver(cls, halves, **kw):
    wires = [_dipole(1.6 * i, h) for i, h in enumerate(halves)]
    return cls(
        wires=wires,
        n_per_edge_per_wire=[[12]] * len(halves),
        wavelength=LAM,
        wire_radius=RADIUS,
        feeds=[(0, halves[0], 1 + 0j)],
        degree=2,
        **kw,
    )


def _solve(cls, halves, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        z = complex(_solver(cls, halves, **kw).compute_impedance()[0])
    return z, [w for w in caught if issubclass(w.category, ArrayBlockNoRepeats)]


# Four IDENTICAL dipoles: the class's own case.
LATTICE = [2.0, 2.0, 2.0, 2.0]
# Four DIFFERENT dipoles: a Yagi in miniature — multi-element, no repeats.
NO_REPEATS = [2.0, 1.9, 1.8, 1.7]


def test_a_lattice_says_nothing():
    """G-972-1. Delete-the-line: the predicate must not fire on an array."""
    _z, adv = _solve(ArrayBlockSolver, LATTICE)
    assert adv == [], [str(a.message) for a in adv]


def test_a_structure_with_no_repeats_says_so_once():
    """G-972-2. One advisory, carrying the counts that justify it."""
    _z, adv = _solve(ArrayBlockSolver, NO_REPEATS)
    assert len(adv) == 1, [str(a.message) for a in adv]
    msg = str(adv[0].message)
    assert "4 elements, 4 distinct shapes" in msg, msg
    # It has to say what to do, not only what happened (the #1264 rule).
    assert "HMatrixSolver" in msg and "BSplineSolver" in msg, msg


def test_a_single_connected_structure_is_the_same_case():
    """One element of one shape — no special branch, and singular grammar."""
    _z, adv = _solve(ArrayBlockSolver, [2.0])
    assert len(adv) == 1
    assert "1 element, 1 distinct shape" in str(adv[0].message)


def test_it_is_emitted_once_per_solver_not_once_per_partition_call():
    """G-972-3. The fill, the preconditioner and the swept path all call
    `array_partition()`; three copies of one sentence teach nothing."""
    sim = _solver(ArrayBlockSolver, NO_REPEATS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            sim.array_partition()
        sim.compute_impedance()
    assert sum(issubclass(w.category, ArrayBlockNoRepeats) for w in caught) == 1


def test_once_survives_the_partition_cache_being_dropped():
    """The `_no_repeats_warned` flag, exercised.

    Repeated `array_partition()` calls are already silent because the
    PARTITION is cached — so the flag looked load-bearing and was not: removing
    it passed every other test in this file. What it actually guards is the
    cache going away (an invalidation, a swept path that rebuilds), and that is
    what this drops by hand. Without the flag this emits twice.
    """
    sim = _solver(ArrayBlockSolver, NO_REPEATS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.array_partition()
        sim._array_partition = None  # what an invalidation would do
        sim.array_partition()
    assert sum(issubclass(w.category, ArrayBlockNoRepeats) for w in caught) == 1


def test_the_answer_is_unchanged_and_is_the_hmatrix_one():
    """G-972-4. The advisory claims the H-matrix path ran; this is that claim.

    Bit-identity is the right bar here: ArrayBlock with nothing to block does
    not merely agree with HMatrixSolver, it IS HMatrixSolver's code path.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z_ab = complex(_solver(ArrayBlockSolver, NO_REPEATS).compute_impedance()[0])
        z_hm = complex(_solver(HMatrixSolver, NO_REPEATS).compute_impedance()[0])
    assert z_ab == z_hm, (z_ab, z_hm)


def test_the_advisory_is_discoverable_the_way_antennaknobs_finds_them():
    """G-972-5. antennaknobs records momwire advisories by MODULE ROOT rather
    than by a class list, so a new one arrives without an edit there. That
    only works if the class lives under `momwire.`."""
    assert ArrayBlockNoRepeats.__module__.split(".")[0] == "momwire"
    assert issubclass(ArrayBlockNoRepeats, UserWarning)


@pytest.mark.slow
def test_the_catalog_split_is_what_the_advisory_was_argued_from():
    """The census the refuse-vs-advise decision rested on, kept honest.

    If a future change made repeats the common case, refusing would become
    reasonable and this number is the thing that would have to move first.
    Geometry only — no solves.
    """
    pytest.importorskip("antennaknobs")
    import importlib
    import pkgutil

    import antennaknobs.designs as pkg
    from antennaknobs.engines.momwire import MomwireEngine

    from momwire.array_block import element_groups

    repeats, no_repeats, errors = [], [], []
    for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        if m.ispkg:
            continue
        try:
            builder = importlib.import_module(m.name).Builder
        except Exception:  # noqa: BLE001 — a design that will not import is not this test's business
            continue
        name = m.name.split("antennaknobs.designs.")[-1]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                eng = MomwireEngine(
                    builder(),
                    solver=ArrayBlockSolver,
                    solver_kwargs={"degree": 2},
                    ground=("finite", 13.0, 0.005),
                )
                part = element_groups(
                    eng._make_solver(wavelength=eng._wavelength_for(builder().freq))
                )
        except Exception:  # noqa: BLE001 — buried decks refuse before the question arises
            errors.append(name)
            continue
        (repeats if part.n_shapes < part.n_elem else no_repeats).append(name)

    assert len(repeats) == 27, sorted(repeats)
    assert len(no_repeats) == 72, len(no_repeats)
    assert len(errors) == 4, sorted(errors)
    # The Yagi class is the reason this advises: multi-element, all distinct.
    for yagi in ("beams.owa_yagi", "beams.moxon", "broadband.lpda"):
        assert yagi in no_repeats
