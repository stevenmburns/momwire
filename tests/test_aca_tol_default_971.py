"""The ACA truncation default, and what it does and does not fix (momwire#971).

`aca_tol` bounded the rank truncation of ONE admissible block; the error in Z
is the accumulation over all of them. Measured, that runs 10^2 to 10^3 times
`aca_tol`, erratically — so the old 1e-4 read as "0.1 % accurate" and left up
to 7.4 % on the driving-point impedance of repeated-element decks.

It is 1e-6 since #971. Four of the five designs the catalog ladder flagged
improve by two to three orders of magnitude.

THE FIFTH IS NOT A TOLERANCE PROBLEM AND THIS FILE SAYS SO RATHER THAN
LOOSENING TO FIT. `loops.skyloop_lmatch` returns the SAME answer at 1e-4 and
1e-6 — 4.79e-03 relative, unchanged to five decimals — and then collapses to
5e-08 somewhere between 3e-08 and 1e-08:

    aca_tol   1e-04    1e-06    1e-07    3e-08    1e-08
    rel       4.80e-3  4.80e-3  4.80e-3  3.32e-4  3.59e-7

That is a step, not convergence: a block whose ACA stops one rank early and
needs a tolerance two orders tighter to take the extra iteration. `aca_eta`
moves it not at all. Tightening the default far enough to catch it would cost
9.4x dense on that deck, so it is filed separately rather than paid for here,
and gated below as UNCHANGED — this PR neither fixes nor breaks it.
"""

from __future__ import annotations

import importlib
import warnings

import pytest

from momwire import ArrayBlockSolver, BSplineSolver, HMatrixSolver
from momwire.hmatrix import DEFAULT_ACA_TOL

# The five the catalog ladder flagged over 1e-3 (antennaknobs#1282 rows).
TOLERANCE_FIXED = [
    "arrays.folded_invveearray",
    "arrays.moxonarray",
    "wire.sterba",
    "wire.sterba_bl",
]
STEP_NOT_TOLERANCE = "loops.skyloop_lmatch"
GROUND = ("finite", 13.0, 0.005)

pytest.importorskip("antennaknobs", reason="the ladder's decks live in antennaknobs")


def _z(design, mult, cls, **kw):
    from antennaknobs.engines.momwire import MomwireEngine

    builder = importlib.import_module(f"antennaknobs.designs.{design}").Builder
    b = builder()
    b.nominal_nsegs = int(b.nominal_nsegs) * mult
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return complex(
            MomwireEngine(
                b, solver=cls, solver_kwargs={"degree": 2, **kw}, ground=GROUND
            ).impedance()[0]
        )


def test_the_default_is_the_one_written_down_once():
    """`ArrayBlockSolver` subclasses `HMatrixSolver` and inherits its
    constructor, so there is one literal. A second one is how the two
    accelerators would come to disagree about their own default.

    CONSTRUCTED, not introspected. `ArrayBlockSolver.__init__` forwards through
    `**kwargs`, so `inspect.signature` has no `aca_tol` parameter to report and
    the introspective version of this test raised `KeyError` — the same trap
    momwire has hit before with `HMatrixSolver` reporting zero keyword
    arguments while accepting the whole B-spline option set. Build the object
    and read what it actually holds.
    """
    import numpy as np

    wires = [np.array([(0.0, 0.0, -2.0), (0.0, 0.0, 2.0)])]
    for cls in (HMatrixSolver, ArrayBlockSolver):
        s = cls(
            wires=wires,
            n_per_edge_per_wire=[[8]],
            wavelength=8.0,
            wire_radius=0.01,
            feeds=[(0, 2.0, 1 + 0j)],
            degree=2,
        )
        assert s.aca_tol == DEFAULT_ACA_TOL, cls.__name__
    assert DEFAULT_ACA_TOL == 1e-6


@pytest.mark.slow
@pytest.mark.parametrize("cls", [HMatrixSolver, ArrayBlockSolver], ids=["hm", "ab"])
@pytest.mark.parametrize("mult", [1, 2], ids=["default", "refined"])
@pytest.mark.parametrize("design", TOLERANCE_FIXED)
def test_the_accelerators_agree_with_dense_on_the_flagged_designs(design, mult, cls):
    """Within 1e-3 relative of dense b-spline degree 2 at the same mesh.

    Measured worst over these sixteen cells: 1.15e-04 — an order of magnitude
    inside the bar, which is stated rather than pinned so a regression that
    stays under 1e-3 is still a regression someone can see.
    """
    ref = _z(design, mult, BSplineSolver)
    got = _z(design, mult, cls)
    rel = abs(got - ref) / abs(ref)
    assert rel < 1e-3, (
        f"{design} x{mult} {cls.__name__}: {rel:.3e} ({got!r} vs {ref!r})"
    )


@pytest.mark.slow
@pytest.mark.parametrize("mult", [1, 2], ids=["default", "refined"])
def test_the_old_default_fails_that_bar(mult):
    """Delete-the-line. `arrays.folded_invveearray` is the worst cell in the
    ladder and 1e-4 misses the bar on it by an order of magnitude at the
    default mesh and two at the refined one."""
    ref = _z("arrays.folded_invveearray", mult, BSplineSolver)
    old = _z("arrays.folded_invveearray", mult, ArrayBlockSolver, aca_tol=1e-4)
    rel = abs(old - ref) / abs(ref)
    assert rel > 1e-3, (
        f"the old 1e-4 default now passes the bar at x{mult} ({rel:.3e}) — the "
        f"gate above no longer demonstrates anything"
    )


@pytest.mark.slow
@pytest.mark.parametrize("mult", [1, 2], ids=["default", "refined"])
def test_the_step_design_is_unchanged_by_this_default(mult):
    """`loops.skyloop_lmatch` is NOT fixed by the tolerance, and this asserts
    exactly that rather than quietly widening the bar to cover it.

    Both halves matter. That the two tolerances AGREE is the evidence the
    mechanism is not truncation — a converging scheme would move. That the
    error is bounded is what stops this reading as permission for any error at
    all.
    """
    ref = _z(STEP_NOT_TOLERANCE, mult, BSplineSolver)
    old = _z(STEP_NOT_TOLERANCE, mult, HMatrixSolver, aca_tol=1e-4)
    new = _z(STEP_NOT_TOLERANCE, mult, HMatrixSolver)
    rel = abs(new - ref) / abs(ref)
    moved = abs(new - old) / abs(ref)
    # As a RATIO against the error, not an absolute: the two tolerances do
    # differ in the last bits (a different ACA rank path), and the claim is
    # not that they are identical but that the tolerance is irrelevant to the
    # error. Measured, it moves the answer by ~1e-6 of the error it leaves.
    assert moved < rel * 1e-4, (
        f"the tolerance now moves this design by {moved:.3e} against an error "
        f"of {rel:.3e}; it is filed as a separate step-function defect on the "
        f"premise that it does not"
    )
    assert 1e-3 < rel < 6e-3, f"skyloop_lmatch moved: {rel:.3e}"


@pytest.mark.slow
def test_the_step_really_is_a_step_and_not_slow_convergence():
    """The claim the separate filing rests on: flat, then a cliff.

    If this were ordinary convergence the middle rungs would improve
    gradually and the right answer would be to tighten the default further.
    They do not.
    """
    ref = _z(STEP_NOT_TOLERANCE, 2, BSplineSolver)
    rels = {}
    for tol in (1e-4, 1e-6, 1e-7, 1e-8):
        got = _z(STEP_NOT_TOLERANCE, 2, HMatrixSolver, aca_tol=tol)
        rels[tol] = abs(got - ref) / abs(ref)
    flat = [rels[t] for t in (1e-4, 1e-6, 1e-7)]
    assert max(flat) / min(flat) < 1.01, f"the plateau is not flat: {rels}"
    assert rels[1e-8] < flat[0] / 1e3, f"no cliff between 1e-7 and 1e-8: {rels}"
