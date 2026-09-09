"""momwire#973: the global Sommerfeld ACA can stagnate, and its own stopping
rule cannot see it.

`aca_partial` stops when the newest rank-1 update is small relative to the
accumulated norm. Partial pivoting can exhaust a subspace, and then the updates
go small because the pivots are trapped rather than because the approximation
is good. Measured on the geometry below (an antennaknobs `skyloop_lmatch`, a
triangular sky loop at 15 m over Sommerfeld ground, reduced to the two
polylines momwire actually receives):

    rank   ||u||*||v||   tol*||A~||   TRUE rel err
       5     4.02e-01     2.23e-06      7.8237e-01
       6     4.35e-02     2.23e-06      7.8229e-01
       9     7.62e-05     2.23e-06      7.8242e-01
      13     8.64e-07     2.23e-06      7.8242e-01   <- stops here

The true error is FROZEN while the tested quantity falls six orders of
magnitude, so the two are anti-correlated and no threshold on the tested one
separates "converged" from "stuck". At `aca_tol=1e-8` it stays frozen for forty
further ranks before a pivot escapes. That is why the fix is a sampled residual
checked from outside the loop, not a tighter tolerance.
"""

import numpy as np
import pytest

from momwire import BSplineSolver, HMatrixSolver

# The two polylines antennaknobs' skyloop_lmatch hands the solver: a 0.1 m
# feed stub and the loop that closes on it. Kept as a literal so this test has
# no antennaknobs dependency.
FEED_STUB = np.array([[-0.05, 0.0, 15.0], [0.05, 0.0, 15.0]])
LOOP = np.array(
    [
        [0.05, 0.0, 15.0],
        [13.806232, -23.826492, 15.0],
        [13.756232, -23.913095, 15.0],
        [-13.756232, -23.913095, 15.0],
        [-13.806232, -23.826492, 15.0],
        [-0.05, 0.0, 15.0],
    ]
)
KW = dict(
    wires=[FEED_STUB, LOOP],
    n_per_edge_per_wire=[[1], [59, 1, 59, 1, 59]],
    # Driven at 1 V. antennaknobs hands the solver 0 V here and resolves the
    # feed through its L-match network afterwards, which would leave every
    # current zero and every relative error NaN. The stagnation is a property
    # of the Sommerfeld remainder OPERATOR, not of the excitation, so driving
    # the gap directly reproduces it and keeps the test self-contained.
    feeds=[(0, 0.05, 1.0 + 0j)],
    junctions=[[(0, "start"), (1, "end")], [(0, "end"), (1, "start")]],
    wavelength=16.563119226519337,
    wire_radius=0.0005,
    ground_z=0.0,
    ground_eps=(13.0, 0.005),
    ground_model="sommerfeld",
    degree=2,
)


def _z(solver_cls, **extra):
    s = solver_cls(**{**KW, **extra})
    return s, np.asarray(s.compute_impedance()[1])


@pytest.fixture(scope="module")
def dense_currents():
    _s, c = _z(BSplineSolver)
    return c


def _rel(a, b):
    na = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / na) if na else float("nan")


@pytest.mark.slow
def test_the_old_rule_misses_the_stagnation_and_the_new_check_catches_it(
    dense_currents,
):
    """The load-bearing gate: BOTH halves, so it cannot pass vacuously.

    `somm_residual_tol=inf` is exactly the pre-#973 behaviour — the sampled
    check can never fire — so the first assertion re-measures the defect and
    the second measures the fix. Delete the check from
    `_sommerfeld_global_lowrank` and the second assertion fails; weaken it to
    "always fall back" and the run stops being an H-matrix at all, which the
    rank assertion below catches.
    """
    _s_off, c_off = _z(HMatrixSolver, aca_tol=1e-6, somm_residual_tol=float("inf"))
    _s_on, c_on = _z(HMatrixSolver, aca_tol=1e-6)

    err_off = _rel(c_off, dense_currents)
    err_on = _rel(c_on, dense_currents)

    assert err_off > 1e-3, (
        f"the pre-#973 rule should MISS this ({err_off:.3e}); if it no longer "
        "does, the stagnation premise moved and this gate is measuring nothing"
    )
    assert err_on < 1e-4, f"the sampled check should catch it, got {err_on:.3e}"
    assert err_on < err_off / 100.0


@pytest.mark.slow
def test_the_fallback_fires_on_this_geometry_and_says_so():
    s, _ = _z(HMatrixSolver, aca_tol=1e-6)
    assert s._last_somm_fallback is True
    assert s._last_somm_residual > 0.5, s._last_somm_residual


@pytest.mark.slow
def test_the_check_is_flat_across_aca_tol_where_the_old_rule_had_a_step(
    dense_currents,
):
    """Six rungs. The pre-#973 behaviour is a plateau at ~4.8e-3 from 1e-4 to
    1e-7 and a cliff at 1e-8; with the check every rung is small."""
    tols = (1e-4, 1e-5, 1e-6, 1e-7, 3e-8, 1e-8)
    errs = []
    for t in tols:
        _s, c = _z(HMatrixSolver, aca_tol=t)
        errs.append(_rel(c, dense_currents))
    assert max(errs) < 1e-4, dict(zip(tols, errs, strict=True))
    # and the step is gone: no rung is 100x another
    assert max(errs) / max(min(errs), 1e-18) < 100.0, dict(zip(tols, errs, strict=True))


# --- #984: the threshold sits in a measured window, not on a slope ---------
# Both edges are measurements, so a future move that crosses either one fails
# here rather than silently changing which decks fall back.
HEALTHY_MAX_PROBE = 2.52e-3  # loops.triangular_skyloop, measured on this revision
MISSED_CELL_PROBE = 9.74e-3  # skyloop d=1 N=90, #984 (|dZ|/|Z| 1.26e-3)


def test_g984_the_threshold_catches_the_missed_cell_and_no_healthy_deck():
    from momwire.hmatrix import DEFAULT_SOMM_RESIDUAL_TOL as tol

    assert tol > HEALTHY_MAX_PROBE, (
        f"tol {tol:g} would fall back loops.triangular_skyloop, which probes "
        f"{HEALTHY_MAX_PROBE:g} and is healthy -- a false positive is a silent "
        "catalog-wide dense fill"
    )
    assert tol < MISSED_CELL_PROBE, (
        f"tol {tol:g} no longer catches the #984 cell at {MISSED_CELL_PROBE:g}"
    )


def test_g984_a_healthy_geometry_still_does_not_fall_back():
    """The absence assertion. Without it the threshold could 'work' by
    falling back everywhere, which is a dense fill wearing a check."""
    s, _ = _z(HMatrixSolver, aca_tol=1e-6)  # the stagnating skyloop
    assert s._last_somm_fallback is True

    # The 52-basis rung of the same geometry is healthy (probe 1.24e-3,
    # |dZ| 1.18e-05) and must stay served by the low-rank route at 4e-3.
    coarse = dict(KW)
    coarse["n_per_edge_per_wire"] = [[1], [15, 1, 15, 1, 15]]
    healthy = HMatrixSolver(**coarse, aca_tol=1e-6)
    healthy.compute_impedance()
    assert healthy._last_somm_fallback is False, healthy._last_somm_residual
    assert healthy._last_somm_residual < HEALTHY_MAX_PROBE


# --- #984: the gate that names decks, not constants ------------------------
# The PRIMARY check. When the healthy edge drifts again -- it moved 1.66e-3 ->
# 2.52e-3 on loops.triangular_skyloop between two momwire revisions with its
# |dZ| unchanged -- a red here names a deck and a verdict, so the fix is a
# re-measure of one number rather than an argument about what the number meant.
#
# Verdicts measured at DEFAULT_SOMM_RESIDUAL_TOL = 4e-3:
#     triangular_skyloop     probe 2.522e-03  ->  served   (rel dZ 2.60e-06)
#     skyloop_lmatch         probe 1.205e+00  ->  fallback (rel dZ 4.67e-07)
#     dipole_turnstile       probe 1.000e+00  ->  fallback (rel dZ 1.24e-10)
#     horizontal_loop        probe 9.735e-01  ->  fallback (rel dZ 2.62e-06)
#     rhombic                probe 3.002e-05  ->  served   (rel dZ 2.66e-07)
#     diamond_loop_turnstile probe 1.205e-04  ->  served   (rel dZ 1.64e-06)
FALLBACK_VERDICTS = {
    "loops.triangular_skyloop": False,
    "loops.skyloop_lmatch": True,
    "dipoles.dipole_turnstile": True,
    "loops.horizontal_loop": True,
    "wire.rhombic": False,
    "loops.diamond_loop_turnstile": False,
}


@pytest.mark.slow
@pytest.mark.parametrize(
    ("design", "expect_fallback"), sorted(FALLBACK_VERDICTS.items())
)
def test_g984_the_catalog_verdicts_are_what_the_threshold_promises(
    design, expect_fallback
):
    pytest.importorskip("antennaknobs", reason="the decks live in antennaknobs")
    import importlib

    import momwire.hmatrix as HM
    from antennaknobs.engines.momwire import MomwireEngine

    builder_cls = importlib.import_module(f"antennaknobs.designs.{design}").Builder
    b = builder_cls()
    b.nominal_nsegs = b.nominal_nsegs * 2

    seen = []
    original = HM._sampled_residual

    def spy(*a, **kw):
        out = original(*a, **kw)
        seen.append(out[0])
        return out

    HM._sampled_residual = spy
    try:
        MomwireEngine(
            b,
            solver=HMatrixSolver,
            solver_kwargs={"degree": 2},
            ground=("finite", 13.0, 0.005),
        ).impedance()
    finally:
        HM._sampled_residual = original

    assert seen, f"{design}: no global Sommerfeld factorisation was probed"
    probe = max(seen)
    got = probe > HM.DEFAULT_SOMM_RESIDUAL_TOL
    assert got is expect_fallback, (
        f"{design}: probe {probe:.3e} against tol "
        f"{HM.DEFAULT_SOMM_RESIDUAL_TOL:g} gives fallback={got}, expected "
        f"{expect_fallback}. If this deck is still healthy, re-measure its "
        "probe and move the threshold; do not flip the expectation."
    )
