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
