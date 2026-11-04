"""momwire#1799 PROTOTYPE: the junction constraints eliminated before the solve.

With MOMWIRE_BS2_JUNCTION_REDUCE=1, `compute_port_solution` solves
T^T Z T y = T^T B (T the null space of the KCL rows, x = T y) instead of the
saddle-point Schur solve. The two are the same solution in exact arithmetic,
so the gate is roundoff: Y agrees to 1e-10 relative, the reduced columns
satisfy the constraints exactly, and n_free = n_basis - n_constraints.
"""

import numpy as np
import pytest

from momwire.bspline import BSplineSolver

from fixtures_refl_coef_geoms import GEOMS

SOMM = {"ground_eps": (10.0, 0.002), "ground_model": "sommerfeld"}


def test_null_space_spans_the_constraints():
    rng = np.random.default_rng(1799)
    n = 40
    A = np.zeros((6, n))
    cols = rng.permutation(n)[:18].reshape(6, 3)
    for r, c in enumerate(cols):
        A[r, c] = rng.choice([-1.0, 1.0], size=3)
    T, free = BSplineSolver._junction_null_space(A, n)
    assert T.shape == (n, n - 6) and free.size == n - 6
    assert np.abs(A @ T.toarray()).max() == 0.0
    assert np.linalg.matrix_rank(T.toarray()) == n - 6


@pytest.mark.parametrize(
    "name, frac", [("inverted_l", 0.1), ("yagi", 0.2), ("inverted_l", 0.02)]
)
def test_reduced_solve_is_the_schur_solve(name, frac, monkeypatch):
    kw = dict(GEOMS[(name, frac)], ground_z=0.0, **SOMM)
    monkeypatch.delenv("MOMWIRE_BS2_JUNCTION_REDUCE", raising=False)
    ref = BSplineSolver(**kw).compute_port_solution()
    monkeypatch.setenv("MOMWIRE_BS2_JUNCTION_REDUCE", "1")
    s = BSplineSolver(**kw)
    got = s.compute_port_solution()
    rec = getattr(s, "_last_junction_reduce", None)
    if rec is None:
        pytest.skip("no junction constraint on this deck")
    assert rec["n_free"] < rec["n_basis"]
    Y0, Y1 = np.asarray(ref.y), np.asarray(got.y)
    assert np.abs(Y1 - Y0).max() <= 1e-10 * np.abs(Y0).max()
