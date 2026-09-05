"""momwire#898 — the column fill runs with the BLAS pool pinned to ONE
thread, and releases it afterwards.

The defect was a timing one (the route ran 2× SLOWER at the default eight
threads than at one, because OpenBLAS threads its small per-column gemm
and then spins, stealing the next column's numpy and scipy work), but the
gate here is deterministic: it reads the BLAS thread count from INSIDE the
fill, through the same `threadpoolctl` the pin uses, and again after. A
timing gate would be a statement about the box; this one is a statement
about the code. The timing itself is probe4 (`scratch/893-study`), run on
a box, and its number is in the #898 record.
"""

import numpy as np
import pytest
from threadpoolctl import ThreadpoolController

from momwire import _ground_refl, _near_interface as ni

C0 = 299792458.0
F7 = 7e6
K7 = 2.0 * np.pi * F7 / C0
OM7 = 2.0 * np.pi * F7
EPS0 = 8.8541878128e-12
SOIL_A = _ground_refl.eps_tilde((13.0, 0.005), OM7, EPS0)


def _blas_threads():
    """Thread count of every loaded BLAS, as `threadpoolctl` reads it now."""
    return [
        i["num_threads"]
        for i in ThreadpoolController().info()
        if i["user_api"] == "blas"
    ]


def test_g898_1_column_fill_pins_blas_to_one_thread_and_releases(monkeypatch):
    """Inside the column loop every BLAS reports one thread; after the
    call every BLAS is back to what it was. Read through a spy on
    `six_columns`, so the observation is made exactly where the gemm runs."""
    before = _blas_threads()
    if not before:
        pytest.skip("no BLAS loaded that threadpoolctl can see")
    seen, real = [], ni.six_columns

    def spy(*args, **kw):
        seen.append(_blas_threads())
        return real(*args, **kw)

    monkeypatch.setattr(ni, "six_columns", spy)
    monkeypatch.setattr(ni, "_ROUTE", "column")
    rho = np.array([0.3, 0.5, 13.6])
    z = np.array([[0.2], [1.0], [4.0]])
    ni.designed_tables(SOIL_A, K7, rho, z, -0.2)
    assert seen, "the column route did not serve"
    assert all(all(n == 1 for n in counts) for counts in seen), seen
    assert _blas_threads() == before


def test_g898_2_point_route_does_not_touch_the_pool(monkeypatch):
    """The pin is the column route's, not the module's: on the point route
    the controller is never consulted, so a caller who chose that route
    for a bisect sees the pool untouched."""
    monkeypatch.setattr(ni, "_ROUTE", "point")
    calls = []
    monkeypatch.setattr(ni, "_blas_single_thread", lambda: calls.append(1))
    ni.designed_tables(SOIL_A, K7, 0.3, 0.2, -0.2)
    assert calls == []


def test_g898_3_controller_is_built_once():
    """One library scan per process (~0.7 ms); the context itself is what
    every fill pays (~20 µs)."""
    assert ni._blas_controller() is ni._blas_controller()
