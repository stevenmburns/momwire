"""`momwire._vec3`: 3-vector length and dot product in one fixed order.

The order is ``(x*x + y*y) + z*z``, each operation rounded on its own. That
is what numpy's ``linalg.norm(axis=-1)`` computes for a ``(..., 3)`` array on
every machine, and what BLAS ``ddot`` computes for a single 3-vector on the
reference machines (Skylake/Haswell) — but NOT on AVX-512 ones, where ddot
fuses. These tests pin the order itself (against exact rational arithmetic)
and the array-path identity; the ddot identity is machine-dependent by
nature and is recorded in the module docstring instead.
"""

import math
from fractions import Fraction

import numpy as np
import pytest

from momwire._vec3 import dot3, norm3


def _plain(u, v):
    # Each step rounded to double, left to right, no FMA: Python floats.
    return (u[0] * v[0] + u[1] * v[1]) + u[2] * v[2]


@pytest.fixture(scope="module")
def vecs():
    rng = np.random.default_rng(1479)
    scale = rng.uniform(1e-3, 1e3, (4000, 1))
    return rng.standard_normal((4000, 3)) * scale, rng.standard_normal((4000, 3))


def test_array_norm_is_numpys_axis_norm_to_the_bit(vecs):
    a, _ = vecs
    assert norm3(a).tobytes() == np.linalg.norm(a, axis=1).tobytes()
    b = a.reshape(40, 100, 3)
    assert norm3(b).tobytes() == np.linalg.norm(b, axis=-1).tobytes()


def test_single_vector_is_the_plain_left_to_right_sum(vecs):
    a, b = vecs
    for u, v in zip(a[:500].tolist(), b[:500].tolist(), strict=True):
        assert norm3(u) == math.sqrt(_plain(u, u))
        assert dot3(u, v) == _plain(u, v)
        assert norm3(np.array(u)) == norm3(u)


def test_order_is_not_fused():
    # A vector where the fused and the plain orders disagree: the plain
    # answer is the one returned.
    rng = np.random.default_rng(7)
    for u in rng.standard_normal((2000, 3)).tolist():
        p = [Fraction(c) for c in u]
        fused = float(
            p[2] * p[2] + Fraction(float(p[1] * p[1] + Fraction(u[0] * u[0])))
        )
        plain = _plain(u, u)
        if fused != plain:
            assert dot3(u, u) == plain
            return
    pytest.fail("no vector separated the two orders")


def test_array_dot_matches_single_vector_dot(vecs):
    a, b = vecs
    got = dot3(a, b)
    want = np.array([_plain(u, v) for u, v in zip(a.tolist(), b.tolist(), strict=True)])
    assert got.tobytes() == want.tobytes()
    assert dot3(a[:, None, :], b[None, :5, :]).shape == (4000, 5)


def test_wrong_trailing_axis_is_refused():
    with pytest.raises(ValueError):
        norm3(np.zeros((4, 2)))
    with pytest.raises(ValueError):
        norm3([1.0, 2.0])
    with pytest.raises(ValueError):
        dot3(np.zeros((4, 3)), np.zeros((4, 2)))
