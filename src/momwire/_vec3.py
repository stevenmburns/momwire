"""Length and dot product of 3-vectors, in one fixed order on every CPU.

The solver's geometry (edge and segment lengths, junction and feed
tolerances, tangent dots on single vectors) used to go through
``np.linalg.norm(v)`` and ``np.dot(a, b)`` on single 3-vectors. Both hand
the work to BLAS ``ddot``, whose answer depends on the machine: on
Skylake/Haswell OpenBLAS sums ``(a0*b0 + a1*b1) + a2*b2`` with plain
roundings, on AVX-512 machines (Rocketlake, GitHub's runners) it fuses the
products into FMAs, and on the 453-deck corpus that moved 34 decks'
razor-2p geometry. A reference that a second implementation (the C++
standalone path) has to match bit for bit cannot depend on which BLAS
kernel the host picked.

The spelling here is the plain one, ``(x*x + y*y) + z*z`` with every
operation rounded on its own, and then a correctly rounded square root.
It is bit for bit what:

* BLAS ``ddot`` returns on Skylake and Haswell (the reference machines), so
  replacing a 1-D ``np.linalg.norm``/``np.dot`` with it moves no bit there;
* ``np.linalg.norm(x, axis=-1)`` returns for a ``(..., 3)`` array on every
  machine (numpy's ``add.reduce`` over a length-3 axis sums left to right),
  so the array sites keep their bits everywhere.

Python floats and numpy ufuncs never contract a multiply and an add into
an FMA, so the order written is the order computed. Products of 3-vector
MATRICES (``tangents @ tangents.T``) are deliberately not covered: OpenBLAS
``dgemm`` is an FMA chain there, and an unfused spelling would move bits on
the reference machines too.
"""

import math

import numpy as np

__all__ = ["norm3", "dot3"]


def norm3(v):
    """``|v|`` over the last axis of a real ``(..., 3)`` array.

    A single 3-vector (any sequence of three reals) gives a Python
    ``float``; an array of them gives an array of the leading shape.
    """
    a = np.asarray(v, dtype=np.float64)
    if a.ndim == 1:
        x, y, z = a.tolist()
        return math.sqrt((x * x + y * y) + z * z)
    if a.shape[-1] != 3:
        raise ValueError(f"norm3 wants a trailing axis of 3, got shape {a.shape}")
    x, y, z = a[..., 0], a[..., 1], a[..., 2]
    return np.sqrt((x * x + y * y) + z * z)


def dot3(u, v):
    """``u . v`` over the last axis of two real ``(..., 3)`` arrays (which
    broadcast). Two single 3-vectors give a Python ``float``."""
    a = np.asarray(u, dtype=np.float64)
    b = np.asarray(v, dtype=np.float64)
    if a.ndim == 1 and b.ndim == 1:
        x0, y0, z0 = a.tolist()
        x1, y1, z1 = b.tolist()
        return (x0 * x1 + y0 * y1) + z0 * z1
    if a.shape[-1] != 3 or b.shape[-1] != 3:
        raise ValueError(f"dot3 wants trailing axes of 3, got {a.shape}, {b.shape}")
    return (a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]) + a[..., 2] * b[..., 2]
