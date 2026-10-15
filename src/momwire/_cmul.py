"""The fused elementwise complex multiply (momwire#1497).

numpy's elementwise complex ``*`` rounds differently by CPU. Its x86-64-v3,
AVX-512 and ARM ASIMD loops fuse each part into one multiply-add::

    re = fma(a.re, b.re, -(a.im * b.im))
    im = fma(a.re, b.im,   a.im * b.re)

and its baseline x86-64 loop rounds the product and the sum separately. So a
product such as ``v * scale`` on the run path had last bits that depended on
which loop numpy picked on the machine at hand. :func:`cmul` is that fused
formula, the same on every CPU: on the machines where numpy fuses it is
numpy's ``*`` to the bit, and on a pre-AVX2 x86 it is the fused answer where
numpy's would not have been.

Broadcasting is numpy's, and the result is a new C-contiguous complex128
array of the broadcast shape, as ``a * b`` gives. A product whose common dtype
is not complex128 (two real arrays, a complex64 one) is ``a * b`` unchanged.

:data:`CALLS` counts the calls by the route each took, so a gate can show the
fused route actually ran.
"""

from __future__ import annotations

import math

import numpy as np

from . import _accel

_C128 = np.dtype(np.complex128)

# Which route each call took: the accelerator's `std::fma`, Python's
# `math.fma`, or numpy's own `*`.
CALLS = {"accel": 0, "math_fma": 0, "numpy": 0}

_fused_cmul = getattr(_accel.acc, "fused_cmul", None) if _accel.LOADED else None
_math_fma = getattr(math, "fma", None)


def _python_fused(x, y):
    """The fused formula per element with `math.fma`, or None when an element
    is one `math.fma` refuses (it raises on an invalid or overflowing
    operation where C's `fma` returns inf or nan)."""
    out = np.empty(x.shape, dtype=_C128)
    try:
        for i, (p, q) in enumerate(zip(x.tolist(), y.tolist())):
            out[i] = complex(
                _math_fma(p.real, q.real, -(p.imag * q.imag)),
                _math_fma(p.real, q.imag, p.imag * q.real),
            )
    except (ValueError, OverflowError):
        return None
    return out


def cmul(a, b):
    """``a * b`` for complex operands, each part one fused multiply-add.

    Without the accelerator this uses ``math.fma`` where Python provides it
    (3.13 and later); on an older Python, or for an element ``math.fma``
    refuses, it is numpy's ``a * b``, whose rounding is then the CPU's again.
    No lane runs without the accelerator, so that fallback is a convenience
    and not a contract.
    """
    if np.result_type(a, b) != _C128:
        CALLS["numpy"] += 1
        return a * b
    x, y = np.broadcast_arrays(np.asarray(a, dtype=_C128), np.asarray(b, dtype=_C128))
    shape = x.shape
    xf = np.ascontiguousarray(x).reshape(-1)
    yf = np.ascontiguousarray(y).reshape(-1)
    out = None
    if _fused_cmul is not None:
        out = _fused_cmul(xf, yf)
        CALLS["accel"] += 1
    elif _math_fma is not None:
        out = _python_fused(xf, yf)
        if out is not None:
            CALLS["math_fma"] += 1
    if out is None:
        CALLS["numpy"] += 1
        return a * b
    out = out.reshape(shape)
    return out[()] if out.ndim == 0 else out
