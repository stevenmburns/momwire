"""The one seam every dense LU factor and solve on the Z path goes through,
and every dense complex matrix product on the NEC-5 run path.

Why it exists. momwire factors its moment matrices with whatever LAPACK the
installed scipy and numpy carry -- and those are two different libraries
(scipy's wheel bundles one OpenBLAS build, numpy's another). A host process
that embeds Python (the planned standalone executable) wants the whole
process to factor with ONE library: its own. This module is where it plugs in.

DEFAULT (nothing published): every function here calls exactly what the
call site called before the seam existed, with the same arguments, so no bit
moves anywhere:

==========================  ===========================================
here                        default call
==========================  ===========================================
:func:`lu_factor`           ``scipy.linalg.lu_factor``
:func:`lu_solve`            ``scipy.linalg.lu_solve``
:func:`solve`               ``scipy.linalg.solve`` (default arguments)
:func:`solve_stacked`       ``numpy.linalg.solve``
:func:`inv_stacked`         ``numpy.linalg.inv``
:func:`matmul`              ``a @ b``
==========================  ===========================================

The scipy entry points are looked up on ``scipy.linalg`` at call time, so a
test that wraps ``scipy.linalg.lu_factor`` still sees every factorization.

HOST ROUTE. A host publishes its routines BEFORE momwire factors anything,
as ``sys._mw_host_lapack``, a mapping::

    {
        "abi": 2,             # or 1: the LU routines only
        "zgetrf": <address>,  # required
        "zgetrs": <address>,  # required
        "dgetrf": <address>,  # optional, real float64
        "dgetrs": <address>,  # optional, real float64
        "zgemm": <address>,   # ABI 2: required
        "zgemv": <address>,   # ABI 2: required
    }

Each address is an integer function pointer with the LP64 (32-bit
integer) Fortran calling convention of OpenBLAS's BLAS / LAPACK interface
(``zgetrf_``, ``zgemm_``, ...): every argument by reference, no hidden
string-length arguments. ABI 1's routines::

    void zgetrf(int *m, int *n, double _Complex *a, int *lda,
                int *ipiv, int *info);
    void zgetrs(char *trans, int *n, int *nrhs, double _Complex *a,
                int *lda, int *ipiv, double _Complex *b, int *ldb,
                int *info);

(``dgetrf`` / ``dgetrs`` the same over ``double``.) Matrices are passed
column-major; ``ipiv`` is LAPACK's 1-based pivot vector.

ABI 2 adds the two BLAS products, Fortran ``zgemm`` / ``zgemv``::

    void zgemm(char *transa, char *transb, int *m, int *n, int *k,
               double _Complex *alpha, double _Complex *a, int *lda,
               double _Complex *b, int *ldb, double _Complex *beta,
               double _Complex *c, int *ldc);
    void zgemv(char *trans, int *m, int *n, double _Complex *alpha,
               double _Complex *a, int *lda, double _Complex *x, int *incx,
               double _Complex *beta, double _Complex *y, int *incy);

and :func:`matmul` asks for exactly ONE call per shape, always with
``alpha = 1``, ``beta = 0``, unit increments and tight leading dimensions,
on C-contiguous (row-major) complex128 copies of the operands -- a float64
operand is cast first, as numpy's matmul casts it. A row-major ``(r, c)``
array read column-major is its ``(c, r)`` transpose, so:

==========================  ===================================================
``a @ b``                   the host call
==========================  ===================================================
(m, k) @ (k, n) -> (m, n)   ``zgemm('N', 'N', n, m, k, 1, b, n, a, k, 0, c, n)``
(m, k) @ (k,)   -> (m,)     ``zgemv('T', k, m, 1, a, k, x, 1, 0, y, 1)``
(k,) @ (k, n)   -> (n,)     ``zgemv('N', n, k, 1, b, n, x, 1, 0, y, 1)``
(k,) @ (k,)     -> scalar   ``zgemv('T', k, 1, 1, a, k, b, 1, 0, y, 1)``
==========================  ===================================================

The table is fixed by shape alone, so an executable that replays a Python
run makes the same call on the same library and gets the same bits. numpy's
own matmul makes these calls in the general case but not always (a dot for
a 1 x 1 result, a gemv when a matrix has one row or column, a transpose
flag in place of a copy), so a host route never promises numpy's bits --
only its own library's, the same way every time.
Anything else -- a real or single-precision product, a stacked (3-D) or
empty operand -- takes ``a @ b``. An ABI 1 table changes no product.

With a host published, every call whose arrays the host can take (complex128,
or float64 when the real pair is present; a square 2-D matrix per slice;
non-empty) factors with the host's getrf and solves with its getrs. Anything
else -- another dtype, a rectangular or empty matrix, a 1 x 1
``scipy.linalg.solve``, which scipy answers by a plain division without
LAPACK -- takes the default call. The route keeps the replaced call's
contract:

- ``lu_factor``/``lu_solve``: ``check_finite`` (``ValueError``), the
  ``LinAlgWarning`` on an exactly zero pivot, the 0-based ``piv`` that
  ``scipy.linalg.lu_solve`` accepts (factors from either route interoperate),
  ``overwrite_a`` factoring an F-contiguous complex128 matrix in place,
  ``trans`` 0/1/2, one or many right-hand sides.
- ``solve``: ``check_finite``, ``LinAlgError`` on an exactly singular
  matrix, and the ill-conditioning ``LinAlgWarning`` when the reciprocal
  condition estimate falls below machine epsilon. The estimate (a 1-norm and
  gecon) comes from scipy's LAPACK reading the host's factors; it only
  decides the warning, never a value.
- ``solve_stacked``/``inv_stacked``: numpy's broadcasting of stacked
  matrices, a 1-D right-hand side read as a vector, the common result dtype,
  ``LinAlgError`` on an exactly singular slice, and no finiteness check.

``numpy.linalg.solve`` is gesv, which is getrf then getrs; ``numpy.linalg.inv``
is gesv against the identity. So the host route runs the same algorithm, on
the host's library. What it cannot promise is numpy's or scipy's bits: those
are other libraries, which is the point of publishing one.

:data:`HOST_CALLS` counts the host's getrf, getrs, gemm and gemv calls.

NOT HERE: numpy's ELEMENTWISE complex multiply. numpy picks its loop by CPU
at run time, and its x86-64-v3 / AVX-512 / ASIMD loops round
``a.re*b.re - a.im*b.im`` once (a fused multiply-add) where its baseline
x86-64 loop rounds twice, so no published library decides those bits. The
run path's complex elementwise products go through :func:`momwire._cmul.cmul`
instead, which always fuses with numpy's formula: a host that replays a run
fuses too, whatever loop the embedded numpy would have taken.
"""

from __future__ import annotations

import ctypes
import sys
import warnings

import numpy as np
import scipy.linalg
from scipy.linalg import LinAlgWarning

ABI = 2
#: Every table version this module reads: ABI 1 (the LU pair only) and ABI 2
#: (the LU pair and the products).
ABIS = (1, 2)

#: Host calls since import (tests read it to prove the route ran).
HOST_CALLS = {"getrf": 0, "getrs": 0, "gemm": 0, "gemv": 0}

_INT = ctypes.c_int
_P_INT = ctypes.POINTER(_INT)
_PTR = ctypes.c_void_p
_GETRF = ctypes.CFUNCTYPE(None, _P_INT, _P_INT, _PTR, _P_INT, _PTR, _P_INT)
_GETRS = ctypes.CFUNCTYPE(
    None, ctypes.c_char_p, _P_INT, _P_INT, _PTR, _P_INT, _PTR, _PTR, _P_INT, _P_INT
)
# zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
_GEMM = ctypes.CFUNCTYPE(
    None,
    ctypes.c_char_p,
    ctypes.c_char_p,
    _P_INT,
    _P_INT,
    _P_INT,
    _PTR,
    _PTR,
    _P_INT,
    _PTR,
    _P_INT,
    _PTR,
    _PTR,
    _P_INT,
)
# zgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
_GEMV = ctypes.CFUNCTYPE(
    None,
    ctypes.c_char_p,
    _P_INT,
    _P_INT,
    _PTR,
    _PTR,
    _P_INT,
    _PTR,
    _P_INT,
    _PTR,
    _PTR,
    _P_INT,
)
_TRANS = {0: b"N", 1: b"T", 2: b"C"}
_C128 = np.dtype(np.complex128)
_INT_MAX = 2**31 - 1

# (the published table, its LU routines by dtype, its (zgemm, zgemv) or None)
_bound: tuple[object, dict, tuple | None] | None = None


def _bind(table):
    """Check a published table and bind its routines (once per table)."""
    try:
        abi = table["abi"]
    except (KeyError, TypeError):
        raise RuntimeError(
            "the host's LAPACK table names no ABI version; momwire reads ABI 1 "
            "and ABI 2"
        ) from None
    if abi not in ABIS:
        raise RuntimeError(
            f"the host's LAPACK table is ABI {abi}; momwire reads ABI 1 and ABI 2"
        )
    routines = {}
    for dtype, prefix, required in (
        (_C128, "z", True),
        (np.dtype(np.float64), "d", False),
    ):
        getrf, getrs = table.get(prefix + "getrf"), table.get(prefix + "getrs")
        if getrf is None and getrs is None and not required:
            continue
        if not getrf or not getrs:
            raise RuntimeError(
                f"the host's LAPACK table must give both {prefix}getrf and "
                f"{prefix}getrs"
            )
        routines[dtype] = (_GETRF(int(getrf)), _GETRS(int(getrs)))
    products = None
    if abi >= 2:
        gemm, gemv = table.get("zgemm"), table.get("zgemv")
        if not gemm or not gemv:
            raise RuntimeError(
                "the host's LAPACK table is ABI 2, which must give both zgemm and zgemv"
            )
        products = (_GEMM(int(gemm)), _GEMV(int(gemv)))
    return routines, products


def _host_bound():
    """``(LU routines by dtype, products or None)`` of the published table, or
    None when nothing is published. Re-read on every call (a getattr), bound
    once per table."""
    global _bound
    table = getattr(sys, "_mw_host_lapack", None)
    if table is None:
        return None
    if _bound is None or _bound[0] is not table:
        _bound = (table, *_bind(table))
    return _bound[1], _bound[2]


def _host_for(dtype):
    """The host's (getrf, getrs) for `dtype`, or None for the default route."""
    bound = _host_bound()
    if bound is None:
        return None
    return bound[0].get(np.dtype(dtype))


def _host_products():
    """The host's (zgemm, zgemv), or None for the default route (nothing
    published, or an ABI 1 table)."""
    bound = _host_bound()
    return None if bound is None else bound[1]


def _getrf(fns, a):
    """Factor the F-contiguous square `a` in place; returns (ipiv, info)."""
    n = a.shape[0]
    ipiv = np.empty(n, dtype=np.int32)
    nn, info = _INT(n), _INT(0)
    HOST_CALLS["getrf"] += 1
    fns[0](
        ctypes.byref(nn),
        ctypes.byref(nn),
        a.ctypes.data,
        ctypes.byref(nn),
        ipiv.ctypes.data,
        ctypes.byref(info),
    )
    return ipiv, info.value


def _getrs(fns, lu, ipiv, b, trans=0):
    """Solve in place into the F-contiguous (n, nrhs) `b`; returns info."""
    n = lu.shape[0]
    nn, nrhs, info = _INT(n), _INT(b.shape[1]), _INT(0)
    HOST_CALLS["getrs"] += 1
    fns[1](
        _TRANS[trans],
        ctypes.byref(nn),
        ctypes.byref(nrhs),
        lu.ctypes.data,
        ctypes.byref(nn),
        ipiv.ctypes.data,
        b.ctypes.data,
        ctypes.byref(nn),
        ctypes.byref(info),
    )
    return info.value


def _fortran_copy(x, dtype):
    return np.array(x, dtype=dtype, order="F", copy=True)


def _square(a):
    return a.ndim == 2 and a.shape[0] == a.shape[1] and a.shape[0] > 0


# ---------------------------------------------------------------------------
# scipy.linalg.lu_factor / lu_solve
# ---------------------------------------------------------------------------


def lu_factor(a, overwrite_a=False, check_finite=True):
    """``scipy.linalg.lu_factor``, on the host's getrf when one is published."""
    a1 = np.asarray(a)
    fns = _host_for(a1.dtype)
    if fns is None or not _square(a1):
        return scipy.linalg.lu_factor(
            a, overwrite_a=overwrite_a, check_finite=check_finite
        )
    if check_finite and not np.isfinite(a1).all():
        raise ValueError("array must not contain infs or NaNs")
    if not (overwrite_a and a1.flags.f_contiguous and a1.flags.writeable):
        a1 = _fortran_copy(a1, a1.dtype)
    ipiv, info = _getrf(fns, a1)
    if info < 0:
        raise ValueError(
            f"illegal value in argument {-info} of the host's getrf (lu_factor)"
        )
    if info > 0:
        warnings.warn(
            f"Diagonal number {info} is exactly zero. Singular matrix.",
            LinAlgWarning,
            stacklevel=2,
        )
    return a1, ipiv - 1


def lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True):
    """``scipy.linalg.lu_solve``, on the host's getrs when one is published."""
    lu, piv = lu_and_piv
    lu = np.asarray(lu)
    b0 = np.asarray(b)
    fns = _host_for(lu.dtype)
    if (
        fns is None
        or not _square(lu)
        or b0.ndim not in (1, 2)
        or b0.size == 0
        or np.result_type(lu.dtype, b0.dtype) != lu.dtype
        or trans not in _TRANS
    ):
        return scipy.linalg.lu_solve(
            lu_and_piv,
            b,
            trans=trans,
            overwrite_b=overwrite_b,
            check_finite=check_finite,
        )
    if check_finite and not np.isfinite(b0).all():
        raise ValueError("array must not contain infs or NaNs")
    n = lu.shape[0]
    if b0.shape[0] != n:
        raise ValueError(
            f"the right-hand side has {b0.shape[0]} rows; the factors are {n} by {n}"
        )
    if not lu.flags.f_contiguous:
        lu = np.asfortranarray(lu)
    b2 = b0.reshape(n, 1) if b0.ndim == 1 else b0
    in_place = (
        overwrite_b
        and b0.ndim == 2
        and b2.dtype == lu.dtype
        and b2.flags.f_contiguous
        and b2.flags.writeable
    )
    x = b2 if in_place else _fortran_copy(b2, lu.dtype)
    ipiv = np.asarray(piv, dtype=np.int32) + 1
    info = _getrs(fns, lu, ipiv, x, trans)
    if info != 0:
        raise ValueError(
            f"illegal value in argument {-info} of the host's getrs (lu_solve)"
        )
    return x.reshape(b0.shape)


# ---------------------------------------------------------------------------
# scipy.linalg.solve (default arguments: general matrix, check_finite)
# ---------------------------------------------------------------------------


def solve(a, b):
    """``scipy.linalg.solve(a, b)``, on the host's getrf + getrs when one is
    published."""
    a1 = np.asarray(a)
    b0 = np.asarray(b)
    dtype = np.result_type(a1.dtype, b0.dtype)
    fns = _host_for(dtype)
    if (
        fns is None
        or not _square(a1)
        or a1.shape[0] == 1
        or b0.ndim not in (1, 2)
        or b0.size == 0
        or b0.shape[0] != a1.shape[0]
    ):
        return scipy.linalg.solve(a, b)
    if not (np.isfinite(a1).all() and np.isfinite(b0).all()):
        raise ValueError("array must not contain infs or NaNs")
    lu = _fortran_copy(a1, dtype)
    lange, gecon = scipy.linalg.get_lapack_funcs(("lange", "gecon"), (lu,))
    anorm = lange("1", lu)  # for the warning only, as scipy's solve takes it
    ipiv, info = _getrf(fns, lu)
    if info != 0:
        if info > 0:
            raise np.linalg.LinAlgError("singular matrix")
        raise ValueError(f"illegal value in argument {-info} of the host's getrf")
    n = a1.shape[0]
    x = _fortran_copy(b0.reshape(n, 1) if b0.ndim == 1 else b0, dtype)
    info = _getrs(fns, lu, ipiv, x)
    if info != 0:
        raise ValueError(f"illegal value in argument {-info} of the host's getrs")
    rcond, _info = gecon(lu, anorm, norm="1")
    if rcond < np.finfo(dtype).eps:
        warnings.warn(
            f"An ill-conditioned matrix detected: rcond = {rcond:.6g}.",
            LinAlgWarning,
            stacklevel=2,
        )
    return x.reshape(b0.shape)


# ---------------------------------------------------------------------------
# numpy.linalg.solve / numpy.linalg.inv (stacked)
# ---------------------------------------------------------------------------


def _stacked_host(a1, dtype):
    if a1.ndim < 2 or a1.shape[-1] != a1.shape[-2] or a1.size == 0:
        return None
    return _host_for(dtype)


def _solve_slices(fns, a, b, dtype):
    """getrf + getrs over the broadcast stack: a (..., n, n), b (..., n, k)."""
    batch = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    a = np.broadcast_to(a, batch + a.shape[-2:])
    b = np.broadcast_to(b, batch + b.shape[-2:])
    out = np.empty(batch + b.shape[-2:], dtype=dtype)
    for idx in np.ndindex(*batch):
        lu = _fortran_copy(a[idx], dtype)
        ipiv, info = _getrf(fns, lu)
        if info > 0:
            raise np.linalg.LinAlgError("Singular matrix")
        if info < 0:
            raise ValueError(f"illegal value in argument {-info} of the host's getrf")
        x = _fortran_copy(b[idx], dtype)
        info = _getrs(fns, lu, ipiv, x)
        if info != 0:
            raise ValueError(f"illegal value in argument {-info} of the host's getrs")
        out[idx] = x
    return out


def solve_stacked(a, b):
    """``numpy.linalg.solve(a, b)``, on the host's getrf + getrs when one is
    published."""
    a1 = np.asarray(a)
    b1 = np.asarray(b)
    dtype = np.result_type(a1.dtype, b1.dtype, np.float64)
    fns = _stacked_host(a1, dtype)
    if fns is None or b1.size == 0:
        return np.linalg.solve(a, b)
    if b1.ndim == 1:
        if b1.shape[0] != a1.shape[-1]:
            return np.linalg.solve(a, b)  # numpy's own shape error
        return _solve_slices(fns, a1, b1[:, None], dtype)[..., 0]
    if b1.shape[-2] != a1.shape[-1]:
        return np.linalg.solve(a, b)
    try:
        np.broadcast_shapes(a1.shape[:-2], b1.shape[:-2])
    except ValueError:
        return np.linalg.solve(a, b)
    return _solve_slices(fns, a1, b1, dtype)


def inv_stacked(a):
    """``numpy.linalg.inv(a)``, on the host's getrf + getrs (against the
    identity, as numpy's gesv does) when one is published."""
    a1 = np.asarray(a)
    dtype = np.result_type(a1.dtype, np.float64)
    fns = _stacked_host(a1, dtype)
    if fns is None:
        return np.linalg.inv(a)
    eye = np.eye(a1.shape[-1], dtype=dtype)
    return _solve_slices(fns, a1, eye, dtype)


# ---------------------------------------------------------------------------
# the matrix products: a @ b (ABI 2)
# ---------------------------------------------------------------------------

_ONE = np.ones(1, dtype=_C128)  # alpha
_ZERO = np.zeros(1, dtype=_C128)  # beta


def _gemm(fns, m, n, k, a, b, c):
    """``c = a @ b`` for C-contiguous complex128 a (m, k), b (k, n), c (m, n):
    column-major, that is ``c^T = b^T a^T``, so the host is asked for
    ``zgemm('N', 'N', n, m, k, 1, b, n, a, k, 0, c, n)``."""
    mm, nn, kk = _INT(m), _INT(n), _INT(k)
    HOST_CALLS["gemm"] += 1
    fns[0](
        b"N",
        b"N",
        ctypes.byref(nn),
        ctypes.byref(mm),
        ctypes.byref(kk),
        _ONE.ctypes.data,
        b.ctypes.data,
        ctypes.byref(nn),
        a.ctypes.data,
        ctypes.byref(kk),
        _ZERO.ctypes.data,
        c.ctypes.data,
        ctypes.byref(nn),
    )


def _gemv(fns, trans, rows, cols, a, x, y):
    """``y = op(a) x`` for the column-major complex128 ``a`` (rows, cols),
    contiguous ``x`` and ``y``: ``zgemv(trans, rows, cols, 1, a, rows, x, 1,
    0, y, 1)``."""
    r, c, one = _INT(rows), _INT(cols), _INT(1)
    HOST_CALLS["gemv"] += 1
    fns[1](
        trans,
        ctypes.byref(r),
        ctypes.byref(c),
        _ONE.ctypes.data,
        a.ctypes.data,
        ctypes.byref(r),
        x.ctypes.data,
        ctypes.byref(one),
        _ZERO.ctypes.data,
        y.ctypes.data,
        ctypes.byref(one),
    )


def matmul(a, b):
    """``a @ b``, on the host's zgemm / zgemv when an ABI 2 table is published.

    With no table, or an ABI 1 table, this IS ``a @ b``. With an ABI 2 table,
    a product of two 1-D / 2-D non-empty operands whose common dtype is
    complex128 (a float64 operand is cast to complex128 first, as numpy's
    matmul casts it) is computed by the host, on C-contiguous complex128
    copies, with one fixed call per shape (module docstring); the result is
    C-contiguous, as numpy's is. Anything else takes ``a @ b``.
    """
    fns = _host_products()
    if fns is None:
        return a @ b
    a1, b1 = np.asarray(a), np.asarray(b)
    if (
        a1.ndim not in (1, 2)
        or b1.ndim not in (1, 2)
        or np.result_type(a1.dtype, b1.dtype) != _C128
        or a1.shape[-1] != b1.shape[0]
        or a1.size == 0
        or b1.size == 0
        or max(a1.shape + b1.shape) > _INT_MAX
    ):
        return a @ b
    a2 = np.ascontiguousarray(a1, dtype=_C128)
    b2 = np.ascontiguousarray(b1, dtype=_C128)
    k = b2.shape[0]
    if a2.ndim == 2 and b2.ndim == 2:
        m, n = a2.shape[0], b2.shape[1]
        out = np.zeros((m, n), dtype=_C128)
        _gemm(fns, m, n, k, a2, b2, out)
        return out
    if a2.ndim == 2:  # (m, k) @ (k,): a's rows are columns of its transpose
        out = np.zeros(a2.shape[0], dtype=_C128)
        _gemv(fns, b"T", k, a2.shape[0], a2, b2, out)
        return out
    if b2.ndim == 2:  # (k,) @ (k, n): b read column-major is b^T
        out = np.zeros(b2.shape[1], dtype=_C128)
        _gemv(fns, b"N", b2.shape[1], k, b2, a2, out)
        return out
    out = np.zeros(1, dtype=_C128)  # (k,) @ (k,): a as a k x 1 column
    _gemv(fns, b"T", k, 1, a2, b2, out)
    return out[0]
