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
:func:`solve_transposed`    SciPy's getrf on ``a.T`` in place, getrs 'T'
                            (momwire#2071; it replaced ``solve`` at its
                            call sites, not a scipy call)
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
- ``lu_factor_strict``/``factor_and_solve``: ``lu_factor`` under the one
  singular-matrix policy every dense route shares (momwire#1981): an
  exactly singular matrix raises ``LinAlgError("singular matrix")``, as
  ``solve`` does, and the condition estimate runs only under
  :func:`condition_check`.
- ``solve``: ``check_finite``, ``LinAlgError`` on an exactly singular
  matrix, and the ill-conditioning ``LinAlgWarning`` when the reciprocal
  condition estimate (a 1-norm and gecon, scipy's LAPACK reading the
  factors) falls below machine epsilon. The estimate only decides the
  warning, never a value, and it costs about five single-threaded passes
  over the factors (0.09 s a frequency on the whip at n = 4392, against
  0.01-0.025 s for getrs), so it runs only when asked
  (``MOMWIRE_CONDITION_CHECK=1``, :func:`condition_check`) or when the
  free pivot screen (:func:`_pivots_suspect`) says the factors look
  near-singular. A call this route hands to ``scipy.linalg.solve`` itself
  (a 1 x 1, a rectangular matrix, a structure scipy would not solve as
  general) warns as scipy does.
- ``solve_transposed``: ``solve``'s errors, warning, pivot screen and
  condition estimate (rcond₁ of ``a``, from the factors of ``aᵀ``), on
  ``a``'s own storage; its docstring says what differs.
- ``solve`` / ``solve_transposed`` / ``factor_and_solve`` under
  :func:`mixed_lu_scope` (the opt-in ``mixed_lu`` feature, momwire#2125):
  complex64 getrf plus complex128 iterative refinement where it converges,
  else the ordinary route and its bits (:func:`_mixed_refine`). Never on the
  host route.
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

import contextlib
import contextvars
import ctypes
import functools
import os
import sys
import threading
import time
import warnings

import math

import numpy as np
import scipy.linalg
from scipy.linalg import LinAlgWarning

from . import _thread_policy, _times

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


# Wall seconds this thread has spent in the factor stage, always counted
# (two clock reads per LU call): a printout's MATRIX TIMING reads the
# difference around a solve so FACTOR= is the O(N^3) LU and FILL= the
# O(N^2) fill, apart. Per thread, so a resident server's concurrent requests
# keep their own counts.
_factor_clock = threading.local()


def factor_seconds() -> float:
    """This thread's running total of seconds in the LU routines."""
    return getattr(_factor_clock, "total", 0.0)


def set_factor_sink(sink) -> None:
    """Add this thread's later LU seconds to ``sink.factor_s`` as well: the
    fill report's factor time (momwire#2016), which belongs to the solver
    that filled last on this thread. A running total alone cannot say which
    solver a stretch of it was, and is not readable from another thread."""
    _factor_clock.sink = sink


def _factor_stage(fn):
    """``fn`` in MOMWIRE_TIMES's ``factor`` stage (:mod:`momwire._times`; a
    no-op when nothing is timed), and in this thread's `factor_seconds`. Its
    warnings name the caller's line as before: they carry one more level for
    this frame."""

    @functools.wraps(fn)
    def timed(*args, **kwargs):
        # The outermost call counts; a factor wrapper reached from inside
        # another (solve's fallback) is already on the clock.
        if getattr(_factor_clock, "depth", 0):
            with _times.stage("factor"):
                return fn(*args, **kwargs)
        _factor_clock.depth = 1
        t0 = time.perf_counter()
        try:
            with _times.stage("factor"):
                return fn(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - t0
            _factor_clock.total = factor_seconds() + elapsed
            _factor_clock.depth = 0
            sink = getattr(_factor_clock, "sink", None)
            if sink is not None:
                sink.factor_s += elapsed

    return timed


# ---------------------------------------------------------------------------
# scipy.linalg.lu_factor / lu_solve
# ---------------------------------------------------------------------------


@_factor_stage
def lu_factor(a, overwrite_a=False, check_finite=True):
    """``scipy.linalg.lu_factor``, on the host's getrf when one is published."""
    _thread_policy.hold()
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
            stacklevel=3,
        )
    return a1, ipiv - 1


@_factor_stage
def lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True):
    """``scipy.linalg.lu_solve``, on the host's getrs when one is published.

    A `MixedLU` (`lu_factor_kept` under `mixed_lu_scope`) is served by its
    own `MixedLU.solve`, refined against the matrix it keeps."""
    if isinstance(lu_and_piv, MixedLU):
        return lu_and_piv.solve(
            b, trans=trans, overwrite_b=overwrite_b, check_finite=check_finite
        )
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
# the dense routes' LU: one singular-matrix policy (momwire#1981)
# ---------------------------------------------------------------------------

#: What every dense route raises on an EXACTLY singular matrix: the message
#: `solve` and `scipy.linalg.solve` give, so a launcher prints one sentence
#: whichever route the deck took.
SINGULAR = "singular matrix"


def lu_factor_strict(a, overwrite_a=False, check_finite=True):
    """`lu_factor`, under the one singular-matrix policy every dense route
    shares (momwire#1981): an exactly zero pivot raises
    ``LinAlgError(SINGULAR)`` instead of returning factors whose solve is
    inf/NaN. Returns ``(lu, piv)`` as `lu_factor` does.

    The factor is `lu_factor`'s getrf on the same matrix, so the factors are
    its bits; `lu_factor`'s ``LinAlgWarning`` on the zero pivot still comes
    first. The 1-norm and gecon `solve` takes are NOT taken unless
    :func:`condition_check` is on (momwire#1290 dropped them from the
    factor-and-solve routes: memory-bound O(N^2) passes that drove only a
    warning). On, the norm is taken before getrf can overwrite ``a`` and an
    rcond below machine epsilon warns as `solve` does.
    """
    _thread_policy.hold()
    anorm = None
    if condition_check():
        a1 = np.asarray(a)
        if _square(a1) and a1.dtype.kind in "fc":
            lange = scipy.linalg.get_lapack_funcs("lange", (a1,))
            anorm = lange("1", a1)
    lu, piv = lu_factor(a, overwrite_a=overwrite_a, check_finite=check_finite)
    if not np.all(np.diagonal(lu)):
        raise np.linalg.LinAlgError(SINGULAR)
    if anorm is not None:
        gecon = scipy.linalg.get_lapack_funcs("gecon", (lu,))
        rcond, _info = gecon(lu, anorm, norm="1")
        if rcond < np.finfo(lu.dtype).eps:
            warnings.warn(
                f"An ill-conditioned matrix detected: rcond = {rcond:.6g}.",
                LinAlgWarning,
                stacklevel=2,
            )
    return lu, piv


def factor_and_solve(a, b, overwrite_a=False, overwrite_b=False):
    """Solve ``a x = b`` by `lu_factor_strict` then `lu_solve`: the same
    getrf and getrs `scipy.linalg.solve` runs on a general matrix, so its
    answer to the bit, without its condition estimate (see
    `lu_factor_strict`). ``overwrite_a`` factors an F-contiguous ``a`` in its
    own storage (``a`` then holds the factors); ``overwrite_b`` lets an
    F-ordered ``b`` take the solution.

    Under :func:`mixed_lu_scope` (the ``mixed_lu`` feature, momwire#2125) the
    system is first tried by `_mixed_factor_and_solve`, which leaves ``a``
    intact whatever ``overwrite_a`` says; where it falls back or declines,
    the two lines below run exactly as with the flag off, ``overwrite_a``
    and all, so the answer is the flag-off one to the bit."""
    if _MIXED.get():
        x = _mixed_factor_and_solve(a, b)
        if x is not None:
            return x
    lu_piv = lu_factor_strict(a, overwrite_a=overwrite_a)
    return lu_solve(lu_piv, b, overwrite_b=overwrite_b)


@_factor_stage
def _mixed_factor_and_solve(a, b):
    """`factor_and_solve`'s mixed-precision attempt: the solution, or None
    for the ordinary route. A matrix in either order is read in place: an
    F-ordered ``a`` (the dense fills' layout) is cast to complex64 as it
    stands and solved with trans 0; a C-ordered one is cast as ``aᵀ`` and
    solved with trans 1, as `solve_transposed` does. Any other layout, or an
    ``a`` whose dtype is not the solve's, declines. A non-finite system goes
    on uncounted to the ordinary route, which raises. In the factor stage,
    so a printout's FACTOR time is the whole attempt; a fallback's own
    factor is counted after it."""
    a1, b0 = np.asarray(a), np.asarray(b)
    if not (
        _square(a1)
        and a1.shape[0] > 1
        and b0.ndim in (1, 2)
        and b0.size
        and b0.shape[0] == a1.shape[0]
    ):
        return None
    dtype = np.result_type(a1.dtype, b0.dtype)
    if not _mixed_wanted(_host_for(dtype), dtype, b0):
        return None
    if a1.dtype != dtype or not (a1.flags.f_contiguous or a1.flags.c_contiguous):
        _mixed_declined()
        return None
    # The matrix norm is the one `_finite_1norm_c` reads in place, threaded:
    # ||a1||_1 of a C-ordered a1, ||a1||_inf (= ||a1ᵀ||_1) of an F-ordered
    # one, whose transpose is C-ordered. scipy's single-threaded lange("1")
    # on the F-ordered matrix took 0.16 s at n = 4096 (8 threads), most of
    # the factor saving.
    if a1.flags.f_contiguous:
        finite, anorm = _finite_1norm_c(a1.T)
        norm, trans, src = "inf", 0, a1
    else:
        finite, anorm = _finite_1norm_c(a1)
        norm, trans, src = "1", 1, a1.T

    def at32():
        return np.array(src, dtype=np.complex64, order="F")

    if not (finite and np.isfinite(b0).all()):
        return None
    _thread_policy.hold()
    n = a1.shape[0]
    b2 = _fortran_copy(b0.reshape(n, 1) if b0.ndim == 1 else b0, dtype)
    x = _mixed_refine(a1, at32, trans, b2, anorm, norm)
    return None if x is None else x.reshape(b0.shape)


# ---------------------------------------------------------------------------
# scipy.linalg.solve (default arguments: general matrix, check_finite)
# ---------------------------------------------------------------------------


def _scipy_takes_gesv(a, dtype):
    """True only when `scipy.linalg.solve(a, b)` provably runs the general
    getrf + getrs path. SciPy >= 1.18 first detects structure (diagonal,
    tridiagonal, banded, triangular, symmetric, Hermitian) and solves those
    with other routines -- an exactly symmetric complex system takes zsytrf,
    which moved bits on the bspline kcl_sym decks when this fallback once
    ran getrf unconditionally. Each case is ruled out in O(n), never by an
    O(n^2) scan: a nonzero a[n-1, 0] makes the lower bandwidth n - 1 and a
    nonzero a[0, n-1] the upper, which excludes every banded, triangular
    and diagonal form; one first-row/first-column pair with
    a[0, j] != a[j, 0] proves the matrix not symmetric, and one with
    a[0, j] != conj(a[j, 0]) not Hermitian. Anything not proven general
    goes to SciPy unchanged, so a False here costs nothing but the speed."""
    if dtype not in (np.float32, np.float64, np.complex64, np.complex128):
        return False
    n = a.shape[0]
    if a[n - 1, 0] == 0 or a[0, n - 1] == 0:
        return False
    row, col = a[0, 1:], a[1:, 0]
    if not np.any(row != col):
        return False
    return bool(np.any(row != np.conj(col)))


#: `solve` prepares A with the accelerator's `fortran_copy_1norm` when it
#: carries it (momwire#1916); False keeps numpy's isfinite, numpy's copy and
#: scipy's lange, the reference that kernel is gated against.
_SOLVE_PREP_ACCEL = True

#: Below this order the three numpy / scipy passes cost less than a call.
_SOLVE_PREP_MIN_N = 64


def _solve_prep(a1, dtype):
    """`(Fortran copy, all finite, lange("1") norm)` of `a1` in one threaded
    pass (`_accel_solve_prep.cpp`), or None where that kernel does not apply:
    switched off, no accelerator, a small or non-C-contiguous matrix, or a
    dtype it does not take (or that the solve casts to)."""
    if not _SOLVE_PREP_ACCEL or a1.shape[0] < _SOLVE_PREP_MIN_N:
        return None
    if a1.dtype != dtype or dtype not in (np.complex128, np.float64):
        return None
    if not a1.flags.c_contiguous:
        return None
    try:
        from ._accel import acc
    except ImportError:  # pragma: no cover - the accelerator module is optional
        return None
    if acc is None or not getattr(acc, "solve_prep_1916", False):
        return None
    lu, finite, anorm = acc.fortran_copy_1norm(a1)
    return lu, bool(finite), float(anorm)


#: The environment variable that turns `solve`'s condition estimate on.
CONDITION_CHECK_ENV = "MOMWIRE_CONDITION_CHECK"


def condition_check() -> bool:
    """Whether `solve` estimates the reciprocal condition (gecon) on every
    matrix, not only on the ones the pivot screen flags:
    ``MOMWIRE_CONDITION_CHECK`` set to anything but empty or ``0``. Read per
    call, so a process can turn it on mid-run."""
    return os.environ.get(CONDITION_CHECK_ENV, "") not in ("", "0")


#: The pivot screen's bar: `solve` runs gecon when the smallest pivot of the
#: factors is below this fraction of the largest. The warning itself still
#: takes gecon's rcond below machine epsilon, so the bar only decides when
#: the estimate is paid for. min|u_ii| / max|u_ii| is not a bound on rcond
#: (row growth in L can hide ill-conditioning from U's diagonal), so the bar
#: sits six orders above epsilon: the ELT whip, well conditioned, reads
#: 7.4e-4 to 7.7e-4 across its 13 frequencies (n = 4392).
PIVOT_SCREEN = 1e-10


def _pivots_suspect(lu) -> bool:
    """The free screen: True when U's diagonal spans more than
    1 / `PIVOT_SCREEN` (or holds a zero or a NaN). n reads of the factors."""
    d = np.abs(np.diagonal(lu))
    hi = d.max()
    return not (hi > 0 and d.min() >= PIVOT_SCREEN * hi)


# ---------------------------------------------------------------------------
# mixed_lu (momwire#2125): complex64 getrf, complex128 iterative refinement
# ---------------------------------------------------------------------------
#
# The design is LAPACK's own zcgesv (factor in single, refine with a
# double-precision residual, fall back to the double factor on failure),
# with four changes: a tighter stop (MIXED_TOLERANCE), the refinement must
# CONTRACT at a fixed rate on every sweep that has not yet converged, it
# takes at least two corrections, and it serves the transposed factor
# `solve_transposed` needs (zcgesv has no trans). The original complex128 matrix is never written: the residuals
# need it, and the fallback factors it exactly as the flag-off route does,
# so a fallback returns the flag-off bits.

# Whether `solve` / `solve_transposed` try the mixed route: set by a solver
# around its own solve (`mixed_lu_scope`), so the flag belongs to the
# request that built the solver and a concurrent one can say no. A
# ContextVar, so each thread (and each asyncio task) has its own.
_MIXED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "momwire_mixed_lu", default=False
)


@contextlib.contextmanager
def mixed_lu_scope(on: bool):
    """Run the body with the ``mixed_lu`` route on (`on` True) or off for
    `solve` and `solve_transposed` in this thread / context."""
    token = _MIXED.set(bool(on))
    try:
        yield
    finally:
        _MIXED.reset(token)


#: u, LAPACK's dlamch('E'): the unit roundoff of complex128 (2**-53).
_U64 = float(np.finfo(np.float64).eps) / 2
#: The unit roundoff of complex64 (2**-24).
_U32 = float(np.finfo(np.float32).eps) / 2

#: The stop, per column, on the normwise backward error in the 1-norm,
#: ``eta = ||b - A x||_1 / (||A||_1 ||x||_1)``: a column is converged when
#: eta <= MIXED_TOLERANCE (u64), or when eta <= sqrt(n) u64 (zcgesv's own
#: stop, `_mixed_accept`) AND the last sweep cut the residual by less than
#: MIXED_CONTRACTION: it has reached the floor of a complex128 residual and
#: no further sweep can improve it.
#:
#: Why not zcgesv's bar alone: on the W8IO whip (n = 4392, 13 frequencies)
#: two corrections land at eta 4e-15 to 7e-15, under sqrt(n) u64 = 7.3e-15,
#: and leave the solution 4e-12 to 6e-12 (max-norm, relative) from the
#: ordinary route's, where two ordinary complex128 factorizations (pivots on
#: A and on Aᵀ) differ by 5e-13 to 1e-12. That iterate is still contracting
#: (by 1e-4), so it is not at the floor; one more correction reaches eta
#: 1.4e-17 to 1.9e-17 and 5e-13 to 1e-12 from the ordinary route, inside the
#: scatter. The ordinary route's own eta there is 6.6e-16 to 1.2e-15.
#: Why not u64 alone: on small decks the floor sits a little above it
#: (1.6e-16 to 1.8e-16 on two n = 83, 86 corpus decks), where sweeps stop
#: contracting with the answer already as good as it gets.
MIXED_TOLERANCE = _U64


#: The forward half of the stop (momwire#2125, from the sg census): a
#: normwise backward error at u64 is not enough on a badly scaled matrix.
#: On a 45-unknown sg deck whose rows and columns span 1.5e5 (kappa_1 3e7)
#: two corrections reach eta 5e-17 and leave the answer 6e-11 from the
#: ordinary route's, where a second complex128 factor sits 1e-12 from it;
#: its componentwise backward error is still 9e-11. So a column must also
#: have stopped changing: its last correction d_k, as ||d_k||_inf /
#: ||x||_inf, times that correction's contraction d_k / d_(k-1) (the error
#: it leaves), at or under MIXED_FORWARD_TOLERANCE, 2**-44 (5.7e-14), a
#: seventeenth of the declared drift and under the complex128 scatter; or,
#: once the corrections stop contracting (MIXED_CONTRACTION), the last one
#: at or under MIXED_FORWARD_FLOOR.
#: (Not sqrt(n) u64: on a bs1 deck, n = 404, the corrections contract by
#: 2e-3 while the residual is already at its floor, and that bar takes all
#: five sweeps for an answer 4e-13 from the ordinary one.)
MIXED_FORWARD_TOLERANCE = 2.0**-44

#: MIXED_FORWARD_FLOOR is set from the declared drift (2e-11 when set; 1e-10 since), not from a
#: small deck's scatter. It was 2**-40 (9.1e-13), the two-factor scatter of
#: decks of a few hundred unknowns; but that scatter grows with the order,
#: and a correction that has stopped contracting overstates the error it
#: leaves. Measured on a 10-element Yagi in free space (laptop, max-norm of
#: the currents relative to the largest; momwire#2125 thread):
#:
#:   engine    n      Z vs Z^T     mixed vs both     last correction
#:   bs2       4307   2.1e-13      4.3-5.4e-13       5.4e-13
#:   razor2p   5922   1.2e-12      0.4-1.0e-12       1.7e-12
#:   bs2       5931   4.2e-13      4.0-4.4e-13       6.2e-12
#:   razor2p   7528   7.6e-13      2.8e-12           2.9e-11
#:   bs2       9329   1.4e-12      1.9-2.4e-12       3.5e-11
#:   razor2p   9318   6.2e-12      2.6-5.0e-12       5.3e-11
#:
#: ("Z vs Z^T": two complex128 factors, pivots on Z and on Z^T.) The last
#: correction is up to 10x the error left, and at 2**-40 every rung from 5.9k
#: fell back after paying for the complex64 factor and its sweeps. At 2**-34
#: (5.8e-11) every rung refines, each answer within 4x of the two-factor
#: scatter and within a quarter of the drift. The badly scaled sg deck above
#: is decided by MIXED_FORWARD_TOLERANCE while it still contracts, so the
#: floor does not reach it (test_a_badly_scaled_matrix_refines_past_its_backward_error).
MIXED_FORWARD_FLOOR = 2.0**-34


def _mixed_accept(n: int) -> float:
    """zcgesv's stop, sqrt(n) u64: the backward error a column may stop at
    once it has stopped contracting (MIXED_TOLERANCE)."""
    return float(np.sqrt(n)) * _U64


#: The most refinement corrections a solve may take before it falls back,
#: at the smallest orders; `_mixed_sweeps_cap` grows it with n.
MIXED_SWEEPS_MAX = 5
#: The fewest corrections a solve takes before it may stop: one correction
#: can land under the tolerance by luck of a single step; a second one that
#: holds it there is the evidence the iteration has settled.
MIXED_SWEEPS_MIN = 2

#: Every sweep that has not yet converged must cut each column's residual
#: 1-norm by at least this factor, or the solve falls back. Derivation: the
#: complex64 solution's backward error starts at the complex64 level (about
#: u32 times the growth: 3e-8 to 5e-7 on the whip and the corpus) and must
#: end at MIXED_TOLERANCE, u64, a fall of about u64 / u32 = 2**-29; at a
#: steady rate theta that
#: takes log(2**-29) / log(theta) sweeps, so a cap of MIXED_SWEEPS_MAX = 5
#: is only reachable when theta <= 2**(-29/5) = 2**-5.8. theta = 2**-6 is
#: that bound rounded down: a solve contracting more slowly could not finish
#: inside the cap, and is sent to the ordinary route at its first slow sweep
#: instead of after five. The rate is about kappa(A) u32 (Higham, ASNA 2nd
#: ed., ch. 12), so this also declines an ill-conditioned matrix on its own
#: evidence, with no condition estimate: on random complex test matrices
#: (n = 120) the rate is 1e-3 at kappa_2 1e5, 1e-2 at 1e6 (refined), 2e-2
#: at 3e6 and 0.1 at 1e7 (fallback). The W8IO whip (kappa_1 1.2e5 to
#: 1.5e5) contracts by 1e-4 to 3e-4 a sweep.
MIXED_CONTRACTION = 2.0**-6

#: The sweep cap grows with the order. A correction costs O(n^2 nrhs) (a
#: complex128 residual and a complex64 getrs) against the O(n^3) the complex64
#: factor saves, so the corrections the saving pays for grow as n / nrhs.
#: Measured on Skylake (i7-6700K, 8 threads, one right-hand side, the
#: 10-element Yagi's Z, cgetrf and the refined solves' sweeps against zgetrf):
#: the saving covers about 7 corrections at n = 2153, 13 at 4307 and 33 at
#: 7539. The cap is about half of that, n / MIXED_SWEEPS_N, so a solve that
#: uses every correction still saves; never under MIXED_SWEEPS_MAX.
#: Why it must grow: the rate is about kappa(Z) u32, and Z's condition grows
#: with the mesh (the Yagi contracts by 4e-4 at n = 1079, 8e-3 at 4307, 2e-2
#: at 9329), so a fixed cap of five sends every large solve back to the
#: ordinary route, after paying for the complex64 factor (1.5x the ordinary
#: factor on that ladder).
MIXED_SWEEPS_N = 640
#: The loosest stall bar the growing cap may reach: kappa u32 <= 1/4.
MIXED_STALL_LOOSEST = 2.0**-2


def _mixed_sweeps_cap(n: int, nrhs: int) -> int:
    """The most corrections a solve of order `n` with `nrhs` columns takes."""
    return max(MIXED_SWEEPS_MAX, n // (MIXED_SWEEPS_N * max(nrhs, 1)))


def _mixed_stall(cap: int) -> float:
    """The slowest per-sweep contraction that can still finish inside `cap`
    corrections: the fall of 2**-29 (u64 / u32) at rate theta takes
    29 / -log2(theta) sweeps, so theta = 2**-ceil(29 / cap). At the base cap
    of five that is MIXED_CONTRACTION, 2**-6, the rule it replaces; never
    looser than MIXED_STALL_LOOSEST."""
    return min(2.0 ** -math.ceil(29 / cap), MIXED_STALL_LOOSEST)


#: Where the route declines by size: below MIXED_MIN_N unknowns, or above
#: `_mixed_nrhs_cap(n)` right-hand sides. The complex64 factor saves O(n^3)
#: and each sweep costs O(n^2 nrhs); at small n neither threads well.
#: Measured on Rocketlake (`factor_and_solve`, F-ordered, overwrite_a,
#: random complex matrices refining in 2 sweeps, medians of 5, ordinary over
#: mixed), 8 threads: n = 512 0.95x, 768 1.04x, 1024 1.14x, 2048 1.57x,
#: 4096 1.53x at one right-hand side; at three, 1024 0.93x, 1536 1.00x,
#: 2048 1.32x, 4096 1.21x. One thread: 512 0.88x, 768 1.05x, 1024 1.15x,
#: 4096 1.63x. And the whip's own Z (n = 4392, 8 threads, 3 sweeps):
#: 1.50x at 1, 1.27x at 3, 1.16x at 8, 1.04x at 64, 0.94x at 96. So
#: n >= 1024, and the cap grows as n^2 to 64 at n = 4392 (3 at 1024, 13 at
#: 2048) and as n past it, the asymptote the O(n^3) / O(n^2 nrhs) ratio
#: gives (182 at the bs2 whip's 12,534).
MIXED_MIN_N = 1024
MIXED_MAX_NRHS = 64
MIXED_NRHS_N = 4392


def _mixed_nrhs_cap(n: int) -> int:
    """The most right-hand sides the route takes at order `n`."""
    r = n / MIXED_NRHS_N
    return int(MIXED_MAX_NRHS * min(r * r, r))


# The cast guard, on the matrix 1-norm the solve already reads. No entry
# exceeds ||A||_1, so ||A||_1 <= _CAST_HI keeps every entry 2**32 below
# complex64's overflow (headroom for pivot growth and the back-substitution's
# sums). An entry that matters, one within u32 of the largest (at least
# ||A||_1 / n), stays a normal complex64 number when ||A||_1 / n >= _CAST_LO.
_CAST_HI = float(np.finfo(np.float32).max) * 2.0**-32
_CAST_LO = float(np.finfo(np.float32).tiny) * 2.0**24

#: mixed-route outcomes since import (tests and gates read them to prove the
#: route ran): "refined" (answered by the route), "fallback" (tried, then
#: answered by the ordinary route), "declined" (asked for but not tried:
#: a host LAPACK, a dtype other than complex128, too many right-hand sides),
#: and "sweeps", the refinement corrections taken by refined solves.
MIXED_SOLVES = {"refined": 0, "fallback": 0, "declined": 0, "sweeps": 0}
#: Each fallback by its reason: "range" (the cast guard), "singular" (an
#: exactly singular complex64 factor), "nonfinite" (an inf or NaN in a
#: complex64 solve or a residual), "stall" (a sweep short of
#: `_mixed_stall`), "cap" (`_mixed_sweeps_cap` corrections, no convergence).
MIXED_FALLBACKS = {"range": 0, "singular": 0, "nonfinite": 0, "stall": 0, "cap": 0}
#: The last mixed attempt in this process: its outcome, corrections, and per
#: check the largest normwise backward error over the columns and the
#: largest residual ratio (sweep over the sweep before).
MIXED_LAST: dict = {}


def _mixed_declined() -> None:
    MIXED_SOLVES["declined"] += 1


def _mixed_fallback(reason: str, record: dict) -> None:
    MIXED_SOLVES["fallback"] += 1
    MIXED_FALLBACKS[reason] += 1
    record["outcome"] = reason
    MIXED_LAST.clear()
    MIXED_LAST.update(record)


def _mixed_correction(getrs, lu32, ipiv, r, trans):
    """Solve ``op(A) d = r`` with the complex64 factors, `r` (n, k) complex128.

    Each column is scaled by a power of two to a largest magnitude in
    [0.5, 1) before the cast and scaled back after, both exact, so a
    residual near complex128's roundoff never underflows complex64."""
    m = np.abs(r).max(axis=0)
    e = np.where(m > 0, np.frexp(m)[1], 0).astype(np.float64)
    r32 = np.asfortranarray(r * np.exp2(-e), dtype=np.complex64)
    d32, info = getrs(lu32, ipiv, r32, trans=trans, overwrite_b=True)
    if info != 0:
        raise ValueError(f"illegal value in argument {-info} of getrs")
    return d32.astype(np.complex128) * np.exp2(e)


def _mixed_refine(a1, at32, trans, b2, anorm, norm="1"):
    """`a1 x = b2` by mixed-precision iterative refinement, or None to fall
    back to the ordinary complex128 route.

    `a1` is the complex128 matrix (read, never written), `at32` a callable
    returning a fresh F-contiguous complex64 matrix that LAPACK's getrf
    factors, and `trans` the getrs flag that turns its factors into a solve
    with `a1` (`solve_transposed`: ``a1ᵀ`` and 1; `solve`: ``a1`` and 0;
    `factor_and_solve` either, by ``a1``'s order).
    `b2` is the (n, k) complex128 right-hand side (not written), `anorm`
    ||a1|| in the norm `norm` names ("1", or "inf" for `factor_and_solve`'s
    F-ordered matrix), and the backward error is taken in that norm, the
    vector norms to match. Returns the (n, k) complex128 solution,
    F-ordered.

    Stop: every column's normwise backward error in the 1-norm,
    ``||r||_1 / (||A||_1 ||x||_1)``, at or under MIXED_TOLERANCE (u64), or
    under `_mixed_accept` once it no longer contracts, after at least
    MIXED_SWEEPS_MIN corrections. A column above `_mixed_accept` that has
    not cut its residual by `_mixed_stall` since the last check is a stall:
    the solve falls back. At most `_mixed_sweeps_cap` corrections.
    Memory: the complex64 factors (half of a1) and (n, k) vectors; the
    factors are dropped before returning, either way."""
    record = _mixed_record(a1.shape[0], b2.shape[1])
    if not _mixed_in_range(a1.shape[0], anorm):
        _mixed_fallback("range", record)
        return None
    factors = _mixed_factor32(at32)
    if factors is None:
        _mixed_fallback("singular", record)
        return None
    return _mixed_sweeps(a1, *factors, trans, b2, anorm, norm, record)


def _mixed_record(n: int, nrhs: int) -> dict:
    """A fresh MIXED_LAST record for one solve of order `n`, `nrhs` columns."""
    return {
        "n": n,
        "nrhs": nrhs,
        "sweeps": 0,
        "eta": [],
        "ratio": [],
        "delta": [],
        "cap": _mixed_sweeps_cap(n, nrhs),
    }


def _mixed_in_range(n: int, anorm: float) -> bool:
    """The cast guard (`_CAST_HI`, `_CAST_LO`) on ||A|| of order `n`."""
    return n * _CAST_LO <= anorm <= _CAST_HI


def _mixed_factor32(at32):
    """complex64 getrf of the fresh F-ordered matrix `at32()` returns:
    ``(lu32, ipiv, getrs)``, or None when a pivot is exactly zero."""
    lu32 = at32()
    getrf, getrs = scipy.linalg.get_lapack_funcs(("getrf", "getrs"), (lu32,))
    lu32, ipiv, info = getrf(lu32, overwrite_a=True)
    if info != 0:
        return None
    return lu32, ipiv, getrs


def _mixed_sweeps(a1, lu32, ipiv, getrs, trans, b2, anorm, norm, record):
    """`_mixed_refine`'s iteration on complex64 factors already taken: the
    (n, k) F-ordered solution, or None (the fallback counted and recorded).
    `lu32` is only read, so a kept factor (`MixedLU`) serves every solve
    from the one cgetrf."""
    n = a1.shape[0]
    cap = record["cap"]
    stall = _mixed_stall(cap)
    tau, accept = MIXED_TOLERANCE, _mixed_accept(n)
    x = _mixed_correction(getrs, lu32, ipiv, b2, trans)
    prev = None
    # The last correction's size, ||d||_inf / ||x||_inf per column, and the
    # one before it (the forward test; None until two corrections exist).
    delta = dprev = None
    k = 0
    while True:
        if not np.isfinite(x).all():
            reason = "nonfinite"
            break
        r = b2 - a1 @ x
        if norm == "1":
            rn, xn = np.abs(r).sum(axis=0), np.abs(x).sum(axis=0)
        else:
            rn, xn = np.abs(r).max(axis=0), np.abs(x).max(axis=0)
        if not (np.isfinite(rn).all() and np.isfinite(xn).all()):
            reason = "nonfinite"
            break
        with np.errstate(divide="ignore", invalid="ignore"):
            eta = np.where(rn > 0, rn / (anorm * xn), 0.0)
        record["eta"].append(float(eta.max()))
        slow = np.zeros(eta.shape, dtype=bool)
        stalled = slow
        if prev is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(prev > 0, rn / prev, 0.0)
            record["ratio"].append(float(ratio.max()))
            slow = ratio > MIXED_CONTRACTION
            stalled = ratio > stall
        # Backward: at u64, or at the floor (no longer contracting) under
        # zcgesv's bar. Not contracting above that bar is a stall.
        back = (eta <= tau) | (slow & (eta <= accept))
        # Forward (`MIXED_FORWARD_TOLERANCE`): the error the last correction
        # leaves, predicted as its size times its contraction; or the
        # correction stopped contracting at a size under the floor. One that
        # stops contracting above the floor stalls.
        fwd = np.zeros(eta.shape, dtype=bool)
        fwd_slow = np.zeros(eta.shape, dtype=bool)
        fwd_stalled = fwd_slow
        if dprev is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                rho = np.where(dprev > 0, delta / dprev, 0.0)
            fwd_slow = rho > MIXED_CONTRACTION
            fwd_stalled = rho > stall
            fwd = (delta * np.minimum(rho, 1.0) <= MIXED_FORWARD_TOLERANCE) | (
                fwd_slow & (delta <= MIXED_FORWARD_FLOOR)
            )
            record["delta"].append(float(delta.max()))
        done = back & fwd
        # A slow sweep (MIXED_CONTRACTION) above the bars is a stall only
        # when it is also too slow to finish inside the cap (`_mixed_stall`);
        # MIXED_CONTRACTION alone still marks the floor in `back` and `fwd`.
        if np.any(stalled & ~back) or np.any(fwd_stalled & ~fwd):
            reason = "stall"
            break
        if k >= MIXED_SWEEPS_MIN and done.all():
            record["sweeps"] = k
            record["outcome"] = "refined"
            MIXED_SOLVES["refined"] += 1
            MIXED_SOLVES["sweeps"] += k
            MIXED_LAST.clear()
            MIXED_LAST.update(record)
            return np.asfortranarray(x)
        if k == cap:
            reason = "cap"
            break
        d = _mixed_correction(getrs, lu32, ipiv, r, trans)
        x = x + d
        with np.errstate(divide="ignore", invalid="ignore"):
            xm = np.abs(x).max(axis=0)
            dprev, delta = delta, np.where(xm > 0, np.abs(d).max(axis=0) / xm, 0.0)
        if dprev is None:
            # x0 was the first correction: the whole of x.
            dprev = np.ones_like(delta)
        prev = rn
        k += 1
        record["sweeps"] = k
    _mixed_fallback(reason, record)
    return None


def _mixed_wanted(fns, dtype, b0) -> bool:
    """Whether a `solve` / `solve_transposed` / `factor_and_solve` call tries
    the mixed route: asked for (`mixed_lu_scope`), and not declined (counted)
    for a host LAPACK (it publishes no complex64 getrf), a dtype other than
    complex128, fewer than MIXED_MIN_N unknowns, or more right-hand sides
    than `_mixed_nrhs_cap` allows at this order."""
    if not _MIXED.get():
        return False
    return _mixed_takes(fns, dtype, b0.shape[0], 1 if b0.ndim == 1 else b0.shape[1])


def _mixed_takes(fns, dtype, n: int, nrhs: int) -> bool:
    """`_mixed_wanted`'s size, dtype and host rule for an order-`n` solve of
    `nrhs` columns, the scope aside; a refusal counts as declined."""
    if (
        fns is not None
        or dtype != _C128
        or n < MIXED_MIN_N
        or nrhs > _mixed_nrhs_cap(n)
    ):
        _mixed_declined()
        return False
    return True


@_factor_stage
def solve(a, b):
    """``scipy.linalg.solve(a, b)``, on the host's getrf + getrs when one is
    published. The condition estimate runs under :func:`condition_check`
    or when :func:`_pivots_suspect` flags the factors.

    Under :func:`mixed_lu_scope` a C-ordered complex128 system on the
    no-host route is first tried by :func:`_mixed_refine`; where that
    falls back, everything below runs as with the flag off."""
    a1 = np.asarray(a)
    b0 = np.asarray(b)
    dtype = np.result_type(a1.dtype, b0.dtype)
    fns = _host_for(dtype)
    if (
        not _square(a1)
        or a1.shape[0] == 1
        or b0.ndim not in (1, 2)
        or b0.size == 0
        or b0.shape[0] != a1.shape[0]
        or (fns is None and not _scipy_takes_gesv(a1, dtype))
    ):
        return scipy.linalg.solve(a, b)
    if _mixed_wanted(fns, dtype, b0):
        # Before the Fortran copy below, so the route holds a1 and its
        # complex64 copy, never a1 twice. A C-ordered complex128 a1 only:
        # its 1-norm is read in place (`_finite_1norm_c`); any other layout
        # declines. A non-finite system goes on to the ordinary error.
        x = None
        if a1.dtype == dtype and a1.flags.c_contiguous:
            a_finite, anorm = _finite_1norm_c(a1)
            if a_finite and np.isfinite(b0).all():
                n = a1.shape[0]
                x = _mixed_refine(
                    a1,
                    lambda: np.array(a1, dtype=np.complex64, order="F"),
                    0,
                    _fortran_copy(b0.reshape(n, 1) if b0.ndim == 1 else b0, dtype),
                    anorm,
                )
        else:
            _mixed_declined()
        if x is not None:
            return x.reshape(b0.shape)
    always = condition_check()
    prep = _solve_prep(a1, dtype)
    if prep is not None:
        # The copy, the finiteness check and lange's 1-norm in one threaded
        # pass (momwire#1916): the same matrix, error and norm.
        lu, a_finite, anorm = prep
        if not (a_finite and np.isfinite(b0).all()):
            raise ValueError("array must not contain infs or NaNs")
        lange, gecon = scipy.linalg.get_lapack_funcs(("lange", "gecon"), (lu,))
    else:
        if not (np.isfinite(a1).all() and np.isfinite(b0).all()):
            raise ValueError("array must not contain infs or NaNs")
        lu = _fortran_copy(a1, dtype)
        lange, gecon = scipy.linalg.get_lapack_funcs(("lange", "gecon"), (lu,))
        # For the estimate only, as scipy's solve takes it. Taken whether or
        # not the estimate runs: the screen decides after getrf has
        # overwritten A. The prepared pass above has it for free.
        anorm = lange("1", lu)
    n = a1.shape[0]
    x = _fortran_copy(b0.reshape(n, 1) if b0.ndim == 1 else b0, dtype)
    if fns is not None:
        ipiv, info = _getrf(fns, lu)
        if info != 0:
            if info > 0:
                raise np.linalg.LinAlgError("singular matrix")
            raise ValueError(f"illegal value in argument {-info} of the host's getrf")
        info = _getrs(fns, lu, ipiv, x)
        if info != 0:
            raise ValueError(f"illegal value in argument {-info} of the host's getrs")
        if not (always or _pivots_suspect(lu)):
            return x.reshape(b0.shape)
    else:
        # No host LAPACK, and a matrix SciPy is proven to solve as general
        # (`_scipy_takes_gesv`): SciPy's own getrf + getrs, the pair its
        # gesv path runs, without its batched wrapper -- 0.56 s against
        # 0.96 s at n = 4392, complex, 8 threads (Rocketlake), about 5 s of
        # the ELT whip's 13-frequency sinusoidal run. Bit-identical to
        # scipy.linalg.solve on such a matrix.
        getrf, getrs = scipy.linalg.get_lapack_funcs(("getrf", "getrs"), (lu,))
        lu, ipiv, info = getrf(lu, overwrite_a=True)
        if info != 0:
            if info > 0:
                raise np.linalg.LinAlgError("singular matrix")
            raise ValueError(f"illegal value in argument {-info} of getrf")
        estimate = always or _pivots_suspect(lu)
        beside = estimate and n >= _GECON_BESIDE_MIN_N
        cond = _gecon_beside(gecon, lu, anorm) if beside else None
        try:
            x, info = getrs(lu, ipiv, x, overwrite_b=True)
        finally:
            rcond = None if cond is None else cond()
        if info != 0:
            raise ValueError(f"illegal value in argument {-info} of getrs")
        if not estimate:
            return x.reshape(b0.shape)
        if rcond is not None:
            return _warn_ill(rcond, dtype, x.reshape(b0.shape))
    rcond, _info = gecon(lu, anorm, norm="1")
    return _warn_ill(rcond, dtype, x.reshape(b0.shape))


# ---------------------------------------------------------------------------
# solve_transposed: a row-major A factored as Aᵀ in its own storage
# ---------------------------------------------------------------------------

#: `solve_transposed` calls since import, by how A reached getrf: "in_place"
#: (A's own storage) or "copied" (A was not a writeable C-contiguous array of
#: the solve's dtype). Tests read it to prove the route ran.
TRANSPOSED_SOLVES = {"in_place": 0, "copied": 0}


def _finite_1norm_c(a1):
    """``(all finite, ‖a1‖₁)`` of the C-contiguous square `a1`, read in place.

    The norm is lange("1") of `a1`'s Fortran copy as `solve` takes it: per
    column, ``|a(i, j)|`` summed from zero for i ascending, then the
    maximum. The accelerator's `finite_1norm_c` is that pass with no copy;
    without it, lange("I") of the F-ordered view ``a1.T`` sums the same
    column chains (zlange's infinity-norm loop: row i of ``a1.T`` is column
    i of ``a1``, accumulated in ascending column order of ``a1.T``)."""
    dtype = a1.dtype
    if (
        _SOLVE_PREP_ACCEL
        and a1.shape[0] >= _SOLVE_PREP_MIN_N
        and dtype in (np.complex128, np.float64)
    ):
        try:
            from ._accel import acc
        except ImportError:  # pragma: no cover - the accelerator module is optional
            acc = None
        if acc is not None and getattr(acc, "solve_prep_2071", False):
            finite, anorm = acc.finite_1norm_c(a1)
            return bool(finite), float(anorm)
    if not np.isfinite(a1).all():
        return False, float("nan")
    lange = scipy.linalg.get_lapack_funcs("lange", (a1.T,))
    return True, float(lange("I", a1.T))


@_factor_stage
def solve_transposed(a, b):
    """Solve ``a x = b`` by factoring ``aᵀ`` IN ``a``'s OWN STORAGE.

    A C-ordered (row-major) ``a`` read column-major is ``aᵀ``, so getrf on
    ``a.T`` factors it with no copy and getrs with trans='T' solves
    ``(aᵀ)ᵀ x = b``. One dense matrix lives through the factor, where `solve`
    holds ``a`` and its Fortran copy. ``a`` is DESTROYED (it holds the
    factors of ``aᵀ``) when it is a writeable C-contiguous array of the
    solve's dtype; any other ``a`` is copied once into C order first, which
    changes no bit of the answer.

    `solve`'s contract otherwise, on both routes (host, and scipy's own
    getrf / getrs):

    - a non-finite ``a`` or ``b`` raises ``ValueError``;
    - an exactly singular ``a`` raises ``LinAlgError("singular matrix")``;
    - the condition estimate runs under :func:`condition_check` or when
      :func:`_pivots_suspect` flags the factors, and an rcond below machine
      epsilon warns. It estimates `solve`'s quantity, rcond₁(a): on factors
      of ``aᵀ`` that is gecon with norm='I' and anorm = ‖aᵀ‖_∞ = ‖a‖₁, the
      maximum column sum of ``a`` read in place (:func:`_finite_1norm_c`,
      the bits `solve` hands its gecon).

    NOT `solve`'s answer to the bit: partial pivoting on ``aᵀ`` picks other
    pivots than on ``a``, so the solution differs in the last digits (a
    backward-stable LU either way). Nor does it run SciPy's structure
    detection (symmetric, Hermitian, banded, triangular): every square ``a``
    of order >= 2 is factored as general, so a caller's answer never depends
    on whether its matrix happens to look structured, and an executable that
    replays the call is one getrf + getrs. A 1 x 1, a rectangular or empty
    system, or a dtype LAPACK's getrf does not take is `scipy.linalg.solve`'s.

    Under :func:`mixed_lu_scope` (the ``mixed_lu`` feature, momwire#2125) a
    complex128 system on the no-host route that the size rule takes
    (`_mixed_wanted`) is first tried by :func:`_mixed_refine`, which factors
    a complex64 copy of ``aᵀ`` and leaves ``a`` INTACT; it answers only when
    the refinement reaches complex128's backward error, so no condition
    warning could fire there and the estimate is not run. Where it falls
    back, the route below runs on the untouched ``a`` and returns the
    flag-off answer to the bit. The factor clock counts the whole call,
    refinement and any fallback included.
    """
    a1 = np.asarray(a)
    b0 = np.asarray(b)
    dtype = np.result_type(a1.dtype, b0.dtype)
    if (
        not _square(a1)
        or a1.shape[0] == 1
        or b0.ndim not in (1, 2)
        or b0.size == 0
        or b0.shape[0] != a1.shape[0]
        or dtype not in (np.float32, np.float64, np.complex64, np.complex128)
    ):
        return scipy.linalg.solve(a, b)
    if a1.dtype == dtype and a1.flags.c_contiguous and a1.flags.writeable:
        TRANSPOSED_SOLVES["in_place"] += 1
    else:
        a1 = np.array(a1, dtype=dtype, order="C", copy=True)
        TRANSPOSED_SOLVES["copied"] += 1
    # Taken before getrf overwrites a, whether or not the estimate runs: the
    # screen decides after the factor.
    a_finite, anorm = _finite_1norm_c(a1)
    if not (a_finite and np.isfinite(b0).all()):
        raise ValueError("array must not contain infs or NaNs")
    at = a1.T  # F-contiguous: aᵀ in LAPACK's order, a's own bytes
    always = condition_check()
    n = a1.shape[0]
    x = _fortran_copy(b0.reshape(n, 1) if b0.ndim == 1 else b0, dtype)
    fns = _host_for(dtype)
    if _mixed_wanted(fns, dtype, b0):
        # aᵀ cast to complex64 (a straight cast: `at` is already in LAPACK's
        # order) and factored there; a and x are left as they are, so a
        # fallback runs the lines below on exactly what the flag-off call
        # sees.
        xm = _mixed_refine(
            a1, lambda: np.array(at, dtype=np.complex64, order="F"), 1, x, anorm
        )
        if xm is not None:
            return xm.reshape(b0.shape)
    gecon = scipy.linalg.get_lapack_funcs("gecon", (at,))
    if fns is not None:
        ipiv, info = _getrf(fns, at)
        if info != 0:
            if info > 0:
                raise np.linalg.LinAlgError(SINGULAR)
            raise ValueError(f"illegal value in argument {-info} of the host's getrf")
        info = _getrs(fns, at, ipiv, x, trans=1)
        if info != 0:
            raise ValueError(f"illegal value in argument {-info} of the host's getrs")
        lu = at
        if not (always or _pivots_suspect(lu)):
            return x.reshape(b0.shape)
    else:
        getrf, getrs = scipy.linalg.get_lapack_funcs(("getrf", "getrs"), (at,))
        # overwrite_a on an F-contiguous array of getrf's own dtype is f2py's
        # in-place case: lu IS a's storage.
        lu, ipiv, info = getrf(at, overwrite_a=True)
        if info != 0:
            if info > 0:
                raise np.linalg.LinAlgError(SINGULAR)
            raise ValueError(f"illegal value in argument {-info} of getrf")
        estimate = always or _pivots_suspect(lu)
        beside = estimate and n >= _GECON_BESIDE_MIN_N
        cond = _gecon_beside(gecon, lu, anorm, norm="I") if beside else None
        try:
            x, info = getrs(lu, ipiv, x, trans=1, overwrite_b=True)
        finally:
            rcond = None if cond is None else cond()
        if info != 0:
            raise ValueError(f"illegal value in argument {-info} of getrs")
        if not estimate:
            return x.reshape(b0.shape)
        if rcond is not None:
            return _warn_ill(rcond, dtype, x.reshape(b0.shape))
    rcond, _info = gecon(lu, anorm, norm="I")
    return _warn_ill(rcond, dtype, x.reshape(b0.shape))


# ---------------------------------------------------------------------------
# a kept factor under mixed_lu: complex64 LU + the complex128 matrix
# ---------------------------------------------------------------------------


class MixedLU:
    """The factor a solver KEEPS for later solves, under ``mixed_lu``
    (momwire#2125): the complex64 LU of ``m`` and ``m`` itself, intact.

    Made only by `lu_factor_kept`; used wherever ``(lu, piv)`` from
    `lu_factor_strict` would be, through `lu_solve`. Every solve,
    ``op(m) x = b`` with ``trans`` 0 (``m``) or 1 (``mᵀ``), is refined by
    `_mixed_sweeps` against ``m`` from the one complex64 factor, so a later
    right-hand side (the adjoint, another drive) costs O(n^2) sweeps, not a
    factor. The first solve that falls back (stall, cap, nonfinite), or one
    the route does not take (``trans`` 2, more columns than
    `_mixed_nrhs_cap`, a non-complex128 ``b``), drops the complex64 factor
    and factors ``m`` IN PLACE by `lu_factor_strict`, the call the flag-off
    route makes on the same untouched bytes; that solve and every later one
    are then the ordinary `lu_solve`, the flag-off bits.

    Memory: ``m`` (Z) and the complex64 factor (0.5 Z) until a fallback,
    then the complex128 factor in ``m``'s storage alone (Z). Never a
    complex128 factor beside ``m``: the opt-in allowance is 1.5 Z + 512 MB.

    Unpacking it (``lu, piv = f``, as `scipy.linalg.lu_solve` does) factors
    ``m`` first, so the pair is the flag-off one; that costs the complex128
    factor and ends the refinement."""

    __slots__ = ("_m", "_anorm", "_lu32", "_ipiv", "_getrs", "_lu")

    def __init__(self, m, anorm_t, lu32, ipiv, getrs):
        self._m = m  # F-contiguous, writeable, complex128; owned from here
        # ||op(m)||_1 by trans: 1 is ||mᵀ||_1 (taken with the factor, as
        # `solve_transposed` takes it); 0, ||m||_1, on its first use.
        self._anorm = {1: anorm_t}
        self._lu32, self._ipiv, self._getrs = lu32, ipiv, getrs
        self._lu = None

    @property
    def refining(self) -> bool:
        """True until a solve has fallen back to the complex128 factor."""
        return self._lu is None

    def __iter__(self):
        return iter(self._factored())

    def _factored(self):
        """``(lu, piv)`` of ``m``, factored in place on first call."""
        if self._lu is None:
            self._lu32 = self._ipiv = self._getrs = None  # before m's factor
            self._lu = lu_factor_strict(self._m, overwrite_a=True)
            self._m = None
        return self._lu

    def _anorm_for(self, trans: int) -> float:
        if trans not in self._anorm:
            lange = scipy.linalg.get_lapack_funcs("lange", (self._m,))
            self._anorm[trans] = float(lange("1", self._m))
        return self._anorm[trans]

    def solve(self, b, trans=0, overwrite_b=False, check_finite=True):
        """`lu_solve`'s contract on the kept factor (see the class)."""
        if self._lu is not None:
            return lu_solve(self._lu, b, trans, overwrite_b, check_finite)
        b0 = np.asarray(b)
        n = self._m.shape[0]
        if b0.ndim in (1, 2) and b0.shape[0] != n:
            raise ValueError(
                f"the right-hand side has {b0.shape[0]} rows; "
                f"the factors are {n} by {n}"
            )
        nrhs = 1 if b0.ndim == 1 else (b0.shape[1] if b0.ndim == 2 else 0)
        if (
            b0.ndim not in (1, 2)
            or b0.size == 0
            or trans not in (0, 1)
            or np.result_type(_C128, b0.dtype) != _C128
            or nrhs > _mixed_nrhs_cap(n)
        ):
            _mixed_declined()
            return lu_solve(self._factored(), b, trans, overwrite_b, check_finite)
        if check_finite and not np.isfinite(b0).all():
            raise ValueError("array must not contain infs or NaNs")
        _thread_policy.hold()
        b2 = _fortran_copy(b0.reshape(n, 1) if b0.ndim == 1 else b0, _C128)
        # op(m) as the residual product reads it: m.T is C-contiguous, m
        # F-contiguous; matmul reads either in place.
        a1 = self._m.T if trans == 1 else self._m
        x = _mixed_sweeps(
            a1,
            self._lu32,
            self._ipiv,
            self._getrs,
            trans,
            b2,
            self._anorm_for(trans),
            "1",
            _mixed_record(n, b2.shape[1]),
        )
        if x is not None:
            return x.reshape(b0.shape)
        return lu_solve(self._factored(), b, trans, overwrite_b, check_finite)


@_factor_stage
def lu_factor_kept(a, overwrite_a=False, check_finite=True):
    """`lu_factor_strict` for a factor the caller KEEPS and solves through
    `lu_solve` (the sinusoidal impedance's ``Z_factors``).

    Outside `mixed_lu_scope`, or where the route does not take ``a``, this
    IS ``lu_factor_strict(a, overwrite_a, check_finite)``: the same bits.
    Under the scope, a complex128 ``a`` that `_mixed_takes` at one
    right-hand side and that can be given away (``overwrite_a`` on an
    F-contiguous writeable array: the factor would have destroyed it
    anyway) is cast to complex64 and factored, and a `MixedLU` holding both
    is returned; ``a`` stays intact inside it for the residuals. A matrix
    out of the cast's range, or exactly singular in complex64, counts its
    fallback and is factored here by `lu_factor_strict`, so an exactly
    singular ``a`` raises its ordinary error. Any other ``a`` (not
    giveable, not complex128) counts as declined."""
    if not _MIXED.get():
        return lu_factor_strict(a, overwrite_a=overwrite_a, check_finite=check_finite)
    a1 = np.asarray(a)
    if not (_square(a1) and a1.shape[0] > 1):
        return lu_factor_strict(a, overwrite_a=overwrite_a, check_finite=check_finite)
    n = a1.shape[0]
    if not _mixed_takes(_host_for(a1.dtype), a1.dtype, n, 1):
        return lu_factor_strict(a, overwrite_a=overwrite_a, check_finite=check_finite)
    if not (overwrite_a and a1.flags.f_contiguous and a1.flags.writeable):
        _mixed_declined()
        return lu_factor_strict(a, overwrite_a=overwrite_a, check_finite=check_finite)
    # ||a1ᵀ||_1, the norm `solve_transposed` reads for the same system: a1ᵀ
    # is C-contiguous, read in place and threaded.
    finite, anorm_t = _finite_1norm_c(a1.T)
    if not finite:
        # The ordinary route's error, uncounted (as `_mixed_factor_and_solve`).
        return lu_factor_strict(a, overwrite_a=overwrite_a, check_finite=check_finite)
    _thread_policy.hold()
    record = _mixed_record(n, 1)
    if not _mixed_in_range(n, anorm_t):
        _mixed_fallback("range", record)
        return lu_factor_strict(a1, overwrite_a=True, check_finite=check_finite)
    factors = _mixed_factor32(lambda: np.array(a1, dtype=np.complex64, order="F"))
    if factors is None:
        _mixed_fallback("singular", record)
        return lu_factor_strict(a1, overwrite_a=True, check_finite=check_finite)
    return MixedLU(a1, anorm_t, *factors)


#: At and above this order the no-host `solve` runs gecon on a second
#: thread beside getrs (momwire#1916). Both only read the factors, so neither
#: moves the other's bits; gecon is single-threaded, so it costs a core for
#: as long as getrs takes, which then costs nothing: 0.09 s of gecon and
#: 0.01-0.025 s of getrs per whip frequency (n = 4392, Rocketlake 8T).
_GECON_BESIDE_MIN_N = 1024


def _gecon_beside(gecon, lu, anorm, norm="1"):
    """Start `gecon(lu, anorm, norm=norm)` on a thread; returns the join,
    which gives rcond, or None where the thread raised (the caller then runs
    gecon itself, so its error is the caller's)."""
    box = {}

    def run():
        try:
            box["rcond"] = gecon(lu, anorm, norm=norm)[0]
        except Exception:  # noqa: BLE001 - the caller reruns gecon and sees it
            pass

    t = threading.Thread(target=run, name="momwire-gecon", daemon=True)
    t.start()

    def join():
        t.join()
        return box.get("rcond")

    return join


def _warn_ill(rcond, dtype, x):
    """`solve`'s ill-conditioning warning, as scipy.linalg.solve raises it
    (named at `solve`'s caller); returns `x`."""
    if rcond < np.finfo(dtype).eps:
        warnings.warn(
            f"An ill-conditioned matrix detected: rcond = {rcond:.6g}.",
            LinAlgWarning,
            stacklevel=4,
        )
    return x


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


@_factor_stage
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
