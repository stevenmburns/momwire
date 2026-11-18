"""The junction constraints eliminated before the factor (momwire#1898, the
opt-in feature `kcl_nullspace`).

`BSplineSolver` enforces current conservation at its junctions as constraint
rows A (`kcl_A`) and solves the saddle-point system [Z Aᵀ; A 0] by a Schur
complement: one factor of Z, with the n_c columns of Aᵀ back-substituted as
extra right-hand sides (`_solve_with_kcl` and its three siblings). On the
B-spline lanes A has a shape that makes the null space of A trivially sparse:

- every entry is +1, -1 or 0;
- no two rows share a column (each wire end is in at most one junction, and a
  clamped B-spline has exactly one basis nonzero at a wire end).

So at each junction one end e (the lowest index, for determinism) is a
signed sum of the others: s_e x_e = -sum_k s_k x_k, i.e. x_e = sum_k a_k x_k
with a_k = -s_k s_e = +-1. That is x = Q y, with Q the identity on every
other unknown plus one entry a_k at (e, k). A Q = 0 exactly and Q has full
column rank, so the Galerkin system on the constraint subspace is

    (Qᵀ Z Q) y = Qᵀ v,    x = Q y,

the Lagrange/Schur route's solution up to rounding, with n_c fewer unknowns
and no extra right-hand sides.

Q is never formed. Z Q is a column operation (column k gains a_k times
column e), and Qᵀ (Z Q) the same on rows, applied column by column while the
eliminated rows and columns are squeezed out, all in Z's own buffer: about
2 n_c N complex adds and one pass over the kept part of Z. `Qᵀ v` and `Q y` are a
scatter-add and a gather.

Any A not of that shape (an entry other than +-1, a column in two rows, an
empty row) is not reduced: `plan` returns the reason, the caller takes the
Schur route, and `COUNTS` records it.

The sector route (`rotational_symmetry=True`, momwire#1983) solves each DFT
harmonic under its own rows, which are the same disjoint pattern with
weights: sqrt(N) on harmonic 0's sector unknowns, and exp(2 pi j h u / N)
phases on harmonic h's. `plan_weighted` reduces those: each row eliminates
its largest entry, x_e = sum_k (-c_k / c_e) x_k, so |a| <= 1. Harmonic h's
multiplier columns are the CONJUGATE rows, so its test space is conj(Q) and
the reduced system is Q^H Z Q (`Plan.conj_test`); on real rows the two are
the same. `solve_reduced` is the one reduced solve both routes call.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

from . import _lapack

__all__ = [
    "COUNTS",
    "Plan",
    "expand",
    "factor_reduced",
    "plan",
    "plan_weighted",
    "record",
    "reduce_matrix",
    "reduce_rhs",
    "solve_factored",
    "solve_reduced",
]

#: Per process: how often each route was taken. ``"reduced"`` counts reduced
#: solves; ``"schur: <reason>"`` the solves that fell back, by reason.
COUNTS: Counter = Counter()

# Columns (or rows) moved per vectorised step of the column and row
# operations: bounds the temporary at _CHUNK x N complex.
_CHUNK = 256


@dataclass(frozen=True)
class Plan:
    """The null-space basis Q of one constraint matrix, as index arrays.

    `keep` are the unknowns that survive (sorted), `elim` the eliminated end
    of each row (one per row). `k`, `e`, `a` list Q's off-identity entries:
    x[e[i]] gains a[i] * x[k[i]]. `conj_test` tests with conj(Q), for rows
    whose multiplier columns are their conjugates (`plan_weighted`).
    """

    n: int
    keep: np.ndarray
    elim: np.ndarray
    k: np.ndarray
    e: np.ndarray
    a: np.ndarray
    conj_test: bool = False

    @property
    def n_free(self) -> int:
        return int(self.keep.size)

    @property
    def row_a(self) -> np.ndarray:
        """The test side's weights: `a`, or conj(a) under `conj_test`."""
        return np.conj(self.a) if self.conj_test else self.a


def plan(kcl_A):
    """`Plan` for `kcl_A` (n_c, n), or a short reason string when the rows
    are not of the exact shape (+-1 entries, disjoint rows, none empty)."""
    A = np.asarray(kcl_A)
    if A.ndim != 2 or A.shape[0] == 0:
        return "no constraint rows"
    if np.iscomplexobj(A):
        if np.any(A.imag):
            return "complex constraint entries"
        A = A.real
    rows, cols = np.nonzero(A)
    vals = A[rows, cols]
    if not np.all(np.abs(vals) == 1.0):
        return "an entry other than +-1"
    if np.unique(cols).size != cols.size:
        return "a column in two constraint rows"
    n_c, n = A.shape
    counts = np.bincount(rows, minlength=n_c)
    if np.any(counts == 0):
        return "an empty constraint row"
    # np.nonzero is row-major, so each row's columns come out ascending and
    # the first of each row is its lowest index: that end is eliminated.
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    first = np.zeros(cols.size, dtype=bool)
    first[starts] = True
    elim = cols[first]
    s_e = vals[first][rows[~first]]
    k = cols[~first]
    e = elim[rows[~first]]
    a = -vals[~first] * s_e
    mask = np.ones(n, dtype=bool)
    mask[elim] = False
    return Plan(
        n=n,
        keep=np.flatnonzero(mask),
        elim=elim,
        k=k.astype(np.intp),
        e=e.astype(np.intp),
        a=a.astype(np.float64),
    )


#: `plan_weighted`'s floor: an entry below this fraction of its row's
#: largest is refused, not taken. A harmonic row whose two contributions to
#: one unknown cancel comes out at roundoff rather than an exact zero, and
#: dividing by it would be a ruinous pivot; a genuine weight is O(1).
WEIGHT_FLOOR = 1e-8


def plan_weighted(rows, *, conj_test=False):
    """`Plan` for constraint rows of `plan`'s disjoint pattern with any
    nonzero weights, real or complex (the sector route's harmonic rows,
    momwire#1983), or a short reason string. Each row eliminates its largest
    entry (the lowest index among equals), so every |a| <= 1."""
    A = np.asarray(rows)
    if A.ndim != 2 or A.shape[0] == 0:
        return "no constraint rows"
    rr, cols = np.nonzero(A)
    vals = A[rr, cols]
    if np.iscomplexobj(vals) and not np.any(vals.imag):
        vals = vals.real
    if np.unique(cols).size != cols.size:
        return "a column in two constraint rows"
    n_c, n = A.shape
    counts = np.bincount(rr, minlength=n_c)
    if np.any(counts == 0):
        return "an empty constraint row"
    mag = np.abs(vals)
    big = np.zeros(n_c)
    np.maximum.at(big, rr, mag)
    if np.any(mag < WEIGHT_FLOOR * big[rr]):
        return "a near-zero constraint entry"
    # Row-major, so a stable sort on -|c| keeps each row's columns ascending
    # among equals: the first of each row is its largest, lowest index.
    order = np.lexsort((-mag, rr))
    rr, cols, vals = rr[order], cols[order], vals[order]
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    first = np.zeros(cols.size, dtype=bool)
    first[starts] = True
    elim = cols[first]
    c_e = vals[first][rr[~first]]
    k = cols[~first]
    e = elim[rr[~first]]
    a = -vals[~first] / c_e
    if not np.iscomplexobj(a):
        a = a.astype(np.float64)
    mask = np.ones(n, dtype=bool)
    mask[elim] = False
    return Plan(
        n=n,
        keep=np.flatnonzero(mask),
        elim=elim,
        k=k.astype(np.intp),
        e=e.astype(np.intp),
        a=a,
        conj_test=bool(conj_test) and np.iscomplexobj(a),
    )


def _reduce_fortran(Z, p):
    """Qᵀ Z Q of an F-contiguous square `Z`, computed in Z's buffer; returns
    the (n_free, n_free) F-contiguous view of the buffer that holds it."""
    n = p.n
    k, e, a = p.k, p.e, p.a
    # Z Q: column k gains a * column e, a chunk of columns at a time.
    for s in range(0, k.size, _CHUNK):
        kk, ee, aa = k[s : s + _CHUNK], e[s : s + _CHUNK], a[s : s + _CHUNK]
        Z[:, kk] += Z[:, ee] * aa[None, :]
    # Qᵀ (Z Q), squeezed: column j of the result is column keep[j] of Z Q,
    # its row k gaining a * its row e, restricted to the kept rows. Done one
    # contiguous column at a time, so the row operation never walks a row.
    # A destination never overtakes an unread source (keep[j] >= j and
    # n_free <= n), and each source is gathered before its write.
    keep = p.keep
    m = keep.size
    k_at = np.searchsorted(keep, k)
    ra = p.row_a
    buf = Z.reshape(-1, order="F")
    for j in range(m):
        c = int(keep[j])
        col = buf[c * n : (c + 1) * n]
        out = col[keep]
        out[k_at] += ra * col[e]
        buf[j * m : (j + 1) * m] = out
    return buf[: m * m].reshape((m, m), order="F")


def reduce_matrix(Z, p, *, overwrite=False):
    """Qᵀ Z Q for a square 2-D `Z`. With `overwrite` it is computed in Z's own
    buffer when Z is contiguous (Z is consumed); otherwise in a copy. The
    result is F- or C-contiguous, ready for `_lu_solve(overwrite_a=True)`."""
    if not overwrite or not (Z.flags.f_contiguous or Z.flags.c_contiguous):
        Z = np.array(Z, dtype=np.complex128, order="F", copy=True)
    if Z.flags.f_contiguous:
        return _reduce_fortran(Z, p)
    # C order: Zᵀ is F-contiguous and Qᵀ Zᵀ Q = (Qᵀ Z Q)ᵀ. Under
    # `conj_test`, (Q^H Z Q)ᵀ = Qᵀ Zᵀ conj(Q): the two sides' weights swap.
    if p.conj_test:
        p = replace(p, a=np.conj(p.a))
    return _reduce_fortran(Z.T, p).T


def reduce_rhs(V, p):
    """Qᵀ V (Q^H V under `conj_test`) for V of shape (n,) or (n, r): a new
    array of n_free rows."""
    W = np.array(V, dtype=np.complex128, copy=True)
    ra = p.row_a
    if W.ndim == 1:
        W[p.k] += ra * W[p.e]
    else:
        W[p.k] += ra[:, None] * W[p.e]
    return np.ascontiguousarray(W[p.keep])


def expand(y, p):
    """Q y for y of shape (n_free,) or (n_free, r): the full coefficients."""
    x = np.zeros((p.n,) + y.shape[1:], dtype=np.result_type(y, np.complex128))
    x[p.keep] = y
    contrib = x[p.k] * (p.a if y.ndim == 1 else p.a[:, None])
    np.add.at(x, p.e, contrib)
    return x


def record(owner, planned):
    """The route `planned` (a `Plan`, or a reason string) takes, recorded on
    `owner.kcl_nullspace_route` and in `COUNTS`: ``"reduced"``, or
    ``"schur: <reason>"``. Returns the `Plan`, or None for the Schur route."""
    if isinstance(planned, str):
        route = f"schur: {planned}"
        planned = None
    else:
        route = "reduced"
    owner.kcl_nullspace_route = route
    COUNTS[route] += 1
    return planned


def solve_reduced(Z, V, p, *, overwrite=False):
    """The reduced solve, x = Q y with (Qᵀ Z Q) y = Qᵀ V (Q^H under
    `conj_test`): the one step the dense routes (`BSplineSolver.
    _solve_reduced`) and the sector route's harmonics share. `V` is (n,) or
    (n, r); `overwrite` lets Z's own buffer hold the reduction and its LU.
    An exactly singular reduced matrix raises as every dense route does
    (`_lapack.lu_factor_strict`)."""
    Zr = reduce_matrix(Z, p, overwrite=overwrite)
    Vr = np.asfortranarray(reduce_rhs(V, p))
    y = _lapack.factor_and_solve(Zr, Vr, overwrite_a=True, overwrite_b=True)
    return expand(y, p)


def factor_reduced(Z, p, *, overwrite=False):
    """`solve_reduced`'s factor half: the LU of the reduced matrix, for a
    caller that solves it against several right-hand sides later
    (`solve_factored`)."""
    return _lapack.lu_factor_strict(
        reduce_matrix(Z, p, overwrite=overwrite), overwrite_a=True
    )


def solve_factored(lu_piv, V, p):
    """`solve_reduced`'s solve half, on `factor_reduced`'s factors."""
    Vr = np.asfortranarray(reduce_rhs(V, p))
    return expand(_lapack.lu_solve(lu_piv, Vr, overwrite_b=True), p)
