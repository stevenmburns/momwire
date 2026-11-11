"""`pair_reuse` on the point-matched sinusoidal fill (momwire#1911).

The fill's field tensors Φ (one value per shape: constant, sin, cos) at an
(observer point, source segment) pair depend on the pair's geometry only
through four scalars, measured in the SOURCE's frame (origin at its centre,
z along its natural tangent):

* ``z``   -- the observer centre's coordinate along the source axis;
* ``rho`` -- its distance from that axis;
* ``td``  -- the cosine between the observer and source tangents;
* ``rdt`` -- the radial vector dotted with the observer tangent;

and, beyond those, on the source's half-length, the observer's radius and,
under the extended kernel, the source's radius and its two end codes. A
straight source's current is axisymmetric, so its field has no azimuthal
part and the projection onto the observer's tangent needs nothing more.
Both C++ kernels (`sinusoidal_field_tensor`, `_ek`) read exactly these
quantities (`_sinusoidal_core.cpp`, stage A), so two pairs agreeing on all
of them have the same Φ: every rigid motion, mirrored pairs included.

THE KEY quantises the four scalars with a step of 2^-KEY_BITS times the
pair's singular distance (the observer centre to the source segment, within
sqrt(2), floored at the observer radius), rounded down to a power of two so
the step is a function of the value alone; the source's half-length
is quantised the same way, and the radii and end codes are exact. A member
differs from its representative by at most 2^-(KEY_BITS+1) relative to the
scale at which Φ varies (k r times that in phase), and takes its Φ. Pairs
that straddle a quantum land in neighbouring classes, which costs
compression, never accuracy. The classes are exact-equality groups of the
quantised integers (the `factorize_ints` hash kernel, staged over its three
columns), so no two different keys can share a class.

EXCLUDED: a pair whose observer lies within `NEAR_LENGTHS` source lengths of
the source's centre (in the max norm of |z| and rho: the self term and its
neighbours) is not shared, and is evaluated by the shipped kernel in the
ORIGINAL frame, exactly as the fill would (`near_blocks`). Those pairs carry
the near-singular field that the basis sums over neighbouring sources
cancel, so any rounding of their geometry other than the structure's own,
including the canonical frame's, is amplified into the answer.

EVALUATION is the shipped kernel in the OBSERVER's frame: one observer row
at the origin along +z, and each class's source placed so the four scalars
come out as the representative's, its half-length, radius and end codes
taken from the representative's own source. A row against many sources is
the kernel's efficient shape; the rows are split across a thread pool whose
workers each run the kernel on one OpenMP thread. Classes are found once per
geometry (the key holds no k), Φ per class once per frequency, then gathered
into each observer band, so the reduction (`_reduce_phi_band`) is
unchanged. Memory: an (N_obs, N_src) int32 class map, a quarter of Z, plus
the class table; no Φ-sized copy.
"""

from __future__ import annotations

import functools
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from ._accel import acc as _acc

#: Relative resolution of the key, in bits.
KEY_BITS = 48

#: Pairs whose observer lies within this many source lengths of the source's
#: centre are not shared: each is evaluated directly, in the original frame.
#: Their field is near-singular, and the basis sums over neighbouring
#: sources cancel most of it, so a geometry rounding other than the
#: structure's own is amplified there (measured: canonical-frame near pairs
#: moved a 16-bowtie array's input impedance 7.7e-11, original-frame ones
#: 6e-14).
NEAR_LENGTHS = 4.0

#: Observer rows per direct near-pair kernel call (their near sources'
#: union is the call's source list).
NEAR_ROWS = 16

_NEAR_SENTINEL = -1  # below every packed exponent-and-type column

#: Sources per kernel call in `evaluate` (its per-row scratch is ~360 bytes
#: per source).
EVAL_CHUNK = 8192

#: `pair_reuse="verify"`: per-pair agreement bound, |Φ_reused - Φ_direct|
#: over the pair's largest |Φ_direct| among the three shapes.
VERIFY_TOL = 1e-10

#: `pair_reuse="verify"`: pairs checked per observer band.
VERIFY_SAMPLES = 512

_HAVE_FACTORIZE = _acc is not None and bool(
    getattr(_acc, "exact_factorize_1224", False)
)


class PairReuseMismatch(AssertionError):
    """`pair_reuse="verify"` found a reused Φ that the direct kernel does not
    reproduce."""


@dataclass
class Plan:
    """Which class each (observer, source) pair belongs to, and how to
    evaluate each class once."""

    cls: np.ndarray  # (N_obs, N_src) int32 class of each pair
    canon: np.ndarray  # (n_cls, 4) float64 rho, z, rdt, td of the representative
    rep_src: np.ndarray  # (n_cls,) int64 the representative's source segment
    rep_obs: np.ndarray  # (n_cls,) int64 the representative's observer segment
    near_flat: np.ndarray  # (n_near,) int64 sorted flat index m * N_src + n
    near_blocks: list  # (m0, m1, sources, row, col) per direct kernel call

    @property
    def n_near(self) -> int:
        return int(self.near_flat.size)

    @property
    def n_pairs(self) -> int:
        return int(self.cls.size)

    @property
    def n_classes(self) -> int:
        return int(self.canon.shape[0])


def _types(cols):
    """Dense ids for the distinct rows of a few per-segment columns."""
    stacked = np.stack([np.asarray(c) for c in cols], axis=1)
    _, ids = np.unique(stacked, axis=0, return_inverse=True)
    return np.asarray(ids, dtype=np.int64).reshape(-1)


def _quantise_length(h):
    """(exponent, mantissa) of a positive length at KEY_BITS relative bits."""
    m, e = np.frexp(h)
    return e.astype(np.int64), np.rint(np.ldexp(m, KEY_BITS)).astype(np.int64)


def segment_types(seg_h, a_obs, src_a=None, ind1=None, ind2=None):
    """Per-segment source-type and observer-type ids.

    The source type holds what Φ reads of the source besides its pose: its
    half-length (quantised like the key) and, under the extended kernel, its
    radius and end codes (exact). The observer type is its radius (exact),
    the scalar each kernel call takes."""
    e, q = _quantise_length(np.asarray(seg_h, dtype=np.float64))
    cols = [e, q]
    if src_a is not None:
        cols += [
            np.asarray(src_a, dtype=np.float64).view(np.int64),
            np.asarray(ind1, dtype=np.int64),
            np.asarray(ind2, dtype=np.int64),
        ]
    src_type = _types(cols)
    obs_type = _types([np.asarray(a_obs, dtype=np.float64).view(np.int64)])
    return src_type, obs_type


def pair_scalars(obs_c, obs_t, src_c, src_t):
    """z, rho, td, rdt for every (observer, source) pair, (R, N) each, in
    the kernels' own order of operations."""
    rvx = obs_c[:, None, 0] - src_c[None, :, 0]
    rvy = obs_c[:, None, 1] - src_c[None, :, 1]
    rvz = obs_c[:, None, 2] - src_c[None, :, 2]
    tnx, tny, tnz = src_t[None, :, 0], src_t[None, :, 1], src_t[None, :, 2]
    z = rvx * tnx + rvy * tny + rvz * tnz
    rvx -= z * tnx
    rvy -= z * tny
    rvz -= z * tnz
    rho = np.sqrt(rvx * rvx + rvy * rvy + rvz * rvz)
    tmx, tmy, tmz = obs_t[:, None, 0], obs_t[:, None, 1], obs_t[:, None, 2]
    td = tmx * tnx + tmy * tny + tmz * tnz
    rdt = rvx * tmx + rvy * tmy + rvz * tmz
    return z, rho, td, rdt


def _band_keys(z, rho, td, rdt, H, a_obs, type_id, near=False):
    """The quantised key columns of one band (int64, (R*N,) each).

    With `near=True`, a NEAR pair (`NEAR_LENGTHS`) takes the one sentinel
    key: its value is the direct kernel's, written over the gather. Returns
    the columns and the near mask (or None)."""
    # The pair's singular distance, from the observer centre to the source
    # SEGMENT (its axis beside the span, its nearer end beyond it), within
    # a factor sqrt(2) as max(rho, |z| - H), floored at the observer radius.
    az = np.abs(z)
    near = (
        (np.maximum(az, rho) < (2.0 * NEAR_LENGTHS) * H[None, :]).reshape(-1)
        if near and NEAR_LENGTHS > 0
        else None
    )
    az -= H[None, :]
    scale = np.maximum(az, rho, out=az)
    np.maximum(scale, a_obs[:, None], out=scale)
    # frexp's exponent and 2^(KEY_BITS - e), off the bits (scale is a
    # positive normal number): the same integers, without two slow ufuncs.
    e = (scale.view(np.int64) >> 52) - 1022
    del scale, az
    step_inv = ((1023 + KEY_BITS - e) << 52).view(np.float64)
    zq = np.rint(z * step_inv).astype(np.int64)
    rq = np.rint(rho * step_inv).astype(np.int64)
    dq = np.rint(rdt * step_inv).astype(np.int64)
    del step_inv
    tq = np.rint(td * float(1 << KEY_BITS)).astype(np.int64)
    # The exponent and the type share a column: |e| < 2^11 always, and the
    # type id is below N^2 < 2^40.
    e += 1 << 11
    e <<= 40
    e += type_id
    cols = [c.reshape(-1) for c in (e, zq, rq, dq, tq)]
    if near is not None and near.any():
        idx = np.flatnonzero(near)
        cols[0][idx] = _NEAR_SENTINEL
        for c in cols[1:]:
            c[idx] = 0
    return cols, near


def _group(cols):
    """`(first, inverse)`: exact-equality groups of the rows of int64
    columns. The hash kernel takes three columns, so five are grouped in
    two stages, the first stage's ids a column of the second; without it, numpy's
    sorted spelling (other group numbers, the same groups)."""
    if _HAVE_FACTORIZE and cols[0].shape[0] < 2**31 - 2:
        _, ids = _acc.factorize_ints(list(cols[:3]))
        rest = list(cols[3:])
        while rest:
            take, rest = rest[:2], rest[2:]
            first, ids = _acc.factorize_ints([np.asarray(ids, np.int64), *take])
        return np.asarray(first, np.int64), np.asarray(ids, np.int64).reshape(-1)
    _, first, inv = np.unique(
        np.stack(cols, axis=1), axis=0, return_index=True, return_inverse=True
    )
    return np.asarray(first, np.int64), np.asarray(inv, np.int64).reshape(-1)


def _limit_openmp():
    """A pool worker's kernels run on one OpenMP thread: the pool is the
    parallelism. libgomp keeps the thread count per calling thread, so this
    leaves every other thread's as it was."""
    from threadpoolctl import ThreadpoolController

    ThreadpoolController().select(user_api="openmp").limit(limits=1)


@functools.lru_cache(maxsize=1)
def _pool():
    """The census's, the evaluation's and the gather's pool: as many workers
    as this process may run on, held to OMP_NUM_THREADS where the caller
    pinned one."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    try:
        n = min(n, int(os.environ.get("OMP_NUM_THREADS", "") or n))
    except ValueError:
        pass
    return ThreadPoolExecutor(max_workers=max(1, n), initializer=_limit_openmp)


def _workers():
    return _pool()._max_workers


def _run(fn, items):
    """`fn` over `items`, on the pool when it has more than one worker."""
    if _workers() == 1 or len(items) == 1:
        return [fn(x) for x in items]
    return list(_pool().map(fn, items))


def build_plan(obs_c, obs_t, src_c, src_t, seg_h, a_obs, src_type, obs_type, rows):
    """The class map over every (observer, source) pair, found in observer
    bands whose transients together hold `rows` rows (O(rows * N_src))."""
    obs_c = np.asarray(obs_c, dtype=np.float64)
    obs_t = np.asarray(obs_t, dtype=np.float64)
    src_c = np.asarray(src_c, dtype=np.float64)
    src_t = np.asarray(src_t, dtype=np.float64)
    a_obs = np.asarray(a_obs, dtype=np.float64)
    H = 0.5 * np.asarray(seg_h, dtype=np.float64)
    M, N = obs_c.shape[0], src_c.shape[0]
    n_obs_types = int(obs_type.max()) + 1
    cls = np.empty((M, N), dtype=np.int32)
    workers = _workers()
    rows = max(1, int(rows) // workers)
    # Enough bands to share out, none so small that its per-band cost shows.
    rows = min(rows, max(16, -(-M // (4 * workers))))
    starts = list(range(0, M, rows))

    def band(i0):
        i1 = min(M, i0 + rows)
        z, rho, td, rdt = pair_scalars(obs_c[i0:i1], obs_t[i0:i1], src_c, src_t)
        type_id = src_type[None, :] * n_obs_types + obs_type[i0:i1, None]
        cols, near = _band_keys(z, rho, td, rdt, H, a_obs[i0:i1], type_id, True)
        del type_id
        near = (
            np.zeros(0, np.int64)
            if near is None
            else np.flatnonzero(near).astype(np.int64) + i0 * N
        )
        first, inv = _group(cols)
        cls[i0:i1] = inv.reshape(i1 - i0, N)
        keys = [c[first] for c in cols]
        canon = np.stack(
            [
                rho.reshape(-1)[first],
                z.reshape(-1)[first],
                rdt.reshape(-1)[first],
                td.reshape(-1)[first],
            ],
            axis=1,
        )
        r_first, n_first = np.divmod(first, N)
        return keys, canon, r_first + i0, n_first, near

    results = _run(band, starts)
    # The bands' classes merged: a key seen in two bands is one class.
    keys = [np.concatenate([r[0][j] for r in results]) for j in range(5)]
    gfirst, ginv = _group(keys)
    del keys
    off = 0
    for (k, *_), i0 in zip(results, starts):
        i1 = min(M, i0 + rows)
        n_b = k[0].shape[0]
        cls[i0:i1] = ginv[off : off + n_b].astype(np.int32)[cls[i0:i1]]
        off += n_b
    canon = np.concatenate([r[1] for r in results])[gfirst]
    rep_obs = np.concatenate([r[2] for r in results])[gfirst]
    rep_src = np.concatenate([r[3] for r in results])[gfirst]
    near_flat = np.concatenate([r[4] for r in results])  # sorted: bands in order
    return Plan(
        cls=cls,
        canon=canon,
        rep_src=rep_src.astype(np.int64),
        rep_obs=rep_obs.astype(np.int64),
        near_flat=near_flat,
        near_blocks=near_blocks(near_flat, N, a_obs),
    )


def near_blocks(near_flat, n_src, a_obs, rows=None):
    """The direct kernel calls that evaluate the near pairs: per run of up to
    NEAR_ROWS observer rows of one radius, the union of their near sources,
    and where each near pair sits in that call's (rows, sources) block."""
    rows = NEAR_ROWS if rows is None else rows
    m_all, n_all = np.divmod(near_flat, n_src)
    out = []
    if near_flat.size == 0:
        return out
    # Cut at every NEAR_ROWS boundary and at every change of observer radius.
    cut = np.flatnonzero(np.diff(a_obs)) + 1
    edges = np.union1d(np.arange(0, a_obs.shape[0] + 1, rows), cut)
    edges = np.union1d(edges, [a_obs.shape[0]])
    lo = np.searchsorted(m_all, edges[:-1], side="left")
    hi = np.searchsorted(m_all, edges[1:], side="left")
    for m0, m1, p0, p1 in zip(edges[:-1], edges[1:], lo, hi):
        if p1 == p0:
            continue
        ns, col = np.unique(n_all[p0:p1], return_inverse=True)
        out.append(
            (int(m0), int(m1), ns, m_all[p0:p1] - m0, col.reshape(-1), int(p0), int(p1))
        )
    return out


def evaluate_near(plan, direct):
    """The near pairs' Φ, (3, n_near): `direct(m0, m1, ns)` is the shipped
    kernel on observer rows m0:m1 against source segments `ns`, in the
    original frame."""
    out = np.empty((3, plan.n_near), dtype=np.complex128)

    def one(block):
        m0, m1, ns, r, c, p0, p1 = block
        phis = direct(m0, m1, ns)
        for sh in range(3):
            out[sh, p0:p1] = np.asarray(phis[sh])[r, c]

    _run(one, plan.near_blocks)
    return out


def canonical_sources(canon):
    """Source centres and tangents placing each class in the canonical
    observer frame: the observer at the origin along +z.

    With the source tangent u = (s, 0, td), s = sqrt(1 - td^2), the radial
    vector is alpha p + beta q over p = (-td, 0, s) and q = (0, 1, 0), both
    normal to u: alpha s = rdt (its projection on the observer tangent) and
    alpha^2 + beta^2 = rho^2. The observer sits at rho_vec + z u from the
    source's centre."""
    rho, z, rdt, td = canon[:, 0], canon[:, 1], canon[:, 2], canon[:, 3]
    s = np.sqrt(np.maximum(0.0, 1.0 - td * td))
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.where(s > 0.0, rdt / s, 0.0)
    alpha = np.clip(alpha, -rho, rho)
    beta = np.sqrt(np.maximum(0.0, rho * rho - alpha * alpha))
    n = canon.shape[0]
    t = np.zeros((n, 3))
    t[:, 0] = s
    t[:, 2] = td
    rv = np.empty((n, 3))  # observer minus source
    rv[:, 0] = -alpha * td + z * s
    rv[:, 1] = beta
    rv[:, 2] = alpha * s + z * td
    return -rv, t


def evaluate(plan, call, obs_type):
    """Φ per class: (3, n_cls) complex. `call(src_c, src_t, ns, g_obs)` runs
    the kernel for the canonical observer at observer segment `g_obs`'s
    radius against canonical sources standing for source segments `ns`,
    returning its (3, 1, len(ns)) tuple."""
    out = np.empty((3, plan.n_classes), dtype=np.complex128)
    src_c, src_t = canonical_sources(plan.canon)
    otype = obs_type[plan.rep_obs]
    pieces = []
    for t in np.unique(otype):
        idx = np.flatnonzero(otype == t)
        g_obs = int(plan.rep_obs[idx[0]])
        for c0 in range(0, idx.size, EVAL_CHUNK):
            pieces.append((idx[c0 : c0 + EVAL_CHUNK], g_obs))

    def one(piece):
        idx, g_obs = piece
        phis = call(
            np.ascontiguousarray(src_c[idx]),
            np.ascontiguousarray(src_t[idx]),
            plan.rep_src[idx],
            g_obs,
        )
        for sh in range(3):
            out[sh, idx] = phis[sh][0]

    _run(one, pieces)
    return out


def scatter(rep, plan, rows, near=None):
    """The band's three Φ, gathered from the class values (on the pool, by
    row pieces), the near pairs then written from `near` (`evaluate_near`)."""
    i0, i1 = rows
    c = plan.cls[i0:i1]
    out = tuple(np.empty(c.shape, dtype=np.complex128) for _ in range(3))
    workers = _workers()
    step = max(1, -(-c.shape[0] // workers))
    pieces = [(s, r) for s in range(3) for r in range(0, c.shape[0], step)]

    def take(piece):
        s, r = piece
        np.take(rep[s], c[r : r + step], out=out[s][r : r + step], mode="clip")

    _run(take, pieces)
    if near is not None and plan.n_near:
        n_src = c.shape[1]
        p0, p1 = np.searchsorted(plan.near_flat, [i0 * n_src, i1 * n_src])
        local = plan.near_flat[p0:p1] - i0 * n_src
        for sh in range(3):
            out[sh].reshape(-1)[local] = near[sh, p0:p1]
    return out


def verify_band(phis, rows, n_src, direct, samples=VERIFY_SAMPLES, tol=VERIFY_TOL):
    """Check `samples` pairs of the band against the direct kernel.

    `direct(m, ns)` returns the kernel's tuple for observer row `m` against
    source segments `ns`, in the original frame. Returns the largest
    relative difference seen; raises `PairReuseMismatch` above `tol`.

    Relative to the pair's largest |Φ| over the three shapes, floored at
    1e-6 of the sample's largest: a far pair whose Φ is itself a
    cancellation (perpendicular, say) carries the input rounding of any
    frame at a relative size no reuse could hold."""
    i0, i1 = rows
    n_pairs = (i1 - i0) * n_src
    flat = np.unique(
        np.linspace(0, n_pairs - 1, min(samples, n_pairs)).astype(np.int64)
    )
    r, n = np.divmod(flat, n_src)
    diff, scale = [], []
    for m in np.unique(r):
        ns = n[r == m]
        d = direct(int(m) + i0, ns)
        want = [np.asarray(d[s])[0] for s in range(3)]
        diff.append(
            np.maximum.reduce([np.abs(phis[s][m, ns] - want[s]) for s in range(3)])
        )
        scale.append(np.maximum.reduce([np.abs(w) for w in want]))
    diff, scale = np.concatenate(diff), np.concatenate(scale)
    floor = 1e-6 * float(scale.max()) if scale.size else 0.0
    den = np.maximum(scale, floor)
    worst = float(np.max(diff / np.where(den > 0, den, 1.0))) if diff.size else 0.0
    if worst > tol:
        raise PairReuseMismatch(
            f"pair_reuse: a reused field tensor differs from the direct kernel "
            f"by {worst:.3g} relative (bound {tol:g})"
        )
    return worst
