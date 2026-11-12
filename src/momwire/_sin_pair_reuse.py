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
import threading
from collections import OrderedDict
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
#: per source). At 8192 each worker's scratch spilled out of its core's
#: cache: on the ELT whip at 8 workers (Rocketlake) the kernel's summed
#: seconds were 3.7x one worker's. Measured per frequency there, wall:
#: 8192 0.086 s, 4096 0.059, 2048 0.048, 1024 0.080 (the Python per call
#: holds the GIL), 512 0.135 (momwire#1916). A speed knob, but not a
#: bit-free one: the kernel's batched sincos is a vector loop with a scalar
#: tail, and which a source falls in can move its Φ by an ulp.
EVAL_CHUNK = 2048

#: Classes per `canonical_sources` call in `build_plan` (its transients are
#: about 16 float64 per class).
_CANON_CHUNK = 1 << 18

#: `pair_reuse="verify"`: per-pair agreement bound, |Φ_reused - Φ_direct|
#: over the pair's largest |Φ_direct| among the three shapes.
VERIFY_TOL = 1e-10

#: `pair_reuse="verify"`: pairs checked per observer band.
VERIFY_SAMPLES = 512

_HAVE_FACTORIZE = _acc is not None and bool(
    getattr(_acc, "exact_factorize_1224", False)
)


#: Bytes of plans the process keeps between solvers (`shared_plan`). A plan
#: is mostly its int32 class map, 4 bytes per pair: the ELT whip's direct
#: block is ~35 MB, so the budget holds a few decks' direct and image blocks.
SHARED_PLAN_BYTES = 1 << 30

_shared_plans: OrderedDict = OrderedDict()
_shared_lock = threading.Lock()


def plan_nbytes(plan) -> int:
    """The arrays a plan holds, in bytes (its lists of slices are small)."""
    return sum(
        v.nbytes for v in vars(plan).values() if isinstance(v, np.ndarray)
    ) + sum(
        x.nbytes for blk in plan.near_blocks for x in blk if isinstance(x, np.ndarray)
    )


def shared_plan(key, build):
    """The plan for `key` from the process's cache, else `build()`'s, kept.

    A plan holds no k, and its key is the geometry's bytes, so it outlives
    the solver that found it: the portal builds one solver per frequency and
    a design server one per knob turn back to a value, and each would pay the
    census again with a per-instance cache (momwire#1916 part C). Plans are
    never written after `build_plan` returns, so solvers share one object.
    Least recently used first out past `SHARED_PLAN_BYTES`, the newest kept
    whatever its size. Two threads missing the same key both build; the
    second's plan replaces the first's, and either is the same classes.
    """
    with _shared_lock:
        held = _shared_plans.get(key)
        if held is not None:
            _shared_plans.move_to_end(key)
            return held[0]
    plan = build()
    size = plan_nbytes(plan)
    with _shared_lock:
        _shared_plans[key] = (plan, size)
        _shared_plans.move_to_end(key)
        total = sum(v[1] for v in _shared_plans.values())
        while total > SHARED_PLAN_BYTES and len(_shared_plans) > 1:
            _, (_, dropped) = _shared_plans.popitem(last=False)
            total -= dropped
    return plan


def clear_shared_plans() -> None:
    """Drop every kept plan (tests, and a caller done with its decks)."""
    with _shared_lock:
        _shared_plans.clear()


class PairReuseMismatch(AssertionError):
    """`pair_reuse="verify"` found a reused Φ that the direct kernel does not
    reproduce."""


@dataclass
class Plan:
    """Which class each (observer, source) pair belongs to, and how to
    evaluate each class once."""

    cls: np.ndarray  # (N_obs, N_src) int32 class of each pair
    rep_src: np.ndarray  # (n_cls,) int64 the representative's source segment
    near_flat: np.ndarray  # (n_near,) int64 sorted flat index m * N_src + n
    near_blocks: list  # (m0, m1, sources, row, col) per direct kernel call
    # The canonical evaluation's inputs, k-independent, so built with the
    # classes (momwire#1916): each class's source in the observer frame
    # (`canonical_sources` of the representative's rho, z, rdt, td), and the
    # runs of classes sharing an observer radius, as (c0, c1, observer
    # segment). Classes are numbered so that each run is contiguous.
    src_c: np.ndarray  # (n_cls, 3) float64
    src_t: np.ndarray  # (n_cls, 3) float64
    runs: list

    @property
    def n_near(self) -> int:
        return int(self.near_flat.size)

    @property
    def n_pairs(self) -> int:
        return int(self.cls.size)

    @property
    def n_classes(self) -> int:
        return int(self.rep_src.shape[0])


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


#: Tasks per worker `_run` cuts a long item list into. One future per item
#: cost ~1,300 submits per frequency on the ELT whip (1,032 near blocks and
#: ~280 evaluation pieces), each a submit, a lock and a wake-up; a few
#: contiguous runs per worker keep the load balanced at a fraction of that
#: (momwire#1916 part D). Each item is still its own `fn` call on its own
#: slice, so no value moves.
RUN_TASKS_PER_WORKER = 4


def _run(fn, items):
    """`fn` over `items`, on the pool when it has more than one worker:
    contiguous runs of items per task once there are more items than
    `RUN_TASKS_PER_WORKER` tasks per worker. Results in `items`' order."""
    workers = _workers()
    if workers == 1 or len(items) == 1:
        return [fn(x) for x in items]
    items = list(items)
    n_tasks = workers * RUN_TASKS_PER_WORKER
    if len(items) <= n_tasks:
        return list(_pool().map(fn, items))
    cuts = [len(items) * t // n_tasks for t in range(n_tasks + 1)]
    runs = [items[a:b] for a, b in zip(cuts[:-1], cuts[1:])]

    def run(chunk):
        return [fn(x) for x in chunk]

    return [r for chunk in _pool().map(run, runs) for r in chunk]


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
    # Classes renumbered so those sharing an observer radius are contiguous
    # (stable within a radius): `evaluate` then runs on slices. A class's
    # members and value are unchanged; only its number moves.
    rep_obs_g = np.concatenate([r[2] for r in results])[gfirst]
    perm = np.argsort(obs_type[rep_obs_g], kind="stable")
    renumber = np.empty(perm.size, dtype=np.int64)
    renumber[perm] = np.arange(perm.size, dtype=np.int64)
    ginv = renumber[ginv]
    gfirst = gfirst[perm]
    del renumber, rep_obs_g
    off = 0
    for (k, *_), i0 in zip(results, starts):
        i1 = min(M, i0 + rows)
        n_b = k[0].shape[0]
        cls[i0:i1] = ginv[off : off + n_b].astype(np.int32)[cls[i0:i1]]
        off += n_b
    rep_obs = np.concatenate([r[2] for r in results])[gfirst].astype(np.int64)
    rep_src = np.concatenate([r[3] for r in results])[gfirst].astype(np.int64)
    near_flat = np.concatenate([r[4] for r in results])  # sorted: bands in order
    canon = np.concatenate([r[1] for r in results])[gfirst]
    del results, gfirst, ginv
    n_cls = canon.shape[0]
    src_c = np.empty((n_cls, 3))
    src_t = np.empty((n_cls, 3))
    for c0 in range(0, n_cls, _CANON_CHUNK):  # bounded transients
        c1 = min(n_cls, c0 + _CANON_CHUNK)
        src_c[c0:c1], src_t[c0:c1] = canonical_sources(canon[c0:c1])
    del canon
    otype = obs_type[rep_obs]
    cuts = np.flatnonzero(np.diff(otype)) + 1
    edges = [0, *cuts.tolist(), int(otype.size)]
    runs = [
        (c0, c1, int(rep_obs[c0])) for c0, c1 in zip(edges[:-1], edges[1:]) if c1 > c0
    ]
    return Plan(
        cls=cls,
        rep_src=rep_src,
        near_flat=near_flat,
        near_blocks=near_blocks(near_flat, N, a_obs),
        src_c=src_c,
        src_t=src_t,
        runs=runs,
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
    """The near pairs' Φ, (n_near, 3): `direct(m0, m1, ns)` is the shipped
    kernel on observer rows m0:m1 against source segments `ns`, in the
    original frame."""
    out = np.empty((plan.n_near, 3), dtype=np.complex128)

    def one(block):
        m0, m1, ns, r, c, p0, p1 = block
        phis = direct(m0, m1, ns)
        for sh in range(3):
            out[p0:p1, sh] = np.asarray(phis[sh])[r, c]

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


def evaluate(plan, call, obs_type=None):
    """Φ per class: (n_cls, 3) complex. `call(src_c, src_t, ns, g_obs)` runs
    the kernel for the canonical observer at observer segment `g_obs`'s
    radius against canonical sources standing for source segments `ns`,
    returning its (3, 1, len(ns)) tuple.

    The plan carries the canonical sources and the radius runs
    (`build_plan`), so each call is on slices of them and writes a slice:
    the Python around the kernel, which holds the GIL and so ran one worker
    at a time, is a few views per call (momwire#1916). `obs_type` is
    unused, the runs being the plan's. Calls are `EVAL_CHUNK` sources, small
    enough that a worker's kernel scratch stays in its core's cache."""
    del obs_type
    out = np.empty((plan.n_classes, 3), dtype=np.complex128)
    src_c, src_t, rep_src = plan.src_c, plan.src_t, plan.rep_src
    pieces = [
        (c0, min(c1, c0 + EVAL_CHUNK), g_obs)
        for r0, c1, g_obs in plan.runs
        for c0 in range(r0, c1, EVAL_CHUNK)
    ]

    def one(piece):
        c0, c1, g_obs = piece
        phis = call(src_c[c0:c1], src_t[c0:c1], rep_src[c0:c1], g_obs)
        for sh in range(3):
            out[c0:c1, sh] = phis[sh][0]

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
        np.take(rep[:, s], c[r : r + step], out=out[s][r : r + step], mode="clip")

    _run(take, pieces)
    if near is not None and plan.n_near:
        local, vals = _band_near(plan, rows, near)
        for sh in range(3):
            out[sh].reshape(-1)[local] = vals[:, sh]
    return out


def _band_near(plan, rows, near):
    """The band's near pairs: band-local flat indices and their (k, 3) Φ."""
    i0, i1 = rows
    n_src = plan.cls.shape[1]
    p0, p1 = np.searchsorted(plan.near_flat, [i0 * n_src, i1 * n_src])
    return plan.near_flat[p0:p1] - i0 * n_src, near[p0:p1]


_HAVE_REDUCE = _acc is not None and bool(getattr(_acc, "pair_reuse_reduce_1916", False))


def reduce_takes(Ms, out):
    """Whether `reduce_band` can take this band: the fused kernel is built,
    and the reduction is `reduce_phi_band`'s sparse regime (CSC complex128
    Ms) into complex128 rows."""
    import scipy.sparse

    if not _HAVE_REDUCE or out.ndim != 2 or out.dtype != np.complex128:
        return False
    if out.shape[1] > 1 and out.strides[1] != 16:
        return False
    return all(
        isinstance(M, scipy.sparse.csc_matrix)
        and M.dtype == np.complex128
        and M.shape[1] == out.shape[1]
        for M in Ms
    )


def reduce_band(blocks, rows, Ms, out):
    """`out = Φ₀@M₀ + Φ₁@M₁ + Φ₂@M₂` for the band, Φ gathered by class
    inside the reduction and never materialised (`_accel_sin_pair_reuse.cpp`),
    the bits of `scatter` then `reduce_phi_band`. `blocks` is
    `[(plan, rep, near)]`: the direct block, then the PEC image's, which is
    subtracted. Call only where `reduce_takes`."""
    i0, i1 = rows
    args = []
    for plan, rep, near in blocks:
        local, vals = _band_near(plan, rows, near)
        args.append(
            (
                plan.cls[i0:i1],
                rep,
                np.ascontiguousarray(local, dtype=np.int64),
                np.ascontiguousarray(vals),
            )
        )
    m_args = []
    for M in Ms:
        m_args += [M.indptr, M.indices, M.data]
    image = args[1] if len(args) > 1 else None
    _acc.pair_reuse_reduce_band(out, *args[0], image, *m_args)
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
