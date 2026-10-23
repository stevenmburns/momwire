"""Razor's exact-kernel correction, split into what geometry decides and what
each call computes, with a small cross-call cache of the first (momwire#1454).

`RazorSolver._add_exact_kernel_correction` used to rebuild, on every call,
everything below that depends on geometry alone: the coaxial groups, each
group's axial frame, the column and row tables of the wing algebra, the
observer points and the observer windows. On a small deck that was most of
the correction (measured on Dan's UR0GT 7.3 MHz deck, warm, one thread:
`coaxial_groups` ~2.4 ms and the tables ~1.5 ms of a ~7.9 ms correction).
In the resident server a SimNEC client asks for the same deck again and
again, so this module keeps that work, keyed on everything it is computed
from.

It also keeps each group's `PointRowsAccel` -- the C++ point-moment source,
which remembers every pair it has evaluated -- for the wavenumber it was
built at. A second call at the same k then reads every pair moment from that
object's own cache instead of re-integrating it (the other ~2.7 ms). Pair
moments depend on k, so a different k never sees them.

EXACT, NOT APPROXIMATE. Every number the correction adds is computed by the
same operations on the same values as before:

* The plan holds the inputs the old loop derived from geometry, and the
  per-call arithmetic (`row_cA`, `row_cP`) is the old expression applied to
  the same arrays, part by part, at the same shapes, so even an elementwise
  kernel that treated a tail differently could not move a bit.
* A `PointRowsAccel` reused at the same k on the same observer windows
  returns what a fresh one would. Its cache is filled in first-met order
  and never evicted, so after any prefix of the windows it holds exactly
  the values a fresh object would have stored by that point (each from the
  key's first-met pair), and a key it declined at the cap is recomputed
  from the same first-met pair of its window every time. The same windows
  in the same order therefore read the same values.

The key is the arrays themselves (bytes, dtype and shape), not a hash, so a
geometry, radius, testing-path, label or quadrature-lane change cannot meet
a stale plan. The ground is deliberately not in it: the correction is the
free-space block only, so a plan and its moments are the same over any
ground, and what a ground does reach -- the grounded rows, the EK labels'
mirror -- is in the arrays compared;
the rows key adds k, the kernel's quadrature rules and whether the C++
route serves. Only the C++ route's objects are kept: the numpy reference
(`PointRows`) is the one a test swaps a kernel into.

Step 3 adds what does NOT depend on k: where a group's segments are short
enough at the call's wavenumber (`_exact_kernel_series.serves`), its exact
moments come from a power series in k whose coefficients are geometry-only,
and the plan keeps that group's series source (`PointSeriesAccel`) for its
own lifetime, under a key that holds no wavenumber. A frequency sweep then
integrates each pair once. That route is not bit-identical to the direct
rule (it agrees to ~1e-15 of a moment), so it is behind
`_exact_kernel_series.ENABLED`; with the switch off this module is
momwire#1455's, bit for bit.

Cross-radius jobs (momwire#1438). A coaxial line with a radius step adds one
job after the one-radius groups (`_exact_kernel_seam.coaxial_lines`): the
same tables over the line's segments, plus each segment's radius label and
each observer's radius and label, so the correction can take the two-radius
moments on exactly the pairs whose labels differ. Those tables join the plan:
the line, its labels and its observers are functions of the segment
geometry and radii, which the key already holds byte for byte, and they
count toward `_PLAN_MAX_BYTES` like a group's. The job's moment source does
NOT join the rows cache: it is `_exact_kernel.point_cross_rows`'s (C++
where it serves; the numpy `PointCrossRows`, the route a test swaps
`ring_kernel2` into, otherwise), rebuilt every call, and it never takes the
k-series, whose coefficients are one radius's. A deck with
no step lists no line, so its plan is the one-radius plan exactly.

Chopped rows (momwire#1619 stage 1). On a deck with buried wires the
correction runs on the above medium's own sub-geometry, where each crossing
tent is a half tent: its other wing is a ghost (sigma 0, on the live wing's
segment) and its T2 ends at the knot in the plane instead of at the ghost's
centroid. The knot observer keeps the fill's kernel, so `chop` names those
rows and the plan zeroes the ghost wing's T2 coefficient, the way a grounded
row's plane endpoint is zeroed; T1 needs nothing, since the ghost half's
tangent is zero. `chop` is part of the key. Without it the plan is the one it
always was.

Bounded: `_PLAN_ENTRIES` geometries (least recently used out), each with at
most `_ROWS_ENTRIES` wavenumbers, and a geometry whose tables would exceed
`_PLAN_MAX_BYTES` is computed as before and not stored. The resident server
solves concurrently, so a plan is immutable once built and a wavenumber's
row objects are checked OUT for the length of one correction (a concurrent
call at the same k builds its own, and the first to finish puts its set
back).
"""

from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np

from . import _exact_kernel, _exact_kernel_seam, _exact_kernel_series
from ._vec3 import dot3

# Switch, for gates that compare the cached and uncached routes in one
# process. Read at call time.
ENABLED = True

_PLAN_ENTRIES = 4
_ROWS_ENTRIES = 4
_PLAN_MAX_BYTES = 64 << 20

_LOCK = threading.Lock()
_PLANS: OrderedDict = OrderedDict()

# Hit/miss counters, for gates that must show the cache actually served.
STATS = {"plan_hit": 0, "plan_miss": 0, "rows_hit": 0, "rows_miss": 0}

# The static/dynamic split's counters (step 3, `_exact_kernel_series`): a
# correction that found a geometry's series sources (`hit`) or had to make
# them (`miss`), then per unique pair: coefficient sets integrated (`built`,
# the static cache's misses), pairs found already built (`reused`, its
# hits) and Horner evaluations of the series (`evaluated`).
SERIES_STATS = {"hit": 0, "miss": 0, "built": 0, "reused": 0, "evaluated": 0}


def clear():
    """Drop every cached plan and zero the counters."""
    with _LOCK:
        _PLANS.clear()
        for key in STATS:
            STATS[key] = 0
        for key in SERIES_STATS:
            SERIES_STATS[key] = 0


def _arr_key(a):
    a = np.ascontiguousarray(a)
    return (a.dtype.str, a.shape, a.tobytes())


def _readonly(a):
    a = np.asarray(a)
    a.flags.writeable = False
    return a


class _Group:
    """One coaxial group's geometry-only tables (see `build_plan`)."""

    __slots__ = (
        "n",
        "x0",
        "sgn",
        "h",
        "a",
        "reach",
        "lab",
        "obs_lab",
        "obs_a",
        "col_ptr",
        "B",
        "B64",
        "ent",
        "a_parts",
        "p_parts",
        "by_o",
        "r_o",
        "r_m",
        "obs_pt",
        "obs_x",
        "sub",
        "windows",
        "nbytes",
    )


class Plan:
    """The geometry-only half of the correction; immutable once built."""

    def __init__(self, groups, nbytes):
        self.groups = groups
        self.nbytes = nbytes
        self._rows = OrderedDict()  # rows key -> [PointRowsAccel per group]
        self._series = {}  # series key -> [PointSeriesAccel per group]

    def take_rows(self, key):
        """The row objects stored for `key`, checked out (None on a miss)."""
        with _LOCK:
            got = self._rows.pop(key, None)
            STATS["rows_hit" if got is not None else "rows_miss"] += 1
            return got

    def give_rows(self, key, rows):
        """Put a finished correction's row objects back for `key`."""
        with _LOCK:
            self._rows[key] = rows
            self._rows.move_to_end(key)
            while len(self._rows) > _ROWS_ENTRIES:
                self._rows.popitem(last=False)

    def take_series(self, key):
        """The series sources stored for `key`, checked out (None on a
        miss). They hold no wavenumber, so one set serves every frequency
        for the plan's lifetime."""
        with _LOCK:
            got = self._series.pop(key, None)
            SERIES_STATS["hit" if got is not None else "miss"] += 1
            return got

    def give_series(self, key, series):
        """Put a finished correction's series sources back. One set per
        plan: a set built under other rules replaces the one held."""
        with _LOCK:
            self._series.clear()
            self._series[key] = series


def series_key():
    """What a group's series coefficients depend on besides the plan."""
    return (
        _exact_kernel._N_GL,
        _exact_kernel._N_SPLIT,
        _exact_kernel._GRADE,
        _exact_kernel._T_SWITCH,
        _exact_kernel_series.X_MAX,
        _exact_kernel_series.X_CENTRED,
        _exact_kernel_series.N_RING,
        _exact_kernel_series.SERIES_TOL,
    )


def count_series(built, reused, evaluated):
    with _LOCK:
        SERIES_STATS["built"] += built
        SERIES_STATS["reused"] += reused
        SERIES_STATS["evaluated"] += evaluated


def rows_key(k):
    """What a group's point-moment values depend on besides the plan."""
    return (
        float(k),
        _exact_kernel._N_GL,
        _exact_kernel._N_SPLIT,
        _exact_kernel._GRADE,
        _exact_kernel._T_SWITCH,
    )


def rows_cacheable(k):
    """Whether the row objects at `k` are the C++ route's (the only ones
    kept)."""
    return _exact_kernel.point_accel_serves(k, 2)


def build_plan(geom, seg_a, pts, tans, wts, src_lab, on_cent, chunk_elems, chop=None):
    """The geometry-only tables of every coaxial group that carries a wing,
    in `coaxial_groups`' order, then of every stepped line's cross job
    (momwire#1438). The expressions are the ones
    `RazorSolver._add_exact_kernel_correction` evaluated inline before
    momwire#1454, so the tables hold the same values.

    `chop` (momwire#1619): the sorted chopped rows of a medium's
    sub-geometry, or None. A chopped row's ghost wing (sigma 0) names the
    endpoint its T2 moved to the knot, whose observer stays uncorrected."""
    # Every 3-vector product below is `dot3`'s fixed order, not `@`/einsum
    # (numpy's BLAS FMA chain / einsum's lane split), so the plan is one
    # that a C++ replay (momwire#1479) forms bit for bit.
    seg_p0, seg_t, seg_h = geom["seg_p0"], geom["seg_t"], geom["seg_h"]
    n_seg = seg_h.size
    seg_r = seg_p0 + seg_h[:, None] * seg_t
    wing_seg, wing_sigma = geom["wing_seg"], geom["wing_sigma"]
    wing_rise = geom["wing_rise"]
    n_basis = wing_seg.shape[0]
    grounded = np.zeros(n_basis, dtype=bool)
    grounded[np.asarray(geom["grounded_bases"], dtype=np.int64)] = True
    # The chopped rows' ghost wings: the T2 endpoint at the knot.
    ghost = None
    if chop is not None and len(chop):
        chopped = np.zeros(n_basis, dtype=bool)
        chopped[np.asarray(chop, dtype=np.int64)] = True
        ghost = chopped[:, None] & (wing_sigma == 0.0)
    n_path = pts.shape[1]
    half = n_path // 2
    node_seg = np.concatenate(
        [
            np.repeat(wing_seg[:, :1], half, 1),
            np.repeat(wing_seg[:, 1:], half, 1),
        ],
        1,
    )
    cent = seg_p0 + 0.5 * seg_h[:, None] * seg_t
    obs_rad = None
    if src_lab is not None:
        obs_rad = None if np.all(seg_a == seg_a[0]) else seg_a
    labels, groups = _exact_kernel_seam.coaxial_groups(seg_p0, seg_r, seg_t, seg_a)
    # The one-radius groups, then the cross-radius pairs of every line with a
    # radius step (momwire#1438): their own jobs, after the groups', owning
    # the (observer, source) pairs whose segments carry different radius
    # labels, so the two corrections add. A deck with no step lists no line.
    jobs = [(S, None) for _g, S in groups]
    jobs += _exact_kernel_seam.coaxial_lines(seg_p0, seg_r, seg_t, seg_a, labels)
    out = []
    total = 0
    for S, lab in jobs:
        n = S.size
        axis, x0, sgn, h = _exact_kernel_seam.group_frame(S, seg_p0, seg_r, seg_t)
        loc = np.full(n_seg, -1, dtype=np.int64)
        loc[S] = np.arange(n)
        # Columns: every live wing on the group, grouped by basis (CSR).
        eb, ej, et, er, eq = [], [], [], [], []
        for w in (0, 1):
            ss = wing_seg[:, w]
            b = np.flatnonzero((loc[ss] >= 0) & (wing_sigma[:, w] != 0))
            j = loc[ss[b]]
            eb.append(b)
            ej.append(j)
            et.append(wing_sigma[b, w] * sgn[j])
            er.append(wing_rise[b, w])
            eq.append(wing_sigma[b, w] * np.where(wing_rise[b, w], 1.0, -1.0) / h[j])
        eb = np.concatenate(eb)
        if eb.size == 0:
            continue
        order = np.argsort(eb, kind="stable")
        eb = eb[order]
        ent = [np.ascontiguousarray(np.concatenate(x)[order]) for x in (ej, et, er, eq)]
        ent[0] = ent[0].astype(np.int64)
        ent[2] = ent[2].astype(np.uint8)
        B, first = np.unique(eb, return_index=True)
        col_ptr = np.append(first, eb.size).astype(np.int64)
        # Observers: the group's centroids (T2; T1 too under
        # `nec5_quadrature`), then the path nodes on the group otherwise.
        # Rows (m, observer) here; their jωμ- and 1/jωε-side coefficients
        # are kept as the factors the call multiplies (`row_cA`, `row_cP`).
        obs_pt, obs_seg = [cent[S]], [S]
        r_m, r_o, a_parts, p_parts = [], [], [], []
        for w, sign in ((0, -1.0), (1, 1.0)):
            rows = np.flatnonzero(loc[wing_seg[:, w]] >= 0)
            zero = grounded[rows] if w == 0 else None
            if ghost is not None and ghost[rows, w].any():
                zero = ghost[rows, w] if zero is None else zero | ghost[rows, w]
            p_parts.append((sign, rows.size, zero))
            if on_cent:
                a_parts.append((wts[rows, w], dot3(tans[rows, w], axis)))
            else:
                a_parts.append((None, rows.size))
            r_m.append(rows)
            r_o.append(loc[wing_seg[rows, w]])
        if not on_cent:
            mm, pp = np.nonzero(loc[node_seg] >= 0)
            obs_pt.append(pts[mm, pp])
            obs_seg.append(node_seg[mm, pp])
            r_m.append(mm)
            r_o.append(n + np.arange(mm.size))
            a_parts.append((wts[mm, pp], dot3(tans[mm, pp], axis)))
            p_parts.append((None, mm.size, None))
        obs_pt = np.ascontiguousarray(np.concatenate(obs_pt))
        obs_seg = np.concatenate(obs_seg)
        obs_x = dot3(obs_pt - seg_p0[S[0]], axis)
        r_o = np.concatenate(r_o)
        by_o = np.argsort(r_o, kind="stable")
        r_o = r_o[by_o]
        r_m = np.concatenate(r_m)[by_o].astype(np.int64)
        n_obs = obs_x.size
        chunk = max(1, chunk_elems // max(1, 4 * n))
        windows = []
        for o0 in range(0, n_obs, chunk):
            o1 = min(n_obs, o0 + chunk)
            e0, e1 = np.searchsorted(r_o, [o0, o1])
            ek = None
            if src_lab is not None:
                o = obs_seg[o0:o1]
                ek = (src_lab[o], src_lab[S], None if obs_rad is None else obs_rad[o])
            windows.append((o0, o1, int(e0), int(e1), r_o[e0:e1] - o0, ek))
        g = _Group()
        g.n = n
        g.x0, g.sgn, g.h = x0, sgn, np.ascontiguousarray(h)
        if lab is None:
            g.a = float(seg_a[S[0]])
            g.reach = _exact_kernel_series.group_reach(h, g.a)
            g.lab = g.obs_lab = g.obs_a = None
        else:
            # A cross job: per-segment radii and labels, and each observer's
            # own (`_exact_kernel.PointCrossRows`). The k-series covers one
            # radius only, so no reach: the direct rule always serves it.
            g.a = np.ascontiguousarray(seg_a[S])
            g.reach = None
            g.lab = np.ascontiguousarray(lab)
            lab_of = np.full(n_seg, -1, dtype=np.int64)
            lab_of[S] = lab
            g.obs_lab = np.ascontiguousarray(lab_of[obs_seg])
            g.obs_a = np.ascontiguousarray(seg_a[obs_seg])
        g.col_ptr, g.B, g.B64, g.ent = col_ptr, B, B.astype(np.int64), ent
        g.a_parts, g.p_parts = a_parts, p_parts
        g.by_o, g.r_o, g.r_m = by_o, r_o, r_m
        g.obs_pt, g.obs_x = obs_pt, obs_x
        g.sub = {
            "seg_p0": seg_p0[S],
            "seg_t": seg_t[S],
            "seg_h": seg_h[S],
            "seg_a": seg_a[S],
        }
        g.windows = windows
        g.nbytes = _group_nbytes(g)
        total += g.nbytes
        out.append(g)
    return out, total


def _group_nbytes(g):
    arrs = [g.x0, g.sgn, g.h, g.col_ptr, g.B, g.B64, g.by_o, g.r_o, g.r_m]
    arrs += [g.obs_pt, g.obs_x, *g.ent, *g.sub.values()]
    if g.lab is not None:
        arrs += [g.a, g.lab, g.obs_lab, g.obs_a]
    for part in g.a_parts:
        arrs += [x for x in part if isinstance(x, np.ndarray)]
    for _s, _n, mask in g.p_parts:
        if mask is not None:
            arrs.append(mask)
    for w in g.windows:
        arrs.append(w[4])
        if w[5] is not None:
            arrs += [x for x in w[5] if x is not None]
    nbytes = sum(a.nbytes for a in arrs)
    if g.lab is not None:
        # A cross job keeps no moment source (`point_cross_rows`' is rebuilt
        # each call), so its tables are all it holds.
        return nbytes
    # Each group's PointRowsAccel holds at most 4n + 1024 pairs (~80 B each),
    # one per kept wavenumber.
    nbytes += 80 * (4 * g.n + 1024) * _ROWS_ENTRIES
    if _exact_kernel_series.ENABLED:
        # Its series source holds the same pairs once, each with at most
        # N_CAP real coefficients per moment -- and never more pairs than
        # observers times segments, which is what bounds a two-segment
        # wire's source (a deck of many short wires is many groups).
        pairs = min(4 * g.n + 1024, g.obs_x.size * g.n)
        nbytes += _exact_kernel_series.PAIR_NBYTES * pairs
    return nbytes


def _freeze(groups):
    """Make every array a plan hands out read-only, so a callee that wrote
    into one would raise rather than poison the next call."""
    for g in groups:
        for name in ("x0", "sgn", "h", "col_ptr", "B", "B64", "by_o", "r_o", "r_m"):
            _readonly(getattr(g, name))
        _readonly(g.obs_pt)
        _readonly(g.obs_x)
        if g.lab is not None:
            for a in (g.a, g.lab, g.obs_lab, g.obs_a):
                _readonly(a)
        for a in g.ent:
            _readonly(a)
        for a in g.sub.values():
            _readonly(a)
        for part in g.a_parts:
            for x in part:
                if isinstance(x, np.ndarray):
                    _readonly(x)
        for _s, _n, mask in g.p_parts:
            if mask is not None:
                _readonly(mask)
        for w in g.windows:
            _readonly(w[4])
            if w[5] is not None:
                for x in w[5]:
                    if x is not None:
                        _readonly(x)


def plan_for(
    geom, seg_a, pts, tans, wts, src_lab, on_cent, chunk_elems, context=(), chop=None
):
    """The plan for these inputs: from the cache when the very same inputs
    were planned before, else built (and stored, within the bounds).

    `context` is the solver's own description of its route (class, basis
    flags), compared as part of the key. The plan reads none of it
    beyond what the arrays already carry; it is there so that a solver
    configured differently never shares an entry, whatever the arrays say."""
    if chop is not None and not len(chop):
        chop = None
    if not ENABLED:
        groups, nbytes = build_plan(
            geom, seg_a, pts, tans, wts, src_lab, on_cent, chunk_elems, chop
        )
        return Plan(groups, nbytes), False
    key = (
        _arr_key(geom["seg_p0"]),
        _arr_key(geom["seg_t"]),
        _arr_key(geom["seg_h"]),
        _arr_key(seg_a),
        _arr_key(geom["wing_seg"]),
        _arr_key(geom["wing_sigma"]),
        _arr_key(geom["wing_rise"]),
        _arr_key(np.asarray(geom["grounded_bases"], dtype=np.int64)),
        _arr_key(pts),
        _arr_key(tans),
        _arr_key(wts),
        None if src_lab is None else _arr_key(src_lab),
        bool(on_cent),
        int(chunk_elems),
        tuple(context),
        # the byte bound a plan was admitted under counts the series sources
        bool(_exact_kernel_series.ENABLED),
        None if chop is None else _arr_key(np.asarray(chop, dtype=np.int64)),
    )
    with _LOCK:
        plan = _PLANS.get(key)
        if plan is not None:
            _PLANS.move_to_end(key)
            STATS["plan_hit"] += 1
            return plan, True
        STATS["plan_miss"] += 1
    groups, nbytes = build_plan(
        geom, seg_a, pts, tans, wts, src_lab, on_cent, chunk_elems, chop
    )
    plan = Plan(groups, nbytes)
    if nbytes <= _PLAN_MAX_BYTES:
        _freeze(groups)
        with _LOCK:
            _PLANS[key] = plan
            _PLANS.move_to_end(key)
            while len(_PLANS) > _PLAN_ENTRIES:
                _PLANS.popitem(last=False)
        return plan, True
    return plan, False


def row_cA(g, c_A):
    """The rows' jωμ-side coefficients, in observer order: the expression
    the correction always used, part by part."""
    parts = []
    for W, P in g.a_parts:
        if W is None:
            parts.append(np.zeros(P, dtype=np.complex128))
        else:
            parts.append(c_A * W * P)
    return np.ascontiguousarray(np.concatenate(parts)[g.by_o])


def row_cP(g, c_Phi):
    """The rows' 1/jωε-side coefficients, in observer order."""
    parts = []
    for sign, size, zero in g.p_parts:
        if sign is None:
            parts.append(np.zeros(size, dtype=np.complex128))
            continue
        cP = np.full(size, sign / c_Phi, dtype=np.complex128)
        if zero is not None:
            cP[zero] = 0.0
        parts.append(cP)
    return np.ascontiguousarray(np.concatenate(parts)[g.by_o])
