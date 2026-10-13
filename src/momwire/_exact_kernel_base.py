"""The exact-kernel correction's BASE moments, deduplicated (momwire#1421).

`BSplineSolver._add_exact_kernel_correction` adds Z_exact - Z_base over each
coaxial group, where Z_base is the moments the free-space fill put on the
group's pairs, recomputed through the fill's own kernels so the correction
removes them (`BSplineSolver._exact_kernel_base_J`). That recompute evaluated
every pair: O(n²) kernel work, ~3.4 s of the 4.2 s correction on one
3,000-segment fat wire, while the exact moments beside it cost O(n)
(`_exact_kernel.CoaxialRows`). `BaseRows` returns the same observer-row
windows from O(n) kernel work on a uniform run.

What a base moment depends on
-----------------------------
Read off the calls `_exact_kernel_base_J` makes (and must keep making):

* an OFF-EDGE pair (different edges of the group) is
  `_seg_seg_full_moments_offedge(seg_l, seg_r [i and j], a = seg_a[i], k, d,
  n_qp_pair, ek, ladder)`. Per pair that is: both segments' endpoints -- on
  one coaxial group, each segment's length h, its direction σ along the
  axis, its edge's line (the perpendicular offset, constant along an edge),
  and the axial offset between the two; the OBSERVER's radius `a_i`; the EK
  fields `group_i[i]`, `group_j[j]` (the extend-this-pair mask) and the
  tube radii `b_i[i]`, `b_j[j]` (momwire#1368), plus the spec's `ek.a`; and
  the ladder. The order a pair runs at is the ladder tier its ratio (centre
  distance over the longer length) meets, kept only if both lengths pass the
  phase guard (`_phase_split_needed` / `_ladder_for_block`, per pair since
  momwire#920): a function of (h_i, h_j, offset, k, ladder), so of the pair.
* a SAME-EDGE pair is `_seg_seg_static_moments(arc, a_w, d, ek_se)` plus
  `_seg_seg_reg_moments_from_geometry(_seg_seg_reg_geometry(arc, a_w, d,
  n_qp_pair_same_edge, ek_se), k)`: per pair, the two lengths and the arc
  offset between them, on the edge's radius `a_w`, with whole-block EK
  (`_EK_SAME_EDGE`: no per-segment fields).
* constants of one group in one solve, the scope of a `BaseRows` and so of
  its cache: k, d, n_qp_pair, n_qp_pair_same_edge, the ladder, `ek.a`,
  `extended_kernel`, and the kernel route each call dispatches to (which
  reads only those and the shapes).

How it deduplicates
-------------------
A RUN is a maximal stretch of the group's local order over which every
per-segment dependency above is constant: the same edge (so the same line,
radius `a_w` and same-edge kernel), the same σ, `a_i`, and EK `group_i`,
`group_j`, `b_i`, `b_j`, and segments that are one segment stepped rigidly
along the run (the "h" field of `RUN_FIELDS`; see the next section). For
runs A and B with the same step, a pair's separation is a function of ONE
integer, m = σ_B·j' - σ_A·i' (i', j' the positions in the runs), and so is
its whole moment: the (A, B) block is Toeplitz (Hankel for opposite σ). Each
needed m is evaluated once, from a pair on the block's boundary (every
diagonal meets it), by the SAME kernel call on the SAME segments' real
coordinates, and the block's rows are gathered from those values. Pairs on
one diagonal differ only by a translation, so the substitution moves a
moment by the coordinates' roundoff, never by a different rule. Pairs not in
such a block -- unequal steps, short runs, a graded mesh -- are recomputed
exactly as before, in one call per run.

What "the same step" means (the drift bound)
--------------------------------------------
A gathered value serves every pair on its diagonal, up to
min(n_A, n_B) - 1 steps away from the pair it was evaluated on. A tolerance
on segment LENGTHS does not bound what that costs: two runs whose pitches
differ by δh put a pair s steps along a diagonal s·δh off its
representative's separation, and s is as large as the run. So the rule is
on POSITIONS, in 3-D, with an absolute tolerance

    ρ = RUN_TOL_EPS · eps · X,   X = the group's largest |coordinate|.

* A run [s0, s1) is an arithmetic progression: with the step
  w = (l[s1-1] - l[s0]) / (s1 - 1 - s0) of its left ends, every left end
  sits within ρ of l[s0] + i·w and every right end within ρ of r[s0] + i·w.
  A stretch that fails is halved until its pieces pass.
* Runs A and B gather only if (min(n_A, n_B) - 1)·|σ_Aσ_B·w_B - w_A| <= ρ:
  the relative drift over their longest diagonal, axial and perpendicular
  (an edge tilted inside the coaxial rule's 1e-6·a is held to the same ρ).

Every gathered pair is then its representative's two segments translated,
each endpoint to within 2ρ, plus at most ρ of relative drift: any
separation the kernel reads differs from the representative's by at most
5ρ, whatever the run lengths.

Why ρ is counted in eps·X and not in h: a stored coordinate is good to
eps·X, so a uniform mesh's own lengths scatter by ~eps·X (a relative 6e-13
on 3,000 segments over 9 m, 3e-11 on 4,000 segments 20 m from the origin; a
relative length tolerance of 1e-13 would cut such an edge into fragments).
No tolerance under that scale can call a uniform mesh uniform, and none
over it is needed. `_build_geometry` puts a node at (1 - t)·p0 + t·p1,
t = i/n: three roundings of magnitudes up to X, 1.5·eps·X per component, so
the ends sit within ~5·eps·X per component (~9·eps·X in norm) of the fitted
progression at worst. Measured on Haswell over axis-aligned and oblique
edges of 37 to 4,000 segments, up to 34 m from the origin: at most
2.3·eps·X. RUN_TOL_EPS = 16 is the next power of two over the worst case,
so a uniform edge is always one run (the kernel counts of a 600-segment wire
and of 2 x 300 are what a length rule gives), and what the rule admits
beyond a perfect mesh, 5ρ = 80·eps·X of separation, is of the size of the
coordinate roundoff the per-pair route carries itself
(tests/test_exact_kernel_base_dedup_1421.py derives its gate from that).

Branches keyed on pair geometry (the tie hazard)
------------------------------------------------
"A function of m" holds for a moment only if every branch the kernel takes
on the pair's geometry answers the same way along the diagonal. Where a
branch compares a geometric quantity to a threshold and the diagonal sits ON
the threshold, coordinate roundoff decides it pair by pair, and a gathered
value would copy the representative's answer to pairs that took the other
branch. Audited, numpy and C++ (momwire#1421 review):

* the LADDER TIER, `ratio >= threshold` with ratio = centre distance over
  the longer length (`seg_seg_full_moments_bspline_tiered` and `..._ek_tiered`
  in _accel_bspline.cpp, `_pair_ratio` + `np.where(ratio >= r)` in
  `_seg_seg_full_moments_offedge`'s numpy twin), every threshold of the
  ladder (free space 16, buried 2 and 16, or an explicit ladder). Collinear
  equal-length edges put the ratio on an integer, so this tie is common.
  Handled: a diagonal within TIE_REL of a threshold is evaluated pair by pair
  (`_tie_diagonals`);
* the PHASE GUARD, |k|·h <= 0.5 per segment (`_phase_row_mask`, with
  `_phase_split_needed` / `_ladder_for_block`), when the ladder has a
  phase-limited tier. Handled: a run whose |k|·h is within TIE_REL of the
  ceiling never gathers off-edge;
* the EK pair mask, `group_i == group_j and >= 0`: integer labels, constant
  on a run (RUN_FIELDS), no tie;
* the EK radius floor, `max(R, b)` per quadrature point: continuous in R,
  equal on both sides of the tie, no jump;
* route splits on the block (`_ek_src_runs` by b_j, the mixed-radius row
  runs by a, `_serves_n_qp`, the SIMD lanes vs the scalar walk): each pair's
  arithmetic is its own in every route ("operation for operation" in the
  C++), so they move nothing but bookkeeping;
* same-edge: the reg kernel (C++ `seg_seg_reg_moments_bspline_swept_window`
  and `..._ek_window`, numpy einsum) has no geometric branch, and the static
  Toeplitz table is exact per diagonal (no translation);
* `expm1_neg_jkR` has no branch on R for real k.

The m values are remembered across windows per (A, B), capped at
`max_cached` entries in all; past the cap a block is still deduplicated
inside its window, only not remembered. `evaluated` counts the pairs handed
to a kernel (the O(n) claim's gate); `toeplitz_blocks` counts the blocks
served by the gather.
"""

import numpy as np

from ._bspline_kernels import (
    _LADDER_PHASE_KL_CEILING,
    _LADDER_PHASE_LIMITED_BELOW,
    _seg_seg_full_moments_offedge,
    _static_is_toeplitz,
    _seg_seg_reg_geometry,
    _seg_seg_reg_geometry_pairs,
    _seg_seg_reg_moments_from_geometry,
    _seg_seg_reg_nodes,
    _seg_seg_static_moments,
)

# The per-segment fields a run must hold constant. Tests drop one to prove
# the collision gate sees it (a red control); nothing else should touch it.
RUN_FIELDS = ("edge", "h", "sgn", "a", "group_i", "group_j", "b_i", "b_j")

# A block is gathered only when it saves a kernel call's worth of pairs: an
# (A ∩ window) x B block with fewer pairs than this is recomputed directly.
# Tests set 0 to push small decks through the gather.
TOEPLITZ_MIN_PAIRS = 4096

# The run rule's tolerance on positions, in units of eps·X (X the group's
# largest |coordinate|): "the drift bound" in the module docstring.
RUN_TOL_EPS = 16

# A diagonal (or a run) whose tier decision sits within this RELATIVE distance
# of a threshold is evaluated pair by pair, never gathered: on it the kernel's
# own `>=` is decided by coordinate roundoff and may differ pair to pair
# (momwire#1421 review: collinear equal-length edges put every off-edge ratio
# on an integer, so the free-space ladder's 16 is met exactly). Roundoff moves
# a ratio by ~|x|·eps/h, under 1e-12 on any mesh here; the margin is 1000x
# that. Tests set 0 to prove the gate sees the tie (a red control).
TIE_REL = 1e-9


def _per_segment(field, S, n):
    """A per-segment label array (or None / a scalar) restricted to S."""
    if field is None or np.ndim(field) == 0:
        return np.full(n, np.nan if field is None else float(field))
    return np.asarray(field, dtype=np.float64)[S]


class BaseRows:
    """Observer-row windows of one coaxial group's base moments:
    ``rows(r0, r1)`` is ``_exact_kernel_base_J(..., rows=slice(r0, r1))``
    to roundoff, with a uniform run's kernel work O(n) over all windows.

    The constructor takes what `_exact_kernel_base_J` reads: the mesh
    arrays, the group's local order `S` and its edges `g_edges` [(slice,
    arc, a_w)], the fill's (ek, ladder), the same-edge EK spec `ek_se`, and
    the quadrature orders."""

    def __init__(
        self,
        *,
        seg_l,
        seg_r,
        seg_a,
        tangents,
        S,
        g_edges,
        k,
        d,
        n_qp_pair,
        n_qp_same_edge,
        ek,
        ladder,
        ek_se,
        n_segs_total,
        max_cached=None,
    ):
        self.seg_l, self.seg_r, self.seg_a = seg_l, seg_r, seg_a
        self.S = np.asarray(S)
        self.n = n = self.S.size
        self.k, self.d, self.nd = k, d, d + 1
        self.n_qp_pair, self.n_qp_same_edge = n_qp_pair, n_qp_same_edge
        self.ek, self.ladder, self.ek_se = ek, ladder, ek_se
        self.max_cached = max_cached
        self.evaluated = 0
        self.toeplitz_blocks = 0
        self.tie_pairs = 0  # pairs evaluated one by one on a tie diagonal
        self._cache = {}
        self._n_cached = 0

        # the group's edges in local order
        loc = np.full(int(n_segs_total), -1, dtype=np.int64)
        loc[self.S] = np.arange(n)
        self.edges = []
        edge_of = np.full(n, -1, dtype=np.int64)
        for e, (sl, arc, a_w) in enumerate(g_edges):
            l0 = int(loc[sl.start])
            l1 = l0 + (sl.stop - sl.start)
            self.edges.append((l0, l1, arc, float(a_w)))
            edge_of[l0:l1] = e
        self._nodes = {}
        self._toeplitz_static = {}
        self.edge_bounds = sorted(l0 for l0, *_ in self.edges)

        axis = tangents[self.S[0]]
        Ls = np.asarray(seg_l, dtype=np.float64)[self.S]
        Rs = np.asarray(seg_r, dtype=np.float64)[self.S]
        u = Rs - Ls
        h = np.linalg.norm(u, axis=1)
        X = max(float(np.max(np.abs(Ls))), float(np.max(np.abs(Rs)))) if n else 0.0
        self.rho = rho = RUN_TOL_EPS * np.finfo(np.float64).eps * X
        fields = {
            "edge": edge_of.astype(np.float64),
            "sgn": np.where(tangents[self.S] @ axis >= 0.0, 1.0, -1.0),
            "a": np.asarray(seg_a, dtype=np.float64)[self.S],
            "group_i": _per_segment(None if ek is None else ek.group_i, self.S, n),
            "group_j": _per_segment(None if ek is None else ek.group_j, self.S, n),
            "b_i": _per_segment(None if ek is None else ek.b_i, self.S, n),
            "b_j": _per_segment(None if ek is None else ek.b_j, self.S, n),
        }
        exact = [fields[f] for f in RUN_FIELDS if f != "h"]
        # "h" stands for the whole geometric rule (the segment vector, the
        # progression, the pair drift): the red control drops all of it
        self._geometric = use_h = "h" in RUN_FIELDS
        # runs: [start, stop), each with its length (the first segment's),
        # direction and edge
        change = np.zeros(n, dtype=bool)
        for f in exact:
            same = (f[1:] == f[:-1]) | (np.isnan(f[1:]) & np.isnan(f[:-1]))
            change[1:] |= ~same
        starts = [0, *np.flatnonzero(change).tolist()]
        pieces = list(zip(starts, starts[1:] + [n])) if n else []
        if use_h:
            # a first cut where the segment vector leaves the run's first by
            # more than ρ (a pitch or a direction change), then the
            # progression check, which is what bounds the drift
            pieces = [
                p
                for s0, s1 in pieces
                for c0, c1 in _same_vector(u, s0, s1, rho)
                for p in _progressions(Ls, Rs, c0, c1, rho)
            ]
        self.runs = [
            (s0, s1, float(h[s0]), float(fields["sgn"][s0]), int(edge_of[s0]))
            for s0, s1 in pieces
        ]
        # each run's step vector (a one-segment run: its own vector)
        self._step = [
            (Ls[s1 - 1] - Ls[s0]) / (s1 - 1 - s0) if s1 - s0 > 1 else u[s0]
            for s0, s1 in pieces
        ]
        self.run_of = np.empty(n, dtype=np.int64)
        for r, (s0, s1, *_rest) in enumerate(self.runs):
            self.run_of[s0:s1] = r
        # length classes: runs whose lengths agree within ρ, a superset of
        # the runs `_same_step` lets a Toeplitz block pair (a graded mesh is
        # all singletons, so its rows find no partner in O(1))
        # the phase guard (`_phase_row_mask`): a run whose |k|·h sits on the
        # ceiling may answer it differently segment to segment
        phase_limited = any(n < _LADDER_PHASE_LIMITED_BELOW for _r, n in ladder or ())
        self._phase_tie = [
            phase_limited
            and abs(abs(k) * run[2] - _LADDER_PHASE_KL_CEILING)
            <= TIE_REL * _LADDER_PHASE_KL_CEILING
            for run in self.runs
        ]
        self._thresholds = np.array([r for r, _n in ladder or ()], dtype=np.float64)
        order = sorted(range(len(self.runs)), key=lambda r: self.runs[r][2])
        self._class_of = np.empty(len(self.runs), dtype=np.int64)
        self._classes = []
        for r in order:
            if (
                self._classes
                and self.runs[r][2] - self.runs[self._classes[-1][0]][2] <= rho
            ):
                self._classes[-1].append(r)
            else:
                self._classes.append([r])
            self._class_of[r] = len(self._classes) - 1
        # partners in O(useful work): a class longest run first, so a row
        # block stops at the first partner too short to gather; same-edge
        # partners per edge
        for c in self._classes:
            c.sort(key=lambda r: -(self.runs[r][1] - self.runs[r][0]))
        self._edge_runs = [[] for _ in self.edges]
        for r, run in enumerate(self.runs):
            self._edge_runs[run[4]].append(r)

    def cache_entries_needed(self):
        """Diagonal values the cross-window cache can ever hold: n_A + n_B - 1
        per run pair of one length class that can gather at all (n_A·n_B at
        least the gather threshold). 2n - 1 on a uniform run, 0 on a graded
        mesh or one of many tiny edges, so the windows give up only what
        the cache can use (`BSplineSolver._exact_kernel_base_cache_bytes`)."""
        need = 0
        for c in self._classes:  # longest run first
            lens = [self.runs[r][1] - self.runs[r][0] for r in c]
            for A, la in zip(c, lens):
                for B, lb in zip(c, lens):
                    if not self._eligible(la, lb):
                        break
                    if self._same_step(A, B):
                        need += la + lb - 1
        return need

    def _same_step(self, A, B):
        """Whether runs A and B step alike closely enough to gather: their
        relative drift over the longest diagonal, min(n_A, n_B) - 1 steps,
        is within ρ (the module docstring's drift bound). Two runs of one
        edge share σ, so the product is 1 for the same-edge kernel's arc
        coordinates too."""
        if A == B or not self._geometric:
            return True
        a0, a1, _hA, sA, _eA = self.runs[A]
        b0, b1, _hB, sB, _eB = self.runs[B]
        steps = min(a1 - a0, b1 - b0) - 1
        if steps <= 0:
            return True
        return steps * _norm(sA * sB * self._step[B] - self._step[A]) <= self.rho

    # ------------------------------------------------------------------
    # the two kernels, on index arrays of the group's local order
    # ------------------------------------------------------------------
    def _offedge(self, rows, cols):
        S, ek = self.S, self.ek
        Sr, Sc = S[rows], S[cols]
        ek_c = None
        if ek is not None:
            ek_c = ek._replace(
                group_i=ek.group_i[Sr],
                group_j=ek.group_j[Sc],
                b_i=None if ek.b_i is None else ek.b_i[Sr],
                b_j=None if ek.b_j is None else ek.b_j[Sc],
            )
        self.evaluated += len(rows) * len(cols)
        return _seg_seg_full_moments_offedge(
            self.seg_l[Sr],
            self.seg_r[Sr],
            self.seg_l[Sc],
            self.seg_r[Sc],
            self.seg_a[Sr],
            self.k,
            self.d,
            self.n_qp_pair,
            ek=ek_c,
            ladder=self.ladder,
        )

    def _reg(self, e, rows, cols):
        """Same-edge smooth-kernel moments of edge `e`; rows and cols are
        local indices inside it."""
        l0, l1, arc, a_w = self.edges[e]
        self.evaluated += len(rows) * len(cols)
        rows, cols = np.asarray(rows), np.asarray(cols)
        if (
            cols.size == l1 - l0
            and cols[0] == l0
            and cols[-1] == l1 - 1
            and rows[-1] - rows[0] == rows.size - 1
        ):
            # a contiguous row window against the whole edge: the per-pair
            # route's own call (cheaper than the pairs path on many tiny
            # edges, and the same arithmetic)
            win = slice(int(rows[0]) - l0, int(rows[-1]) + 1 - l0)
            geo = _seg_seg_reg_geometry(
                arc,
                a_w,
                max_d=self.d,
                n_qp=self.n_qp_same_edge,
                ek=self.ek_se,
                rows=win,
            )
            return _seg_seg_reg_moments_from_geometry(geo, self.k)
        nodes = self._nodes.get(e)
        if nodes is None:
            nodes = _seg_seg_reg_nodes(arc, self.d, self.n_qp_same_edge)
            self._nodes = {e: nodes}  # one edge's O(N) tables at a time
        geo = _seg_seg_reg_geometry_pairs(
            nodes,
            a_w,
            self.n_qp_same_edge,
            np.asarray(rows) - l0,
            np.asarray(cols) - l0,
            ek=self.ek_se,
        )
        return _seg_seg_reg_moments_from_geometry(geo, self.k)

    # ------------------------------------------------------------------
    # the windows
    # ------------------------------------------------------------------
    def rows(self, r0, r1):
        """(nd, nd, r1 - r0, n): observer rows [r0, r1) of the base block."""
        out = np.zeros((self.nd, self.nd, r1 - r0, self.n), dtype=complex)
        self._accumulate(out, r0, r1, 1.0)
        return out

    def subtract_from(self, out, r0, r1):
        """out -= rows(r0, r1), in place, without the window ever existing:
        each pair's value is subtracted where it is produced."""
        self._accumulate(out, r0, r1, -1.0)

    def _accumulate(self, out, r0, r1, sign):
        self._out, self._r0, self._sign = out, r0, sign
        try:
            if len(self.edges) > 1:
                self._fill(r0, r1, "off", None)
            for e, (l0, l1, arc, a_w) in enumerate(self.edges):
                lo, hi = max(l0, r0), min(l1, r1)
                if lo >= hi:
                    continue
                A = int(self.run_of[l0])
                if self._eligible(hi - lo, l1 - l0) and self._foldable(e, A):
                    # one run on a uniform edge: the static table rides in
                    # the same diagonals as the reg moments
                    self._toeplitz(A, lo, hi, A, "reg", e, static=True)
                    continue
                win = None if (lo, hi) == (l0, l1) else slice(lo - l0, hi - l0)
                static = _seg_seg_static_moments(
                    arc, a_w, max_d=self.d, ek=self.ek_se, rows=win
                )
                if len(self._edge_runs[e]) == 1 and not self._eligible(
                    hi - lo, l1 - l0
                ):
                    # one run, too few rows to gather: the per-pair route's
                    # own block, static + reg, in one put
                    self._put(
                        lo,
                        hi,
                        l0,
                        l1,
                        static + self._reg(e, np.arange(lo, hi), np.arange(l0, l1)),
                    )
                    continue
                self._put(lo, hi, l0, l1, static)
                self._fill(lo, hi, "reg", e)
        finally:
            self._out = None

    def _put(self, a, b, c, f, blk):
        """out[rows a:b, cols c:f] += sign * blk."""
        dst = self._out[:, :, a - self._r0 : b - self._r0, c:f]
        if self._sign > 0:
            dst += blk
        else:
            dst -= blk

    def _foldable(self, e, A):
        l0, l1, arc = self.edges[e][:3]
        run = self.runs[A]
        if (run[0], run[1]) != (l0, l1):
            return False
        hit = self._toeplitz_static.get(e)
        if hit is None:
            hit = _static_is_toeplitz(arc)
            self._toeplitz_static[e] = hit
        return hit

    @staticmethod
    def _eligible(n_rows, n_cols):
        return n_rows * n_cols >= max(TOEPLITZ_MIN_PAIRS, 2)

    def _eval(self, kind, e, rows, cols):
        if kind == "reg":
            return self._reg(e, rows, cols)
        return self._offedge(rows, cols)

    def _fill(self, lo, hi, kind, e):
        """Rows [lo, hi) over the kind's columns: off-edge every column off
        the row's own edge, same-edge ("reg", onto the static block) edge
        e's own. Each pair is written once."""
        short = []
        for A in range(int(self.run_of[lo]), int(self.run_of[hi - 1]) + 1):
            a0, a1, _hA, _sA, eA = self.runs[A]
            ra0, ra1 = max(a0, lo), min(a1, hi)
            toe = []
            if kind == "reg":
                for B in self._edge_runs[eA]:
                    b0, b1 = self.runs[B][:2]
                    if (
                        self._class_of[B] == self._class_of[A]
                        and self._eligible(ra1 - ra0, b1 - b0)
                        and self._same_step(A, B)
                    ):
                        toe.append(B)
            elif not self._phase_tie[A]:
                for B in self._classes[self._class_of[A]]:
                    b0, b1, _hB, _sB, eB = self.runs[B]
                    if not self._eligible(ra1 - ra0, b1 - b0):
                        break  # longest first: the rest are shorter
                    if eB != eA and not self._phase_tie[B] and self._same_step(A, B):
                        toe.append(B)
            if not toe:
                short.append((ra0, ra1))
                continue
            for B in toe:
                self._toeplitz(A, ra0, ra1, B, kind, e)
            rest = _subtract(self._span(kind, eA), [self.runs[B][:2] for B in toe])
            if rest:
                self._direct([(ra0, ra1)], rest, kind, e)
        if not short:
            return
        short = _merge(short)
        if kind == "reg":
            self._direct(short, [self.edges[e][:2]], kind, e)
            return
        # off-edge: one call over every column, as the per-pair route makes
        # it, each row range then taking only the columns off its own edge
        # (each row's own-edge block zeroed in the call's output, so one put
        # per row range writes every pair once; the same-edge pass fills it)
        rows = np.concatenate([np.arange(a, b) for a, b in short])
        vals = self._eval(kind, e, rows, np.arange(self.n))
        i = 0
        for a, b in short:
            for x, y in _split_by_edge(a, b, self.edge_bounds):
                l0, l1 = self.edges[int(self.runs[int(self.run_of[x])][4])][:2]
                vals[:, :, i + x - a : i + y - a, l0:l1] = 0.0
            self._put(a, b, 0, self.n, vals[:, :, i : i + b - a, :])
            i += b - a

    def _span(self, kind, e):
        """The columns a row on edge e takes for this kind."""
        l0, l1 = self.edges[e][:2]
        if kind == "reg":
            return [(l0, l1)]
        return [r for r in ((0, l0), (l1, self.n)) if r[0] < r[1]]

    def _direct(self, row_ranges, col_ranges, kind, e):
        rows = np.concatenate([np.arange(a, b) for a, b in row_ranges])
        cols = np.concatenate([np.arange(a, b) for a, b in col_ranges])
        vals = self._eval(kind, e, rows, cols)
        i = 0
        for a, b in row_ranges:
            j = 0
            for c, f in col_ranges:
                self._put(a, b, c, f, vals[:, :, i : i + b - a, j : j + f - c])
                j += f - c
            i += b - a

    def _static_diagonals(self, e):
        """Edge e's static block as its 2N - 1 diagonals, t = j - i + N - 1:
        rows 0 and N - 1 of the method's own Toeplitz gather."""
        l0, l1, arc, a_w = self.edges[e]
        N = l1 - l0
        last = _seg_seg_static_moments(
            arc, a_w, max_d=self.d, ek=self.ek_se, rows=slice(N - 1, N)
        )[:, :, 0, :]
        first = _seg_seg_static_moments(
            arc, a_w, max_d=self.d, ek=self.ek_se, rows=slice(0, 1)
        )[:, :, 0, :]
        return np.concatenate([last[:, :, :-1], first], axis=2)

    def _toeplitz(self, A, ra0, ra1, B, kind, e, static=False):
        """Rows [ra0, ra1) of run A against run B, gathered from one value per
        diagonal m = σ_B·j' - σ_A·i'; with `static`, run A is its whole
        uniform edge and each diagonal also carries the static table's
        entry."""
        a0, a1, _h, sA, _eA = self.runs[A]
        b0, b1, _h, sB, _eB = self.runs[B]
        if kind == "reg":
            sA = sB = 1.0  # arc coordinates along the one edge
        sA, sB = int(sA), int(sB)
        nA, nB = a1 - a0, b1 - b0
        m_min = min(0, sB * (nB - 1)) - max(0, sA * (nA - 1))
        size = nA + nB - 1
        key = (A, B, static)
        entry = self._cache.get(key)
        if entry is None:
            F = np.empty((self.nd, self.nd, size), dtype=complex)
            have = np.zeros(size, dtype=bool)
            ties = np.zeros(0, dtype=np.int64)
            if kind == "off" and self._thresholds.size:
                ties = self._tie_diagonals(a0, b0, nA, nB, sA, sB, m_min, size)
                F[:, :, ties] = 0.0  # evaluated per pair in the gather
                have[ties] = True
            entry = (F, have, ties)
            if self.max_cached is None or self._n_cached + size <= self.max_cached:
                self._cache[key] = entry
                self._n_cached += size
        F, have, ties = entry
        # the diagonals the rows need
        ip = np.arange(ra0 - a0, ra1 - a0)
        t_ends = np.concatenate([-sA * ip, sB * (nB - 1) - sA * ip]) - m_min
        tlo, thi = int(t_ends.min()), int(t_ends.max())
        need = tlo + np.flatnonzero(~have[tlo : thi + 1])
        if need.size:
            for line, at, ok, other in _boundary_pairs(need + m_min, nA, nB, sA, sB):
                if line == "row":
                    vals = self._eval(kind, e, np.array([a0 + at]), b0 + other)
                    F[:, :, need[ok]] = vals[:, :, 0, :]
                else:
                    vals = self._eval(kind, e, a0 + other, np.array([b0 + at]))
                    F[:, :, need[ok]] = vals[:, :, :, 0]
            if static:
                # same indexing: m = j - i, m_min = -(N - 1)
                F[:, :, need] += self._static_diagonals(e)[:, :, need]
            have[need] = True
        self.toeplitz_blocks += 1
        for i in range(ra0, ra1):
            t0 = -sA * (i - a0) - m_min
            if sB > 0:
                src = F[:, :, t0 : t0 + nB]
            else:
                src = F[:, :, t0 - nB + 1 : t0 + 1][:, :, ::-1]
            self._put(i, i + 1, b0, b1, src[:, :, None, :])
        if ties.size:
            # the tie diagonals hold zeros above; each pair on them is its own
            m_tie = ties + m_min
            for i in range(ra0, ra1):
                jp = sB * (m_tie + sA * (i - a0))
                jp = jp[(jp >= 0) & (jp < nB)]
                if jp.size:
                    self.tie_pairs += int(jp.size)
                    vals = self._eval(kind, e, np.array([i]), b0 + jp)
                    for c, j in enumerate(jp.tolist()):
                        self._put(
                            i, i + 1, b0 + j, b0 + j + 1, vals[:, :, :, c : c + 1]
                        )

    def _tie_diagonals(self, a0, b0, nA, nB, sA, sB, m_min, size):
        """The diagonals of an off-edge (A, B) block whose ladder ratio --
        centre distance over the longer length, the kernels' own selector
        (`_pair_ratio`; `ratio >= ladder.ratio[t]` in _accel_bspline.cpp) --
        sits within TIE_REL of a threshold, read off each diagonal's
        boundary pair."""
        m = np.arange(size) + m_min
        ii = np.empty(size, dtype=np.int64)
        jj = np.empty(size, dtype=np.int64)
        for line, at, ok, other in _boundary_pairs(m, nA, nB, sA, sB):
            ii[ok] = at if line == "row" else other
            jj[ok] = other if line == "row" else at
        I, J = self.S[a0 + ii], self.S[b0 + jj]
        sl, sr = self.seg_l, self.seg_r
        dist = np.linalg.norm(0.5 * (sl[I] + sr[I]) - 0.5 * (sl[J] + sr[J]), axis=1)
        longer = np.maximum(
            np.linalg.norm(sr[I] - sl[I], axis=1), np.linalg.norm(sr[J] - sl[J], axis=1)
        )
        ratio = dist / longer
        thr = self._thresholds
        near = np.abs(ratio[:, None] - thr[None, :]) <= TIE_REL * thr[None, :]
        return np.flatnonzero(near.any(axis=1))


def _norm(v):
    """|v| of one 3-vector (cheaper than `np.linalg.norm` in a loop)."""
    return float(np.sqrt(v @ v))


def _same_vector(u, s0, s1, rho):
    """[s0, s1) cut wherever a segment vector `u[i]` leaves its piece's FIRST
    by more than `rho`, so a slow change cannot walk a piece. Looks ahead in
    doubling blocks: one pass over a uniform stretch, and no more than a
    block's work per piece on a graded one."""
    out, a = [], s0
    while a < s1:
        b, block = a + 1, 64
        while b < s1:
            e = min(s1, b + block)
            d = u[b:e] - u[a]
            off = np.flatnonzero(np.einsum("ij,ij->i", d, d) > rho * rho)
            if off.size:
                b += int(off[0])
                break
            b, block = e, 2 * block
        out.append((a, b))
        a = b
    return out


def _progressions(Ls, Rs, s0, s1, rho):
    """[s0, s1) cut into pieces whose left ends `Ls` and right ends `Rs` each
    follow an arithmetic progression with the piece's own step (fitted
    through its first and last left ends) to within `rho`, by halving until
    they do: one piece on a uniform edge, one-segment pieces at worst."""
    out, todo = [], [(s0, s1)]
    while todo:
        a, b = todo.pop()
        if b - a > 1:
            line = np.arange(b - a)[:, None] * ((Ls[b - 1] - Ls[a]) / (b - 1 - a))
            dev = max(
                np.max(np.sum((Ls[a:b] - Ls[a] - line) ** 2, axis=1)),
                np.max(np.sum((Rs[a:b] - Rs[a] - line) ** 2, axis=1)),
            )
            if dev > rho * rho:
                mid = (a + b) // 2
                todo += [(a, mid), (mid, b)]
                continue
        out.append((a, b))
    return sorted(out)


def _boundary_pairs(m, nA, nB, sA, sB):
    """A boundary pair on each diagonal m of an nA x nB block: row 0, row
    nA - 1, column 0, column nB - 1, the first that holds it (every diagonal
    meets one). Yields (line, at, mask into m, the other index)."""
    left = np.ones(m.size, dtype=bool)
    for line, at in (("row", 0), ("row", nA - 1), ("col", 0), ("col", nB - 1)):
        if not left.any():
            break
        if line == "row":
            other = sB * (m + sA * at)
            ok = left & (other >= 0) & (other < nB)
        else:
            other = sA * (sB * at - m)
            ok = left & (other >= 0) & (other < nA)
        if ok.any():
            yield line, at, ok, other[ok]
        left &= ~ok
    assert not left.any(), "a diagonal missed every boundary line"


def _split_by_edge(a, b, bounds):
    """[a, b) cut at the edge starts in `bounds` (sorted)."""
    lo, hi = np.searchsorted(bounds, [a, b], side="right")
    cuts = [x for x in bounds[lo:hi] if a < x < b]
    pts = [a, *cuts, b]
    return list(zip(pts[:-1], pts[1:]))


def _subtract(ranges, holes):
    """[start, stop) ranges minus the hole ranges, empty pieces dropped."""
    out = []
    holes = sorted(holes)
    for a, b in ranges:
        cur = a
        for c, d in holes:
            if d <= cur or c >= b:
                continue
            if c > cur:
                out.append((cur, c))
            cur = max(cur, d)
        if cur < b:
            out.append((cur, b))
    return out


def _merge(ranges):
    """Sorted, coalesced [start, stop) ranges."""
    out = []
    for a, b in sorted(ranges):
        if out and out[-1][1] == a:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out
