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
radius `a_w` and same-edge kernel), the same length within `q` of the run's
first, the same σ, `a_i`, and EK `group_i`, `group_j`, `b_i`, `b_j`
(`RUN_FIELDS`). Inside a run the segments are contiguous, so for runs A and B
of equal length h a pair's axial offset is a function of ONE integer,
m = σ_B·j' - σ_A·i' (i', j' the positions in the runs), and so is its whole
moment: the (A, B) block is Toeplitz (Hankel for opposite σ). Each needed m
is evaluated once, from a pair on the block's boundary (every diagonal meets
it), by the SAME kernel call on the SAME segments' real coordinates, and the
block's rows are gathered from those values. Pairs on one diagonal differ
only by a translation along the axis, so the substitution moves a moment by
the coordinates' roundoff (and a length by at most `q`, the exact dedup's
own key resolution), never by a different rule. Pairs not in such a block
-- unequal lengths, short runs, a graded mesh -- are recomputed exactly as
before, in one call per run.

The m values are remembered across windows per (A, B), capped at
`max_cached` entries in all; past the cap a block is still deduplicated
inside its window, only not remembered. `evaluated` counts the pairs handed
to a kernel (the O(n) claim's gate); `toeplitz_blocks` counts the blocks
served by the gather.
"""

import numpy as np

from ._bspline_kernels import (
    _seg_seg_full_moments_offedge,
    _static_is_toeplitz,
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
        h = np.linalg.norm(seg_r[self.S] - seg_l[self.S], axis=1)
        self.q = 1e-11 * float(np.max(h)) if n else 1.0
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
        use_h = "h" in RUN_FIELDS
        # runs: [start, stop), each with its length (the first segment's),
        # direction and edge
        change = np.zeros(n, dtype=bool)
        for f in exact:
            same = (f[1:] == f[:-1]) | (np.isnan(f[1:]) & np.isnan(f[:-1]))
            change[1:] |= ~same
        starts = [0]
        for i in range(1, n):
            # a length is held to the RUN's first, so a slow drift cannot
            # walk a run past q
            if change[i] or (use_h and abs(h[i] - h[starts[-1]]) > self.q):
                starts.append(i)
        stops = starts[1:] + [n]
        self.runs = [
            (s0, s1, float(h[s0]), float(fields["sgn"][s0]), int(edge_of[s0]))
            for s0, s1 in zip(starts, stops)
        ]
        self.run_of = np.empty(n, dtype=np.int64)
        for r, (s0, s1, *_rest) in enumerate(self.runs):
            self.run_of[s0:s1] = r
        # length classes: runs whose lengths agree within q, the only runs a
        # Toeplitz block can pair (a graded mesh is all singletons, so its
        # rows find no partner in O(1))
        order = sorted(range(len(self.runs)), key=lambda r: self.runs[r][2])
        self._class_of = np.empty(len(self.runs), dtype=np.int64)
        self._classes = []
        for r in order:
            if (
                self._classes
                and self.runs[r][2] - self.runs[self._classes[-1][0]][2] <= self.q
            ):
                self._classes[-1].append(r)
            else:
                self._classes.append([r])
            self._class_of[r] = len(self._classes) - 1

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
        l0, _l1, arc, a_w = self.edges[e]
        nodes = self._nodes.get(e)
        if nodes is None:
            nodes = _seg_seg_reg_nodes(arc, self.d, self.n_qp_same_edge)
            self._nodes = {e: nodes}  # one edge's O(N) tables at a time
        self.evaluated += len(rows) * len(cols)
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
                if self._foldable(e, A) and self._eligible(hi - lo, l1 - l0):
                    # one run on a uniform edge: the static table rides in
                    # the same diagonals as the reg moments
                    self._toeplitz(A, lo, hi, A, "reg", e, static=True)
                    continue
                win = None if (lo, hi) == (l0, l1) else slice(lo - l0, hi - l0)
                self._put(
                    lo,
                    hi,
                    l0,
                    l1,
                    _seg_seg_static_moments(
                        arc, a_w, max_d=self.d, ek=self.ek_se, rows=win
                    ),
                )
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
            for B in self._classes[self._class_of[A]]:
                b0, b1, _hB, _sB, eB = self.runs[B]
                if (eB == eA) != (kind == "reg"):
                    continue
                if self._eligible(ra1 - ra0, b1 - b0):
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
        rows = np.concatenate([np.arange(a, b) for a, b in short])
        vals = self._eval(kind, e, rows, np.arange(self.n))
        i = 0
        for a, b in short:
            for x, y in _split_by_edge(a, b, self.edge_bounds):
                eX = int(self.runs[int(self.run_of[x])][4])
                for c, f in self._span(kind, eX):
                    self._put(x, y, c, f, vals[:, :, i + x - a : i + y - a, c:f])
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
            entry = (
                np.empty((self.nd, self.nd, size), dtype=complex),
                np.zeros(size, dtype=bool),
            )
            if self.max_cached is None or self._n_cached + size <= self.max_cached:
                self._cache[key] = entry
                self._n_cached += size
        F, have = entry
        # the diagonals the rows need
        ip = np.arange(ra0 - a0, ra1 - a0)
        t_ends = np.concatenate([-sA * ip, sB * (nB - 1) - sA * ip]) - m_min
        tlo, thi = int(t_ends.min()), int(t_ends.max())
        need = tlo + np.flatnonzero(~have[tlo : thi + 1])
        if need.size:
            m = need + m_min
            # a boundary pair on each diagonal: row 0, row nA-1, column 0,
            # column nB-1, the first that holds it
            left = np.ones(m.size, dtype=bool)
            for line, at in (("row", 0), ("row", nA - 1), ("col", 0), ("col", nB - 1)):
                if not left.any():
                    break
                if line == "row":
                    other = sB * (m + sA * at)
                    ok = left & (other >= 0) & (other < nB)
                    if ok.any():
                        vals = self._eval(kind, e, np.array([a0 + at]), b0 + other[ok])
                        F[:, :, need[ok]] = vals[:, :, 0, :]
                else:
                    other = sA * (sB * at - m)
                    ok = left & (other >= 0) & (other < nA)
                    if ok.any():
                        vals = self._eval(kind, e, a0 + other[ok], np.array([b0 + at]))
                        F[:, :, need[ok]] = vals[:, :, :, 0]
                left &= ~ok
            assert not left.any(), "a diagonal missed every boundary line"
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


def _split_by_edge(a, b, bounds):
    """[a, b) cut at the edge starts in `bounds` (sorted)."""
    cuts = [x for x in bounds if a < x < b]
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
