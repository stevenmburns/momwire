"""Layer 2 of the fill: the SCHEDULE (momwire#1337).

`docs/design/solver-architecture.md` §E splits a matrix fill in two. Layer 1,
the kernel algebra, decides what arithmetic runs: pair geometry, kernel
variant, mirror map, ground weighting, composition order. This module is
layer 2. It decides when buffers are touched -- which observer rows a window
holds, how many fit the byte budget, which source blocks each window meets,
which rows a restricted fill keeps, where a finished window folds, and where
the solve may be cancelled -- and it never reads a moment, a weight or a
wavenumber. The formulation hands it callbacks; it calls them.

**The unit of work is a block**: observer rows ``[i0, i1)`` against source
columns ``[j0, j1)`` at one chunk of the k axis -- the contiguous case of
§F's ``zblock(I, J)``. The three axes are the interface: `sweep` walks
observer chunks x source blocks (and `walk` a list of blocks), `k_chunks`
walks the k axis, and `prepare` / `replay` split a fill into the half that
does not depend on k and the half that does.

**The schedule moves no float.** Every walk here is the loop it replaced:
the same chunk boundaries, in the same order, with the checkpoints at the
same seams. That is what lets the §3.1 schedule-invariance gates certify a
migration onto it. It is also why the budget arithmetic is NOT unified: a
B-spline window's row count comes from bytes against `swept_mem_mb`, a razor
window's from complex elements against `_CHUNK_ELEMS`, and each trunk keeps
its own rule (`mb_rows`, `elem_rows`). A different chunk boundary is a
different reduction order in B-spline's windowed assembler (the reason its
§3.1 gate is 1e-10 and not bitwise), so one shared rule would move floats.

**Lifetimes are structural.** A block's buffers are its callback's locals
and die when it returns, before the next block is built -- the rule #338
and #347 enforced with ``del`` at every rebind, now a property of the walk.
A callback that releases a buffer EARLY (before an assembler call, so the
call's own transient does not ride on top of it) still says so.

**Cancellation (#1342)**: `chunks`, `sweep`, `walk`, `k_chunks` and
`replay` call the solver's ``checkpoint`` once per chunk, block, k-chunk or
replayed window -- exactly where the per-solver loops called it.

Sinusoidal is on it (phase 2 of momwire#1337): its single-medium fill's
observer bands are `mb_rows` + `chunks`, a mixed deck's class bands are
`chunks` over `runs` of the class (the fused Z bands are `chunks` inside
each, with no checkpoint), and the Sommerfeld remainder's observer replay
is `elem_rows` + `chunks`. Its `row_group` alignment (round DOWN to whole
groups, raise on a partial one) and the dense-threshold one-chunk override
stay in the trunk's step rule; that remainder replay walks observers over a
per-fill source dict, not `prepare` / `replay`'s window list.

SG is on it too (phase 2 of momwire#1337): its default mixed-potential
fill's observer windows (`_sinusoidal_mp.WindowFill.accumulate`) are
`chunks`, as are the direct fill's test-segment bands (`_segment_bands`,
which both banded fills walk) and the extended-kernel bracket's row bands.
`_segment_bands` rounds a band UP to whole numpy-fill blocks (its
``align=``) in its own step rule -- the opposite direction to sinusoidal's
`row_group`, which is why neither rounding is an argument of `chunks` --
and its consumers keep their checkpoint, which skips an empty band.

Not yet on it (phase 2 of momwire#1337): hmatrix does NOT fit `sweep`: its
blocks are a cluster partition over basis index sets, chosen by
admissibility, and its far blocks are filled by ACA one row or column at a
time on demand. It can take `walk` for its block loops; the partition and
ACA stay its own schedule, and `zblock(I, J)` stays the general block this
module's contiguous block is a case of.

`STATS` counts what the walks did, so a gate can prove a fill came through
here rather than through a path that no longer exists (the green-gate trap).
It is a process-global tally for tests and probes; nothing reads it.
"""

import numpy as np

# What the walks did, process-wide. Tests and probes reset and read it.
STATS = {
    "sweeps": 0,
    "chunks": 0,
    "blocks": 0,
    "walked": 0,
    "k_chunks": 0,
    "prepares": 0,
    "replays": 0,
    "folds": 0,
}


def reset_stats():
    for key in STATS:
        STATS[key] = 0


# ----------------------------------------------------------------------
# budgets
# ----------------------------------------------------------------------


def mb_rows(budget_mb, row_bytes):
    """Observer rows per chunk: `row_bytes` per row against a budget in MB
    (B-spline's rule, `swept_mem_mb`). At least one row, so a budget
    smaller than one row still makes progress."""
    return max(1, int(budget_mb * 1024 * 1024 // row_bytes))


def elem_rows(budget_elems, row_elems):
    """Observer rows per chunk: `row_elems` complex elements per row against
    an element budget (razor's rule, `_CHUNK_ELEMS`)."""
    return max(1, budget_elems // max(1, row_elems))


def fits_mb(nbytes, budget_mb):
    """Whether one `nbytes` allocation fits a budget in MB."""
    return nbytes <= (budget_mb << 20)


def k_chunk(n_k, budget_mb, bytes_per_k, reserve_bytes=0):
    """Wavenumbers per k-chunk: `bytes_per_k` per point against the budget
    less `reserve_bytes` held for the whole chunk's build (#330's one-edge
    R table). At least one point, at most the sweep."""
    budget = max((budget_mb << 20) - reserve_bytes, 0)
    return max(1, min(n_k, budget // max(bytes_per_k, 1)))


# ----------------------------------------------------------------------
# windows
# ----------------------------------------------------------------------


def runs(idx):
    """`[(start, stop), ...]` -- the maximal runs of consecutive integers in
    a sorted index array (a subset that is a union of whole wires is a few
    of these)."""
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) != 1) + 1
    starts = np.concatenate(([0], cuts))
    stops = np.concatenate((cuts, [idx.size]))
    return [(int(idx[s0]), int(idx[s1 - 1]) + 1) for s0, s1 in zip(starts, stops)]


def chunks(spans, chunk, checkpoint=None):
    """Yield ``(i0, i1)``: each ``(r0, r1)`` of `spans` cut into `chunk`-row
    chunks, in order, with `checkpoint` called at the top of each."""
    for r0, r1 in spans:
        for i0 in range(r0, r1, chunk):
            if checkpoint is not None:
                checkpoint()
            STATS["chunks"] += 1
            yield i0, min(i0 + chunk, r1)


def sweep(spans, chunk, block, *, sources, checkpoint=None, restrict=None):
    """Call ``block(i0, i1, j0, j1)`` for every block of a fill.

    The observer axis is `spans` cut by `chunks` (one checkpoint per chunk);
    under `restrict` (an `ObserverRows`) each chunk keeps only its requested
    sub-windows, at the UNRESTRICTED chunk's boundaries, which is what makes
    a restricted row the dense row bit for bit; and each window meets every
    source span of `sources`, in order. Returns the number of blocks."""
    STATS["sweeps"] += 1
    n = 0
    for c0, c1 in chunks(spans, chunk, checkpoint):
        wins = [(c0, c1)] if restrict is None else restrict.windows(c0, c1)
        for i0, i1 in wins:
            for j0, j1 in sources:
                block(i0, i1, j0, j1)
                n += 1
    STATS["blocks"] += n
    return n


def walk(blocks, fill, *, checkpoint=None, keep=None):
    """Call ``fill(*block)`` for each entry of `blocks` that `keep` (if
    given) accepts, with a checkpoint before each: a fill whose blocks are a
    list rather than a grid -- the per-edge correction passes (same-edge,
    near-image) that follow a sweep. `keep` may raise: a restriction that
    covers part of an edge refuses rather than skips."""
    for block in blocks:
        if keep is not None and not keep(block):
            continue
        if checkpoint is not None:
            checkpoint()
        STATS["walked"] += 1
        fill(*block)


def k_chunks(n_k, chunk, checkpoint=None):
    """Yield ``(c0, c1)`` over a sweep's k axis, a checkpoint at the top of
    each k-chunk."""
    for c0 in range(0, n_k, chunk):
        if checkpoint is not None:
            checkpoint()
        STATS["k_chunks"] += 1
        yield c0, min(c0 + chunk, n_k)


# ----------------------------------------------------------------------
# observer-row restriction and fold destinations
# ----------------------------------------------------------------------


class ObserverRows:
    """A fill restricted to the observer rows of a set of whole wires
    (momwire#1131, the twin of the buried fill's `rows=` / `compact=`).

    `seg_rows` is a sorted set of global segment indices made of whole
    wires; `basis_rows` (R) is the basis rows whose whole live support lies
    inside it, which only the formulation can name (`basis_rows_of` maps
    `seg_rows` to R). A writer asks this object three things, and nothing
    else about the restriction:

    * `windows(i0, i1)` -- the parts of an observer chunk ``[i0, i1)`` that
      hold requested segments, as contiguous sub-windows. The walk keeps the
      DENSE fill's chunk boundaries and only drops the unrequested segments
      inside each chunk. A basis's wings in one chunk all lie in one run of
      whole wires, so they fall in one sub-window, and each ``(m, n)``
      entry receives the same addend, in the same order, as the dense chunk
      gave it -- which is why a requested row is the dense chunked fill's
      row bit for bit, not merely to roundoff.
    * `held(m_idx)` -- `m_idx` less the rows the fill does not write. A
      basis outside R can touch a requested window only through a PADDED
      support slot (a zero-padded support names segment 0), which adds an
      exact zero to a row nobody reads.
    * `covers(sl)` -- whether an edge's correction block is requested. All
      or nothing: an edge is part of one wire.

    `row_of` is None on the square target (Z stays ``(n, n)`` with every
    unrequested row exactly zero) and the compact map (basis row m held at
    row ``row_of[m]``, -1 elsewhere) on the row-compact one. `loading_map`
    is the compact map, or the identity on R, so a writer that takes a row
    map leaves the square target's unrequested rows zero too.
    """

    __slots__ = ("basis_rows", "loading_map", "row_of", "runs", "seg_mask", "_held")

    def __init__(self, seg_rows, n_segs, n_basis, basis_rows_of, *, compact):
        seg_rows = np.asarray(seg_rows, dtype=np.int64)
        if seg_rows.ndim != 1 or np.any(np.diff(seg_rows) <= 0):
            raise ValueError("rows= must be a sorted 1-D array of distinct segments")
        if seg_rows.size and (seg_rows[0] < 0 or seg_rows[-1] >= n_segs):
            raise ValueError(f"rows= holds a segment outside [0, {n_segs})")
        n_basis = int(n_basis)
        self.basis_rows = basis_rows_of(seg_rows)
        self.seg_mask = np.zeros(int(n_segs), dtype=bool)
        self.seg_mask[seg_rows] = True
        self.runs = runs(seg_rows)
        self._held = np.zeros(n_basis, dtype=bool)
        self._held[self.basis_rows] = True
        loading_map = np.full(n_basis, -1, dtype=np.int64)
        if compact:
            loading_map[self.basis_rows] = np.arange(
                self.basis_rows.size, dtype=np.int64
            )
            self.row_of = loading_map
        else:
            loading_map[self.basis_rows] = self.basis_rows
            self.row_of = None
        self.loading_map = loading_map

    def new_Z(self, n_basis):
        if self.row_of is None:
            # Column-major, as an unrestricted square target is.
            return np.zeros((n_basis, n_basis), dtype=np.complex128, order="F")
        # C order: nothing factors a row-compact Z (momwire#1132's reason).
        return np.zeros((self.basis_rows.size, n_basis), dtype=np.complex128)

    def windows(self, i0, i1):
        out = []
        for r0, r1 in self.runs:
            a0, a1 = max(r0, i0), min(r1, i1)
            if a0 < a1:
                out.append((a0, a1))
        return out

    def held(self, m_idx):
        return m_idx[self._held[m_idx]]

    def covers(self, sl):
        inside = self.seg_mask[sl]
        if inside.all():
            return True
        if inside.any():
            raise ValueError(
                f"rows= covers part of the edge [{sl.start}, {sl.stop}); the "
                f"above-ground fill restricts by whole wires (momwire#1131)"
            )
        return False

    def kwargs(self):
        """The writers' extra argument: only the compact target passes one,
        so the square target's calls are the unrestricted ones."""
        return {} if self.row_of is None else {"row_of": self.row_of}


def new_target(n_basis, restrict=None):
    """``(Z, row_kw)``: the zeroed destination a fill accumulates into and
    the extra argument its writers take. Unrestricted, Z is square and
    Fortran order -- an in-place LU (`_lu_solve(overwrite_a=True)`) factors
    only a column-major matrix and would silently copy a C-order one
    (issue #136)."""
    if restrict is None:
        return np.zeros((n_basis, n_basis), dtype=np.complex128, order="F"), {}
    return restrict.new_Z(n_basis), restrict.kwargs()


def row_window(into, lo, hi, n_cols):
    """Rows ``[lo, hi)`` of `into` to write a window in, or a fresh
    complex128 window when there is no whole destination."""
    if into is not None:
        return into[lo:hi]
    return np.empty((hi - lo, n_cols), dtype=np.complex128)


def fold_rows(out, windows):
    """``out[lo:hi] -= rows`` for each ``(lo, hi, rows)`` of `windows`, so a
    block folds into its destination a window at a time and never exists
    whole beside it. Elementwise, so it is the same matrix as
    ``out -= block``. Returns `out`."""
    for lo, hi, rows in windows:
        STATS["folds"] += 1
        out[lo:hi] -= rows
    return out


# ----------------------------------------------------------------------
# prepare / replay
# ----------------------------------------------------------------------


class Prepared:
    """The k-independent half of a row-windowed fill: ``[(lo, hi, *payload)]``
    built once by `prepare`, replayed at every wavenumber by `replay`.

    Iterable, because a window list is what every consumer wants. A
    formulation that must bind the windows to what built them (razor binds
    the source set, momwire#745) subclasses it and adds the binding."""

    __slots__ = ("windows",)

    def __init__(self, windows):
        self.windows = windows

    def __iter__(self):
        return iter(self.windows)


def prepare(n_rows, step, build, *, checkpoint=None):
    """``[(lo, hi, *build(lo, hi))]`` over ``[0, n_rows)`` in `step`-row
    windows: a fill's k-independent half, built once. `build` returns the
    window's payload as a tuple."""
    STATS["prepares"] += 1
    return [
        (lo, hi, *build(lo, hi)) for lo, hi in chunks([(0, n_rows)], step, checkpoint)
    ]


def replay(windows, checkpoint=None):
    """Yield each prepared window in order, a checkpoint before each: the
    k-dependent half's walk over what `prepare` built."""
    for window in windows:
        if checkpoint is not None:
            checkpoint()
        STATS["replays"] += 1
        yield window
