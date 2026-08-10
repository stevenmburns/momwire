"""Element-aware block-low-rank solver for antenna arrays.

`ArrayBlockSolver` is a structural accelerator for the B-spline MoM, a sibling
of `HMatrixSolver`. Where the generic H-matrix partitions the impedance matrix
with a geometry-blind binary space-partition cluster tree, this solver uses the
*structural* partition an array hands it for free: a `P`-element array is a
`P x P` grid of blocks — strong dense self-blocks on the diagonal, weak
low-rank coupling blocks off it.

See `docs/array_block_solver_plan.md` for the design and the validated
measurements behind it. This file is being built up phase by phase:

  * P0: element grouping (the key enabler) + the assumption verification
    harness. `element_groups` maps each basis to its array element via
    connected components of the wire graph and labels elements by geometric
    shape class; `ArrayPartition` bundles the grouping. The verification
    routines confirmed the design's load-bearing measurements (self-blocks
    identical within a shape class, weak + low-rank coupling).

  * P1: `ArrayBlockSolver.build_array_blocks()` assembles the impedance matrix
    as an `ArrayBlock` — one dense self-block per distinct shape class (reused
    across same-shape elements; the P0-verified ~2e-12 identity) plus an
    ACA-compressed low-rank coupling block per element pair, with the
    complex-symmetry `Z_ba = Z_ab^T` halving the ACA work. The container has a
    fast `matvec` that reproduces the dense `Z @ x`.

  * P2: the constrained solve. `ArrayBlock` exposes its dense self-blocks as
    `.near`, so `ArrayBlockSolver` runs the inherited `_solve_hmatrix`
    augmented-GMRES verbatim — the block-diagonal self-blocks become the
    block-Jacobi preconditioner and the KCL junction constraints go in the
    saddle rows. Because coupling is ~1e-4 of the self-blocks, GMRES converges
    in a handful of iterations (5 on `invveearray`, 9 on `bowtiearray2x4`);
    impedance/Y match dense `BSplineSolver` to ~1e-5.

  * P3: identical-element + block-Toeplitz reuse. Self-blocks are one-per-shape.
    Coupling blocks are deduplicated by `(shape_a, shape_b, displacement)`:
    free-space translation invariance makes all pairs with that key the same
    block, so ACA runs once per unique key (+complex-symmetry transpose),
    collapsing the `P(P-1)` pairs to a handful of displacements on a regular
    grid — 56→13 on `bowtiearray2x4`, 12→5 on `invveearray`, 12→3 on a uniform
    4-element line.

  * P4 (this commit): the animation factor-cache. Two module-level caches let
    an animation sweep reuse the expensive work across frames:
      - operator cache (keyed by full geometry + k + tol): a *phase/excitation*
        sweep holds geometry fixed, so Z and its factored preconditioner are
        reused wholesale — each frame is just new-RHS back-substitutions.
      - self-block cache (keyed by an element's translation-invariant geometry
        signature + k + radius): a *spacing* sweep keeps identical elements, so
        the dense self-block assembly is reused across frames and only the
        cheap coupling blocks recompute.
    The factorisation itself is cached on the operator (see
    `HMatrixSolver._factored_solve`), so a reused operator never refactors.

  * Solve internals (post-profiling): the constrained solve is a
    left-preconditioned *block* GMRES over all RHS at once (batched `matmat` +
    batched preconditioner apply, BLAS-3), and the preconditioner is a
    per-element block-Jacobi (`_BlockJacobiAugPrecond`): because elements are
    electrically separate the augmented saddle is block-diagonal, so it factors
    once per shape with a dense LU and applies as one wide `lu_solve` per shape.
    Both are exact reformulations — convergence and accuracy are unchanged — and
    cut the dominant repeated-apply costs the profiler flagged (~2× on the
    preconditioner apply for `bowtiearray2x4`).

`ArrayBlockSolver` subclasses `HMatrixSolver`, reusing its `_context`, `zblock`,
the C++ off-edge block assembler, and `aca_partial` verbatim; the grouping
reuses BSplineSolver's geometry/basis build. Nothing here touches the kernel.
"""

import itertools

import numpy as np
from scipy.fft import fftn, ifftn, next_fast_len
from scipy.linalg import lu_factor, lu_solve

from ._aca import aca_partial
from ._cancel import _Cancelable
from .hmatrix import (
    _HAVE_OFFEDGE_BLOCK_ACCEL,
    _OFFEDGE_BLOCK_ACCEL_MAX_D,
    HMatrixSolver,
)


# ----------------------------------------------------------------------
# Animation factor-cache (P4): module-level reuse across solves/frames
# ----------------------------------------------------------------------

# Whole-operator cache (keyed by geometry + k + tol): a phase/excitation sweep
# holds geometry fixed, so the assembled ArrayBlock and its factored
# preconditioner are reused wholesale and each frame only re-solves the RHS.
_ARRAY_OP_CACHE: dict = {}
_ARRAY_OP_CACHE_MAX = 8

# Self-block cache (keyed by an element's translation-invariant geometry
# signature + k + radius + basis): a spacing sweep keeps identical elements, so
# the dense self-block assembly is reused while only coupling recomputes.
_SELF_BLOCK_CACHE: dict = {}
_SELF_BLOCK_CACHE_MAX = 64

# Instrumentation so tests (and the demo script) can prove reuse happened.
_CACHE_STATS = {
    "operator_build": 0,
    "operator_hit": 0,
    "self_block_build": 0,
    "self_block_hit": 0,
    "hmatrix_fallback": 0,
}


def reset_array_caches():
    """Clear the animation caches and zero the hit/build counters."""
    _ARRAY_OP_CACHE.clear()
    _SELF_BLOCK_CACHE.clear()
    for key in _CACHE_STATS:
        _CACHE_STATS[key] = 0


def cache_stats():
    """A copy of the cache hit/build counters."""
    return dict(_CACHE_STATS)


def _cache_put(cache, key, value, max_size):
    """Insert with a crude FIFO bound so long sweeps don't grow unbounded."""
    if len(cache) >= max_size:
        cache.pop(next(iter(cache)))
    cache[key] = value


# ----------------------------------------------------------------------
# Element grouping (P0 — the key enabler)
# ----------------------------------------------------------------------


def _basis_to_wire(supp_seg, wire_basis_global):
    """Map each global basis index to the polyline (wire) that owns it.

    Bases are emitted wire-by-wire in `_build_basis_polynomials` (the global
    index increments through each wire's kept bases in order), so every wire
    owns a contiguous range of basis indices. Returns an (n_basis,) int array.
    """
    n_basis = supp_seg.shape[0]
    b2w = np.empty(n_basis, dtype=np.int64)
    m = 0
    for w_idx, (kept, _local_to_global) in enumerate(wire_basis_global):
        nb = len(kept)
        b2w[m : m + nb] = w_idx
        m += nb
    assert m == n_basis, f"basis count {m} != n_basis {n_basis}"
    return b2w


def _wire_to_element(wires_polylines, tol=1e-6):
    """Group polylines into electrically connected elements by shared anchors.

    Two polylines belong to the same element when any of their anchor points
    coincide (within `tol`): wires inside one array element meet end-to-end at
    junction nodes, while distinct elements are spatially separated. Pure
    geometry — no junction list or builder metadata needed.

    Returns (wire_elem, n_elem) where `wire_elem[w]` is the element id of wire
    `w`, with element ids assigned in order of first appearance (so wire 0 is
    always in element 0).
    """
    n_w = len(wires_polylines)
    parent = list(range(n_w))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # Map each rounded anchor coordinate to the wires that touch it, unioning
    # every wire that shares a node. Rounding to a `tol` grid merges anchors
    # that are coincident to ~1e-12 (consecutive segments / junctions) while
    # keeping distinct elements (separated by >> tol) apart.
    coord_to_wire = {}
    for w, pl in enumerate(wires_polylines):
        for pt in np.asarray(pl, dtype=float):
            key = tuple(np.round(pt / tol).astype(np.int64))
            prev = coord_to_wire.get(key)
            if prev is None:
                coord_to_wire[key] = w
            else:
                union(prev, w)

    # Relabel component roots to dense element ids in order of first appearance.
    roots = [find(w) for w in range(n_w)]
    label_of_root = {}
    wire_elem = np.empty(n_w, dtype=np.int64)
    for w, r in enumerate(roots):
        if r not in label_of_root:
            label_of_root[r] = len(label_of_root)
        wire_elem[w] = label_of_root[r]
    return wire_elem, len(label_of_root)


def _element_segment_groups(geom, wire_elem, n_elem):
    """Segments owned by each element, as a sorted index array per element.

    Segments are laid out contiguously per wire (`geom["seg_offsets"]`), and an
    element is a set of whole wires, so each element owns a clean union of
    per-wire segment ranges — no basis-support padding to filter out.
    """
    seg_off = geom["seg_offsets"]
    seg_groups = [[] for _ in range(n_elem)]
    for w, e in enumerate(wire_elem):
        seg_groups[e].append(np.arange(seg_off[w], seg_off[w + 1], dtype=np.int64))
    return [np.concatenate(s) if s else np.zeros(0, dtype=np.int64) for s in seg_groups]


class ArrayPartition:
    """The structural partition of an array's impedance matrix into elements.

    Attributes
    ----------
    n_basis : int
        Total number of bases.
    n_elem : int
        Number of array elements found.
    elem_of_basis : (n_basis,) int
        Element id of each basis.
    groups : list[np.ndarray]
        `groups[e]` is the sorted array of basis indices in element `e`.
        Within an element bases are sorted ascending; because the array builder
        emits each element from the same element generator with identical
        segmentation, that ascending order corresponds segment-for-segment
        across elements of the same shape (the property the identical-
        self-block reuse relies on).
    sizes : (n_elem,) int
        `len(groups[e])`.
    elem_of_wire : (n_wires,) int
        Element id of each polyline.
    seg_groups : list[np.ndarray]
        `seg_groups[e]` is the sorted array of segment indices in element `e`.
    shape_of_elem : (n_elem,) int
        Shape-class id of each element: elements that are translates of one
        another (identical geometry up to a rigid shift) share a class.
        Generally up to 4 classes for a 2x4 array (itop/ibot/otop/obot), up to
        2 for a 2x2 (top/bot), 1 for a 1x2; fewer when params coincide (e.g.
        the default 2x4 has inner==outer, collapsing 4 nominal shapes to 2).
    """

    def __init__(
        self, n_basis, elem_of_basis, groups, elem_of_wire, seg_groups, shape_of_elem
    ):
        self.n_basis = n_basis
        self.n_elem = len(groups)
        self.elem_of_basis = elem_of_basis
        self.groups = groups
        self.sizes = np.array([g.size for g in groups], dtype=np.int64)
        self.elem_of_wire = elem_of_wire
        self.seg_groups = seg_groups
        self.shape_of_elem = shape_of_elem
        self.n_shapes = int(shape_of_elem.max()) + 1 if len(shape_of_elem) else 0

    def shape_representatives(self):
        """First element id of each shape class, in class-id order."""
        reps = {}
        for e, s in enumerate(self.shape_of_elem):
            if s not in reps:
                reps[int(s)] = e
        return [reps[s] for s in range(self.n_shapes)]

    def __repr__(self):
        return (
            f"ArrayPartition(n_basis={self.n_basis}, n_elem={self.n_elem}, "
            f"n_shapes={self.n_shapes}, sizes={self.sizes.tolist()})"
        )


def _shape_classes(geom, seg_groups, seg_a, tol=1e-6):
    """Cluster elements into geometric shape classes (translation-invariant).

    Each element's signature is its segment midpoints recentred on the element
    centroid, sorted lexicographically and rounded to `tol`. Two elements are
    translates of one another iff their signatures match, so this finds the
    true number of distinct shapes from geometry alone — 4 for a 2x4 with
    distinct inner/outer/top/bot params, fewer when params coincide. (These
    arrays differ only by translation, not rotation/reflection; a rotated
    element would land in its own class, which is the safe behaviour.)

    `seg_a` (the per-segment conductor radius) joins the signature in the
    same canonical order: two geometric translates whose wires carry
    different radii have different impedance blocks, so they must not share
    a shape class (stevenmburns/momwire#147).
    """
    seg_l, seg_r = geom["seg_l"], geom["seg_r"]
    sigs = []
    for segs in seg_groups:
        mids = 0.5 * (seg_l[segs] + seg_r[segs])
        mids = mids - mids.mean(axis=0)
        key = np.round(mids / tol).astype(np.int64)
        order = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
        sigs.append(
            key[order].tobytes()
            + bytes(str(key.shape), "ascii")
            + np.asarray(seg_a[segs], dtype=np.float64)[order].tobytes()
        )
    label_of_sig = {}
    shape_of_elem = np.empty(len(seg_groups), dtype=np.int64)
    for e, s in enumerate(sigs):
        if s not in label_of_sig:
            label_of_sig[s] = len(label_of_sig)
        shape_of_elem[e] = label_of_sig[s]
    return shape_of_elem


def element_groups(sim, tol=1e-6):
    """Partition `sim`'s bases into array elements.

    Composes the exact basis→wire map (contiguous, from the basis build) with
    a wire→element connected-components grouping (shared anchors), then labels
    the elements by geometric shape class. Returns an `ArrayPartition`. A
    single connected structure yields one element of all bases — correct
    degenerate behaviour, just no array structure to exploit.
    """
    geom = sim._build_geometry()
    supp_seg, _polys, _kcl_A, _wk, wire_basis_global = sim._build_basis_polynomials(
        geom
    )
    n_basis = supp_seg.shape[0]
    b2w = _basis_to_wire(supp_seg, wire_basis_global)
    wire_elem, n_elem = _wire_to_element(sim.wires_polylines, tol=tol)
    elem_of_basis = wire_elem[b2w]
    groups = [
        np.sort(np.nonzero(elem_of_basis == e)[0].astype(np.int64))
        for e in range(n_elem)
    ]
    seg_groups = _element_segment_groups(geom, wire_elem, n_elem)
    shape_of_elem = _shape_classes(geom, seg_groups, sim._seg_radius(geom), tol=tol)
    return ArrayPartition(
        n_basis, elem_of_basis, groups, wire_elem, seg_groups, shape_of_elem
    )


# ----------------------------------------------------------------------
# Block container + matvec (P1)
# ----------------------------------------------------------------------


class ArrayBlock(_Cancelable):
    """Element-block decomposition of the impedance matrix with a fast matvec.

    The `P x P` grid of element blocks exactly tiles `Z` (the element groups
    partition all bases): diagonal blocks are dense self-blocks, off-diagonal
    blocks are low-rank coupling blocks.

    groups : list[np.ndarray]
        Per-element basis indices (from `ArrayPartition`).
    shape_of_elem : (P,) int
        Shape class of each element.
    shape_blocks : dict[int, np.ndarray]
        One dense `(N_s, N_s)` self-block per distinct shape class, applied to
        every element of that shape (same-shape self-blocks are identical to
        ~2e-12 in free space — the P0-verified reuse).
    coupling : list[(a, b, U, V)]
        Low-rank factors for every ordered off-diagonal pair: block `(a, b)`
        is `U @ V`. Stored for both directions (the `(b, a)` entry reuses the
        transposed factors of `(a, b)`).
    """

    def __init__(
        self,
        n,
        groups,
        shape_of_elem,
        shape_blocks,
        coupling,
        extra_lowrank=None,
        cancel=None,
    ):
        self.n = n
        self.groups = groups
        self.shape_of_elem = shape_of_elem
        self.shape_blocks = shape_blocks
        self.coupling = coupling
        # Optional global low-rank terms [(I, J, U, V)] on top of the
        # element decomposition — the Sommerfeld smooth remainder rides
        # here (docs/sommerfeld-everywhere-plan.md Phase 3), OUTSIDE the
        # block cache, so shape/displacement dedup and the per-shape
        # factorisation reuse are untouched. Not part of `near`, so the
        # block-Jacobi preconditioner ignores it (smooth perturbation).
        self.extra_lowrank = extra_lowrank or []
        self._cancel = cancel
        # Dense self-blocks as (I, J, D) triples, so the block decomposition is
        # a drop-in for `HMatrixSolver._solve_hmatrix`: its near-field
        # preconditioner becomes the block-diagonal of Z (block-Jacobi), which
        # — because coupling is ~1e-4 of the self-blocks — drives GMRES to a
        # handful of iterations. `precond_extra` (the H-matrix's first-ring
        # strengthening) has no analogue here, so it is empty.
        self.near = [
            (g, g, shape_blocks[int(shape_of_elem[e])]) for e, g in enumerate(groups)
        ]
        self.precond_extra = []

    def matvec(self, x):
        self._checkpoint()  # one check per GMRES matvec (no-op without a token)
        x = np.asarray(x)
        y = np.zeros(self.n, dtype=np.complex128)
        for e, g in enumerate(self.groups):
            y[g] += self.shape_blocks[int(self.shape_of_elem[e])] @ x[g]
        for a, b, U, V in self.coupling:
            y[self.groups[a]] += U @ (V @ x[self.groups[b]])
        for I, J, U, V in self.extra_lowrank:
            y[I] += U @ (V @ x[J])
        return y

    def matmat(self, X):
        """Apply the operator to all columns of X (n, nrhs) at once.

        Same block decomposition as `matvec`, but every block product becomes a
        BLAS-3 matrix-matrix multiply (`(N_s, N_s) @ (N_s, nrhs)`), which is far
        more cache- and overhead-efficient than nrhs separate matvecs — the
        point of the batched multi-RHS solve."""
        self._checkpoint()
        X = np.asarray(X)
        Y = np.zeros((self.n, X.shape[1]), dtype=np.complex128)
        for e, g in enumerate(self.groups):
            Y[g] += self.shape_blocks[int(self.shape_of_elem[e])] @ X[g]
        for a, b, U, V in self.coupling:
            Y[self.groups[a]] += U @ (V @ X[self.groups[b]])
        for I, J, U, V in self.extra_lowrank:
            Y[I] += U @ (V @ X[J])
        return Y

    def storage(self):
        """Complex scalars stored (distinct self-blocks + coupling factors)."""
        s = sum(D.size for D in self.shape_blocks.values())
        s += sum(U.size + V.size for _, _, U, V in self.coupling)
        s += sum(U.size + V.size for _, _, U, V in self.extra_lowrank)
        return s

    def stats(self):
        ranks = [U.shape[1] for _, _, U, _ in self.coupling]
        return {
            "n": self.n,
            "n_elem": len(self.groups),
            "n_shapes": len(self.shape_blocks),
            "n_coupling": len(self.coupling),
            "storage": self.storage(),
            "dense_storage": self.n * self.n,
            "compression": self.storage() / (self.n * self.n),
            "max_rank": max(ranks) if ranks else 0,
            "mean_rank": float(np.mean(ranks)) if ranks else 0.0,
        }

    def to_dense(self):
        """Reconstruct the full dense matrix (validation / small n only)."""
        Z = np.zeros((self.n, self.n), dtype=np.complex128)
        for e, g in enumerate(self.groups):
            Z[np.ix_(g, g)] = self.shape_blocks[int(self.shape_of_elem[e])]
        for a, b, U, V in self.coupling:
            Z[np.ix_(self.groups[a], self.groups[b])] = U @ V
        for I, J, U, V in self.extra_lowrank:
            Z[np.ix_(I, J)] += U @ V
        return Z


def _detect_lattice(cen, tol=1e-6):
    """Fit element centroids to a regular 1-D or 2-D integer lattice.

    Basis vectors are the shortest and shortest-non-parallel centroid
    differences from element 0; every centroid must then land on an integer
    combination of them within `tol` (metres — the same absolute grid the
    displacement keys round to). Any misfit returns None and the caller keeps
    the per-pair path. The lattice may sit in any plane at any orientation and
    need not be fully occupied (an array with holes still fits; absent sites
    simply stay zero in the convolution).

    Returns (coords, basis): `coords` (P, dim) 0-based integer lattice
    coordinates, `basis` (dim, 3) lattice vectors — or None.
    """
    cen = np.asarray(cen, dtype=float)
    P = cen.shape[0]
    if P < 4:
        return None
    d = cen - cen[0]
    norms = np.linalg.norm(d, axis=1)
    nz = np.argsort(norms)
    nz = nz[norms[nz] > tol]
    if nz.size == 0:
        return None
    v1 = d[nz[0]]
    v2 = None
    n1 = np.linalg.norm(v1)
    for idx in nz[1:]:
        if np.linalg.norm(np.cross(v1, d[idx])) > 1e-9 * n1 * norms[idx]:
            v2 = d[idx]
            break
    A = np.array([v1]) if v2 is None else np.array([v1, v2])
    m = np.linalg.solve(A @ A.T, A @ d.T).T  # (P, dim) real coordinates
    mi = np.round(m)
    if np.abs(d - mi @ A).max() > tol:
        return None
    coords = mi.astype(np.int64)
    coords -= coords.min(axis=0)
    if len({tuple(c) for c in coords}) != P:
        return None
    return coords, A


class LatticeArrayBlock(ArrayBlock):
    """`ArrayBlock` specialised for a regular lattice of same-shape elements:
    the coupling is block-Toeplitz in every lattice dimension, held as one
    dense `(n_e, n_e)` block per displacement in FFT (spectral) form, and the
    coupling matvec is an FFT convolution over the element grid —
    O(N log P) per apply with no per-pair loop and no ACA truncation
    (displacement blocks are stored exactly). Self-blocks, block-Jacobi
    preconditioning, KCL saddle rows, and the Sommerfeld `extra_lowrank`
    remainder are inherited unchanged.

    Parameters beyond `ArrayBlock`: `coords` (P, dim) integer lattice
    coordinates; `fft_shape` the padded per-dimension FFT sizes (each
    ≥ 2·span−1, so the circular convolution never wraps onto an occupied
    site); `k_hat` the kernel spectrum `(*fft_shape, n_e, n_e)` with the
    zero-displacement slot empty (self-blocks stay separate);
    `n_kernel_blocks` the count of stored displacement blocks (for stats).
    """

    def __init__(
        self,
        n,
        groups,
        shape_of_elem,
        shape_blocks,
        coords,
        fft_shape,
        k_hat,
        n_kernel_blocks,
        extra_lowrank=None,
        cancel=None,
    ):
        super().__init__(
            n,
            groups,
            shape_of_elem,
            shape_blocks,
            coupling=[],
            extra_lowrank=extra_lowrank,
            cancel=cancel,
        )
        self.coords = coords
        self.fft_shape = tuple(fft_shape)
        self.k_hat = k_hat
        self.n_kernel_blocks = n_kernel_blocks
        self._axes = tuple(range(coords.shape[1]))
        self._site_idx = tuple(coords.T)
        # Deep-restart hint for the block GMRES: edge-resonance modes on a
        # resonant lattice converge slowly, and restart truncation at the
        # generic depth (50) can stall the solve outright — see
        # `_AugmentedFactoredSolve._block_gmres` and docs/lattice-fft-plan.md.
        # 600 covers the measured need through a 100×100 half-wave lattice
        # (245 iters at 64×64; >300 at 100×100, where a 300-deep restart
        # still converged but paid ~3× in cycles).
        self.gmres_restart = 600
        # (P, n_e) gather map — same-shape elements all have n_e bases and
        # the groups partition the basis range, so this is a permutation.
        self.G = np.stack(groups)

    def _apply(self, X):
        self._checkpoint()
        n_e = self.G.shape[1]
        nrhs = X.shape[1]
        Y = np.zeros((self.n, nrhs), dtype=np.complex128)
        Xg = X[self.G]  # (P, n_e, nrhs)
        for sid, S in self.shape_blocks.items():
            mask = self.shape_of_elem == sid
            Y[self.G[mask]] += np.matmul(S, Xg[mask])
        Xgrid = np.zeros(self.fft_shape + (n_e, nrhs), dtype=np.complex128)
        Xgrid[self._site_idx] = Xg
        Yhat = np.matmul(self.k_hat, fftn(Xgrid, axes=self._axes, workers=-1))
        Ygrid = ifftn(Yhat, axes=self._axes, workers=-1)
        Y[self.G] += Ygrid[self._site_idx]
        for I, J, U, V in self.extra_lowrank:
            Y[I] += U @ (V @ X[J])
        return Y

    def matvec(self, x):
        return self._apply(np.asarray(x)[:, None])[:, 0]

    def matmat(self, X):
        return self._apply(np.asarray(X))

    def storage(self):
        s = sum(D.size for D in self.shape_blocks.values())
        s += self.k_hat.size
        s += sum(U.size + V.size for _, _, U, V in self.extra_lowrank)
        return s

    def stats(self):
        st = super().stats()
        st["n_coupling"] = self.n_kernel_blocks
        st["lattice_fft"] = True
        st["fft_shape"] = self.fft_shape
        return st

    def to_dense(self):
        """Validation only (small grids): reconstruct dense Z from the
        spatial kernel recovered by inverse FFT of the stored spectrum."""
        Kgrid = ifftn(self.k_hat, axes=self._axes, workers=-1)
        F = np.array(self.fft_shape)
        Z = np.zeros((self.n, self.n), dtype=np.complex128)
        for e, g in enumerate(self.groups):
            Z[np.ix_(g, g)] = self.shape_blocks[int(self.shape_of_elem[e])]
        P = len(self.groups)
        for a in range(P):
            for b in range(P):
                if a == b:
                    continue
                idx = tuple((self.coords[a] - self.coords[b]) % F)
                Z[np.ix_(self.groups[a], self.groups[b])] = Kgrid[idx]
        for I, J, U, V in self.extra_lowrank:
            Z[np.ix_(I, J)] += U @ V
        return Z


class _LatticeFloquetAugPrecond:
    """Block-circulant (Floquet) preconditioner for a `LatticeArrayBlock`.

    Block-Jacobi ignores *all* coupling, and on a resonant lattice the GMRES
    iteration count grows fast with array size (measured 12 → 46 → 178 on
    4×4 → 8×8 → 16×16 half-wave dipoles at 0.7λ). The lattice structure
    offers the principled fix: fold the displacement kernel onto a
    span-sized circulant grid (Chan-style, `c[m] = Σ_{Δ≡m (mod span)} T(Δ)`),
    so the preconditioner is the impedance operator of the *periodic* array —
    exact for the infinite lattice, leaving GMRES only the edge effects. It
    diagonalises over Floquet bins: one dense `(n_e+nce, n_e+nce)` augmented
    saddle per bin (the per-element KCL rows `A_e` are identical across a
    single shape class), inverted batched once; each apply is two lattice
    FFTs plus one batched per-bin matmul.

    Raises ValueError when per-element constraint blocks differ (caller falls
    back to block-Jacobi).
    """

    def __init__(self, H, kcl_A):
        n = H.n
        nc = kcl_A.shape[0] if kcl_A is not None else 0
        self.n, self.nc = n, nc
        coords = H.coords
        P, n_e = H.G.shape
        self.G = H.G
        self.n_e = n_e
        span = tuple(int(s) for s in coords.max(axis=0) + 1)
        self.span = span
        axes = tuple(range(coords.shape[1]))
        self.axes = axes
        self.site_idx = H._site_idx

        # Per-element constraint rows; must be identical across elements
        # (same shape class ⇒ same local KCL structure, but verify).
        if nc > 0:
            row_map = []
            A_ref = None
            for e in range(P):
                g = H.groups[e]
                rows = np.nonzero(np.abs(kcl_A[:, g]).sum(axis=1) > 0)[0]
                A_e = kcl_A[np.ix_(rows, g)].astype(np.complex128)
                if A_ref is None:
                    A_ref = A_e
                elif A_e.shape != A_ref.shape or not np.array_equal(A_e, A_ref):
                    raise ValueError("per-element KCL blocks differ")
                row_map.append(rows)
            row_map = np.stack(row_map)  # (P, nce)
            if len(np.unique(row_map)) != nc:
                raise ValueError("constraint rows not partitioned by element")
            self.row_map = row_map
            nce = A_ref.shape[0]
        else:
            self.row_map = None
            A_ref = None
            nce = 0
        d = n_e + nce
        self.d = d

        # Chan fold of the displacement kernel onto the span-sized circulant.
        Kgrid = ifftn(H.k_hat, axes=axes, workers=-1)
        F = np.asarray(H.fft_shape)
        Sarr = np.asarray(span)
        c = np.zeros(span + (n_e, n_e), dtype=np.complex128)
        for delta in itertools.product(*(range(-(s - 1), s) for s in span)):
            if all(v == 0 for v in delta):
                continue
            c[tuple(np.mod(delta, Sarr))] += Kgrid[tuple(np.mod(delta, F))]
        del Kgrid
        M = fftn(c, axes=axes, workers=-1)  # (*span, n_e, n_e)
        del c
        S_blk = H.shape_blocks[next(iter(H.shape_blocks))]
        if nce == 0:
            M = M + S_blk
        else:
            Maug = np.zeros(span + (d, d), dtype=np.complex128)
            Maug[..., :n_e, :n_e] = M + S_blk
            Maug[..., :n_e, n_e:] = A_ref.T
            Maug[..., n_e:, :n_e] = A_ref
            M = Maug
        self.Minv = np.linalg.inv(M.reshape((-1, d, d))).reshape(M.shape)

    def solve(self, R):
        R = np.asarray(R)
        s = R.shape[1]
        n, n_e, d = self.n, self.n_e, self.d
        Rg = np.zeros(self.span + (d, s), dtype=np.complex128)
        Rg[self.site_idx + (slice(None, n_e),)] = R[:n][self.G]
        if self.nc:
            Rg[self.site_idx + (slice(n_e, None),)] = R[n + self.row_map]
        Xg = ifftn(
            np.matmul(self.Minv, fftn(Rg, axes=self.axes, workers=-1)),
            axes=self.axes,
            workers=-1,
        )
        out = np.empty_like(R)
        body = Xg[self.site_idx]  # (P, d, s)
        out[:n][self.G] = body[:, :n_e]
        if self.nc:
            out[n + self.row_map] = body[:, n_e:]
        return out


class _BlockJacobiAugPrecond:
    """Block-Jacobi factorisation of the augmented near-field preconditioner
    `[Zn A^T; A 0]` for an `ArrayBlock`.

    Array elements are electrically separate, so the block-diagonal self-blocks
    `Zn` *and* the KCL constraint rows `A` are both block-diagonal by element:
    each junction's directional bases live in a single element (the grouping is
    connected-components on exactly those shared nodes). The augmented saddle is
    therefore itself block-diagonal — one small dense saddle
    `[S_e A_e^T; A_e 0]` per element. This is the *same* matrix the generic
    sparse-LU preconditioner factors, so GMRES convergence is identical; but it
    is factored once per distinct shape (`S_e` and `A_e` coincide across
    same-shape elements) with a dense LU. The apply stacks all same-shape
    elements' right-hand sides into one wide solve per shape (BLAS-3), so a
    `P`-element array costs `n_shapes` dense `lu_solve` calls per Krylov step
    instead of `P` separate ones. `.solve(R)` applies `M^{-1}` to an augmented
    block R (N, nrhs).
    """

    def __init__(self, ablock, kcl_A):
        self.n = ablock.n
        nc = kcl_A.shape[0] if kcl_A is not None else 0
        self.nc = nc
        groups = ablock.groups
        shape_of_elem = ablock.shape_of_elem
        shape_blocks = ablock.shape_blocks

        # Group elements that share an augmented factorisation (same shape and
        # same local KCL structure) so their solves batch into one wide RHS.
        by_key = {}  # (shape_id, A_e bytes) -> dict(fac, Ns, members=[(g, rows)])
        claimed = 0
        for e, g in enumerate(groups):
            Ns = g.size
            if nc > 0:
                rows = np.nonzero(np.abs(kcl_A[:, g]).sum(axis=1) > 0)[0].astype(
                    np.int64
                )
                A_e = (
                    kcl_A[np.ix_(rows, g)].astype(np.complex128) if rows.size else None
                )
            else:
                rows = np.empty(0, dtype=np.int64)
                A_e = None
            claimed += rows.size
            sid = int(shape_of_elem[e])
            key = (sid, A_e.tobytes() if A_e is not None else b"")
            grp = by_key.get(key)
            if grp is None:
                S = shape_blocks[sid]
                if A_e is not None:
                    nce = rows.size
                    M = np.zeros((Ns + nce, Ns + nce), dtype=np.complex128)
                    M[:Ns, :Ns] = S
                    M[:Ns, Ns:] = A_e.T
                    M[Ns:, :Ns] = A_e
                else:
                    M = np.ascontiguousarray(S)
                grp = {"fac": lu_factor(M), "Ns": Ns, "members": []}
                by_key[key] = grp
            grp["members"].append((g, rows))
        self._shapes = list(by_key.values())
        # Every constraint row must belong to exactly one element (no junction
        # bridges elements — that is what the connectivity grouping guarantees).
        assert claimed == nc, f"KCL rows {claimed} claimed != {nc} (cross-element?)"

    def solve(self, R):
        R = np.asarray(R)
        out = np.empty_like(R)
        n = self.n
        s = R.shape[1]
        for grp in self._shapes:
            fac, Ns, members = grp["fac"], grp["Ns"], grp["members"]
            k = len(members)
            nce = fac[0].shape[0] - Ns
            rhs = np.empty((Ns + nce, k * s), dtype=np.complex128)
            for j, (g, rows) in enumerate(members):
                rhs[:Ns, j * s : (j + 1) * s] = R[g]
                if nce:
                    rhs[Ns:, j * s : (j + 1) * s] = R[n + rows]
            sol = lu_solve(fac, rhs, check_finite=False, overwrite_b=True)
            for j, (g, rows) in enumerate(members):
                out[g] = sol[:Ns, j * s : (j + 1) * s]
                if nce:
                    out[n + rows] = sol[Ns:, j * s : (j + 1) * s]
        return out


class LatticeFFTUnavailable(RuntimeError):
    """Raised by `ArrayBlockSolver(require_lattice_fft=True)` when the
    lattice-FFT coupling path cannot engage. The message names the unmet
    gate condition (the same vocabulary `solver_diag()["reason"]` reports,
    antennaknobs#613) and how to meet it (antennaknobs#616)."""


class ArrayBlockSolver(HMatrixSolver):
    """Element-aware block-low-rank accelerator for arrays of identical (or
    few-shape) elements. Drop-in for `HMatrixSolver` (same constructor, plus
    `array_max_elem_bases`).

    P1 adds `array_partition()` (cached element grouping) and
    `build_array_blocks()` (the `ArrayBlock` assembly + matvec). The solve and
    `compute_impedance` / `compute_y_matrix` overrides arrive in P2; until then
    they resolve to the dense `BSplineSolver` path via the base class.

    On a mesh with no repeated-block structure to exploit (a single connected
    structure, or all-distinct element shapes) — or when one element is too
    large to densify — the solve degrades to the parent H-matrix instead of
    forcing the block decomposition (issue #143); see `_degenerate_partition`.
    """

    def __init__(
        self,
        *args,
        array_max_elem_bases=2048,
        lattice_fft="auto",
        require_lattice_fft=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Block-Toeplitz FFT coupling for regular same-shape lattices
        # (docs/lattice-fft-plan.md). `lattice_fft` is a representation
        # *preference*, never a guarantee: "auto" (default) switches to the
        # FFT representation when a lattice is detected and P ≥ 16; True
        # prefers it at any element count but still yields to the other
        # gates (one block-shape class, a regular lattice fit); False keeps
        # the per-pair path. To *assert* the FFT path engaged, pass
        # `require_lattice_fft=True` — then any solve whose operator is not
        # the lattice-FFT one raises `LatticeFFTUnavailable` naming the
        # unmet gate condition (antennaknobs#616).
        self.lattice_fft = lattice_fft
        self.require_lattice_fft = bool(require_lattice_fft)
        if self.require_lattice_fft:
            if not self.lattice_fft:
                raise ValueError(
                    "require_lattice_fft=True contradicts lattice_fft=False"
                )
            if self.use_singular_enrichment:
                # Known at construction: enrichment forces the dense-path
                # fallback (`_hmatrix_unsupported`), so no solve could ever
                # engage the FFT path. Fail fast rather than at solve time.
                raise self._lattice_fft_unavailable(
                    "use_singular_enrichment forces the dense-path fallback",
                    "Disable singular enrichment",
                )
        # Largest element (in bases) the block decomposition will densify.
        # A dense self-block build materialises an (m, m, degree+1, degree+1)
        # complex gather — 576 MiB at m=2048, degree 2, but 21.6 GiB at the
        # whip deck's m=12,682 (issue #143) — and the resulting (m, m) block
        # must then be LU-factored for the block-Jacobi preconditioner.
        # Beyond this the parent H-matrix (ACA partition + bounded near
        # blocks) is the better tool, so `_build_operator` falls back to it.
        self.array_max_elem_bases = int(array_max_elem_bases)

    def _hmatrix_unsupported(self):
        """ArrayBlock supports PEC ground via the per-block image term (the
        free-space block reuse survives the image method for a grid array — see
        `build_array_blocks`), the reflection-coefficient finite ground (the
        Fresnel weights depend only on relative displacement + heights,
        exactly what the grounded block keys already carry), and the
        Sommerfeld finite ground: the singular part is the C2-scaled PEC
        image (same keys again — C2 is a per-frequency constant), and the
        smooth remainder rides `ArrayBlock.extra_lowrank` as ONE global
        low-rank term outside the block cache, so shape/displacement dedup
        and per-shape factorisation reuse are untouched
        (docs/sommerfeld-everywhere-plan.md Phase 3). Falls back to the
        dense path only for singular enrichment."""
        return self.use_singular_enrichment

    def array_partition(self, tol=1e-6):
        """Element/shape partition of the bases (cached)."""
        cached = getattr(self, "_array_partition", None)
        if cached is None:
            cached = element_groups(self, tol=tol)
            self._array_partition = cached
        return cached

    def _self_block_key(self, ctx, segs, basis_idx, k):
        """Content-addressed key for an element's dense self-block: the
        element's segment endpoints recentred on its own centroid (so it is
        translation-invariant — identical elements at different array positions
        share a key), rounded and canonically ordered, plus the parameters the
        self-impedance depends on (k, wire radius, degree, quadrature).

        Under PEC ground the self-block also carries the self-image term, which
        depends on the element's *height above the ground plane* (the image sits
        at ``2·ground_z − z``), so the centroid height ``cen_z − ground_z`` joins
        the key — same-height translates still share a block (the grid case),
        while a free-space block (``ground_z is None``) never aliases a grounded
        one and elements at different heights get distinct blocks."""
        seg_l, seg_r = ctx["seg_l"][segs], ctx["seg_r"][segs]
        cen = 0.5 * (seg_l + seg_r).mean(axis=0)
        rel = np.hstack([seg_l - cen, seg_r - cen])
        keyarr = np.round(rel / 1e-6).astype(np.int64)
        mid = 0.5 * (keyarr[:, :3] + keyarr[:, 3:])
        order = np.lexsort((mid[:, 2], mid[:, 1], mid[:, 0]))
        # Per-segment conductor radii in the same canonical order — the
        # module-scope cache must never alias two translation-identical
        # elements with different (or differently-arranged) radii
        # (stevenmburns/momwire#147).
        sig = keyarr[order].tobytes() + ctx["seg_a"][segs][order].tobytes()
        # Basis-composition signature (issue #240): segment geometry alone
        # doesn't determine this element's basis set — a wire end kept as a
        # junction/ground directional basis vs dropped as free changes the
        # block's actual size and content (_build_basis_polynomials), with
        # no visible trace in the segments above. Two solver instances with
        # identical wires/segmentation but different self.junctions must not
        # alias here. Same recentre-round-lexsort-tobytes treatment as the
        # segment signature, over basis centroids, so it is translation-
        # invariant and also separates a same-count composition swap (the
        # extra tip basis moving from one wire end to another).
        bcen = ctx["basis_centroid"][basis_idx] - cen
        bkeyarr = np.round(bcen / 1e-6).astype(np.int64)
        border = np.lexsort((bkeyarr[:, 2], bkeyarr[:, 1], bkeyarr[:, 0]))
        sig += bkeyarr[border].tobytes()
        gkey = (
            None
            if self.ground_z is None
            else int(np.round((cen[2] - self.ground_z) / 1e-6))
        )
        # The Fresnel-weighted image block depends on the ground constants
        # and the Φ mode too — a PEC-image block must never alias a
        # ground_eps one (or one for different ground constants) in the
        # module-scope cache. ground_model joins the key: a sommerfeld
        # self-block (C2-scaled image) must never alias a refl-coef one
        # for the same constants.
        ekey = (
            None
            if self.ground_eps is None
            else (repr(self.ground_eps), self.ground_phi_mode, self.ground_model)
        )
        # `extended_kernel` changes the operator (a different Green's
        # function on the coaxial pairs), so it joins the key for the same
        # reason `junctions` did in issue #240: two solver instances with
        # identical geometry, radius, k and degree but different kernels
        # must not alias in this module-scope cache.
        return (
            sig,
            float(k),
            self.degree,
            self.n_qp_pair,
            gkey,
            ekey,
            self.extended_kernel,
        )

    def _lattice_fft_unavailable(self, reason, hint):
        """Build (not raise) the `require_lattice_fft` failure, so callers
        read as `raise self._lattice_fft_unavailable(...)`. `reason` is the
        gate condition in `solver_diag()` vocabulary; `hint` is the concrete
        way to meet it."""
        return LatticeFFTUnavailable(
            f"lattice-FFT path unavailable: {reason}. "
            f"{hint} — or drop require_lattice_fft."
        )

    def _degenerate_partition(self):
        """True when the element partition offers nothing the block solver
        can exploit: no shape class has two members (so neither the self-block
        reuse nor the coupling-displacement dedup ever fires), or some element
        exceeds `array_max_elem_bases` (its dense self-block would materialise
        a multi-GiB gather — issue #143). The whip-style mesh is the extreme
        case: one connected structure ⇒ one element holding every basis."""
        part = self.array_partition()
        if part.n_elem == 0:
            return True
        counts = np.bincount(part.shape_of_elem, minlength=part.n_shapes)
        no_reuse = not bool((counts >= 2).any())
        oversize = int(part.sizes.max()) > self.array_max_elem_bases
        return no_reuse or oversize

    def _build_operator(self):
        """Build the array-block operator (cached) for the constrained solve.

        `compute_impedance` / `compute_y_matrix` (inherited from
        `HMatrixSolver`) run the same GMRES on it via `_solve_hmatrix`, with the
        block-diagonal self-blocks as the block-Jacobi preconditioner.

        The assembled operator is cached at module scope keyed by the full
        geometry + k + tol, so an animation *phase/excitation* sweep — geometry
        fixed, only the RHS changes — reuses both the operator and the
        factorisation cached on it (`HMatrixSolver._factored_solve`), making each
        frame a cheap multi-RHS back-substitution."""
        if self._degenerate_partition():
            if self.require_lattice_fft:
                part = self.array_partition()
                if part.n_elem == 0:
                    reason, hint = "empty element partition", "Add elements"
                elif int(part.sizes.max()) > self.array_max_elem_bases:
                    reason = (
                        f"largest element holds {int(part.sizes.max())} bases "
                        f"> array_max_elem_bases={self.array_max_elem_bases} "
                        f"(H-matrix fallback, issue #143)"
                    )
                    hint = "Raise array_max_elem_bases or mesh the element finer"
                else:
                    reason = (
                        "no shape class has two members, so there is no "
                        "block structure to exploit (H-matrix fallback, "
                        "issue #143)"
                    )
                    hint = "Use repeated identical elements"
                raise self._lattice_fft_unavailable(reason, hint)
            # Nothing to exploit (or an element too big to densify): degrade
            # to the parent H-matrix — ACA partition + bounded near blocks —
            # instead of feeding whole-matrix index sets to zblock. On the
            # whip benchmark (12,682 bases, one connected element) the block
            # path gathered a 21.6 GiB intermediate where the H-matrix
            # solves the same deck in ~2.1 GB (issue #143).
            _CACHE_STATS["hmatrix_fallback"] += 1
            self._last_n_coupling_aca = 0
            H = super()._build_operator()
            # solver_diag() (antennaknobs#613) reads this straight off the
            # operator — a plain `HMatrix` instance identifies the fallback
            # by type, but only this branch knows *why* it fell back.
            H._fft_reason = (
                "fell back to the H-matrix (no exploitable element partition)"
            )
            return H
        key = (
            self._geometry_cache_key(),
            float(self.k),
            self._radius_per_wire.tobytes(),
            self.degree,
            self.n_qp_pair,
            float(self.aca_tol),
            repr(self.lattice_fft),
            None if self.ground_z is None else float(self.ground_z),
            None
            if self.ground_eps is None
            else (repr(self.ground_eps), self.ground_phi_mode, self.ground_model),
            # junctions change which boundary bases are kept vs dropped
            # (bspline.py's _build_basis_polynomials), so they change the
            # basis count/composition — same geometry + k, different
            # junctions must not alias (issue #240). junction_ports is
            # deliberately excluded: per #234/_same_constraints, a port
            # moves a KCL row out of the constraint set without touching
            # the assembled operator, so it cannot stale this cache.
            tuple(tuple((w, e) for (w, e) in jw) for jw in self.junctions),
            # Same staleness class as junctions (#240): the extended kernel
            # is a different operator on the same geometry, so it must not
            # alias a reduced-kernel operator here (momwire#249).
            self.extended_kernel,
        )
        op = _ARRAY_OP_CACHE.get(key)
        if op is None:
            op = self.build_array_blocks()
            op._n_coupling_aca = self._last_n_coupling_aca
            _cache_put(_ARRAY_OP_CACHE, key, op, _ARRAY_OP_CACHE_MAX)
            _CACHE_STATS["operator_build"] += 1
        else:
            _CACHE_STATS["operator_hit"] += 1
        if self.require_lattice_fft and not isinstance(op, LatticeArrayBlock):
            # Cache hits skip build_array_blocks' gate walk, so re-assert
            # here from the reason/hint the build stored on the operator.
            raise self._lattice_fft_unavailable(
                op._fft_reason, getattr(op, "_fft_hint", "Meet the gate above")
            )
        self._last_n_coupling_aca = op._n_coupling_aca
        return op

    def _make_preconditioner(self, H, kcl_A):
        """Per-element block-Jacobi factorisation of the augmented near-field
        preconditioner — the same matrix the generic sparse-LU path factors,
        but block-diagonal by element so it factors once per shape and applies
        as batched dense solves (see `_BlockJacobiAugPrecond`).

        When `_build_operator` fell back to the parent H-matrix (degenerate
        partition, issue #143) the operator has no element blocks — delegate
        to the generic sparse-LU preconditioner."""
        if not isinstance(H, ArrayBlock):
            return super()._make_preconditioner(H, kcl_A)
        if isinstance(H, LatticeArrayBlock):
            # Periodic-array (Floquet) preconditioner: exact for the infinite
            # lattice, so GMRES only iterates on edge effects — block-Jacobi's
            # iteration count grows fast with array size on resonant lattices.
            try:
                return _LatticeFloquetAugPrecond(H, kcl_A)
            except ValueError:
                pass  # irregular constraint structure: block-Jacobi still valid
        return _BlockJacobiAugPrecond(H, kcl_A)

    def _coupling_aca(self, ctx, I, J, k, tol, use_accel):
        """ACA low-rank factors (U, V) of the off-diagonal element block
        Z[I][:, J]. The two elements share no segments, so the block is purely
        off-edge (no same-edge analytic overwrite) and well separated ⇒ low
        rank. Reuses the shared off-edge ACA evaluators, which fold in the PEC
        image term (``Z_free − Z_image``) when ground is on — one factor pair
        per pair, so the matvec and block-Jacobi machinery are unchanged.
        Sharing those evaluators is also what makes `extended_kernel` work
        here: they force `use_accel=False` under EK, so this fill inherits
        the numpy block path without knowing the flag exists
        (momwire#249)."""
        get_row, get_col, _dense = self._offedge_aca_evaluators(ctx, I, J, k, use_accel)
        U, V = aca_partial(get_row, get_col, I.size, J.size, tol=tol)
        return U, V

    def _build_lattice_operator(
        self, ctx, part, shp, shape_blocks, coords, k, use_accel, elem_a, extra_lowrank
    ):
        """Assemble the block-Toeplitz coupling kernel for a regular lattice
        of same-shape elements and return a `LatticeArrayBlock`.

        One dense `(n_e, n_e)` off-edge block per lattice displacement —
        filled with the same ground-aware evaluators the ACA path samples,
        but kept exact (no truncation) since the FFT convolution needs the
        full block anyway. `T(−Δ) = T(Δ)ᵀ` halves the fills under the same
        uniform-radius gate the per-pair dedup uses. Displacements that never
        occur between occupied sites (sparse lattices) stay zero: the
        convolution only ever reads them into cropped-away rows.

        `extended_kernel` needs nothing here. Coaxial-and-equal-radius
        eligibility is translation-invariant — a lattice translate of a
        segment is coaxial with the original iff the lattice vector lies
        along the axis, which is a property of the lattice, not of the site
        — so every displacement block keeps the value it would have had and
        the block-Toeplitz structure is preserved. Argued, and gated (G8 in
        tests/test_extended_kernel_bspline.py: a COLLINEAR lattice, where
        cross-element pairs really are eligible, against the per-pair path).
        """
        n = ctx["n_basis"]
        sizes = np.asarray(part.sizes)
        assert sizes.min() == sizes.max(), "single shape class ⇒ equal sizes"
        n_e = int(sizes[0])
        span = coords.max(axis=0) + 1
        fft_shape = tuple(int(next_fast_len(2 * int(s) - 1)) for s in span)
        F = np.asarray(fft_shape)
        site = {tuple(c): e for e, c in enumerate(coords)}
        coords_list = list(site)
        sym = elem_a[0] is not None and all(a == elem_a[0] for a in elem_a)
        Kgrid = np.zeros(fft_shape + (n_e, n_e), dtype=np.complex128)
        filled = set()
        n_fill = 0
        ranges = (range(-(int(s) - 1), int(s)) for s in span)
        for delta in itertools.product(*ranges):
            # delta = observer coord − source coord; slot (a−b) holds Z_ab.
            if all(v == 0 for v in delta) or delta in filled:
                continue
            self._checkpoint()  # per displacement fill (cancellable build)
            cb = tuple(max(0, -v) for v in delta)  # full-grid anchor: source
            ca = tuple(c + v for c, v in zip(cb, delta))  # observer
            if cb not in site or ca not in site:
                ca = cb = None
                for cb2 in coords_list:  # sparse lattice: scan for any pair
                    ca2 = tuple(c + v for c, v in zip(cb2, delta))
                    if ca2 in site:
                        ca, cb = ca2, cb2
                        break
                if ca is None:
                    continue  # displacement absent from the occupied set
            a, b = site[ca], site[cb]
            T = self._offedge_aca_evaluators(
                ctx, part.groups[a], part.groups[b], k, use_accel
            )[2]()
            n_fill += 1
            Kgrid[tuple(np.mod(delta, F))] = T
            filled.add(delta)
            if sym:
                mdelta = tuple(-v for v in delta)
                Kgrid[tuple(np.mod(mdelta, F))] = T.T
                filled.add(mdelta)
        axes = tuple(range(coords.shape[1]))
        k_hat = fftn(Kgrid, axes=axes, workers=-1)
        del Kgrid
        self._last_n_coupling_aca = n_fill
        return LatticeArrayBlock(
            n,
            part.groups,
            shp,
            shape_blocks,
            coords,
            fft_shape,
            k_hat,
            len(filled),
            extra_lowrank=extra_lowrank,
            cancel=self._cancel,
        )

    def build_array_blocks(self, tol=None, k=None):
        """Assemble the impedance matrix as an `ArrayBlock`.

        One dense self-block per shape class (built once from a representative
        element and reused across same-shape elements), plus a low-rank
        coupling block per element pair.

        Coupling reuse (block-Toeplitz, generalised): a coupling block depends
        only on the two elements' shapes and their relative displacement (the
        free-space kernel is translation-invariant), so all pairs sharing a
        `(shape_a, shape_b, displacement)` key are the *same* block — ACA runs
        once per unique key. On a regular grid this collapses the `P(P-1)`
        pairs to a handful of displacements. Complex symmetry `Z_ba = Z_ab^T`
        is folded in too: a key whose reverse is already cached reuses the
        transposed factors instead of a fresh ACA. `self._last_n_coupling_aca`
        records how many ACA solves actually ran.
        """
        if tol is None:
            tol = self.aca_tol
        if k is None:
            k = self.k
        part = self.array_partition()
        ctx = self._context()
        n = ctx["n_basis"]
        somm = (
            self.ground_z is not None
            and self.ground_eps is not None
            and self.ground_model == "sommerfeld"
        )
        if somm:
            eps_t, c2 = self._somm_eps_c2()

        # Element centroids, from each element's own segment midpoints (translated
        # elements differ by exactly the displacement, so rounding to the grouping
        # tol is safe). Computed from seg_groups, not ctx["basis_centroid"] — the
        # latter is polluted by boundary-basis support padding, which would
        # perturb the displacement/height keys.
        seg_mid = 0.5 * (ctx["seg_l"] + ctx["seg_r"])
        cen = np.array([seg_mid[sg].mean(axis=0) for sg in part.seg_groups])
        disp_tol = 1e-6

        # Block-shape classes. In free space a block depends only on an element's
        # (translation-invariant) geometric shape. Under PEC ground both the
        # self-image and the coupling-image terms also depend on the element's
        # height above the plane (the image sits at 2·ground_z − z), so refine the
        # shape classes by height: same-shape elements at the same height share a
        # block — the single-height grid case (e.g. bowtiearray2x4), full reuse —
        # while different heights get distinct blocks (correct; reuse degrades
        # gracefully, the solve stays fast). `shp[e]` is element e's dense
        # block-class id; free space leaves it equal to the geometric shape id.
        if self.ground_z is None:
            shp = np.asarray(part.shape_of_elem)
            reps = part.shape_representatives()
        else:
            label_of = {}
            reps = []
            shp = np.empty(part.n_elem, dtype=np.int64)
            for e in range(part.n_elem):
                hk = int(np.round((cen[e][2] - self.ground_z) / disp_tol))
                key = (int(part.shape_of_elem[e]), hk)
                lab = label_of.get(key)
                if lab is None:
                    lab = len(label_of)
                    label_of[key] = lab
                    reps.append(e)
                shp[e] = lab

        # Dense self-block per distinct block-shape, from a representative element.
        # Cached by the element's translation-invariant geometry signature (plus
        # height under ground) so a spacing sweep reuses the assembly.
        shape_blocks = {}
        for s, e in enumerate(reps):
            self._checkpoint()  # per distinct self-block dense fill
            g = part.groups[e]
            sb_key = self._self_block_key(ctx, part.seg_groups[e], g, k)
            blk = _SELF_BLOCK_CACHE.get(sb_key)
            if blk is None:
                blk = self.zblock(g, g, k=k)
                if self.ground_z is not None:
                    # Self-image reaction: the element against its own mirror. One
                    # block per (shape, height) class, so the block-Jacobi
                    # preconditioner stays (near-)exact and the
                    # one-factorisation-per-class cost is unchanged. Under
                    # ground_eps the weighted image replaces the PEC one; its
                    # weights depend only on displacement + heights, which the
                    # block key already carries. Sommerfeld's singular part is
                    # C2 x the PEC image (the smooth remainder rides the
                    # global extra_lowrank term, outside the block cache).
                    if somm:
                        blk = blk - c2 * self._zblock_image(g, g, k=k)
                    elif self.ground_eps is not None:
                        blk = blk - self._zblock_image_refl(g, g, k=k)
                    else:
                        blk = blk - self._zblock_image(g, g, k=k)
                _cache_put(_SELF_BLOCK_CACHE, sb_key, blk, _SELF_BLOCK_CACHE_MAX)
                _CACHE_STATS["self_block_build"] += 1
            else:
                _CACHE_STATS["self_block_hit"] += 1
            shape_blocks[s] = blk

        use_accel = (
            _HAVE_OFFEDGE_BLOCK_ACCEL
            and self.degree <= _OFFEDGE_BLOCK_ACCEL_MAX_D
            and self.hmatrix_use_accel
        )

        extra_lowrank = None
        if somm:
            U, V = self._sommerfeld_global_lowrank(k, eps_t)
            idx = np.arange(n, dtype=np.int64)
            extra_lowrank = [(idx, idx, U, -V)]  # Z subtracts the remainder

        # Per-element uniform radius (or None when internally mixed). Complex
        # symmetry Z_ab = Z_ba^T needs the a²-regularisation to be symmetric
        # across the pair: mixed per-wire radii regularise each OBSERVER row
        # with its own wire's radius (stevenmburns/momwire#147), so transposed
        # factors/blocks are only reusable when every wire in BOTH elements
        # carries one and the same radius.
        seg_a = ctx["seg_a"]
        elem_a = []
        for sg in part.seg_groups:
            av = seg_a[sg]
            elem_a.append(float(av[0]) if np.all(av == av[0]) else None)

        P = part.n_elem
        # Lattice-FFT path (docs/lattice-fft-plan.md): one block-shape class on
        # a regular lattice ⇒ the coupling is block-Toeplitz per lattice
        # dimension. Represent it as a dense displacement-block kernel in FFT
        # form and skip the O(P²) pair loop entirely. "auto" requires enough
        # elements for the FFT bookkeeping to matter; `lattice_fft=True`
        # forces it (tests / benchmarks), False disables.
        #
        # fft_reason (antennaknobs#613) walks the same gate in the same
        # order and records, in the gate's own vocabulary, which condition
        # kept the path from engaging — solver_diag() surfaces it verbatim
        # so a user can act on it (constant height, 16+ elements, a regular
        # grid) instead of just seeing `lattice_fft: false`.
        if not self.lattice_fft:
            fft_reason = "lattice FFT explicitly disabled"
            fft_hint = "Pass lattice_fft='auto' or True"
        elif len(shape_blocks) > 1:
            if len(shape_blocks) > part.n_shapes:
                # Only ground refinement splits block-shape classes beyond
                # the geometric shape count (`shp` construction above).
                fft_reason = (
                    f"shape classes split by height above ground "
                    f"({len(shape_blocks)} classes); the FFT path needs 1"
                )
                fft_hint = "Lay the array out at constant height above the ground plane"
            else:
                fft_reason = (
                    f"{len(shape_blocks)} distinct element shapes; the FFT path needs 1"
                )
                fft_hint = "Make every element geometrically identical"
        elif self.lattice_fft is not True and P < 16:
            fft_reason = f"only {P} elements; the FFT path engages at 16+"
            fft_hint = "Pass lattice_fft=True to engage below 16 elements"
        else:
            lattice = _detect_lattice(cen, disp_tol)
            if lattice is not None:
                op = self._build_lattice_operator(
                    ctx,
                    part,
                    shp,
                    shape_blocks,
                    lattice[0],
                    k,
                    use_accel,
                    elem_a,
                    extra_lowrank,
                )
                op._fft_reason = None
                return op
            fft_reason = "element centroids don't fit a regular lattice"
            fft_hint = "Lay the elements out on a regular (possibly sparse) grid"
        if self.require_lattice_fft:
            # Raise here, before the O(P²) per-pair coupling build below —
            # the caller asked for an assertion, not a slow fallback.
            raise self._lattice_fft_unavailable(fft_reason, fft_hint)

        # Coupling reuse: a block depends only on the two elements' block-shapes
        # and their relative displacement (the free-space kernel is
        # translation-invariant; under ground the block-shape already encodes the
        # height the image term needs, so the same key keys both the free and the
        # image contribution). All pairs sharing `(block_shape_a, block_shape_b,
        # displacement)` are the same block — ACA runs once per unique key. On a
        # single-height grid this collapses the P(P-1) pairs to a handful of
        # displacements. Complex symmetry Z_ab = Z_ba^T survives the image method
        # (both free and image blocks are symmetric), so a key whose reverse is
        # cached reuses the transposed factors instead of a fresh ACA.
        coupling = []
        cache = {}  # (block_shape_a, block_shape_b, disp_key) -> (U, V)
        n_aca = 0
        for a in range(P):
            self._checkpoint()  # per element-row of the coupling ACA grid
            for b in range(P):
                if a == b:
                    continue
                sa, sb = int(shp[a]), int(shp[b])
                dkey = tuple(np.round((cen[b] - cen[a]) / disp_tol).astype(np.int64))
                key = (sa, sb, dkey)
                hit = cache.get(key)
                if hit is None:
                    rkey = (sb, sa, tuple(-d for d in dkey))
                    rhit = cache.get(rkey)
                    symmetric_pair = elem_a[a] is not None and elem_a[a] == elem_a[b]
                    if rhit is not None and symmetric_pair:
                        # Z_ab = Z_ba^T = (U_r V_r)^T = V_r^T U_r^T.
                        hit = (rhit[1].T.copy(), rhit[0].T.copy())
                    else:
                        hit = self._coupling_aca(
                            ctx, part.groups[a], part.groups[b], k, tol, use_accel
                        )
                        n_aca += 1
                    cache[key] = hit
                coupling.append((a, b, hit[0], hit[1]))

        self._last_n_coupling_aca = n_aca

        op = ArrayBlock(
            n,
            part.groups,
            shp,
            shape_blocks,
            coupling,
            extra_lowrank=extra_lowrank,
            cancel=self._cancel,
        )
        op._fft_reason = fft_reason
        op._fft_hint = fft_hint
        return op

    def solver_diag(self):
        """Advisory diagnostics for which coupling path the last solve used
        (antennaknobs#613): the fast lattice-FFT path, the per-pair
        `ArrayBlock` path, or a degenerate fallback to the parent H-matrix
        (issue #143) — plus, when the FFT path did not engage, why not.

        Pulled from the operator `compute_impedance()` / `compute_y_matrix()`
        already built and cached on `self._hmatrix` — reading it needs no
        re-solve. Returns None before any solve has run.
        """
        op = getattr(self, "_hmatrix", None)
        if op is None:
            return None
        stats = op.stats()
        return {
            "operator": type(op).__name__,
            "lattice_fft": bool(stats.get("lattice_fft", False)),
            "n_elem": stats.get("n_elem"),
            "n_shapes": stats.get("n_shapes"),
            "reason": getattr(op, "_fft_reason", None),
        }
