"""Hierarchical (H-matrix / ACA) B-spline Galerkin MoM solver.

`HMatrixSolver` is a distance-based accelerator built on top of
`BSplineSolver`. It reuses BSplineSolver's geometry build, basis-polynomial
extraction, kernels, source vectors, and KCL machinery verbatim; the only
thing it replaces is the *dense* O(N²) impedance-matrix assembly + dense LU
solve.

The plan (phased — see project notes):

  * Phase 0 (this file, to start): an on-demand block evaluator
    `zblock(I, J)` that computes any rectangular sub-block Z[I][:, J] of the
    exact dense bspline Z *without* materialising the full (d+1, d+1, N, N)
    moment tensor. Well-separated (admissible) blocks contain no same-edge
    segment pairs, so they need only the off-edge full-kernel quadrature;
    near blocks additionally overwrite same-edge pairs with the analytic
    static + regularised split, identical to `BSplineSolver._build_J_blocks`.
    This evaluator is the foundation everything else stands on: ACA can only
    be as correct as the entries it samples.

  * Phase 1: binary-space-partition cluster tree over basis centroids;
    admissibility min(diam(s), diam(t)) <= eta * dist(s, t); recursive
    block tree → {admissible far, dense near} leaves.

  * Phase 2: partial-pivoted ACA low-rank approximation of admissible
    blocks; dense near blocks; fast H-matvec.

  * Phase 3: preconditioned block GMRES on the batched operator
    (`matmat`) + near-field preconditioner, KCL junction constraints in the
    augmented system. All RHS share one block Krylov space, so the operator
    and preconditioner applies are batched (BLAS-3) across the columns.

Why distance matters here: the moment integral kernel
G = exp(-jkR)/(4πR) is smooth and asymptotically smooth once the two
B-spline basis supports are well separated, so the corresponding Z block is
numerically low rank and ACA captures it from O(r·(m+n)) sampled entries
instead of the full m·n.
"""

import math

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import splu

from .bspline import BSplineSolver, _SplineBasis, _EK_SAME_EDGE, _ek_slice
from ._port_solution import PortSolution

from . import _ground_mirror, _ground_refl, _potential_ground, _sommerfeld
from ._bspline_kernels import (
    _ek_radius,
    _refuse_complex_k,
    _seg_seg_full_moments_offedge,
    _seg_seg_reg_moments,
    _seg_seg_static_moments,
)
from ._aca import (
    HMatrix,
    aca_partial,
    admissible,
    build_block_tree,
    build_cluster_tree,
    partition_stats,
)
from ._quadrature import leggauss

from ._accel import acc as _acc

# momwire#553 U5 — the fast operator has no medium. Named here rather than
# inside the guard so `capabilities` and the tests quote one sentence.
#
# The `{who}: ` naming the raising method moved OUT of this constant to the
# raise site in momwire#792, leaving the constant a per-deck preamble
# plus a constant reason like every other geometry refusal in the tree — which
# is what lets `capabilities.refusals["buried"]` be this string rather than a
# copy of it. The raised message is unchanged.
_BURIED_FAST_OPERATOR_REFUSAL = (
    "this deck has a wire below the ground plane, and the fast "
    "operator has no per-segment medium. Admissibility is a purely "
    "geometric distance test with no notion of which side of the interface "
    "a cluster is on; the fused near/far block kernels take a `double k` and "
    "would truncate the in-medium k_m = k0*sqrt(eps_tilde) to its real part; "
    "and the Sommerfeld composition is carried as ONE global low-rank "
    "remainder over ONE grid, where a buried deck needs three blocks over "
    "three grids (momwire#553 U5). BSplineSolver serves the deck through its "
    "dense fill - this class deliberately does NOT fall back to it, because "
    "it exists for decks the dense fill cannot hold, and a silent fallback "
    "on a buried array is an out-of-memory rather than a slow answer"
)

_HAVE_OFFEDGE_BLOCK_ACCEL = _acc is not None and hasattr(
    _acc, "bspline_assemble_offedge_block"
)
_HAVE_OFFEDGE_BLOCK_REFL_ACCEL = _acc is not None and hasattr(
    _acc, "bspline_assemble_offedge_block_refl"
)
# The fused off-edge assembler's extended-kernel twin (momwire#270 unit 3):
# a build without it degrades to the numpy zblock(..., same_edge=False)
# fallback under `extended_kernel` rather than silently using the reduced
# fused assembler (which would be numerically wrong under EK) — see
# `_offedge_aca_evaluators`'s `use_accel` gate.
_HAVE_OFFEDGE_BLOCK_EK_ACCEL = _acc is not None and hasattr(
    _acc, "bspline_assemble_offedge_block_ek"
)
# The refl-coef image's own EK twin (momwire#269). A build without it takes
# the numpy `_zblock_image_refl` closures for the far-block image under
# `extended_kernel` + `ground_eps` — correct (that path routes through the
# EK-aware `_seg_seg_full_moments_offedge`), just unfused. Falling back to
# `bspline_assemble_offedge_block_refl` would be silently REDUCED, the same
# trap `_HAVE_OFFEDGE_BLOCK_EK_ACCEL` guards on the free-space side.
_HAVE_OFFEDGE_BLOCK_REFL_EK_ACCEL = _acc is not None and hasattr(
    _acc, "bspline_assemble_offedge_block_refl_ek"
)

_OFFEDGE_BLOCK_ACCEL_MAX_D = 2


class _SparseAugPrecond:
    """Generic single sparse-LU factorisation of the augmented near-field
    preconditioner `[Zn A^T; A 0]`. Used by the H-matrix path, where the near
    band has no exploitable block structure. `.solve(R)` applies `M^{-1}` to an
    augmented block R (N, nrhs) via one SuperLU back-substitution."""

    def __init__(self, Zn, kcl_A):
        nc = kcl_A.shape[0] if kcl_A is not None else 0
        if nc > 0:
            A_sp = sp.csr_matrix(kcl_A.astype(np.complex128))
            Saug = sp.bmat([[Zn, A_sp.T], [A_sp, None]], format="csc")
        else:
            Saug = Zn.tocsc() if sp.issparse(Zn) else Zn
        self.lu = splu(Saug)

    def solve(self, R):
        return self.lu.solve(R)


def _same_constraints(a, b):
    """True when two constraint-row matrices define the same augmented system.

    A cached `_AugmentedFactoredSolve` is only reusable for constraint rows it
    was built on. `junction_ports` (#234) moves rows out of the constraint set
    without touching the operator, so an operator-cache hit can hand back a
    factorisation of a *different* saddle system; row count alone does not
    separate the cases (two solvers can port different junctions)."""
    if a is b:
        return True
    a_empty = a is None or a.shape[0] == 0
    b_empty = b is None or b.shape[0] == 0
    if a_empty or b_empty:
        return a_empty and b_empty
    return a.shape == b.shape and np.array_equal(a, b)


class _AugmentedFactoredSolve:
    """A factored augmented near-field preconditioner plus the preconditioned
    block GMRES on the operator `H`.

    The preconditioner is a pluggable object with a `.solve(R)` method (sparse
    LU for the H-matrix, per-element block-Jacobi for the array-block solver).
    Holds no reference to the solver instance, so it can be cached on the
    operator and reused across solves that share it: the factorisation is
    RHS-independent, so an animation phase/excitation sweep re-solves with
    cached back-substitutions (`X0 = M^{-1} B`) plus a handful of block-Krylov
    steps, never refactoring.
    """

    def __init__(self, H, kcl_A, precond):
        self.H = H
        self.kcl_A = kcl_A
        self.precond = precond
        n = H.n
        self.n = n
        nc = kcl_A.shape[0] if kcl_A is not None else 0
        self.nc = nc
        self.N = n + nc

    def _aug_matmat(self, Z):
        """Apply the augmented operator [Z A^T; A 0] to a block Z (N, nrhs)."""
        n, nc = self.n, self.nc
        out = np.empty((self.N, Z.shape[1]), dtype=np.complex128)
        out[:n] = self.H.matmat(Z[:n])
        if nc > 0:
            out[:n] += self.kcl_A.T @ Z[n:]
            out[n:] = self.kcl_A @ Z[:n]
        return out

    def solve(self, B, rtol):
        """Solve the augmented system for every RHS column of B (n, nrhs) at
        once with left-preconditioned block GMRES. Returns (X (n, nrhs),
        iters). All RHS share one block Krylov space and every operator/
        preconditioner apply is batched across the columns (BLAS-3), so the
        cost is ~one batched matmat + one batched back-substitution per Krylov
        step instead of nrhs separate matvec/solve pairs per step."""
        n, N = self.n, self.N
        Baug = np.zeros((N, B.shape[1]), dtype=np.complex128)
        Baug[:n] = B
        X, iters = self._block_gmres(Baug, rtol)
        return X[:n], iters

    def _block_gmres(self, Baug, rtol, restart=None, maxiter=2000):
        """Left-preconditioned restarted block GMRES on the augmented system.

        Solves M^{-1} A X = M^{-1} B for all columns simultaneously (M = the
        factored near-field preconditioner, A = the augmented operator). The
        preconditioned initial guess `X0 = M^{-1} B` is the same excellent
        starting point the per-RHS path used. Convergence is the per-column
        preconditioned-residual norm relative to ‖M^{-1} b_j‖ — matching SciPy
        gmres's left-preconditioned stopping test, so accuracy is unchanged.

        The restart depth defaults to the operator's `gmres_restart` hint (50
        when absent). Resonant lattices need deep Krylov spaces: a 32×32
        half-wave-dipole grid converges in 226 unrestarted iterations but
        *stalls indefinitely* at restart 50 — truncation discards exactly the
        slowly-converging edge-resonance directions (`LatticeArrayBlock` hints
        300). The depth is capped so the stored Krylov block (m·N·nrhs
        complex) stays within ~2 GB — a tighter clamp can land back inside
        stall territory and defeat the hint (a 100×100 lattice, N=1.4e5,
        stalled when an earlier 500 MB cap cut the depth to 223).
        """
        if restart is None:
            restart = getattr(self.H, "gmres_restart", 50)
            restart = max(50, min(restart, int(2e9 / (16 * self.N * Baug.shape[1]))))
        prec = self.precond.solve
        Bt = prec(Baug)  # M^{-1} B
        bnorms = np.linalg.norm(Bt, axis=0)
        bnorms = np.where(bnorms == 0.0, 1.0, bnorms)
        X = Bt.copy()  # X0 = M^{-1} B
        s = Baug.shape[1]
        m = min(restart, self.N)
        total_iters = 0
        for _ in range(maxiter // m + 1):
            R = Bt - prec(self._aug_matmat(X))  # preconditioned residual
            if np.all(np.linalg.norm(R, axis=0) <= rtol * bnorms):
                break
            Q, beta = np.linalg.qr(R)  # R = Q @ beta; Q (N,s), beta (s,s)
            # Krylov basis as ONE contiguous block: orthogonalisation is then
            # two BLAS-3 products per iteration (classical Gram-Schmidt done
            # twice — CGS2, as stable as modified GS in practice) instead of
            # a Python loop of k rank-s updates. At deep restarts (lattice
            # operators hint 600) the loop version dominated the whole solve.
            V = np.empty((self.N, (m + 1) * s), dtype=np.complex128)
            V[:, :s] = Q
            # Incremental QR of the block Hessenberg: Rbar accumulates the
            # triangular factor, g the transformed RHS (whose trailing block
            # IS the least-squares residual, giving the same stopping test
            # the old per-iteration lstsq computed), and rots[j] the small
            # (2s, 2s) unitary that eliminated column block j's subdiagonal.
            # O(k·s³) small work per iteration instead of a fresh O(k³)
            # lstsq + O(k²) Python rebuild of the Hessenberg.
            Rbar = np.zeros(((m + 1) * s, m * s), dtype=np.complex128)
            g = np.zeros(((m + 1) * s, s), dtype=np.complex128)
            g[:s] = beta
            rots = []
            kb = 0
            converged = False
            for k in range(m):
                total_iters += 1
                p = (k + 1) * s
                W = prec(self._aug_matmat(V[:, p - s : p]))
                Vp = V[:, :p]
                # Vp^H W = (W^H Vp)^H — conjugate the (N, s) block, never
                # the (N, p) basis.
                H1 = (W.conj().T @ Vp).conj().T
                W -= Vp @ H1
                H2 = (W.conj().T @ Vp).conj().T
                W -= Vp @ H2
                Qk, Hkk = np.linalg.qr(W)
                V[:, p : p + s] = Qk
                c = np.empty((p + s, s), dtype=np.complex128)
                c[:p] = H1 + H2
                c[p:] = Hkk
                for j, Qh in enumerate(rots):  # accumulated eliminations
                    r0 = j * s
                    c[r0 : r0 + 2 * s] = Qh.conj().T @ c[r0 : r0 + 2 * s]
                Qh, Rh = np.linalg.qr(c[p - s : p + s], mode="complete")
                rots.append(Qh)
                c[p - s : p] = Rh[:s]
                Rbar[:p, p - s : p] = c[:p]
                g[p - s : p + s] = Qh.conj().T @ g[p - s : p + s]
                kb = k + 1
                rk = np.linalg.norm(g[p : p + s], axis=0)
                if np.all(rk <= rtol * bnorms):
                    converged = True
                    break
            psz = kb * s
            Rtop = Rbar[:psz, :psz]
            diag = np.abs(np.diagonal(Rtop))
            if np.all(diag > 1e-14 * max(float(diag.max(initial=0.0)), 1.0)):
                Y = solve_triangular(Rtop, g[:psz], check_finite=False)
            else:  # (near-)breakdown: rank-deficient small system
                Y = np.linalg.lstsq(Rtop, g[:psz], rcond=None)[0]
            X = X + V[:, :psz] @ Y
            if converged:
                break
        return X, [total_iters] * s


class HMatrixSolver(BSplineSolver):
    """Distance-based hierarchical accelerator for the B-spline MoM.

    Drop-in for `BSplineSolver` (same constructor); Phase 0 only adds the
    block-evaluator plumbing. `compute_impedance` / `compute_y_matrix` still
    resolve to the dense BSplineSolver path until later phases override them,
    so the class is usable and correct from the start.
    """

    # momwire#792: `BSplineSolver`'s row in every cell but ONE. This
    # class used to inherit that row whole, on the survey's finding that
    # enrichment merely forces a dense-path fallback here rather than being
    # refused — true, and it missed the cell that is genuinely different.
    # `_refuse_buried_fast_operator` refuses a buried deck by NAME, on
    # purpose (a silent dense fallback on a buried array is an
    # out-of-memory rather than a slow answer), while the inherited row said
    # buried was served: a consumer reading the declaration was told to send
    # exactly the deck this class raises on.
    #
    # `_replace(refusals=...)` REPLACES the mapping rather than merging it —
    # the trap that lost `SinusoidalGalerkinSolver` a cell — so the parent's
    # entries are spread in by hand, including the two ground rows that
    # refuse buried before this guard is ever reached.
    #
    # `ArrayBlockSolver` inherits THIS row (no override): it is
    # `HMatrixSolver`'s solve with a different structural decomposition and
    # the same guard, on the same fills.
    capabilities = BSplineSolver.capabilities._replace(
        # Same physics as BSplineSolver, different assembly — which is the
        # pair antennaknobs#1006 opens with: nothing in a name says these two
        # are REQUIRED to agree numerically and differ only here.
        axes={**BSplineSolver.capabilities.axes, "solve_strategy": ("aca",)},
        buried=False,
        refusals={
            **BSplineSolver.capabilities.refusals,
            "buried": _BURIED_FAST_OPERATOR_REFUSAL,
        },
    )

    # ------------------------------------------------------------------
    # Shared, k-independent geometry/basis context
    # ------------------------------------------------------------------

    def _context(self):
        """Build (and memoise) the k-independent geometry + basis tables and
        the per-segment edge map that the block evaluator needs.

        Returns a dict with:
          geom, supp_seg, polys, kcl_A, wire_knots, wire_basis_global,
          seg_l, seg_r, tangents, n_basis, n_segs,
          seg_edge_id   (N_segs,)   global edge index of each segment
          seg_edge_loc  (N_segs,)   within-edge local index of each segment
          edge_arc      list[ (edge_arc_edges array) ] indexed by edge id
          basis_centroid (n_basis, 3), basis_lo/basis_hi (n_basis, 3)
          seg_a   (N_segs,)  conductor radius of each segment's wire
          edge_a  (n_edges,) conductor radius of each edge's wire
          basis_a (n_basis,) conductor radius of each basis's wire
        (the *_a tables are the per-wire radius broadcasts the block fills
        regularise observer rows with — stevenmburns/momwire#147)
        """
        cached = getattr(self, "_hm_context", None)
        if cached is not None:
            return cached

        geom = self._build_geometry()
        supp_seg, polys, kcl_A, wire_knots, wire_basis_global = (
            self._build_basis_polynomials(geom)
        )

        seg_l = geom["seg_l"]
        seg_r = geom["seg_r"]
        tangents = geom["tangents"]
        n_segs = seg_l.shape[0]
        n_basis = supp_seg.shape[0]

        # Per-segment edge map: each (wire, edge) pair gets a global edge id;
        # record which edge every global segment lives on and its local index
        # within that edge's contiguous segment run. Used to overwrite
        # same-edge pairs with the analytic static + reg moments.
        seg_edge_id = np.full(n_segs, -1, dtype=np.int64)
        seg_edge_loc = np.full(n_segs, -1, dtype=np.int64)
        edge_arc = []
        edge_a = []
        per_wire = geom["per_wire"]
        seg_off = geom["seg_offsets"]
        eid = 0
        for w in range(len(per_wire)):
            pw = per_wire[w]
            ed_off = pw["edge_offsets"]
            ed_arc = pw["edge_arc_edges"]
            base = seg_off[w]
            a_w = float(self._radius_per_wire[w])
            for i_e in range(len(ed_off) - 1):
                lo = base + ed_off[i_e]
                hi = base + ed_off[i_e + 1]
                seg_edge_id[lo:hi] = eid
                seg_edge_loc[lo:hi] = np.arange(hi - lo, dtype=np.int64)
                edge_arc.append(np.asarray(ed_arc[i_e], dtype=float))
                edge_a.append(a_w)
                eid += 1

        # Basis geometric extent: axis-aligned bounding box (and centroid)
        # of the union of the segment endpoints in the basis support. The
        # cluster tree partitions on the boxes so admissibility reflects the
        # true spatial extent of each basis function's current support.
        #
        # Wings must be masked to the basis's REAL support: on wires shorter
        # than d+1 segments, `_build_basis_polynomials` zero-pads the unused
        # supp_seg slots with global segment 0 (zero poly weight, so every
        # numeric consumer is unaffected). Trusting supp_seg raw here drags
        # every padded basis's box out to segment 0's location; on a
        # junction-split multi-wire mesh that makes all boxes overlap, kills
        # admissibility, and degenerates the H-matrix to dense (issue #137).
        basis_lo = np.empty((n_basis, 3), dtype=float)
        basis_hi = np.empty((n_basis, 3), dtype=float)
        basis_centroid = np.empty((n_basis, 3), dtype=float)
        basis_a = np.empty(n_basis, dtype=float)
        seg_a = self._seg_radius(geom)
        live_wing = np.abs(polys).sum(axis=2) > 0.0  # (n_basis, n_wings)
        for m in range(n_basis):
            segs = np.unique(supp_seg[m][live_wing[m]])
            pts = np.vstack([seg_l[segs], seg_r[segs]])
            basis_lo[m] = pts.min(axis=0)
            basis_hi[m] = pts.max(axis=0)
            basis_centroid[m] = pts.mean(axis=0)
            # Every live segment of a basis lives on one wire (bases are
            # wire-local; junctions couple through KCL, not shared bases),
            # so the first live segment pins the basis's conductor radius.
            basis_a[m] = seg_a[segs[0]]

        ctx = {
            "geom": geom,
            "supp_seg": supp_seg,
            "polys": polys,
            "kcl_A": kcl_A,
            "wire_knots": wire_knots,
            "wire_basis_global": wire_basis_global,
            "seg_l": seg_l,
            "seg_r": seg_r,
            "tangents": tangents,
            "n_basis": n_basis,
            "n_segs": n_segs,
            "seg_edge_id": seg_edge_id,
            "seg_edge_loc": seg_edge_loc,
            "edge_arc": edge_arc,
            "basis_centroid": basis_centroid,
            "basis_lo": basis_lo,
            "basis_hi": basis_hi,
            "seg_a": seg_a,
            "edge_a": np.asarray(edge_a, dtype=float),
            "basis_a": basis_a,
        }
        self._hm_context = ctx
        return ctx

    # ------------------------------------------------------------------
    # Same-edge analytic blocks (cached per k)
    # ------------------------------------------------------------------

    def _same_edge_band(self, ctx, edge_id, k, lo, hi):
        """Analytic static + regularised same-edge moments over the contiguous
        within-edge segment sub-range [lo, hi] (inclusive) of one edge. Shape
        (d+1, d+1, W, W) with W = hi - lo + 1, indexed by (local_idx - lo).

        Same formula as `BSplineSolver._build_J_blocks`'s same-edge overwrite,
        but restricted to the sub-range a block actually touches — O(W²)
        instead of the full O(N_edge²). A near block spans ~2·leaf_size
        segments, so W stays small and the same-edge fill is O(N·leaf) total
        rather than O(N²). The static moments are translation-invariant along a
        straight edge and the reg geometry depends only on arc differences, so
        the sub-range gives exactly the corresponding entries of the full
        edge block. Cached per (edge_id, lo, hi); the per-k cache is reset in
        `zblock` when k changes. `extended_kernel` is a solver-level
        constant and this cache lives on the solver, so it needs no place in
        the key.
        """
        cache = self._hm_se_cache
        key = (edge_id, lo, hi)
        blk = cache.get(key)
        if blk is not None:
            return blk
        # An edge lives on one wire, so the whole band uses that wire's
        # radius (stevenmburns/momwire#147) — and, under EK, is eligible in
        # its entirety, sub-range included (a sub-range of one straight run
        # is still one straight run).
        a = float(ctx["edge_a"][edge_id])
        d = self.degree
        ek = _EK_SAME_EDGE if self.extended_kernel else None
        sub_arc = ctx["edge_arc"][edge_id][lo : hi + 2]
        A_st = _seg_seg_static_moments(sub_arc, a, max_d=d, ek=ek)
        # SAME-EDGE (momwire#743): follows n_qp_pair_same_edge, not the
        # cross-edge knob. The offedge sites below keep self.n_qp_pair.
        # Getting this wrong makes HMatrix disagree with dense.
        A_reg = _seg_seg_reg_moments(
            sub_arc, a, k, max_d=d, n_qp=self.n_qp_pair_same_edge, ek=ek
        )
        blk = A_st + A_reg
        cache[key] = blk
        return blk

    # ------------------------------------------------------------------
    # On-demand moment sub-tensor
    # ------------------------------------------------------------------

    def _moment_subtensor(self, ctx, seg_I, seg_J, k, same_edge=True):
        """Moment tensor J_pq between the global segment lists seg_I, seg_J,
        shape (d+1, d+1, |seg_I|, |seg_J|), matching the corresponding slice
        of `BSplineSolver._build_J_blocks`.

        Off-edge pairs use the full-kernel GL quadrature. When `same_edge` is
        True, pairs that share an edge are overwritten with the analytic
        static + reg block (essential for the near-singular diagonal). When
        False, the overwrite is skipped entirely — valid for an *admissible*
        (well-separated) block, where every pair, even two segments on the
        same long wire, is far enough apart that the a²-regularised GL kernel
        is accurate (~1e-5). Skipping it is what makes far blocks cheap:
        no O(N_edge²) same-edge block is ever built for a single-wire mesh.
        """
        d = self.degree
        seg_l = ctx["seg_l"]
        seg_r = ctx["seg_r"]

        # Per-OBSERVER-row radius (stevenmburns/momwire#147); the kernel
        # helper collapses a uniform slice back to the scalar fast path.
        # The EK labels are a whole-mesh array (they have to be globally
        # comparable), restricted here to the block's own segment lists.
        Jsub = _seg_seg_full_moments_offedge(
            seg_l[seg_I],
            seg_r[seg_I],
            seg_l[seg_J],
            seg_r[seg_J],
            ctx["seg_a"][seg_I],
            k,
            d,
            self.n_qp_pair,
            ek=_ek_slice(
                self._ek_spec(ctx["geom"]) if self.extended_kernel else None,
                rows=seg_I,
                cols=seg_J,
            ),
        )

        if not same_edge:
            return Jsub

        eid = ctx["seg_edge_id"]
        loc = ctx["seg_edge_loc"]
        eid_I = eid[seg_I]
        eid_J = eid[seg_J]
        shared = np.intersect1d(eid_I, eid_J)
        for e in shared:
            rows = np.nonzero(eid_I == e)[0]
            cols = np.nonzero(eid_J == e)[0]
            li = loc[seg_I[rows]]
            lj = loc[seg_J[cols]]
            lo = int(min(li.min(), lj.min()))
            hi = int(max(li.max(), lj.max()))
            blk = self._same_edge_band(ctx, int(e), k, lo, hi)
            sub = blk[:, :, (li - lo)[:, None], (lj - lo)[None, :]]
            Jsub[:, :, rows[:, None], cols[None, :]] = sub
        return Jsub

    # ------------------------------------------------------------------
    # Restricted Z assembly (numpy reference, mirrors _assemble_Z)
    # ------------------------------------------------------------------

    def _assemble_Z_block(
        self, Jsub, supp_I_local, polys_I, supp_J_local, polys_J, td_sub
    ):
        """Galerkin-assemble Z[I][:, J] from a local moment sub-tensor.

        `supp_*_local` are the wing→local-segment index tables (values index
        into Jsub's last two axes / td_sub), `polys_*` the per-basis poly
        coefficients, `td_sub` the tangent-dot table over the local segments.
        Bit-for-bit the same arithmetic as `BSplineSolver._assemble_Z`'s numpy
        fallback, restricted to the block.
        """
        d = self.degree
        nI = supp_I_local.shape[0]
        nJ = supp_J_local.shape[0]
        Z_A = np.zeros((nI, nJ), dtype=np.complex128)
        Z_Phi = np.zeros((nI, nJ), dtype=np.complex128)
        p_vec = np.arange(1, d + 1, dtype=np.float64) if d >= 1 else None

        for a in range(d + 1):
            sm = supp_I_local[:, a]
            for b in range(d + 1):
                sn = supp_J_local[:, b]
                J_blk = Jsub[:, :, sm[:, None], sn[None, :]]
                td_blk = td_sub[sm[:, None], sn[None, :]]
                inner_A = np.einsum(
                    "mp,pPmn,nP->mn", polys_I[:, a, :], J_blk, polys_J[:, b, :]
                )
                Z_A += td_blk * inner_A
                if d >= 1:
                    deriv_m = polys_I[:, a, 1:] * p_vec[None, :]
                    deriv_n = polys_J[:, b, 1:] * p_vec[None, :]
                    J_blk_lo = J_blk[:d, :d]
                    Z_Phi += np.einsum("mp,pPmn,nP->mn", deriv_m, J_blk_lo, deriv_n)

        Z_A = 1j * self.omega * self.mu * Z_A
        Z_Phi = Z_Phi / (1j * self.omega * self.eps)
        return Z_A + Z_Phi

    # ------------------------------------------------------------------
    # Public block evaluator
    # ------------------------------------------------------------------

    def zblock(self, I, J, k=None, same_edge=True):
        """Return the dense sub-block Z[I][:, J] (shape (|I|, |J|), complex).

        `I`, `J` are 1-D integer arrays of *basis* indices. Computed on demand
        from only the segments in the supports of I and J — no full Z or full
        moment tensor is formed.

        With `same_edge=True` (default) this equals the dense bspline Z slice
        exactly. With `same_edge=False` the same-edge analytic overwrite is
        skipped — correct (~1e-5) only for *admissible* far blocks, where it
        avoids ever materialising a same-edge block; used by the ACA fill.
        """
        if k is None:
            k = self.k
        # HMatrixSolver has no medium concept (no per-segment medium, no
        # in-medium admissibility, and a fused C++ assembler taking
        # `double k`), so an in-medium k reaching here is a caller error,
        # not a capability gap to paper over — momwire#553 unit 1.
        _refuse_complex_k(k, "HMatrixSolver.zblock")
        self._refuse_buried_fast_operator("HMatrixSolver.zblock")
        ctx = self._context()
        # Reset the per-k same-edge cache when k changes.
        if getattr(self, "_hm_se_cache_k", None) != k:
            self._hm_se_cache = {}
            self._hm_se_cache_k = k

        I = np.asarray(I, dtype=np.int64)
        J = np.asarray(J, dtype=np.int64)
        supp_seg = ctx["supp_seg"]
        polys = ctx["polys"]
        tangents = ctx["tangents"]

        # Union of segments touched by the two basis index sets, with a
        # global→local remap so the moment sub-tensor and td table are small.
        seg_I = np.unique(supp_seg[I].ravel())
        seg_J = np.unique(supp_seg[J].ravel())
        loc_of_I = {int(s): i for i, s in enumerate(seg_I)}
        loc_of_J = {int(s): i for i, s in enumerate(seg_J)}

        Jsub = self._moment_subtensor(ctx, seg_I, seg_J, k, same_edge=same_edge)

        supp_I_local = np.vectorize(loc_of_I.__getitem__)(supp_seg[I])
        supp_J_local = np.vectorize(loc_of_J.__getitem__)(supp_seg[J])
        td_sub = tangents[seg_I] @ tangents[seg_J].T

        Zb = self._assemble_Z_block(
            Jsub, supp_I_local, polys[I], supp_J_local, polys[J], td_sub
        )
        # Distributed wire loading rides the free-space block (the image
        # block is subtracted separately and carries none). Exact for any
        # block partition: entry (i, j) appears in exactly one block, and
        # far/admissible blocks hold no same-wire basis overlaps anyway.
        if self._loading_active:
            Zb += self._loading_block(I, J)
        return Zb

    # ------------------------------------------------------------------
    # The near image on a horizontal edge (momwire#634)
    # ------------------------------------------------------------------

    def _near_image_edge_spans(self, geom):
        """`_near_image_edge_blocks` as (start, stop, arc, a_eff) spans.

        Memoised per geometry object: the block fills below ask this once per
        sub-block, and the answer is a property of the deck alone.
        """
        if self.ground_z is None:
            return []  # no plane, no image, nothing to correct
        cached = self._cached_near_image_spans
        if cached is not None and cached[0] is geom:
            return cached[1]
        spans = [
            (int(sl.start), int(sl.stop), arc, a_eff)
            for sl, arc, a_eff in self._near_image_edge_blocks(geom)
        ]
        self._cached_near_image_spans = (geom, spans)
        return spans

    def _near_image_edge_block_cached(self, e, arc, a_eff, k):
        """One edge's whole `(d+1, d+1, N, N)` analytic near-image block.

        The dense route builds this per edge and writes it in whole
        (`_build_J_image_blocks`). A block solver cannot: its row and column
        ranges are cluster ranges, which do not align with edge boundaries —
        the obstacle momwire#634 records.

        It dissolves once you notice the block does not have to be built to
        fit. `J_static_moment` and `_seg_seg_reg_geometry` read the edge's ARC
        alone, so entry (i, j) of this block is the same number whatever
        sub-block asks for it. So build the whole edge once, cache it per
        edge for the current k, and let each sub-block gather the rectangle
        it needs. No rectangular kernel, no alignment, and the value a
        sub-block reads cannot depend on how the partition happened to cut.

        The cache holds ONE k. `compute_impedance_swept` walks `_set_k` on
        this same object, and a block is `(d+1)^2 N^2` complex per edge, so
        keying on k would keep every frequency's blocks alive for the whole
        sweep (measured 576 KiB per k at N = 64, degree 2). A new k drops
        the old blocks; within one k every sub-block reads the same edge.
        """
        kc = complex(k)
        cached = self._cached_near_image_blocks
        if cached is None or cached[0] != kc:
            cached = (kc, {})
            self._cached_near_image_blocks = cached
        block = cached[1].get(e)
        if block is None:
            block = cached[1][e] = self._near_image_analytic_block(arc, a_eff, k)
        return block

    def _apply_near_image_analytic(self, Jsub, seg_I, seg_J, k, geom):
        """Overwrite `Jsub`'s near-image entries with the closed form.

        momwire#634. `_zblock_image` / `_zblock_image_refl` integrate every
        image pair at `n_qp_pair`, on the premise that the mirror always
        separates the image from the original. At grazing that premise fails
        — a segment's own image sits 2h away, 3.6 cm under a 2.48 m segment
        on momwire#631's deck — and the two fast solvers kept the pre-#631
        answer while the dense route was corrected (measured 79.7% apart over
        PEC on that deck before this).

        Any (observer, source) pair whose BOTH segments lie on one horizontal
        near-image edge is replaced by that edge's analytic static + reg split
        at `a_eff`, exactly as the dense route replaces the whole block. Pairs
        that straddle edges, or that sit on a non-qualifying edge, keep the
        quadrature fill.

        Mutates and returns `Jsub`.
        """
        spans = self._near_image_edge_spans(geom)
        if not spans:
            return Jsub
        for e, (s0, s1, arc, a_eff) in enumerate(spans):
            ri = np.flatnonzero((seg_I >= s0) & (seg_I < s1))
            ci = np.flatnonzero((seg_J >= s0) & (seg_J < s1))
            if ri.size == 0 or ci.size == 0:
                continue
            full = self._near_image_edge_block_cached(e, arc, a_eff, k)
            Jsub[:, :, ri[:, None], ci[None, :]] = full[
                :, :, (seg_I[ri] - s0)[:, None], (seg_J[ci] - s0)[None, :]
            ]
        return Jsub

    def _zblock_image(self, I, J, k=None):
        """Return the PEC-image sub-block for the basis pair (I, J): the real
        test bases I reacting against the trial bases J mirrored across
        ``z = ground_z`` (positions reflected, tangents z-flipped to (tx, ty,
        -tz)). The full impedance block under PEC ground is

            zblock(I, J) - _zblock_image(I, J)

        a single combined minus sign capturing both the image current's
        anti-parallel horizontal direction and the image charge sign flip (see
        `BSplineSolver.compute_impedance`). The mirror always separates the image
        from the real geometry, so every pair is off-edge — full GL quadrature
        throughout, no same-edge analytic overwrite (mirrors
        `BSplineSolver._build_J_image_blocks`)."""
        if k is None:
            k = self.k
        ctx = self._context()
        supp_seg = ctx["supp_seg"]
        polys = ctx["polys"]
        tangents = ctx["tangents"]
        seg_l = ctx["seg_l"]
        seg_r = ctx["seg_r"]
        d = self.degree

        I = np.asarray(I, dtype=np.int64)
        J = np.asarray(J, dtype=np.int64)
        seg_I = np.unique(supp_seg[I].ravel())
        seg_J = np.unique(supp_seg[J].ravel())
        loc_of_I = {int(s): i for i, s in enumerate(seg_I)}
        loc_of_J = {int(s): i for i, s in enumerate(seg_J)}

        # Observer rows are the REAL test segments — the per-observer
        # radius convention applies to the image block unchanged. EK scores
        # eligibility against the MIRRORED source geometry (`mirror=True`),
        # the block form of `BSplineSolver._build_J_image_blocks`.
        Jsub = _seg_seg_full_moments_offedge(
            seg_l[seg_I],
            seg_r[seg_I],
            self._image_positions(seg_l[seg_J]),
            self._image_positions(seg_r[seg_J]),
            ctx["seg_a"][seg_I],
            k,
            d,
            self.n_qp_pair,
            ek=_ek_slice(
                self._ek_spec(ctx["geom"], mirror=True)
                if self.extended_kernel
                else None,
                rows=seg_I,
                cols=seg_J,
            ),
        )
        self._apply_near_image_analytic(Jsub, seg_I, seg_J, k, ctx["geom"])
        supp_I_local = np.vectorize(loc_of_I.__getitem__)(supp_seg[I])
        supp_J_local = np.vectorize(loc_of_J.__getitem__)(supp_seg[J])
        td_sub = tangents[seg_I] @ self._image_tangent_dot_cols(tangents[seg_J])
        return self._assemble_Z_block(
            Jsub, supp_I_local, polys[I], supp_J_local, polys[J], td_sub
        )

    @staticmethod
    def _image_tangent_dot_cols(tangents_J):
        """The (tx, ty, -tz)-flipped trial tangents as columns, so
        ``t_I @ _image_tangent_dot_cols(t_J)`` is the image tangent-dot table
        (rows = real test, cols = image trial) — the block form of
        `BSplineSolver._image_tangent_dot`."""
        return _ground_mirror.mirror_tangents(tangents_J).T

    def _assemble_Z_block_weighted(
        self, Jsub, supp_I_local, polys_I, supp_J_local, polys_J, wA_sub, wPhi_sub
    ):
        """`_assemble_Z_block` with per-segment-pair weight tables on BOTH
        terms — the block form of `BSplineSolver._image_Z_refl`'s numpy
        fallback (wA on the A term in place of the tangent dot, wPhi on the
        charge term the PEC path leaves unweighted)."""
        d = self.degree
        nI = supp_I_local.shape[0]
        nJ = supp_J_local.shape[0]
        Z_A = np.zeros((nI, nJ), dtype=np.complex128)
        Z_Phi = np.zeros((nI, nJ), dtype=np.complex128)
        p_vec = np.arange(1, d + 1, dtype=np.float64) if d >= 1 else None

        for a in range(d + 1):
            sm = supp_I_local[:, a]
            for b in range(d + 1):
                sn = supp_J_local[:, b]
                J_blk = Jsub[:, :, sm[:, None], sn[None, :]]
                wA_blk = wA_sub[sm[:, None], sn[None, :]]
                inner_A = np.einsum(
                    "mp,pPmn,nP->mn", polys_I[:, a, :], J_blk, polys_J[:, b, :]
                )
                Z_A += wA_blk * inner_A
                if d >= 1:
                    wPhi_blk = wPhi_sub[sm[:, None], sn[None, :]]
                    deriv_m = polys_I[:, a, 1:] * p_vec[None, :]
                    deriv_n = polys_J[:, b, 1:] * p_vec[None, :]
                    J_blk_lo = J_blk[:d, :d]
                    Z_Phi += wPhi_blk * np.einsum(
                        "mp,pPmn,nP->mn", deriv_m, J_blk_lo, deriv_n
                    )

        Z_A = 1j * self.omega * self.mu * Z_A
        Z_Phi = Z_Phi / (1j * self.omega * self.eps)
        return Z_A + Z_Phi

    def _refl_weight_tables(self, ctx, seg_I, seg_J):
        """(wA, wPhi) tables over a (seg_I, seg_J) segment-index rectangle,
        from the REAL (unmirrored) geometry. O(|I|·|J|) closed-form; no
        global (N, N) table is ever built, so large-N fast-solver problems
        stay out of quadratic weight memory.

        Since momwire#429 unit 1 this is `PotentialGround.weight_windows`
        taken as ONE full-width window over a named observer set and a
        named source set — which is exactly the shape momwire#398 unit 4
        gave that producer for razor, so the block path needed no
        generalisation at all to consume it (the audit's rank-2 finding,
        confirmed). The weights this reads are ω-dependent and the object
        is built per call for that reason; ε̃ comes from the factory's own
        hoist rather than a second `eps_tilde` call here."""
        seg_mid = 0.5 * (ctx["seg_l"] + ctx["seg_r"])
        tangents = ctx["tangents"]
        ground = _potential_ground.potential_ground_for(self, ctx, self.k, self.omega)
        windows = ground.weight_windows(
            observers=(seg_mid[seg_I], tangents[seg_I]),
            sources=(seg_mid[seg_J], tangents[seg_J]),
        )
        return windows(0, seg_I.shape[0])

    def _somm_eps_c2(self):
        """(eps_t, C2) for the sommerfeld decomposition at the current
        omega. C2 = (eps-1)/(eps+1) is the exact-image coefficient: the
        C2-weighted image block is just C2 x the PEC-image block, so the
        fully-accelerated PEC image paths serve it with one scalar."""
        eps_t = _ground_refl.eps_tilde(self.ground_eps, self.omega, self.eps)
        return eps_t, (eps_t - 1.0) / (eps_t + 1.0)

    def _somm_nodes(self, ctx):
        """k-independent per-geometry quadrature tables for the rectangular
        Sommerfeld-remainder sampler: GL nodes along every segment, the
        u^p moment weights W (the quadrature dual of the J-block moments,
        so `polys` applies unchanged), and the GLOBAL grid extent — every
        rectangular block must query ONE shared grid, so the extent comes
        from all endpoint pairs, not the block's own. Memoised on the
        hm-context dict (same lifetime as the geometry)."""
        cached = ctx.get("somm_nodes")
        if cached is not None:
            return cached
        seg_l = ctx["seg_l"]
        seg_r = ctx["seg_r"]
        h = ctx["geom"]["h_per_seg"]
        d = self.degree
        gz_key = self.ground_z
        # momwire#647: the SAME keying the dense route uses. This sampler
        # took `self.n_qp_sommerfeld` raw, so the fast solvers ordered the
        # remainder on the constructor default however close the deck came
        # to the plane, while `BSplineSolver._Z_sommerfeld_remainder` keyed
        # it on the grazing height (momwire#510 / #631). At h/lambda = 1.09e-4
        # that left the two routes 1.27 apart AFTER momwire#634 corrected the
        # image; raising `n_qp_sommerfeld` by hand closed it (7.7e-9 at 96),
        # which is what said the residual was the ORDER and not the fill.
        # `_remainder_qp` is inherited and reads only geometry, so this is
        # the dense rule itself and not a second copy of it — and it is a
        # max-with-base, so a deck with nothing grazing keeps the base order
        # it always had and is bit-identical.
        q = self._remainder_qp(seg_l, seg_r, gz_key)
        xg, wg = leggauss(q)
        tq = 0.5 * (xg + 1.0)
        nodes = seg_l[:, None, :] + tq[None, :, None] * (seg_r - seg_l)[:, None, :]
        u_phys = h[:, None] * tq[None, :]
        w_node = 0.5 * h[:, None] * wg[None, :]
        W = w_node[None] * u_phys[None] ** np.arange(d + 1)[:, None, None]
        gz = self.ground_z
        r1_max = _sommerfeld.max_image_distance(seg_l, seg_r, gz)
        zmin = min(seg_l[:, 2].min(), seg_r[:, 2].min()) - gz
        # Touching (zmin == 0) is allowed since #151: the ground-junction
        # basis handles contact, and the remainder quadrature samples
        # Gauss nodes strictly interior to segments, so z+z' > 0 holds
        # even for a wire ending in the plane. Only genuinely submerged
        # geometry is rejected (already caught at geometry build too).
        if zmin < -1e-12:
            raise ValueError(
                "ground_model='sommerfeld' requires every wire at or "
                f"above ground_z (min height above plane: {zmin:.3g})"
            )
        cached = {"nodes": nodes, "W": W, "q": q, "r1_max": r1_max}
        ctx["somm_nodes"] = cached
        return cached

    def _zblock_sommerfeld_remainder(self, I, J, k=None, eps_t=None, grid_args=None):
        """Rectangular Galerkin block Q[I][:, J] of the smooth Sommerfeld
        remainder — `BSplineSolver._Z_sommerfeld_remainder` restricted to
        a basis rectangle. This is the ACA sampler for the global low-rank
        remainder term: rows keep I small, and the kernel is smooth
        everywhere (the image term absorbed the singularity), so no
        same-edge special-casing exists.

        Uses the fused C++ `sommerfeld_remainder_bspline_Q` (rectangular
        obs/src form) when available — one call does interpolate + project
        + moment-quadrature + basis-assembly, the same kernel the dense
        block uses. `grid_args` lets the ACA driver pre-marshal the grid
        once and reuse it across the O(rank) samples; falls back to the
        vectorized numpy `remainder_field_proj` path otherwise."""
        if k is None:
            k = self.k
        ctx = self._context()
        supp_seg = ctx["supp_seg"]
        polys = ctx["polys"]
        tang = ctx["tangents"]
        if eps_t is None:
            eps_t, _c2 = self._somm_eps_c2()
        sn = self._somm_nodes(ctx)
        nodes, W, q = sn["nodes"], sn["W"], sn["q"]
        grid = self._somm_grid(eps_t, sn["r1_max"])

        I = np.asarray(I, dtype=np.int64)
        J = np.asarray(J, dtype=np.int64)
        seg_I = np.unique(supp_seg[I].ravel())
        seg_J = np.unique(supp_seg[J].ravel())
        self._checkpoint()  # per sampled rectangle (ACA row/col granularity)
        loc_I = np.searchsorted(seg_I, supp_seg[I])
        loc_J = np.searchsorted(seg_J, supp_seg[J])

        if _acc is not None and hasattr(_acc, "sommerfeld_remainder_bspline_Q"):
            if grid_args is None:
                grid_args = _sommerfeld.grid_cpp_args(grid)
            return _acc.sommerfeld_remainder_bspline_Q(
                np.ascontiguousarray(nodes[seg_I], dtype=np.float64),
                np.ascontiguousarray(tang[seg_I], dtype=np.float64),
                np.ascontiguousarray(W[:, seg_I], dtype=np.float64),
                np.ascontiguousarray(nodes[seg_J], dtype=np.float64),
                np.ascontiguousarray(tang[seg_J], dtype=np.float64),
                np.ascontiguousarray(W[:, seg_J], dtype=np.float64),
                np.ascontiguousarray(loc_I, dtype=np.int64),
                np.ascontiguousarray(polys[I], dtype=np.float64),
                np.ascontiguousarray(loc_J, dtype=np.int64),
                np.ascontiguousarray(polys[J], dtype=np.float64),
                self.ground_z,
                k,
                *grid_args,
                int(self._cancel_flag),
            )

        proj = _sommerfeld.remainder_field_proj(
            nodes[seg_I].reshape(-1, 3),
            np.repeat(tang[seg_I], q, axis=0),
            nodes[seg_J].reshape(-1, 3),
            np.repeat(tang[seg_J], q, axis=0),
            self.ground_z,
            k,
            grid,
        )
        fq = proj.reshape(seg_I.size, q, seg_J.size, q)
        Jf = np.einsum("piq,iqjr,Pjr->pPij", W[:, seg_I], fq, W[:, seg_J])

        d = self.degree
        pI = polys[I]
        pJ = polys[J]
        Q = np.zeros((I.size, J.size), dtype=np.complex128)
        for a in range(d + 1):
            sm = loc_I[:, a]
            for b in range(d + 1):
                sc = loc_J[:, b]
                J_blk = Jf[:, :, sm[:, None], sc[None, :]]
                Q += np.einsum("mp,pPmn,nP->mn", pI[:, a, :], J_blk, pJ[:, b, :])
        return Q

    def _sommerfeld_global_lowrank(self, k, eps_t):
        """One global ACA factorization (U, V) of the remainder block Q
        over ALL bases. Q is smooth everywhere, hence globally low rank;
        carrying it as a single extra low-rank term leaves the block
        partition, matvec, and preconditioners untouched (the operator
        subtracts it: Z = free - C2*image - Q). Rank is recorded on
        `self._last_somm_rank` — if it ever grows past ~50 at native
        sizes, per-pair remainder fills become the better trade
        (docs/sommerfeld-everywhere-plan.md Phase 3 decision point)."""
        ctx = self._context()
        n = ctx["n_basis"]
        idx = np.arange(n, dtype=np.int64)

        # Marshal the grid once and reuse it across every ACA sample.
        grid_args = None
        if _acc is not None and hasattr(_acc, "sommerfeld_remainder_bspline_Q"):
            sn = self._somm_nodes(ctx)
            grid = self._somm_grid(eps_t, sn["r1_max"])
            grid_args = _sommerfeld.grid_cpp_args(grid)

        def get_row(i):
            return self._zblock_sommerfeld_remainder(
                idx[i : i + 1], idx, k=k, eps_t=eps_t, grid_args=grid_args
            ).ravel()

        def get_col(j):
            return self._zblock_sommerfeld_remainder(
                idx, idx[j : j + 1], k=k, eps_t=eps_t, grid_args=grid_args
            ).ravel()

        U, V = aca_partial(get_row, get_col, n, n, tol=self.aca_tol)
        self._last_somm_rank = U.shape[1]
        return U, V

    def _zblock_image_refl(self, I, J, k=None):
        """Fresnel-weighted image sub-block for `ground_eps` — the
        reflection-coefficient counterpart of `_zblock_image`, subtracted
        from `zblock(I, J)` with the same single global minus sign. Same
        image moment fill; only the assembly weights differ."""
        if k is None:
            k = self.k
        ctx = self._context()
        supp_seg = ctx["supp_seg"]
        polys = ctx["polys"]
        seg_l = ctx["seg_l"]
        seg_r = ctx["seg_r"]
        d = self.degree

        I = np.asarray(I, dtype=np.int64)
        J = np.asarray(J, dtype=np.int64)
        seg_I = np.unique(supp_seg[I].ravel())
        seg_J = np.unique(supp_seg[J].ravel())
        loc_of_I = {int(s): i for i, s in enumerate(seg_I)}
        loc_of_J = {int(s): i for i, s in enumerate(seg_J)}

        # Same mirrored-source eligibility as `_zblock_image`, and live since
        # momwire#269 lifted the `extended_kernel` + `ground_eps` refusal:
        # this is the near-block image fill of every grounded H-matrix solve
        # (and the far-block one on a build without the fused refl EK twin).
        # The Fresnel weights below are applied after the fill, so the moment
        # kernel choice is independent of them.
        Jsub = _seg_seg_full_moments_offedge(
            seg_l[seg_I],
            seg_r[seg_I],
            self._image_positions(seg_l[seg_J]),
            self._image_positions(seg_r[seg_J]),
            ctx["seg_a"][seg_I],
            k,
            d,
            self.n_qp_pair,
            ek=_ek_slice(
                self._ek_spec(ctx["geom"], mirror=True)
                if self.extended_kernel
                else None,
                rows=seg_I,
                cols=seg_J,
            ),
        )
        self._apply_near_image_analytic(Jsub, seg_I, seg_J, k, ctx["geom"])
        supp_I_local = np.vectorize(loc_of_I.__getitem__)(supp_seg[I])
        supp_J_local = np.vectorize(loc_of_J.__getitem__)(supp_seg[J])
        wA_sub, wPhi_sub = self._refl_weight_tables(ctx, seg_I, seg_J)
        return self._assemble_Z_block_weighted(
            Jsub, supp_I_local, polys[I], supp_J_local, polys[J], wA_sub, wPhi_sub
        )

    # ------------------------------------------------------------------
    # Cluster / block tree (Phase 1)
    # ------------------------------------------------------------------

    def build_partition(self, eta=None, leaf_size=None):
        """Build (and memoise) the block-cluster partition of the n_basis x
        n_basis impedance matrix into far (admissible, compressible) and near
        (dense) leaf blocks.

        Returns a dict with the cluster-tree `root`, the `far`/`near` block
        lists (each a list of (Cluster, Cluster) pairs), and `stats`.
        """
        if eta is None:
            eta = self.aca_eta
        if leaf_size is None:
            leaf_size = self.aca_leaf_size

        cached = getattr(self, "_hm_partition", None)
        if (
            cached is not None
            and cached["eta"] == eta
            and (cached["leaf_size"] == leaf_size)
        ):
            return cached

        ctx = self._context()
        n = ctx["n_basis"]
        root = build_cluster_tree(
            np.arange(n), ctx["basis_lo"], ctx["basis_hi"], leaf_size=leaf_size
        )
        far, near = build_block_tree(root, root, eta)
        stats = partition_stats(n, far, near)
        part = {
            "root": root,
            "far": far,
            "near": near,
            "stats": stats,
            "eta": eta,
            "leaf_size": leaf_size,
        }
        self._hm_partition = part
        return part

    # ------------------------------------------------------------------
    # H-matrix assembly (Phase 2): dense near blocks + ACA far blocks
    # ------------------------------------------------------------------

    def build_hmatrix(self, eta=None, leaf_size=None, tol=None, k=None):
        """Assemble the impedance matrix as an `HMatrix`: near blocks dense
        (via `zblock`, including the same-edge analytic path), far blocks
        compressed by partial-pivoted ACA on the off-edge kernel.

        A far block whose ACA factors would cost as much as the dense block
        falls back to dense storage, so the H-matvec is never worse than the
        dense block-by-block product.

        Under PEC ground every block carries the per-block image term
        (`Z_free − Z_image`): near blocks subtract the dense `_zblock_image`,
        far blocks fold the image into the ACA target via
        `_offedge_aca_evaluators`. The `HMatrix` container, matvec, and
        preconditioner are ground-agnostic — they bake whatever block values
        they are handed — and the cluster partition is built on the real
        geometry, so it is reused unchanged.
        """
        if tol is None:
            tol = self.aca_tol
        if k is None:
            k = self.k
        self._refuse_buried_fast_operator("HMatrixSolver.build_hmatrix")
        part = self.build_partition(eta=eta, leaf_size=leaf_size)
        ctx = self._context()
        n = ctx["n_basis"]
        grounded = self.ground_z is not None
        somm = (
            grounded
            and self.ground_eps is not None
            and (self.ground_model == "sommerfeld")
        )
        if somm:
            eps_t, c2 = self._somm_eps_c2()

        near_blocks = []
        for s, t in part["near"]:
            self._checkpoint()  # per near-block dense fill
            I, J = s.indices, t.indices
            D = self.zblock(I, J, k=k)
            if grounded:
                if somm:
                    # C2-weighted exact image = C2 x the PEC-image block;
                    # the smooth remainder rides the single global
                    # low-rank term appended below.
                    D = D - c2 * self._zblock_image(I, J, k=k)
                elif self.ground_eps is not None:
                    D = D - self._zblock_image_refl(I, J, k=k)
                else:
                    D = D - self._zblock_image(I, J, k=k)
            near_blocks.append((I, J, D))

        use_accel = (
            _HAVE_OFFEDGE_BLOCK_ACCEL
            and self.degree <= _OFFEDGE_BLOCK_ACCEL_MAX_D
            and self.hmatrix_use_accel
            # momwire#769: the off-edge block kernels carry the same capped
            # scratch as the dense pair kernels.
            and self._accel_serves_n_qp_pair
        )

        far_blocks = []
        precond_extra = []  # first-ring far blocks, dense, for the preconditioner
        p_eta = self.precond_eta
        for s, t in part["far"]:
            self._checkpoint()  # per far-block ACA build
            I, J = s.indices, t.indices
            mI, nJ = I.size, J.size

            get_row, get_col, dense = self._offedge_aca_evaluators(
                ctx, I, J, k, use_accel
            )

            U, V = aca_partial(get_row, get_col, mI, nJ, tol=tol)
            r = U.shape[1]
            if r * (mI + nJ) >= mI * nJ:
                # No compression — store dense (off-edge kernel; the block is
                # admissible so the same-edge analytic path is unnecessary).
                near_blocks.append((I, J, dense()))
            else:
                far_blocks.append((I, J, U, V))
                # "First ring": leaf-scale blocks adjacent to the near band —
                # admissible for the operator but not at the tighter
                # precond_eta. Fold a dense reconstruction into the
                # preconditioner (free: reuses the low-rank factors). The
                # leaf-size cap keeps it a *thin* ring: large far blocks (big
                # well-separated clusters) are genuinely far and excluded, so
                # the preconditioner stays sparse.
                if (
                    p_eta < self.aca_eta
                    and max(mI, nJ) <= self.aca_leaf_size
                    and not admissible(s, t, p_eta)
                ):
                    precond_extra.append((I, J, U @ V))

        if somm:
            # The smooth remainder as ONE extra global low-rank far block
            # (Z subtracts Q, so the factors carry the minus). Not in the
            # near list, so the GMRES preconditioner ignores it — a smooth
            # perturbation it converges through.
            U, V = self._sommerfeld_global_lowrank(k, eps_t)
            idx = np.arange(n, dtype=np.int64)
            far_blocks.append((idx, idx, U, -V))

        return HMatrix(
            n, near_blocks, far_blocks, precond_extra=precond_extra, cancel=self._cancel
        )

    def _gl01(self):
        """Gauss-Legendre nodes/weights mapped to [0, 1] (cached)."""
        cached = getattr(self, "_hm_gl01", None)
        if cached is None:
            xi, w = leggauss(self.n_qp_pair)
            cached = (
                np.ascontiguousarray(0.5 * (xi + 1.0)),
                np.ascontiguousarray(0.5 * w),
            )
            self._hm_gl01 = cached
        return cached

    def _offedge_block_evaluators(self, ctx, I, J, k, mirror_J=False, refl=False):
        """Build (get_row, get_col, dense) closures for an admissible far block
        backed by the fused C++ off-edge assembler.

        The C++ assembler regularises with ONE scalar a²; under mixed
        per-wire radii (stevenmburns/momwire#147) each observer basis row
        uses its own wire's radius, so the block dispatches one uniform
        sub-evaluator per constant-radius row group and scatters the rows
        back into block order. Bases are wire-contiguous, so a cluster
        spans few radii; a uniform block (every scalar-radius solve) takes
        the single-call fast path unchanged.

        Under `extended_kernel` (momwire#270 unit 3) nothing here needs to
        split the EK labels alongside `a`, unlike `_seg_seg_full_moments_
        offedge`'s own per-row-run recursion (which slices `ek.group_i` by
        hand because it is handed one whole-mesh array up front): each
        `_offedge_block_evaluators_uniform` sub-call rebuilds its own
        `segI`/`segJ` union from the `I[rows]`/`J` it is actually given and
        re-derives `_ek_spec(...)` from that, so the labels a row group sees
        are already restricted correctly. A row-group boundary is drawn
        purely from `a`, and eligible pairs share a radius by construction,
        so a boundary never splits a coaxial group — same invariant unit 2's
        own recursion relies on.
        """
        aI = ctx["basis_a"][np.asarray(I, dtype=np.int64)]
        if np.all(aI == aI[0]):
            return self._offedge_block_evaluators_uniform(
                ctx, I, J, k, float(aI[0]) ** 2, mirror_J=mirror_J, refl=refl
            )

        nI, nJ = I.size, J.size
        groups = []
        grp_of = np.empty(nI, dtype=np.int64)
        loc_of = np.empty(nI, dtype=np.int64)
        for a_val in np.unique(aI):
            rows = np.flatnonzero(aI == a_val)
            row_g, col_g, dense_g = self._offedge_block_evaluators_uniform(
                ctx, I[rows], J, k, float(a_val) ** 2, mirror_J=mirror_J, refl=refl
            )
            grp_of[rows] = len(groups)
            loc_of[rows] = np.arange(rows.size)
            groups.append((rows, row_g, col_g, dense_g))

        def get_row(i):
            return groups[grp_of[i]][1](loc_of[i])

        def get_col(j):
            out = np.empty(nI, dtype=np.complex128)
            for rows, _row_g, col_g, _dense_g in groups:
                out[rows] = col_g(j)
            return out

        def dense():
            out = np.empty((nI, nJ), dtype=np.complex128)
            for rows, _row_g, _col_g, dense_g in groups:
                out[rows] = dense_g()
            return out

        return get_row, get_col, dense

    def _offedge_block_evaluators_uniform(self, ctx, I, J, k, a2, mirror_J, refl):
        """`_offedge_block_evaluators` for a block whose observer bases all
        share the conductor radius √a2 — the fused C++ assembler
        `bspline_assemble_offedge_block` takes that single scalar.

        The block-wide I/J segment unions and local support maps are resolved
        once here; each row/column call passes only the single basis it needs
        on its own axis (so the C++ side never precomputes positions for unused
        segments) against the precomputed full opposite axis. `_call`'s first
        two arguments are always the GLOBAL segment ids that call's I-side and
        J-side geometry arrays were built from (`segI`/`segJ` for the block-
        wide axis, the single basis's own `seg_i`/`seg_j` for the other) — the
        EK branch below needs them to slice the group labels the same way,
        since a per-call fresh I/J-side rebuild (get_row's `seg_i`, get_col's
        `seg_j`) is not indexed by `segI`/`segJ` at all.

        With `mirror_J=True` the trial (J) segment endpoints are reflected across
        ``z = ground_z`` and their tangents z-flipped, so the kernel's internal R
        distances and tangent dot products reproduce the PEC-image reaction — the
        C++ counterpart of `_zblock_image` (the result is *subtracted* from the
        free-space block). The C++ assembler uses the trial tangents only through
        the dot product, so flipping their z is exactly the image-current sign
        flip. Group labels do not need a parallel mirror step: `_ek_spec(...,
        mirror=mirror_J)` already returns the JOINT real+image labelling (the
        #249 `_ek_axis_labels` scan), keyed by the REAL global segment ids —
        the same ids `seg_l`/`seg_r`/`tangents` are indexed by before the
        `mirror_pos`/`mirror_tan` transform is applied for the kernel call.

        With `refl=True` (implies the mirrored-J image; `ground_eps` set) the
        closures call `bspline_assemble_offedge_block_refl` instead: the same
        pre-mirrored inputs, with the Fresnel dyad computed in-kernel from
        ε̃ and the Φ weight as w_Φ = c0 + c1·ρ_v — the C++ counterpart of
        `_zblock_image_refl`. Under `extended_kernel` it calls that
        assembler's own EK twin, `bspline_assemble_offedge_block_refl_ek`
        (momwire#269), which takes the same group labels as the free-space
        twin ALONGSIDE the Fresnel arguments — the dyad is a per-pair scalar
        applied after the Galerkin contraction and the coaxial factor
        multiplies G before it, so the two compose without interacting.
        `_offedge_aca_evaluators` gates on the twin's presence before asking
        for `refl=True` under EK, so a build without it never lands on the
        reduced refl assembler here.
        """
        # momwire#270 unit 3 / #269: the fused block assembler's EK twins.
        # `ek` is the whole-mesh (group_i, group_j) label spec — sliced per
        # call below, not once here, because get_row/get_col rebuild their
        # single-basis side fresh each call (see the docstring).
        ek = (
            self._ek_spec(ctx["geom"], mirror=mirror_J or refl)
            if self.extended_kernel
            else None
        )
        have_twin = (
            _HAVE_OFFEDGE_BLOCK_REFL_EK_ACCEL if refl else _HAVE_OFFEDGE_BLOCK_EK_ACCEL
        )
        use_ek_accel = ek is not None and have_twin
        a_ek = float(_ek_radius(ek, math.sqrt(a2))) if use_ek_accel else 0.0

        def _ek_groups(seg_ids_I, seg_ids_J):
            return (
                np.ascontiguousarray(ek.group_i[seg_ids_I], dtype=np.int64),
                np.ascontiguousarray(ek.group_j[seg_ids_J], dtype=np.int64),
            )

        if refl:
            mirror_J = True
            eps_t = _ground_refl.eps_tilde(self.ground_eps, self.omega, self.eps)
            phi_c0, phi_c1 = _ground_refl.phi_mode_coeffs(self.ground_phi_mode, eps_t)
            if use_ek_accel:

                def _call(seg_ids_I, seg_ids_J, *args):
                    group_i, group_j = _ek_groups(seg_ids_I, seg_ids_J)
                    return _acc.bspline_assemble_offedge_block_refl_ek(
                        *args,
                        group_i,
                        group_j,
                        a_ek,
                        eps_t,
                        phi_c0,
                        phi_c1,
                        self._cancel_flag,
                    )
            else:

                def _call(seg_ids_I, seg_ids_J, *args):
                    del seg_ids_I, seg_ids_J
                    return _acc.bspline_assemble_offedge_block_refl(
                        *args, eps_t, phi_c0, phi_c1, self._cancel_flag
                    )
        else:
            if use_ek_accel:

                def _call(seg_ids_I, seg_ids_J, *args):
                    group_i, group_j = _ek_groups(seg_ids_I, seg_ids_J)
                    return _acc.bspline_assemble_offedge_block_ek(
                        *args, group_i, group_j, a_ek, self._cancel_flag
                    )
            else:

                def _call(seg_ids_I, seg_ids_J, *args):
                    del seg_ids_I, seg_ids_J
                    return _acc.bspline_assemble_offedge_block(*args, self._cancel_flag)

        supp_seg = ctx["supp_seg"]
        polys = ctx["polys"]
        seg_l = ctx["seg_l"]
        seg_r = ctx["seg_r"]
        tangents = ctx["tangents"]
        d = self.degree
        glt, glw = self._gl01()
        omega, eps, mu = self.omega, self.eps, self.mu

        def mirror_pos(p):
            return self._image_positions(p) if mirror_J else p

        def mirror_tan(t):
            return _ground_mirror.mirror_tangents(t) if mirror_J else t

        segI = np.unique(supp_seg[I].ravel())
        segJ = np.unique(supp_seg[J].ravel())
        sIl = np.searchsorted(segI, supp_seg[I]).astype(np.int64)
        sJl = np.searchsorted(segJ, supp_seg[J]).astype(np.int64)
        pI = np.ascontiguousarray(polys[I])
        pJ = np.ascontiguousarray(polys[J])
        slI, srI, tI = seg_l[segI], seg_r[segI], tangents[segI]
        slJ, srJ, tJ = (
            mirror_pos(seg_l[segJ]),
            mirror_pos(seg_r[segJ]),
            mirror_tan(tangents[segJ]),
        )
        one_supp = np.arange(d + 1, dtype=np.int64)[None, :]

        def get_row(i):
            seg_i = supp_seg[I[i]]
            return _call(
                seg_i,
                segJ,
                one_supp,
                polys[I[i]][None],
                seg_l[seg_i],
                seg_r[seg_i],
                tangents[seg_i],
                sJl,
                pJ,
                slJ,
                srJ,
                tJ,
                a2,
                k,
                omega,
                eps,
                mu,
                d,
                glt,
                glw,
            ).ravel()

        def get_col(j):
            seg_j = supp_seg[J[j]]
            return _call(
                segI,
                seg_j,
                sIl,
                pI,
                slI,
                srI,
                tI,
                one_supp,
                polys[J[j]][None],
                mirror_pos(seg_l[seg_j]),
                mirror_pos(seg_r[seg_j]),
                mirror_tan(tangents[seg_j]),
                a2,
                k,
                omega,
                eps,
                mu,
                d,
                glt,
                glw,
            ).ravel()

        def dense():
            return _call(
                segI,
                segJ,
                sIl,
                pI,
                slI,
                srI,
                tI,
                sJl,
                pJ,
                slJ,
                srJ,
                tJ,
                a2,
                k,
                omega,
                eps,
                mu,
                d,
                glt,
                glw,
            )

        return get_row, get_col, dense

    def _offedge_aca_evaluators(self, ctx, I, J, k, use_accel):
        """`(get_row, get_col, dense)` for the off-edge block Z[I][:, J] — the
        ACA-fill interface for an admissible far block (or a distinct array-
        element pair). "Off-edge" means no same-edge analytic overwrite, valid
        because the clusters are well separated; the C++ assembler is used when
        available, else the numpy `zblock(..., same_edge=False)` fallback.

        Under PEC ground the block is the *grounded* off-edge block,
        `Z_free − Z_image` (real cluster I against J, plus I against J's mirror
        image), with the image folded into every evaluator so a single ACA
        compresses the combined block — one factor pair, leaving the matvec and
        preconditioner unchanged. Rank rises modestly (real + image content); a
        block whose combined rank no longer compresses falls back to dense in
        the caller, which is correct (the image stays as low-rank as the real
        block for an antenna above the plane — reflection only increases the
        cluster separation).

        Under `extended_kernel` the dispatch is capability-gated rather than
        forced off outright (momwire#270 unit 3): `_offedge_block_evaluators`
        /`_offedge_block_evaluators_uniform` build the whole-mesh `_ek_spec`
        and pass its per-segment labels through to the fused
        `bspline_assemble_offedge_block_ek` twin whenever the extension has
        it (`_HAVE_OFFEDGE_BLOCK_EK_ACCEL`); only a build that LACKS the twin
        still falls back to the numpy `zblock(..., same_edge=False)` branch
        below, which routes through `_moment_subtensor` and has been
        EK-aware since #249 (falling back to the REDUCED fused assembler
        would be silently wrong under EK, not merely slow, so the gate is
        "twin present" rather than "EK off"). Every ArrayBlock coupling fill
        (`_coupling_aca`, `_build_lattice_operator`) comes through here, so
        they inherit the decision rather than repeating it."""
        use_accel = (
            use_accel
            and (not self.extended_kernel or _HAVE_OFFEDGE_BLOCK_EK_ACCEL)
            and self._accel_serves_n_qp_pair
        )
        if use_accel:
            row_f, col_f, dense_f = self._offedge_block_evaluators(ctx, I, J, k)
        else:

            def row_f(i):
                return self.zblock(I[i : i + 1], J, k=k, same_edge=False).ravel()

            def col_f(j):
                return self.zblock(I, J[j : j + 1], k=k, same_edge=False).ravel()

            def dense_f():
                return self.zblock(I, J, k=k, same_edge=False)

        if self.ground_z is None:
            return row_f, col_f, dense_f

        # Sommerfeld: the singular ground part is C2 x the PEC image (the
        # smooth remainder lives in the operator's single global low-rank
        # term, NOT in per-block fills), so the PEC-image evaluators serve
        # it with one scalar. Refl-coef keeps its Fresnel-weighted fills.
        somm = self.ground_eps is not None and self.ground_model == "sommerfeld"
        refl = self.ground_eps is not None and not somm
        scale = self._somm_eps_c2()[1] if somm else 1.0
        # The refl image needs its OWN twin under EK (momwire#269) — the
        # reduced `_refl` assembler would be silently wrong, not merely
        # slow, exactly as `use_accel`'s own EK gate above.
        refl_accel = _HAVE_OFFEDGE_BLOCK_REFL_ACCEL and (
            not self.extended_kernel or _HAVE_OFFEDGE_BLOCK_REFL_EK_ACCEL
        )
        if use_accel and (refl_accel or not refl):
            row_i, col_i, dense_i = self._offedge_block_evaluators(
                ctx, I, J, k, mirror_J=True, refl=refl
            )
        elif refl:

            def row_i(i):
                return self._zblock_image_refl(I[i : i + 1], J, k=k).ravel()

            def col_i(j):
                return self._zblock_image_refl(I, J[j : j + 1], k=k).ravel()

            def dense_i():
                return self._zblock_image_refl(I, J, k=k)

        else:

            def row_i(i):
                return self._zblock_image(I[i : i + 1], J, k=k).ravel()

            def col_i(j):
                return self._zblock_image(I, J[j : j + 1], k=k).ravel()

            def dense_i():
                return self._zblock_image(I, J, k=k)

        def get_row(i):
            return row_f(i) - scale * row_i(i)

        def get_col(j):
            return col_f(j) - scale * col_i(j)

        def dense():
            return dense_f() - scale * dense_i()

        return get_row, get_col, dense

    # ------------------------------------------------------------------
    # Iterative solve (Phase 3): GMRES on the H-matvec + near-field
    # preconditioner, KCL constraints via the augmented saddle system
    # ------------------------------------------------------------------

    def _near_sparse(self, H, n):
        """Assemble the near-field approximation of Z into one sparse (n, n)
        matrix for the GMRES preconditioner: the operator's dense near blocks
        plus the first-ring far blocks (H.precond_extra), the latter folded in
        as dense reconstructions of their low-rank factors to give a stronger
        preconditioner than the operator's own near band."""
        rows, cols, data = [], [], []
        for I, J, D in H.near + H.precond_extra:
            rr = np.repeat(I, J.size)
            cc = np.tile(J, I.size)
            rows.append(rr)
            cols.append(cc)
            data.append(D.ravel())
        if not rows:
            return sp.csc_matrix((n, n), dtype=np.complex128)
        return sp.coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n),
        ).tocsc()

    def _make_preconditioner(self, H, kcl_A):
        """Build the augmented near-field preconditioner for operator `H`. The
        generic H-matrix path uses a single sparse LU; subclasses with block
        structure (e.g. `ArrayBlockSolver`) override this with a cheaper
        block-wise factorisation."""
        return _SparseAugPrecond(self._near_sparse(H, H.n), kcl_A)

    def _factored_solve(self, H, kcl_A):
        """The augmented preconditioner factorisation for operator `H`, built
        once and cached on `H` itself. Because the factorisation depends only
        on `H` (its near blocks) and the KCL rows — not on the RHS or the
        excitation — caching it on `H` lets a *reused* operator (e.g. an
        animation phase sweep, where geometry and Z are fixed and only the RHS
        changes) skip the factorisation entirely. A freshly-built `H` (the
        generic H-matrix path) just factors once per solve as before.

        The cached factorisation is keyed on the constraint rows themselves:
        `ArrayBlockSolver`'s operator cache is junction-port-blind (ports do
        not change the basis set or Z), so the same operator can be handed
        different constraint sets across solvers."""
        fac = getattr(H, "_factored", None)
        if fac is None or not _same_constraints(fac.kcl_A, kcl_A):
            fac = _AugmentedFactoredSolve(H, kcl_A, self._make_preconditioner(H, kcl_A))
            H._factored = fac
        return fac

    def _solve_hmatrix(self, H, kcl_A, B):
        """Solve the constrained system  [Z A^T; A 0][x; λ] = [b; 0]  for each
        RHS column of B (n, nrhs), with Z applied via the H-matvec and a
        sparse-LU near-field preconditioner. Returns X (n, nrhs).

        With no junctions (kcl_A empty) this is a plain GMRES on Z.
        """
        fac = self._factored_solve(H, kcl_A)
        X, self._last_solve_iters = fac.solve(B, self.solve_tol)
        return X

    def _hmatrix_unsupported(self):
        """The H-matrix path supports free space, PEC ground (per-block
        image term folded into the near/far block fill — see
        `build_hmatrix` and `_offedge_aca_evaluators`), the reflection-
        coefficient finite ground (`ground_eps`, Phase 5: near blocks via
        `_zblock_image_refl`, far blocks via the in-kernel Fresnel
        weighting in `bspline_assemble_offedge_block_refl`), and the
        Sommerfeld finite ground (C2-scaled PEC-image blocks + one global
        low-rank remainder term — docs/sommerfeld-everywhere-plan.md
        Phase 3). Falls back to the dense path only for singular
        enrichment (its image reaction isn't implemented; the constructor
        already forbids enrichment + ground).

        A BURIED wire is NOT on that list and does not fall back either — it
        refuses, in `zblock` and in `build_hmatrix`, by name. Falling back
        would be the wrong kindness: this class exists for decks too large
        for the dense fill, so a silent dense route on a buried array is an
        out-of-memory rather than a slow answer (momwire#553 U5)."""
        return self.use_singular_enrichment

    def _refuse_buried_fast_operator(self, who):
        """The fast operator has no medium — momwire#553 U5.

        U1 put `_refuse_complex_k` on every fill it did not widen, and that
        guard reads the wavenumber it is handed. A buried deck hands this
        class a REAL k₀ and a geometry whose lower half needs k_m, so the
        wavenumber guard cannot see it: the tell is the geometry, and the
        refusal has to be spelled here. Nothing about the H-matrix carries a
        per-segment medium — admissibility is a purely geometric distance
        test, the fused near/far block kernels take a `double k`, and the
        Sommerfeld remainder is ONE global low-rank term over one grid.
        """
        if self.ground_z is not None and self._has_buried_wires():
            raise NotImplementedError(f"{who}: {_BURIED_FAST_OPERATOR_REFUSAL}")

    def _build_operator(self):
        """Build the fast operator the constrained solve runs GMRES on. The
        generic accelerator returns its H-matrix; subclasses with a different
        structural decomposition (e.g. `ArrayBlockSolver`) override this and
        feed the result through the same `_solve_hmatrix` machinery (the solve
        only needs `.n`, `.matvec`, and `.near`/`.precond_extra`)."""
        return self.build_hmatrix()

    def compute_y_matrix(self):
        """Short-circuit admittance matrix — the `y` field of
        `compute_port_solution()` and nothing else (#232)."""
        return self.compute_port_solution().y

    def compute_port_solution(self, same_edge_prep=None):
        """Solve every port from ONE operator fill and ONE block-GMRES group.

        Same contract as `BSplineSolver.compute_port_solution` — ports are
        [gap feeds…, junction ports…], `coeffs` column j is the amplitude
        vector for a 1 V drive at port j, `y` is identical to
        `compute_y_matrix()` — but the constrained solve is the iterative one:
        the H-matrix (or, on `ArrayBlockSolver`, the lattice operator) is built
        once and every port column is carried through one factored-solve group,
        so the port count costs back-substitutions, not fills. Columns are
        converged to `solve_tol`, not to machine precision.

        Falls back to the dense B-spline path exactly where
        `compute_y_matrix` does (singular enrichment), including its column
        conventions.

        `basis` is an opaque handle, stable across the ports of this one
        solution and NOT across solves.

        `same_edge_prep` rides through to the dense fallback for the same
        reason `compute_impedance` accepts it — the dense swept loop calls
        back in with the sweep's hoisted same-edge block — and is ignored on
        the accelerated path, which builds its own blocks.
        """
        if self._hmatrix_unsupported():
            return super().compute_port_solution(same_edge_prep=same_edge_prep)
        ctx = self._context()
        geom = ctx["geom"]
        n = ctx["n_basis"]
        H = self._build_operator()
        # Junction-port columns (#172 dense contract, lifted onto the
        # iterative paths by #234): the ported junction's KCL row leaves the
        # constraint set and becomes a port column, drive vector == readout
        # vector as for a gap feed, so the mixed Y stays symmetric. Built by
        # the dense family's helper so the two never diverge (#252).
        B, kcl_con = self._port_columns(
            geom, ctx["wire_knots"], ctx["wire_basis_global"], n, ctx["kcl_A"]
        )
        X = self._solve_hmatrix(H, kcl_con, B)
        self._hmatrix = H
        Y = B.T @ X
        return PortSolution(
            y=Y,
            coeffs=X,
            port_currents=Y,  # the same object: the readout IS the Y matrix
            basis=_SplineBasis(
                geom=geom,
                supp_seg=ctx["supp_seg"],
                polys=ctx["polys"],
                wire_knots=ctx["wire_knots"],
                wire_basis_global=ctx["wire_basis_global"],
                n_poly=n,
            ),
        )

    def compute_impedance(self, same_edge_prep=None):
        """Driving-point impedance on the accelerated operator.

        `same_edge_prep` is the dense sweep's hoisted same-edge reg-moment
        block for this k. The accelerated fill builds its own blocks and has
        no use for it, but the signature has to accept it: on a configuration
        the accelerator does not serve, `compute_impedance_swept` hands the
        whole sweep to the dense base, whose per-k loop calls back into THIS
        method with the hoist. Before it did, an enriched sweep on an
        H-matrix solver raised `TypeError` from inside the base sweep.
        """
        if self._hmatrix_unsupported():
            return super().compute_impedance(same_edge_prep=same_edge_prep)
        ctx = self._context()
        geom = ctx["geom"]
        n = ctx["n_basis"]
        H = self._build_operator()

        v_per_feed = []
        for w_i, arc_i, _v in self.feeds:
            arc_at_knot = geom["per_wire"][w_i]["arc_at_knot"]
            s_f_i = arc_i if arc_i is not None else arc_at_knot[-1] / 2.0
            v_per_feed.append(
                self._build_source_vector(
                    geom,
                    ctx["wire_knots"],
                    ctx["wire_basis_global"],
                    n,
                    wi=w_i,
                    s_f=s_f_i,
                )
            )
        voltages = np.array([v for _, _, v in self.feeds], dtype=np.complex128)
        v = np.zeros(n, dtype=np.complex128)
        for V_i, v_i in zip(voltages, v_per_feed):
            v += V_i * v_i

        # Junction ports (#234). A driven port row differs from a zero
        # constraint row only in where it lands: the constraint set keeps the
        # rows whose outflow is forced to zero, while a port row multiplies
        # its voltage into the RHS (its Lagrange multiplier is *given*, not
        # solved for) and reads its current back off the same row. The
        # augmented GMRES therefore needs no port concept — it is handed the
        # shrunken constraint matrix and an already-driven RHS.
        kcl_con, port_A, port_V = self._split_kcl_ports(ctx["kcl_A"])
        v += port_V @ port_A
        # Node gaps (#305): same additive drive/readout as the dense
        # family's `_feed_drive_and_readout` — without this the accelerated
        # route solved a node-gap-driven deck with a ZERO drive (empty z,
        # silent) while the dense fallback answered correctly.
        gap_cols, gap_V = self._node_gap_columns(ctx["wire_basis_global"], n)
        v += gap_cols @ gap_V
        port_vectors = (
            v_per_feed
            + list(port_A)
            + [gap_cols[:, p] for p in range(gap_cols.shape[1])]
        )
        all_voltages = np.concatenate([voltages, port_V, gap_V])

        coeffs = self._solve_hmatrix(H, kcl_con, v[:, None])[:, 0]
        self._hmatrix = H

        currents = np.array([u @ coeffs for u in port_vectors], dtype=np.complex128)
        z_per = all_voltages / currents
        z = z_per[0] if z_per.shape[0] == 1 else z_per
        return z, coeffs

    # ------------------------------------------------------------------
    # Size-based swept dispatch (issue #262)
    # ------------------------------------------------------------------

    #: Basis-count ceiling at or below which BOTH swept entry points hand the
    #: whole sweep to the dense batched route instead of rebuilding the
    #: accelerated operator per frequency. `_swept_prefers_dense` carries the
    #: measurement table and the memory arithmetic behind the number.
    SWEPT_DENSE_MAX_BASES = 1024

    def _swept_dense_dispatch_ok(self):
        """Whether the size-based swept dispatch may speak for this class.

        True on the generic H-matrix accelerator, where the crossover really
        is a function of the basis count. `ArrayBlockSolver` overrides it to
        False: its wall clock against the dense sweep is set by the ARRAY
        structure (element count, shape classes, lattice-FFT eligibility),
        not by n, and measurably so — same 3-point sweep, same box, dipole
        line array vs the batched dense route:

            240 bases   0.283 s vs 0.060 s dense   (dense 4.7x faster)
            480 bases   0.205 s vs 0.249 s dense   (dense 1.2x SLOWER)
           1000 bases   1.901 s vs 0.712 s dense   (dense 2.7x faster)

        A basis-count threshold cannot predict that ordering — at 480 it
        would dispatch to the slower engine and pay 2.3x the memory for the
        privilege. Measuring a second threshold in the array-structure
        coordinates is the work that would lift this hook; guessing one is
        exactly what #262 asks us not to do.
        """
        return True

    def _swept_prefers_dense(self):
        """The ONE predicate both swept entry points consult (issue #262).

        Below a measured basis count the batched dense sweep beats this
        class's per-k operator rebuild outright, so `compute_impedance_swept`
        and `_port_solutions_swept` both hand the sweep to
        `BSplineSolver`'s batched route. Having exactly one predicate is the
        point: #241 was a bug in which one solver answered swept Y and swept Z
        on two different engines, and a size rule spelled twice would grow the
        same defect back. Both entry points call this; nothing else decides.

        Measured on this box (3-point sweep, gap-fed linear array of
        half-wave dipoles, degree 2, free space, one driven port; wall is the
        whole sweep, MB is peak RSS of the process):

            n_basis   dense batched        accelerated per-k     dense win
              120     0.054 s /   90 MB    0.072 s /  86 MB        1.3x
              240     0.052 s /  113 MB    0.200 s /  89 MB        3.9x
              480     0.291 s /  211 MB    0.572 s / 113 MB        2.0x
              800     0.526 s /  397 MB    1.193 s / 126 MB        2.3x
             1000     0.745 s /  428 MB    1.479 s / 137 MB        2.0x
             1600     1.767 s /  919 MB    3.741 s / 173 MB        2.1x
             2000     2.505 s / 1387 MB    4.040 s / 231 MB        1.6x
             3360     7.697 s / 3700 MB    8.991 s / 297 MB        1.2x

        The dense advantage is real but it decays — 2x through the middle,
        1.6x by 2,000 bases, 1.2x by 3,360 — while its memory grows as n².
        (#262's own table, measured before the extended-kernel C++ work and
        before #263, had the dense side 3.0-5.8x ahead; the accelerated route
        has closed most of that gap on its own.) So the ceiling belongs where
        the win is still worth paying for, and the memory arithmetic says the
        same thing:

        * a single-k dense Z at the ceiling is 1024² · 16 B = 16.8 MB;
        * the batched route's real peak is the all-pairs moment tensor,
          (chunk, (d+1)², N, N) complex. Post-#263 that tensor is chunked
          under `swept_mem_mb` (default 256 MB) — which is what makes this
          dispatch safe at all — but `_swept_batched_z_chunks` floors the
          chunk at 1, so ONE k's tensor is the point past which the budget
          stops being honoured. At d = 2 that is 9 · 1024² · 16 = 151 MB
          (inside the budget); at d = 3, 268 MB (at it). Above ~1,365 bases
          at d = 2 the floor bites, and the measured peaks leave the
          few-hundred-MB range accordingly: 919 MB at 1,600, 3.7 GB at 3,360;
        * #143's whip — 12,682 bases, the model `HMatrixSolver` exists for —
          is 12x the ceiling and can never dispatch. Its dense Z alone would
          be 2.57 GB, its moment tensor 23 GB.

        `swept_dense_max_bases` is the per-instance escape hatch: 0 pins the
        accelerated route at every size (a memory-constrained caller, or a
        test that wants to exercise the accelerator on a small model), None
        takes `SWEPT_DENSE_MAX_BASES`.

        Dispatch also requires `_swept_batched_available()`. Without it the
        dense base sweep falls onto its per-k loop, which calls back into
        THIS class's `compute_impedance` / `compute_port_solution` — i.e. it
        would run the accelerated route anyway, just with a wasted same-edge
        hoist in front of it. The dispatch is a choice of engine, so it only
        fires where the other engine is actually reachable.
        """
        cap = self.swept_dense_max_bases
        if cap <= 0 or not self._swept_dense_dispatch_ok():
            return False
        if not self._swept_batched_available():
            return False
        return self._context()["n_basis"] <= cap

    def compute_impedance_swept(self, k_array):
        """Frequency sweep on the accelerated operator: rebind k per point and
        reuse the fast `compute_impedance` (which assembles the H-matrix /
        array block for that k). Overrides the dense base sweep, whose
        `same_edge_prep` batching argument the accelerated `compute_impedance`
        doesn't accept — calling it would `TypeError`.

        Hands the whole sweep to the dense base in two cases: where the
        accelerator is unsupported for this configuration (then the base
        sweep's batched same-edge precompute is worth having), and where
        `_swept_prefers_dense` says the model is small enough that the
        batched dense route simply wins (#262). `_port_solutions_swept`
        consults the same predicate, so swept Z and swept Y are never on
        different engines.
        """
        if self._hmatrix_unsupported() or self._swept_prefers_dense():
            return super().compute_impedance_swept(k_array)
        _refuse_complex_k(k_array, "HMatrixSolver.compute_impedance_swept")
        k_array = np.asarray(k_array, dtype=float)
        n_ports = self._port_count()
        if n_ports == 1:
            z_out = np.zeros(k_array.shape[0], dtype=np.complex128)
        else:
            z_out = np.zeros((k_array.shape[0], n_ports), dtype=np.complex128)
        with self._k_restored():
            for i, kk in enumerate(k_array):
                self._set_k(kk)
                z, _ = self.compute_impedance()
                z_out[i] = z
        return z_out

    def _port_solutions_swept(self, k_array):
        """Swept ports on the accelerated operator — the `compute_y_matrix_swept`
        / `compute_port_solution_swept` twin of `compute_impedance_swept`
        (issue #241).

        Without this, a swept Y on an accelerated solver fell straight through
        to `BSplineSolver`'s implementation and ran the DENSE fill per k: the
        batched dense assembly (or, with that accelerator absent, a dense
        `_compute_Z_operator`) for every frequency, silently bypassing the very
        operator the caller asked for. It was never wrong — the dense path is
        the reference, junction ports and all — just quietly not accelerated.
        Now the sweep rebinds k per point and reuses the fast
        `compute_port_solution`, exactly as the impedance sweep above reuses
        the fast `compute_impedance`; `ArrayBlockSolver` inherits it and gets
        its per-k operator-cache hits along the way.

        The tradeoff is the same one `compute_impedance_swept` makes, and it
        is worth stating because it does NOT run one way. Re-measured on this
        box post-#263 (3-point sweep, gap-fed linear array, 1 port, d=2;
        wall / peak RSS), dense batched route vs this one:

             480 bases   0.29 s /  211 MB   →   0.57 s / 113 MB
            1000 bases   0.75 s /  428 MB   →   1.48 s / 137 MB
            1600 bases   1.77 s /  919 MB   →   3.74 s / 173 MB
            2000 bases   2.51 s / 1387 MB   →   4.04 s / 231 MB
            3360 bases   7.70 s / 3700 MB   →   8.99 s / 297 MB

        The batched dense sweep amortizes one C++ assembly and one stacked
        LAPACK factorisation across a whole k-chunk; this route rebuilds an
        ACA operator and re-runs GMRES per frequency, so at small and middling
        n it loses on wall clock. What it does not do is grow
        as n²: the dense route is already at 3.7 GB by 3,360 bases and does
        not survive the models `HMatrixSolver` exists for at all (issue #143's
        whip: 12,682 bases, 21.6 GiB dense).

        Which is why the size rule is now a dispatch rather than an argument.
        Below `_swept_prefers_dense`'s measured ceiling this generator hands
        the sweep to the dense base itself; above it, the accelerated route is
        the only one that finishes, and picking the accelerated solver and
        silently getting a dense sweep would be an OOM waiting for a big
        enough model. The one thing that must not happen is the #241 defect —
        one solver answering swept Y here and swept Z somewhere else — so the
        rule lives in a single predicate that `compute_impedance_swept` calls
        too, and the escape hatch (`swept_dense_max_bases=0`) pins BOTH back
        onto the accelerator at once.

        The per-k rebuild is still not fundamental, and dispatch does not fix
        it — above the ceiling every frequency pays a full fill. The ACA block
        tree and admissibility are k-independent (`build_partition` is already
        memoised across the sweep; only the block CONTENTS move with
        frequency), so a swept H-matrix fill could batch the k axis the way
        the dense assembly does. Profiled at 1,600 bases the per-k wall splits
        ~57% near-block dense fill, ~25% far-block ACA, ~18% GMRES, so
        k-batching the near fill is the piece with something to win and the
        ACA/GMRES 43% is genuinely per-frequency. #262 holds that idea.

        Falls back to the dense base sweep wherever the accelerator is
        unsupported for this configuration (singular enrichment), which is
        also where the base sweep's batched same-edge precompute is worth
        having.
        """
        if self._hmatrix_unsupported() or self._swept_prefers_dense():
            yield from super()._port_solutions_swept(k_array)
            return
        _refuse_complex_k(k_array, "HMatrixSolver._port_solutions_swept")
        with self._k_restored():
            for kk in np.asarray(k_array, dtype=float):
                self._checkpoint()  # top of each frequency iteration
                self._set_k(kk)
                yield self.compute_port_solution()

    def __init__(
        self,
        *args,
        aca_eta=1.0,
        aca_leaf_size=32,
        aca_tol=1e-4,
        solve_tol=1e-6,
        hmatrix_use_accel=True,
        precond_eta=None,
        swept_dense_max_bases=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.aca_eta = float(aca_eta)
        self.aca_leaf_size = int(aca_leaf_size)
        self.aca_tol = float(aca_tol)
        self.solve_tol = float(solve_tol)
        # Preconditioner near-field admissibility. The GMRES preconditioner
        # uses a *stronger* (tighter-eta) near-field than the operator: every
        # operator-far block that is inadmissible at `precond_eta` (the "first
        # ring" just outside the operator's near band) is folded into the
        # sparse preconditioner as a dense reconstruction of its already-
        # computed low-rank factors — no extra kernel work, operator stays
        # compressed. None ⇒ 0.5·aca_eta. Set equal to aca_eta to disable.
        # The MoM EFIE operator is non-normal, so spectral (coarse-space)
        # deflation does not help; a denser near-field does.
        self.precond_eta = (
            0.5 * self.aca_eta if precond_eta is None else float(precond_eta)
        )
        # Use the fused C++ off-edge block assembler for ACA far blocks when
        # available; set False to force the pure-numpy zblock path (testing).
        self.hmatrix_use_accel = bool(hmatrix_use_accel)
        # Swept size dispatch (#262): at or below this many bases both swept
        # entry points take the batched dense route instead of rebuilding the
        # accelerated operator per frequency. None ⇒ the class default; 0 (or
        # anything ≤ 0) ⇒ never dispatch, i.e. always accelerated — the opt-out
        # for a memory-constrained caller. See `_swept_prefers_dense`.
        self.swept_dense_max_bases = (
            int(self.SWEPT_DENSE_MAX_BASES)
            if swept_dense_max_bases is None
            else int(swept_dense_max_bases)
        )
        self._hm_context = None
        self._hm_gl01 = None
        self._hm_se_cache = {}
        self._hm_se_cache_k = None
        self._hm_partition = None
