"""Higher-order B-spline Galerkin MoM solver.

The retired TriangularSolver was the degree-1 (tent) special case of this
solver — d=1 reproduces it to roundoff on knot-fed meshes (see
tests/test_tent_parity.py); this
module extends to arbitrary degree d on multi-wire polylines with K-wire
junctions, primarily as an in-codebase arbiter for the hentenna question
(NEXT_STEPS.md items 9, 13, 14): does the tent basis converge to the
correct value, or is it converged-to-the-wrong-place?

Scope:
  * arbitrary number of wires; each wire is a polyline (M ≥ 2 anchors)
  * uniform segments per edge, possibly non-uniform across edges
  * free space, thin-wire kernel with a² wire-radius regularization;
    `wire_radius` is a scalar or a per-wire sequence (momwire#147) — mixed
    radii regularize each observer row with ITS wire's radius (the
    observer-surface convention oracle-validated in
    docs/sinusoidal_basis_design.md "Per-wire radius")
  * delta-gap "applied-E" source on one feed wire
  * degree d ∈ {1, 2}  (d=1 reproduces the tent basis up to feed convention)
  * K-wire junctions with KCL constraint (Σ outflow currents = 0)
  * ground junctions (#151): a wire end at `ground_z` keeps its value-1
    boundary basis (end current is a dof; the image is the return path);
    a junction at the plane keeps its directional bases but drops the
    KCL row (current may flow into ground)
  * ground: PEC image (`ground_z`), NEC-gn 0-style reflection-coefficient
    finite ground (`ground_eps`, docs/refl-coef-ground-plan.md), and
    NEC-gn 2-style Sommerfeld finite ground (`ground_model="sommerfeld"`,
    docs/sommerfeld-ground-plan.md)

Tent-basis generalisation: a polynomial-of-degree-d on each segment
instead of just a linear ramp. Each interior basis Φ_m spans up to d+1
contiguous segments within a single wire; on each segment in its support
("wing") the basis equals Σ_p C[m, w, p] · u^p with u local arc length.

J_pq[i, j] = ∫∫ u^p u'^q · exp(-jkR)/(4πR) du' du
with R² = |r_i(u) - r_j(u')|² + a²

Galerkin assembly:
    Z_A[m,n]   = jωμ Σ_{a,b} (t_i · t_j) · Σ_{p,q} C[m,a,p] C[n,b,q]
                 · J_{pq}[supp_seg[m,a], supp_seg[n,b]]
    Z_Φ[m,n]   = (1/jωε) Σ_{a,b} Σ_{p≥1,q≥1} p·q · C[m,a,p] C[n,b,q]
                 · J_{p-1,q-1}[supp_seg[m,a], supp_seg[n,b]]

Junction directional bases: at every junction node with K connected wire-
ends we add K boundary bases (B_0 or B_{N+d-1} of each connected wire,
the ones with value 1 at the junction) and enforce KCL via a Lagrange-
multiplier row (the same treatment the retired TriangularSolver used).

Feed: v_m = Φ_m(s_f), Z_drive = 1 / (v^T c).
"""

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse
from scipy.interpolate import BSpline

from ._bspline_kernels import (
    _EK,
    _HAVE_BSPLINE_OFFEDGE_SWEPT_ACCEL,
    _ek_axis_groups,
    _seg_seg_full_moments_offedge,
    _seg_seg_full_moments_offedge_swept,
    _seg_seg_reg_geometry,
    _seg_seg_reg_moments,
    _seg_seg_reg_moments_from_geometry,
    _seg_seg_reg_moments_from_geometry_swept,
    _seg_seg_static_moments,
)
from ._quadrature import leggauss

from . import _ground_refl
from . import _sommerfeld
from . import _wire_loading
from ._accel import acc as _acc
from ._cancel import _Cancelable
from ._element_currents import _ElementCurrents
from ._port_solution import PortSolution, _SweptPortSolutions

_HAVE_BSPLINE_ASSEMBLE_ACCEL = _acc is not None and hasattr(_acc, "assemble_Z_bspline")
_HAVE_BSPLINE_ASSEMBLE_W_ACCEL = _acc is not None and hasattr(
    _acc, "assemble_Z_bspline_weighted"
)
_HAVE_ENRICH_ACCEL = _acc is not None and hasattr(_acc, "assemble_Z_enrich")
_HAVE_BSPLINE_SWEPT_ASSEMBLE_ACCEL = _acc is not None and hasattr(
    _acc, "assemble_Z_bspline_swept"
)
_HAVE_BSPLINE_WINDOWED_ASSEMBLE_ACCEL = _acc is not None and hasattr(
    _acc, "assemble_Z_bspline_windowed"
)
_HAVE_BSPLINE_W_WINDOWED_ASSEMBLE_ACCEL = _acc is not None and hasattr(
    _acc, "assemble_Z_bspline_weighted_windowed"
)

_BSPLINE_ASSEMBLE_ACCEL_MAX_D = 2


# Constant Vandermonde inverses for uniform sample points [0, 1/d, ..., 1].
# Used by `_build_basis_polynomials` to convert per-segment basis values at
# d+1 uniform local-u sample points to polynomial coefficients without a
# per-segment scipy.linalg.solve. With u_local = h_seg * [0, 1/d, ..., 1],
# the Vandermonde factors as Vmat = V_unit @ diag(1, h, h², ..., h^d), so
# coeffs_p = (V_unit_inv @ vals)_p / h_seg^p — pure matmul + column scaling.
_V_UNIT_INV: dict[int, np.ndarray] = {
    1: np.array([[1.0, 0.0], [-1.0, 1.0]]),
    2: np.array([[1.0, 0.0, 0.0], [-3.0, 4.0, -1.0], [2.0, -4.0, 2.0]]),
}


def _design_matrix_rows(dm_csr, d):
    """Split a `BSpline.design_matrix` CSR into per-row (values, columns).

    Returns two (n_rows, d+1) arrays. The dense form of that CSR is
    (n_rows, n_basis) with only d+1 nonzeros per row, so materialising it
    costs O(n_rows · n_basis) for O(n_rows · d) of content (#329); every
    consumer here reads inside the band, so none of them needs it.

    The two structural facts this relies on — exactly d+1 stored entries
    per row, at consecutive columns in ascending order — are scipy's
    output shape rather than anything we control, so they are checked
    here (O(n_rows · d), the same order as the gather itself) instead of
    assumed.
    """
    width = d + 1
    n_rows = dm_csr.shape[0]
    if not np.array_equal(np.diff(dm_csr.indptr), np.full(n_rows, width)):
        raise AssertionError(
            "BSpline.design_matrix no longer stores exactly d+1 entries per row"
        )
    cols = dm_csr.indices.reshape(n_rows, width)
    if width > 1 and not np.all(np.diff(cols, axis=1) == 1):
        raise AssertionError(
            "BSpline.design_matrix columns are no longer consecutive ascending"
        )
    return dm_csr.data.reshape(n_rows, width), cols


def _design_matrix_band(dm_csr, n_seg, d):
    """Per-segment basis values as (n_seg, d+1 samples, d+1 bases).

    Band column b holds basis (seg + b): segment s lies inside the
    support of exactly bases s .. s+d, so this band is all of the dense
    design matrix that any per-segment consumer can read.

    Sample rows are gathered by each row's OWN column start, not by a
    shared per-segment offset. A segment's last sample sits on its
    right-hand knot, and scipy resolves a sample on an interior knot into
    the NEXT span; such a row starts one column higher and carries a
    trailing entry for the basis whose support opens at that knot, which
    is zero there. Whether a given segment's endpoint sample lands that
    way is a floating-point question (arc[s] + h[s] need not round to
    arc[s+1]), so the offset is read per row and the out-of-band column
    is checked to be zero rather than reasoned about.
    """
    width = d + 1
    data, cols = _design_matrix_rows(dm_csr, d)
    data = data.reshape(n_seg, width, width)
    starts = cols.reshape(n_seg, width, width)[:, :, 0]
    off = starts - np.arange(n_seg)[:, None]
    if off.min() < 0 or off.max() > 1:
        raise AssertionError(
            f"design-matrix span offset outside [0, 1]: {off.min()}..{off.max()}"
        )
    padded = np.zeros((n_seg, width, width + 1), dtype=np.float64)
    np.put_along_axis(padded, off[:, :, None] + np.arange(width), data, axis=2)
    if np.any(padded[:, :, width] != 0.0):
        raise AssertionError("nonzero design-matrix entry outside the segment band")
    return padded[:, :, :width].copy()


# Module-level caches for `_build_geometry` and `_build_basis_polynomials`.
# Both functions are pure functions of immutable geometry inputs (wires +
# n_per_edge_per_wire, plus degree + junctions for the basis case) — they
# don't depend on `k` / wavelength / feed location. The instance-level
# caches in `_cached_geometry` / `_cached_basis_polynomials` only help the
# swept path (where one solver instance handles many k's). The engine
# wrapper instantiates a fresh BSplineSolver per impedance() call, so the
# instance cache is dead for the interactive UI sweep. These module-level
# caches survive across instances and turn a band-sweep of N freqs into
# 1 cold call + (N−1) hot calls for the geometry/basis stages.
#
# FIFO with a small bound — typical interactive use has 1–3 active
# (geometry, degree) combinations at a time.
_GEOMETRY_CACHE: dict = {}
_BASIS_POLY_CACHE: dict = {}
_GEOMETRY_CACHE_MAX = 32
_BASIS_POLY_CACHE_MAX = 32

# The SommerfeldGrid cache lives in `_sommerfeld.get_grid` (module-level,
# keyed `(eps_t, k, r1_bucket, omega, mu)`, r1_max bucketed UP in ~25%
# geometric steps) so SinusoidalSolver and the fast solvers share fills
# with this module — docs/sommerfeld-everywhere-plan.md Phase 1.


def _evict_fifo(cache: dict, limit: int) -> None:
    while len(cache) >= limit:
        cache.pop(next(iter(cache)))


# The extended-kernel spec for a SAME-EDGE moment block (momwire#249). One
# edge is one straight run of one wire at one radius, so every pair in the
# block is coaxial-and-equal-radius by construction and the labels carry no
# information — the all-None spec is the kernel layer's spelling for "the
# whole block is eligible".
_EK_SAME_EDGE = _EK(a=None, group_i=None, group_j=None)


def _ek_slice(ek, rows=None, cols=None):
    """Restrict an `_EK` spec's per-segment labels to a block's rows/cols.

    The solver builds ONE label array over the whole mesh (labels have to be
    globally comparable — `group_i[i] == group_j[j]` is the eligibility
    test), and every windowed/blocked fill site then hands the kernel the
    labels of just the segments it is filling. `rows`/`cols` are anything
    numpy indexes a 1-D array with: a slice, or a fancy-index array.
    None (both here and for a spec's own labels) means "unrestricted".
    """
    if ek is None:
        return None
    gi = ek.group_i if rows is None or ek.group_i is None else ek.group_i[rows]
    gj = ek.group_j if cols is None or ek.group_j is None else ek.group_j[cols]
    return _EK(a=ek.a, group_i=gi, group_j=gj)


def _xfem_projection_coeffs(d):
    """Coefficients c such that P_bubble Φ_sing (t) = Σ_p c_p t^p, where
    P_bubble is the L²-orthogonal projection of Φ_sing(t) = t·log(t) onto
    the subspace of P_d on [0, 1] whose elements vanish at t=0 and t=1.

    Returns an array of length d+1, the monomial coefficients of the
    projection. Pure constants — h-independent and geometry-independent.

    Used to build the "stable" XFEM enrichment basis
        Φ_sing_stable(t) = t·log(t) − Σ_p c_p t^p
    Φ_sing_stable retains both endpoint BCs of Φ_sing — vanishing at
    t=0 (the junction node, required for finite-current KCL at the
    K-wire junction — the KCL constraint only sees polynomial bases,
    so enrichment must self-zero there) and at t=1 (the segment's far
    end, required for current continuity with the adjacent non-enriched
    segment). And on the BC-compatible bubble subspace, the enrichment
    is L²-orthogonal: α_enrich = 0 exactly when the truth's
    BC-compatible-bubble part lives in the polynomial subspace — the
    small-N transient where the original Φ_sing absorbs polynomial
    discretization error is eliminated.

    Bubble basis: b_k(t) = t^(k+1) − t^(k+2) = t·(1−t)·t^k for k = 0..d−2.
    Dimension is d−1 (for d=2 it's 1D = span{t(1−t)}; for d=1 it's empty,
    matching the empirical "d=1 enrichment is a no-op" finding in
    NEXT_STEPS item 15(b)).
    """
    if d < 2:
        return np.zeros(d + 1)
    n_b = d - 1
    # Bubble Gram matrix: ⟨b_i, b_j⟩ = ∫₀¹ (t^(i+1)-t^(i+2))(t^(j+1)-t^(j+2)) dt
    #                  = 1/(i+j+3) − 2/(i+j+4) + 1/(i+j+5)
    G = np.array(
        [
            [
                1.0 / (i + j + 3) - 2.0 / (i + j + 4) + 1.0 / (i + j + 5)
                for j in range(n_b)
            ]
            for i in range(n_b)
        ]
    )
    # Moment vector: ⟨Φ_sing, b_i⟩ = ∫₀¹ t·log(t)·(t^(i+1)-t^(i+2)) dt
    #              = ∫ t^(i+2) log(t) dt − ∫ t^(i+3) log(t) dt
    #              = −1/(i+3)² + 1/(i+4)²    [using ∫₀¹ t^n log(t) dt = −1/(n+1)²]
    m = np.array([-1.0 / (i + 3) ** 2 + 1.0 / (i + 4) ** 2 for i in range(n_b)])
    alpha = np.linalg.solve(G, m)
    # Bubble-coefficient α → monomial coefficients c:
    # P(t) = Σ_k α_k · (t^(k+1) − t^(k+2))  ⇒  c[k+1] += α_k, c[k+2] −= α_k.
    coeffs = np.zeros(d + 1)
    for k in range(n_b):
        coeffs[k + 1] += alpha[k]
        coeffs[k + 2] -= alpha[k]
    return coeffs


@dataclass(frozen=True)
class _SplineBasis:
    """Opaque `PortSolution.basis` payload for the B-spline families.

    The per-solve context needed to read a coefficient column: the geometry
    tables, the basis support map and per-basis polynomials, the knot vectors
    and the basis-index map, plus `n_poly` — the number of polynomial dofs,
    i.e. where the singular-enrichment block starts in a column (== the column
    length when enrichment is off). Private on purpose — #232 hands consumers
    an OPAQUE handle, not an interface; nothing here promises it survives the
    next solve.
    """

    geom: dict
    supp_seg: object
    polys: object
    wire_knots: object
    wire_basis_global: object
    n_poly: int


class BSplineSolver(_ElementCurrents, _SweptPortSolutions, _Cancelable):
    """Degree-d B-spline Galerkin MoM, multi-wire polylines with junctions.

    **Usable Δ/a floor (issue #248).** Like every reduced-kernel
    (filament-source) solver, this one answers a question that stops being
    well-posed when segments get shorter than the wire radius: below
    Δ/a ≈ 1 the solved current develops a cell-alternating spurious mode at
    the feed that dominates the driving-point reading, and mesh refinement
    at fixed radius makes it diverge rather than settle (|I_feed| grows
    without bound; Z → 0 — measured for every momwire basis, diagnosis on
    the issue). The answer there is basis-dependent, not physical;
    `SinusoidalSolver` reproduces nec2c's EK-OFF column on those rungs only
    because it is NEC-2's own discretization. Against that oracle at NS=41
    this solver holds ≤0.7% for Δ/a ≥ 24, ≤2.1% for Δ/a ≥ 6, ≤2.9% for
    Δ/a ≥ 2.4, then 5.9% / 18.5% / 33.8% at Δ/a = 1.22 / 0.81 / 0.41.
    Treat **Δ/a ≥ 2 as the usable floor** (≈3%), Δ/a ≥ 6 for ≈2%; the
    residual also grows with absolute a/λ at fixed Δ/a. Fat conductors need
    the extended kernel AND a mesh that keeps Δ/a above ~1 — they are not
    rescued by refining the mesh.

    `extended_kernel=True` (issue #249) is the first half of that: it buys
    ~10× on the same-edge moments at Δ/a ≥ 2 (measured against the exact
    tube kernel). It does NOT move the floor. EK stops improving the moments
    below Δ/a ≈ 1, and by Δ/a ≈ 0.5 it is *worse* than the reduced kernel on
    the nearest-neighbour moment — an engine-free confirmation of the same
    floor #248 found from the divergence side. The reason is that EK's O(a²)
    tube expansion is a truncation in b/R, and the self moment integrates
    through the ζ ≲ a region where that truncation is still O(1) wrong.

    Parameters
    ----------
    wires : list of (M, 3) polyline arrays, M ≥ 2 anchors each.
    n_per_edge_per_wire : list of (int | sequence | None). Per-wire segment
        counts per edge. None for a wire ⇒ use `nsegs` on every edge; int ⇒
        same count for every edge; sequence ⇒ explicit per-edge count.
    degree : B-spline degree (1 ≤ degree ≤ 2 currently; static-moment file
        only covers max_d=2). d=1 IS the tent basis (it reproduced the
        retired TriangularSolver to roundoff whenever the feed lands on a
        knot; for a between-knots feed arclength triangular snapped to the
        nearest knot while this solver excites the exact arclength).
    feed_wire_index : index of the wire carrying the delta-gap source.
    feed_arclength : arc length along the feed wire at which to evaluate
        Φ_m(s_f). Default: feed wire midpoint.
    feed_model : gap-source model, `"point"` (default) or `"segment"`
        (stevenmburns/momwire#216). `"point"` is this solver's native
        zero-width drive, E_app = V·δ(s − s_f), whose Galerkin drive column
        collapses on the delta to v_m = Φ_m(s_f). `"segment"` is NEC's
        segment-wide gap, E_app = V/Δ uniform over the mesh cell containing
        s_f, giving v_m = (1/Δ)∫_cell Φ_m ds — the same source
        `SinusoidalSolver` hard-codes and `SinusoidalGalerkinSolver` defaults
        to, so it is what makes a feed-matched comparison against those
        solvers possible from this side (report §19). Mutually exclusive with
        `feed_smoothing_factor`. Note the readout follows: Z = 1/(vᵀc) is
        always the drive's dual, so `"segment"` reads the gap-AVERAGED
        current, matching `SinusoidalGalerkinSolver(feed_readout=
        "variational")` rather than its default centre readout.
    junctions : list of [(wire_idx, "start"|"end"), ...] tuples, each entry
        one junction node where K wire endpoints meet.
    n_qp_pair : Gauss-Legendre nodes per segment per axis for the smooth-
        kernel piece of same-edge pairs and for all cross-edge / cross-wire
        pairs (full kernel with a² regularization).
    extended_kernel : NEC's EK card for this basis family (#249). False —
        the default, and what this solver did before #249 — is the reduced
        ("thin-wire") kernel: the source current is a filament on the wire
        axis and the conductor's girth survives only as the a²
        regularization of R. True is NEC's EXTENDED thin-wire kernel, Eq 89
        of the theory manual, the O(a²) azimuthal average of the Green's
        function over a source tube of radius a. Unlike `SinusoidalSolver`,
        which transcribes NEC's per-END IND1/IND2 gating, this Galerkin fill
        applies EK to a segment PAIR iff the two segments are coaxial and of
        equal radius (`_ek_axis_groups`) — a symmetric rule, so the Galerkin
        symmetry of Z survives as an error detector. It agrees with NEC
        exactly on straight wires and on perpendicular ground contacts (via
        the mirrored source) and is strictly more conservative at bends,
        radius steps and K ≥ 3 junctions, where NEC still extends the
        cross-arm pairs and this rule does not (~1 % of Z at Δ/a = 2, O(h)
        in the refinement limit — #249 §4.3).

        Every fill an EK-on solve takes is C++-served (momwire#270): the
        same-edge static and reg moments, the off-edge moments single-k and
        batched-over-k, and `HMatrixSolver`/`ArrayBlockSolver`'s fused ACA
        block assembler all have extended-kernel twins. Measured cost of
        turning EK on, 400-segment monopole, degree 2: ~1.1x on the dense
        path and ~1.1x on the H-matrix, the same factor in free space, over
        PEC ground and over both finite grounds. Refused, rather than
        half-served, with `use_singular_enrichment` — see the
        `NotImplementedError` in `__init__`.

        **Ground under EK.** PEC ground (`ground_z` alone) and both finite-
        ground models (`ground_eps`) are served (momwire#269). The image
        blocks are extended: eligibility is scored against the MIRRORED
        source geometry (`_ek_axis_labels(mirror=True)`, one joint scan), so
        a vertical monopole extends against its own image — NEC's IND = 0
        ground-contact branch — while a horizontal wire, whose image is
        parallel but offset by twice the height, does not. `ground_eps`
        costs nothing extra over PEC: its Fresnel dyad / image-charge tables
        are per-segment-pair weights applied by the assembler AFTER the
        moment fill, so the same EK moment twins serve them (dense route:
        `_accumulate_Z_image_chunked`; H-matrix near blocks:
        `_zblock_image_refl`; H-matrix far blocks: the fused
        `bspline_assemble_offedge_block_refl_ek` twin).

        What stays REDUCED under EK is the Sommerfeld remainder
        (`_Z_sommerfeld_remainder`, `_sommerfeld_global_lowrank`) — the
        smooth ground-wave correction NEC's eqs 143-147 add on top of the
        C2-scaled exact image. That is deliberate, not an omission: EK is an
        O((a/R)²) tube correction, and the remainder's source is the ground
        reflection, so R ≥ 2h for a wire at height h and the un-applied
        correction is O((a/2h)²). Measured, by building the extended
        remainder outright (the same field azimuthally averaged over the
        source tube) and re-solving: for a wire CLEAR of the plane it moves
        |Z| by ≤ 1.1e-4 relative — three orders below the EK shift the image
        blocks do carry, two below this basis's own accuracy at Δ/a ≥ 2. At
        ground CONTACT, where the (a/2h)² estimate degenerates and only the
        remainder's smoothness bounds it, the measured cost is 3-4e-3
        relative: still an order below the basis error, but ~45% of that
        deck's own EK shift, so it is the one place the mixture is visible.
        The full table, the O(a²) confirmation and the tolerances are in
        tests/test_extended_kernel_bspline.py Gate 18.
    wavelength, halfdriver_factor, wire_radius, nsegs : shared solver
        conventions (see SinusoidalSolver for the same surface).
    wire_conductivity : distributed conductor loss (#131). None (default)
        = PEC; a scalar σ [S/m] applies to every wire; a per-wire sequence
        with NaN entries switches individual wires off. Adds the exact
        round-conductor internal impedance Z'_int(ω) (valid DC → strong
        skin effect) as a series loading over same-wire basis overlaps.
        `wire_loss_power(coeffs)` reads back the dissipated watts.
    insulation_radius, insulation_eps_r : dielectric jacket (#131), given
        together (scalar / per-wire like wire_conductivity). Adds King's
        quasi-static series inductance μ₀/2π·(1−1/εr)·ln(b/a) per meter —
        the insulated-wire velocity-factor effect (the wire tunes a few
        percent long). Purely reactive; no dissipation modeled.
    swept_mem_mb : memory budget (MB, default 256) for dense moment tensors
        and for the batched swept path's per-chunk transients — the
        all-pairs J moment tensor plus
        the same-edge reg moment blocks (see `_swept_batched_z_chunks`;
        the per-k fallback sweeps chunk their same-edge hoist under the
        same budget, `_same_edge_prep_swept_chunks`).
        Peak transient memory of a sweep ≈ this budget, so a
        memory-constrained deployment caps it per solve (e.g. 64 on a
        small shared host). Speed saturates by ~256 (chunk ≈ 8-16 on
        production shapes); below ~64 the batching win starts eroding
        (chunk=1 costs ~+75% on the worst shapes).
    """

    eps = 8.8541878188e-12
    mu = 1.25663706127e-6

    def __init__(
        self,
        *,
        wires,
        n_per_edge_per_wire=None,
        degree=2,
        feed_wire_index=0,
        feed_arclength=None,
        feeds=None,
        feed_model="point",
        feed_smoothing_factor=None,
        junctions=None,
        junction_ports=None,
        node_gaps=None,
        n_qp_pair=4,
        n_qp_source=16,
        extended_kernel=False,
        wavelength=22,
        halfdriver_factor=0.962,
        wire_radius=0.0005,
        wire_conductivity=None,
        insulation_radius=None,
        insulation_eps_r=None,
        nsegs=101,
        ground_z=None,
        ground_eps=None,
        ground_phi_mode="normal",
        ground_model="refl-coef",
        n_qp_sommerfeld=3,
        use_singular_enrichment=False,
        n_qp_sing=32,
        enrichment_min_k=3,
        enrichment_variant="raw",
        tikhonov_lambda=1e-3,
        auto_tap_ratio_threshold=0.3,
        swept_mem_mb=256,
        cancel=None,
    ):
        self._cancel = cancel
        if degree < 1:
            raise ValueError(f"degree must be >= 1, got {degree}")
        if degree > 2:
            raise NotImplementedError(
                "degree > 2 needs scripts/derive_bspline_static_moments.py "
                "to be re-run with a larger MAX_D"
            )
        if not wires:
            raise ValueError("wires must be non-empty")
        if use_singular_enrichment and (
            wire_conductivity is not None or insulation_radius is not None
        ):
            raise NotImplementedError(
                "use_singular_enrichment + distributed wire loading together "
                "not supported yet — the enrichment bases don't carry the "
                "loading overlap term"
            )

        self.degree = int(degree)
        self.wavelength = wavelength
        self.halfdriver_factor = halfdriver_factor
        self.wire_radius = wire_radius
        self.nsegs = nsegs
        self.ground_z = ground_z
        # Finite ground via NEC-style reflection-coefficient weighting of the
        # image block (docs/refl-coef-ground-plan.md). None → PEC image
        # (today's behavior). A complex ε̃ or (eps_r, sigma) tuple → Fresnel-
        # weighted image; needs ground_z. `ground_phi_mode` picks the image-
        # charge (Φ-term) weighting candidate — see _ground_refl.PHI_MODES.
        #
        # Validity window (#153): this mixed-potential refl-coef path is
        # accurate for wires 0.1–0.5λ above the plane — the Φ-term has no
        # exact Fresnel weight, and its θ=0 approximation degrades in the
        # quasi-static near field (|ΔΓ| ~0.02 at 0.1λ, ~0.13 at 0.05λ,
        # worse at contact). Below ~0.1λ or for ground-touching wires,
        # prefer `ground_model="sommerfeld"` (exact everywhere, contact-
        # capable since #151) or the field-based SinusoidalSolver, which
        # applies NEC's dyad exactly at any height.
        if ground_eps is not None and ground_z is None:
            raise ValueError("ground_eps requires ground_z to be set")
        if ground_phi_mode not in _ground_refl.PHI_MODES:
            raise ValueError(
                f"ground_phi_mode must be one of {_ground_refl.PHI_MODES}, "
                f"got {ground_phi_mode!r}"
            )
        self.ground_eps = ground_eps
        self.ground_phi_mode = ground_phi_mode
        # `ground_model` picks the finite-ground physics when `ground_eps`
        # is set: "refl-coef" (NEC gn 0 style, the default) or "sommerfeld"
        # (NEC gn 2 style: exact image scaled by C2 = (eps-1)/(eps+1) plus
        # the interpolated Sommerfeld remainder — see
        # docs/sommerfeld-ground-plan.md). `ground_phi_mode` applies to
        # "refl-coef" only; the sommerfeld image coefficient is exact and
        # has no knob. `n_qp_sommerfeld` is the per-segment Gauss order of
        # the remainder block's field-form Galerkin quadrature (the kernel
        # is smooth — the image point is below the plane — so 3 converges;
        # guarded by a q-vs-q+2 test).
        if ground_model not in ("refl-coef", "sommerfeld"):
            raise ValueError(
                "ground_model must be 'refl-coef' or 'sommerfeld', "
                f"got {ground_model!r}"
            )
        if ground_model == "sommerfeld" and ground_eps is None:
            raise ValueError("ground_model='sommerfeld' requires ground_eps")
        self.ground_model = ground_model
        self.n_qp_sommerfeld = int(n_qp_sommerfeld)
        # NEC's extended thin-wire kernel (momwire#249). See the class
        # docstring for what it is and what it costs; `_ek_spec` builds the
        # per-fill eligibility spec and every fill site guards on this flag,
        # so an EK-off solve never enters a line of EK code.
        self.extended_kernel = bool(extended_kernel)
        # (geom, {mirror: (group_i, group_j)}) — per-geometry-object cache of
        # the axis-group labels, identity-checked, same pattern as
        # `SinusoidalSolver._cached_ek_gating`.
        self._cached_ek_groups = None
        if self.extended_kernel:
            # One refusal rather than a half-served path (#249 §5). The
            # second (finite ground) was lifted by momwire#269 — see below.
            if use_singular_enrichment:
                raise NotImplementedError(
                    "extended_kernel=True + use_singular_enrichment=True not "
                    "supported yet — the enrichment DOFs bypass the moment "
                    "kernels entirely (they carry their own Φ_sing "
                    "quadrature), they exist only at K >= 3 junctions where "
                    "NEC's own gating turns EK off, and the O(a²) tube "
                    "expansion was never derived for the s^(-1/2) shapes "
                    "(stevenmburns/momwire#249 follow-up C)"
                )
            # Finite ground (`ground_eps`, both models) is served since
            # momwire#269. What #249 refused as "an ungated mixture" now has
            # its gates: the refl-coef image rides the same EK-aware moment
            # blocks as the PEC image (the Fresnel tables are per-pair
            # weights applied AFTER the fill, so the kernel choice is
            # orthogonal to them), and the Sommerfeld remainder /global
            # low-rank term deliberately stay REDUCED — see the class
            # docstring's "Finite ground under EK" note for the (a/2h)²
            # negligibility arithmetic and
            # tests/test_extended_kernel_bspline.py's G16-G19 for the
            # measurements behind it.
        self.swept_mem_mb = int(swept_mem_mb)
        if self.swept_mem_mb < 1:
            raise ValueError(f"swept_mem_mb must be >= 1, got {swept_mem_mb}")
        # Source smoothing: when None (default) → delta-gap at exact midpoint
        # of feed wire. When set to a float α, the delta-gap is replaced by a
        # cos² bump of width w = α · h_feed_segment_at_source centered on s_f,
        # giving basis-limited convergence instead of the O(1/N) delta-gap
        # source-singularity rate. α ≈ 2-4 is a sensible starting point;
        # larger α gives faster basis-limited convergence but the smoothing
        # error from the bump's finite width takes longer to vanish.
        self.feed_smoothing_factor = feed_smoothing_factor
        # Gap-source model (stevenmburns/momwire#216), the mirror of #192's
        # option on the sinusoidal siblings. "point" (default) is this
        # solver's native zero-width drive E_app = V·δ(s − s_f); "segment" is
        # NEC's segment-wide gap, E_app = V/Δ uniform over the mesh cell
        # containing s_f (Eq 187's convention). Both are gap models, so
        # combining "segment" with the cos²-bump `feed_smoothing_factor` is
        # meaningless — each replaces the point drive outright.
        if feed_model not in ("point", "segment"):
            raise ValueError(
                f"feed_model must be 'point' or 'segment', got {feed_model!r}"
            )
        if feed_model == "segment" and feed_smoothing_factor is not None:
            raise ValueError(
                "feed_model='segment' and feed_smoothing_factor are mutually "
                "exclusive — both replace the point drive with a spread one"
            )
        self.feed_model = feed_model
        self.n_qp_source = int(n_qp_source)

        self.c = 1 / np.sqrt(self.eps * self.mu)
        self.freq = self.c / self.wavelength
        self.omega = 2 * np.pi * self.freq
        self.k = self.omega / self.c
        self.halfdriver = self.halfdriver_factor * self.wavelength / 4

        self.wires_polylines = [np.asarray(w, dtype=float) for w in wires]
        for i, pl in enumerate(self.wires_polylines):
            if pl.ndim != 2 or pl.shape[0] < 2 or pl.shape[1] != 3:
                raise ValueError(f"wire {i}: polyline must be (M, 3) with M >= 2")

        n_w = len(self.wires_polylines)

        # Per-wire conductor radius (stevenmburns/momwire#147): a scalar
        # applies to every wire; a length-n_wires sequence gives each wire
        # (polyline) its own. `_uniform_radius` is the scalar fast path —
        # it keeps the historical scalar code paths (and the single-`a`
        # C++ kernel arguments) bit-identical whenever all wires share one
        # radius, including when that radius arrived as a uniform array.
        # Mixed radii use the OBSERVER wire's radius in the a²-regularised
        # kernel — see docs/sinusoidal_basis_design.md "Per-wire radius"
        # for the convention and its PyNEC oracle validation.
        radius = np.asarray(wire_radius, dtype=float)
        if radius.ndim == 0:
            radius = np.full(n_w, float(radius))
        elif radius.shape != (n_w,):
            raise ValueError(
                f"wire_radius: expected a scalar or a length-{n_w} sequence "
                f"(one entry per wire), got shape {radius.shape}"
            )
        if not np.all(np.isfinite(radius)) or np.any(radius <= 0.0):
            raise ValueError(
                f"wire_radius entries must be positive and finite, got {radius}"
            )
        self._radius_per_wire = radius
        self._uniform_radius = float(radius[0]) if np.all(radius == radius[0]) else None
        if use_singular_enrichment and self._uniform_radius is None:
            raise NotImplementedError(
                "use_singular_enrichment + mixed per-wire radii together "
                "not supported yet — the enrichment kernels take a single "
                "radius (stevenmburns/momwire#147)"
            )

        # Distributed series wire impedance (stevenmburns/momwire#131):
        # finite conductivity (skin-effect internal impedance) and/or a
        # dielectric jacket (series inductance → velocity factor). Each is
        # None (off, today's PEC behavior), a scalar (every wire), or a
        # per-wire sequence (NaN entries switch a wire off). The loading
        # enters Z as Σ_w Z'_w(ω)·S_w over same-wire basis overlaps — see
        # `_loading_gram` / `_apply_loading`.
        self.wire_conductivity = _wire_loading.normalize_per_wire(
            wire_conductivity, n_w, "wire_conductivity"
        )
        self.insulation_radius = _wire_loading.normalize_per_wire(
            insulation_radius, n_w, "insulation_radius"
        )
        self.insulation_eps_r = _wire_loading.normalize_per_wire(
            insulation_eps_r, n_w, "insulation_eps_r"
        )
        if (self.insulation_radius is None) != (self.insulation_eps_r is None):
            raise ValueError(
                "insulation_radius and insulation_eps_r must be given together"
            )
        if self.insulation_radius is not None:
            finite_b = np.isfinite(self.insulation_radius)
            if not np.array_equal(finite_b, np.isfinite(self.insulation_eps_r)):
                raise ValueError(
                    "insulation_radius and insulation_eps_r must be finite "
                    "on the same wires (NaN switches a wire off in both)"
                )
            # Fail fast on bad geometry/material values (the same checks
            # run inside insulation_inductance, but per-solve is too late).
            for w in np.nonzero(finite_b)[0]:
                _wire_loading.insulation_inductance(
                    self._radius_per_wire[w],
                    self.insulation_radius[w],
                    self.insulation_eps_r[w],
                )
        if self.wire_conductivity is not None:
            for w in np.nonzero(np.isfinite(self.wire_conductivity))[0]:
                if self.wire_conductivity[w] <= 0.0:
                    raise ValueError(
                        f"wire_conductivity[{w}] must be > 0 S/m, "
                        f"got {self.wire_conductivity[w]}"
                    )
        self._loading_active = self.wire_conductivity is not None or (
            self.insulation_radius is not None
        )
        # Per-instance cache for the k-independent loading Gram structure
        # (rows, cols, vals, wire_of_nnz) — see `_loading_gram`.
        self._cached_loading_gram = None

        if n_per_edge_per_wire is None:
            n_per_edge_per_wire = [None] * n_w
        if len(n_per_edge_per_wire) != n_w:
            raise ValueError(
                f"n_per_edge_per_wire length {len(n_per_edge_per_wire)} != n_wires {n_w}"
            )

        self.n_per_edge_per_wire = []
        for i, (pl, npe) in enumerate(zip(self.wires_polylines, n_per_edge_per_wire)):
            n_edges_w = pl.shape[0] - 1
            if npe is None:
                npe = self.nsegs
            if np.isscalar(npe):
                npe = [int(npe)] * n_edges_w
            npe = list(npe)
            if len(npe) != n_edges_w:
                raise ValueError(
                    f"wire {i}: n_per_edge length {len(npe)} != n_edges {n_edges_w}"
                )
            self.n_per_edge_per_wire.append(npe)

        if feeds is None:
            if not (0 <= feed_wire_index < n_w):
                raise ValueError(f"feed_wire_index {feed_wire_index} out of range")
            self.feeds = [(int(feed_wire_index), feed_arclength, 1.0 + 0.0j)]
        else:
            if len(feeds) == 0 and not junction_ports and not node_gaps:
                # Junction ports (issue #172) and node gaps (issue #305) are
                # drive/readout ports in their own right, so a solve driven
                # entirely through them needs no gap feed at all.
                raise ValueError("feeds must contain at least one entry")
            norm = []
            for i, f in enumerate(feeds):
                if len(f) != 3:
                    raise ValueError(
                        f"feeds[{i}]: expected (wire_index, arclength, voltage), got {f!r}"
                    )
                w_i, arc_i, v_i = f
                if not (0 <= w_i < n_w):
                    raise ValueError(
                        f"feeds[{i}]: wire_index {w_i} out of range [0, {n_w})"
                    )
                arc_i = None if arc_i is None else float(arc_i)
                norm.append((int(w_i), arc_i, complex(v_i)))
            self.feeds = norm

        # Back-compat scalars — None when driven entirely through junction
        # ports (feeds=[] with junction_ports, issue #172).
        self.feed_wire_index = self.feeds[0][0] if self.feeds else None
        self.feed_arclength = self.feeds[0][1] if self.feeds else None
        self.n_qp_pair = int(n_qp_pair)

        # Singular basis enrichment at K≥`enrichment_min_k` junctions.
        # When enabled, adds ONE extra basis per (wire, end_pos) tuple at each
        # qualifying junction, with shape Φ_sing(u) = (u/h)·log(u/h) on the
        # adjacent segment (u measured from the junction node, so Φ_sing(0)=0
        # matching the finite-current condition while dΦ_sing/du has a log
        # singularity that captures the classical K≥3 junction charge-density
        # singularity). On hentenna-class geometries this flips the R-rate
        # from O(1/N) to ~O(1/N^(d+1)) (basis-limited). Quadrature is GL with
        # `n_qp_sing` nodes per axis (default 32) routed through the C++
        # `assemble_Z_enrich` accelerator.
        self.use_singular_enrichment = bool(use_singular_enrichment)
        self.n_qp_sing = int(n_qp_sing)
        self.enrichment_min_k = int(enrichment_min_k)
        # `enrichment_variant` picks the singular basis shape:
        #   "raw"    → Φ_sing(t) = t·log(t), the unmodified PR #47 shape.
        #             Mixed behavior: captures real cusps where they exist
        #             (balanced K=3 like Y-fixture: ~0.08 Ω R correction)
        #             but also absorbs polynomial discretization error at
        #             small N (hentenna n=21: ~0.26 Ω X transient post-#51
        #             sign fix). PR #45 / #47 default.
        #   "stable" → Φ_sing_stable(t) = t·log(t) − P_bubble(t·log(t)) where
        #             P_bubble is the L²-orthogonal projection onto the
        #             polynomial bubble subspace of P_d that vanishes at
        #             both t=0 and t=1 (preserves Φ_sing's endpoint BCs;
        #             required because the KCL constraint only sees the
        #             polynomial bases). Trade-offs measured on probe
        #             scripts: hentenna large-N converges faster
        #             (X-rate p≈4.10 vs raw's 2.7); fan-dipole gap closes
        #             to noise floor; **hentenna small-N gets a larger
        #             transient (~0.79 Ω X at n=21)**; Y-fixture loses
        #             its 0.08 Ω R cusp benefit. Not "universally safe."
        #             For d=1 the bubble subspace is empty so "stable"
        #             reduces to "raw" identically.
        # "tikhonov" → raw Φ_sing basis, but add λ·s·I to the enrichment
        # block Z_ee at solve time, where s is the average diagonal
        # magnitude of Z_ee (so λ is a dimensionless relative-strength
        # knob). Penalizes ||α_enr||² in the augmented objective; shrinks
        # spurious-large α at small N without re-deriving the basis.
        # λ → 0 ⇒ raw; λ → ∞ ⇒ enrichment effectively off.
        # "auto" → two-pass selectivity. Solve once without enrichment,
        # measure tap_ratio = min|I_wire|/max|I_wire| over wires meeting
        # at each K≥enrichment_min_k junction (the same diagnostic as
        # scripts/probe_k3_junction_imbalance.py), and apply raw
        # enrichment only at junctions where tap_ratio exceeds
        # `auto_tap_ratio_threshold`. Cleanly separates dominant-pair
        # K=3 (hentenna ≈ 0.16, fan-dipole ≈ 0.03) from balanced 3-way
        # (Y-fixture ≈ 0.50). One extra solve per compute_impedance
        # call; second solve skipped if no junction qualifies.
        if enrichment_variant not in ("raw", "stable", "tikhonov", "auto"):
            raise ValueError(
                f"enrichment_variant must be one of 'raw', 'stable', "
                f"'tikhonov', 'auto', got {enrichment_variant!r}"
            )
        self.enrichment_variant = enrichment_variant
        self.tikhonov_lambda = float(tikhonov_lambda)
        self.auto_tap_ratio_threshold = float(auto_tap_ratio_threshold)
        # Populated by compute_impedance when variant="auto" runs the
        # two-pass solve so currents_at_knots can index the enrichment
        # block consistently. None means "ignore auto-selection".
        self._auto_active_junctions = None

        self.junctions = []
        if junctions is not None:
            for j, jw in enumerate(junctions):
                # A 1-entry group is legal (issue #172): as a plain junction
                # it enforces I_end = 0 through the KCL row — numerically a
                # free end — and as a junction PORT it is the natural form
                # of a lone-conductor-end attachment.
                if len(jw) < 1:
                    raise ValueError(f"junction {j}: need >= 1 wire-end")
                normalized = []
                for w, end in jw:
                    if not (0 <= w < n_w):
                        raise ValueError(
                            f"junction {j}: wire_idx {w} out of range [0, {n_w})"
                        )
                    if end not in ("start", "end"):
                        raise ValueError(
                            f"junction {j}: end must be 'start' or 'end', got {end!r}"
                        )
                    normalized.append((int(w), end))
                self.junctions.append(normalized)

        # Junction ports (issue #172): junction groups promoted to network
        # ports. Each entry is (junction_index, voltage) — a plain int means
        # voltage 0. A port junction's KCL closure row leaves the constraint
        # set (the grounded-junction #151 move) and becomes the port's
        # source/readout vector instead: voltage V drives the excitation
        # v += V·A_p, and the port current reads I_p = A_p·coeffs — the same
        # Galerkin-reciprocity pairing as delta-gap feeds, so mixed-port Y
        # matrices stay symmetric by construction. Ports are ordered after
        # the gap feeds everywhere: [feeds..., junction_ports...].
        self.junction_ports = []
        if junction_ports is not None:
            seen = set()
            for p in junction_ports:
                if isinstance(p, (int, np.integer)):
                    j_idx, volt = int(p), 0j
                else:
                    j_idx, volt = int(p[0]), complex(p[1])
                if not (0 <= j_idx < len(self.junctions)):
                    raise ValueError(
                        f"junction_ports: junction index {j_idx} out of range "
                        f"[0, {len(self.junctions)})"
                    )
                if j_idx in seen:
                    raise ValueError(f"junction_ports: junction {j_idx} listed twice")
                seen.add(j_idx)
                self.junction_ports.append((j_idx, volt))

        # Node gaps (issue #305): a SERIES delta-gap EMF at a junction node,
        # in series with a named wire-end — the apex feed. Each entry is
        # (wire_index, "start"|"end", voltage). The named end must belong to
        # a junction group; the junction's KCL row STAYS in the constraint
        # set (current is continuous through a series EMF — contrast a #172
        # junction port, whose row leaves it to drive net inflow). The port's
        # drive/readout vector is the σ-signed unit indicator of that
        # wire-end's directional basis (σ = +1 start / −1 end, the KCL
        # outflow sign), so I_port is the current flowing from the node into
        # the named wire and the pairing is Galerkin-reciprocal — mixed-port
        # Y stays symmetric alongside gap feeds and junction ports. At a
        # degree-2 vertex the member choice does not change Z (through
        # current, both orientations agree); at degree ≥ 3 it names which
        # arm the gap separates, NEC-5's tag/segment/end addressing. Ports
        # order [feeds..., junction_ports..., node_gaps...].
        self.node_gaps = []
        if node_gaps is not None:
            end_to_junction = {}
            for j, jw in enumerate(self.junctions):
                for member in jw:
                    end_to_junction[member] = j
            seen_members = set()
            seen_junctions = set()
            ported = {j for j, _v in self.junction_ports}
            for i, g in enumerate(node_gaps):
                if len(g) != 3:
                    raise ValueError(
                        f"node_gaps[{i}]: expected (wire_index, 'start'|'end',"
                        f" voltage), got {g!r}"
                    )
                w_i, end_i, v_i = g
                if not (0 <= w_i < n_w):
                    raise ValueError(
                        f"node_gaps[{i}]: wire_index {w_i} out of range [0, {n_w})"
                    )
                if end_i not in ("start", "end"):
                    raise ValueError(
                        f"node_gaps[{i}]: end must be 'start' or 'end', got {end_i!r}"
                    )
                member = (int(w_i), end_i)
                j_idx = end_to_junction.get(member)
                if j_idx is None:
                    raise ValueError(
                        f"node_gaps[{i}]: wire {w_i} {end_i!r} is not a member "
                        "of any junction group — a series node gap lives at a "
                        "junction; for a feed inside a wire use feeds="
                    )
                if len(self.junctions[j_idx]) < 2:
                    raise ValueError(
                        f"node_gaps[{i}]: junction {j_idx} has a single member "
                        "— there is no through-current path to be in series "
                        "with (a lone-end attachment is junction_ports=)"
                    )
                if member in seen_members:
                    raise ValueError(
                        f"node_gaps[{i}]: wire {w_i} {end_i!r} listed twice"
                    )
                if j_idx in seen_junctions:
                    raise ValueError(
                        f"node_gaps[{i}]: junction {j_idx} already carries a "
                        "node gap — one series gap per junction"
                    )
                if j_idx in ported:
                    raise ValueError(
                        f"node_gaps[{i}]: junction {j_idx} is also a junction "
                        "port — the shunt (#172) and series (#305) ports of "
                        "one node cannot be driven together yet"
                    )
                seen_members.add(member)
                seen_junctions.add(j_idx)
                self.node_gaps.append((int(w_i), end_i, complex(v_i)))

        # `compute_impedance(...)` and `currents_at_knots(coeffs)` both call
        # `_build_geometry()` + `_build_basis_polynomials(geom)` from scratch
        # — on the N=21 hentenna width-sweep harness that's ~90 ms/step of
        # repeated Python work (see scripts/vtune_hentenna_width_sweep.py).
        # Cache both on the instance. Neither depends on `k` (only on the
        # immutable geometry inputs + degree + junctions), so
        # `compute_impedance_swept`'s per-k loop also benefits.
        self._cached_geometry: dict | None = None
        self._cached_basis_polynomials: tuple | None = None
        # (geom, specular tables) for the ground_eps weighted image — same
        # lifetime as the geometry cache; k-independent, so swept solves
        # reuse it across the whole frequency loop.
        self._cached_image_refl_prep: tuple | None = None
        # SommerfeldGrid lives in the module-level cache in
        # `_sommerfeld.get_grid` so it survives across solver instances.

    # ------------------------------------------------------------------
    # Geometry build
    # ------------------------------------------------------------------

    def _build_geometry(self):
        """Discretize all wires, concatenate to global arrays.

        Per-wire metadata is preserved so the basis-polynomial extraction
        (which operates on each wire's clamped knot vector independently)
        can be done wire-by-wire.

        Returns a `geom` dict with:
          per_wire: list of per-wire dicts (seg_l, seg_r, tangents, h_per_seg,
              edge_offsets, edge_arc_edges, arc_at_knot, n_total)
          seg_offsets: list[n_w+1] of global segment index of wire start
          n_segs_total: total segment count across all wires
          h_per_seg: (N_total,) per-segment edge length
          tangents: (N_total, 3) per-segment tangent unit vector
          seg_l, seg_r: (N_total, 3) per-segment 3D endpoint
        """
        if self._cached_geometry is not None:
            return self._cached_geometry
        # Module cache hit → reuse the geom dict identity-stably across
        # solver instances. The basis-polynomial instance cache keys on
        # `cached_geom is geom`, so returning the same object also lets
        # the basis-poly cache resolve through the module path below.
        geom_key = self._geometry_cache_key()
        cached = _GEOMETRY_CACHE.get(geom_key)
        if cached is not None:
            self._cached_geometry = cached
            return cached
        per_wire = []
        seg_offsets = [0]
        h_list = []
        tangents_list = []
        seg_l_list_all = []
        seg_r_list_all = []
        for w_idx, (pl, npe_list) in enumerate(
            zip(self.wires_polylines, self.n_per_edge_per_wire)
        ):
            seg_l_w = []
            seg_r_w = []
            tan_w = []
            h_w_list = []
            edge_offsets = [0]
            edge_arc_edges = []
            for e_idx in range(pl.shape[0] - 1):
                p0 = pl[e_idx]
                p1 = pl[e_idx + 1]
                edge_vec = p1 - p0
                edge_len = float(np.linalg.norm(edge_vec))
                if edge_len < 1e-15:
                    raise ValueError(f"wire {w_idx} edge {e_idx} has zero length")
                tan = edge_vec / edge_len
                n_e = npe_list[e_idx]
                h_e = edge_len / n_e

                t_node = np.linspace(0.0, 1.0, n_e + 1)
                pts = (1 - t_node[:, None]) * p0[None, :] + t_node[:, None] * p1[
                    None, :
                ]
                seg_l_w.append(pts[:-1])
                seg_r_w.append(pts[1:])
                tan_w.append(np.tile(tan, (n_e, 1)))
                h_w_list.append(np.full(n_e, h_e))
                edge_arc_edges.append(np.linspace(0.0, edge_len, n_e + 1))
                edge_offsets.append(edge_offsets[-1] + n_e)

            seg_l = np.vstack(seg_l_w)
            seg_r = np.vstack(seg_r_w)
            tangents_w = np.vstack(tan_w)
            h_per_seg_w = np.concatenate(h_w_list)
            n_total_w = seg_l.shape[0]
            arc_at_knot = np.concatenate([[0.0], np.cumsum(h_per_seg_w)])

            per_wire.append(
                {
                    "seg_l": seg_l,
                    "seg_r": seg_r,
                    "tangents": tangents_w,
                    "h_per_seg": h_per_seg_w,
                    "edge_offsets": edge_offsets,
                    "edge_arc_edges": edge_arc_edges,
                    "arc_at_knot": arc_at_knot,
                    "n_total": n_total_w,
                }
            )
            seg_offsets.append(seg_offsets[-1] + n_total_w)
            h_list.append(h_per_seg_w)
            tangents_list.append(tangents_w)
            seg_l_list_all.append(seg_l)
            seg_r_list_all.append(seg_r)

        h_per_seg_global = np.concatenate(h_list)
        tangents_global = np.vstack(tangents_list)
        seg_l_global = np.vstack(seg_l_list_all)
        seg_r_global = np.vstack(seg_r_list_all)

        self._cached_geometry = {
            "per_wire": per_wire,
            "seg_offsets": seg_offsets,
            "n_segs_total": seg_offsets[-1],
            "h_per_seg": h_per_seg_global,
            "tangents": tangents_global,
            "seg_l": seg_l_global,
            "seg_r": seg_r_global,
        }
        _evict_fifo(_GEOMETRY_CACHE, _GEOMETRY_CACHE_MAX)
        _GEOMETRY_CACHE[geom_key] = self._cached_geometry
        return self._cached_geometry

    def _geometry_cache_key(self):
        # Bytes view of each wire's float64 polyline + per-wire segmentation.
        # Both are immutable post-__init__, and the geom dict depends on
        # exactly these (see _build_geometry body).
        return (
            tuple(w.tobytes() for w in self.wires_polylines),
            tuple(tuple(npe) for npe in self.n_per_edge_per_wire),
        )

    # ------------------------------------------------------------------
    # Endpoint status (free vs junction)
    # ------------------------------------------------------------------

    def _wire_endpoint_status(self):
        """For each wire, return the (start, end) endpoint condition:
        "free", "ground", or the index of the junction connecting it.

        "ground" marks an un-junctioned endpoint lying in an active ground
        plane (|z − ground_z| ≤ 1e-6 of the wire's length): the wire is
        electrically connected to ground, so its end current must NOT be
        pinned to zero — the image supplies the return path/continuation
        (issue #151). A *junctioned* endpoint at ground keeps its junction
        index; the grounded-junction handling (KCL row dropped — current may
        flow into ground) lives in `_grounded_junctions`.
        """
        n_w = len(self.wires_polylines)
        start_status = ["free"] * n_w
        end_status = ["free"] * n_w
        for j_idx, jw in enumerate(self.junctions):
            for w, end in jw:
                if end == "start":
                    start_status[w] = j_idx
                else:
                    end_status[w] = j_idx
        gz = self.ground_z
        if gz is not None:
            for w_idx, pl in enumerate(self.wires_polylines):
                tol = self._ground_touch_tol(pl)
                pl_arr = np.asarray(pl, dtype=np.float64)
                if float(pl_arr[:, 2].min()) < gz - tol:
                    raise ValueError(
                        f"wire {w_idx} dips below the ground plane "
                        f"(min z = {pl_arr[:, 2].min():.6g} < ground_z = {gz:g})"
                    )
                z_at = np.abs(pl_arr[:, 2] - gz) <= tol
                if np.any(z_at[:-1] & z_at[1:]):
                    raise ValueError(
                        f"wire {w_idx} has an edge lying in the ground plane "
                        "(both endpoints at ground_z) — degenerate over a "
                        "conducting ground"
                    )
                if start_status[w_idx] == "free" and z_at[0]:
                    start_status[w_idx] = "ground"
                if end_status[w_idx] == "free" and z_at[-1]:
                    end_status[w_idx] = "ground"
        return start_status, end_status

    def _ground_touch_tol(self, polyline):
        """Snap distance for "this endpoint touches the ground plane":
        1e-6 of the wire's polyline length — loose enough for deck-import
        float noise at z=0, far tighter than any deliberate clearance (a
        1 mm stand-off on a 10 m vertical is 100× the tolerance)."""
        pl = np.asarray(polyline, dtype=np.float64)
        length = float(np.sum(np.linalg.norm(np.diff(pl, axis=0), axis=1)))
        return 1e-6 * max(length, 1e-30)

    def _grounded_junctions(self):
        """Indices of junctions whose shared point lies in the ground plane.

        Their KCL row (Σ signed outflows = 0) is dropped: at a grounded
        node current may flow into the ground stake (completed by the
        image), so enforcing closure among the real wires alone would be
        wrong physics."""
        gz = self.ground_z
        if gz is None or not self.junctions:
            return frozenset()
        grounded = set()
        for j_idx, jw in enumerate(self.junctions):
            w, end = jw[0]
            pl = self.wires_polylines[w]
            pt = pl[0] if end == "start" else pl[-1]
            if abs(pt[2] - gz) <= self._ground_touch_tol(pl):
                grounded.add(j_idx)
        return frozenset(grounded)

    def _split_kcl_ports(self, kcl_A):
        """Split the assembled KCL matrix into (constraint rows, port rows,
        port voltages) per `self.junction_ports` (issue #172).

        `_build_basis_polynomials` emits one KCL row per non-grounded
        junction, in junction-index order; a junction port's row moves from
        the constraint set to the port set. Returns
        ``(kcl_con, port_A, port_V)`` where ``port_A`` rows follow
        `self.junction_ports` order and ``port_V`` is the matching complex
        voltage vector. With no junction ports this is
        ``(kcl_A, (0, n) empty, (0,) empty)`` — the exact passthrough.
        """
        if not self.junction_ports:
            return (
                kcl_A,
                np.zeros((0, kcl_A.shape[1]), dtype=np.float64),
                np.zeros(0, dtype=np.complex128),
            )
        grounded = self._grounded_junctions()
        row_of = {}
        row = 0
        for j in range(len(self.junctions)):
            if j not in grounded:
                row_of[j] = row
                row += 1
        assert row == kcl_A.shape[0], (row, kcl_A.shape)
        port_rows = []
        for j_idx, _v in self.junction_ports:
            if j_idx in grounded:
                raise ValueError(
                    f"junction {j_idx} is both grounded and a junction port — "
                    "a grounded node's voltage is pinned by the ground image, "
                    "so it cannot also be a driven port"
                )
            port_rows.append(row_of[j_idx])
        keep = [r for r in range(kcl_A.shape[0]) if r not in set(port_rows)]
        kcl_con = kcl_A[keep, :]
        port_A = kcl_A[port_rows, :]
        port_V = np.array([v for _j, v in self.junction_ports], dtype=np.complex128)
        return kcl_con, port_A, port_V

    # ------------------------------------------------------------------
    # Basis polynomial extraction
    # ------------------------------------------------------------------

    def _build_basis_polynomials(self, geom):
        """Extract polynomial coefficients per (basis, wing).

        For each wire:
          * Build clamped knot vector on the wire's cumulative arc.
          * Determine which of the d+1 boundary bases per end are kept:
              - Free end: drop all d+1 boundary bases (Φ(end) = 0 strictly,
                  AND derivative 0, etc. — for d ≤ 2 this means drop just
                  B_0 because only B_0 has nonzero value, and the higher
                  boundary bases are kept as ordinary interior bases since
                  their value at the end is 0).
              - Junction end: keep the value-1 boundary basis B_0 as a
                  directional basis; keep B_1..B_{d-1} as interior bases.
          * Extract per-segment polynomial coefficients via BSpline +
            Vandermonde (uniform within each segment's local-u range).

        Returns
        -------
        supp_seg, polys : as in the single-wire case, concatenated globally.
        kcl_A : (n_junctions, n_basis_total) Lagrange-multiplier rows
            (+1 / -1 outflow sign per directional basis).
        wire_knots : list of per-wire knot vectors (for the source vector).
        wire_basis_global : list of per-wire (kept_idx, global_basis_idx)
            tuples for the source-vector mapping.
        """
        # Cache key is geometry identity: the result depends only on `geom`
        # (per-wire arc knots), `self.degree`, and `self.junctions` (via
        # _wire_endpoint_status); none change after __init__, so a cached
        # result computed against the same geom dict is still valid.
        cached_geom = self._cached_geometry
        if cached_geom is geom and self._cached_basis_polynomials is not None:
            return self._cached_basis_polynomials
        # Module cache promotes the per-instance memoization across solver
        # instances (the engine wrapper recreates the solver per impedance()
        # call). Key is geometry signature + degree + junctions; the result
        # is k-independent.
        basis_key = (
            self._geometry_cache_key(),
            self.degree,
            tuple(tuple((w, e) for (w, e) in j) for j in self.junctions),
            # Endpoint conditions depend on the ground plane (ground ends /
            # grounded junctions, #151); geometry alone no longer keys them.
            self.ground_z,
        )
        cached_basis = _BASIS_POLY_CACHE.get(basis_key)
        if cached_basis is not None:
            if cached_geom is geom:
                self._cached_basis_polynomials = cached_basis
            return cached_basis
        d = self.degree
        n_wings = d + 1
        n_poly = d + 1

        start_status, end_status = self._wire_endpoint_status()

        all_supp_seg = []
        all_polys = []
        wire_knots = []
        wire_basis_global = []
        # Track per-junction the list of (directional-basis global idx,
        # outflow sign).
        junction_dirs = {j: [] for j in range(len(self.junctions))}

        m_global = 0
        for w_idx, pw in enumerate(geom["per_wire"]):
            arc = pw["arc_at_knot"]
            wire_arc = arc[-1]
            interior_knots = arc.copy()
            knots = np.concatenate(
                [np.full(d, 0.0), interior_knots, np.full(d, wire_arc)]
            )
            wire_knots.append(knots)
            n_basis_w = len(knots) - d - 1  # = N_w + d

            # Determine kept bases. For d ∈ {1, 2}:
            #   B_0 is the value-1 boundary basis at the start
            #   B_{n_basis_w - 1} is the value-1 boundary basis at the end
            #   B_1, ..., B_{n_basis_w - 2} are interior (value 0 at endpoints)
            kept = []  # list of (basis_j, kind, junction_idx-or-None)
            # Start boundary basis (B_0)
            if start_status[w_idx] == "free":
                pass  # drop
            elif start_status[w_idx] == "ground":
                # Ground junction: keep the value-1 end basis so the end
                # current is a real dof — its image (integrated by the
                # ground blocks like every basis's) is the continuation
                # through the plane. No KCL partner: the image IS the
                # return path.
                kept.append((0, "gnd", None, "start"))
            else:
                kept.append((0, "dir", start_status[w_idx], "start"))
            # Truly interior bases
            for j in range(1, n_basis_w - 1):
                kept.append((j, "int", None, None))
            # End boundary basis (B_{n_basis_w - 1})
            if end_status[w_idx] == "free":
                pass  # drop
            elif end_status[w_idx] == "ground":
                kept.append((n_basis_w - 1, "gnd", None, "end"))
            else:
                kept.append((n_basis_w - 1, "dir", end_status[w_idx], "end"))

            seg_off = geom["seg_offsets"][w_idx]
            h_per_seg_w = pw["h_per_seg"]
            arc_at_knot_w = pw["arc_at_knot"]
            n_total_w = pw["n_total"]

            # Vectorize: per-wire single BSpline.design_matrix + constant
            # V_unit_inv lookup replaces per-(basis, wing) BSpline
            # construction + per-segment linspace/vander/solve.
            #
            # Sample points: d+1 uniform u within each segment, in global
            # arc. shape (n_total_w, d+1) → flatten for design_matrix.
            unit = np.linspace(0.0, 1.0, d + 1)  # (d+1,) shared across segs
            u_local_per_seg = h_per_seg_w[:, None] * unit[None, :]  # (N, d+1)
            u_global_per_seg = arc_at_knot_w[:-1, None] + u_local_per_seg
            u_flat = u_global_per_seg.reshape(-1)

            # All basis values at all sample points in one design_matrix
            # call, kept in the (n_total_w, d+1, d+1) band: band column b
            # of segment s is basis s+b, the only bases that segment sees.
            DM_seg = _design_matrix_band(
                BSpline.design_matrix(u_flat, knots, d), n_total_w, d
            )

            # V_unit_inv @ vals: convert d+1 basis values per segment to
            # poly coeffs (in u_local). Then divide by h_seg^p column-wise
            # to recover coeffs in u_local = h_seg · u_unit terms.
            V_unit_inv = _V_UNIT_INV[d]
            inv_h_powers = h_per_seg_w[:, None] ** (-np.arange(d + 1))
            # → (N, d+1, d+1): for each segment, polynomial coeff p of
            # each in-band basis expressed as Σ_p coeffs_p · u_local^p
            poly_per_seg = np.einsum("ij,sjk->sik", V_unit_inv, DM_seg)
            poly_per_seg *= inv_h_powers[:, :, None]

            # Per-basis support range as half-open segment indices [lo, hi).
            # knots = [0]*d + arc + [wire_arc]*d, so knots[j] sits at
            # segment index max(0, j - d), and knots[j+d+1] sits at
            # min(N, j+1). Result: basis j has wings = segments
            # max(0, j-d) .. min(N, j+1) - 1.

            per_basis_local_to_global = {}
            for kept_idx, (j, kind, junc_idx, end_pos) in enumerate(kept):
                seg_lo = max(0, j - d)
                seg_hi = min(n_total_w, j + 1)
                n_actual = seg_hi - seg_lo

                supp_seg_m = np.zeros(n_wings, dtype=np.int64)
                polys_m = np.zeros((n_wings, n_poly), dtype=np.float64)
                seg_rows = np.arange(seg_lo, seg_hi)
                supp_seg_m[:n_actual] = seg_off + seg_rows
                # Basis j sits at band column j - s of segment s, which
                # walks d, d-1, ... down the wing (or j, j-1, ... for the
                # clamped start bases, whose support is truncated).
                polys_m[:n_actual, :] = poly_per_seg[
                    seg_rows[:, None],
                    np.arange(n_poly)[None, :],
                    (j - seg_rows)[:, None],
                ]

                all_supp_seg.append(supp_seg_m)
                all_polys.append(polys_m)
                per_basis_local_to_global[kept_idx] = m_global

                if kind == "dir":
                    sign = +1.0 if end_pos == "start" else -1.0
                    junction_dirs[junc_idx].append((m_global, sign))

                m_global += 1

            wire_basis_global.append((kept, per_basis_local_to_global))

        supp_seg = (
            np.stack(all_supp_seg, axis=0)
            if all_supp_seg
            else (np.zeros((0, n_wings), dtype=np.int64))
        )
        polys = (
            np.stack(all_polys, axis=0)
            if all_polys
            else (np.zeros((0, n_wings, n_poly), dtype=np.float64))
        )
        n_basis_total = supp_seg.shape[0]

        # Grounded junctions keep their directional bases but lose the KCL
        # closure row — current may leave through the ground image (#151).
        grounded = self._grounded_junctions()
        kcl_rows = [j for j in range(len(self.junctions)) if j not in grounded]
        kcl_A = np.zeros((len(kcl_rows), n_basis_total), dtype=np.float64)
        for row, j_idx in enumerate(kcl_rows):
            for m_g, sign in junction_dirs[j_idx]:
                kcl_A[row, m_g] = sign

        result = (supp_seg, polys, kcl_A, wire_knots, wire_basis_global)
        if cached_geom is geom:
            self._cached_basis_polynomials = result
        _evict_fifo(_BASIS_POLY_CACHE, _BASIS_POLY_CACHE_MAX)
        _BASIS_POLY_CACHE[basis_key] = result
        return result

    # ------------------------------------------------------------------
    # J moment integrals
    # ------------------------------------------------------------------

    def _image_positions(self, positions):
        """Mirror an array of 3D positions across z = ground_z."""
        out = positions.copy()
        out[..., 2] = 2 * self.ground_z - out[..., 2]
        return out

    def _image_tangent_dot(self, tangents):
        """t_m · t_image_n with t_image_n = (t_n_x, t_n_y, -t_n_z)."""
        return tangents @ (tangents * np.array([1.0, 1.0, -1.0])).T

    def _build_J_image_blocks(self, geom, k):
        """Build the J moment tensor with j-segments mirrored across the
        PEC ground plane. The image is always far enough from the original
        that the analytic same-edge static + reg split doesn't apply — full
        off-edge quadrature handles every (i, j) pair uniformly.
        """
        d = self.degree
        seg_l = geom["seg_l"]
        seg_r = geom["seg_r"]
        seg_l_img = self._image_positions(seg_l)
        seg_r_img = self._image_positions(seg_r)
        # Observers (rows) are the real segments — the per-observer radius
        # convention applies to the image block unchanged. Under EK the
        # source side is the MIRRORED geometry, so eligibility is scored
        # against it (`mirror=True`): a vertical monopole is coaxial with
        # its own image and extends, a horizontal wire is not and does not.
        return _seg_seg_full_moments_offedge(
            seg_l,
            seg_r,
            seg_l_img,
            seg_r_img,
            self._seg_radius(geom),
            k,
            d,
            self.n_qp_pair,
            ek=self._ek_spec(geom, mirror=True) if self.extended_kernel else None,
        )

    def _image_refl_prep(self, geom):
        """k-independent per-pair specular tables (cos θ, PEC mirror dot,
        out-of-plane dyad component) for the `ground_eps` weighted image.
        Cached per geometry object so swept callers pay for the O(N²)
        build once, not per frequency.
        """
        cached = self._cached_image_refl_prep
        if cached is not None and cached[0] is geom:
            return cached[1]
        seg_c = 0.5 * (geom["seg_l"] + geom["seg_r"])
        tables = _ground_refl.specular_pair_tables(
            seg_c, geom["tangents"], self.ground_z
        )
        self._cached_image_refl_prep = (geom, tables)
        return tables

    def _image_refl_weights(self, prep, omega):
        """Per-frequency weight tables from the k-independent specular prep:
        ε̃(ω) → ρ_v/ρ_h at each pair's specular angle → A-term dyad table
        w_A and Φ-term image-charge table w_Φ (mode: `ground_phi_mode`).
        """
        cos_th, td_img, P = prep
        eps_t = _ground_refl.eps_tilde(self.ground_eps, omega, self.eps)
        rho_v, rho_h = _ground_refl.fresnel_rho(eps_t, cos_th)
        w_A = _ground_refl.a_term_weights(rho_v, rho_h, td_img, P)
        w_Phi = _ground_refl.phi_term_weights(self.ground_phi_mode, eps_t, rho_v)
        if np.ndim(w_Phi) == 0:
            w_Phi = np.full(w_A.shape, complex(w_Phi))
        return w_A, w_Phi

    def _image_weight_window_fn(self, geom):
        """Producer of the image weight WINDOWS the chunked accumulator
        consumes: `weights_fn(i0, i1) -> (w_A, w_Phi)`, each complex128 of
        shape (i1-i0, n_segs) — observer rows [i0, i1) against every source.

        The chunked path used to be handed the same global (N, N) tables the
        tensor path builds and slice them per chunk, which put 2× the dense
        Z on the peak of every grounded solve for weights only ever read one
        row-band at a time (issue #323). Each mode's algebra is row-local, so
        the window is produced directly and nothing N² is ever allocated:

          PEC        — the mirror tangent dot t_m·(M·t_n) on A, unit charge;
          sommerfeld — the same dot scaled by the exact-image constant
                       C2 = (ε̃−1)/(ε̃+1), with a constant C2 charge weight
                       (the smooth remainder is a separate, already-chunked
                       term);
          refl-coef  — the rectangular form of the Fresnel producers
                       (`specular_pair_tables` already takes an observer×
                       source block), so the k-independent N² specular cache
                       `_image_refl_prep` is bypassed entirely here. It stays
                       for the tensor and enrichment paths, which do want the
                       whole table.

        ε̃ (and with it C2) is frequency- but not row-dependent, so it is
        computed once when the closure is built.
        """
        tangents = geom["tangents"]
        mirror = np.array([1.0, 1.0, -1.0])

        if self.ground_eps is None:

            def pec_weights(i0, i1):
                # float64 gemm → C-contiguous; astype gives the complex128
                # the assembler wants without a wrapper copy on top.
                w_A = (tangents[i0:i1] @ (tangents * mirror).T).astype(np.complex128)
                return w_A, np.ones_like(w_A)

            return pec_weights

        eps_t = _ground_refl.eps_tilde(self.ground_eps, self.omega, self.eps)

        if self.ground_model == "sommerfeld":
            c2 = (eps_t - 1.0) / (eps_t + 1.0)

            def sommerfeld_weights(i0, i1):
                # complex scalar × float64 array → complex128 C-contiguous.
                w_A = c2 * (tangents[i0:i1] @ (tangents * mirror).T)
                return w_A, np.full(w_A.shape, c2)

            return sommerfeld_weights

        seg_c = 0.5 * (geom["seg_l"] + geom["seg_r"])
        phi_mode = self.ground_phi_mode

        def refl_weights(i0, i1):
            cos_th, td_img, P = _ground_refl.specular_pair_tables(
                seg_c[i0:i1],
                tangents[i0:i1],
                self.ground_z,
                src_centers=seg_c,
                src_tangents=tangents,
            )
            rho_v, rho_h = _ground_refl.fresnel_rho(eps_t, cos_th)
            w_A = _ground_refl.a_term_weights(rho_v, rho_h, td_img, P)
            w_Phi = _ground_refl.phi_term_weights(phi_mode, eps_t, rho_v)
            if np.ndim(w_Phi) == 0:
                # "image"/"normal" are pair-independent; "rho_v"/"blend"
                # already come back as per-pair windows.
                w_Phi = np.full(w_A.shape, complex(w_Phi))
            return w_A, w_Phi

        return refl_weights

    def _image_Z_refl(self, J_img, supp_seg, polys, geom):
        """Fresnel-weighted image sub-assembly for `ground_eps` (NEC-style
        reflection-coefficient finite ground). Returns the matrix to SUBTRACT
        from the free-space Z — same global-minus convention as the PEC
        image, which this reproduces exactly in the ε̃ → ∞ limit.

        Structure mirrors `_assemble_Z`'s numpy fallback, with two per-
        segment-pair weight tables instead of one: the A term takes the
        Fresnel dyad tangent table w_A (in place of the PEC mirror tangent
        dot), and the Φ term — which the PEC path leaves unweighted — takes
        the per-pair image-charge weight w_Φ picked by `ground_phi_mode`.

        Hot path is the C++ `assemble_Z_bspline_weighted` — the PEC assembly
        kernel with complex per-pair weight tables on both terms, added in
        Phase 2 after profiling showed Python-side weighted assembly at
        ~3× the PEC image assembly cost on a 41-freq grounded sweep
        (scripts/perf_refl_coef_sweep.py). The einsum loop below stays as
        the no-accelerator fallback and bit-exact reference.
        """
        w_A, w_Phi = self._image_refl_weights(self._image_refl_prep(geom), self.omega)
        return self._image_Z_weighted(J_img, supp_seg, polys, w_A, w_Phi)

    def _image_Z_weighted(self, J_img, supp_seg, polys, w_A, w_Phi):
        """Weighted image assembly core: complex per-pair tables w_A on the
        A term and w_Phi on the charge term, through the C++
        `assemble_Z_bspline_weighted` kernel when available with the numpy
        einsum loop as the bit-exact fallback. Shared by the refl-coef
        ground (Fresnel tables) and the Sommerfeld ground's exact-image
        part (constant C2 tables)."""
        d = self.degree
        if _HAVE_BSPLINE_ASSEMBLE_W_ACCEL and d <= _BSPLINE_ASSEMBLE_ACCEL_MAX_D:
            return _acc.assemble_Z_bspline_weighted(
                np.ascontiguousarray(J_img, dtype=np.complex128),
                np.ascontiguousarray(supp_seg, dtype=np.int64),
                np.ascontiguousarray(polys, dtype=np.float64),
                np.ascontiguousarray(w_A, dtype=np.complex128),
                np.ascontiguousarray(w_Phi, dtype=np.complex128),
                float(self.omega),
                float(self.eps),
                float(self.mu),
                int(d),
                self._cancel_flag,
            )

        n_basis, n_wings, n_poly = polys.shape
        assert n_wings == d + 1 and n_poly == d + 1
        Z_A = np.zeros((n_basis, n_basis), dtype=np.complex128)
        Z_Phi = np.zeros((n_basis, n_basis), dtype=np.complex128)
        p_vec = np.arange(1, d + 1, dtype=np.float64) if d >= 1 else None

        for a in range(n_wings):
            sm = supp_seg[:, a]
            for b in range(n_wings):
                sn = supp_seg[:, b]
                J_blk = J_img[:, :, sm[:, None], sn[None, :]]
                wA_blk = w_A[sm[:, None], sn[None, :]]

                inner_A = np.einsum(
                    "mp,pPmn,nP->mn", polys[:, a, :], J_blk, polys[:, b, :]
                )
                Z_A += wA_blk * inner_A

                if d >= 1:
                    wPhi_blk = w_Phi[sm[:, None], sn[None, :]]
                    deriv_m = polys[:, a, 1:] * p_vec[None, :]
                    deriv_n = polys[:, b, 1:] * p_vec[None, :]
                    J_blk_lo = J_blk[:d, :d]
                    inner_Phi = np.einsum("mp,pPmn,nP->mn", deriv_m, J_blk_lo, deriv_n)
                    Z_Phi += wPhi_blk * inner_Phi

        Z_A = 1j * self.omega * self.mu * Z_A
        Z_Phi = Z_Phi / (1j * self.omega * self.eps)
        return Z_A + Z_Phi

    def _ground_finite_Z(self, J_img, supp_seg, polys, geom):
        """Finite-ground matrix to SUBTRACT from the free-space Z (the
        seams' `Z - ...` convention, shared with the PEC image).

        refl-coef: the Fresnel-weighted image (`_image_Z_refl`).

        sommerfeld: NEC's decomposition (theory manual eqs 136-147) — the
        exact image scaled by the constant C2 = (eps-1)/(eps+1), which
        absorbs all the singular behavior and reuses the weighted-image
        kernel with constant tables, plus the smooth Sommerfeld remainder
        block. In the eps->inf limit C2 -> 1 and the remainder vanishes,
        reproducing the PEC image exactly; at eps -> 1 both terms vanish,
        reproducing free space. Both limits are unit-tested.
        """
        if self.ground_model != "sommerfeld":
            return self._image_Z_refl(J_img, supp_seg, polys, geom)
        eps_t = _ground_refl.eps_tilde(self.ground_eps, self.omega, self.eps)
        c2 = (eps_t - 1.0) / (eps_t + 1.0)
        td_img = self._image_tangent_dot(geom["tangents"])
        w_A = c2 * td_img.astype(np.complex128)
        w_Phi = np.full_like(w_A, c2)
        M = self._image_Z_weighted(J_img, supp_seg, polys, w_A, w_Phi)
        return M + self._Z_sommerfeld_remainder(geom, supp_seg, polys, eps_t)

    def _somm_grid(self, eps_t, r1_max):
        return _sommerfeld.get_grid(
            eps_t,
            self.k,
            r1_max,
            omega=self.omega,
            mu=self.mu,
            cancel_flag=self._cancel_flag,
        )

    def _Z_sommerfeld_remainder(self, geom, supp_seg, polys, eps_t):
        """Galerkin block Q[m,n] = ∫∫ f_m f_n · t_m·F(r, r')·t_n of the
        smooth Sommerfeld remainder field F (theory manual eqs 143-147:
        the ground field minus its C2-scaled exact-image part). The EFIE
        contribution is -Q; `_ground_finite_Z` returns C2-image + Q so the
        seams' single subtraction lands both terms.

        Field-form, not mixed-potential: F is the total E-field of a unit
        current element over ground (endpoint charges inherent in the
        element superposition), interpolated from the four SommerfeldGrid
        surfaces, combined per source-tangent vertical/horizontal
        decomposition (eqs 143-147 azimuth factors), projected on the
        observer tangent, and integrated with the basis polynomials by
        per-segment Gauss quadrature. Chunked over observer segments to
        bound the (nodes x nodes) working set.
        """
        gz = self.ground_z
        seg_l = geom["seg_l"]
        seg_r = geom["seg_r"]
        tang = geom["tangents"]
        h = geom["h_per_seg"]
        n_seg = seg_l.shape[0]
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

        d = self.degree
        q = self.n_qp_sommerfeld
        xg, wg = leggauss(q)
        tq = 0.5 * (xg + 1.0)
        nodes = seg_l[:, None, :] + tq[None, :, None] * (seg_r - seg_l)[:, None, :]
        u_phys = h[:, None] * tq[None, :]  # (N, q) physical arc offsets
        w_node = 0.5 * h[:, None] * wg[None, :]
        # Moment weights W[p, i, qi] = w_qi * u_qi^p — the quadrature dual
        # of the J-block u^p moments, so `polys` applies unchanged.
        W = w_node[None] * u_phys[None] ** np.arange(d + 1)[:, None, None]

        # Grid extent: obs-to-image-point distance is convex in the two
        # segment parameters, so its max over all pairs is attained at
        # endpoint pairs.
        r1_max = _sommerfeld.max_image_distance(seg_l, seg_r, gz)
        grid = self._somm_grid(eps_t, r1_max)

        # Fully-fused C++ path: interpolate + project + moment-quadrature +
        # basis-assemble straight into Q, skipping the Jf tensor and the two
        # Galerkin einsums (sommerfeld-perf-plan Phase 4b stage 2). The dense
        # block is the symmetric obs==src case of the rectangular kernel, with
        # the support map == supp_seg (segment set is all segments).
        if _acc is not None and hasattr(_acc, "sommerfeld_remainder_bspline_Q"):
            nodes_c = np.ascontiguousarray(nodes, dtype=np.float64)
            tang_c = np.ascontiguousarray(tang, dtype=np.float64)
            W_c = np.ascontiguousarray(W, dtype=np.float64)
            supp_c = np.ascontiguousarray(supp_seg, dtype=np.int64)
            polys_c = np.ascontiguousarray(polys, dtype=np.float64)
            return _acc.sommerfeld_remainder_bspline_Q(
                nodes_c,
                tang_c,
                W_c,
                nodes_c,
                tang_c,
                W_c,
                supp_c,
                polys_c,
                supp_c,
                polys_c,
                float(gz),
                float(self.k),
                *_sommerfeld.grid_cpp_args(grid),
                int(self._cancel_flag),
            )

        n_nodes = n_seg * q
        src = nodes.reshape(n_nodes, 3)
        t_src = np.repeat(tang, q, axis=0)

        Jf = np.empty((d + 1, d + 1, n_seg, n_seg), dtype=np.complex128)
        chunk = max(1, (1 << 19) // max(n_nodes * q, 1))
        for i0 in range(0, n_seg, chunk):
            self._checkpoint()  # per observer chunk of the eval+einsum block
            i1 = min(i0 + chunk, n_seg)
            obs = nodes[i0:i1].reshape(-1, 3)
            t_obs = np.repeat(tang[i0:i1], q, axis=0)
            proj = _sommerfeld.remainder_field_proj(
                obs, t_obs, src, t_src, gz, self.k, grid
            )
            fq = proj.reshape(i1 - i0, q, n_seg, q)
            Jf[:, :, i0:i1, :] = np.einsum("piq,iqjr,Pjr->pPij", W[:, i0:i1], fq, W)

        n_basis = polys.shape[0]
        Q = np.zeros((n_basis, n_basis), dtype=np.complex128)
        for a in range(d + 1):
            sm = supp_seg[:, a]
            for b in range(d + 1):
                sn = supp_seg[:, b]
                J_blk = Jf[:, :, sm[:, None], sn[None, :]]
                Q += np.einsum("mp,pPmn,nP->mn", polys[:, a, :], J_blk, polys[:, b, :])
        return Q

    def _Q_sommerfeld_remainder_enrich(
        self, geom, supp_seg_poly, polys_poly, spec_seg, spec_origin, eps_t
    ):
        """Sommerfeld remainder-field reaction for the enrichment DOFs (#167
        stage 3): the (Q_pe, Q_ep, Q_ee) counterpart of
        `_Z_sommerfeld_remainder`.

        Same smooth remainder field F (theory eqs 143-147, interpolated from
        the SommerfeldGrid via `_sommerfeld.remainder_field_proj`), projected
        between basis quad nodes and integrated with the basis shapes. F is the
        field of a unit current MOMENT — basis-agnostic — so the enrichment
        side simply weights its nodes by ``w·Φ_sing`` where the polynomial side
        uses the ``u^p`` moment weights. Field-form, so the singular basis's
        endpoint charges are inherent (no separate Φ term).

        The returned blocks are SUBTRACTED alongside the C2 exact-image blocks
        (the sommerfeld branch of `_enrichment_Z_assemble`), matching the
        polynomial `_ground_finite_Z` convention (C2-image + Q, one minus).
        numpy-only and negligible: the enrichment node count is a handful.
        """
        gz = self.ground_z
        seg_l = geom["seg_l"]
        seg_r = geom["seg_r"]
        tang = geom["tangents"]
        h = geom["h_per_seg"]
        d = self.degree
        n_enrich = spec_seg.shape[0]
        n_poly, n_wings = supp_seg_poly.shape
        n_seg = seg_l.shape[0]

        # Same grid extent as _Z_sommerfeld_remainder: endpoint pairs bound the
        # obs-to-image distance, and every interior quad node sits inside it.
        r1_max = _sommerfeld.max_image_distance(seg_l, seg_r, gz)
        grid = self._somm_grid(eps_t, r1_max)

        # Enrichment-side nodes: the singular basis value Φ_sing(u) sampled at
        # the enrichment quadrature (n_qp_sing), times the node weight. Same
        # Φ_sing shape as the free-space / image enrichment assembly.
        gl_xi, gl_w = leggauss(self.n_qp_sing)
        te = 0.5 * (gl_xi + 1.0)
        we = 0.5 * gl_w
        q_e = te.shape[0]
        if self.enrichment_variant == "stable":
            proj_coeffs = _xfem_projection_coeffs(d)
        else:
            proj_coeffs = np.zeros(d + 1)
        polyval = np.polynomial.polynomial.polyval
        eps_tiny = 1e-300

        pos_e = np.zeros((n_enrich, q_e, 3))
        wphi_e = np.zeros((n_enrich, q_e))  # node weight · Φ_sing
        t_e = np.zeros((n_enrich, 3))
        for e in range(n_enrich):
            se = int(spec_seg[e])
            orig = int(spec_origin[e])
            u_norm = te if orig == 0 else (1.0 - te)
            u_safe = np.where(u_norm > eps_tiny, u_norm, eps_tiny)
            sing = u_norm * np.log(u_safe) - polyval(u_norm, proj_coeffs)
            wphi_e[e] = (we * h[se]) * sing
            pos_e[e] = seg_l[se] + te[:, None] * (seg_r[se] - seg_l[se])
            t_e[e] = tang[se]

        obs_e = pos_e.reshape(n_enrich * q_e, 3)
        tobs_e = np.repeat(t_e, q_e, axis=0)

        # Q_ee: enrichment observer vs enrichment source. F is smooth, so the
        # Φ_sing-weighted double sum is a plain contraction (no chunking — the
        # (n_enrich·q_e)² block is tiny).
        proj_ee = _sommerfeld.remainder_field_proj(
            obs_e, tobs_e, obs_e, tobs_e, gz, self.k, grid, self._cancel_flag
        ).reshape(n_enrich, q_e, n_enrich, q_e)
        Q_ee = np.einsum("eq,eqfr,fr->ef", wphi_e, proj_ee, wphi_e)

        Q_pe = np.zeros((n_poly, n_enrich), dtype=np.complex128)
        Q_ep = np.zeros((n_enrich, n_poly), dtype=np.complex128)
        if n_poly == 0:
            return Q_pe, Q_ep, Q_ee

        # Polynomial-side nodes + u^p moment weights, exactly as
        # _Z_sommerfeld_remainder (n_qp_sommerfeld order).
        q = self.n_qp_sommerfeld
        xg, wg = leggauss(q)
        tq = 0.5 * (xg + 1.0)
        nodes_p = seg_l[:, None, :] + tq[None, :, None] * (seg_r - seg_l)[:, None, :]
        u_phys = h[:, None] * tq[None, :]
        w_node = 0.5 * h[:, None] * wg[None, :]
        Wm = w_node[None] * u_phys[None] ** np.arange(d + 1)[:, None, None]
        obs_p = nodes_p.reshape(n_seg * q, 3)
        tobs_p = np.repeat(tang, q, axis=0)

        # Q_pe: poly observer m, enrichment source e. Poly side assembled with
        # the u^p moments + basis polynomials; enrichment side with w·Φ_sing.
        proj_pe = _sommerfeld.remainder_field_proj(
            obs_p, tobs_p, obs_e, tobs_e, gz, self.k, grid, self._cancel_flag
        ).reshape(n_seg, q, n_enrich, q_e)
        field_pe = np.einsum("iqer,er->iqe", proj_pe, wphi_e)  # (n_seg, q, n_enrich)
        Jf_pe = np.einsum("piq,iqe->pie", Wm, field_pe)  # (d+1, n_seg, n_enrich)
        for a in range(n_wings):
            sm = supp_seg_poly[:, a]
            Q_pe += np.einsum("mp,pme->me", polys_poly[:, a, :], Jf_pe[:, sm, :])

        # Q_ep: enrichment observer e, poly source m.
        proj_ep = _sommerfeld.remainder_field_proj(
            obs_e, tobs_e, obs_p, tobs_p, gz, self.k, grid, self._cancel_flag
        ).reshape(n_enrich, q_e, n_seg, q)
        field_ep = np.einsum("eq,eqir->eir", wphi_e, proj_ep)  # (n_enrich, n_seg, q)
        # Jf_ep[e, i, P] = Σ_r field_ep[e,i,r] · Wm[P,i,r]
        Jf_ep = np.einsum("eir,Pir->eiP", field_ep, Wm)  # (n_enrich, n_seg, d+1)
        for b in range(n_wings):
            sn = supp_seg_poly[:, b]
            # Jf_ep[:, sn, :] -> (n_enrich, n_poly, d+1); polys_poly[:, b, :] -> (n_poly, d+1)
            Q_ep += np.einsum("enP,nP->en", Jf_ep[:, sn, :], polys_poly[:, b, :])

        return Q_pe, Q_ep, Q_ee

    def _seg_radius(self, geom):
        """(n_segs_total,) per-segment radius — each segment inherits its
        wire's (stevenmburns/momwire#147). The kernel helpers collapse a
        uniform array back to the scalar fast path, so passing this
        everywhere keeps scalar-radius solves bit-identical."""
        seg_off = np.asarray(geom["seg_offsets"], dtype=np.int64)
        return np.repeat(self._radius_per_wire, np.diff(seg_off))

    def _ek_axis_labels(self, geom, mirror):
        """Cached coaxial-and-equal-radius labels for this geometry.

        Returns `(group_i, group_j)`, both (n_segs,), for the observer and
        source sides of a fill. `mirror=False` is the free-space case, where
        both sides are the same real segments and therefore the same labels.

        `mirror=True` is the PEC-image block, where the SOURCE segments are
        the real ones reflected through z = ground_z. The two label arrays
        must stay COMPARABLE — eligibility is `group_i[i] == group_j[j]` —
        so the labels are built by one scan over the CONCATENATION of the
        real and mirrored segments and then split, not by two independent
        scans. Two independent scans would be actively wrong: a horizontal
        wire and its image would both be labelled 0 and every real/image
        pair would be declared coaxial, when in fact they are parallel and
        offset by twice the height. With the joint scan, a vertical monopole
        on the plane mirrors onto its own axis and IS one group (NEC's
        IND = 0 ground-contact branch, #249 §4.3), while the horizontal wire
        splits into two.

        Cached per geometry OBJECT (identity check) and per mirror flag, so
        a grounded swept solve pays for the O(N·G) scan once, not per k.
        """
        cached = self._cached_ek_groups
        if cached is None or cached[0] is not geom:
            cached = (geom, {})
            self._cached_ek_groups = cached
        hit = cached[1].get(mirror)
        if hit is not None:
            return hit

        seg_l = geom["seg_l"]
        seg_r = geom["seg_r"]
        tangents = geom["tangents"]
        seg_a = self._seg_radius(geom)
        if mirror:
            n = seg_l.shape[0]
            flip = np.array([1.0, 1.0, -1.0])
            joint = _ek_axis_groups(
                np.vstack([seg_l, self._image_positions(seg_l)]),
                np.vstack([seg_r, self._image_positions(seg_r)]),
                np.vstack([tangents, tangents * flip]),
                np.concatenate([seg_a, seg_a]),
            )
            hit = (joint[:n], joint[n:])
        else:
            labels = _ek_axis_groups(seg_l, seg_r, tangents, seg_a)
            hit = (labels, labels)
        cached[1][mirror] = hit
        return hit

    def _ek_spec(self, geom, mirror=False):
        """The `_EK` spec for a whole-mesh fill, or None when EK is off.

        Call sites guard on `self.extended_kernel` before calling, so an
        EK-off solve enters no EK code at all (the monkeypatch-counter gate
        in tests/test_extended_kernel_bspline.py pins that); the `not`
        branch here is belt-and-braces for a direct caller.

        Windowed and blocked fills restrict the returned spec's labels to
        their own rows/columns with `_ek_slice`; same-edge blocks use
        `_EK_SAME_EDGE` instead (whole-block eligibility).
        """
        if not self.extended_kernel:
            return None
        group_i, group_j = self._ek_axis_labels(geom, mirror)
        # a=None: each kernel call's own regularisation radius IS the EK
        # radius, because eligibility requires equal radii and the off-edge
        # kernel already regularises each observer row with its own wire's.
        return _EK(a=None, group_i=group_i, group_j=group_j)

    def _same_edge_prep(self, geom):
        """k-independent per-same-edge precompute hoisted out of the swept-k
        loop: each edge's analytic static-moment block plus the reg-kernel
        quadrature geometry (R table + weighted powers). Returns a list of
        `(global_slice, A_static, reg_geometry)`. Bounded memory — one edge's
        tables at a time, identical footprint to the per-k path; only the
        cheap `exp(-jkR)` + einsum is left per k.

        Same-edge pairs live on a single wire, so each edge's block uses
        that wire's own radius — and, under `extended_kernel`, is eligible
        in its entirety (`_EK_SAME_EDGE`). The spec rides into the reg
        GEOMETRY dict rather than the per-k call, so the swept callers that
        share this prep across frequencies carry EK for free.
        """
        d = self.degree
        ek = _EK_SAME_EDGE if self.extended_kernel else None
        per_wire = geom["per_wire"]
        seg_off = geom["seg_offsets"]
        prep = []
        for w in range(len(per_wire)):
            pw = per_wire[w]
            ed_off = pw["edge_offsets"]
            ed_arc = pw["edge_arc_edges"]
            base = seg_off[w]
            a_w = float(self._radius_per_wire[w])
            for i_e in range(len(ed_off) - 1):
                sl = slice(base + ed_off[i_e], base + ed_off[i_e + 1])
                A_st = _seg_seg_static_moments(ed_arc[i_e], a_w, max_d=d, ek=ek)
                reg_geo = _seg_seg_reg_geometry(
                    ed_arc[i_e], a_w, max_d=d, n_qp=self.n_qp_pair, ek=ek
                )
                prep.append((sl, A_st, reg_geo))
        return prep

    def _same_edge_prep_swept_chunks(self, prep, k_array):
        """Yield `(ki, k, same_edge_k)` per sweep point, where `same_edge_k`
        is the per-k `same_edge_prep` list the fallback sweeps hand to
        `compute_port_solution` / `compute_impedance`.

        The swept same-edge reg-moment hoist is chunked over k so the
        hoisted blocks stay under the `swept_mem_mb` budget (issue #263):
        the whole-sweep hoist is an O(n_k · nm² · ΣN_e²) transient that the
        budget never saw — the same corner as #238's per-k tensor, one
        multiplicative n_k worse. The chunk arithmetic mirrors the
        same-edge term of `_swept_batched_z_chunks` (nm² ΣN_e² complex per
        k). Chunking is pure re-batching: each k's moment block is computed
        independently of its chunk mates, so a sweep that fits in one chunk
        is byte-for-byte the old whole-sweep call, and a multi-chunk sweep
        matches it to the last ulp.
        """
        k_array = np.asarray(k_array, dtype=float)
        n_k = k_array.shape[0]
        nm = self.degree + 1
        sum_ne2 = sum((sl.stop - sl.start) ** 2 for sl, _A_st, _g in prep)
        bytes_per_k = nm * nm * sum_ne2 * 16
        chunk = max(1, min(n_k, (self.swept_mem_mb << 20) // max(bytes_per_k, 1)))
        for c0 in range(0, n_k, chunk):
            self._checkpoint()  # before each chunk's batched reg-moment build
            ks = k_array[c0 : c0 + chunk]
            reg_chunk = [
                _seg_seg_reg_moments_from_geometry_swept(reg_geo, ks)
                for _sl, _A_st, reg_geo in prep
            ]
            for i in range(ks.shape[0]):
                same_edge_k = [
                    (sl, A_st, reg_chunk[e][i]) for e, (sl, A_st, _g) in enumerate(prep)
                ]
                yield c0 + i, ks[i], same_edge_k

    def _build_J_blocks(self, geom, k, same_edge_prep=None):
        """All polynomial moment integrals J_pq[i, j] for p, q ∈ {0..d} and
        every (i, j) global segment pair. Returns shape (d+1, d+1, N, N).

        Fused build: first compute
        every pair by full GL quadrature on the regularized full kernel
        G = exp(-jkR)/(4πR), R² = |Δr|² + a²; then overwrite same-edge
        blocks with the analytic static + GL-regularized split (essential
        for the log-singular diagonal).

        `same_edge_prep` (from `_same_edge_prep`) lets a swept-k caller share
        the k-independent static + reg-geometry across frequencies; when None
        the same quantities are computed inline (single-k path). (Fully
        batched sweeps bypass this method entirely —
        `_compute_impedance_swept_batched` builds its own chunked
        (n_k, d+1, d+1, N, N) tensors.)
        """
        d = self.degree
        a_row = self._seg_radius(geom)
        seg_l = geom["seg_l"]
        seg_r = geom["seg_r"]
        ek = self._ek_spec(geom) if self.extended_kernel else None
        ek_se = _EK_SAME_EDGE if self.extended_kernel else None

        # All-pairs full kernel (same a² regularization handles touching
        # segments at kink corners and at junctions to within ~1e-5 at
        # antenna scales; off-segment-pair accuracy is what GL is good at).
        # Per-observer-row radius under mixed per-wire radii.
        J = _seg_seg_full_moments_offedge(
            seg_l, seg_r, seg_l, seg_r, a_row, k, d, self.n_qp_pair, ek=ek
        )  # (d+1, d+1, N, N) complex

        # Overwrite each same-edge block with analytic static + reg
        if same_edge_prep is None:
            per_wire = geom["per_wire"]
            seg_off = geom["seg_offsets"]
            for w in range(len(per_wire)):
                pw = per_wire[w]
                ed_off = pw["edge_offsets"]
                ed_arc = pw["edge_arc_edges"]
                base = seg_off[w]
                a_w = float(self._radius_per_wire[w])
                for i_e in range(len(ed_off) - 1):
                    sl = slice(base + ed_off[i_e], base + ed_off[i_e + 1])
                    A_st = _seg_seg_static_moments(ed_arc[i_e], a_w, max_d=d, ek=ek_se)
                    A_reg = _seg_seg_reg_moments(
                        ed_arc[i_e], a_w, k, max_d=d, n_qp=self.n_qp_pair, ek=ek_se
                    )
                    J[:, :, sl, sl] = A_st + A_reg
        else:
            for sl, A_st, reg in same_edge_prep:
                # `reg` is either a reg-geometry dict (compute this k's block
                # now) or a precomputed (max_d+1, max_d+1, N, N) moment block
                # for this k (swept caller batched it across frequencies).
                A_reg = (
                    _seg_seg_reg_moments_from_geometry(reg, k)
                    if isinstance(reg, dict)
                    else reg
                )
                J[:, :, sl, sl] = A_st + A_reg

        return J

    # ------------------------------------------------------------------
    # Z assembly
    # ------------------------------------------------------------------

    def _assemble_Z(self, J, supp_seg, polys, geom, td_all=None):
        """Assemble the (n_basis, n_basis) complex Z matrix.

        Uses the templated C++ accelerator `assemble_Z_bspline` when
        available and `self.degree` is in its instantiation set; otherwise
        falls back to a numpy-einsum implementation that's a bit-exact
        reference target.

        `td_all` defaults to the free-space tangent dot product matrix
        derived from `geom["tangents"]`. The PEC image build passes its
        own (tx, ty, -tz)-modified table here so the same assembly fuses
        the image-current sign flip.
        """
        d = self.degree
        n_basis, n_wings, n_poly = polys.shape
        assert n_wings == d + 1 and n_poly == d + 1

        if td_all is None:
            tangents = geom["tangents"]
            td_all = tangents @ tangents.T

        if _HAVE_BSPLINE_ASSEMBLE_ACCEL and d <= _BSPLINE_ASSEMBLE_ACCEL_MAX_D:
            return _acc.assemble_Z_bspline(
                np.ascontiguousarray(J, dtype=np.complex128),
                np.ascontiguousarray(supp_seg, dtype=np.int64),
                np.ascontiguousarray(polys, dtype=np.float64),
                np.ascontiguousarray(td_all, dtype=np.float64),
                float(self.omega),
                float(self.eps),
                float(self.mu),
                int(d),
                self._cancel_flag,
            )

        Z_A = np.zeros((n_basis, n_basis), dtype=np.complex128)
        Z_Phi = np.zeros((n_basis, n_basis), dtype=np.complex128)
        p_vec = np.arange(1, d + 1, dtype=np.float64) if d >= 1 else None

        for a in range(n_wings):
            sm = supp_seg[:, a]
            for b in range(n_wings):
                sn = supp_seg[:, b]
                J_blk = J[:, :, sm[:, None], sn[None, :]]
                td_blk = td_all[sm[:, None], sn[None, :]]

                inner_A = np.einsum(
                    "mp,pPmn,nP->mn", polys[:, a, :], J_blk, polys[:, b, :]
                )
                Z_A += td_blk * inner_A

                if d >= 1:
                    deriv_m = polys[:, a, 1:] * p_vec[None, :]
                    deriv_n = polys[:, b, 1:] * p_vec[None, :]
                    J_blk_lo = J_blk[:d, :d]
                    inner_Phi = np.einsum("mp,pPmn,nP->mn", deriv_m, J_blk_lo, deriv_n)
                    Z_Phi += inner_Phi

        Z_A = 1j * self.omega * self.mu * Z_A
        Z_Phi = Z_Phi / (1j * self.omega * self.eps)
        return Z_A + Z_Phi

    def _compute_Z_dense_chunked(self, geom, k, supp_seg, polys, same_edge_prep=None):
        """Free-space dense Z without materialising the (d+1, d+1, N, N)
        moment tensor (issue #136).

        Observer-row chunks of the all-pairs full-kernel moments accumulate
        straight into Z through the windowed C++ assembler — the
        (zA, zPhi) → Z mixing is linear, so per-window accumulation equals
        the all-at-once assembly. Same-edge blocks are then fixed up
        per edge with a correction window (analytic static + regularised
        split MINUS the full-kernel block the sweep already added), the
        chunked equivalent of `_build_J_blocks`'s overwrite. Identical
        quadrature and algebra to `_build_J_blocks` + `_assemble_Z`; the
        peak transient is one row chunk (bounded by `swept_mem_mb`, the
        same fill-transient budget the batched sweep uses) instead of the
        full tensor — the difference between ~3 GB and ~0.25 GB on a
        4,700-segment mesh.

        No N²-scale transient survives the fill: the pair tangent dot is
        formed inside the assembler from the (N, 3) tangent table, rather
        than from an N×N dot matrix that would sit alongside Z for the
        whole build at half its size (issue #318).
        """
        d = self.degree
        a_row = self._seg_radius(geom)
        seg_l = geom["seg_l"]
        seg_r = geom["seg_r"]
        n_segs = geom["n_segs_total"]
        n_basis = supp_seg.shape[0]
        tangents = geom["tangents"]
        ek = self._ek_spec(geom) if self.extended_kernel else None
        ek_se = _EK_SAME_EDGE if self.extended_kernel else None

        # Fortran order: scipy.linalg.solve(overwrite_a=True) can only
        # factor in place on a column-major matrix — C order would silently
        # cost a full n_basis-squared copy at solve time (issue #136).
        Z = np.zeros((n_basis, n_basis), dtype=np.complex128, order="F")
        supp_c = np.ascontiguousarray(supp_seg, dtype=np.int64)
        polys_c = np.ascontiguousarray(polys, dtype=np.float64)
        tan_c = np.ascontiguousarray(tangents, dtype=np.float64)
        all_n = np.arange(n_basis, dtype=np.int64)

        def _accumulate(J_win, i0, i1, j0, j1, m_idx, n_idx):
            # Producer contract, not a conversion: every window handed here
            # is already C-contiguous complex128, so the ascontiguousarray
            # this used to wrap it in returned the SAME object on every
            # path (issue #318 audit — C++ reduced kernel, C++ EK twin,
            # numpy einsum fallback, the mixed-radius `concatenate`, and
            # the same-edge `(A_st + A_reg) - J_edge` difference, which
            # promotes float64 A_st to complex128 by itself). The assert
            # pins the contract and vanishes under -O; the pybind
            # `c_style | forcecast` on the assembler stays the safety net.
            assert J_win.dtype == np.complex128 and J_win.flags.c_contiguous, (
                f"moment window must be C-contiguous complex128, got {J_win.dtype}"
            )
            _acc.assemble_Z_bspline_windowed(
                J_win,
                supp_c,
                polys_c,
                tan_c,
                m_idx,
                n_idx,
                int(i0),
                int(i1),
                int(j0),
                int(j1),
                float(self.omega),
                float(self.eps),
                float(self.mu),
                Z,
                self._cancel_flag,
            )

        def _bases_touching(lo, hi):
            mask = ((supp_c >= lo) & (supp_c < hi)).any(axis=1)
            return np.nonzero(mask)[0].astype(np.int64)

        # Row-chunk budget: bytes per observer row of the (d+1, d+1, ·, N)
        # chunk, against the same transient budget the swept path uses.
        row_bytes = (d + 1) ** 2 * n_segs * 16
        chunk = max(1, int(self.swept_mem_mb * 1024 * 1024 // row_bytes))
        for i0 in range(0, n_segs, chunk):
            self._checkpoint()  # per observer chunk of the fill+assemble
            i1 = min(i0 + chunk, n_segs)
            J_chunk = _seg_seg_full_moments_offedge(
                seg_l[i0:i1],
                seg_r[i0:i1],
                seg_l,
                seg_r,
                a_row[i0:i1],
                k,
                d,
                self.n_qp_pair,
                ek=_ek_slice(ek, rows=slice(i0, i1)),
            )
            _accumulate(J_chunk, i0, i1, 0, n_segs, _bases_touching(i0, i1), all_n)
        del J_chunk

        # Same-edge fixup: the sweep above added the full-kernel block for
        # every pair; each same-edge block must instead be the analytic
        # static + regularised split, so accumulate the difference.
        if same_edge_prep is None:
            per_wire = geom["per_wire"]
            seg_off = geom["seg_offsets"]
            same_edge_prep = []
            for w in range(len(per_wire)):
                pw = per_wire[w]
                ed_off = pw["edge_offsets"]
                ed_arc = pw["edge_arc_edges"]
                base = seg_off[w]
                a_w = float(self._radius_per_wire[w])
                for i_e in range(len(ed_off) - 1):
                    sl = slice(base + ed_off[i_e], base + ed_off[i_e + 1])
                    A_st = _seg_seg_static_moments(ed_arc[i_e], a_w, max_d=d, ek=ek_se)
                    reg_geo = _seg_seg_reg_geometry(
                        ed_arc[i_e], a_w, max_d=d, n_qp=self.n_qp_pair, ek=ek_se
                    )
                    same_edge_prep.append((sl, A_st, reg_geo))
        for sl, A_st, reg in same_edge_prep:
            self._checkpoint()  # per same-edge correction block
            A_reg = (
                _seg_seg_reg_moments_from_geometry(reg, k)
                if isinstance(reg, dict)
                else reg
            )
            # The correction subtracts what the sweep above already added for
            # these pairs, so it must be filled with exactly the sweep's own
            # EK treatment — the sliced whole-mesh spec, not `ek_se`. (They
            # agree pair by pair on a same-edge block; slicing keeps the two
            # windows the same arithmetic rather than merely the same value.)
            J_edge = _seg_seg_full_moments_offedge(
                seg_l[sl],
                seg_r[sl],
                seg_l[sl],
                seg_r[sl],
                a_row[sl],
                k,
                d,
                self.n_qp_pair,
                ek=_ek_slice(ek, rows=sl, cols=sl),
            )
            corr = (A_st + A_reg) - J_edge
            e_idx = _bases_touching(sl.start, sl.stop)
            _accumulate(corr, sl.start, sl.stop, sl.start, sl.stop, e_idx, e_idx)

        return Z

    def _accumulate_Z_image_chunked(self, Z, geom, k, supp_seg, polys, weights_fn):
        """Chunked ground-image accumulation: subtract the weighted image
        sub-assembly from Z without materialising the (d+1, d+1, N, N)
        image tensor OR an intermediate n_basis² matrix (issue #136,
        ground scope). Observer-row chunks of the mirrored-source
        full-kernel moments feed the weighted windowed assembler with
        scale = -1 (the seams' `Z - image` convention). Image pairs are
        never singular (`_build_J_image_blocks` docstring), so there is
        no same-edge correction pass. The complex weights serve all three
        grounds: PEC (mirror tangent dot / ones), refl-coef (Fresnel
        dyad / image charge), Sommerfeld exact image (constant C2).

        Weights arrive as `weights_fn(i0, i1) -> (w_A, w_Phi)`, called once
        per observer chunk for WINDOWS of shape (i1-i0, n_segs) aligned with
        the moment window's trailing axes — not as global (N, N) tables
        (issue #323). Producing them per chunk is what retires the 2× dense-Z
        residency this path used to carry: nothing N² in the weights is ever
        allocated. See `_image_weight_window_fn` for the per-mode producers."""
        d = self.degree
        a_row = self._seg_radius(geom)
        seg_l = geom["seg_l"]
        seg_r = geom["seg_r"]
        seg_l_img = self._image_positions(seg_l)
        seg_r_img = self._image_positions(seg_r)
        n_segs = geom["n_segs_total"]
        n_basis = supp_seg.shape[0]
        ek = self._ek_spec(geom, mirror=True) if self.extended_kernel else None

        supp_c = np.ascontiguousarray(supp_seg, dtype=np.int64)
        polys_c = np.ascontiguousarray(polys, dtype=np.float64)
        all_n = np.arange(n_basis, dtype=np.int64)

        row_bytes = (d + 1) ** 2 * n_segs * 16
        chunk = max(1, int(self.swept_mem_mb * 1024 * 1024 // row_bytes))
        for i0 in range(0, n_segs, chunk):
            self._checkpoint()  # per observer chunk of the image fill
            i1 = min(i0 + chunk, n_segs)
            J_chunk = _seg_seg_full_moments_offedge(
                seg_l[i0:i1],
                seg_r[i0:i1],
                seg_l_img,
                seg_r_img,
                a_row[i0:i1],
                k,
                d,
                self.n_qp_pair,
                ek=_ek_slice(ek, rows=slice(i0, i1)),
            )
            m_mask = ((supp_c >= i0) & (supp_c < i1)).any(axis=1)
            # Same producer contract as `_accumulate` in the free-space
            # chunked fill (issue #318 audit): the offedge producer emits
            # C-contiguous complex128 on every path, so the wrapper this
            # replaced was the same dead no-op. `forcecast` on the
            # assembler stays the safety net.
            assert J_chunk.dtype == np.complex128 and J_chunk.flags.c_contiguous, (
                f"moment window must be C-contiguous complex128, got {J_chunk.dtype}"
            )
            # The window producers are gemm/elementwise expressions, so they
            # emit C-contiguous complex128 of exactly the chunk's shape
            # already — same producer contract as the moment window above,
            # asserted rather than re-wrapped.
            w_A_win, w_Phi_win = weights_fn(i0, i1)
            assert all(
                w.shape == (i1 - i0, n_segs)
                and w.dtype == np.complex128
                and w.flags.c_contiguous
                for w in (w_A_win, w_Phi_win)
            ), f"weight windows must be C-contiguous complex128 ({i1 - i0}, {n_segs})"
            _acc.assemble_Z_bspline_weighted_windowed(
                J_chunk,
                supp_c,
                polys_c,
                # The j-window is the full [0, n_segs), so the producers hand
                # back whole rows.
                w_A_win,
                w_Phi_win,
                np.nonzero(m_mask)[0].astype(np.int64),
                all_n,
                int(i0),
                int(i1),
                0,
                int(n_segs),
                float(self.omega),
                float(self.eps),
                float(self.mu),
                complex(-1.0),
                Z,
                self._cancel_flag,
            )

    # ------------------------------------------------------------------
    # Distributed series wire loading (stevenmburns/momwire#131)
    # ------------------------------------------------------------------

    def _loading_gram(self):
        """COO triplets of the loading Gram matrix, tagged per wire.

        S[m, n] = ∫ Φ_m(l)·Φ_n(l) dl over the segments the two bases share
        — nonzero only for overlapping bases on the same wire, so the
        structure is banded and block-diagonal by wire. The full loading
        term is Σ_w Z'_w(ω)·S_w; the triplets carry `wire_of_nnz` so one
        structure serves every ω (only the per-wire scale changes).

        Involves no kernel (no exp(-jkR)) — on each shared segment the
        integral is the closed-form polynomial moment
        Σ_pq C[m,a,p]·C[n,b,q]·h^(p+q+1)/(p+q+1). k-independent; cached
        per instance (geometry is immutable after __init__).

        Returns (rows, cols, vals, wire_of_nnz) int64/float64 arrays.
        Duplicate (row, col) entries are intentional (one per shared
        segment) — consumers accumulate (np.add.at / COO semantics).
        """
        cached = self._cached_loading_gram
        if cached is not None:
            return cached
        geom = self._build_geometry()
        supp_seg, polys, _kcl_A, _wk, _wbg = self._build_basis_polynomials(geom)
        n_basis, n_wings, n_poly = polys.shape
        h_per_seg = geom["h_per_seg"]
        seg_offsets = np.asarray(geom["seg_offsets"], dtype=np.int64)

        # segment → [(basis, wing), ...]. Padded wings (beyond a boundary
        # basis's actual support) have all-zero poly rows; skip them so a
        # padding segment index of 0 can't alias real segment 0.
        seg_map: dict[int, list[tuple[int, int]]] = {}
        nonzero_wing = np.any(polys != 0.0, axis=2)
        for m in range(n_basis):
            for a in range(n_wings):
                if nonzero_wing[m, a]:
                    seg_map.setdefault(int(supp_seg[m, a]), []).append((m, a))

        pq = np.arange(n_poly)
        pq_sum = pq[:, None] + pq[None, :] + 1
        rows, cols, vals, wire_ids = [], [], [], []
        for s, entries in seg_map.items():
            hs = h_per_seg[s]
            # H[p, q] = ∫₀^h u^p·u^q du = h^(p+q+1)/(p+q+1)
            H = hs**pq_sum / pq_sum
            C = polys[[m for m, _ in entries], [a for _, a in entries], :]
            M = C @ H @ C.T
            w = int(np.searchsorted(seg_offsets, s, side="right") - 1)
            n_e = len(entries)
            for i in range(n_e):
                mi = entries[i][0]
                for j in range(n_e):
                    rows.append(mi)
                    cols.append(entries[j][0])
                    vals.append(M[i, j])
                    wire_ids.append(w)

        result = (
            np.asarray(rows, dtype=np.int64),
            np.asarray(cols, dtype=np.int64),
            np.asarray(vals, dtype=np.float64),
            np.asarray(wire_ids, dtype=np.int64),
        )
        self._cached_loading_gram = result
        return result

    def _loading_zw(self, omega):
        """Per-wire series impedance Z'_w(ω) [Ω/m]; (n_w,) or (n_w, n_k)."""
        return _wire_loading.series_impedance_per_wire(
            omega,
            self._radius_per_wire,
            self.wire_conductivity,
            self.insulation_radius,
            self.insulation_eps_r,
        )

    def _apply_loading(self, Z, omega=None):
        """Add the loading term into Z in place; no-op when loading is off.

        Z is (n_basis, n_basis) with scalar `omega` (default self.omega),
        or a swept chunk (n_k, n_basis, n_basis) with `omega` (n_k,).
        Returns Z for call-site convenience.
        """
        if not self._loading_active:
            return Z
        rows, cols, vals, wire_ids = self._loading_gram()
        if omega is None:
            omega = self.omega
        zw = self._loading_zw(omega)  # (n_w,) or (n_w, n_k)
        if Z.ndim == 2:
            np.add.at(Z, (rows, cols), zw[wire_ids] * vals)
        else:
            data = zw[wire_ids, :].T * vals[None, :]  # (n_k, nnz)
            for ki in range(Z.shape[0]):
                np.add.at(Z[ki], (rows, cols), data[ki])
        return Z

    def _loading_block(self, I, J, omega=None):
        """Dense loading sub-block L[I][:, J] for restricted evaluators
        (HMatrixSolver.zblock). The scaled CSR is cached per ω — the
        H-matrix fill calls zblock many times per k."""
        rows, cols, vals, wire_ids = self._loading_gram()
        if omega is None:
            omega = self.omega
        cache = getattr(self, "_loading_csr_cache", None)
        if cache is None or cache[0] != omega:
            geom = self._build_geometry()
            supp_seg, _p, _k, _wk, _wbg = self._build_basis_polynomials(geom)
            n = supp_seg.shape[0]
            zw = self._loading_zw(omega)
            L = scipy.sparse.coo_matrix(
                (zw[wire_ids] * vals, (rows, cols)), shape=(n, n)
            ).tocsr()
            self._loading_csr_cache = (omega, L)
        L = self._loading_csr_cache[1]
        return L[I][:, J].toarray()

    def wire_loss_power(self, coeffs, omega=None):
        """Ohmic power dissipated in the wire metal, from a solve's coeffs.

        P_wire = ½ Σ_w Re[Z'_w(ω)] · (c^H S_w c) — the ∫ R'(l)·|I(l)|² dl
        readout the downstream power budget reports. Insulation loading is
        purely reactive and contributes nothing here. Trailing KCL
        Lagrange-multiplier entries in `coeffs` are ignored.

        Returns (total_watts, per_wire_watts ndarray (n_wires,)).
        """
        n_w = len(self.wires_polylines)
        per_wire = np.zeros(n_w, dtype=np.float64)
        if not self._loading_active:
            return 0.0, per_wire
        rows, cols, vals, wire_ids = self._loading_gram()
        geom = self._build_geometry()
        supp_seg, _p, _k, _wk, _wbg = self._build_basis_polynomials(geom)
        c = np.asarray(coeffs)[: supp_seg.shape[0]]
        r_w = np.real(self._loading_zw(self.omega if omega is None else omega))
        contrib = 0.5 * r_w[wire_ids] * np.real(np.conj(c[rows]) * c[cols]) * vals
        np.add.at(per_wire, wire_ids, contrib)
        return float(per_wire.sum()), per_wire

    # ------------------------------------------------------------------
    # Source vector
    # ------------------------------------------------------------------

    def _build_source_vector(
        self,
        geom,
        wire_knots,
        wire_basis_global,
        n_basis_total,
        wi=None,
        s_f=None,
    ):
        """Galerkin RHS for a delta-gap, segment-gap or smoothed source.

        Delta-gap (`feed_model="point"`, no smoothing — the default):
        v_m = Φ_m(s_f).

        Segment gap (`feed_model="segment"`, stevenmburns/momwire#216): NEC's
        Eq 187 convention, E_app = V/Δ uniform over the mesh cell [s_lo, s_hi]
        containing s_f, so v_m = (1/Δ)∫_{s_lo}^{s_hi} Φ_m ds with Δ = s_hi −
        s_lo. Every Φ_m is a polynomial of degree d on that cell (its
        endpoints ARE knots), so the existing `n_qp_source`-node Gauss rule —
        exact through degree 2·n_qp_source − 1, i.e. 31 at the default 16 —
        integrates it exactly; there is no accuracy knob to add here. This is
        the same source `SinusoidalSolver` hard-codes and
        `SinusoidalGalerkinSolver` defaults to, so it is the feed-matched
        setting for a comparison against either (report §19). It shares the
        smoothed source's cure for the delta gap's O(1/N) term below: the
        drive is a bounded function, not a distribution.

        Smoothed source: replace V·δ(s − s_f) with V·g_w(s − s_f) where g_w
        is a cos² bump of integral 1 and half-width w/2 = α·h_feed/2 (with
        α = self.feed_smoothing_factor). Then v_m = ⟨Φ_m, g_w(. − s_f)⟩,
        computed by Gauss-Legendre quadrature on the bump's support. The
        impedance extraction in `compute_impedance` is unchanged:
        I_in = v^T c gives the smoothing-weighted current, and Z = 1/I_in.
        In the α → 0 limit g_w → δ and both v and I_in revert to the
        delta-gap formulas.

        Why this fixes the convergence rate. The delta-gap source produces a
        log singularity in the current at s_f that no polynomial basis can
        represent; the integrated impedance picks up an O(1/N) error term
        regardless of basis degree. The smoothed source has no singularity,
        so the convergence is basis-limited (O(1/N³) for d=2).

        For unit excitation V=1 at a single (wi, s_f); the multi-feed
        caller scales each per-feed vector by V_i and sums them. wi/s_f
        default to self.feeds[0] for back-compat with single-feed callers.
        """
        d = self.degree
        if wi is None:
            wi = self.feeds[0][0]
        arc = geom["per_wire"][wi]["arc_at_knot"]
        wire_arc = arc[-1]
        if s_f is None:
            arc_req = self.feeds[0][1]
            s_f = arc_req if arc_req is not None else wire_arc / 2.0
        knots = wire_knots[wi]
        kept, local_to_global = wire_basis_global[wi]

        if self.feed_smoothing_factor is None and self.feed_model == "point":
            # Delta-gap (original)
            DM = BSpline.design_matrix(np.array([s_f]), knots, d).toarray()[0]
            v = np.zeros(n_basis_total, dtype=np.complex128)
            for kept_idx, (j, _kind, _junc_idx, _end_pos) in enumerate(kept):
                m_global = local_to_global[kept_idx]
                v[m_global] = DM[j]
            return v

        if self.feed_model == "segment":
            # NEC's segment-wide gap: E_app = V/Δ uniform over the mesh cell
            # holding s_f. Same cell location as the smoothing branch below
            # (a feed exactly on a knot takes the cell to its right, and a
            # feed at the wire end is clipped to the last cell).
            arc_at_knot = arc
            seg_idx = int(np.searchsorted(arc_at_knot, s_f, side="right")) - 1
            seg_idx = max(0, min(seg_idx, len(arc_at_knot) - 2))
            s_lo = float(arc_at_knot[seg_idx])
            s_hi = float(arc_at_knot[seg_idx + 1])
            h_cell = s_hi - s_lo
            gl_xi, gl_w = leggauss(self.n_qp_source)
            t = 0.5 * (s_hi + s_lo) + 0.5 * h_cell * gl_xi
            # (1/h)·∫_cell Φ_m ds — exact, the integrand being degree d on
            # exactly one knot span (see the docstring).
            weights = (0.5 * h_cell * gl_w) / h_cell
            DM = BSpline.design_matrix(t, knots, d).toarray()
            v_full = DM.T @ weights
            v = np.zeros(n_basis_total, dtype=np.complex128)
            for kept_idx, (j, _kind, _junc_idx, _end_pos) in enumerate(kept):
                m_global = local_to_global[kept_idx]
                v[m_global] = v_full[j]
            return v

        # Smoothed source: find the feed segment to set the smoothing width
        # w = α·h_feed. The "feed segment" is the segment containing s_f.
        h_per_seg = geom["per_wire"][wi]["h_per_seg"]
        arc_at_knot = arc
        # Locate segment such that arc_at_knot[seg] <= s_f < arc_at_knot[seg+1]
        seg_idx = int(np.searchsorted(arc_at_knot, s_f, side="right")) - 1
        seg_idx = max(0, min(seg_idx, len(h_per_seg) - 1))
        h_feed = float(h_per_seg[seg_idx])
        alpha = float(self.feed_smoothing_factor)
        smoothing_w = alpha * h_feed
        half_w = smoothing_w / 2.0

        # Clip to wire arc range — if the feed is too close to a wire end,
        # the bump may not fit; in that case the integral is just over the
        # available portion (consistent with the smoothed source convention
        # but breaks symmetry at the wire end).
        s_lo = max(0.0, s_f - half_w)
        s_hi = min(wire_arc, s_f + half_w)
        if s_lo >= s_hi:
            raise ValueError(
                "feed_smoothing_factor too large for wire — bump doesn't fit"
            )

        gl_xi, gl_w = leggauss(self.n_qp_source)
        t = 0.5 * (s_hi + s_lo) + 0.5 * (s_hi - s_lo) * gl_xi
        weights = 0.5 * (s_hi - s_lo) * gl_w

        # Cos² bump on |x| < smoothing_w/2:
        #   g_w(x) = (2/smoothing_w) · cos²(π x / smoothing_w)
        # so ∫ g_w = 1 (since ∫_{-w/2}^{w/2} cos²(πx/w) dx = w/2).
        delta = t - s_f
        in_support = np.abs(delta) < half_w
        g_vals = np.where(
            in_support,
            (2.0 / smoothing_w) * np.cos(np.pi * delta / smoothing_w) ** 2,
            0.0,
        )

        # Evaluate every basis at the quadrature points and integrate.
        DM = BSpline.design_matrix(t, knots, d).toarray()  # (n_qp, n_basis_w_full)
        v_full = np.einsum("qj,q,q->j", DM, g_vals, weights)  # (n_basis_w_full,)

        v = np.zeros(n_basis_total, dtype=np.complex128)
        for kept_idx, (j, _kind, _junc_idx, _end_pos) in enumerate(kept):
            m_global = local_to_global[kept_idx]
            v[m_global] = v_full[j]
        return v

    # ------------------------------------------------------------------
    # KCL solve (Schur complement)
    # ------------------------------------------------------------------

    def _solve_with_kcl(self, Z, v, kcl_A, overwrite=False):
        """Constrained solve [Z A^T; A 0] [I; λ] = [v; 0] via Schur.

        If kcl_A is empty (no junctions), do a plain solve.

        `overwrite=True` lets LAPACK factor Z in place instead of copying
        it — a full n_basis²-complex saving (2.5 GB at whip-benchmark
        scale). Callers pass it only where Z is dead after the solve. The
        locally-built rhs is always overwritten; the caller's `v` never is.
        """
        if kcl_A.shape[0] == 0:
            return scipy.linalg.solve(Z, v, overwrite_a=overwrite)
        n_b = Z.shape[0]
        n_c = kcl_A.shape[0]
        rhs = np.empty((n_b, 1 + n_c), dtype=np.complex128, order="F")
        rhs[:, 0] = v
        rhs[:, 1:] = kcl_A.T
        sol = scipy.linalg.solve(Z, rhs, overwrite_a=overwrite, overwrite_b=True)
        w = sol[:, 0]
        X = sol[:, 1:]
        lam = scipy.linalg.solve(kcl_A @ X, kcl_A @ w)
        return w - X @ lam

    def _solve_with_kcl_ports(self, Z, V, kcl_A, overwrite=False):
        """Multi-port KCL-constrained Schur solve. V: (n_b, n_p), returns
        (n_b, n_p). Matrix-RHS generalisation of `_solve_with_kcl` — all
        n_p source columns share one LU factorisation with the n_c
        constraint columns. `overwrite` as in `_solve_with_kcl`; the
        caller's V is never overwritten.
        """
        if kcl_A.shape[0] == 0:
            return scipy.linalg.solve(Z, V, overwrite_a=overwrite)
        n_b, n_p = V.shape
        n_c = kcl_A.shape[0]
        rhs = np.empty((n_b, n_p + n_c), dtype=np.complex128, order="F")
        rhs[:, :n_p] = V
        rhs[:, n_p:] = kcl_A.T
        sol = scipy.linalg.solve(Z, rhs, overwrite_a=overwrite, overwrite_b=True)
        W = sol[:, :n_p]
        X = sol[:, n_p:]
        Lam = scipy.linalg.solve(kcl_A @ X, kcl_A @ W)
        return W - X @ Lam

    def _solve_with_kcl_batch(self, Z, v, kcl_A):
        """k-batched KCL-constrained Schur solve. Z: (n_k, n_b, n_b),
        v: (n_b,) shared across k. Returns (n_k, n_b).

        The Schur algebra is basis-agnostic (it only sees Z and kcl_A):
        solves the saddle-point system [Z Aᵀ; A 0][I; λ] = [v; 0] without
        materializing the augmented matrix per k. Packs the
        (1 + n_c) right-hand sides into one stacked np.linalg.solve so
        the per-k LU factorisation is paid once.
        """
        n_k, n_b = Z.shape[0], Z.shape[1]
        if kcl_A.shape[0] == 0:
            rhs = np.broadcast_to(v[None, :, None], (n_k, n_b, 1))
            return np.linalg.solve(Z, rhs)[:, :, 0]
        n_c = kcl_A.shape[0]
        rhs = np.empty((n_k, n_b, 1 + n_c), dtype=np.complex128)
        rhs[:, :, 0] = v[None, :]
        rhs[:, :, 1:] = kcl_A.T[None, :, :]
        sol = np.linalg.solve(Z, rhs)  # (n_k, n_b, 1 + n_c)
        w = sol[:, :, 0]
        X = sol[:, :, 1:]
        S = np.einsum("cm,kmn->kcn", kcl_A, X)
        Aw = np.einsum("cm,km->kc", kcl_A, w)
        lam = np.linalg.solve(S, Aw[:, :, None])[:, :, 0]
        return w - np.einsum("kmc,kc->km", X, lam)

    def _solve_with_kcl_swept_ports(self, Z, V, kcl_A):
        """k- and port-batched KCL-constrained Schur solve.
        Z: (n_k, n_b, n_b), V: (n_b, n_p) shared across k. Returns
        (n_k, n_b, n_p) — the matrix-RHS generalisation of
        `_solve_with_kcl_batch`.
        """
        n_k = Z.shape[0]
        n_b, n_p = V.shape
        if kcl_A.shape[0] == 0:
            rhs = np.broadcast_to(V[None, :, :], (n_k, n_b, n_p))
            return np.linalg.solve(Z, rhs)
        n_c = kcl_A.shape[0]
        rhs = np.empty((n_k, n_b, n_p + n_c), dtype=np.complex128)
        rhs[:, :, :n_p] = V[None, :, :]
        rhs[:, :, n_p:] = kcl_A.T[None, :, :]
        sol = np.linalg.solve(Z, rhs)
        W = sol[:, :, :n_p]
        X = sol[:, :, n_p:]
        S = np.einsum("cm,kmn->kcn", kcl_A, X)
        AW = np.einsum("cm,kmp->kcp", kcl_A, W)
        Lam = np.linalg.solve(S, AW)
        return W - np.einsum("kmc,kcp->kmp", X, Lam)

    # ------------------------------------------------------------------
    # Driver impedance
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Singular basis enrichment at K≥3 junctions
    # ------------------------------------------------------------------

    def _enrichment_specs(self, geom, active_junction_indices=None):
        """For each (wire, end_pos) at a K≥enrichment_min_k junction, return
        the global segment index of the adjacent segment and the "orientation"
        flag: u_local from junction = u_seg_left if end_pos == "start",
        h - u_seg_left if end_pos == "end".

        When `active_junction_indices` is provided (a container of indices
        into self.junctions), only those junctions get specs — for the
        "auto" variant's per-junction enrichment selectivity. None means
        all qualifying junctions enrich (raw / stable / tikhonov path).
        """
        specs = []  # list of (junction_idx, wire_w, end_pos, seg_idx, u_origin)
        active_set = (
            None if active_junction_indices is None else set(active_junction_indices)
        )
        for j_idx, jw in enumerate(self.junctions):
            if len(jw) < self.enrichment_min_k:
                continue
            if active_set is not None and j_idx not in active_set:
                continue
            for wire_w, end_pos in jw:
                if end_pos == "start":
                    seg_idx = geom["seg_offsets"][wire_w]
                    u_origin = "left"  # u from junction = u_seg_left
                else:
                    seg_idx = geom["seg_offsets"][wire_w + 1] - 1
                    u_origin = "right"  # u from junction = h - u_seg_left
                specs.append((j_idx, wire_w, end_pos, seg_idx, u_origin))
        return specs

    def _junction_tap_ratios(self, coeffs_poly):
        """Compute tap_ratio = min(|I_wire|) / max(|I_wire|) at each
        junction node, using the polynomial-only coefficient vector from
        a no-enrichment solve. Returns a list of length len(self.junctions);
        entries are None for junctions with K < enrichment_min_k.

        At a junction node only the bspline directional bases are nonzero
        (one per wire end), so the wire-by-wire current magnitudes can be
        read directly out of `currents_at_knots` at the junction-side knot.
        Per-wire |I| values at a K-junction sum to zero by KCL (vector
        sum), but individual magnitudes need not be equal — the ratio
        captures how lopsided the split is.
        """
        # Temporarily disable enrichment so currents_at_knots doesn't try
        # to index past the polynomial block.
        saved = self.use_singular_enrichment
        self.use_singular_enrichment = False
        try:
            I_per_wire = self.currents_at_knots(coeffs_poly)
        finally:
            self.use_singular_enrichment = saved

        ratios = []
        for jw in self.junctions:
            if len(jw) < self.enrichment_min_k:
                ratios.append(None)
                continue
            mags = []
            for wire_w, end_pos in jw:
                knot_idx = 0 if end_pos == "start" else -1
                mags.append(abs(I_per_wire[wire_w][knot_idx]))
            m_max = max(mags)
            ratios.append(0.0 if m_max == 0.0 else min(mags) / m_max)
        return ratios

    @staticmethod
    def _assemble_Z_enrich_numpy(
        spec_seg,
        spec_origin,
        seg_l,
        seg_r,
        h_per_seg,
        td_all,
        supp_seg_poly,
        polys_poly,
        a_squared,
        k,
        omega,
        eps_,
        mu_,
        gl_t01,
        gl_w01,
        proj_coeffs,
    ):
        """Pure-numpy reference for the C++ `assemble_Z_enrich` kernel.

        Mirrors the C++ structure (precompute Φ_sing values/derivatives
        and 3D positions at the enrichment-side quadrature nodes, then
        nested-loop Galerkin sums for Z_ee / Z_pe / Z_ep) so the parity
        test in tests/test_momwire.py catches any drift between the two.

        Also lets BSplineSolver(use_singular_enrichment=True) work without
        the C++ accelerator at all — Windows users hit this path because
        setup.py skips the Pybind11Extension under MSVC (the GCC-only
        `-fopenmp` / `-mavx2` / `-lmvec` flags don't link there).

        Same argument names and semantics as the C++ binding. `proj_coeffs`
        of length d+1 selects raw (all zeros → Φ_sing = t·log(t)) vs
        stable XFEM (subtract Σ_p proj_coeffs[p] t^p from both Φ and its
        derivative).
        """
        n_enrich = spec_seg.shape[0]
        n_poly, n_wings = supp_seg_poly.shape
        d_plus_1 = polys_poly.shape[2]
        n_qp = gl_t01.shape[0]

        Z_pe = np.zeros((n_poly, n_enrich), dtype=np.complex128)
        Z_ep = np.zeros((n_enrich, n_poly), dtype=np.complex128)
        Z_ee = np.zeros((n_enrich, n_enrich), dtype=np.complex128)
        if n_enrich == 0:
            return Z_pe, Z_ep, Z_ee

        inv_4pi = 1.0 / (4.0 * np.pi)
        omega_mu = omega * mu_
        inv_omega_eps = 1.0 / (omega * eps_)
        eps_tiny = 1e-300

        # Derivative coefficients in monomial basis for the proj polynomial:
        # P(t) = Σ_p c_p t^p ⇒ P'(t) = Σ_{p≥1} p·c_p t^(p-1).
        proj_deriv_coeffs = proj_coeffs[1:] * np.arange(1, d_plus_1)

        # Per-enrichment precompute.
        pos_e_all = np.zeros((n_enrich, n_qp, 3))
        sing_val_all = np.zeros((n_enrich, n_qp))
        sing_dval_all = np.zeros((n_enrich, n_qp))
        w_e_all = np.zeros((n_enrich, n_qp))
        polyval = np.polynomial.polynomial.polyval
        for e in range(n_enrich):
            se = int(spec_seg[e])
            orig = int(spec_origin[e])
            he = h_per_seg[se]
            dphi_sign = 1.0 if orig == 0 else -1.0
            t = gl_t01
            u_norm = t if orig == 0 else (1.0 - t)
            u_safe = np.where(u_norm > eps_tiny, u_norm, eps_tiny)
            log_u = np.log(u_safe)
            poly_val = polyval(u_norm, proj_coeffs)
            if proj_deriv_coeffs.size > 0:
                poly_dval = polyval(u_norm, proj_deriv_coeffs)
            else:
                poly_dval = np.zeros_like(u_norm)
            sing_val_all[e] = u_norm * log_u - poly_val
            sing_dval_all[e] = dphi_sign * (log_u + 1.0 - poly_dval) / he
            w_e_all[e] = gl_w01 * he
            pos_e_all[e, :, 0] = (1.0 - t) * seg_l[se, 0] + t * seg_r[se, 0]
            pos_e_all[e, :, 1] = (1.0 - t) * seg_l[se, 1] + t * seg_r[se, 1]
            pos_e_all[e, :, 2] = (1.0 - t) * seg_l[se, 2] + t * seg_r[se, 2]

        seg_e_arr = spec_seg.astype(np.int64, copy=False)

        # Z_ee: symmetric. Fill upper triangle, mirror.
        for e in range(n_enrich):
            for f in range(e, n_enrich):
                td = td_all[seg_e_arr[e], seg_e_arr[f]]
                diff = pos_e_all[e, :, None, :] - pos_e_all[f, None, :, :]
                R = np.sqrt(np.sum(diff * diff, axis=-1) + a_squared)
                iR_4pi = inv_4pi / R
                phase = -k * R
                Gre = np.cos(phase) * iR_4pi
                Gim = np.sin(phase) * iR_4pi
                wprod_A = (w_e_all[e] * sing_val_all[e])[:, None] * (
                    w_e_all[f] * sing_val_all[f]
                )[None, :]
                wprod_P = (w_e_all[e] * sing_dval_all[e])[:, None] * (
                    w_e_all[f] * sing_dval_all[f]
                )[None, :]
                IA_re = np.sum(wprod_A * Gre)
                IA_im = np.sum(wprod_A * Gim)
                IP_re = np.sum(wprod_P * Gre)
                IP_im = np.sum(wprod_P * Gim)
                # Z = jωμ·td·I_A + I_Φ / (jωε)
                Zre = -omega_mu * td * IA_im + IP_im * inv_omega_eps
                Zim = omega_mu * td * IA_re - IP_re * inv_omega_eps
                Z_ee[e, f] = complex(Zre, Zim)
                if e != f:
                    Z_ee[f, e] = Z_ee[e, f]

        # Z_pe / Z_ep. Loop over polynomial bases and their wings;
        # for each wing, precompute poly value/derivative + 3D quad-point
        # positions on the wing's segment, then loop over enrichments.
        # Z_pe and Z_ep are computed independently (no .T shortcut) to
        # mirror the C++ kernel — the kernel is symmetric, so the totals
        # agree to floating-point rounding either way.
        for m in range(n_poly):
            for w_idx in range(n_wings):
                cw = polys_poly[m, w_idx]
                if not np.any(cw != 0.0):
                    continue
                seg_m = int(supp_seg_poly[m, w_idx])
                hm = h_per_seg[seg_m]
                t = gl_t01
                u_arc = t * hm
                pv = polyval(u_arc, cw)
                cw_deriv = cw[1:] * np.arange(1, d_plus_1)
                if cw_deriv.size > 0:
                    dv = polyval(u_arc, cw_deriv)
                else:
                    dv = np.zeros_like(u_arc)
                w_m = gl_w01 * hm
                pos_m = np.empty((n_qp, 3))
                pos_m[:, 0] = (1.0 - t) * seg_l[seg_m, 0] + t * seg_r[seg_m, 0]
                pos_m[:, 1] = (1.0 - t) * seg_l[seg_m, 1] + t * seg_r[seg_m, 1]
                pos_m[:, 2] = (1.0 - t) * seg_l[seg_m, 2] + t * seg_r[seg_m, 2]
                for e in range(n_enrich):
                    seg_e = int(seg_e_arr[e])
                    td_me = td_all[seg_m, seg_e]
                    td_em = td_all[seg_e, seg_m]
                    # Z_pe leg: i = m-axis, j = e-axis
                    diff = pos_m[:, None, :] - pos_e_all[e, None, :, :]
                    R = np.sqrt(np.sum(diff * diff, axis=-1) + a_squared)
                    iR_4pi = inv_4pi / R
                    phase = -k * R
                    Gre = np.cos(phase) * iR_4pi
                    Gim = np.sin(phase) * iR_4pi
                    wprod_A = (w_m * pv)[:, None] * (w_e_all[e] * sing_val_all[e])[
                        None, :
                    ]
                    wprod_P = (w_m * dv)[:, None] * (w_e_all[e] * sing_dval_all[e])[
                        None, :
                    ]
                    pe_IA_re = np.sum(wprod_A * Gre)
                    pe_IA_im = np.sum(wprod_A * Gim)
                    pe_IP_re = np.sum(wprod_P * Gre)
                    pe_IP_im = np.sum(wprod_P * Gim)
                    # Z_ep leg: i = e-axis, j = m-axis
                    diff = pos_e_all[e, :, None, :] - pos_m[None, :, :]
                    R = np.sqrt(np.sum(diff * diff, axis=-1) + a_squared)
                    iR_4pi = inv_4pi / R
                    phase = -k * R
                    Gre = np.cos(phase) * iR_4pi
                    Gim = np.sin(phase) * iR_4pi
                    wprod_A = (w_e_all[e] * sing_val_all[e])[:, None] * (w_m * pv)[
                        None, :
                    ]
                    wprod_P = (w_e_all[e] * sing_dval_all[e])[:, None] * (w_m * dv)[
                        None, :
                    ]
                    ep_IA_re = np.sum(wprod_A * Gre)
                    ep_IA_im = np.sum(wprod_A * Gim)
                    ep_IP_re = np.sum(wprod_P * Gre)
                    ep_IP_im = np.sum(wprod_P * Gim)
                    Z_pe[m, e] += complex(
                        -omega_mu * td_me * pe_IA_im + pe_IP_im * inv_omega_eps,
                        omega_mu * td_me * pe_IA_re - pe_IP_re * inv_omega_eps,
                    )
                    Z_ep[e, m] += complex(
                        -omega_mu * td_em * ep_IA_im + ep_IP_im * inv_omega_eps,
                        omega_mu * td_em * ep_IA_re - ep_IP_re * inv_omega_eps,
                    )

        return Z_pe, Z_ep, Z_ee

    @staticmethod
    def _assemble_Z_enrich_image_numpy(
        spec_seg,
        spec_origin,
        seg_l,
        seg_r,
        h_per_seg,
        w_A_all,
        w_Phi_all,
        supp_seg_poly,
        polys_poly,
        a_squared,
        k,
        omega,
        eps_,
        mu_,
        gl_t01,
        gl_w01,
        proj_coeffs,
        ground_z,
    ):
        """Ground-image reaction blocks for the enrichment DOFs (#167).

        The image counterpart of `_assemble_Z_enrich_numpy`: the reaction of
        each real observer basis against the ground-plane image of every
        source basis. Same Galerkin kernel and mixing algebra
        (`Z = jωμ·w_A·I_A + w_Φ·I_Φ/(jωε)`), with the two changes that define
        the polynomial image path (`_image_Z_weighted`):

          * the source-side quadrature positions are mirrored across
            `z = ground_z` (the `_image_positions` reflection), and
          * the free-space tangent dot on the A term becomes the per-segment-
            pair weight table `w_A_all[seg_m, seg_n]`, and the charge term
            picks up `w_Phi_all[seg_m, seg_n]`.

        The same weight tables the polynomial block uses, so both grounds are
        one code path: **PEC image** passes `w_A = t_m·(t_n,x, t_n,y, -t_n,z)`
        (the mirror tangent dot) and `w_Φ = 1`; the **fast finite ground**
        (refl-coef) passes the Fresnel dyad table `_ground_refl.a_term_weights`
        and the image-charge table `_ground_refl.phi_term_weights`. The caller
        SUBTRACTS the returned blocks from the free-space (Z_pe, Z_ep, Z_ee):
        the single global minus captures both the image current's anti-parallel
        horizontal direction and the image charge's sign flip (PEC), or is
        absorbed into the complex Fresnel weights (finite).

        numpy-only: there are only a few enrichment DOFs (one per
        K≥enrichment_min_k junction), so this O(n_enrich·(n_enrich + n_poly))
        assembly is negligible beside the O(N²) polynomial image fill and
        needs no C++ twin. `a_squared` regularises the reduced kernel exactly
        as in the free-space path, so a ground-touching wire (image coincident
        with the wire) stays finite.
        """
        n_enrich = spec_seg.shape[0]
        n_poly, n_wings = supp_seg_poly.shape
        d_plus_1 = polys_poly.shape[2]
        n_qp = gl_t01.shape[0]

        Z_pe = np.zeros((n_poly, n_enrich), dtype=np.complex128)
        Z_ep = np.zeros((n_enrich, n_poly), dtype=np.complex128)
        Z_ee = np.zeros((n_enrich, n_enrich), dtype=np.complex128)
        if n_enrich == 0:
            return Z_pe, Z_ep, Z_ee

        inv_4pi = 1.0 / (4.0 * np.pi)
        omega_mu = omega * mu_
        inv_omega_eps = 1.0 / (omega * eps_)
        eps_tiny = 1e-300

        proj_deriv_coeffs = proj_coeffs[1:] * np.arange(1, d_plus_1)

        # Per-enrichment precompute — identical to the free-space reference
        # (kept in lockstep by inspection); then mirror the positions to serve
        # as the ground-plane image sources.
        pos_e_all = np.zeros((n_enrich, n_qp, 3))
        sing_val_all = np.zeros((n_enrich, n_qp))
        sing_dval_all = np.zeros((n_enrich, n_qp))
        w_e_all = np.zeros((n_enrich, n_qp))
        polyval = np.polynomial.polynomial.polyval
        for e in range(n_enrich):
            se = int(spec_seg[e])
            orig = int(spec_origin[e])
            he = h_per_seg[se]
            dphi_sign = 1.0 if orig == 0 else -1.0
            t = gl_t01
            u_norm = t if orig == 0 else (1.0 - t)
            u_safe = np.where(u_norm > eps_tiny, u_norm, eps_tiny)
            log_u = np.log(u_safe)
            poly_val = polyval(u_norm, proj_coeffs)
            if proj_deriv_coeffs.size > 0:
                poly_dval = polyval(u_norm, proj_deriv_coeffs)
            else:
                poly_dval = np.zeros_like(u_norm)
            sing_val_all[e] = u_norm * log_u - poly_val
            sing_dval_all[e] = dphi_sign * (log_u + 1.0 - poly_dval) / he
            w_e_all[e] = gl_w01 * he
            pos_e_all[e, :, 0] = (1.0 - t) * seg_l[se, 0] + t * seg_r[se, 0]
            pos_e_all[e, :, 1] = (1.0 - t) * seg_l[se, 1] + t * seg_r[se, 1]
            pos_e_all[e, :, 2] = (1.0 - t) * seg_l[se, 2] + t * seg_r[se, 2]

        # Image sources: mirror z across the ground plane.
        pos_e_img = pos_e_all.copy()
        pos_e_img[..., 2] = 2.0 * ground_z - pos_e_img[..., 2]

        seg_e_arr = spec_seg.astype(np.int64, copy=False)

        # Z_ee image: real observer e against image source f. The image
        # reaction is symmetric (a mirror is an isometry, so reciprocity
        # holds), but both halves are computed independently to mirror the
        # free-space "no .T shortcut" convention. Complex per-pair weights
        # w_A / w_Φ (PEC: real td / 1; finite: Fresnel) fold in via
        # Z = jωμ·w_A·I_A + w_Φ·I_Φ/(jωε).
        for e in range(n_enrich):
            for f in range(n_enrich):
                w_A = w_A_all[seg_e_arr[e], seg_e_arr[f]]
                w_Phi = w_Phi_all[seg_e_arr[e], seg_e_arr[f]]
                diff = pos_e_all[e, :, None, :] - pos_e_img[f, None, :, :]
                R = np.sqrt(np.sum(diff * diff, axis=-1) + a_squared)
                iR_4pi = inv_4pi / R
                phase = -k * R
                Gre = np.cos(phase) * iR_4pi
                Gim = np.sin(phase) * iR_4pi
                wprod_A = (w_e_all[e] * sing_val_all[e])[:, None] * (
                    w_e_all[f] * sing_val_all[f]
                )[None, :]
                wprod_P = (w_e_all[e] * sing_dval_all[e])[:, None] * (
                    w_e_all[f] * sing_dval_all[f]
                )[None, :]
                I_A = complex(np.sum(wprod_A * Gre), np.sum(wprod_A * Gim))
                I_P = complex(np.sum(wprod_P * Gre), np.sum(wprod_P * Gim))
                Z_ee[e, f] = (
                    1j * omega_mu * w_A * I_A - 1j * w_Phi * I_P * inv_omega_eps
                )

        # Z_pe / Z_ep image: real polynomial observer against image
        # enrichment source (Z_pe) and real enrichment observer against image
        # polynomial source (Z_ep). Both mirror one side and use the image
        # tangent dot; the two agree by reciprocity but are computed apart.
        for m in range(n_poly):
            for w_idx in range(n_wings):
                cw = polys_poly[m, w_idx]
                if not np.any(cw != 0.0):
                    continue
                seg_m = int(supp_seg_poly[m, w_idx])
                hm = h_per_seg[seg_m]
                t = gl_t01
                u_arc = t * hm
                pv = polyval(u_arc, cw)
                cw_deriv = cw[1:] * np.arange(1, d_plus_1)
                if cw_deriv.size > 0:
                    dv = polyval(u_arc, cw_deriv)
                else:
                    dv = np.zeros_like(u_arc)
                w_m = gl_w01 * hm
                pos_m = np.empty((n_qp, 3))
                pos_m[:, 0] = (1.0 - t) * seg_l[seg_m, 0] + t * seg_r[seg_m, 0]
                pos_m[:, 1] = (1.0 - t) * seg_l[seg_m, 1] + t * seg_r[seg_m, 1]
                pos_m[:, 2] = (1.0 - t) * seg_l[seg_m, 2] + t * seg_r[seg_m, 2]
                pos_m_img = pos_m.copy()
                pos_m_img[:, 2] = 2.0 * ground_z - pos_m_img[:, 2]
                for e in range(n_enrich):
                    seg_e = int(seg_e_arr[e])
                    wA_me = w_A_all[seg_m, seg_e]
                    wPhi_me = w_Phi_all[seg_m, seg_e]
                    wA_em = w_A_all[seg_e, seg_m]
                    wPhi_em = w_Phi_all[seg_e, seg_m]
                    # Z_pe leg: real poly m vs image enrichment e.
                    diff = pos_m[:, None, :] - pos_e_img[e, None, :, :]
                    R = np.sqrt(np.sum(diff * diff, axis=-1) + a_squared)
                    iR_4pi = inv_4pi / R
                    phase = -k * R
                    Gre = np.cos(phase) * iR_4pi
                    Gim = np.sin(phase) * iR_4pi
                    wprod_A = (w_m * pv)[:, None] * (w_e_all[e] * sing_val_all[e])[
                        None, :
                    ]
                    wprod_P = (w_m * dv)[:, None] * (w_e_all[e] * sing_dval_all[e])[
                        None, :
                    ]
                    pe_I_A = complex(np.sum(wprod_A * Gre), np.sum(wprod_A * Gim))
                    pe_I_P = complex(np.sum(wprod_P * Gre), np.sum(wprod_P * Gim))
                    # Z_ep leg: real enrichment e vs image poly m.
                    diff = pos_e_all[e, :, None, :] - pos_m_img[None, :, :]
                    R = np.sqrt(np.sum(diff * diff, axis=-1) + a_squared)
                    iR_4pi = inv_4pi / R
                    phase = -k * R
                    Gre = np.cos(phase) * iR_4pi
                    Gim = np.sin(phase) * iR_4pi
                    wprod_A = (w_e_all[e] * sing_val_all[e])[:, None] * (w_m * pv)[
                        None, :
                    ]
                    wprod_P = (w_e_all[e] * sing_dval_all[e])[:, None] * (w_m * dv)[
                        None, :
                    ]
                    ep_I_A = complex(np.sum(wprod_A * Gre), np.sum(wprod_A * Gim))
                    ep_I_P = complex(np.sum(wprod_P * Gre), np.sum(wprod_P * Gim))
                    Z_pe[m, e] += (
                        1j * omega_mu * wA_me * pe_I_A
                        - 1j * wPhi_me * pe_I_P * inv_omega_eps
                    )
                    Z_ep[e, m] += (
                        1j * omega_mu * wA_em * ep_I_A
                        - 1j * wPhi_em * ep_I_P * inv_omega_eps
                    )

        return Z_pe, Z_ep, Z_ee

    def _enrichment_Z_assemble(
        self, geom, supp_seg_poly, polys_poly, active_junction_indices=None
    ):
        """Assemble the (Z_pe, Z_ep, Z_ee) enrichment blocks via the C++
        accelerator (`momwire._accelerators.assemble_Z_enrich`).

        Z = [[Z_pp, Z_pe],
             [Z_ep, Z_ee]]
        with n_poly + n_enrich basis functions. Z_pp is built by the
        existing polynomial assembly; the three new blocks come from
        Gauss-Legendre quadrature over the (u·log(u/h))-shaped singular
        basis adjacent to each K≥`enrichment_min_k` junction. Z_pe and
        Z_ep are computed independently — the Galerkin .T shortcut is
        mathematically exact for symmetric kernels but the productized
        path computes both halves so a future quadrature change can't
        silently break symmetry.
        """
        specs = self._enrichment_specs(
            geom, active_junction_indices=active_junction_indices
        )
        n_enrich = len(specs)
        if n_enrich == 0:
            return None  # no qualifying junctions → no-op

        spec_seg = np.fromiter((s[3] for s in specs), dtype=np.int64, count=n_enrich)
        spec_origin = np.fromiter(
            (0 if s[4] == "left" else 1 for s in specs),
            dtype=np.int64,
            count=n_enrich,
        )

        tangents = geom["tangents"]
        td_all = np.ascontiguousarray(tangents @ tangents.T, dtype=np.float64)

        gl_xi, gl_w = leggauss(self.n_qp_sing)
        t01 = 0.5 * (gl_xi + 1.0)
        w01 = 0.5 * gl_w

        # "stable" variant subtracts the L²-projection of Φ_sing onto
        # the BC-preserving polynomial bubble subspace; "raw" and
        # "tikhonov" both send zero coefficients (the tikhonov knob is
        # applied at solve time to Z_ee, not at the basis level).
        if self.enrichment_variant == "stable":
            proj_coeffs = _xfem_projection_coeffs(self.degree)
        else:
            proj_coeffs = np.zeros(self.degree + 1)

        seg_l_arr = np.ascontiguousarray(geom["seg_l"], dtype=np.float64)
        seg_r_arr = np.ascontiguousarray(geom["seg_r"], dtype=np.float64)
        h_arr = np.ascontiguousarray(geom["h_per_seg"], dtype=np.float64)
        supp_arr = np.ascontiguousarray(supp_seg_poly, dtype=np.int64)
        polys_arr = np.ascontiguousarray(polys_poly, dtype=np.float64)
        a_squared = float(self._uniform_radius) ** 2
        t01_arr = np.ascontiguousarray(t01, dtype=np.float64)
        w01_arr = np.ascontiguousarray(w01, dtype=np.float64)
        proj_arr = np.ascontiguousarray(proj_coeffs, dtype=np.float64)

        kernel_args = (
            spec_seg,
            spec_origin,
            seg_l_arr,
            seg_r_arr,
            h_arr,
            td_all,
            supp_arr,
            polys_arr,
            a_squared,
            float(self.k),
            float(self.omega),
            float(self.eps),
            float(self.mu),
            t01_arr,
            w01_arr,
            proj_arr,
        )
        if _HAVE_ENRICH_ACCEL:
            Z_pe, Z_ep, Z_ee = _acc.assemble_Z_enrich(*kernel_args)
        else:
            Z_pe, Z_ep, Z_ee = self._assemble_Z_enrich_numpy(*kernel_args)

        if self.ground_z is not None:
            # Ground image reaction for the enrichment DOFs (#167). Same
            # global-minus + per-segment-pair weight convention as the
            # polynomial image block. Three grounds, one image kernel with the
            # matching weight tables:
            #   PEC        — mirror tangent dot on A, unit charge weight;
            #   refl-coef  — Fresnel weight tables (`_image_refl_weights`);
            #   sommerfeld — the C2 exact-image weights, PLUS the smooth
            #                remainder-field reaction added below.
            # numpy-only — the handful of enrichment DOFs make the cost
            # negligible beside the poly image fill.
            sommerfeld = (
                self.ground_eps is not None and self.ground_model == "sommerfeld"
            )
            if sommerfeld:
                eps_t = _ground_refl.eps_tilde(self.ground_eps, self.omega, self.eps)
                c2 = (eps_t - 1.0) / (eps_t + 1.0)
                td_img = self._image_tangent_dot(tangents)
                w_A_all = np.ascontiguousarray(c2 * td_img, dtype=np.complex128)
                w_Phi_all = np.full_like(w_A_all, c2)
            elif self.ground_eps is not None:
                w_A_all, w_Phi_all = self._image_refl_weights(
                    self._image_refl_prep(geom), self.omega
                )
                w_A_all = np.ascontiguousarray(w_A_all, dtype=np.complex128)
                w_Phi_all = np.ascontiguousarray(w_Phi_all, dtype=np.complex128)
            else:
                w_A_all = np.ascontiguousarray(
                    self._image_tangent_dot(tangents), dtype=np.complex128
                )
                w_Phi_all = np.ones_like(w_A_all)
            Z_pe_img, Z_ep_img, Z_ee_img = self._assemble_Z_enrich_image_numpy(
                spec_seg,
                spec_origin,
                seg_l_arr,
                seg_r_arr,
                h_arr,
                w_A_all,
                w_Phi_all,
                supp_arr,
                polys_arr,
                a_squared,
                float(self.k),
                float(self.omega),
                float(self.eps),
                float(self.mu),
                t01_arr,
                w01_arr,
                proj_arr,
                float(self.ground_z),
            )
            Z_pe = Z_pe - Z_pe_img
            Z_ep = Z_ep - Z_ep_img
            Z_ee = Z_ee - Z_ee_img

            if sommerfeld:
                # The exact-image part above is only the C2-scaled image; the
                # Sommerfeld ground adds the smooth remainder field. Subtract
                # its enrichment reaction (one minus, matching the poly
                # `_ground_finite_Z` = C2-image + Q convention). In the
                # ε̃ → ∞ limit C2 → 1 and Q → 0, so this collapses to PEC.
                Q_pe, Q_ep, Q_ee = self._Q_sommerfeld_remainder_enrich(
                    geom, supp_arr, polys_arr, spec_seg, spec_origin, eps_t
                )
                Z_pe = Z_pe - Q_pe
                Z_ep = Z_ep - Q_ep
                Z_ee = Z_ee - Q_ee

        return {
            "specs": specs,
            "n_enrich": n_enrich,
            "Z_pe": Z_pe,
            "Z_ep": Z_ep,
            "Z_ee": Z_ee,
        }

    # ------------------------------------------------------------------
    # Driver impedance
    # ------------------------------------------------------------------

    def _dense_tensor_fits_budget(self, n_segs):
        """Whether one dense polynomial-moment tensor fits the memory budget."""
        tensor_bytes = (
            (self.degree + 1) ** 2 * int(n_segs) ** 2 * np.dtype(np.complex128).itemsize
        )
        return tensor_bytes <= (self.swept_mem_mb << 20)

    def _compute_Z_operator(self, geom, supp_seg, polys, same_edge_prep=None):
        """Loaded (free-space or grounded) dense Z for one k — the operator
        construction shared by `compute_impedance` and `compute_y_matrix`.

        Dispatches to the chunked fill+assemble (issue #136) whenever the
        (d+1, d+1, N, N) moment tensor would blow the `swept_mem_mb`
        budget. `compute_y_matrix` used to bypass that dispatch and
        materialise the full tensor unconditionally — a flat ~12x n²·16 B
        peak on exactly the entry point the SimNEC portal and the array
        benchmarks drive (issue #235).
        """
        dense_tensor_fits = self._dense_tensor_fits_budget(geom["n_segs_total"])

        self._checkpoint()  # after geometry/basis, before the J-block fill
        if (
            _HAVE_BSPLINE_WINDOWED_ASSEMBLE_ACCEL
            and self.degree <= _BSPLINE_ASSEMBLE_ACCEL_MAX_D
            and not dense_tensor_fits
        ):
            # Chunked fill+assemble: never materialises the full
            # (d+1, d+1, N, N) tensor (issue #136). Identical algebra to
            # the tensor path below, bounded transients.
            Z = self._compute_Z_dense_chunked(
                geom, self.k, supp_seg, polys, same_edge_prep=same_edge_prep
            )
        else:
            J = self._build_J_blocks(geom, self.k, same_edge_prep=same_edge_prep)
            Z = self._assemble_Z(J, supp_seg, polys, geom)
            del J

        if self.ground_z is not None:
            self._checkpoint()  # between fills: before the image J-block fill
            # PEC image method: subtract the same-shape assembly built from
            # J integrals over image segments + (tx, ty, -tz)-modified
            # tangent dot products. The minus sign captures both the
            # image current's horizontal anti-parallel direction and the
            # image charge's sign flip (one minus combined).
            if (
                _HAVE_BSPLINE_W_WINDOWED_ASSEMBLE_ACCEL
                and self.degree <= _BSPLINE_ASSEMBLE_ACCEL_MAX_D
            ):
                # Chunked image subtraction — no (d+1, d+1, N, N) image
                # tensor, no intermediate n_basis² matrix (issue #136), and
                # no (N, N) weight table either: `_image_weight_window_fn`
                # picks the mode and produces each observer chunk's window
                # on demand (issue #323). The Sommerfeld remainder Q is a
                # separate, already observer-chunked term.
                self._accumulate_Z_image_chunked(
                    Z,
                    geom,
                    self.k,
                    supp_seg,
                    polys,
                    self._image_weight_window_fn(geom),
                )
                if self.ground_eps is not None and self.ground_model == "sommerfeld":
                    eps_t = _ground_refl.eps_tilde(
                        self.ground_eps, self.omega, self.eps
                    )
                    Z -= self._Z_sommerfeld_remainder(geom, supp_seg, polys, eps_t)
            else:
                J_img = self._build_J_image_blocks(geom, self.k)
                if self.ground_eps is not None:
                    # Finite ground: Fresnel-weighted image (same J fill,
                    # per-pair weight tables on both terms) instead of the
                    # PEC dot.
                    Z = Z - self._ground_finite_Z(J_img, supp_seg, polys, geom)
                else:
                    td_img = self._image_tangent_dot(geom["tangents"])
                    Z = Z - self._assemble_Z(
                        J_img, supp_seg, polys, geom, td_all=td_img
                    )

        # Distributed series wire loading (independent of ground: it's a
        # wire property, added once to the final Z).
        return self._apply_loading(Z)

    def _port_count(self):
        """Ports `compute_port_solution` returns: [gap feeds…, junction
        ports…, node gaps…]. Answered from the configuration, without
        solving."""
        return len(self.feeds) + len(self.junction_ports) + len(self.node_gaps)

    def _node_gap_columns(self, wire_basis_global, n_basis_total):
        """`(cols, volts)` for the series node gaps (issue #305), in
        `self.node_gaps` order: column p is the σ-signed unit indicator of the
        named wire-end's directional basis — drive AND readout vector, like
        every other port column, so mixed-port Y stays symmetric. σ is the
        KCL outflow sign (+1 start / −1 end): I_port is the current flowing
        from the node into the named wire, which makes the port invariant
        under re-parametrizing the wire. k-independent."""
        cols = np.zeros((n_basis_total, len(self.node_gaps)), dtype=np.float64)
        volts = np.zeros(len(self.node_gaps), dtype=np.complex128)
        if not self.node_gaps:
            return cols, volts
        grounded = self._grounded_junctions()
        end_to_junction = {}
        for j, jw in enumerate(self.junctions):
            for member in jw:
                end_to_junction[member] = j
        for p, (w_i, end_i, v_i) in enumerate(self.node_gaps):
            j_idx = end_to_junction[(w_i, end_i)]
            if j_idx in grounded:
                raise ValueError(
                    f"node_gaps[{p}]: junction {j_idx} is grounded — a series "
                    "gap between a wire and the ground stake is not supported "
                    "(#151 grounds the node through the image instead)"
                )
            kept, local_to_global = wire_basis_global[w_i]
            m_global = None
            for kept_idx, (_j, kind, junc_idx, end_pos) in enumerate(kept):
                if kind == "dir" and junc_idx == j_idx and end_pos == end_i:
                    m_global = local_to_global[kept_idx]
                    break
            assert m_global is not None, (w_i, end_i, j_idx)
            cols[m_global, p] = +1.0 if end_i == "start" else -1.0
            volts[p] = v_i
        return cols, volts

    def _gap_source_vectors(self, geom, wire_knots, wire_basis_global, n_basis_total):
        """One unit Galerkin source vector per configured gap feed, in feed
        order. k-independent (it only integrates basis shapes against the
        delta gap), which is why every swept path hoists it out of the loop.
        """
        v_per_feed = []
        for w_i, arc_i, _v in self.feeds:
            arc_at_knot = geom["per_wire"][w_i]["arc_at_knot"]
            s_f_i = arc_i if arc_i is not None else arc_at_knot[-1] / 2.0
            v_per_feed.append(
                self._build_source_vector(
                    geom,
                    wire_knots,
                    wire_basis_global,
                    n_basis_total,
                    wi=w_i,
                    s_f=s_f_i,
                )
            )
        return v_per_feed

    def _port_columns(self, geom, wire_knots, wire_basis_global, n_basis_total, kcl_A):
        """`(B, kcl_con)`: the unit-drive/readout column per port and the
        constraint rows left over, shared by every entry point that solves
        all ports at once (#252).

        Column j of `B` is port j's Galerkin source vector AND its readout
        vector — the reciprocity that makes `Y = Bᵀ X` symmetric. Ports run
        [gap feeds…, junction ports…, node gaps…]: a ported junction's KCL row
        (#172) leaves the constraint set and becomes a column here, which is
        why the returned `kcl_con` is shorter than the `kcl_A` handed in; a
        node gap's column (#305) is the σ-signed indicator of its wire-end's
        directional basis, and its junction's KCL row stays a constraint. All
        of it is k-independent, so the swept paths build it once for the sweep.
        """
        B = np.zeros((n_basis_total, len(self.feeds)), dtype=np.complex128)
        for j, v_j in enumerate(
            self._gap_source_vectors(geom, wire_knots, wire_basis_global, n_basis_total)
        ):
            B[:, j] = v_j
        kcl_con, port_A, _port_V = self._split_kcl_ports(kcl_A)
        gap_cols, _gap_V = self._node_gap_columns(wire_basis_global, n_basis_total)
        return (
            np.hstack(
                [B, port_A.T.astype(np.complex128), gap_cols.astype(np.complex128)]
            ),
            kcl_con,
        )

    def _feed_drive_and_readout(
        self, geom, wire_knots, wire_basis_global, n_basis_total, kcl_A
    ):
        """The single-excitation drive/readout algebra shared by
        `compute_impedance` and its batched sweep (#252).

        Returns `(v, port_vectors, vpf_T, all_voltages, kcl_con)`: the RHS for
        the configured voltages, the per-port readout vectors as a list and as
        the `(n_basis, n_ports)` matrix the batched sweep contracts against,
        the port voltages in [gap feeds…, junction ports…, node gaps…] order,
        and the constraint rows left after the ported junctions become drives.
        """
        n_feeds = len(self.feeds)
        v_per_feed = self._gap_source_vectors(
            geom, wire_knots, wire_basis_global, n_basis_total
        )
        voltages = np.array([v for _, _, v in self.feeds], dtype=np.complex128)
        v = np.zeros(n_basis_total, dtype=np.complex128)
        for V_i, v_i in zip(voltages, v_per_feed):
            v += V_i * v_i

        # Junction ports (issue #172): drive v += V_p·A_p per port and
        # append the A_p rows to the readout set; the port-junction KCL
        # rows leave the constraint matrix.
        kcl_con, port_A, port_V = self._split_kcl_ports(kcl_A)
        v += port_V @ port_A
        # Node gaps (issue #305): same drive/readout algebra on the σ-signed
        # wire-end indicator columns; their junctions' KCL rows stay put.
        gap_cols, gap_V = self._node_gap_columns(wire_basis_global, n_basis_total)
        v += gap_cols @ gap_V
        port_vectors = (
            v_per_feed
            + [port_A[i] for i in range(port_A.shape[0])]
            + [gap_cols[:, p] for p in range(gap_cols.shape[1])]
        )
        all_voltages = np.concatenate([voltages, port_V, gap_V])
        # Reshape keeps this 2-D at n_feeds == 0: np.array([]) is (0,) and
        # would break the hstack against port_A's rows (issue #175).
        vpf = np.asarray(v_per_feed, dtype=np.complex128).reshape(
            n_feeds, n_basis_total
        )
        vpf_T = np.hstack(
            [
                vpf.T,
                port_A.T.astype(np.complex128),
                gap_cols.astype(np.complex128),
            ]
        )
        return v, port_vectors, vpf_T, all_voltages, kcl_con

    def compute_impedance(self, same_edge_prep=None):
        geom = self._build_geometry()
        supp_seg, polys, kcl_A, wire_knots, wire_basis_global = (
            self._build_basis_polynomials(geom)
        )
        n_basis_total = supp_seg.shape[0]
        Z = self._compute_Z_operator(
            geom, supp_seg, polys, same_edge_prep=same_edge_prep
        )

        # Per-feed unit Galerkin source vectors. For multi-feed, the
        # combined RHS is Σ_i V_i · v_i, and each per-feed driving-point
        # current is I_i = v_i^T coeffs by reciprocity of the Galerkin
        # inner product (V=1 source at port i gives v_i; the current
        # sampled at port j by another source is then v_j^T · solve).
        v, port_vectors, _vpf_T, all_voltages, kcl_con = self._feed_drive_and_readout(
            geom, wire_knots, wire_basis_global, n_basis_total, kcl_A
        )
        n_ports_total = len(port_vectors)

        def _per_feed_z(coeffs_full):
            """Drive-point impedance per port (gap feeds, then junction
            ports). coeffs_full may include the enrichment block; every port
            vector is zero on that block by convention (gap feeds are not on
            enriched segments; singular enrichment bases vanish AT junctions),
            so the inner product naturally restricts to the polynomial block.
            """
            currents = np.array(
                [u_i @ coeffs_full[: u_i.shape[0]] for u_i in port_vectors],
                dtype=np.complex128,
            )
            z_per = all_voltages / currents
            return z_per[0] if n_ports_total == 1 else z_per

        # Clear any leftover per-junction selection from a prior solve
        # (variant="auto" repopulates this below; everything else leaves
        # it as None so the standard "all qualifying junctions" path
        # runs in _enrichment_specs).
        self._auto_active_junctions = None

        active_junctions = None
        if self.use_singular_enrichment and self.enrichment_variant == "auto":
            # Pass 1: solve raw (no enrichment) to read tap_ratio at each
            # K≥enrichment_min_k junction. Below the threshold ⇒ dominant-
            # pair geometry, enrichment would absorb spurious polynomial-
            # discretization error — skip. Above ⇒ genuinely balanced
            # K-way split, enrichment captures real cusp physics — keep.
            self._checkpoint()  # before the enrichment pass-1 probe solve
            coeffs_p1 = self._solve_with_kcl(Z, v, kcl_con)
            ratios = self._junction_tap_ratios(coeffs_p1)
            active_junctions = [
                j
                for j, r in enumerate(ratios)
                if r is not None and r > self.auto_tap_ratio_threshold
            ]
            self._auto_active_junctions = active_junctions
            if not active_junctions:
                # No junction qualifies → pass-1 result is the final answer.
                return _per_feed_z(coeffs_p1), coeffs_p1

        self._checkpoint()  # between passes: before pass-2 enrichment / final solve
        if self.use_singular_enrichment:
            enrich = self._enrichment_Z_assemble(
                geom, supp_seg, polys, active_junction_indices=active_junctions
            )
            if enrich is not None:
                n_p = n_basis_total
                n_e = enrich["n_enrich"]
                n_total = n_p + n_e
                Z_aug = np.zeros((n_total, n_total), dtype=np.complex128)
                Z_aug[:n_p, :n_p] = Z
                Z_aug[:n_p, n_p:] = enrich["Z_pe"]
                Z_aug[n_p:, :n_p] = enrich["Z_ep"]
                Z_aug[n_p:, n_p:] = enrich["Z_ee"]
                if self.enrichment_variant == "tikhonov":
                    # λ·s·I on the enrichment-block diagonal. s is the
                    # mean diagonal magnitude of Z_ee so λ is a
                    # dimensionless knob independent of problem scale.
                    # If Z_ee is empty (n_e=0) this block is skipped above.
                    s = float(np.mean(np.abs(np.diag(enrich["Z_ee"]))))
                    Z_aug[n_p:, n_p:] += self.tikhonov_lambda * s * np.eye(n_e)
                v_aug = np.zeros(n_total, dtype=np.complex128)
                v_aug[:n_p] = v
                # Enrichment KCL: singular bases vanish at junction → 0 outflow
                kcl_aug = np.zeros((kcl_con.shape[0], n_total), dtype=np.float64)
                kcl_aug[:, :n_p] = kcl_con
                # Z_aug is dead after the solve (and the unused `self.z`
                # stash was dropped — it retained the full n_basis² matrix
                # for nothing), so let LAPACK factor in place.
                coeffs = self._solve_with_kcl(Z_aug, v_aug, kcl_aug, overwrite=True)
                return _per_feed_z(coeffs), coeffs

        coeffs = self._solve_with_kcl(Z, v, kcl_con, overwrite=True)
        return _per_feed_z(coeffs), coeffs

    def compute_y_matrix(self) -> np.ndarray:
        """Short-circuit admittance matrix [Y_sc] at the configured feeds.

        Y_sc[i, j] is the current flowing out of port i when port j is
        driven with V_j = 1 and every other port is held at V_k = 0; the
        caller can invert it to recover the open-circuit Z matrix used in
        network analysis. BSpline's per-feed driving-point current uses the Galerkin
        reciprocity I_i = v_i^T · coeffs (inner product of feed i's
        source vector with the solution). Stacking the N source
        vectors as RHS columns and back-substituting once gives
        Y[i, j] = v_i^T · solve(Z, v_j) in one shot.

        Junctions are handled through the matrix-RHS Schur solve in
        `_solve_with_kcl_ports` — KCL is enforced once across all n_p
        source columns, so the augmentation cost stays O(n_c²) per Y
        rather than scaling with the port count.

        Singular enrichment (issue #165) composes exactly as in
        `compute_impedance`: the augmented system [[Z, Z_pe], [Z_ep, Z_ee]]
        with zero RHS and zero KCL rows on the enrichment block, and the
        readout restricted to the polynomial block (the source vectors are
        zero on the enrichment dofs), so for a single feed 1/Y[0,0] equals
        `compute_impedance`'s Z identically. The "auto" variant needs ONE
        consistent operator across all port columns for Y to be symmetric
        and self-consistent, so its pass-1 activates the UNION of junctions
        whose tap_ratio exceeds the threshold under ANY port drive (a
        single-feed solver therefore selects the same set as
        `compute_impedance`).

        This is the `y` field of `compute_port_solution()` and nothing else —
        see there for the per-port solution columns this throws away (#232).
        """
        return self.compute_port_solution().y

    def compute_port_solution(self, same_edge_prep=None) -> PortSolution:
        """Solve every port from ONE fill and ONE factorisation.

        Returns a `PortSolution` whose `y` is identical to
        `compute_y_matrix()` and whose `coeffs` column j is the B-spline
        amplitude vector for a 1 V drive at port j with every other port
        shorted; any other excitation is `coeffs @ V` with no second fill.
        Ports run [gap feeds…, junction ports…]. Lagrange multipliers are
        already eliminated — the columns satisfy the junction KCL constraints,
        they do not carry the multipliers.

        The port algebra stays inside: the Galerkin source vector per gap
        feed, the #172 split that turns a ported junction's KCL row into a
        drive/readout column, and the Schur solve that enforces the remaining
        constraints once across all port columns. Ground models, wire loading
        and singular enrichment ride exactly as for `compute_y_matrix`.

        With singular enrichment active, `coeffs` carries the enrichment
        amplitudes after the polynomial block, so it is longer than the
        polynomial basis; `basis` records where the split is. Readout is
        unaffected — port vectors vanish on the enrichment dofs.

        `basis` is an opaque handle, stable across the ports of this one
        solution and NOT across solves.

        `same_edge_prep` is the sweep's hoisted same-edge reg-moment block for
        this k (see `_same_edge_prep`); it changes nothing about the answer,
        it just spares the per-k rebuild when `_port_solutions_swept` drives
        this method frequency by frequency.
        """
        geom = self._build_geometry()
        supp_seg, polys, kcl_A, wire_knots, wire_basis_global = (
            self._build_basis_polynomials(geom)
        )
        n_basis_total = supp_seg.shape[0]

        Z = self._compute_Z_operator(
            geom, supp_seg, polys, same_edge_prep=same_edge_prep
        )

        # Junction ports (issue #172): their KCL rows become port-vector
        # columns after the gap feeds; the constraint set shrinks to match.
        # Reciprocity holds for them exactly as for gap feeds (drive vector
        # == readout vector), so the mixed Y stays symmetric.
        B, kcl_con = self._port_columns(
            geom, wire_knots, wire_basis_global, n_basis_total, kcl_A
        )
        n_ports = B.shape[1]

        def _solution(Y, X):
            """Wrap a (Y, columns) pair without touching either — the Y
            expressions below are the ones `compute_y_matrix` has always
            evaluated, spelled identically so no reduction order moves."""
            return PortSolution(
                y=Y,
                coeffs=X,
                port_currents=Y,  # the same object: the readout IS the Y matrix
                basis=_SplineBasis(
                    geom=geom,
                    supp_seg=supp_seg,
                    polys=polys,
                    wire_knots=wire_knots,
                    wire_basis_global=wire_basis_global,
                    n_poly=n_basis_total,
                ),
            )

        # Clear any leftover per-junction selection from a prior solve
        # (mirrors compute_impedance; "auto" repopulates it below).
        self._auto_active_junctions = None

        active_junctions = None
        if self.use_singular_enrichment and self.enrichment_variant == "auto":
            # Pass 1: all port columns against the un-enriched operator.
            # A junction activates when its tap_ratio exceeds the threshold
            # under ANY port drive — the union keeps one operator for every
            # column, so Y stays symmetric and internally consistent.
            X1 = self._solve_with_kcl_ports(Z, B, kcl_con)
            active = set()
            for j in range(n_ports):
                ratios = self._junction_tap_ratios(X1[:, j])
                active |= {
                    i
                    for i, r in enumerate(ratios)
                    if r is not None and r > self.auto_tap_ratio_threshold
                }
            active_junctions = sorted(active)
            self._auto_active_junctions = active_junctions
            if not active_junctions:
                return _solution(B.T @ X1, X1)  # pass-1 result is final

        if self.use_singular_enrichment:
            enrich = self._enrichment_Z_assemble(
                geom, supp_seg, polys, active_junction_indices=active_junctions
            )
            if enrich is not None:
                n_p = n_basis_total
                n_e = enrich["n_enrich"]
                n_total = n_p + n_e
                Z_aug = np.zeros((n_total, n_total), dtype=np.complex128)
                Z_aug[:n_p, :n_p] = Z
                Z_aug[:n_p, n_p:] = enrich["Z_pe"]
                Z_aug[n_p:, :n_p] = enrich["Z_ep"]
                Z_aug[n_p:, n_p:] = enrich["Z_ee"]
                if self.enrichment_variant == "tikhonov":
                    s = float(np.mean(np.abs(np.diag(enrich["Z_ee"]))))
                    Z_aug[n_p:, n_p:] += self.tikhonov_lambda * s * np.eye(n_e)
                B_aug = np.zeros((n_total, n_ports), dtype=np.complex128)
                B_aug[:n_p, :] = B
                # Enrichment KCL: singular bases vanish at the junction →
                # zero outflow columns, same padding as compute_impedance.
                kcl_aug = np.zeros((kcl_con.shape[0], n_total), dtype=np.float64)
                kcl_aug[:, :n_p] = kcl_con
                X = self._solve_with_kcl_ports(Z_aug, B_aug, kcl_aug, overwrite=True)
                # Readout restricted to the polynomial block (source
                # vectors are zero on the enrichment dofs); the returned
                # columns keep the enrichment block, which is part of the
                # solved current — `basis.n_poly` marks the split.
                return _solution(B.T @ X[:n_p, :], X)

        X = self._solve_with_kcl_ports(Z, B, kcl_con, overwrite=True)
        return _solution(B.T @ X, X)  # Y[i, j] = v_i^T · solve(Z, v_j)

    def _port_solutions_swept(self, k_array):
        """Per-k `PortSolution` generator behind `compute_y_matrix_swept` and
        `compute_port_solution_swept` (#252).

        Three routes, all producing the same per-k answer:

        * singular enrichment (issue #165) — a bare per-k
          `compute_port_solution` loop; the batched C++ assembly has no
          augmented-system variant, and the same-edge hoist is skipped so the
          enrichment sweep stays byte-for-byte what it was;
        * the fully batched fast path (batched assembly + one k- and
          port-batched KCL Schur solve per chunk) when
          `_swept_batched_available`. This is the one route whose per-k core
          is NOT `compute_port_solution` — thinning it would dissolve the
          chunking that bounds `swept_mem_mb` — so it re-spells only the
          readout, `Y = Bᵀ X`, off the shared `_port_columns`;
        * otherwise a per-k `compute_port_solution` with the sweep's hoisted
          same-edge reg moments. Routing through `compute_port_solution` puts
          the fallback on `_compute_Z_operator`, so it honours the
          `swept_mem_mb` dispatch instead of always materialising the full
          (d+1, d+1, N, N) moment tensor (issue #238).
        """
        k_array = np.asarray(k_array, dtype=float)
        if self.use_singular_enrichment:
            with self._k_restored():
                for kk in k_array:
                    self._checkpoint()  # top of each frequency iteration
                    self._set_k(kk)
                    yield self.compute_port_solution()
            return

        geom = self._build_geometry()
        supp_seg, polys, kcl_A, wire_knots, wire_basis_global = (
            self._build_basis_polynomials(geom)
        )
        n_basis_total = supp_seg.shape[0]

        # Port columns (gap feeds then junction ports, issue #172) are
        # k-independent, so the whole sweep shares one build.
        B, kcl_con = self._port_columns(
            geom, wire_knots, wire_basis_global, n_basis_total, kcl_A
        )

        def _solution(Y, X):
            return PortSolution(
                y=Y,
                coeffs=X,
                port_currents=Y,  # the same object: the readout IS the Y matrix
                basis=_SplineBasis(
                    geom=geom,
                    supp_seg=supp_seg,
                    polys=polys,
                    wire_knots=wire_knots,
                    wire_basis_global=wire_basis_global,
                    n_poly=n_basis_total,
                ),
            )

        if self._swept_batched_available():
            for _c0, ks, Z in self._swept_batched_z_chunks(
                k_array, geom, supp_seg, polys
            ):
                X = self._solve_with_kcl_swept_ports(Z, B, kcl_con)
                del Z  # the chunk's Z stack is dead; let it go before the yields
                for i in range(ks.shape[0]):
                    # Spelled exactly as compute_port_solution's readout, so
                    # the batched path differs from a per-k solve only by the
                    # batched LAPACK/assembly reassociation, never by algebra.
                    yield _solution(B.T @ X[i], X[i])
            return

        # k-independent static + reg-geometry, shared across the sweep; the
        # reg-kernel moment blocks are batched over chunks of k sized to the
        # `swept_mem_mb` budget (one einsum per (edge, chunk) instead of one
        # per (edge, k); whole-sweep hoisting was issue #263).
        prep = self._same_edge_prep(geom)
        with self._k_restored():
            for _ki, kk, same_edge_k in self._same_edge_prep_swept_chunks(
                prep, k_array
            ):
                self._checkpoint()  # top of each frequency iteration
                self._set_k(kk)
                yield self.compute_port_solution(same_edge_prep=same_edge_k)

    def _swept_batched_available(self):
        """True when the fully batched swept fast path can serve this
        instance: batched C++ kernels present, degree instantiated, no
        singular enrichment (two-pass / augmented system), and no finite
        ground (per-k ε̃(ω) weight tables / sommerfeld grids stay on the
        per-k loop). Junctions ARE supported — bspline's assembly is
        already general (directional bases live in supp_seg / polys), so
        only the KCL constraint needs batching, via the Schur solves
        `_solve_with_kcl_batch` / `_solve_with_kcl_swept_ports`.
        """
        return (
            not self.use_singular_enrichment
            and self.ground_eps is None
            and _HAVE_BSPLINE_SWEPT_ASSEMBLE_ACCEL
            and _HAVE_BSPLINE_OFFEDGE_SWEPT_ACCEL
            and self.degree <= _BSPLINE_ASSEMBLE_ACCEL_MAX_D
        )

    def _swept_batched_z_chunks(self, k_array, geom, supp_seg, polys):
        """Yield (c0, ks, Z) stacks of the batched swept assembly, where
        Z is (len(ks), n_basis, n_basis) for k_array[c0 : c0 + len(ks)].

        The k axis is chunked so the transient moment tensors stay under
        the `swept_mem_mb` constructor budget (default 256 MB). The
        budget counts the per-k transients that actually scale with the
        sweep — the all-pairs J tensor (chunk, nm, nm, N, N) AND the
        same-edge reg moment slices (chunk, nm, nm, N_e, N_e per edge),
        both computed per chunk — so peak transient memory ≈ the budget,
        and a memory-constrained deployment can cap it per solve (e.g.
        64–96 on a 2 GB host with concurrent users).

        The tradeoff (measured, single 400-seg dipole, d=2, 41-pt sweep):
        the batched kernels amortize their per-pair R-table hoists across
        the chunk's k axis, so tiny chunks re-derive geometry per k —
        chunk=1 costs ~+75% wall-clock; the win saturates by chunk ≈ 8-16.
        Budgets below ~64 MB buy little memory and cost real time.
        """
        d = self.degree
        n_k = k_array.shape[0]
        seg_l, seg_r = geom["seg_l"], geom["seg_r"]
        tangents = geom["tangents"]
        N = seg_l.shape[0]
        nm = d + 1

        # k-independent: same-edge reg geometry, tangent-dot tables,
        # image segments — all built once for the whole sweep.
        prep = self._same_edge_prep(geom)
        td_free = tangents @ tangents.T
        # Same specs as the per-k fills; the same-edge half already rides
        # `prep` (its static blocks and reg geometry carry `_EK_SAME_EDGE`).
        # Under EK the batched offedge kernels reach their own C++ twin
        # (momwire#270 unit 2); only a build without it stacks per-k calls.
        ek = self._ek_spec(geom) if self.extended_kernel else None
        ek_img = None
        if self.ground_z is not None:
            td_img = self._image_tangent_dot(tangents)
            seg_l_img = self._image_positions(seg_l)
            seg_r_img = self._image_positions(seg_r)
            if self.extended_kernel:
                ek_img = self._ek_spec(geom, mirror=True)

        # Chunk size from the memory budget. Per k, the transients are
        # the all-pairs J tensor (nm² N² complex) plus the per-edge
        # same-edge reg moment blocks (nm² ΣN_e² complex); the PEC image
        # J reuses J's footprint (J is dropped before the image build).
        sum_ne2 = sum((sl.stop - sl.start) ** 2 for sl, _A_st, _g in prep)
        bytes_per_k = nm * nm * (N * N + sum_ne2) * 16
        chunk = max(1, min(n_k, (self.swept_mem_mb << 20) // max(bytes_per_k, 1)))

        def _assemble_swept(J_tensor, td, omega_chunk):
            return _acc.assemble_Z_bspline_swept(
                np.ascontiguousarray(J_tensor, dtype=np.complex128),
                np.ascontiguousarray(supp_seg, dtype=np.int64),
                np.ascontiguousarray(polys, dtype=np.float64),
                np.ascontiguousarray(td, dtype=np.float64),
                np.ascontiguousarray(omega_chunk, dtype=np.float64),
                float(self.eps),
                float(self.mu),
                int(d),
            )

        for c0 in range(0, n_k, chunk):
            self._checkpoint()  # top of each k-chunk
            ks = k_array[c0 : c0 + chunk]
            omega_chunk = ks * self.c
            J = _seg_seg_full_moments_offedge_swept(
                seg_l,
                seg_r,
                seg_l,
                seg_r,
                self._seg_radius(geom),
                ks,
                d,
                self.n_qp_pair,
                ek=ek,
            )
            # Same-edge reg moments for this chunk. Computed per chunk —
            # the streaming kernel amortizes its R hoist over the chunk's
            # k axis, which captures nearly all of the full-sweep hoist's
            # win once chunk ≳ 8 while keeping the allocation inside the
            # memory budget (a full-sweep hoist is O(n_k·nm²·ΣN_e²) —
            # ~1 GB on a 41-pt sweep of a single 400-seg wire).
            for sl, A_st, reg_geo in prep:
                J[:, :, :, sl, sl] = A_st[
                    None
                ] + _seg_seg_reg_moments_from_geometry_swept(reg_geo, ks)
            Z = _assemble_swept(J, td_free, omega_chunk)
            if self.ground_z is not None:
                del J  # let the image tensor reuse J's footprint
                J_img = _seg_seg_full_moments_offedge_swept(
                    seg_l,
                    seg_r,
                    seg_l_img,
                    seg_r_img,
                    self._seg_radius(geom),
                    ks,
                    d,
                    self.n_qp_pair,
                    ek=ek_img,
                )
                Z = Z - _assemble_swept(J_img, td_img, omega_chunk)
            # Loading is Z'(ω)-scaled per k within the chunk (skin R ∝ √ω,
            # insulation X ∝ ω), added after the batched kernel assembly.
            Z = self._apply_loading(Z, omega=omega_chunk)
            yield c0, ks, Z

    def _compute_impedance_swept_batched(self, k_array):
        """Fully batched swept solve: the whole sweep's
        J and Z built in batched C++ calls, one stacked LAPACK solve per
        k-chunk instead of looping compute_impedance per frequency.
        Junctions ride the batched KCL Schur solve.

        The drive and readout algebra is `compute_impedance`'s own
        `_feed_drive_and_readout` — this method owns the batching, not a
        second copy of the port bookkeeping (#252).
        """
        n_k = k_array.shape[0]

        geom = self._build_geometry()
        supp_seg, polys, kcl_A, wire_knots, wire_basis_global = (
            self._build_basis_polynomials(geom)
        )
        n_basis_total = supp_seg.shape[0]

        # Junction ports (issue #172): k-independent drive and readout rows,
        # exactly like the gap source vectors, so they batch for free.
        v, port_vectors, vpf_T, all_voltages, kcl_con = self._feed_drive_and_readout(
            geom, wire_knots, wire_basis_global, n_basis_total, kcl_A
        )
        n_total = len(port_vectors)

        z_out = (
            np.zeros(n_k, dtype=np.complex128)
            if n_total == 1
            else np.zeros((n_k, n_total), dtype=np.complex128)
        )
        for c0, ks, Z in self._swept_batched_z_chunks(k_array, geom, supp_seg, polys):
            coeffs = self._solve_with_kcl_batch(Z, v, kcl_con)
            currents = coeffs @ vpf_T  # (chunk, n_total)
            z_per = all_voltages[None, :] / currents
            z_out[c0 : c0 + ks.shape[0]] = z_per[:, 0] if n_total == 1 else z_per

        return z_out

    def compute_impedance_swept(self, k_array):
        """Driver impedance over a batch of wavenumbers.

        Fully batched fast path (batched C++ J/Z assembly + stacked
        LAPACK solve, junctions via the batched KCL Schur) when
        `_swept_batched_available`; otherwise a per-k loop that rebinds
        self.k / self.omega / self.wavelength per call and restores them
        (enrichment and finite grounds live here).

        Both routes read their drive/readout algebra off
        `_feed_drive_and_readout`, the same helper `compute_impedance` uses,
        so there is no swept copy of it to drift (#252). This entry point
        stays on `compute_impedance`'s single-excitation solve rather than on
        the port columns: at one RHS instead of n_ports it is the cheaper
        solve, and it keeps the swept answer bit-comparable with the per-k
        `compute_impedance` it mirrors.
        """
        k_array = np.asarray(k_array, dtype=float)
        n_total = len(self.feeds) + len(self.junction_ports)
        if n_total == 1:
            z_out = np.zeros(k_array.shape[0], dtype=np.complex128)
        else:
            z_out = np.zeros((k_array.shape[0], n_total), dtype=np.complex128)

        # Fully batched fast path: build the whole sweep's
        # J / Z / solve in batched calls instead of looping compute_impedance
        # per frequency. Junctions included; see _swept_batched_available
        # for the (enrichment / finite-ground / accel) eligibility rules.
        if self._swept_batched_available():
            return self._compute_impedance_swept_batched(k_array)

        # k-independent static + reg-geometry, shared across the sweep; the
        # reg-kernel moment blocks are batched over chunks of k sized to the
        # `swept_mem_mb` budget (one einsum per (edge, chunk) instead of one
        # per (edge, k); whole-sweep hoisting was issue #263).
        prep = self._same_edge_prep(self._build_geometry())
        with self._k_restored():
            for i, kk, same_edge_k in self._same_edge_prep_swept_chunks(prep, k_array):
                self._checkpoint()  # top of each frequency iteration
                self._set_k(kk)
                z, _ = self.compute_impedance(same_edge_prep=same_edge_k)
                z_out[i] = z
        return z_out

    def currents_at_knots(self, coeffs, s_array=None):
        """Per-wire complex current at every mesh knot.

        Evaluates Σ_kept c_g · B_{j_local}(s_knot) per wire using scipy's
        B-spline design matrix on the wire's clamped knot vector.

        When `s_array` is provided as a list of 1D arc-length arrays (one per
        wire), the basis sum is evaluated at those arc positions instead of
        the mesh knots. With `use_singular_enrichment=True`, the enrichment
        basis Φ_sing(u) = (u/h)·log(u/h) — non-zero between knots but exactly
        zero AT the bounding knots — is added at sample positions interior to
        the enriched segments. Φ_sing contributes nothing at mesh knots, so
        the s_array=None path is unchanged.

        KCL Lagrange multipliers (trailing entries beyond the polynomial
        and enrichment blocks of `coeffs`) carry no current shape and are
        ignored by this evaluation.
        """
        coeffs = np.asarray(coeffs)
        geom = self._build_geometry()
        supp_seg, _, _, wire_knots, wire_basis_global = self._build_basis_polynomials(
            geom
        )
        n_poly = supp_seg.shape[0]
        d = self.degree

        enrich_specs = None
        if self.use_singular_enrichment:
            # Match the active-junction subset that compute_impedance used
            # so the spec list lines up with the enrichment block of coeffs.
            specs = self._enrichment_specs(
                geom, active_junction_indices=self._auto_active_junctions
            )
            if specs:
                enrich_specs = specs

        out = []
        for w_idx in range(len(self.wires_polylines)):
            arc_at_knot = geom["per_wire"][w_idx]["arc_at_knot"]
            knots_vec = wire_knots[w_idx]
            if s_array is None:
                s_eval = np.clip(arc_at_knot, knots_vec[0], knots_vec[-1])
            else:
                s_eval = np.clip(
                    np.asarray(s_array[w_idx], dtype=np.float64),
                    knots_vec[0],
                    knots_vec[-1],
                )
            I_out = np.zeros(s_eval.shape[0], dtype=np.complex128)
            kept, local_to_global = wire_basis_global[w_idx]
            if s_eval.shape[0] > 0:
                # design_matrix at [0, wire_arc] — clip tiny FP overshoots that
                # would push the endpoint epsilon outside the clamped knot range.
                DM = BSpline.design_matrix(s_eval, knots_vec, d)
                # Scatter the kept coefficients onto their bases (dropped
                # free-end bases keep their 0) and sum each row over its
                # own d+1 stored entries. Those entries are in ascending
                # column order, so the per-row accumulation order matches
                # the ascending-basis loop this replaces term for term —
                # the terms it drops are the exact zeros off the band.
                c_basis = np.zeros(len(knots_vec) - d - 1, dtype=np.complex128)
                for kept_idx, (j_local, _, _, _) in enumerate(kept):
                    c_basis[j_local] = coeffs[local_to_global[kept_idx]]
                dm_vals, dm_cols = _design_matrix_rows(DM, d)
                for e in range(d + 1):
                    I_out += c_basis[dm_cols[:, e]] * dm_vals[:, e]

            if enrich_specs is not None:
                seg_off_w = geom["seg_offsets"][w_idx]
                arc_at_knot_w = geom["per_wire"][w_idx]["arc_at_knot"]
                h_per_seg_w = geom["per_wire"][w_idx]["h_per_seg"]
                for spec_idx, (_, wire_w, _, seg_idx_global, u_origin) in enumerate(
                    enrich_specs
                ):
                    if wire_w != w_idx:
                        continue
                    seg_local = seg_idx_global - seg_off_w
                    seg_l_arc = arc_at_knot_w[seg_local]
                    seg_r_arc = arc_at_knot_w[seg_local + 1]
                    h_seg = h_per_seg_w[seg_local]
                    mask = (s_eval >= seg_l_arc) & (s_eval <= seg_r_arc)
                    if not np.any(mask):
                        continue
                    if u_origin == "left":
                        u_from_junc = s_eval[mask] - seg_l_arc
                    else:
                        u_from_junc = seg_r_arc - s_eval[mask]
                    u_norm = u_from_junc / h_seg
                    # Match the solver's variant: "raw" subtracts nothing,
                    # "stable" subtracts the bubble-subspace projection.
                    # Both variants preserve Φ_sing(0)=Φ_sing(1)=0.
                    phi = np.zeros_like(u_norm)
                    pos = u_norm > 0.0
                    phi[pos] = u_norm[pos] * np.log(u_norm[pos])
                    if self.enrichment_variant == "stable":
                        proj_coeffs = _xfem_projection_coeffs(self.degree)
                        phi = phi - np.polyval(proj_coeffs[::-1], u_norm)
                    I_out[mask] += coeffs[n_poly + spec_idx] * phi
            out.append(I_out)
        return out
