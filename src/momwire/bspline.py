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

import math
from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse
from scipy.interpolate import BSpline

from ._bspline_kernels import (
    _EK,
    _HAVE_BSPLINE_OFFEDGE_SWEPT_ACCEL,
    _ek_axis_groups,
    _normalize_ladder,
    _refuse_complex_k,
    _seg_seg_full_moments_offedge,
    _seg_seg_full_moments_offedge_swept,
    _seg_seg_reg_geometry,
    _seg_seg_reg_moments,
    _seg_seg_reg_moments_from_geometry,
    _seg_seg_reg_moments_from_geometry_swept,
    _seg_seg_static_moments,
)
from ._quadrature import leggauss

from . import _bspline_kernels
from . import _crossing_fill
from . import _ground_mirror
from . import _ground_refl
from . import _ground_spec
from . import _medium_spec
from . import _potential_ground
from . import _quadrature
from . import _sommerfeld
from . import _sommerfeld_below
from . import _sommerfeld_transmitted
from . import _wire_loading
from . import _wire_spec
from ._accel import acc as _acc
from ._accel import MAX_N_QP as _ACCEL_MAX_N_QP
from . import _surface_height
from ._cancel import _Cancelable
from ._capabilities import Capabilities
from ._element_currents import _ElementCurrents
from ._port_solution import PortSolution, _SweptPortSolutions

# Cross-edge quadrature defaults, resolved per deck by `BSplineSolver.n_qp_pair`.
# Two constants because there are two fills; see that property for the
# measurements and for why the buried order is not applied everywhere.
DEFAULT_N_QP_PAIR = 8
BURIED_N_QP_PAIR = 32

# The distance-adaptive pair-order LADDER (momwire#906), resolved per deck by
# `BSplineSolver.pair_order_ladder` alongside `n_qp_pair`: a pair whose centre
# distance is >= 2 segment lengths takes order 8, >= 16 lengths order 4, the
# rest the base order. Measured on the 654-segment buried screen: every
# ladder tried reproduces uniform-32 Z to the printed digit while ~87 % of
# the pairs run at order 4. Free space keeps NO ladder for now — not because
# the thresholds differ there (the study says they do not) but because that
# fill's pinned constants have not been re-run under one; flipping it is its
# own measured change.
DEFAULT_PAIR_ORDER_LADDER = ()
BURIED_PAIR_ORDER_LADDER = ((2.0, 8), (16.0, 4))

_HAVE_BSPLINE_ASSEMBLE_ACCEL = _acc is not None and hasattr(_acc, "assemble_Z_bspline")
_HAVE_BSPLINE_ASSEMBLE_W_ACCEL = _acc is not None and hasattr(
    _acc, "assemble_Z_bspline_weighted"
)
_HAVE_ENRICH_ACCEL = _acc is not None and hasattr(_acc, "assemble_Z_enrich")
# momwire#910: the in-medium (complex eps~) twins of the two assemblers. The
# buried fill used to take the numpy einsum loop for both of its assemblies
# because the C++ entries' `double eps` would truncate eps~.
_HAVE_BSPLINE_ASSEMBLE_CPLX_EPS_ACCEL = _acc is not None and hasattr(
    _acc, "assemble_Z_bspline_cplx_eps"
)
_HAVE_BSPLINE_ASSEMBLE_W_CPLX_EPS_ACCEL = _acc is not None and hasattr(
    _acc, "assemble_Z_bspline_weighted_cplx_eps"
)
# momwire#915: the complex-eps~ twins of the two WINDOWED assemblers, which
# are what let a buried deck take the chunked fill+assemble route instead of
# refusing when its dense tensor does not fit the budget.
_HAVE_BSPLINE_WINDOWED_CPLX_EPS_ACCEL = _acc is not None and hasattr(
    _acc, "assemble_Z_bspline_windowed_cplx_eps"
)
_HAVE_BSPLINE_W_WINDOWED_CPLX_EPS_ACCEL = _acc is not None and hasattr(
    _acc, "assemble_Z_bspline_weighted_windowed_cplx_eps"
)
# momwire#914 unit 1: the below/below plan extents in C++. Gated on the #914
# capability flag AND the symbol — the flag alone would be satisfied by a .so
# whose binding moved, and the symbol alone by a build predating the contract.
_HAVE_PLAN_EXTENTS_ACCEL = (
    _acc is not None
    and getattr(_acc, "plan_extents_914", False)
    and hasattr(_acc, "pair_extents_below")
)
_HAVE_FIELD_GALERKIN_ACCEL = (
    _acc is not None
    and getattr(_acc, "field_galerkin_914", False)
    and hasattr(_acc, "assemble_field_galerkin")
)
# Which of the accelerator's two routes to the same numbers to take. The
# fused one skips `Jc` entirely and is what production wants (measured
# momwire#914: 0.70 s against 1.47 s on the 48-radial screen). The literal
# transcription stays REACHABLE rather than being a `fused=True` the C++
# hard-codes, for two reasons: an unreachable branch in the TU is untested
# code that still ships, and a second independent route is the strongest
# available cross-check on the fused one's index arithmetic. Tests flip it;
# nothing else should.
_FIELD_GALERKIN_FUSED = True
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

# momwire#631. A horizontal edge's own PEC image is the SAME arc translated
# by -2h in z, so its moment block is the same-edge kernel at an effective
# radius sqrt(a^2 + 4h^2) — exactly, with no quadrature order to key. Off-edge
# quadrature at `n_qp_pair` only agrees with that closed form while the image
# stays far compared with a segment; measured against a 256-point reference on
# a 39.6 m radial, max relative error over the block:
#
#   delta/2h     0.76      7.57      69.4
#   n_qp_pair=4  3.8e-06   4.2e-02   1.6e+00
#   closed form  7.5e-09   1.3e-06   5.0e-06
#
# so the analytic block is better everywhere and the off-edge rule collapses
# above delta/2h ~ 1. (The measured row is at n_qp_pair=4, the default when
# it was taken; momwire#743 moved that default to 8, which tightens the
# off-edge row without touching the conclusion — the collapse above
# delta/2h ~ 1 is the rule losing its premise, not losing quadrature order.) The threshold is set an octave below where the shipped
# rule starts to lose figures, which keeps every non-grazing deck on exactly
# the arithmetic it had (the two agree to ~1e-6 at the crossover, far below
# any tolerance the suite pins) while catching the grazing regime whole.
_NEAR_IMAGE_DELTA_OVER_2H = 0.5

# momwire#631, the remainder half. Same rule and same constants razor took in
# momwire#510 — `_quadrature.remainder_qp` owns the arithmetic; these are
# bspline's own clip, so patching one trunk's cap never moves the other's.
_REMAINDER_QP_CAP = 192
_REMAINDER_QP_C = 1.0


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


# momwire#396: three `use_singular_enrichment` combination refusals, reused
# by their `__init__` raises below and by `capabilities.refusals` — one
# message per combination, not a copy in each.
_ENRICHMENT_WIRE_LOADING_REFUSAL = (
    "use_singular_enrichment + distributed wire loading together "
    "not supported yet — the enrichment bases don't carry the "
    "loading overlap term"
)
_ENRICHMENT_EXTENDED_KERNEL_REFUSAL = (
    "extended_kernel=True + use_singular_enrichment=True not "
    "supported yet — the enrichment DOFs bypass the moment "
    "kernels entirely (they carry their own Φ_sing "
    "quadrature), they exist only at K >= 3 junctions where "
    "NEC's own gating turns EK off, and the O(a²) tube "
    "expansion was never derived for the s^(-1/2) shapes "
    "(stevenmburns/momwire#249 follow-up C)"
)
# The Gauss order the buried fill's field-form blocks run at — see
# `BSplineSolver._n_qp_buried_field` for the measurement that sets it.
_N_QP_BURIED_FIELD = 6

_ENRICHMENT_PER_WIRE_RADIUS_REFUSAL = (
    "use_singular_enrichment + mixed per-wire radii together "
    "not supported yet — the enrichment kernels take a single "
    "radius (stevenmburns/momwire#147)"
)

# The sentence the OTHER families' `singular_enrichment` cell reads
# (momwire#792). It lives here because the enrichment is this
# family's — the same reason `_ground_spec` owns the one contact-under-
# refl-coef sentence the whole tree quotes — and it is imported by
# `razor` and `sinusoidal` (and through the latter by
# `sinusoidal_galerkin`) rather than copied into each.
#
# NOT YET, not never, and the distinction is the point of recording it: the
# enrichment is a junction basis written against THIS family's knot vector
# and its Galerkin testing, and what the same dof looks like under another
# formulation's testing is an open design question (momwire#445), not a
# decision any of those rows has taken. `{cls}` is substituted with the
# declaring class (momwire#564) — these sentences reach a user verbatim
# through `capabilities.refusals`.
SINGULAR_ENRICHMENT_NEVER = (
    "singular enrichment is not built for {cls}, and will not be: the enrichment "
    "in tree (`use_singular_enrichment`) is the B-spline family's junction basis — "
    "an extra dof carrying the s^(-1/2) edge shape, written against that family's "
    "knot vector and integrated by its Galerkin testing. It is kept as a "
    "B-spline-only EXPERIMENTAL feature (maintainer decision, momwire#445, "
    "2026-09-02): it has not yet bought anything measurable, so it is not "
    "extended to any other formulation and may be removed altogether later. "
    "This cell is a NEVER, not a not-yet. There is no `use_singular_enrichment` "
    "keyword on this class at all, so asking for it is a caller typo (a "
    "TypeError) rather than this sentence"
)

# momwire#553 U5: what a BURIED deck may not reach, and the domain a buried
# deck may not leave. Every one of these names the geometry AND the limit in
# the same sentence, because the two buried Sommerfeld families refuse rather
# than clamp — there is no negligible tail to freeze below the interface —
# and a serve-time refusal with no numbers in it is not actionable.
_BURIED_ENRICHMENT_REFUSAL = (
    "use_singular_enrichment=True + a wire below the ground plane is not "
    "served: the enrichment DOFs are a SECOND kernel implementation with "
    "their own quadrature over the s^(-1/2) shapes, in C++ with a `double k` "
    "and in a numpy twin, and neither carries an in-medium wavenumber or the "
    "buried image and remainder blocks (momwire#553 U5 widens the polynomial "
    "moment fill only). Solve the buried deck without singular enrichment"
)
_BURIED_EXTENDED_KERNEL_REFUSAL = (
    "extended_kernel=True + a wire below the ground plane is not served: the "
    "extended kernel's eligibility is a COAXIAL-AND-EQUAL-RADIUS grouping "
    "scored across the whole geometry, and momwire#553 measured neither what "
    "that grouping means for a pair spanning two media (the tube expansion's "
    "O(a^2) term is written at one wavenumber) nor what the mirror labels "
    "mean when the image of a buried source lands in the OTHER medium. "
    "Solve the buried deck with extended_kernel=False, which is the default"
)
_BURIED_DENSE_BUDGET_REFUSAL = (
    "a deck with buried wires is filled through the DENSE moment tensor on "
    "this build and this one does not fit: {n} segments need about {need:.0f} "
    "MB per tensor against a swept_mem_mb budget of {budget} MB. The chunked "
    "fill+assemble route (momwire#915) needs the windowed C++ assemblers' "
    "complex-eps twins, which this accelerator build does not export, so the "
    "fill refuses rather than silently truncating the medium. Rebuild the "
    "accelerators, raise swept_mem_mb, or mesh the deck coarser"
)
_BURIED_PAST_CAP_REFUSAL = (
    "this deck's buried wires reach a below/below pair separation of "
    "R1 = {r1:.6g} m ({wl:.3g} in-medium wavelengths), past the {cap:.6g} m "
    "({capwl:g} in-medium wavelengths, lambda_m = {lam_m:.4g} m) the "
    "below/below remainder is tabulated to. There is no honest clamp out "
    "there: unlike the reflected-wave remainder above the interface, the "
    "below/below remainder GROWS relative to the direct term with range "
    "(measured 12x and 168x direct+image at working range, momwire#553 U2), "
    "so freezing the surface amplitude would return a confident wrong "
    "number. Shrink the buried structure, or densify the far annulus of the "
    "below/below grid (a recorded follow-up, and it wants the C++ twin first)"
)
_BURIED_GRAZING_REFUSAL = (
    "this deck's buried wires reach a below/below pair elevation of "
    "theta = {th:.4g} deg, below the {floor:g} deg grazing floor the "
    "below/below surfaces are tabulated from (the pair's two depths add to "
    "{depth:.4g} m, and theta = atan2(depth sum, horizontal separation)). "
    "Below the floor the surfaces carry the lateral wave's LOGARITHMIC "
    "structure — measured drift 1.1 to 3.3 of scale between 2 and 0.05 deg — "
    "which no uniform lattice resolves, and theta = 0 has no node at all "
    "because h = 0 leaves the tail without its exponential decay. Bury the "
    "wires deeper, shorten them, or wait for the log-spaced grazing band "
    "(momwire#553 U2's recorded follow-up)"
)
_BURIED_CROSS_RANGE_REFUSAL = (
    "this deck's cross-medium pairs reach an observer radius of R = {r:.6g} m "
    "({wl:.3g} free-space wavelengths) about a buried source's ground "
    "projection, past the {cap:.6g} m ({capwl:g} free-space wavelengths) the "
    "transmitted family is tabulated to. The transmitted integral is the "
    "WHOLE field above a buried source, not a remainder, so there is no "
    "negligible tail to freeze and no honest clamp. Extending the log-R axis "
    "is cheap and honest work (7 nodes per doubling of range); extrapolating "
    "past it is not"
)
_BURIED_DEPTH_REFUSAL = (
    "this deck buries a wire {d:.6g} m down, past the {cap:.6g} m "
    "({capwl:g} in-medium wavelengths, lambda_m = {lam_m:.4g} m) the "
    "transmitted family's z' ladder reaches. Beyond a quarter lambda_m the "
    "two-ray (lambda_1/lambda_2 saddle) structure of the transmitted "
    "integral returns and the single e^(-j k_m |z'|) divide-out the whole "
    "ladder architecture rests on no longer flattens it (momwire#524 phase 0 "
    "measured >33 nodes over the deep range at every soil, and a spherical-"
    "phase divide is no better). Bury the wire shallower, or extend the "
    "ladder — about 8 extra rungs per additional quarter lambda_m, each rung "
    "a full (R, theta) fill"
)
_BURIED_CROSS_GRAZING_REFUSAL = (
    "this deck's cross-medium pairs reach an observer elevation of "
    "theta = {th:.4g} deg, below the {floor:.4g} deg this transmitted grid "
    "can pay for. That floor is a COST law rather than a physics one: the "
    "transmitted tail is panelled on the J0(lam rho) zeros and must reach "
    "lam ~ 35/(z + |z'|), so a node costs about 16*cot(theta_true) panels, "
    "and at this deck's range {r:.6g} m over a shallowest buried depth of "
    "{depth:.6g} m the bottom row runs out of budget. A truncated tail here "
    "does NOT degrade gracefully — the acceleration fallback was measured "
    "4.5e+3 relative wrong — so it refuses. Raise the above-ground wires "
    "clear of the plane, bury the wires deeper, or shrink the deck's extent"
)


def _pair_extents_below(x, y, d_b, rows=256):
    """`(r1_max, th_min)` over every node pair for the below/below plan —
    the largest image distance hypot(rho, h_i + h_j) and the shallowest
    angle atan2(h_i + h_j, rho) — without an (n, n) array ever being live.

    momwire#910: the all-pairs spelling built six (n, n) arrays over the
    3,924 buried nodes of a 12-radial screen (123 MB each) and spent 0.6 s,
    a third of it in arctan2, to find two scalars. Rows are walked in
    chunks, the squared distance is accumulated in place, and the angle is
    ONE atan at the end: on the closed quadrant hh, rho >= 0 the map
    atan2(hh, rho) is monotone in hh / rho, so the minimum angle is the
    minimum ratio. rho = 0 only where a node meets itself, where hh > 0 and
    the ratio is +inf — never the minimum, so the divide is ignored. Same
    two numbers to 1e-12 as the all-pairs form (gated).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    d_b = np.asarray(d_b, dtype=np.float64)
    if _HAVE_PLAN_EXTENTS_ACCEL and x.size:
        # momwire#914 unit 1. The C++ twin walks the upper triangle (the pair
        # matrix is exactly symmetric) and minimises hh^2/rho^2, the same
        # argmin on the non-negative quadrant, so the per-pair sqrt is gone:
        # 2.31 s -> 0.062 s over the 246 M pairs of a 48-radial screen.
        # The numpy form below stays the reference and the fallback, and is
        # what G-914-1 gates against.
        r1_max, th_min = _acc.pair_extents_below(x, y, d_b)
        return float(r1_max), float(th_min)
    r1sq_max = 0.0
    ratio_min = np.inf
    for i0 in range(0, x.shape[0], rows):
        i1 = min(i0 + rows, x.shape[0])
        rho2 = (x[i0:i1, None] - x[None, :]) ** 2
        rho2 += (y[i0:i1, None] - y[None, :]) ** 2
        hh = d_b[i0:i1, None] + d_b[None, :]
        r1sq_max = max(r1sq_max, float(np.max(rho2 + hh * hh)))
        np.sqrt(rho2, out=rho2)
        with np.errstate(divide="ignore"):
            ratio_min = min(ratio_min, float(np.min(hh / rho2)))
    return float(np.sqrt(r1sq_max)), float(np.arctan(ratio_min))


def _contiguous_runs(idx):
    """`[(start, stop), ...]` — the maximal runs of consecutive integers in
    a sorted index array (a subset that is a union of whole wires is a few
    of these)."""
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) != 1) + 1
    starts = np.concatenate(([0], cuts))
    stops = np.concatenate((cuts, [idx.size]))
    return [(int(idx[s0]), int(idx[s1 - 1]) + 1) for s0, s1 in zip(starts, stops)]


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
    n_qp_pair : Gauss-Legendre nodes per segment per axis for CROSS-EDGE and
        cross-wire pairs (full kernel with a² regularization). Default 8,
        raised from 4 in momwire#743.

        Two moment paths used to share this number and they want opposite
        things. Cross-edge order is what a split straight wire and a
        radius-step junction need, and it is free in memory because the C++
        kernel streams — `seg_seg_full_moments_bspline_kernel` allocates
        `J({NM, NM, N_i, N_j})`, with no `n_qp` in the shape. Same-edge order
        is the entire memory cost and buys nothing. Measured as a known zero
        (splitting a straight wire at a collinear anchor is geometrically a
        no-op, so `|Z_split − Z_unsplit|` has an exact answer of 0):

            N     qp=4              qp=8
            18    0.761 Ω (0.557%)  0.198 Ω (0.145%)
            30    0.450 Ω (0.329%)  0.112 Ω (0.082%)
            60    0.216 Ω (0.158%)  0.048 Ω (0.035%)

        8 is also the ceiling: the accelerated kernels carry a fixed
        `n_qp² ≤ 64` scratch buffer and raise above it, so this default sits
        exactly at the cap.
    n_qp_pair_same_edge : the same, for SAME-EDGE pairs — the smooth-kernel
        piece of a segment pair on one edge. Default 4, i.e. what
        `n_qp_pair` meant for both paths before momwire#743.

        Raising this is pure cost: `_seg_seg_reg_geometry` materialises one
        edge's `(N_e·n_qp, N_e·n_qp)` R table, quadratic in BOTH, while the
        answer does not move — `Re Z = 135.698155` bit-identical at 2, 4 and
        8 on an N=400 single edge, where peak RSS goes 162 / 177 / 353 MB.
        There is a saving available here (2 is bit-identical to 4 at N=400)
        but it also governs #631's near-image analytic block, which has not
        been measured at 2, so the default stays where it was.
    pair_order_ladder : the distance-adaptive off-edge quadrature ladder
        (momwire#906): a tuple of `(ratio, n_qp)` with ratios strictly
        ascending and orders strictly descending below `n_qp_pair`. A pair
        whose centre distance over the longer segment meets a tier's ratio
        takes that tier's order; the rest pay `n_qp_pair`. Resolved from the
        deck like `n_qp_pair` when None: buried decks get
        `BURIED_PAIR_ORDER_LADDER` = ((2, 8), (16, 4)), free-space decks get
        no ladder (see the constants for why). Tiers at or above the base
        order are dropped, so an explicit `n_qp_pair=8` under the buried
        default leaves ((16, 4),). The per-block phase guard in
        `_seg_seg_full_moments_offedge` drops the order-4 tier when the
        longest segment passes kL = 0.5.

        What it buys: on the 654-segment radial screen the two buried pair
        blocks went from 6.3 s to well under a second at the same Z, because
        87 % of the pairs sit 16+ segment lengths apart and the study
        measured order 4 there at 3e-14 relative to order 32.
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
        Peak transient memory of a sweep ≈ this budget (honest since #338:
        the chunked fills `del` each observer-row window before the next
        one is built, so the loop never holds two at once — it used to,
        which silently doubled the real transient behind this knob), so a
        memory-constrained deployment caps it per solve (e.g. 64 on a
        small shared host). Speed saturates by ~256 (chunk ≈ 8-16 on
        production shapes); below ~64 the batching win starts eroding
        (chunk=1 costs ~+75% on the worst shapes).
    """

    eps = 8.8541878188e-12
    mu = 1.25663706127e-6

    # momwire#396: everything is served on its own — three of the four
    # refusals are `use_singular_enrichment` combinations (`__init__`'s
    # three raises, reused here). `HMatrixSolver` / `ArrayBlockSolver` fall
    # back to the dense path under enrichment rather than refusing it, so
    # they inherit this row unchanged (see those modules — no override).
    #
    # The fourth is momwire#282 stage 1's withdrawal: ground CONTACT under
    # `ground_model="refl-coef"`. Ground contact is not a declared AXIS (no
    # solver has one), so it says so through a combination key, exactly as
    # `RazorSolver` says its own contact refusal through
    # `"contact+finite_ground"`. The ground stays in `grounds`: refl-coef is
    # served, and is still the default, for wires clear of the plane.
    #
    # `buried` and `contact` are declared CELLS since momwire#792, not
    # condition tokens: this is the one family that fills a wire below the
    # interface (momwire#553) and it stands a wire end in the plane
    # (momwire#151), so both read True and the refusals around them are the
    # combinations. Two of those combinations are the ground column itself —
    # a PEC plane and a reflection-coefficient ground have no lower medium to
    # be buried IN — and they were missing entirely: `refusal("buried",
    # "pec")` answered None while `_medium_spec.wire_media` raised.
    capabilities = Capabilities(
        # Compositional row (antennaknobs#1006). `degree=` is a kwarg, so
        # this class IS both tent and quadratic — the set, not one value.
        # `extended_kernel=` likewise gives it both kernels.
        axes={
            "basis": ("bspline-1", "bspline-2"),
            "testing": ("galerkin",),
            "charge_support": ("spline",),
            "kernel": ("reduced", "extended"),
            "quadrature": ("converged",),
            "solve_strategy": ("dense",),
            # BOTH, default first. This row said ("segment-gap",) while the
            # constructor has always defaulted to feed_model="point" and
            # accepted either — so a consumer reading the row described a
            # source this class does not use by default. antennaknobs' new
            # composition line rendered exactly that: "segment gap" on a tab
            # whose stock solve uses a point gap.
            #
            # DEFAULT FIRST is now the row's convention and is gated: the
            # first declared value must be what the constructor picks when
            # nothing is passed, so anything reading the row for "what does
            # this solver do by default" gets the right answer.
            "feed_model": ("point-gap", "segment-gap"),
        },
        grounds=frozenset({"pec", "refl-coef", "sommerfeld"}),
        wire_loading=True,
        extended_kernel=True,
        junction_ports=True,
        node_gaps=True,
        knot_feeds=True,
        # momwire#673: the B-spline family integrates the delta at the
        # arclength itself and never snaps, so it lands on whichever grid the
        # caller named -- both cells True, including at d=1.
        centre_feeds=True,
        per_wire_radius=True,
        singular_enrichment=True,
        buried=True,
        contact=True,
        refusals={
            "wire_loading+singular_enrichment": _ENRICHMENT_WIRE_LOADING_REFUSAL,
            "extended_kernel+singular_enrichment": _ENRICHMENT_EXTENDED_KERNEL_REFUSAL,
            "per_wire_radius+singular_enrichment": _ENRICHMENT_PER_WIRE_RADIUS_REFUSAL,
            "contact+refl-coef": _ground_spec.CONTACT_UNDER_REFL_COEF_REFUSAL,
            # momwire#553 U5 — a wire STRICTLY below a Sommerfeld interface
            # is served; these say what around it is not. The two ground rows
            # come first: buried is a SOMMERFELD capability, and the other two
            # grounds refuse it before any of the rest is reached.
            "buried+pec": _medium_spec.BURIED_PEC_REFUSAL,
            "buried+refl-coef": _medium_spec.BURIED_REFL_REFUSAL,
            "buried+contact": _medium_spec.CONTACT_WITH_BURIED_REFUSAL,
            "buried+singular_enrichment": _BURIED_ENRICHMENT_REFUSAL,
            "buried+extended_kernel": _BURIED_EXTENDED_KERNEL_REFUSAL,
            "buried+crossing": _medium_spec.CROSSING_REFUSAL,
        },
    )

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
        n_qp_pair=None,
        n_qp_pair_same_edge=4,
        pair_order_ladder=None,
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
            raise NotImplementedError(_ENRICHMENT_WIRE_LOADING_REFUSAL)

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
        # quasi-static near field (|ΔΓ| ~0.02 at 0.1λ, ~0.13 at 0.05λ).
        # Below ~0.1λ prefer `ground_model="sommerfeld"` (exact everywhere,
        # contact-capable since #151) or the field-based SinusoidalSolver,
        # which applies NEC's dyad exactly at any height.
        #
        # AT the plane it is no longer advice. Ground CONTACT under this
        # ground model is REFUSED (momwire#282 stage 1, below) — the window
        # does not merely degrade at zero clearance, it ends, and this
        # comment used to be the only thing saying so while the code
        # answered anyway.
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
                raise NotImplementedError(_ENRICHMENT_EXTENDED_KERNEL_REFUSAL)
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

        # momwire#282 stage 1: ground CONTACT under the reflection-
        # coefficient ground is refused, at construction, before any
        # geometry is built. It is checked HERE rather than beside the other
        # ground validation above because it is the one ground check that
        # needs the wires. The condition is exactly `contact_ends` — a wire
        # END in the plane, junctioned or not, which is what
        # `_wire_endpoint_status` will tag `"ground"` — so the refusal and
        # the grounded basis agree on what contact is by construction.
        if self.ground_eps is not None and self.ground_model == "refl-coef":
            touching = _ground_spec.contact_ends(self.wires_polylines, self.ground_z)
            if touching:
                where = ", ".join(f"wire {w} {kind}" for w, kind in touching)
                raise NotImplementedError(
                    f"{where} lies in the ground plane: "
                    f"{_ground_spec.CONTACT_UNDER_REFL_COEF_REFUSAL}"
                )

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
        self._radius_per_wire, self._uniform_radius = _wire_spec.normalize_wire_radius(
            wire_radius, n_w
        )
        if use_singular_enrichment and self._uniform_radius is None:
            raise NotImplementedError(_ENRICHMENT_PER_WIRE_RADIUS_REFUSAL)

        # Distributed series wire impedance (stevenmburns/momwire#131):
        # finite conductivity (skin-effect internal impedance) and/or a
        # dielectric jacket (series inductance → velocity factor). Each is
        # None (off, today's PEC behavior), a scalar (every wire), or a
        # per-wire sequence (NaN entries switch a wire off). The loading
        # enters Z as Σ_w Z'_w(ω)·S_w over same-wire basis overlaps — see
        # `_loading_gram` / `_apply_loading`.
        _wire_loading.configure_loading(
            self, n_w, wire_conductivity, insulation_radius, insulation_eps_r
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
        # `None` means "let the deck choose" — resolved lazily by the
        # `n_qp_pair` property below, because whether this deck is buried is
        # a GEOMETRY question and the wires are not walked yet here. An
        # explicit value always wins and is never second-guessed.
        self._n_qp_pair_arg = None if n_qp_pair is None else int(n_qp_pair)
        self._n_qp_pair_resolved = None
        self.n_qp_pair_same_edge = int(n_qp_pair_same_edge)
        # Same contract as `n_qp_pair`: None defers to the deck, an explicit
        # ladder (including the empty one) always wins (momwire#906).
        self._pair_order_ladder_arg = (
            None
            if pair_order_ladder is None
            else tuple((float(r), int(n)) for r, n in pair_order_ladder)
        )

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

        # momwire#429 rank 8: the spec, its inference and its validation are
        # `_wire_spec.normalize_junctions` -- one owner, because a node-gap
        # port names a MEMBER of a group and every family has to agree with
        # every other about what the members are.
        self.junctions = _wire_spec.normalize_junctions(
            junctions, self.wires_polylines, self.n_per_edge_per_wire
        )

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
        # momwire#603 U4: the spec and its validation are
        # `_wire_spec.normalize_node_gaps` — every rule in it is about the
        # SPEC and the topology, not about this basis.  What stays here is
        # the port's column (`_node_gap_columns`), which is the one part a
        # family cannot share.
        self.node_gaps = _wire_spec.normalize_node_gaps(
            node_gaps,
            self.junctions,
            n_w,
            junction_ports=[j for j, _v in self.junction_ports],
        )

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
        # momwire#634: the near-image analytic edge blocks the fast
        # solvers' sub-block fills gather from, and the spans that say
        # which segments are on one. Both are properties of the deck
        # (and, for the blocks, of k), not of any partition.
        self._cached_near_image_spans: tuple | None = None
        self._cached_near_image_blocks: tuple | None = None  # (k, {edge: block})
        # Per-wire medium labels (momwire#553 U5): geometry plus the three
        # ground kwargs, all frozen after __init__, so one answer per solver.
        self._cached_wire_media: tuple | None = None
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
        _low_stand_off = []  # (h_min, a) per near-ground wire, momwire#865
        if gz is not None:
            media = self._wire_media()
            for w_idx, pl in enumerate(self.wires_polylines):
                tol = _ground_spec.ground_touch_tol(pl)
                pl_arr = np.asarray(pl, dtype=np.float64)
                if media[w_idx] == _medium_spec.BELOW:
                    # Strictly below the interface, in the lower medium
                    # (momwire#553 U5). No end of it touches the plane, so it
                    # has no ground contact to tag and no in-plane edge to
                    # diagnose — both of those are questions about the
                    # INTERFACE, and this wire never reaches it. What used to
                    # stand here was the "dips below the ground plane" raise;
                    # `_wire_media` is where the two geometries still refused
                    # (crossing, and buried over a ground with no lower
                    # medium) name themselves now.
                    continue
                z_at = np.abs(pl_arr[:, 2] - gz) <= tol
                if np.any(z_at[:-1] & z_at[1:]):
                    raise ValueError(
                        f"wire {w_idx} has an edge lying in the ground plane "
                        "(both endpoints at ground_z) — degenerate over a "
                        "conducting ground"
                    )
                # The validity FLOOR for the low-stand-off class (momwire#865).
                # Two forms, because the evidence has two shapes:
                #
                #   bare wire:      h >= 2a
                #   jacketed wire:  h >= b   (the jacket's OUTER radius)
                #
                # The bare bound is mesh stability. The jacketed one is
                # physical and reads better than any ratio: the jacket may
                # REST on the soil but not sink into it. At h = b the jacket
                # exactly touches the interface; below it the jacket
                # intersects the ground, which is a partly-buried wire and a
                # different problem from a wire lying on top.
                #
                # Measured (N = 4, n_rad 10 -> 30, |dZ|/|Z|):
                #
                #   deck                       h/a   h/a'   10->30  20->30
                #   bare at h = 2a (old floor)  2.00   2.00   5.81 %  0.49 %
                #   No.18 b = 1.76a at h = b    1.76   1.21   2.48 %  0.28 %
                #   THIN  b = 1.05a at h = b    1.05   1.02   6.03 %  0.93 %
                #   THIN  b = 1.10a at h = b    1.10   1.04   5.52 %  0.85 %
                #   bare at h = b (1.76a)       1.76    --    5.81 %  0.71 %
                #
                # A THICK jacket is markedly more stable than the bare deck the
                # old floor admitted, and the enlarged a' is why: the
                # a^2-regularised kernel is better conditioned for it.
                #
                # THAT ARGUMENT DOES NOT CARRY THE THIN JACKET, and the thin
                # rows are the ones that justify the bound. At b = 1.05a the
                # equivalent radius is only 1.02a, so the kernel is essentially
                # the bare one at h/a = 1.05 — and it is still no worse than
                # bare at h/a = 2.00 (6.03 % against 5.81 %, and BETTER on
                # dR/R: 1.68 % against 3.97 %). The reason is the pair of bare
                # rows above, which are IDENTICAL at h/a 2.00 and 1.76: in this
                # range the bare instability is not a function of h at all, it
                # is the mesh. A bare-like kernel at h/a 1.05 therefore behaves
                # like a bare-like kernel at h/a 2.00, which is what makes
                # h >= b safe for any jacket rather than only a thick one.
                #
                # Same "both endpoints" shape as the in-plane refusal above, so
                # a vertical whose base merely reaches the plane is untouched —
                # this is about an edge lying ALONGSIDE the interface, not one
                # ending on it.
                #
                # The CONDUCTOR radius, not the kernel one: measuring the
                # stand-off against a' would tighten the bare bound on the
                # strength of a quasi-static charge radius, which is not what
                # the mesh evidence measured.
                a_w = float(self._conductor_radius_per_wire[w_idx])
                jacket_b = None
                if self.insulation_radius is not None and np.isfinite(
                    self.insulation_radius[w_idx]
                ):
                    jacket_b = float(self.insulation_radius[w_idx])
                h_floor = (
                    jacket_b
                    if jacket_b is not None
                    else _surface_height.SURFACE_HEIGHT_CLASS.floor_h_over_a * a_w
                )
                h_edge = pl_arr[:, 2] - gz
                low = (h_edge > tol) & (h_edge < h_floor)
                if np.any(low[:-1] & low[1:]):
                    h_min = float(np.min(h_edge[low]))
                    why = (
                        (
                            f"below b = {jacket_b * 1e3:.3f} mm, its jacket's "
                            "OUTER radius. A jacketed wire may REST on the soil "
                            "but not sink into it: at h = b the jacket exactly "
                            "touches the interface, and below that the jacket "
                            "intersects the ground, which is a partly-buried "
                            "wire and a different problem. Lay it at h = b, or "
                            "model it in the lower medium if it is genuinely "
                            "buried"
                        )
                        if jacket_b is not None
                        else (
                            f"below the h/a = "
                            f"{_surface_height.SURFACE_HEIGHT_CLASS.floor_h_over_a:.0f} "
                            "validity floor for a BARE conductor, which is "
                            "mesh stability rather than physics: the check "
                            "moves "
                            f"{_surface_height.SURFACE_HEIGHT_CLASS.mesh_move_pct_at_floor:.0f} "
                            "% at the floor itself. Raise the wire, or give it "
                            "the jacket it really has — an insulated conductor "
                            "lying on soil is served down to h = b, and for "
                            "No. 18 with a 0.4 mm wall that is h/a = 1.76, "
                            "below this bare bound"
                        )
                    )
                    raise ValueError(
                        f"wire {w_idx} runs at h = {h_min * 1e3:.3f} mm above "
                        f"the interface, h/a = {h_min / a_w:.2f}, {why}. See "
                        f"{_surface_height.SURFACE_HEIGHT_CLASS.issue}"
                    )
                # Advisory candidates: served, but inside the sensitive band.
                adv = (h_edge > tol) & (
                    h_edge
                    < _surface_height.SURFACE_HEIGHT_CLASS.advisory_h_over_a * a_w
                )
                if np.any(adv[:-1] & adv[1:]):
                    _low_stand_off.append((float(np.min(h_edge[adv])), a_w))
                if start_status[w_idx] == "free" and z_at[0]:
                    start_status[w_idx] = "ground"
                if end_status[w_idx] == "free" and z_at[-1]:
                    end_status[w_idx] = "ground"
        if _low_stand_off:
            h_min, a_at = min(_low_stand_off)
            _surface_height.warn_surface_height(h_min, a_at, len(_low_stand_off))
        return start_status, end_status

    # ------------------------------------------------------------------
    # Per-segment medium (momwire#553 U5)
    # ------------------------------------------------------------------

    def _lower_medium(self):
        """Whether this solve's ground has a HALF-SPACE below the interface —
        a medium with a wavenumber of its own, not just a boundary condition.

        Exactly the Sommerfeld ground. `_ground_spec.ground_config` answers
        the same question in its own vocabulary (`mode == "compose"`), but it
        needs an ω to fold ε̃ at and this one is asked at geometry time.
        """
        return self.ground_eps is not None and self.ground_model == "sommerfeld"

    def _wire_media(self):
        """One `_medium_spec` label per wire, cached per instance.

        Raises the crossing / no-lower-medium refusals by name. Geometry and
        the three ground kwargs are all frozen after `__init__`, so this is
        computed once.
        """
        cached = self._cached_wire_media
        if cached is None:
            cached = _medium_spec.wire_media(
                self.wires_polylines,
                self.ground_z,
                lower_medium=self._lower_medium(),
                pec=self.ground_eps is None,
                crossing_ends=self._grounded_junction_ends(),
            )
            self._cached_wire_media = cached
        return cached

    def _grounded_junction_ends(self):
        """The `(wire, "start"|"end")` pairs that participate in a junction
        whose shared point lies IN the ground plane — the crossing-junction
        exemption `_medium_spec.wire_media` keys on (momwire#524 phase 2).

        Pure geometry: no media labels yet (the labels are what this
        feeds). Which of those junctions actually SPAN the interface, and
        whether the deck is inside the crossing serve's scope, is
        `_crossing_junctions`' question, asked after the labels exist.

        Groups with fewer than two members are skipped (momwire#698). A
        one-member group is legal (momwire#172 — its KCL row pins
        I_end = 0, and as a junction PORT it is a lone attachment), but one
        wire end cannot join two media, so such a group can NEVER be the
        crossing junction this exemption is granted for. Admitting it
        handed a contact+buried deck a silent escape from
        `_medium_spec.wire_media`'s refusal while `_crossing_junctions`
        still declined it (one member never spans), dropping the deck onto
        the field-form transmitted block — the O(1)-boundary-term
        configuration the refusal exists to prevent. The count is the
        cheapest NECESSARY condition for crossing and it is pure geometry,
        so it belongs here; the SUFFICIENT condition needs labels and is
        `_crossing_junctions`'. A junction no member of which reaches ABOVE
        the plane is skipped for the same shape of reason and by the same
        argument (momwire#700): it has no above side to cross TO, so it can
        never span the interface either, and admitting it handed a
        WHOLLY-BELOW deck the exemption on a deck where nothing crosses.

        The conditions themselves live in
        `_medium_spec.grounded_crossing_exemption`, shared with
        `RazorSolver._grounded_junction_ends` (momwire#700): the two trunks
        differ only in where the junction GROUPS come from — declared here,
        detected there — and momwire#700 is what two copies of the geometry
        cost.
        """
        if self.ground_z is None or not self.junctions:
            return frozenset()
        return _medium_spec.grounded_crossing_exemption(
            self.wires_polylines, self.ground_z, self.junctions
        )

    def _crossing_junctions(self):
        """Indices of junctions that CROSS the interface — grounded
        junctions joining an ABOVE wire to a BELOW wire — after checking
        the deck against the crossing serve's scope (momwire#524 phase 2).

        The scope is what the phase-2 adjudication validated, refused by
        name past it:

        * exactly ONE above member per crossing junction, N ≥ 1 below
          members — the node fan (a monopole over a buried radial screen
          risen to the node, momwire#524 fan widening). Multiple above
          members share the interface corner between above tents, a pair
          class no adjudicator has measured;
        * ONE wire radius across the deck — the radius rule
          ρ_eff = √(ρ² + a²) is the corner's regularization and a
          per-pair radius has no pinned convention;
        * OTHER junctions only wholly BELOW and off the plane — the
          buried hub (one rise + N radials joined at depth, the screen's
          other spelling). Its by-parts end terms cancel through the
          hub's own KCL row (probe35: fan M+hub ≡ M to the digit), and
          the hub ≡ N-rises gate holds the two spellings of the same
          screen together. An above-side or in-plane other junction
          stays refused: only the below axis's completion machinery has
          that cancellation measured.
        """
        media = self._wire_media()
        if _medium_spec.BELOW not in media or not self.junctions:
            return ()
        grounded = self._grounded_junctions()
        crossing = []
        for j_idx, jw in enumerate(self.junctions):
            if j_idx not in grounded:
                continue
            sides = {media[w] for w, _e in jw}
            if len(sides) == 2:
                crossing.append(j_idx)

        # The exemption audit (momwire#698), before the empty-crossing
        # return because the escape it closes is exactly the empty case.
        #
        # `_grounded_junction_ends` grants its exemption on GEOMETRY — a
        # junction whose shared point lies in the plane — and that
        # exemption silences `_medium_spec.wire_media`'s contact+buried
        # refusal. Only a junction that actually CROSSES earns the silence:
        # the crossing fill is what carries the contact end's current into
        # the lower medium, and a grounded junction that turned out not to
        # span the interface (one member, or every member above it) leaves
        # the deck on the field-form transmitted block with the contact
        # basis's O(1) boundary term unaccounted for. So re-ask the refusal
        # predicate here with the exemption narrowed from "touches the
        # plane" to "validated as crossing".
        #
        # Here and not in `wire_media`: crossing-ness is a MEDIA question
        # and the exemption set is computed before the labels exist. Here
        # and not at the dispatch site: this is the single function both
        # dispatch arms consult, and it already requires the ground attrs
        # (`_wire_media`, `_grounded_junctions`), so it cannot fire on a
        # bare `__new__` probe — the momwire#660 misplacement trap.
        earned = {tuple(m) for j_idx in crossing for m in self.junctions[j_idx]}
        stranded = [
            c
            for c in _ground_spec.contact_ends(self.wires_polylines, self.ground_z)
            if c not in earned
        ]
        if stranded:
            raise ValueError(
                _medium_spec.contact_with_buried_refusal(
                    stranded[0][0], media.index(_medium_spec.BELOW)
                )
            )

        if not crossing:
            return ()
        for j_idx in crossing:
            n_above = sum(
                1 for w, _e in self.junctions[j_idx] if media[w] == _medium_spec.ABOVE
            )
            if n_above != 1:
                raise NotImplementedError(
                    "crossing junction with more than one above member: the "
                    "crossing serve joins ONE above wire to N below wires "
                    "at the interface (momwire#524 fan widening); the "
                    "above-tent × above-tent interface corner has no "
                    "measured convention"
                )
        for j_idx, jw in enumerate(self.junctions):
            if j_idx in crossing:
                continue
            if j_idx in grounded or any(media[w] != _medium_spec.BELOW for w, _e in jw):
                raise NotImplementedError(
                    "a deck with a crossing junction and an above-side or "
                    "in-plane OTHER junction is not served: the complete "
                    "crossing spelling completes every value-1 end on its "
                    "axes, and only the below axis's completions (the "
                    "crossing node and the buried hub) are measured "
                    "(momwire#524 phase 2)"
                )
        radii = np.asarray(self._radius_per_wire, dtype=float)
        if float(radii.max()) - float(radii.min()) > 0.0:
            raise NotImplementedError(
                "crossing serve with per-wire radii: the radius rule "
                "rho_eff = sqrt(rho^2 + a^2) regularizes the corner with "
                "ONE wire radius, and a mixed-radius convention is not "
                "pinned (momwire#524 phase 2)"
            )
        return tuple(crossing)

    @property
    def n_qp_pair(self):
        """Cross-edge quadrature order, resolved from the deck when not given.

        Two defaults because there are two FILLS, not because one knob means
        two things. A deck with wires below the interface dispatches to
        `_compute_Z_operator_buried`, which the comment there calls
        "structurally a different fill, not a flag inside this one": three
        pair classes, two wavenumbers, two permittivities. It gets its own
        quadrature default for the same reason it gets its own code path.

        Why 32 there (momwire#760). The buried/crossing class carries a large
        quadrature CONSTANT — not, as #760 long recorded, a lost convergence
        rate. Measured against a q=256 reference on main @ f729cb5, soil A,
        degree 2:

            deck                          q=8      q=32
            antennaknobs hub (shipped)  0.1717    0.0047
            crossing_deck(1)            0.0929    0.0003
            hub_deck()                  0.0514    0.0001
            fan (coincident rises)      2.5562    0.1049

        Why NOT everywhere. On these decks the Sommerfeld evaluation
        dominates and the order is free (+1-2% at steady state). In free
        space it is the whole cost and it is O(n_qp^2): on a bent 400-segment
        deck, 8 -> 32 is **4.8x**. A single-edge deck is unaffected either way
        (since momwire#759 it never enters an off-edge kernel at all), which
        is exactly why a straight-wire timing would have made this look free.

        Resolution is lazy and cached: `_has_buried_wires()` walks the wires,
        so it cannot be answered in `__init__`.
        """
        if self._n_qp_pair_arg is not None:
            return self._n_qp_pair_arg
        if self._n_qp_pair_resolved is None:
            self._n_qp_pair_resolved = (
                BURIED_N_QP_PAIR
                if self.ground_z is not None and self._has_buried_wires()
                else DEFAULT_N_QP_PAIR
            )
        return self._n_qp_pair_resolved

    @property
    def pair_order_ladder(self):
        """The off-edge pair-order ladder for this deck (momwire#906).

        Explicit wins; otherwise buried decks get `BURIED_PAIR_ORDER_LADDER`
        and free-space decks `DEFAULT_PAIR_ORDER_LADDER`. Tiers that do not
        sit strictly below the resolved `n_qp_pair` are dropped rather than
        refused — a ladder written for the buried 32 is still a valid wish
        under an explicit 8, just a shorter one. What survives is validated
        by the kernel module's `_normalize_ladder`.
        """
        if self._pair_order_ladder_arg is not None:
            ladder = self._pair_order_ladder_arg
        elif self.ground_z is not None and self._has_buried_wires():
            ladder = BURIED_PAIR_ORDER_LADDER
        else:
            ladder = DEFAULT_PAIR_ORDER_LADDER
        base = self.n_qp_pair
        return _normalize_ladder(tuple((r, n) for r, n in ladder if n < base), base)

    @property
    def _accel_serves_n_qp_pair(self):
        # momwire#769: the chunked and swept fills go straight into the capped
        # C++ pair kernels, so they have to ask the same question the per-block
        # path asks. SILENT on purpose — construction does not know whether
        # this deck will ever reach an off-edge kernel (a single-edge deck has
        # not entered one since momwire#759), so the warning is left to the
        # per-block path, which knows. A property now, not an __init__ scalar,
        # because the order it asks about is itself resolved from the deck.
        return self.n_qp_pair <= _ACCEL_MAX_N_QP

    def _has_buried_wires(self):
        """Whether any wire lies strictly below the interface."""
        return _medium_spec.BELOW in self._wire_media()

    def _below_segments(self, geom):
        """`(n_segs,)` bool: this segment is in the lower medium."""
        return _medium_spec.segment_media(self._wire_media(), geom["seg_offsets"])

    def _crossing_node_members(self, crossing, media):
        """A `_crossing_fill.NodeArm` per member of every crossing junction.

        Each arm is walked OUTWARD from the shared point, edge by edge,
        until the walk passes `_crossing_fill.NODE_REACH`. Two lengths
        come back per arm: the finest segment length seen inside that
        reach (`h_resolved`, what the advisory gates) and the length of
        the segment actually touching the node (`h_adjacent`).

        They are gated separately because the touching segment does not
        predict the error — a 50 mm feed gap in front of a chain graded
        to 6 mm is a resolved node, while a single 667 mm tent is not,
        and node-adjacent h reads those 13x apart when their errors are
        ~200x apart. `_crossing_fill`'s constants carry the measurement.

        The first edge always counts, however long it is: an arm made of
        one coarse edge must not read as "unmeasured" and escape.
        """
        reach = _crossing_fill.NODE_REACH
        out = []
        for j_idx in crossing:
            for w, end in self.junctions[j_idx]:
                pl = self.wires_polylines[w]
                npe = self.n_per_edge_per_wire[w]
                # Edge indices ordered from the node outward.
                order = (
                    range(len(npe)) if end == "start" else range(len(npe) - 1, -1, -1)
                )
                h_resolved = h_adjacent = None
                walked = 0.0
                for e in order:
                    if walked > 0.0 and walked >= reach:
                        break
                    length = float(np.linalg.norm(pl[e + 1] - pl[e]))
                    h = length / int(npe[e])
                    if h_adjacent is None:
                        h_adjacent = h
                    h_resolved = h if h_resolved is None else min(h_resolved, h)
                    walked += length
                side = "above" if media[w] == _medium_spec.ABOVE else "below"
                out.append(_crossing_fill.NodeArm(h_resolved, h_adjacent, w, end, side))
        return out

    def _crossing_context(self, geom, supp_seg, polys):
        """What the crossing fill reads off this solver, as data
        (momwire#801): the basis as per-segment polynomials, the five
        geometry columns, the buried medium, and the four scalars. The
        fill never sees the solver; any formulation with a
        piecewise-polynomial basis can build the same record."""
        return _crossing_fill.CrossingContext(
            basis=_crossing_fill.BasisPolynomials(supp_seg, polys, self.degree),
            geom=_crossing_fill.AxisGeometry(
                geom["seg_l"],
                geom["seg_r"],
                geom["h_per_seg"],
                geom["tangents"],
                geom["seg_offsets"],
            ),
            medium=self._buried_medium(),
            ground_z=float(self.ground_z),
            a_wire=float(self._radius_per_wire[0]),
            omega=self.omega,
            mu=self.mu,
            eps=self.eps,
        )

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
            if abs(pt[2] - gz) <= _ground_spec.ground_touch_tol(pl):
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
        return _ground_mirror.mirror_positions(positions, self.ground_z)

    def _image_tangent_dot(self, tangents):
        """t_m · t_image_n with t_image_n = (t_n_x, t_n_y, -t_n_z)."""
        return tangents @ _ground_mirror.mirror_tangents(tangents).T

    def _remainder_qp(self, seg_l, seg_r, gz, cap=None):
        """This fill's Sommerfeld remainder order, keyed to grazing height.

        momwire#631's second half. The closed-form near-image block fixes the
        EXACT image; the remainder Q is a Sommerfeld integral with no closed
        form, so it needs the order rule razor took in momwire#510 — and the
        cross that named this defect showed neither half is sufficient alone
        (image-only left 152 %, remainder-only 306 %, both 0.62 %).

        Observers are the remainder's own Gauss nodes at the BASE order.
        They are strictly interior to their segments, which is what keeps a
        wire ENDING in the plane from reading as zero distance to its own
        mirror and pinning the order at the cap: contact is a legitimate
        geometry here (the basis handles it) and is not what this rule is
        about. Taking them at the base order rather than at the order being
        chosen also keeps the observer set independent of the answer.
        """
        cap = _REMAINDER_QP_CAP if cap is None else cap
        xg, _ = leggauss(int(self.n_qp_sommerfeld))
        tq = 0.5 * (xg + 1.0)
        nodes = seg_l[:, None, :] + tq[None, :, None] * (seg_r - seg_l)[:, None, :]
        return _quadrature.remainder_qp(
            nodes.reshape(-1, 3),
            seg_l,
            seg_r,
            gz,
            self.n_qp_sommerfeld,
            cap,
            _REMAINDER_QP_C,
        )

    def _near_image_edge_blocks(self, geom):
        """Edges whose own image is a NEAR, PARALLEL translate of themselves.

        momwire#631. For a HORIZONTAL edge at height h over `ground_z` the
        mirror is the same arc translated by -2h in z: `mirror_tangents` flips
        only the z component and that component is zero, so the image runs the
        same way at a constant perpendicular offset. The separation of the
        point at arc s on the edge from the point at s' on its image is then

            R^2 = (s - s')^2 + (2h)^2       (+ a^2 for the tube)

        which is exactly what `J_static_moment` and `_seg_seg_reg_geometry`
        integrate, both of which take `a` as nothing but that constant offset.
        So the image block IS the same-edge block at

            a_eff = sqrt(a^2 + 4h^2)

        and the analytic static + regularised split `_build_J_blocks` uses on
        the diagonal serves it unchanged — no order to key, and no premise
        about the image being far.

        Only SELF-edge blocks qualify, and only horizontal ones. A vertical
        edge's image is collinear with it rather than offset (ground contact,
        which the EK path already treats), and a tilted edge's image is
        neither parallel nor constantly offset, so neither reduces to this
        kernel. Cross-edge image pairs are a different geometry again.

        Yields `(slice, arc, a_eff)` per eligible edge, empty when the deck
        has no grazing horizontal run — in which case nothing downstream
        deviates by so much as a bit from what it did before #631.
        """
        seg_l = geom["seg_l"]
        seg_r = geom["seg_r"]
        per_wire = geom["per_wire"]
        seg_off = geom["seg_offsets"]
        gz = self.ground_z
        out = []
        for w in range(len(per_wire)):
            pw = per_wire[w]
            ed_off = pw["edge_offsets"]
            ed_arc = pw["edge_arc_edges"]
            base = seg_off[w]
            a_w = float(self._radius_per_wire[w])
            for i_e in range(len(ed_off) - 1):
                sl = slice(base + ed_off[i_e], base + ed_off[i_e + 1])
                z = np.concatenate([seg_l[sl, 2], seg_r[sl, 2]])
                arc = np.asarray(ed_arc[i_e], dtype=np.float64)
                h = float(z[0]) - gz
                # Horizontal means every endpoint of the run shares one
                # height; `a_w` is the scale that decides what "flat" means,
                # since a deviation below the tube radius is not a tilt this
                # kernel can tell from none.
                if not np.allclose(z - gz, h, rtol=0.0, atol=max(a_w, 1e-12)):
                    continue
                # Strictly ABOVE the plane. h == 0 is ground contact, not a
                # near image; h < 0 is a BURIED wire, whose image lands in the
                # other medium and whose blocks are built by the momwire#553
                # machinery rather than here — the arc identity would still
                # hold geometrically, but nothing in this arc measured it
                # there, so it is not claimed.
                if h <= 0.0:
                    continue
                two_h = 2.0 * h
                delta = float(np.min(np.diff(arc)))
                if delta <= _NEAR_IMAGE_DELTA_OVER_2H * two_h:
                    continue  # image still far compared with a segment
                out.append((sl, arc, float(np.sqrt(a_w * a_w + two_h * two_h))))
        return out

    def _near_image_analytic_block(self, arc, a_eff, k):
        """The closed-form near-image moment block for one horizontal edge.

        The extended kernel is deliberately NOT applied. EK scores image
        eligibility against the mirrored source geometry
        (`_ek_axis_labels(mirror=True)`), and a horizontal wire — whose image
        is parallel but offset rather than coaxial — is not eligible, so the
        off-edge fill this replaces carries no EK on these pairs either.
        """
        d = self.degree
        return _seg_seg_static_moments(
            arc, a_eff, max_d=d, ek=None
        ) + _seg_seg_reg_moments(
            arc, a_eff, k, max_d=d, n_qp=self.n_qp_pair_same_edge, ek=None
        )

    def _build_J_image_blocks(self, geom, k, ground=None):
        """Build the J moment tensor with j-segments mirrored across the
        ground plane. Off-edge quadrature handles every (i, j) pair
        uniformly, EXCEPT the self-edge block of a horizontal edge whose
        image has come close enough to break that rule — momwire#631, see
        `_near_image_edge_blocks` — which is overwritten with the analytic
        static + reg split at the image's effective radius.

        The mirrored source geometry comes from the ground object's
        `image_geometry()` (momwire#398 unit 1) — the mirror map is the
        ground's, and this is the same reading `_image_positions` gave when
        it was spelled here. `ground` is the caller's already-built
        `PotentialGround` when it has one; callers that don't (tests, the
        perf scripts) get one built here, which for every ground momwire
        ships is a handful of scalar operations.
        """
        d = self.degree
        seg_l = geom["seg_l"]
        seg_r = geom["seg_r"]
        if ground is None:
            ground = _potential_ground.potential_ground_for(self, geom, k, self.omega)
        img = ground.image_geometry()
        seg_l_img = img.seg_l
        seg_r_img = img.seg_r
        # Observers (rows) are the real segments — the per-observer radius
        # convention applies to the image block unchanged. Under EK the
        # source side is the MIRRORED geometry, so eligibility is scored
        # against it (`mirror=True`): a vertical monopole is coaxial with
        # its own image and extends, a horizontal wire is not and does not.
        J = _seg_seg_full_moments_offedge(
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
        for sl, arc, a_eff in self._near_image_edge_blocks(geom):
            J[:, :, sl, sl] = self._near_image_analytic_block(arc, a_eff, k)
        return J

    def _image_refl_prep(self, geom):
        """The CACHE over `_potential_ground.specular_prep`: k-independent
        per-pair specular tables (cos θ, PEC mirror dot, out-of-plane dyad
        component) for the `ground_eps` weighted image, memoised per
        geometry object so swept callers pay for the O(N²) build once, not
        per frequency.

        The build itself moved to `_potential_ground` with momwire#429
        unit 1; what is left here is the SCHEDULE, which the fill hands to
        `PotentialGround.weight_tables(prep=…)` exactly as
        `SinusoidalGalerkinSolver` hands its own cache to
        `FieldGround.projector(tables=…)` on the sibling trunk.
        """
        cached = self._cached_image_refl_prep
        if cached is not None and cached[0] is geom:
            return cached[1]
        tables = _potential_ground.specular_prep(geom, self.ground_z)
        self._cached_image_refl_prep = (geom, tables)
        return tables

    def _image_weight_row_bytes(self, n_segs):
        """Bytes-per-observer-row `PotentialGround.weight_windows`' closure
        holds alongside one moment window, for the chunked image fill's
        row-byte budget arithmetic (issue #347 — follow-up to #338, which
        sized that arithmetic on the moment window alone).

        This is the one place on the chunked route that still reads
        `ground_eps` / `ground_model` after momwire#398 unit 1, and it stays
        deliberately: it prices what a chunk allocates, which is budget —
        the same scheduling layer the field trunk's object also keeps out
        (`_field_ground`'s "what stays out"). The three branches below must
        track `weight_windows`' three producers, which is what the memory
        gates measure.

        PEC / sommerfeld (`weight_windows`' `pec_weights` /
        `sommerfeld_weights`): the
        two returned (chunk, n_segs) complex128 windows (w_A, w_Phi) — one
        gemm output plus its `ones_like`/`full` sibling. Nothing else of
        row-scale is built.

        refl-coef (`weight_windows`' `refl_weights`): the first cut of this
        accounting
        (issue #347, first pass) only priced the three arrays
        `specular_pair_tables` RETURNS (cos_th, td_img, P) plus the two
        `fresnel_rho` returns (rho_v, rho_h) plus the two final windows
        (w_A, w_Phi) — 1.06x budget locally, but 1.11x on a CI runner
        with different BLAS/allocator behavior: still dishonest, just by
        less. CPython keeps every LOCAL VARIABLE in a function's frame
        alive until the function returns (or the name is reassigned),
        regardless of whether the code uses it again — not only the
        values a function returns. Instrumented with per-statement
        tracemalloc snapshots at the gate's own config (N=1200,
        swept_mem_mb=8, chunk=30): `specular_ray_tables`' own peak
        (before `specular_pair_tables` even builds P/td_img) already
        holds dx, dy, dz, hyp, rmag, safe, inv_hyp alongside its cos_th,
        px, py returns — none of which the first-pass accounting counted
        because they die when `specular_ray_tables` returns, before
        `weight_windows`' caller ever sees them.
        `specular_pair_tables` itself adds tm_p, tn_p on top of its own
        cos_th/td_img/P/px/py inputs and returns. `fresnel_rho` then adds
        sin2 and root, both of which stay bound for its ENTIRE body (used
        once each, in the `root = sqrt(...)` line, but never reassigned
        after) — alive through both the rho_v and the rho_h expression,
        each of which additionally repeats `eps_t * cos_th` textually and
        so evaluates (and transiently allocates) it twice. Priced
        additively, at true dtype width, the way the rest of this budget
        already prices every named array it counts — not attempting to
        model exactly which of these temporaries overlap in time, the
        same conservative convention `_offedge_fallback_row_bytes` uses:
          float64  (8 B/elem):  cos_th, td_img, P, px, py, tm_p, tn_p,
                                 dx, dy, dz, hyp, rmag, inv_hyp, sin2 (14)
          bool     (1 B/elem):  safe                             (1)
          complex128 (16 B/elem): root, rho_v, rho_h, w_A, w_Phi (5)
        Measured post-fix at the same N=1200/swept_mem_mb=8 config (the
        larger array count shrinks the chunk further than the first pass
        did): 0.72x budget — comfortable local headroom under the 1.1x
        bar for the ~5% CI allocator variance that pushed the first pass
        (1.06x locally) over on a GitHub runner (1.11x there).
        """
        if self.ground_eps is None or self.ground_model == "sommerfeld":
            # w_A, w_Phi: two complex128 (chunk, n_segs) windows.
            return 2 * n_segs * 16
        n_float64 = 14
        n_bool = 1
        n_complex128 = 5
        return n_float64 * n_segs * 8 + n_bool * n_segs * 1 + n_complex128 * n_segs * 16

    def _image_weight_window_fn(self, geom):
        """Producer of the image weight WINDOWS the chunked accumulator
        consumes: `weights_fn(i0, i1) -> (w_A, w_Phi)`, each complex128 of
        shape (i1-i0, n_segs) — observer rows [i0, i1) against every source.

        The chunked path used to be handed the same global (N, N) tables the
        tensor path builds and slice them per chunk, which put 2× the dense
        Z on the peak of every grounded solve for weights only ever read one
        row-band at a time (issue #323). Each mode's algebra is row-local, so
        the window is produced directly and nothing N² is ever allocated.

        Since momwire#398 unit 1 the three per-mode producers, and the
        choice between them, are `PotentialGround.weight_windows` — the
        `(w_A, w_Φ)` weight row the architecture doc §2.2 assigns to this
        trunk's ground object. This stays as the solver-side spelling the
        fill and the existing gates name.
        """
        return _potential_ground.potential_ground_for(
            self, geom, self.k, self.omega
        ).weight_windows()

    def _image_weight_enrich_blocks(self, geom, seg_e_arr, eps_t=None):
        """Rectangular ground-image weight tables for the enrichment
        reaction (issue #328): the (n_enrich, N), (N, n_enrich) and
        (n_enrich, n_enrich) sub-blocks `_assemble_Z_enrich_image_numpy`
        actually reads, in place of the (N, N) `w_A_all`/`w_Phi_all`
        `_enrichment_Z_assemble` used to build (for all three ground modes)
        for a handful of enrichment DOFs.

        `seg_e_arr` is the per-enrichment-DOF segment id (`spec_seg`).
        `eps_t` is the frequency-only complex ε̃, precomputed by the caller
        so the sommerfeld/refl-coef branches don't redo `eps_tilde` (unused
        for PEC). Returns (w_A_row, w_Phi_row, w_A_col, w_Phi_col, w_A_ee,
        w_Phi_ee):

          row (n_enrich, N)        — observer = enrichment segments, source
                                      = every segment; feeds Z_ep's
                                      `w_A_all[seg_e, seg_m]` read.
          col (N, n_enrich)        — observer = every segment, source =
                                      enrichment segments; feeds Z_pe's
                                      `w_A_all[seg_m, seg_e]` read.
          ee (n_enrich, n_enrich)  — both axes enrichment segments; feeds
                                      Z_ee. Sliced out of `row` (whose
                                      source axis already covers every
                                      segment, enrichment ones included)
                                      rather than built a third time.

        Same per-mode algebra as `PotentialGround.weight_windows` (a third
        weight SHAPE, not yet migrated — momwire#398 unit 1). `row` and `col`
        are each their own direct gemm / `specular_pair_tables` call,
        restricted on their own natural axis — not one transposed into the
        other — so bit-exactness against the retired full-table reads rests
        on the #323 identity (slicing a gemm's output axes is exact when
        its reduction axis is untouched), not on the image reaction's
        reciprocity symmetry.
        """
        tangents = geom["tangents"]
        mirror_tangents = _ground_mirror.mirror_tangents
        tan_e = tangents[seg_e_arr]

        if self.ground_eps is None:
            w_A_row = tan_e @ mirror_tangents(tangents).T
            w_A_col = tangents @ mirror_tangents(tan_e).T
            w_Phi_row = np.ones_like(w_A_row)
            w_Phi_col = np.ones_like(w_A_col)
        elif self.ground_model == "sommerfeld":
            c2 = (eps_t - 1.0) / (eps_t + 1.0)
            w_A_row = c2 * (tan_e @ mirror_tangents(tangents).T)
            w_A_col = c2 * (tangents @ mirror_tangents(tan_e).T)
            w_Phi_row = np.full(w_A_row.shape, c2)
            w_Phi_col = np.full(w_A_col.shape, c2)
        else:
            seg_c = 0.5 * (geom["seg_l"] + geom["seg_r"])
            phi_mode = self.ground_phi_mode

            def refl_block(obs_c, obs_t, src_c, src_t):
                cos_th, td_img, P = _ground_refl.specular_pair_tables(
                    obs_c, obs_t, self.ground_z, src_centers=src_c, src_tangents=src_t
                )
                rho_v, rho_h = _ground_refl.fresnel_rho(eps_t, cos_th)
                w_A = _ground_refl.a_term_weights(rho_v, rho_h, td_img, P)
                w_Phi = _ground_refl.phi_term_weights(phi_mode, eps_t, rho_v)
                if np.ndim(w_Phi) == 0:
                    # "image"/"normal" are pair-independent; "rho_v"/"blend"
                    # already come back as per-pair windows.
                    w_Phi = np.full(w_A.shape, complex(w_Phi))
                return w_A, w_Phi

            seg_c_e = seg_c[seg_e_arr]
            w_A_row, w_Phi_row = refl_block(seg_c_e, tan_e, seg_c, tangents)
            w_A_col, w_Phi_col = refl_block(seg_c, tangents, seg_c_e, tan_e)

        w_A_row = np.ascontiguousarray(w_A_row, dtype=np.complex128)
        w_Phi_row = np.ascontiguousarray(w_Phi_row, dtype=np.complex128)
        w_A_col = np.ascontiguousarray(w_A_col, dtype=np.complex128)
        w_Phi_col = np.ascontiguousarray(w_Phi_col, dtype=np.complex128)
        w_A_ee = np.ascontiguousarray(w_A_row[:, seg_e_arr], dtype=np.complex128)
        w_Phi_ee = np.ascontiguousarray(w_Phi_row[:, seg_e_arr], dtype=np.complex128)
        return w_A_row, w_Phi_row, w_A_col, w_Phi_col, w_A_ee, w_Phi_ee

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

        Since momwire#398 unit 1 the dense fill reaches this pairing through
        `PotentialGround.weight_tables()` rather than through this method,
        so what is left here is the SPELLING the block-path tests and the
        perf script name — the whole-geometry refl-coef image, as one call
        — and since momwire#429 unit 1 it reaches it the same way they do,
        handing the object this solver's `_image_refl_prep` cache.
        """
        ground = _potential_ground.potential_ground_for(self, geom, self.k, self.omega)
        w_A, w_Phi = ground.weight_tables(prep=lambda: self._image_refl_prep(geom))
        return self._image_Z_weighted(J_img, supp_seg, polys, w_A, w_Phi)

    def _image_Z_weighted(self, J_img, supp_seg, polys, w_A, w_Phi, eps=None):
        """Weighted image assembly core: complex per-pair tables w_A on the
        A term and w_Phi on the charge term, through the C++
        `assemble_Z_bspline_weighted` kernel when available with the numpy
        einsum loop as the bit-exact fallback. Shared by the refl-coef
        ground (Fresnel tables), the Sommerfeld ground's exact-image part
        (constant C2 tables) and — since momwire#553 U5 — the BURIED image,
        whose constant tables carry A_m = (1 − ε̃)/(1 + ε̃) instead.

        `eps` is the same seam `_assemble_Z` names: the buried image's Φ term
        divides by ε̃_m = ε₀·ε̃. A complex one used to take the numpy branch by
        force because the C++ kernel's signature is `double eps`; since
        momwire#910 it takes the complex-eps twin, with the numpy loop as the
        reference it is gated against."""
        d = self.degree
        eps_z = self.eps if eps is None else eps
        in_medium = np.iscomplexobj(eps_z)
        if (
            _HAVE_BSPLINE_ASSEMBLE_W_ACCEL
            and d <= _BSPLINE_ASSEMBLE_ACCEL_MAX_D
            and not in_medium
        ):
            return _acc.assemble_Z_bspline_weighted(
                np.ascontiguousarray(J_img, dtype=np.complex128),
                np.ascontiguousarray(supp_seg, dtype=np.int64),
                np.ascontiguousarray(polys, dtype=np.float64),
                np.ascontiguousarray(w_A, dtype=np.complex128),
                np.ascontiguousarray(w_Phi, dtype=np.complex128),
                float(self.omega),
                float(eps_z),
                float(self.mu),
                int(d),
                self._cancel_flag,
            )
        if (
            _HAVE_BSPLINE_ASSEMBLE_W_CPLX_EPS_ACCEL
            and d <= _BSPLINE_ASSEMBLE_ACCEL_MAX_D
            and in_medium
        ):
            # momwire#910: the complex-eps~ twin (see `_assemble_Z`).
            return _acc.assemble_Z_bspline_weighted_cplx_eps(
                np.ascontiguousarray(J_img, dtype=np.complex128),
                np.ascontiguousarray(supp_seg, dtype=np.int64),
                np.ascontiguousarray(polys, dtype=np.float64),
                np.ascontiguousarray(w_A, dtype=np.complex128),
                np.ascontiguousarray(w_Phi, dtype=np.complex128),
                float(self.omega),
                complex(eps_z),
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
        Z_Phi = Z_Phi / (1j * self.omega * eps_z)
        return Z_A + Z_Phi

    def _ground_finite_Z(self, J_img, supp_seg, polys, geom, ground=None):
        """Ground-image matrix to SUBTRACT from the free-space Z (the
        seams' `Z - ...` convention), from the moment-tensor route's
        already-built image blocks `J_img`. Serves all three grounds.

        Since momwire#398 unit 1 this reads ONE `PotentialGround` where it
        used to branch on `ground_model` and rebuild ε̃ and C2 by hand:

        * `weight_tables() is None` — PEC. Assemble unweighted, with the
          image tangent-dot table (a different kernel, not a special case
          of the weighted one; see `PotentialGround.weight_tables`).
        * `mode == "fold"` with tables — refl-coef. The Fresnel-weighted
          image, the same pair `_image_Z_refl` builds.
        * `mode == "compose"` — sommerfeld. NEC's decomposition (theory
          manual eqs 136-147): the exact image scaled by the constant
          C2 = (eps-1)/(eps+1), which absorbs all the singular behavior
          and reuses the weighted-image kernel with constant tables, plus
          the smooth Sommerfeld remainder block. The association is the
          mode's whole point — `C2·img + Q` is summed HERE, before the
          caller's single minus. In the eps->inf limit C2 -> 1 and the
          remainder vanishes, reproducing the PEC image exactly; at
          eps -> 1 both terms vanish, reproducing free space. Both limits
          are unit-tested.

        `ground` is the caller's already-built object when it has one.
        """
        if ground is None:
            ground = _potential_ground.potential_ground_for(
                self, geom, self.k, self.omega
            )
        weights = ground.weight_tables(prep=lambda: self._image_refl_prep(geom))
        if weights is None:
            return self._assemble_Z(
                J_img,
                supp_seg,
                polys,
                geom,
                td_all=ground.image_geometry().tangent_dot(),
            )
        M = self._image_Z_weighted(J_img, supp_seg, polys, *weights)
        remainder = ground.remainder()
        if remainder is None:
            return M
        return M + remainder.evaluate(supp_seg, polys)

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
        per-segment Gauss quadrature.

        BOTH paths band the assembly over observer segments so no
        (d+1)^2 * N^2 moment tensor is ever live (issue #343): the fused
        kernel bands internally (64 MiB slab), the numpy fallback below
        assembles each chunk into Q as it goes.
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
        q = self._remainder_qp(seg_l, seg_r, gz)  # momwire#631
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
        # basis-assemble straight into Q, skipping the Python-side Jf tensor
        # and the two Galerkin einsums (sommerfeld-perf-plan Phase 4b stage
        # 2). The dense block is the symmetric obs==src case of the
        # rectangular kernel, with the support map == supp_seg (segment set
        # is all segments). The kernel's own moment slab is banded over
        # observer segments (#343), so the full-N call is bounded too.
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

        # numpy fallback. The observer chunking bounds the field evaluation,
        # but the moment tensor it filled used to be the FULL
        # (d+1, d+1, n_seg, n_seg) block — 144 N^2 bytes, the same ~9x-dense
        # transient the fused kernel carried (issue #343), and the assembly
        # below then gathered a second copy of it per wing pair. Assemble
        # each observer chunk's contribution into Q immediately instead: a
        # basis row only sees the chunks holding its own support segments, so
        # nothing bigger than (d+1, d+1, chunk, n_seg) is ever live.
        n_basis = polys.shape[0]
        Q = np.zeros((n_basis, n_basis), dtype=np.complex128)
        chunk = max(1, (1 << 19) // max(n_nodes * q, 1))
        for i0 in range(0, n_seg, chunk):
            self._checkpoint()  # per observer chunk of the eval+assemble block
            i1 = min(i0 + chunk, n_seg)
            obs = nodes[i0:i1].reshape(-1, 3)
            t_obs = np.repeat(tang[i0:i1], q, axis=0)
            proj = _sommerfeld.remainder_field_proj(
                obs, t_obs, src, t_src, gz, self.k, grid
            )
            fq = proj.reshape(i1 - i0, q, n_seg, q)
            # optimize=True (momwire#910): as one three-operand loop numpy
            # walks every index at once — 16.6 ms per chunk on the 12-radial
            # screen; contracted pairwise it is 3.6 ms, same sum to roundoff.
            Jc = np.einsum("piq,iqjr,Pjr->pPij", W[:, i0:i1], fq, W, optimize=True)
            for a in range(d + 1):
                sm = supp_seg[:, a]
                # Wings of this chunk only; every wing lands in exactly one
                # chunk, so the (a, b) pair sum is complete and disjoint.
                rows = np.nonzero((sm >= i0) & (sm < i1))[0]
                if rows.size == 0:
                    continue
                sml = sm[rows] - i0
                for b in range(d + 1):
                    sn = supp_seg[:, b]
                    J_blk = Jc[:, :, sml[:, None], sn[None, :]]
                    Q[rows] += np.einsum(
                        "mp,pPmn,nP->mn", polys[rows, a, :], J_blk, polys[:, b, :]
                    )
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
        # _Z_sommerfeld_remainder — including its momwire#631 keying, so the
        # enrichment cross-blocks are sampled at the same order as the
        # polynomial block they sit beside rather than at a stale default.
        q = self._remainder_qp(seg_l, seg_r, gz)
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
            joint = _ek_axis_groups(
                np.vstack([seg_l, self._image_positions(seg_l)]),
                np.vstack([seg_r, self._image_positions(seg_r)]),
                np.vstack([tangents, _ground_mirror.mirror_tangents(tangents)]),
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
        loop: each edge's analytic static-moment block, plus the O(N_e)
        ingredients (`ed_arc`, `a_w`) `_seg_seg_reg_geometry` needs to
        rebuild the reg-kernel quadrature geometry on demand. Returns a
        list of `(global_slice, A_static, ed_arc, a_w)`.

        The reg-kernel geometry itself — an `(N_e·n_qp, N_e·n_qp)` R table,
        128·N_e² bytes at the default `n_qp_pair_same_edge=4` — is NOT
        materialised here (issue #330): every swept caller holds this
        return value's list across the WHOLE sweep, so a retained R table
        per edge would be resident for the sweep's entire lifetime rather
        than the one edge's turn it actually needs. `A_static` (72·N_e²
        bytes) stays small enough to keep; `ed_arc` is the (N_e+1,) arc
        array already owned by `geom` (a reference, not a copy) and `a_w`
        is a scalar — both O(N_e). Consumers rebuild the R table from
        these with `_seg_seg_reg_geometry(ed_arc, a_w, max_d=d,
        n_qp=self.n_qp_pair, ek=...)` — same function, same inputs as this
        method used to call, so the rebuilt table is bit-identical to the
        one this used to retain.

        Same-edge pairs live on a single wire, so each edge's block uses
        that wire's own radius — and, under `extended_kernel`, is eligible
        in its entirety (`_EK_SAME_EDGE`). Consumers recompute the SAME
        `_EK_SAME_EDGE if self.extended_kernel else None` spec that built
        `A_static` here, so EK rides through unchanged.
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
                prep.append((sl, A_st, ed_arc[i_e], a_w))
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

        `prep`'s entries no longer carry a materialised reg-geometry table
        (issue #330): each edge's R table is rebuilt HERE, once per (chunk,
        edge) — the same granularity the old code consumed it at, since the
        chunk loop below already called `_seg_seg_reg_moments_from_geometry_swept`
        once per (chunk, edge) on a table the caller had pre-built. Rebuilding
        instead of reusing means only one edge's R table is ever alive at a
        time (the list comprehension drops each one immediately after its
        einsum), instead of every edge's table riding in `prep` for the whole
        generator's lifetime. Same function (`_seg_seg_reg_geometry`), same
        inputs (`ed_arc`, `a_w`, `d`, `n_qp_pair`, `ek`) as the retired
        materialise-once call, so this is bit-identical, not an approximation
        — only the moment CHUNK COUNT changes the wall-clock cost, never the
        answer.
        """
        # `dtype=float` below would silently drop the imaginary part of an
        # in-medium sweep. Unit 1 of momwire#553 widens the moment KERNELS,
        # not the solver: this method also feeds `_assemble_Z`'s
        # `float(self.eps)` prefactor seam and the C++ windowed assembler,
        # neither of which has a medium. Refuse by name rather than truncate.
        _refuse_complex_k(k_array, "the swept B-spline same-edge fill")
        k_array = np.asarray(k_array, dtype=float)
        n_k = k_array.shape[0]
        d = self.degree
        nm = d + 1
        n_qp = self.n_qp_pair_same_edge
        ek = _EK_SAME_EDGE if self.extended_kernel else None
        sum_ne2 = sum((sl.stop - sl.start) ** 2 for sl, _A_st, _arc, _a in prep)
        max_ne2 = max(
            ((sl.stop - sl.start) ** 2 for sl, _A_st, _arc, _a in prep), default=0
        )
        bytes_per_k = nm * nm * sum_ne2 * 16
        # Budget honesty (issue #330): while a chunk's moment blocks are
        # being built, one edge's rebuilt R table is transiently alive
        # too — float64, (N_e·n_qp)² entries, 8 bytes each. Only the
        # LARGEST edge's table is ever live at once (the list
        # comprehension below drops each edge's table before starting the
        # next), so reserve that much of the budget up front and size the
        # k-chunk out of what is left, rather than pretending the rebuild
        # is free.
        transient_bytes = max_ne2 * n_qp * n_qp * 8
        budget = max((self.swept_mem_mb << 20) - transient_bytes, 0)
        chunk = max(1, min(n_k, budget // max(bytes_per_k, 1)))
        for c0 in range(0, n_k, chunk):
            self._checkpoint()  # before each chunk's batched reg-moment build
            ks = k_array[c0 : c0 + chunk]
            reg_chunk = [
                _seg_seg_reg_moments_from_geometry_swept(
                    _seg_seg_reg_geometry(ed_arc, a_w, max_d=d, n_qp=n_qp, ek=ek), ks
                )
                for _sl, _A_st, ed_arc, a_w in prep
            ]
            for i in range(ks.shape[0]):
                same_edge_k = [
                    (sl, A_st, reg_chunk[e][i])
                    for e, (sl, A_st, _arc, _a) in enumerate(prep)
                ]
                yield c0 + i, ks[i], same_edge_k

    @staticmethod
    def _same_edge_slices(geom):
        """The segment ranges that form same-edge blocks, one per edge.

        Same spelling as the overwrite loop in `_build_J_blocks`; factored out
        so the pre-pass can ask what it is about to throw away before paying
        for it (momwire#743)."""
        per_wire = geom["per_wire"]
        seg_off = geom["seg_offsets"]
        out = []
        for w in range(len(per_wire)):
            ed_off = per_wire[w]["edge_offsets"]
            base = seg_off[w]
            for i_e in range(len(ed_off) - 1):
                out.append(slice(base + ed_off[i_e], base + ed_off[i_e + 1]))
        return out

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

        # Which segment ranges are about to be OVERWRITTEN by the analytic
        # same-edge block. Collected before the pre-pass so a deck whose
        # same-edge blocks tile the whole matrix can skip it entirely.
        same_slices = (
            [sl for sl, _A_st, _reg in same_edge_prep]
            if same_edge_prep is not None
            else self._same_edge_slices(geom)
        )
        n_total = int(seg_l.shape[0])

        # All-pairs full kernel (same a² regularization handles touching
        # segments at kink corners and at junctions to within ~1e-5 at
        # antenna scales; off-segment-pair accuracy is what GL is good at).
        # Per-observer-row radius under mixed per-wire radii.
        #
        # ...except when there is nothing to keep. This build is fused: every
        # pair by GL, then same-edge blocks overwritten. On a SINGLE-EDGE deck
        # the one same-edge block IS the whole matrix, so 100% of the pre-pass
        # is discarded — measurably (|Z(n_qp_pair=8) - Z(n_qp_pair=4)| is
        # bit-identical zero there) and expensively (momwire#743: 8.5-22% of
        # the solve depending on how the kernel scales in n_qp). A dipole is a
        # single-edge deck, so this is the commonest shape there is.
        if len(same_slices) == 1 and same_slices[0] == slice(0, n_total):
            J = np.zeros((d + 1, d + 1, n_total, n_total), dtype=complex)
        else:
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
                        ed_arc[i_e],
                        a_w,
                        k,
                        max_d=d,
                        n_qp=self.n_qp_pair_same_edge,
                        ek=ek_se,
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

    def _assemble_Z(self, J, supp_seg, polys, geom, td_all=None, eps=None):
        """Assemble the (n_basis, n_basis) complex Z matrix.

        Uses the templated C++ accelerator `assemble_Z_bspline` when
        available and `self.degree` is in its instantiation set; otherwise
        falls back to a numpy-einsum implementation that's a bit-exact
        reference target.

        `td_all` defaults to the free-space tangent dot product matrix
        derived from `geom["tangents"]`. The PEC image build passes its
        own (tx, ty, -tz)-modified table here so the same assembly fuses
        the image-current sign flip.

        `eps` overrides the permittivity the Φ term divides by — the
        `float(self.eps)` seam momwire#553 U5 finally lands. A BURIED pair
        block's mixed potential is written in the LOWER medium: jωμ₀ still
        multiplies the A term (μ_r = 1 is this arc's scope), and the Φ term
        divides by ε̃_m = ε₀·ε̃ rather than by ε₀. A complex `eps` used to take
        the numpy branch by force — the C++ assembler's signature is `double
        eps` and `float()` on a complex raises, which is the silent-
        truncation class U1 killed one level down; since momwire#910 it takes
        the complex-eps ENTRY POINT instead, gated against that numpy loop.
        `eps=None` is the pre-#553 spelling, same branch and same bytes.
        """
        d = self.degree
        n_basis, n_wings, n_poly = polys.shape
        assert n_wings == d + 1 and n_poly == d + 1

        if td_all is None:
            tangents = geom["tangents"]
            td_all = tangents @ tangents.T

        eps_z = self.eps if eps is None else eps
        in_medium = np.iscomplexobj(eps_z)

        if (
            _HAVE_BSPLINE_ASSEMBLE_ACCEL
            and d <= _BSPLINE_ASSEMBLE_ACCEL_MAX_D
            and not in_medium
        ):
            return _acc.assemble_Z_bspline(
                np.ascontiguousarray(J, dtype=np.complex128),
                np.ascontiguousarray(supp_seg, dtype=np.int64),
                np.ascontiguousarray(polys, dtype=np.float64),
                np.ascontiguousarray(td_all, dtype=np.float64),
                float(self.omega),
                float(eps_z),
                float(self.mu),
                int(d),
                self._cancel_flag,
            )
        if (
            _HAVE_BSPLINE_ASSEMBLE_CPLX_EPS_ACCEL
            and d <= _BSPLINE_ASSEMBLE_ACCEL_MAX_D
            and in_medium
        ):
            # momwire#910: the complex-eps~ ENTRY POINT, never `float()` on a
            # complex (the #553 U1 hazard class). The numpy loop below stays
            # the reference it is gated against.
            return _acc.assemble_Z_bspline_cplx_eps(
                np.ascontiguousarray(J, dtype=np.complex128),
                np.ascontiguousarray(supp_seg, dtype=np.int64),
                np.ascontiguousarray(polys, dtype=np.float64),
                np.ascontiguousarray(td_all, dtype=np.float64),
                float(self.omega),
                complex(eps_z),
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
        Z_Phi = Z_Phi / (1j * self.omega * eps_z)
        return Z_A + Z_Phi

    def _offedge_fallback_row_bytes(self, n_segs):
        """Extra per-observer-row bytes the chunked fills' row-byte budget
        must carry when `_seg_seg_full_moments_offedge` falls back to its
        pure-numpy reference (issue #347 — follow-up to #338, which noted
        but did not fix this: "the pure-Python/numpy fallback producer ...
        genuinely does carry the ~7x internal-intermediate transient the
        issue described").

        Checked against the LIVE flag on the `_bspline_kernels` module
        object (not a name imported at load time) so a test that
        monkeypatches `_bspline_kernels._HAVE_BSPLINE_ACCEL` — the same
        precedent `test_bspline_cpp_kernel_matches_numpy` uses to force the
        numpy path — is honored here too. Zero whenever the C++
        accelerator is available: its per-pair work is fixed-size stack
        scratch inside the kernel, independent of N (issue #338).

        Without the accelerator, `_seg_seg_full_moments_offedge`'s numpy
        path builds three (row, n_qp, n_segs, n_qp[, 3]) pairwise tables
        that all outlive each other before reducing to the (d+1, d+1, row,
        n_segs) window this row-byte budget otherwise prices alone:
        `diff` (float64, plus its own trailing length-3 axis, 24 bytes/
        elem), `R` (float64, 8 bytes/elem), and `G` (complex128, plus the
        `exp(-jkR)` temporary the expression that builds it needs before
        the division, effectively 2x its own 16 bytes/elem). Measured in
        isolation via tracemalloc (fit across several (n_segs, chunk,
        n_qp) combinations): ~55-56 bytes/(pair-quadrature-point) at
        n_qp=4.

        RE-FITTED to 80 for momwire#743's `n_qp_pair=8` default. The old
        64 was fitted at n_qp=4 only, and its error showed up as a
        coefficient that was not scale-free: on the #347 gate deck the
        measured transient/budget ratio ran 0.836 at n_qp=4 but 0.960 at
        n_qp=8. **That q-dependence IS the under-pricing** — this model's
        whole claim is that the transient goes as `n_qp^2 * n_segs *
        chunk`, so if the constant were right the ratio would not care
        what n_qp is. At 80 it does not: 0.716 at n_qp=4 against 0.725 at
        n_qp=8. Re-fit that way rather than by widening the gate, which
        would have hidden a real regression: at 64 the q=8 ratio cleared
        the 1.1 bar locally and FAILED it on CI (1.14), whose allocation
        behaviour sits ~19% above this box's.
        """
        if _bspline_kernels._HAVE_BSPLINE_ACCEL:
            return 0
        n_qp2 = self.n_qp_pair * self.n_qp_pair
        bytes_per_pair_point = 80
        return n_qp2 * n_segs * bytes_per_pair_point

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
        #
        # Honest only if the loop itself never holds two windows at once
        # (issue #338): `J_chunk = producer(...)` allocates the NEW window
        # before rebinding the name, so without the `del` below the OLD
        # window (still referenced by `J_chunk` from the prior iteration)
        # stays resident while its replacement is built — one budget's
        # worth of accidental double-buffering on top of the one the
        # arithmetic accounts for. Measured at 8,320 basis,
        # swept_mem_mb=256: 511 MB transient (1.997x budget) before this
        # `del`, 255 MB (0.996x) after — bit-exact, since nothing about
        # the windows' contents or the accumulation order changes.
        #
        # `_offedge_fallback_row_bytes` adds the numpy-fallback producer's
        # own internal-intermediate overhead (issue #347) — zero, and this
        # collapses to the #338 arithmetic above, whenever the C++
        # accelerator is available (the certified 8,320-basis numbers all
        # exercise that path).
        row_bytes = (d + 1) ** 2 * n_segs * 16 + self._offedge_fallback_row_bytes(
            n_segs
        )
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
            del J_chunk  # drop this window before the next one is built (#338)

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
                        ed_arc[i_e],
                        a_w,
                        max_d=d,
                        n_qp=self.n_qp_pair_same_edge,
                        ek=ek_se,
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
            del J_edge  # same lifetime discipline as the sweep above (#338)
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
        never singular, but momwire#631 found they are not always FAR
        either: a horizontal edge at grazing height sits a fraction of a
        segment from its own image, and off-edge quadrature at
        `n_qp_pair` loses that block entirely (measured 1.6 relative at
        delta/2h = 69). So there is a near-image correction pass after
        the sweep, the image-side twin of the free-space same-edge fixup
        — see `_near_image_edge_blocks`. The complex weights serve all
        three grounds: PEC (mirror tangent dot / ones), refl-coef
        (Fresnel dyad / image charge), Sommerfeld exact image
        (constant C2).

        Weights arrive as `weights_fn(i0, i1) -> (w_A, w_Phi)`, called once
        per observer chunk for WINDOWS of shape (i1-i0, n_segs) aligned with
        the moment window's trailing axes — not as global (N, N) tables
        (issue #323). Producing them per chunk is what retires the 2× dense-Z
        residency this path used to carry: nothing N² in the weights is ever
        allocated. See `PotentialGround.weight_windows` for the per-mode
        producers."""
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

        # Same lifetime discipline as `_compute_Z_dense_chunked` (#338):
        # `del` the window and weight arrays before the next iteration
        # rebinds their names, so the loop never holds an old chunk and its
        # replacement at once.
        #
        # #338 left this arithmetic sizing the moment window ALONE — the
        # weight windows `weights_fn` returns (and, for refl-coef, the
        # specular intermediates it builds them from) ride along on top of
        # every chunk uncounted, which is why the grounded transient only
        # improved to 1.22x (PEC) / 1.68x (refl-coef) budget there instead
        # of the free-space fill's 0.996x. `_image_weight_row_bytes` prices
        # those extra arrays in so the chunk shrinks to actually honor the
        # budget (issue #347). `_offedge_fallback_row_bytes` does the same
        # for the mirrored-source moment window's own numpy-fallback
        # overhead when the C++ accelerator is unavailable.
        row_bytes = (
            (d + 1) ** 2 * n_segs * 16
            + self._image_weight_row_bytes(n_segs)
            + self._offedge_fallback_row_bytes(n_segs)
        )
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
            del J_chunk, w_A_win, w_Phi_win  # (#338)

        # Near-image fixup (momwire#631), the image-side twin of the
        # free-space same-edge fixup in `_compute_Z_dense_chunked`: the sweep
        # above added the off-edge block for every pair, and for a horizontal
        # edge whose image has come close that block is the one thing
        # off-edge quadrature cannot do at this order. Accumulate the
        # DIFFERENCE, so the pairs it does not name keep exactly the
        # arithmetic they had rather than merely the same value.
        for sl, arc, a_eff in self._near_image_edge_blocks(geom):
            self._checkpoint()  # per near-image correction block
            J_edge = _seg_seg_full_moments_offedge(
                seg_l[sl],
                seg_r[sl],
                seg_l_img[sl],
                seg_r_img[sl],
                a_row[sl],
                k,
                d,
                self.n_qp_pair,
                ek=_ek_slice(ek, rows=sl, cols=sl),
            )
            corr = self._near_image_analytic_block(arc, a_eff, k) - J_edge
            del J_edge  # same lifetime discipline as the sweep above (#338)
            w_A_win, w_Phi_win = weights_fn(sl.start, sl.stop)
            e_idx = np.nonzero(((supp_c >= sl.start) & (supp_c < sl.stop)).any(axis=1))[
                0
            ].astype(np.int64)
            _acc.assemble_Z_bspline_weighted_windowed(
                np.ascontiguousarray(corr, dtype=np.complex128),
                supp_c,
                polys_c,
                # `weights_fn` hands back whole rows; this block's j-window is
                # its own columns, so both tables are narrowed to match.
                np.ascontiguousarray(w_A_win[:, sl], dtype=np.complex128),
                np.ascontiguousarray(w_Phi_win[:, sl], dtype=np.complex128),
                e_idx,
                e_idx,
                int(sl.start),
                int(sl.stop),
                int(sl.start),
                int(sl.stop),
                float(self.omega),
                float(self.eps),
                float(self.mu),
                complex(-1.0),
                Z,
                self._cancel_flag,
            )
            del corr, w_A_win, w_Phi_win  # (#338)

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
        # (n_w,) or (n_w, n_k) — the shared spec layer (momwire#428); the
        # Gram is keyed by wire, so this row consumes the per-WIRE form.
        zw = _wire_loading.loading_for(self, omega).z_wire
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
            zw = _wire_loading.loading_for(self, omega).z_wire
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
        r_w = np.real(
            _wire_loading.loading_for(
                self, self.omega if omega is None else omega
            ).z_wire
        )
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
        tangents,
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

        `tangents` is the (n_segs, 3) per-segment unit tangent table
        (issue #334) — each tangent dot `td` is formed here at the single
        (m, e) / (e, f) pair it is needed for, rather than read from a
        precomputed (n_segs, n_segs) `td_all` matrix. `assemble_Z_enrich`
        only ever reads that table at these same handful of pairs, so the
        full N² table was pure transient.
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
                td = float(np.dot(tangents[seg_e_arr[e]], tangents[seg_e_arr[f]]))
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
                    td_me = float(np.dot(tangents[seg_m], tangents[seg_e]))
                    td_em = float(np.dot(tangents[seg_e], tangents[seg_m]))
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
        w_A_row,
        w_Phi_row,
        w_A_col,
        w_Phi_col,
        w_A_ee,
        w_Phi_ee,
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
          * the free-space tangent dot on the A term becomes a per-segment-
            pair weight, and the charge term picks up its own weight —
            `w_A_row`/`w_Phi_row` (observer = enrichment segments, source =
            every segment; Z_ep), `w_A_col`/`w_Phi_col` (observer = every
            segment, source = enrichment segments; Z_pe) and `w_A_ee`/
            `w_Phi_ee` (both axes enrichment segments; Z_ee) — the (N,
            n_enrich)-scale sub-blocks of the (N, N) tables the polynomial
            image block uses, sized to what this function actually reads
            (issue #328; see `_image_weight_enrich_blocks`).

        Same per-mode weights as the polynomial block, so both grounds are
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
        pos_e_img = _ground_mirror.mirror_positions(pos_e_all, ground_z)

        # Z_ee image: real observer e against image source f. The image
        # reaction is symmetric (a mirror is an isometry, so reciprocity
        # holds), but both halves are computed independently to mirror the
        # free-space "no .T shortcut" convention. Complex per-pair weights
        # w_A / w_Φ (PEC: real td / 1; finite: Fresnel) fold in via
        # Z = jωμ·w_A·I_A + w_Φ·I_Φ/(jωε). `w_A_ee`/`w_Phi_ee` are already
        # indexed by enrichment-DOF position (issue #328), no segment-id
        # indirection needed.
        for e in range(n_enrich):
            for f in range(n_enrich):
                w_A = w_A_ee[e, f]
                w_Phi = w_Phi_ee[e, f]
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
                pos_m_img = _ground_mirror.mirror_positions(pos_m, ground_z)
                for e in range(n_enrich):
                    # `w_A_col`/`w_A_row` are already restricted to the
                    # enrichment DOFs on their small axis (issue #328); `e`
                    # indexes that axis directly, `seg_m` the full-N one.
                    wA_me = w_A_col[seg_m, e]
                    wPhi_me = w_Phi_col[seg_m, e]
                    wA_em = w_A_row[e, seg_m]
                    wPhi_em = w_Phi_row[e, seg_m]
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
        # The singular-enrichment fill is a SECOND kernel implementation —
        # its own GL quadrature over the (u·log(u/h))-shaped basis, in C++
        # (`assemble_Z_enrich`, `double k`) and in its numpy twin, plus
        # three more image/remainder assemblers below. momwire#553 unit 1
        # widens the polynomial moment kernels only, so complex k refuses
        # here by name rather than reaching `float(self.k)` and dying as a
        # TypeError that names nothing.
        _refuse_complex_k(self.k, "the B-spline singular-enrichment fill")
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
        # Pass the (n_segs, 3) tangent table straight through — the kernel
        # forms each td = tangents[i]·tangents[j] dot in-kernel at the
        # handful of (spec_seg[e], n) pairs it actually needs (issue #334).
        # The old `tangents @ tangents.T` line built the full (N, N) table
        # even for free-space geometry, and rebuilt it per k in an
        # enrichment sweep.
        tan_arr = np.ascontiguousarray(tangents, dtype=np.float64)

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
            tan_arr,
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

        ground = _potential_ground.potential_ground_for(self, geom, self.k, self.omega)
        if ground is not None:
            # Ground image reaction for the enrichment DOFs (#167). Same
            # global-minus + per-segment-pair weight convention as the
            # polynomial image block. Three grounds, one image kernel with the
            # matching weight tables:
            #   PEC        — mirror tangent dot on A, unit charge weight;
            #   refl-coef  — Fresnel weight tables
            #                (`_potential_ground.refl_weight_tables`);
            #   sommerfeld — the C2 exact-image weights, PLUS the smooth
            #                remainder-field reaction added below.
            # numpy-only — the handful of enrichment DOFs make the cost
            # negligible beside the poly image fill. The weight tables
            # themselves used to be the full (N, N) `w_A_all`/`w_Phi_all`
            # the tensor path builds, even though the blocks below only ever
            # index (N, n_enrich), (n_enrich, N) and (n_enrich, n_enrich)
            # sub-blocks of them — the last N²-scale allocation left on a
            # grounded enrichment solve (issue #328).
            # `_image_weight_enrich_blocks` produces exactly those sub-blocks;
            # momwire#398 unit 1 moved the two hoists it needs — "is this the
            # composing ground" and ε̃ — onto the ground object, but left its
            # per-mode algebra (a THIRD weight shape: row / col / ee
            # rectangular sub-blocks) where it is.
            sommerfeld = ground.mode == "compose"
            eps_t = ground.eps_tilde
            (
                w_A_row,
                w_Phi_row,
                w_A_col,
                w_Phi_col,
                w_A_ee,
                w_Phi_ee,
            ) = self._image_weight_enrich_blocks(geom, spec_seg, eps_t=eps_t)
            Z_pe_img, Z_ep_img, Z_ee_img = self._assemble_Z_enrich_image_numpy(
                spec_seg,
                spec_origin,
                seg_l_arr,
                seg_r_arr,
                h_arr,
                w_A_row,
                w_Phi_row,
                w_A_col,
                w_Phi_col,
                w_A_ee,
                w_Phi_ee,
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
    # The buried fill (momwire#553 U5)
    # ------------------------------------------------------------------

    def _buried_medium(self):
        """`(eps_t, eps_m, k_p, k_m, c2, a_m)` for a buried solve.

        `eps_t` is the ground's RELATIVE ε̃(ω); `eps_m = ε₀·ε̃` is what the
        lower medium's mixed potential divides its Φ term by. `c2` is the
        ±=+ family's exact-image coefficient and `a_m` the ±=− family's,
        `image_coefficient_below` — measured, not derived, and the negative
        of `c2`, which is exactly the kind of coincidence a sign scan exists
        to keep honest (`_sommerfeld_below.image_coefficient_below`).
        """
        return _crossing_fill.buried_medium(
            self.ground_eps, self.omega, self.eps, self.k
        )

    def _n_qp_buried_field(self):
        """Gauss order for the buried fill's THREE field-form blocks.

        **This is momwire#553's fifth inversion, and it is a tolerance
        inherited from "the remainder is small".** `n_qp_sommerfeld` is 3
        because the ±=+ remainder is a smooth correction over a segment — no
        near zone, nothing to resolve. Two of the three blocks here are
        remainders and 3 would serve them. The CROSS block is not a remainder:
        the transmitted integral is the WHOLE field, near zone and all, so the
        quantity a cross pair integrates over a segment falls like 1/R³ and
        3-point Gauss under-resolves it exactly where an above wire and a
        buried wire come close — which on a buried radial screen is every
        pair that matters.

        Measured at ε̃ = 1, where the whole cross block must reproduce the
        free-space mixed-potential block over the same pairs and the two
        disagree only by quadrature and interpolation (worst entry, relative
        to the block's largest):

            deck                       q=3      q=4      q=6      q=8
            10 m mono / 5 m radial,
            3 + 2 segments           6.6e-3   3.1e-3   6.8e-4   6.8e-4
            15 + 10 segments         3.9e-4   1.1e-5   1.0e-5   1.0e-5

        — the floor in each row is the GRID's own interpolation error, which
        the quadrature cannot go below, and q = 6 reaches it on both. Cost is
        q² per pair on a table the grid fill already dwarfs, so the order is
        set at the measurement rather than at the cheapest passing rung.

        `n_qp_sommerfeld` still raises it if a caller asked for more: the
        knob keeps meaning "at least this".

        Since momwire#692 the CROSSING fill's axes no longer route through
        this knob — its density ladder (shallow AND deep rungs) banked its
        own `_NEAR_Q`/`_FAR_Q` in `_crossing_fill`. The q = 6 measurement
        above stays authoritative for the three grid field-form blocks.
        """
        return max(int(self.n_qp_sommerfeld), _N_QP_BURIED_FIELD)

    def _buried_nodes(self, geom, seg_idx):
        """`(points, tangents, W)` for the field-form quadrature over a
        SUBSET of segments — `_Z_sommerfeld_remainder`'s own node rule,
        restricted, at `_n_qp_buried_field`'s order.

        `W[p, i, q] = w_q·u_q^p` is the moment weight the basis polynomials
        apply against, and the nodes are strictly interior to each segment,
        which is what keeps a wire ending IN the plane off its own
        singularity and what bounds every grid extent this fill sizes.
        """
        seg_l = geom["seg_l"][seg_idx]
        seg_r = geom["seg_r"][seg_idx]
        tang = geom["tangents"][seg_idx]
        h = geom["h_per_seg"][seg_idx]
        d = self.degree
        q = self._n_qp_buried_field()
        xg, wg = leggauss(q)
        tq = 0.5 * (xg + 1.0)
        nodes = seg_l[:, None, :] + tq[None, :, None] * (seg_r - seg_l)[:, None, :]
        u_phys = h[:, None] * tq[None, :]
        w_node = 0.5 * h[:, None] * wg[None, :]
        W = w_node[None] * u_phys[None] ** np.arange(d + 1)[:, None, None]
        n = seg_l.shape[0]
        return nodes.reshape(n * q, 3), np.repeat(tang, q, axis=0), W

    def _field_galerkin_block(
        self,
        supp_seg,
        polys,
        proj_fn,
        obs_idx,
        src_idx,
        obs,
        t_obs,
        W_obs,
        src,
        t_src,
        W_src,
    ):
        """`Q[m, n]` — the FIELD-form Galerkin block of a projected pair
        table, over a rectangular (observer segments × source segments)
        subset.

        `_Z_sommerfeld_remainder`'s einsum, generalized on both axes so the
        below/below remainder and the two transmitted blocks consume the same
        assembly. The caller supplies the projected table through
        `proj_fn(obs, t_obs, src, t_src) -> (n_obs·q, n_src·q)`; everything
        here is testing, not physics, so all three families share it and none
        of them can drift on the moment convention.

        The observer axis is banded exactly as the ±=+ fill bands it, so
        nothing bigger than `(d+1, d+1, chunk, n_src)` is ever live.
        """
        d = self.degree
        q = self._n_qp_buried_field()
        n_obs = len(obs_idx)
        n_src = len(src_idx)
        n_basis = polys.shape[0]
        Q = np.zeros((n_basis, n_basis), dtype=np.complex128)
        if n_obs == 0 or n_src == 0:
            return Q
        # Global segment index -> position on each axis; -1 means "not on
        # this axis", which is how a wing that belongs to the other medium
        # drops out of the block rather than being clamped into it.
        n_seg_total = int(np.max(supp_seg)) + 1
        pos_o = np.full(n_seg_total, -1, dtype=np.int64)
        pos_o[obs_idx] = np.arange(n_obs)
        pos_s = np.full(n_seg_total, -1, dtype=np.int64)
        pos_s[src_idx] = np.arange(n_src)

        chunk = max(1, (1 << 19) // max(n_src * q * q, 1))
        for i0 in range(0, n_obs, chunk):
            self._checkpoint()
            i1 = min(i0 + chunk, n_obs)
            proj = proj_fn(
                obs[i0 * q : i1 * q],
                t_obs[i0 * q : i1 * q],
                src,
                t_src,
            )
            if _HAVE_FIELD_GALERKIN_ACCEL:
                # momwire#914 unit 2. The C++ twin fuses both moment sums into
                # a q-vector per wing, so it never materialises `Jc` nor the
                # per-wing-pair gather below; it accumulates into `Q` in place.
                # Everything after this line is the reference it is gated
                # against, and the fallback when the .so predates the contract.
                _acc.assemble_field_galerkin(
                    proj,
                    np.ascontiguousarray(W_obs[:, i0:i1]),
                    W_src,
                    supp_seg,
                    polys,
                    pos_o,
                    pos_s,
                    i0,
                    Q,
                    _FIELD_GALERKIN_FUSED,
                )
                continue
            fq = proj.reshape(i1 - i0, q, n_src, q)
            # optimize=True: pairwise contraction, momwire#910 (see the
            # remainder's twin above for the measurement).
            Jc = np.einsum(
                "piq,iqjr,Pjr->pPij", W_obs[:, i0:i1], fq, W_src, optimize=True
            )
            for a in range(d + 1):
                pm = pos_o[supp_seg[:, a]]
                rows = np.nonzero((pm >= i0) & (pm < i1))[0]
                if rows.size == 0:
                    continue
                pml = pm[rows] - i0
                for b in range(d + 1):
                    pn = pos_s[supp_seg[:, b]]
                    cols = np.nonzero(pn >= 0)[0]
                    if cols.size == 0:
                        continue
                    J_blk = Jc[:, :, pml[:, None], pn[cols][None, :]]
                    Q[np.ix_(rows, cols)] += np.einsum(
                        "mp,pPmn,nP->mn",
                        polys[rows, a, :],
                        J_blk,
                        polys[cols, b, :],
                    )
        return Q

    def _build_J_blocks_subset(self, geom, k, seg_idx, mirror_sources=False):
        """`_build_J_blocks` / `_build_J_image_blocks` over a SUBSET of
        segments, scattered back into a full `(d+1, d+1, N, N)` tensor whose
        other entries stay zero.

        Zeros are the pair MASK: a pair that belongs to another medium
        contributes nothing to this block, and writing that as "the moment is
        zero" rather than as a masked assembly keeps `_assemble_Z` and
        `_image_Z_weighted` unmodified — the prefactors are global multipliers,
        so a zero moment stays zero through both.

        The subset is always a union of whole WIRES (media are labelled per
        wire, `_medium_spec`), so each wire's same-edge overwrite lands on a
        contiguous local slice and the analytic static + regularized split is
        applied exactly where `_build_J_blocks` applies it.
        """
        d = self.degree
        n_total = geom["n_segs_total"]
        if seg_idx.size == 0:
            # A FULLY buried deck has no above segments at all (the phase-0
            # buried dipoles are exactly this), and an empty subset is a legal
            # answer rather than a degenerate one: the class contributes
            # nothing.
            return np.zeros((d + 1, d + 1, n_total, n_total), dtype=np.complex128)
        seg_l = geom["seg_l"]
        seg_r = geom["seg_r"]
        a_row = self._seg_radius(geom)[seg_idx]
        src_l = seg_l[seg_idx]
        src_r = seg_r[seg_idx]
        if mirror_sources:
            src_l = self._image_positions(src_l)
            src_r = self._image_positions(src_r)
        block = _seg_seg_full_moments_offedge(
            seg_l[seg_idx],
            seg_r[seg_idx],
            src_l,
            src_r,
            a_row,
            k,
            d,
            self.n_qp_pair,
            ladder=self.pair_order_ladder,
        )
        if not mirror_sources:
            # Same-edge overwrite, per edge of each subset wire. The image
            # block never needs it: a segment and its own mirror are a
            # distance 2·depth apart, which the off-edge quadrature resolves.
            per_wire = geom["per_wire"]
            seg_off = geom["seg_offsets"]
            local_of = np.full(n_total, -1, dtype=np.int64)
            local_of[seg_idx] = np.arange(len(seg_idx))
            for w in range(len(per_wire)):
                if local_of[seg_off[w]] < 0:
                    continue
                pw = per_wire[w]
                ed_off = pw["edge_offsets"]
                ed_arc = pw["edge_arc_edges"]
                base = seg_off[w]
                a_w = float(self._radius_per_wire[w])
                for i_e in range(len(ed_off) - 1):
                    lo = int(local_of[base + ed_off[i_e]])
                    hi = lo + (ed_off[i_e + 1] - ed_off[i_e])
                    sl = slice(lo, hi)
                    A_st = _seg_seg_static_moments(ed_arc[i_e], a_w, max_d=d)
                    A_reg = _seg_seg_reg_moments(
                        ed_arc[i_e], a_w, k, max_d=d, n_qp=self.n_qp_pair_same_edge
                    )
                    block[:, :, sl, sl] = A_st + A_reg
        J = np.zeros((d + 1, d + 1, n_total, n_total), dtype=np.complex128)
        J[:, :, seg_idx[:, None], seg_idx[None, :]] = block
        return J

    def _refuse_buried_out_of_scope(self, geom):
        """The three solver configurations a buried deck may not reach.

        Each one is a SECOND kernel that has no medium, not a tolerance:
        refusing by name is the same discipline U1 applied one level down to
        the fills it did not widen.
        """
        if self.use_singular_enrichment:
            raise NotImplementedError(_BURIED_ENRICHMENT_REFUSAL)
        if self.extended_kernel:
            raise NotImplementedError(_BURIED_EXTENDED_KERNEL_REFUSAL)
        n = int(geom["n_segs_total"])
        if not self._dense_tensor_fits_budget(n) and not self._buried_chunked_serves:
            raise NotImplementedError(
                _BURIED_DENSE_BUDGET_REFUSAL.format(
                    n=n,
                    need=(self.degree + 1) ** 2 * n * n * 16 / (1 << 20),
                    budget=self.swept_mem_mb,
                )
            )

    @property
    def _buried_chunked_serves(self):
        """Whether the buried fill can take the chunked fill+assemble route
        (momwire#915): both windowed assemblers have their complex-eps~
        twins and the degree is one they instantiate."""
        return (
            _HAVE_BSPLINE_WINDOWED_ASSEMBLE_ACCEL
            and _HAVE_BSPLINE_W_WINDOWED_ASSEMBLE_ACCEL
            and _HAVE_BSPLINE_WINDOWED_CPLX_EPS_ACCEL
            and _HAVE_BSPLINE_W_WINDOWED_CPLX_EPS_ACCEL
            and self.degree <= _BSPLINE_ASSEMBLE_ACCEL_MAX_D
        )

    def _accumulate_Z_subset_chunked(
        self,
        Z,
        geom,
        k,
        seg_idx,
        supp_seg,
        polys,
        *,
        mirror_sources,
        eps,
        scale,
        weight=None,
    ):
        """`_build_J_blocks_subset` + its assembly, accumulated into `Z`
        window by window and never holding a (d+1, d+1, N, N) tensor
        (momwire#915) — the buried fill's twin of `_compute_Z_dense_chunked`
        and `_accumulate_Z_image_chunked`.

        `seg_idx` is a union of whole wires, so it is a few contiguous runs
        of global segment index; every (observer chunk × source run)
        rectangle is one off-edge window handed to the windowed assembler
        with the bases that touch it. `weight` is None for a direct block
        (the assembler forms t_m·t_n itself, as `_assemble_Z`'s default
        `td_all` does) or the scalar the image block multiplies
        (t_m·t_image_n, 1) by — C₂ above, A_m below — produced per window so
        nothing (N, N) is ever built. `scale` is +1 for the direct blocks and
        −1 for the image ones: the dense path's `Z +=` / `Z -=`, folded into
        the accumulator. `eps` complex takes the #915 twins, real the
        shipped entries — the same seam `_assemble_Z` names, one level down.

        The same-edge overwrite of the dense path is the same-edge fixup
        here: for a direct block each edge's (analytic static + regularised)
        block MINUS the off-edge block the sweep added, accumulated as a
        correction window, exactly as the free-space chunked fill does. The
        image block has no fixup on either route.
        """
        d = self.degree
        seg_l = geom["seg_l"]
        seg_r = geom["seg_r"]
        a_row = self._seg_radius(geom)
        tangents = geom["tangents"]
        src_l, src_r = seg_l, seg_r
        src_t = tangents
        if mirror_sources:
            src_l = self._image_positions(seg_l)
            src_r = self._image_positions(seg_r)
            src_t = _ground_mirror.mirror_tangents(tangents)
        supp_c = np.ascontiguousarray(supp_seg, dtype=np.int64)
        polys_c = np.ascontiguousarray(polys, dtype=np.float64)
        tan_c = np.ascontiguousarray(tangents, dtype=np.float64)
        in_medium = np.iscomplexobj(eps)
        eps_arg = complex(eps) if in_medium else float(eps)
        if weight is None:
            fn = (
                _acc.assemble_Z_bspline_windowed_cplx_eps
                if in_medium
                else _acc.assemble_Z_bspline_windowed
            )
        else:
            fn = (
                _acc.assemble_Z_bspline_weighted_windowed_cplx_eps
                if in_medium
                else _acc.assemble_Z_bspline_weighted_windowed
            )
        ladder = self.pair_order_ladder

        def _bases_touching(lo, hi):
            mask = ((supp_c >= lo) & (supp_c < hi)).any(axis=1)
            return np.nonzero(mask)[0].astype(np.int64)

        def _accumulate(J_win, i0, i1, j0, j1):
            J_win = np.ascontiguousarray(J_win, dtype=np.complex128)
            m_idx = _bases_touching(i0, i1)
            n_idx = _bases_touching(j0, j1)
            if m_idx.size == 0 or n_idx.size == 0:
                return
            if weight is None:
                if scale != 1.0:
                    J_win = J_win * scale
                fn(
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
                    eps_arg,
                    float(self.mu),
                    Z,
                    self._cancel_flag,
                )
            else:
                w_A = np.ascontiguousarray(
                    weight * (tangents[i0:i1] @ src_t[j0:j1].T), dtype=np.complex128
                )
                w_Phi = np.full((i1 - i0, j1 - j0), weight, dtype=np.complex128)
                fn(
                    J_win,
                    supp_c,
                    polys_c,
                    w_A,
                    w_Phi,
                    m_idx,
                    n_idx,
                    int(i0),
                    int(i1),
                    int(j0),
                    int(j1),
                    float(self.omega),
                    eps_arg,
                    float(self.mu),
                    complex(scale),
                    Z,
                    self._cancel_flag,
                )

        runs = _contiguous_runs(seg_idx)
        n_sub = int(seg_idx.size)
        row_bytes = (d + 1) ** 2 * n_sub * 16
        chunk = max(1, int(self.swept_mem_mb * 1024 * 1024 // row_bytes))
        for r0, r1 in runs:
            for i0 in range(r0, r1, chunk):
                self._checkpoint()  # per observer chunk of the buried subset fill
                i1 = min(i0 + chunk, r1)
                for j0, j1 in runs:
                    J_win = _seg_seg_full_moments_offedge(
                        seg_l[i0:i1],
                        seg_r[i0:i1],
                        src_l[j0:j1],
                        src_r[j0:j1],
                        a_row[i0:i1],
                        k,
                        d,
                        self.n_qp_pair,
                        ladder=ladder,
                    )
                    _accumulate(J_win, i0, i1, j0, j1)
                    del J_win  # one window live at a time (#338)

        if mirror_sources:
            return
        # Same-edge fixup, per edge of each subset wire (the dense path's
        # overwrite, as a correction window).
        per_wire = geom["per_wire"]
        seg_off = geom["seg_offsets"]
        on_subset = np.zeros(int(geom["n_segs_total"]), dtype=bool)
        on_subset[seg_idx] = True
        for w in range(len(per_wire)):
            if not on_subset[seg_off[w]]:
                continue
            pw = per_wire[w]
            ed_off = pw["edge_offsets"]
            ed_arc = pw["edge_arc_edges"]
            base = seg_off[w]
            a_w = float(self._radius_per_wire[w])
            for i_e in range(len(ed_off) - 1):
                self._checkpoint()  # per same-edge correction block
                sl = slice(base + ed_off[i_e], base + ed_off[i_e + 1])
                A_st = _seg_seg_static_moments(ed_arc[i_e], a_w, max_d=d)
                A_reg = _seg_seg_reg_moments(
                    ed_arc[i_e], a_w, k, max_d=d, n_qp=self.n_qp_pair_same_edge
                )
                J_edge = _seg_seg_full_moments_offedge(
                    seg_l[sl],
                    seg_r[sl],
                    seg_l[sl],
                    seg_r[sl],
                    a_row[sl],
                    k,
                    d,
                    self.n_qp_pair,
                    ladder=ladder,
                )
                corr = (A_st + A_reg) - J_edge
                del J_edge
                _accumulate(corr, sl.start, sl.stop, sl.start, sl.stop)

    def _buried_serve_plan(self, geom, a_idx, obs_a, obs_b, k_p, k_m, crossing=False):
        """Grid extents for the three field-form blocks, or a named refusal.

        Every extent is measured on the QUADRATURE NODES the fill will
        actually query, not on segment endpoints. That is not an
        optimisation: a wire standing in the plane has an endpoint AT the
        interface, whose transmitted-grid radius about a buried source's
        ground projection is zero, and the nodes are strictly interior, so
        the node set is both the honest domain and a smaller one.

        Nothing here clamps. Both buried families' `eval` already refuse
        rather than freeze an amplitude — their remainder is not a negligible
        tail and the transmitted surface is the whole field — so a geometry
        past a cap has no answer to give, and the refusal is raised HERE,
        before an 80-second grid fill, with the deck's own numbers and the
        limit in the same sentence.

        `crossing=True` skips the cross-medium section entirely: a crossing
        deck's cross pair is `_crossing_fill`'s designed DIRECT evaluation —
        no transmitted grid is ever built — so its θ-floor cost law must not
        refuse the deck (a node-graded crossing mesh routinely puts
        quadrature nodes fractions of a millimetre below the plane, which is
        exactly the grazing geometry the grid can't pay for and the designed
        evaluator doesn't care about).
        """
        gz = self.ground_z
        lam_p = 2.0 * np.pi / k_p
        lam_m = 2.0 * np.pi / abs(k_m)
        plan = {}

        # --- below/below: R1 = |two depths added|, theta = atan2(h, rho) ---
        d_b = gz - obs_b[:, 2]
        r1_max, th_min = _pair_extents_below(obs_b[:, 0], obs_b[:, 1], d_b)
        cap = _sommerfeld_below._SOMM_BELOW_R1_CAP_LAMBDA_M * lam_m
        if r1_max > cap:
            raise ValueError(
                _BURIED_PAST_CAP_REFUSAL.format(
                    r1=r1_max,
                    wl=r1_max / lam_m,
                    cap=cap,
                    capwl=_sommerfeld_below._SOMM_BELOW_R1_CAP_LAMBDA_M,
                    lam_m=lam_m,
                )
            )
        floor = math.radians(_sommerfeld_below._SOMM_BELOW_TH_MIN_DEG)
        if th_min < floor:
            raise ValueError(
                _BURIED_GRAZING_REFUSAL.format(
                    th=math.degrees(th_min),
                    floor=_sommerfeld_below._SOMM_BELOW_TH_MIN_DEG,
                    depth=float(np.min(d_b)) + float(np.min(d_b)),
                )
            )
        plan["r1_below"] = r1_max

        if a_idx.size == 0:
            return plan

        # --- above/above: the shipped sizing, over the above segments -----
        plan["r1_above"] = _sommerfeld.max_image_distance(
            geom["seg_l"][a_idx], geom["seg_r"][a_idx], gz
        )

        if crossing:
            return plan

        # --- cross-medium: observer polar radius about the SOURCE's ground
        #     projection, and the source depth ladder -----------------------
        z_a = obs_a[:, 2] - gz
        cdx = obs_a[:, 0][:, None] - obs_b[:, 0][None, :]
        cdy = obs_a[:, 1][:, None] - obs_b[:, 1][None, :]
        crho = np.hypot(cdx, cdy)
        r_obs = np.hypot(crho, z_a[:, None])
        r_lo = float(np.min(r_obs))
        r_hi = float(np.max(r_obs))
        zp_lo = float(np.min(d_b))
        zp_hi = float(np.max(d_b))

        r_cap = _sommerfeld_transmitted._R_CAP_LAMBDA_P * lam_p
        if r_hi > r_cap:
            raise ValueError(
                _BURIED_CROSS_RANGE_REFUSAL.format(
                    r=r_hi,
                    wl=r_hi / lam_p,
                    cap=r_cap,
                    capwl=_sommerfeld_transmitted._R_CAP_LAMBDA_P,
                )
            )
        zp_cap = _sommerfeld_transmitted._ZPRIME_MAX_LAMBDA_M * lam_m
        if zp_hi > zp_cap:
            raise ValueError(
                _BURIED_DEPTH_REFUSAL.format(
                    d=zp_hi,
                    cap=zp_cap,
                    capwl=_sommerfeld_transmitted._ZPRIME_MAX_LAMBDA_M,
                    lam_m=lam_m,
                )
            )
        th_cross = float(np.min(np.arctan2(z_a[:, None], crho)))
        # Against the domain the grid will HAVE, not the one asked for: r_max
        # buckets up and that raises the floor (`grid_extent`).
        _rmin_eff, r_hi_eff, th_floor = _sommerfeld_transmitted.grid_extent(
            k_p, r_hi, zp_lo, r_min=r_lo
        )
        if th_cross < th_floor:
            raise ValueError(
                _BURIED_CROSS_GRAZING_REFUSAL.format(
                    th=math.degrees(th_cross),
                    floor=math.degrees(th_floor),
                    r=r_hi_eff,
                    depth=zp_lo,
                )
            )
        plan["r_cross_max"] = r_hi
        plan["r_cross_min"] = r_lo
        plan["zp_min"] = zp_lo
        plan["zp_max"] = zp_hi
        return plan

    def _compute_Z_operator_buried(self, geom, supp_seg, polys):
        """The mixed-medium dense Z: per-segment media, three pair classes,
        one matrix (momwire#553 U5).

        A deck with buried wires is filled pair class by pair class, and the
        classes are not variants of each other:

        * **above/above** — the shipped composition, unchanged. Direct at k₀
          through the mixed potential in air, minus `C₂·image + Q`.
        * **below/below** — the same SHAPE in the lower medium and nothing
          else shared. Direct at k_m through the mixed potential written in
          the medium (jωμ₀ on A, 1/(jωε̃_m) on Φ — the `float(self.eps)` seam
          this unit lands), minus `A_m·image + Q_below`, with the image
          mirrored through the interface exactly as the ±=+ one is and
          `A_m = (1 − ε̃)/(1 + ε̃)` in C₂'s place. The image of a below source
          is ABOVE, and its interaction with a below observer is the k_m
          direct kernel at the image distance — the phase-0 composition,
          EQUATIONS.md §Regime 2.
        * **cross-medium** — neither. The transmitted integral is the WHOLE
          field across the interface: no direct term, no image term, no
          mixed-potential prefactors, just `⟨E, testing⟩` subtracted like any
          field-form block. Both directions are filled and their agreement is
          the reciprocity gate.

        The three field-form blocks (the two remainders and the transmitted
        pair) all subtract, because a field's contribution to the EFIE
        Galerkin matrix is `−⟨f, E⟩` and the mixed-potential block is that
        same functional written out; the ±=+ path's single `Z -= (C₂·img + Q)`
        is the same convention and this method keeps it verbatim.

        **The fifth inversion lives in that last sentence.** "The
        mixed-potential block is that same functional written out" is true up
        to an integration-by-parts BOUNDARY TERM `[f_m·Φ_n]` at each end of a
        basis's support, and momwire has always mixed the two forms — MP for
        direct and image, field-form for the remainder — because every basis
        that vanishes at its own ends makes that term identically zero. A
        ground CONTACT basis does not vanish there. The shipped path gets away
        with it because its field-form block is a small REMAINDER, so a
        boundary term on a small block is a small error; this fill's
        cross-medium block is the WHOLE interaction between the two media, and
        the same term is O(1) of it. Measured at ε̃ = 1, where the entire
        buried fill must reproduce the free-space fill exactly: 2.5 relative
        on the contact basis and 1e-8 on every other one, unmoved by
        quadrature order, against 1.0e-5 everywhere once the above wire is
        lifted clear of the plane. That is why `_medium_spec` refuses a
        ground contact and a buried wire on the same deck, and it is the same
        shape as the arc's other four inversions: a ±=+ convenience whose
        licence is "the remainder is small", used where nothing is a
        remainder. Undoing it wants the transmitted family's scalar
        POTENTIALS, so the cross block can be written mixed-potential like
        its neighbours — and on a deck with a CROSSING junction that is now
        exactly what happens: the crossing branch below fills the cross pair
        with `_crossing_fill`'s complete designed mixed-potential spelling,
        boundary terms, corner and all (momwire#524 phase 2, adjudicated
        2026-08-26). The contact-plus-buried refusal itself still stands
        while P3 re-scores its anchors under the same machinery.
        """
        below = self._below_segments(geom)
        b_idx = np.nonzero(below)[0]
        a_idx = np.nonzero(~below)[0]
        eps_t, eps_m, k_p, k_m, c2, a_m = self._buried_medium()
        gz = self.ground_z

        self._refuse_buried_out_of_scope(geom)
        obs_a, t_a, W_a = self._buried_nodes(geom, a_idx)
        obs_b, t_b, W_b = self._buried_nodes(geom, b_idx)
        # Asked ONCE, and asked UNCONDITIONALLY (momwire#700). It used to sit
        # inside `bool(a_idx.size and ...)` at the plan site and behind the
        # same guard again below, so on a WHOLLY-below deck — no above
        # segment — Python short-circuited both calls and momwire#698's
        # exemption audit never ran at all. `_crossing_junctions` is a
        # VALIDATION as much as a label (it is where a grounded junction that
        # cannot cross gives its exemption back), and a validation behind a
        # short-circuit is a validation that does not run. Razor asks it
        # unconditionally in `_refuse_buried_geometry`, which is why the two
        # trunks answered differently on the same deck.
        crossing_j = self._crossing_junctions()
        plan = self._buried_serve_plan(
            geom,
            a_idx,
            obs_a,
            obs_b,
            k_p,
            k_m,
            crossing=bool(a_idx.size and crossing_j),
        )

        # --- the two direct blocks and the two image blocks, each in its
        #     own medium. Chunked whenever the windowed assemblers' complex-
        #     eps~ twins are built (momwire#915): the same four terms with the
        #     same signs, never a (d+1, d+1, N, N) tensor, and faster even
        #     where the tensor fits (12 radials: 3.5 -> 2.7 s) because the
        #     zero-padded scatter of `_build_J_blocks_subset` is gone. The
        #     dense route below is the REFERENCE every buried gate was pinned
        #     on and the chunked route is gated against it at 1e-12.
        if not self._buried_chunked_serves:
            self._checkpoint()
            Z = self._assemble_Z(
                self._build_J_blocks_subset(geom, k_m, b_idx),
                supp_seg,
                polys,
                geom,
                eps=eps_m,
            )
            td_img = self._image_tangent_dot(geom["tangents"])
            if a_idx.size:
                self._checkpoint()
                Z += self._assemble_Z(
                    self._build_J_blocks_subset(geom, k_p, a_idx), supp_seg, polys, geom
                )
                self._checkpoint()
                Z -= self._image_Z_weighted(
                    self._build_J_blocks_subset(geom, k_p, a_idx, mirror_sources=True),
                    supp_seg,
                    polys,
                    c2 * td_img.astype(np.complex128),
                    np.full(td_img.shape, c2, dtype=np.complex128),
                )
            self._checkpoint()
            Z -= self._image_Z_weighted(
                self._build_J_blocks_subset(geom, k_m, b_idx, mirror_sources=True),
                supp_seg,
                polys,
                a_m * td_img.astype(np.complex128),
                np.full(td_img.shape, a_m, dtype=np.complex128),
                eps=eps_m,
            )
            del td_img
        else:
            n_basis = supp_seg.shape[0]
            Z = np.zeros((n_basis, n_basis), dtype=np.complex128, order="F")
            self._accumulate_Z_subset_chunked(
                Z,
                geom,
                k_m,
                b_idx,
                supp_seg,
                polys,
                mirror_sources=False,
                eps=eps_m,
                scale=1.0,
            )
            if a_idx.size:
                self._accumulate_Z_subset_chunked(
                    Z,
                    geom,
                    k_p,
                    a_idx,
                    supp_seg,
                    polys,
                    mirror_sources=False,
                    eps=self.eps,
                    scale=1.0,
                )
                self._accumulate_Z_subset_chunked(
                    Z,
                    geom,
                    k_p,
                    a_idx,
                    supp_seg,
                    polys,
                    mirror_sources=True,
                    eps=self.eps,
                    scale=-1.0,
                    weight=complex(c2),
                )
            self._accumulate_Z_subset_chunked(
                Z,
                geom,
                k_m,
                b_idx,
                supp_seg,
                polys,
                mirror_sources=True,
                eps=eps_m,
                scale=-1.0,
                weight=complex(a_m),
            )

        # --- the three field-form blocks -----------------------------------
        if a_idx.size:
            grid_above = self._somm_grid(eps_t, plan["r1_above"])

            def proj_aa(o, to, s, ts):
                return _sommerfeld.remainder_field_proj(
                    o, to, s, ts, gz, k_p, grid_above, cancel_flag=self._cancel_flag
                )

            Z -= self._field_galerkin_block(
                supp_seg, polys, proj_aa, a_idx, a_idx, obs_a, t_a, W_a, obs_a, t_a, W_a
            )

        grid_below = _sommerfeld_below.get_grid_below(
            eps_t, k_p, plan["r1_below"], self.omega, mu=self.mu
        )

        def proj_bb(o, to, s, ts):
            return _sommerfeld_below.remainder_field_proj_below(
                o, to, s, ts, gz, k_p, k_m, grid_below
            )

        Z -= self._field_galerkin_block(
            supp_seg, polys, proj_bb, b_idx, b_idx, obs_b, t_b, W_b, obs_b, t_b, W_b
        )

        crossing = crossing_j if a_idx.size else ()
        if crossing:
            # The node-mesh advisory (momwire#696) goes first, because
            # this is the one place per fill where the crossing serve
            # actually engages: the plan site above asks the same question
            # before the deck is committed to the crossing path.
            _crossing_fill.warn_coarse_node(
                self._crossing_node_members(crossing, self._wire_media())
            )
            # The crossing serve (momwire#524 phase 2): the cross pair is
            # the COMPLETE designed mixed-potential spelling on graded
            # axes. Near / corner-adjacent pairs are direct contour
            # evaluations — no grid, no interpolation to exclude the
            # corner — while admissible far blocks ride the #688
            # admissibility split (coarse axes + low-rank ACA, parity-
            # gated against the dense fill). The transpose is
            # reciprocity, measured on the adjudication decks rather
            # than assumed. The self families get their missing by-parts
            # bnd + corner content on the dense axes; continuity through
            # the node and the AGARD slope condition then emerge from
            # the fill with no constraint row and no merged dof.
            self._checkpoint()
            ctx = self._crossing_context(geom, supp_seg, polys)
            ax_a = _crossing_fill.axis_data(ctx, a_idx)
            ax_b = _crossing_fill.axis_data(ctx, b_idx)
            t_ab = _crossing_fill.cross_complete_block_split(
                ctx, a_idx, b_idx, ax_a, ax_b
            )
            Z -= t_ab
            Z -= t_ab.T
            Z += _crossing_fill.self_completions(ctx, ax_b, ax_a)
        elif a_idx.size:
            grid_t = _sommerfeld_transmitted.get_grid_below_above(
                eps_t,
                k_p,
                plan["r_cross_max"],
                plan["zp_min"],
                plan["zp_max"],
                self.omega,
                mu=self.mu,
                r_min=plan["r_cross_min"],
            )

            def proj_ab(o, to, s, ts):
                return _sommerfeld_transmitted.transmitted_field_proj_below_to_above(
                    o, to, s, ts, gz, k_p, k_m, grid_t
                )

            def proj_ba(o, to, s, ts):
                return _sommerfeld_transmitted.transmitted_field_proj_above_to_below(
                    o, to, s, ts, gz, k_p, k_m, grid_t
                )

            Z -= self._field_galerkin_block(
                supp_seg, polys, proj_ab, a_idx, b_idx, obs_a, t_a, W_a, obs_b, t_b, W_b
            )
            Z -= self._field_galerkin_block(
                supp_seg, polys, proj_ba, b_idx, a_idx, obs_b, t_b, W_b, obs_a, t_a, W_a
            )

        return self._apply_loading(Z)

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
        if self.ground_z is not None and self._has_buried_wires():
            # Per-segment media (momwire#553 U5). Structurally a different
            # fill, not a flag inside this one: three pair classes, two
            # wavenumbers, two permittivities and three field-form blocks.
            # An all-above deck never reaches it, which is what keeps the
            # shipped path byte-identical.
            return self._compute_Z_operator_buried(geom, supp_seg, polys)

        dense_tensor_fits = self._dense_tensor_fits_budget(geom["n_segs_total"])

        self._checkpoint()  # after geometry/basis, before the J-block fill
        if (
            _HAVE_BSPLINE_WINDOWED_ASSEMBLE_ACCEL
            and self.degree <= _BSPLINE_ASSEMBLE_ACCEL_MAX_D
            and not dense_tensor_fits
            and self._accel_serves_n_qp_pair
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

        ground = _potential_ground.potential_ground_for(self, geom, self.k, self.omega)
        if ground is not None:
            self._checkpoint()  # between fills: before the image J-block fill
            # Image method: subtract the same-shape assembly built from
            # J integrals over image segments + the ground's own weights on
            # the A and Φ terms (PEC: the (tx, ty, -tz)-modified tangent dot
            # products, unweighted charge). The minus sign captures both the
            # image current's horizontal anti-parallel direction and the
            # image charge's sign flip (one minus combined) — which is why
            # a `"fold"` ground may take it entry by entry and a
            # `"compose"` one may not (momwire#398 unit 1).
            if (
                _HAVE_BSPLINE_W_WINDOWED_ASSEMBLE_ACCEL
                and self.degree <= _BSPLINE_ASSEMBLE_ACCEL_MAX_D
                and self._accel_serves_n_qp_pair
            ):
                # Chunked image subtraction — no (d+1, d+1, N, N) image
                # tensor, no intermediate n_basis² matrix (issue #136), and
                # no (N, N) weight table either: `weight_windows` produces
                # each observer chunk's window on demand (issue #323).
                # `"compose"`'s remainder Q is a separate, already
                # observer-chunked term, and the composition survives being
                # split across the two calls only because the accumulator's
                # `scale = -1` and this `Z -=` are the SAME single minus.
                self._accumulate_Z_image_chunked(
                    Z,
                    geom,
                    self.k,
                    supp_seg,
                    polys,
                    ground.weight_windows(),
                )
                remainder = ground.remainder()
                if remainder is not None:
                    Z -= remainder.evaluate(supp_seg, polys)
            else:
                J_img = self._build_J_image_blocks(geom, self.k, ground=ground)
                # In-place subtract (issue #334): `Z = Z - ...` held the
                # old Z, the new Z, and the image block — three n_basis²
                # matrices at once. `Z -=` folds the subtraction into Z's
                # own buffer, holding two.
                Z -= self._ground_finite_Z(J_img, supp_seg, polys, geom, ground=ground)

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
                # Z is fully copied into Z_aug's polynomial block and never
                # read again on this path (the branch returns below) — drop
                # it here rather than at function exit, halving the peak
                # (issue #334).
                del Z
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
                # Z is fully copied into Z_aug's polynomial block and never
                # read again on this path (the branch returns below) — drop
                # it here rather than at function exit, halving the peak
                # (issue #334).
                del Z
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
        _refuse_complex_k(k_array, "BSplineSolver._port_solutions_swept")
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
            and self._accel_serves_n_qp_pair
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

        `prep`'s reg geometry is rebuilt per (chunk, edge) rather than held
        for the whole sweep (issue #330) — see `_same_edge_prep_swept_chunks`
        for the full rationale; the chunk arithmetic here mirrors it.
        """
        d = self.degree
        n_k = k_array.shape[0]
        seg_l, seg_r = geom["seg_l"], geom["seg_r"]
        tangents = geom["tangents"]
        N = seg_l.shape[0]
        nm = d + 1
        n_qp = self.n_qp_pair_same_edge

        # k-independent: same-edge static moments + O(N_e) geometry inputs,
        # image segments — all built once for the whole sweep (the
        # reg-kernel R table itself is rebuilt per chunk below). The
        # tangent-dot is NOT hoisted as an (N, N) table here — the C++
        # kernel forms it in-kernel from the (N, 3) tangent table(s) it's
        # handed (issue #333, the swept-batched twin of #318); `tangents`
        # itself is already O(N) and geom's own array, so nothing extra to
        # hold for free space. The image term needs the mirrored copy,
        # which is O(N) too.
        prep = self._same_edge_prep(geom)
        # Same specs as the per-k fills; the same-edge half already rides
        # `prep` (its static blocks carry `_EK_SAME_EDGE`, and the rebuilt
        # reg geometry below uses the same spec). Under EK the batched
        # offedge kernels reach their own C++ twin (momwire#270 unit 2);
        # only a build without it stacks per-k calls.
        ek = self._ek_spec(geom) if self.extended_kernel else None
        ek_se = _EK_SAME_EDGE if self.extended_kernel else None
        ek_img = None
        tangents_mirror = None
        if self.ground_z is not None:
            tangents_mirror = _ground_mirror.mirror_tangents(tangents)
            seg_l_img = self._image_positions(seg_l)
            seg_r_img = self._image_positions(seg_r)
            if self.extended_kernel:
                ek_img = self._ek_spec(geom, mirror=True)

        # Chunk size from the memory budget. Per k, the transients are
        # the all-pairs J tensor (nm² N² complex) plus the per-edge
        # same-edge reg moment blocks (nm² ΣN_e² complex); the PEC image
        # J reuses J's footprint (J is dropped before the image build).
        sum_ne2 = sum((sl.stop - sl.start) ** 2 for sl, _A_st, _arc, _a in prep)
        max_ne2 = max(
            ((sl.stop - sl.start) ** 2 for sl, _A_st, _arc, _a in prep), default=0
        )
        bytes_per_k = nm * nm * (N * N + sum_ne2) * 16
        # Budget honesty (issue #330): rebuilding a chunk's same-edge reg
        # moments transiently holds ONE edge's rebuilt R table at a time
        # (float64, (N_e·n_qp)² entries, 8 bytes each) — the largest edge
        # sets the high-water mark, since the loop below overwrites `reg_geo`
        # edge by edge rather than keeping every edge's table alive together.
        transient_bytes = max_ne2 * n_qp * n_qp * 8
        budget = max((self.swept_mem_mb << 20) - transient_bytes, 0)
        chunk = max(1, min(n_k, budget // max(bytes_per_k, 1)))

        def _assemble_swept(J_tensor, t_row, t_col, omega_chunk):
            return _acc.assemble_Z_bspline_swept(
                np.ascontiguousarray(J_tensor, dtype=np.complex128),
                np.ascontiguousarray(supp_seg, dtype=np.int64),
                np.ascontiguousarray(polys, dtype=np.float64),
                np.ascontiguousarray(t_row, dtype=np.float64),
                np.ascontiguousarray(t_col, dtype=np.float64),
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
            # ~1 GB on a 41-pt sweep of a single 400-seg wire). The R table
            # itself is rebuilt HERE from `ed_arc`/`a_w` rather than reused
            # from `prep` (issue #330): same `_seg_seg_reg_geometry` call,
            # same inputs, bit-identical result, but only one edge's table
            # is alive at a time instead of every edge's riding in `prep`
            # for the whole sweep.
            for sl, A_st, ed_arc, a_w in prep:
                reg_geo = _seg_seg_reg_geometry(
                    ed_arc, a_w, max_d=d, n_qp=n_qp, ek=ek_se
                )
                J[:, :, :, sl, sl] = A_st[
                    None
                ] + _seg_seg_reg_moments_from_geometry_swept(reg_geo, ks)
            Z = _assemble_swept(J, tangents, tangents, omega_chunk)
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
                # Near-image blocks (momwire#631), the image-side twin of the
                # same-edge overwrite this loop already does above: a
                # horizontal edge at grazing height sits a fraction of a
                # segment from its own image, where off-edge quadrature at
                # `n_qp_pair` loses the block. Same closed form, built per k
                # through the same swept reg twin. Without this the swept
                # route answered a grazing deck 194 % away from what
                # `compute_impedance` answered for it.
                for sl, arc, a_eff in self._near_image_edge_blocks(geom):
                    A_st_ni = _seg_seg_static_moments(arc, a_eff, max_d=d, ek=None)
                    reg_ni = _seg_seg_reg_geometry(
                        arc, a_eff, max_d=d, n_qp=self.n_qp_pair_same_edge, ek=None
                    )
                    J_img[:, :, :, sl, sl] = A_st_ni[
                        None
                    ] + _seg_seg_reg_moments_from_geometry_swept(reg_ni, ks)
                # In-place fold (issue #333 part 2): `Z = Z - ...` held the
                # old stack, the image stack, and the difference — three
                # (chunk, n_basis, n_basis) complex stacks at the sweep's
                # peak moment. `Z -=` holds two.
                Z -= _assemble_swept(J_img, tangents, tangents_mirror, omega_chunk)
                # (#347) same rebind-before-del gap #338 fixed on the other
                # two chunked fills: this is a GENERATOR, so `J_img` stays
                # bound across the `yield` below and into the top of the
                # next iteration, where `J = producer(...)` rebuilds a new
                # (chunk, nm, nm, N, N) stack before this one's reference is
                # dropped — one budget's worth of accidental double
                # buffering `bytes_per_k` never accounted for.
                del J_img
            else:
                # (#347) same reasoning as the grounded `del J_img` above,
                # for the free-space branch: without this, `J` rides across
                # the yield and into the next iteration's rebind.
                del J
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
        _refuse_complex_k(k_array, "BSplineSolver.compute_impedance_swept")
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

    def current_slopes(self, coeffs, s_array=None):
        """Per-wire ``dI/ds`` — the solved current's arc-length derivative.

        The twin of :meth:`currents_at_knots`, differentiated in the basis
        rather than around it: a B-spline of degree ``d`` is a polynomial on
        each knot span and scipy's :class:`~scipy.interpolate.BSpline` hands
        back its exact derivative as a spline of degree ``d-1``, so this is
        the same sum evaluated with the same coefficients and no step size
        anywhere.

        Returned per wire, in ``wires_polylines`` order, at the mesh knots
        (``s_array=None``) or at the arc positions given per wire — the same
        two calling conventions, and the same clipping into the clamped knot
        range, that :meth:`currents_at_knots` uses.

        **Why it exists** (momwire#497): the linear charge density a NEC
        printout reports is ``q = -(1/jω)·dI/ds`` at each element's centre,
        and differencing knot currents to get it would report a
        discretisation of a quantity this basis already knows exactly. At
        ``degree >= 2`` the derivative is continuous across a knot; at
        ``degree == 1`` it is piecewise constant and a sample taken AT a knot
        lands on whichever span scipy assigns it, so ask for centres.

        Singular enrichment is refused rather than silently dropped: the
        enrichment shape ``(u/h)·log(u/h)`` contributes nothing to the
        current AT a knot but its slope diverges there, so an evaluation
        that ignored it would be wrong wherever it matters most.
        """
        if self.use_singular_enrichment:
            raise NotImplementedError(
                "current_slopes does not serve use_singular_enrichment=True: "
                "the enrichment shape's slope is singular at the junction "
                "knot, so dropping it would be a silent error rather than an "
                "approximation"
            )
        coeffs = np.asarray(coeffs)
        geom = self._build_geometry()
        _, _, _, wire_knots, wire_basis_global = self._build_basis_polynomials(geom)
        d = self.degree

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
            kept, local_to_global = wire_basis_global[w_idx]
            c_basis = np.zeros(len(knots_vec) - d - 1, dtype=np.complex128)
            for kept_idx, (j_local, _, _, _) in enumerate(kept):
                c_basis[j_local] = coeffs[local_to_global[kept_idx]]
            if s_eval.shape[0] == 0:
                out.append(np.zeros(0, dtype=np.complex128))
                continue
            spline = BSpline(knots_vec, c_basis, d, extrapolate=False)
            out.append(np.asarray(spline.derivative(1)(s_eval), dtype=np.complex128))
        return out

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
