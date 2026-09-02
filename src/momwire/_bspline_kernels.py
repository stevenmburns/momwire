"""Polynomial-moment integrals for the B-spline Galerkin MoM.

Two flavours of same-edge moment integrals are exposed:

  * `_seg_seg_static_moments`: closed-form static-kernel piece
        S_pq[i, j] = ∫∫ (s-α_i)^p (s'-A_j)^q / (4π √((s-s')²+a²)) ds' ds
    for p, q ∈ {0, ..., max_d}, on every (i, j) pair of an edge's N segments.
    The closed form handles the log singularity on the (i, i) diagonal that
    Gauss-Legendre quadrature converges on only logarithmically. It is
    `_bspline_static_far.J_static_stable`, imported below under the name
    `J_static_moment`: momwire#808 re-derived the same-edge family in a
    cancellation-free grouping, so the sympy dump in
    `_bspline_static_moments.py` is no longer what this kernel evaluates
    (`_bspline_ek_moments` still builds on it).

  * `_seg_seg_reg_moments`: smooth-kernel piece
        R_pq[i, j] = ∫∫ (s-α_i)^p (s'-A_j)^q · (exp(-jkR)-1)/(4π R) ds' ds
    by per-segment Gauss-Legendre quadrature on the bounded difference
    (exp(-jkR) - 1)/R (limit -jk at R = 0 — with k COMPLEX too, see the
    in-medium section below; the limit is the analytic continuation and the
    magnitude bound is |k|, not k).

The full moment is S_pq + R_pq on same-edge pairs; for cross-edge / cross-
wire pairs (not in this first-cut single-wire scope) the unregularized
GL quadrature on G = exp(-jkR)/(4π R) is fine because R ≥ a there.

For the tent basis (degree 1) this is bit-for-bit equivalent to the retired
TriangularSolver's same-edge kernels (verified end-to-end: the two solvers
agreed to solve-order roundoff on knot-fed meshes — tests/test_tent_parity.py
pins the values), i.e. equivalent to the
existing `_seg_seg_static_all` + `_seg_seg_reg_all` kernels; the new
module is just the generalization to higher polynomial moments.

THE EXTENDED THIN-WIRE KERNEL (momwire#249)
-------------------------------------------
Everything above is NEC's *reduced* kernel: the source current is a filament
on the wire axis and the conductor's girth survives only as the a²
regularization of R. Passing `ek=` (an `_EK` spec) switches the eligible
segment pairs to NEC's *extended* kernel, Eq 89 of the LLNL theory manual:

    G_ek = (e^{-jkR}/4πR)·(1 + T1·C2 − T2·C1)
    C1 = 1 + jkR    C2 = 3·C1 − (kR)²    T1 = b²ρ²/(4R⁴)    T2 = b²/(2R²)

which is the O(b²) truncation of the azimuthal average of the free-space
Green's function over a source tube of radius b, seen from an observer a
distance ρ off the source axis.

momwire extends only COAXIAL EQUAL-RADIUS pairs (`_ek_axis_groups`), and on
those the whole thing collapses: the observer sits on its own wire's surface
on the same axis, so ρ = a and b = a and R = √(ζ² + a²) is *the same R the
reduced kernel already computes*. Eq 89 becomes a scalar multiplicative
factor of R alone (`_ek_factor`), manifestly symmetric in i ↔ j and
manifestly → 1 as a → 0. NEC's IRA swapped arm is unreachable in this
specialisation (the test `ρ_eval < b` is strict and ρ_eval = b = a).

The factor rides through the existing static/regular split unchanged:

    G_ek,static = (1/4π)·[ 1/R − a²/(2R³) + 3a⁴/(4R⁵) ]      (k → 0)
    G_ek,reg    = (1/4π)·[ (e^{-jkR} − 1)·fac + extra ] / R

`extra = fac − fac_static` is spelled out term by term (`_ek_reg_extra`) so
that no term is a difference of near-equal quantities, and so that a = 0
reduces it to exactly the pre-existing expression rather than to something
that merely rounds to it. The static half's new O(a²) piece is one generated
closed-form family, `D_ek_moment` — one integrand, never 1/R³ and 1/R⁵
separately (see scripts/derive_bspline_static_moments.py for the
cancellation this avoids). The remainder is bounded by |k| on R ≥ a — in
MODULUS, so the statement carries over verbatim to the complex k of the
in-medium kernel below (momwire#553 U1 re-measured `sup_{R≥a} |4π·G_reg|/|k|`
over Δ/a ∈ [0.5, 100]: **1.0000** for the reduced remainder and **1.0000**
for the EK one, at real k₀ and at every #553 SPEC k_m; the pre-#553 note here
said "≈ 3.5k", the same class, just loose). It is resolved by the existing
Gauss–Legendre rule at unchanged `n_qp` (measured: the EK remainder's
quadrature error is within 2–8× the reduced remainder's).

Two honest limits, both measured (momwire#249 design §1.3, §3):

  * EK does NOT fix the self term. The exact tube kernel diverges
    logarithmically as ζ → 0 while EK saturates at 1.25/a, so the diagonal
    MOMENT is only ~10× better than reduced, not the ~10³× the pointwise
    O(a⁴) order suggests.
  * Below Δ/a ≈ 1 EK stops improving the moments, and by Δ/a ≈ 0.5 it is
    *worse* than the reduced kernel on the nearest-neighbour moment. This is
    an engine-free confirmation of the Δ/a ≥ 2 floor #248 established from
    the divergence side.

C++ TWINS (momwire#270)
-----------------------
#249 landed EK as numpy only: every C++ dispatch was guarded by `ek is None`.
#270 adds the accelerated twins, one entry point at a time, each behind its
own `_HAVE_*` capability flag so an older extension still degrades to numpy.
Landed so far (unit 1, the SAME-EDGE kernels):

    _seg_seg_static_moments (uniform edge)  seg_seg_static_moments_bspline_uniform_ek
    _seg_seg_reg_moments_from_geometry      seg_seg_reg_moments_bspline_swept_ek
    ...        _from_geometry_swept         seg_seg_reg_moments_bspline_swept_ek

Unit 2 adds the OFF-EDGE pair: `seg_seg_full_moments_bspline_ek` and
`seg_seg_full_moments_bspline_swept_ek`. Off-edge eligibility is a property
of the (i, j) SEGMENT PAIR rather than of the whole block, so these two
twins additionally take the per-segment coaxial-and-equal-radius group
labels (`group_i`, `group_j`, int64 arrays) that `_ek_pair_mask` consumes on
the numpy side, and apply NEC Eq 89's coaxial factor pair by pair instead of
block-wide.

The EK-off path is untouched, byte for byte: the twins are separate symbols
(and, in C++, separate template instantiations), so the reduced kernels'
signatures and arithmetic are unchanged rather than merely believed to be.
Cross-backend EK comparisons are meaningful only at tolerance — the C++
reduction order is not the einsum's, and the closed forms differ in the last
bits between `np.arcsinh` and `std::asinh` (measured ~1e-15 relative, the same
class the reduced kernels have always had).
"""

from collections import namedtuple

import numpy as np

from ._bspline_ek_moments import D_ek_moment
from ._bspline_static_far import J_static_stable as J_static_moment
from ._quadrature import leggauss
from ._stable import expm1_neg_jkR as _expm1_neg_jkR

from ._accel import acc as _acc
from ._accel import SAME_EDGE_MAX_N_QP as _SAME_EDGE_MAX_N_QP
from ._accel import serves_n_qp as _serves_n_qp

_HAVE_BSPLINE_ACCEL = _acc is not None and hasattr(_acc, "seg_seg_full_moments_bspline")
_HAVE_BSPLINE_STATIC_ACCEL = _acc is not None and hasattr(
    _acc, "seg_seg_static_moments_bspline_uniform"
)
_HAVE_BSPLINE_REG_SWEPT_ACCEL = _acc is not None and hasattr(
    _acc, "seg_seg_reg_moments_bspline_swept"
)
_HAVE_BSPLINE_OFFEDGE_SWEPT_ACCEL = _acc is not None and hasattr(
    _acc, "seg_seg_full_moments_bspline_swept"
)
# The extended-kernel twins of the two SAME-EDGE kernels (momwire#270 unit 1).
# Separate capability flags, following the module's existing one-flag-per-
# symbol pattern: an extension built before #270 lacks these symbols and the
# EK path then falls back to numpy rather than failing to import.
_HAVE_BSPLINE_STATIC_EK_ACCEL = _acc is not None and hasattr(
    _acc, "seg_seg_static_moments_bspline_uniform_ek"
)
_HAVE_BSPLINE_REG_SWEPT_EK_ACCEL = _acc is not None and hasattr(
    _acc, "seg_seg_reg_moments_bspline_swept_ek"
)
# The extended-kernel twins of the two OFF-EDGE kernels (momwire#270 unit 2).
_HAVE_BSPLINE_OFFEDGE_EK_ACCEL = _acc is not None and hasattr(
    _acc, "seg_seg_full_moments_bspline_ek"
)
_HAVE_BSPLINE_OFFEDGE_SWEPT_EK_ACCEL = _acc is not None and hasattr(
    _acc, "seg_seg_full_moments_bspline_swept_ek"
)

# Currently the C++ accelerator has explicit instantiations for D in {1, 2}.
# Extend by adding `seg_seg_full_moments_bspline_kernel<3>(...)` and a switch
# case in src/momwire/_accelerators.cpp.
_HAVE_BSPLINE_OFFEDGE_CPLX_ACCEL = _acc is not None and hasattr(
    _acc, "seg_seg_full_moments_bspline_cplx"
)
_BSPLINE_ACCEL_MAX_D = 2

MAX_D_SUPPORTED = 2


# ----------------------------------------------------------------------
# THE IN-MEDIUM KERNEL: complex k (momwire#553 unit 1)
# ----------------------------------------------------------------------
#
# A segment pair buried in a lossy medium sees the same thin-wire kernel,
# analytically continued to the medium's complex wavenumber
# k_m = k₀·√ε̃, ε̃ = ε_r − jσ/(ωε₀). King & Smith (1981) eqs. (3.1)/(3.5)
# state this outright for the tubular kernel: the functional form is
# unchanged at complex k — no new terms appear — and their k = β − jα with
# e^{+jωt} is momwire's convention already, so k_m enters where k stood.
# What continues with it:
#
#   * the static extraction. King & Smith §7.4 (eqs. 4.1–4.5) puts the
#     kernel's whole coincidence singularity in its REAL part, ~1/R, and
#     that part is k-INDEPENDENT — so `_seg_seg_static_moments` and
#     `D_ek_moment` are k-free closed forms that stay valid verbatim, and
#     the complex-k content lives entirely in the smooth remainder's
#     series 1/R − jk − k²R/2 + jk³R²/6 + … with complex coefficients.
#     `_seg_seg_static_moments` therefore takes no k and needs no widening.
#   * the small-argument expansions. There are none written out here: the
#     remainder is evaluated as the single complex exponential, so the only
#     "expansion" is IEEE's own `exp`.
#   * the extended kernel. `_ek_factor`/`_ek_reg_extra`/`_ek_reg_kernel`
#     are already written in terms of `kr = k·R` and `1j`, so they continue
#     without a character changing (gated, not assumed).
#
# The named TRAP is King & Smith (3.20b,c): a lossy-medium kernel split
# into e^{−αR}cos(βR)/R and e^{−αR}sin(βR)/R. Any code path carrying
# cos(kR)/sin(kR) with a REAL k does NOT generalize by substituting |k|.
# momwire's B-spline fill carries no such split (audited at #553 U1: zero
# occurrences of a trig-of-kR spelling in `_bspline_kernels`,
# `_bspline_static_moments`, `_bspline_ek_moments`), which is exactly why
# the continuation is a dispatch change rather than a rewrite.
#
# DISPATCH RULE — `np.iscomplexobj(k)`, and nothing else.
# One predicate for scalars and for swept k arrays alike (a Python
# `complex`, a `np.complex128`, and a complex-dtype array all answer True;
# a Python float, a `np.float64` and a float64 array all answer False), so
# there is no second spelling to drift. A complex k routes to the numpy
# twins, bypassing every `_HAVE_*` accelerator flag: the C++ kernels take
# `double k` and would truncate. Real k is untouched — same branch, same
# arithmetic, same bytes. A complex k whose imaginary part happens to be
# zero takes the complex (numpy) path by dtype, not by value; it agrees
# with the real path to rounding, and pinning bytes across dtypes is not
# something this module promises anywhere.
#
# CONVENTION — asserted, never repaired. With e^{+jωt}, Im k ≤ 0 is what
# makes e^{−jkR} decay; the opposite branch of the square root gives a
# kernel that grows with distance. `_complex_k` raises rather than
# conjugating, because the branch choice belongs where k_m is DERIVED from
# ε̃ (the precedent is `_sommerfeld._d12`, which conjugates k1 at the point
# of derivation). A conjugation here would silently rescue a caller that
# built the wrong k and leave every other quantity on the wrong branch.
#
# QUADRATURE — `n_qp` does NOT escalate in the medium, and that is a
# measurement (#553 U1 gate G-U1-7), not a hope. The GL rule's difficulty is
# set by |k|·h, and at EQUAL |k|·h the in-medium remainder costs what the
# free-space one costs: n_qp=4 relative error over |k|h ∈ [0.05, 5],
# reduced kernel at Δ/a = 24, is 1.6e-04 → 2.3e-02 at real k₀ and
# 1.6e-04 → 2.7e-02 at the worst SPEC k_m (soil B / 7 MHz, |k_m|/k₀ = 8.92)
# — a spread of ≤1.15×, in both directions. The EK remainder is two to three
# decades below that throughout (worst 3.4e-04 at |k|h = 5), so it never
# sets the rule. What DOES change in the medium is h: a deck meshed against
# the free-space wavelength runs at |k_m|h = |n|·|k₀|h — 8.9× the panel
# phase at soil B / 7 MHz, i.e. ~0.70 rad on a 20-per-quarter-λ₀ mesh,
# where n_qp=4 reads 2.5e-03 instead of 2.5e-04. That is the under-meshing
# U4 fixes by judging segment length against λ_m; it is a MESHING defect,
# not a quadrature-order one, and escalating n_qp here would paper over it.
# (It also could not live here: `n_qp` sizes the k-INDEPENDENT geometry
# `_seg_seg_reg_geometry` builds, which by design never sees k.)
_IM_K_CONVENTION = "e^{+jwt} requires Im k <= 0 so that e^{-jkR} decays"


def _complex_k(k):
    """The complex-k dispatch predicate, plus the convention guard.

    Returns True when `k` (scalar or array) is complex-typed, in which case
    the caller must take the numpy path. Raises when any component has
    Im k > 0 — the growing-exponential branch, which no kernel here will
    evaluate.
    """
    if not np.iscomplexobj(k):
        return False
    if np.any(np.asarray(k).imag > 0.0):
        raise ValueError(
            f"complex k with Im k > 0: {_IM_K_CONVENTION}. Take the other "
            "branch of k_m = k0*sqrt(eps_tilde) where k_m is derived (the "
            "precedent is _sommerfeld._d12's conjugation of k1); this "
            "kernel will not conjugate for you."
        )
    return True


def _refuse_complex_k(k, what):
    """Named refusal for the fill paths momwire#553 unit 1 does NOT widen.

    Unit 1 continues the B-spline moment kernels' numpy twins to complex k
    and nothing else. Every other assembler that could be handed a `k`
    refuses by name here rather than truncating it to its real part, which
    is what `float(k)` / `dtype=float` would have done.
    """
    if np.iscomplexobj(k):
        raise ValueError(
            f"{what} does not serve complex k (an in-medium wavenumber "
            "k_m = k0*sqrt(eps_tilde)): momwire#553 unit 1 widens only the "
            "B-spline moment kernels' numpy path."
        )


def _k_array_asarray(k_array):
    """`np.asarray` for a swept-k vector that cannot silently truncate.

    `np.asarray(k_array, dtype=float)` on a complex array does not raise —
    numpy drops the imaginary part with a ComplexWarning that is a warning,
    not an error, and warnings are filtered. That is the silent-truncation
    class momwire#553 unit 1 kills: complex in, complex out (the numpy
    swept twins serve it); real in, float64 out exactly as before.
    """
    if np.iscomplexobj(k_array):
        return np.asarray(k_array, dtype=np.complex128)
    return np.asarray(k_array, dtype=np.float64)


def _k_scalar(k):
    """One element of a swept-k vector, as the scalar the single-k entry
    points take. `float` for a real k — the pre-#553 spelling and the same
    value — `complex` for an in-medium one, where `float` would raise."""
    if np.iscomplexobj(k):
        return complex(k)
    return float(k)


def _normalize_row_radius(a, n_rows):
    """Normalize the off-edge kernel's `a` argument: scalar, or a per-
    OBSERVER-row (N_i,) array (per-wire radius, stevenmburns/momwire#147).

    Returns a float when the radius is uniform (including a uniform array —
    keeps the scalar code paths bit-identical and C++-servable), else the
    validated (N_i,) float64 array.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    if a_arr.ndim == 0:
        return float(a_arr)
    if a_arr.shape != (n_rows,):
        raise ValueError(
            f"a: expected a scalar or a per-observer-row length-{n_rows} "
            f"array, got shape {a_arr.shape}"
        )
    if np.all(a_arr == a_arr[0]):
        return float(a_arr[0])
    return a_arr


# The extended-kernel spec threaded through every kernel entry point.
#
#   a        EK regularization radius, or None to use the kernel call's own
#            `a` — which is the right answer on every eligible pair by
#            construction (eligibility REQUIRES equal radii, and the
#            off-edge kernel's per-observer-row `a` is already that radius).
#            None is the normal spelling; an explicit value is an override
#            for callers that regularize with something else.
#   group_i  per-observer-segment axis-group labels, or None meaning "every
#            row of this block is eligible" (the same-edge case: one edge is
#            one straight run of one wire at one radius, so its segments are
#            coaxial and equal-radius by construction).
#   group_j  per-source-segment labels, same convention.
#
# A pair is extended iff `group_i[i] == group_j[j]`, which is symmetric in
# (i, j) by construction — the Galerkin symmetry gate stays live as an error
# detector rather than being burnt by the gating rule (momwire#249 §4.1).
_EK = namedtuple("_EK", "a group_i group_j")


def _ek_radius(ek, a):
    """The radius entering the EK factor for a kernel call regularized by `a`."""
    return a if ek.a is None else ek.a


def _ek_factor(R, a, k):
    """NEC Eq 89's `1 + T1·C2 − T2·C1` in the coaxial equal-radius case.

    With the observer on its own wire's surface on the source axis, NEC's
    ρ_eval and its source radius b are both `a`, so T1 = a⁴/(4R⁴) and
    T2 = a²/(2R²) and the whole factor is a function of R alone. Exactly
    1.0 at a = 0, in IEEE and not merely in the limit.
    """
    # Multi-step spelling throughout (momwire#205): a one-expression complex
    # product with a dead operand changes rounding above numpy's
    # temporary-elision threshold, making the fill depend on block size.
    r2 = R * R
    r4 = r2 * r2
    kr = k * R
    kr2 = kr * kr
    a2 = a * a
    a4 = a2 * a2
    c1 = 1.0 + 1j * kr
    c2 = 3.0 * c1 - kr2
    t1 = 0.25 * a4 / r4
    t2 = 0.5 * a2 / r2
    fac = t1 * c2
    fac = fac - t2 * c1
    fac = fac + 1.0
    return fac


def _ek_reg_extra(R, a, k):
    """`fac − fac_static` = T1·(C2 − 3) − T2·(C1 − 1), written out.

    The k → 0 limit of `_ek_factor` is `fac_static = 1 − a²/(2R²) +
    3a⁴/(4R⁴)`, whose moments the generated `D_ek_moment` family carries in
    closed form. What is left for the Gauss–Legendre remainder is this
    difference — spelled as the two surviving terms rather than as a
    subtraction of near-equal factors, so it is exactly `0.0` at a = 0 and
    the remainder collapses term by term onto the reduced kernel's.
    """
    r2 = R * R
    r4 = r2 * r2
    kr = k * R
    kr2 = kr * kr
    a2 = a * a
    a4 = a2 * a2
    t1 = 0.25 * a4 / r4
    t2 = 0.5 * a2 / r2
    # C2 - 3 = 3jkR - (kR)²  and  C1 - 1 = jkR.
    extra = t1 * (3j * kr - kr2)
    extra = extra - t2 * (1j * kr)
    return extra


def _ek_axis_groups(seg_l, seg_r, tangents, seg_a, tol=1e-6):
    """Per-segment coaxial-and-equal-radius eligibility labels.

    Two segments share a label iff they lie on the SAME LINE and have the
    same radius, using NEC's own thresholds:

      * `|t_i · t_j| ≥ 1 − tol` — nec2-1.2.1.2.f:2040-2041, including the
        ABS() there, so an antiparallel collinear pair still counts;
      * `|a_i/a_j − 1| ≤ tol` — f.2042-2043;
      * perpendicular offset between the two axes `≤ tol·a` — momwire's own
        addition. NEC never needs it because its collinearity test is
        applied only to segments already known to share an endpoint; a
        Galerkin fill asks the question of arbitrary pairs, where parallel
        is not the same as coaxial.

    NEC's per-END gating (`SinusoidalSolver._ek_gating`, IND1/IND2) is
    deliberately NOT reused. It reads per-segment neighbour tables the
    B-spline geometry never builds, per-end brackets do not exist in a
    quadrature fill, and a per-source-segment decision would make
    `G(i, j) ≠ G(j, i)` and burn the Galerkin symmetry gate. Against NEC's
    codes this rule:

      IND = 1 (free end)                agrees — coaxial observers extend
      IND = 0 (collinear junction)      agrees — same line, same radius
      IND = 0 (perpendicular ground)    agrees via the image: the mirrored
                                        source of a vertical monopole is
                                        coaxial with, and of equal radius
                                        to, the real wire
      IND = 2 (bend, radius step,       agrees where both ends are reduced;
               K ≥ 3 junction)          strictly MORE conservative on the
                                        cross-arm pairs NEC still extends
                                        (≤ 0.27 % of Z at Δ/a = 2, MEASURED
                                        — momwire#272)

    #249 §4.3 estimated that last cost at "~1 % at Δ/a = 2, and O(h) in the
    refinement limit". momwire#272 measured it by forcing this function to a
    single label — which extends EVERY pair, strictly more than NEC's per-end
    gating does — and diffing against the shipped rule, so the reading is an
    upper bound with no cross-solver noise in it (the straight-wire control
    reads 0.00000 %, by construction: nothing is declined there). Both halves
    of the estimate came back changed:

      the size    ≤ 0.27 % (bent) / 0.21 % (K=3) at Δ/a = 2, four times inside
                  the 1 % the estimate named, and falling to 0.004 % by Δ/a = 25
      the order   O(a), NOT O(h). Refining six-fold at FIXED radius — the only
                  sweep in which h moves and a/λ does not — moves the bound by
                  a few percent of itself, while the radius sweep moves it
                  60-fold. The cost is set by how fat the wire is, not by how
                  finely it is meshed, and it cannot be refined away.

    The practical reading is unchanged, and better: the rule is vindicated at
    any radius a thin-wire code is valid for. Only the variable was wrong.

    seg_l, seg_r: (N, 3) segment endpoints. tangents: (N, 3) unit tangents.
    seg_a: (N,) per-segment radii. Returns an (N,) int64 label array; every
    label is ≥ 0 (a segment is always coaxial with itself), the ≥ 0
    convention leaving room for a future "never extend" marker.

    Cost is O(N·G) with G the number of distinct groups — one pass per
    segment against the existing group representatives, not an O(N²) pair
    scan. G is 1 on a straight wire and small on any real deck.
    """
    seg_l = np.asarray(seg_l, dtype=np.float64)
    seg_r = np.asarray(seg_r, dtype=np.float64)
    tangents = np.asarray(tangents, dtype=np.float64)
    seg_a = np.asarray(seg_a, dtype=np.float64)
    n = seg_l.shape[0]
    centers = 0.5 * (seg_l + seg_r)

    labels = np.full(n, -1, dtype=np.int64)
    rep_c = np.empty((0, 3), dtype=np.float64)
    rep_t = np.empty((0, 3), dtype=np.float64)
    rep_a = np.empty(0, dtype=np.float64)
    for i in range(n):
        if rep_a.size:
            collinear = np.abs(rep_t @ tangents[i]) >= 1.0 - tol
            same_a = np.abs(seg_a[i] / rep_a - 1.0) <= tol
            # Perpendicular component of the center-to-center offset,
            # resolved in each representative's own axis frame.
            dvec = centers[i][None, :] - rep_c
            axial = dvec @ tangents[i]
            perp = dvec - axial[:, None] * tangents[i][None, :]
            coaxial = np.linalg.norm(perp, axis=1) <= tol * rep_a
            hit = np.flatnonzero(collinear & same_a & coaxial)
            if hit.size:
                labels[i] = int(hit[0])
                continue
        labels[i] = rep_a.size
        rep_c = np.vstack([rep_c, centers[i][None, :]])
        rep_t = np.vstack([rep_t, tangents[i][None, :]])
        rep_a = np.append(rep_a, seg_a[i])
    return labels


def _ek_pair_mask(ek, n_i, n_j):
    """The (N_i, N_j) boolean "extend this pair" mask of an `_EK` spec."""
    if ek.group_i is None or ek.group_j is None:
        return np.ones((n_i, n_j), dtype=bool)
    gi = np.asarray(ek.group_i)
    gj = np.asarray(ek.group_j)
    return (gi[:, None] == gj[None, :]) & (gi[:, None] >= 0)


def _seg_seg_static_moments(seg_endpoints, a, max_d, *, ek=None):
    """Closed-form same-edge static-kernel moment integrals.

    seg_endpoints: (N+1,) array of arc lengths along a single straight edge.
    Returns J_static of shape (max_d+1, max_d+1, N, N), with the 1/(4π)
    prefactor folded in.

    Fast path: when the edge segments are uniform-h, J_pq[i, j] depends only
    on (j-i)·h (the integrand is translation-invariant in the arc-length
    direction along the straight edge), so the (N, N) matrix is Toeplitz —
    2N-1 unique values per moment instead of N². At N=21 this is ~10× faster
    than the dense evaluation; at N=81 it's ~40×.

    `ek`: an `_EK` spec turns on the extended kernel's static correction,
    `D_ek_moment`, added moment by moment. A same-edge block is eligible in
    its entirety (one edge is one straight run of one wire at one radius),
    so the spec's group labels are not consulted here. `D` is
    translation-invariant along the edge exactly as `J_static_moment` is, so
    the Toeplitz gather serves it unchanged — in numpy and, since #270, in the
    C++ twin `seg_seg_static_moments_bspline_uniform_ek`, which builds the same
    2N-1 table with `J + D` in each entry. NON-uniform edges take the dense
    numpy branch above under both kernels; there is no C++ path for them.
    """
    if max_d > MAX_D_SUPPORTED:
        raise NotImplementedError(
            f"max_d={max_d}: only {MAX_D_SUPPORTED} pre-derived. Run "
            "scripts/derive_bspline_static_moments.py with a larger MAX_D."
        )
    sl = np.ascontiguousarray(seg_endpoints[:-1], dtype=np.float64)
    sr = np.ascontiguousarray(seg_endpoints[1:], dtype=np.float64)
    N = len(sl)
    h_seg = sr - sl
    n_d = max_d + 1
    inv4pi = 1.0 / (4 * np.pi)

    uniform = N >= 1 and np.allclose(h_seg, h_seg[0], rtol=1e-12, atol=1e-15)
    if not uniform:
        alpha = sl[:, None]
        beta = sr[:, None]
        A = sl[None, :]
        B = sr[None, :]
        out = np.empty((n_d, n_d, N, N), dtype=np.float64)
        for p in range(n_d):
            for q in range(n_d):
                vals = J_static_moment(p, q, alpha, beta, A, B, a)
                if ek is not None:
                    vals = vals + D_ek_moment(
                        p, q, alpha, beta, A, B, _ek_radius(ek, a)
                    )
                out[p, q] = vals * inv4pi
        return out

    # Uniform-h fast paths
    h = float(h_seg[0])
    if _HAVE_BSPLINE_STATIC_ACCEL and max_d <= _BSPLINE_ACCEL_MAX_D and ek is None:
        # C++ inlined sympy-derived closed forms — ~50× faster than numpy
        # because each call escapes per-op dispatch overhead.
        return _acc.seg_seg_static_moments_bspline_uniform(
            float(h), float(a), int(N), int(max_d)
        )
    if (
        ek is not None
        and _HAVE_BSPLINE_STATIC_EK_ACCEL
        and max_d <= _BSPLINE_ACCEL_MAX_D
    ):
        # The EK twin (momwire#270): same Toeplitz table, each entry J + D.
        # Only the UNIFORM edge reaches here — a non-uniform edge returned
        # above on the numpy dense path, which the C++ kernel does not serve
        # for either kernel flavour.
        return _acc.seg_seg_static_moments_bspline_uniform_ek(
            float(h), float(a), int(N), int(max_d), float(_ek_radius(ek, a))
        )

    # numpy Toeplitz fallback: J_pq[i, j] = vals_pq[j - i + (N - 1)]
    delta = np.arange(-(N - 1), N, dtype=np.float64)
    alpha = np.zeros_like(delta)
    beta = np.full_like(delta, h)
    A = delta * h
    B = (delta + 1.0) * h
    j_minus_i = np.arange(N)[None, :] - np.arange(N)[:, None]
    gather_idx = j_minus_i + (N - 1)
    out = np.empty((n_d, n_d, N, N), dtype=np.float64)
    for p in range(n_d):
        for q in range(n_d):
            vals = J_static_moment(p, q, alpha, beta, A, B, a)
            if ek is not None:
                vals = vals + D_ek_moment(p, q, alpha, beta, A, B, _ek_radius(ek, a))
            vals = vals * inv4pi
            out[p, q] = vals[gather_idx]
    return out


def _seg_seg_reg_geometry(seg_endpoints, a, max_d, n_qp, *, ek=None):
    """k-independent precompute for `_seg_seg_reg_moments`.

    Everything in the smooth-kernel moment integral except the `exp(-jkR)`
    phase: the pair-distance table R and the weight-folded local-coordinate
    powers. Hoisting this out of a swept-k loop turns the per-k same-edge
    work into a single `exp(-jkR)` + einsum (see
    `_seg_seg_reg_moments_from_geometry`). Bounded memory — one edge's
    (N·n_qp, N·n_qp) R table at a time, same as the per-k path.

    Returns a dict consumed by `_seg_seg_reg_moments_from_geometry`. The
    extended-kernel spec rides in that dict rather than in the per-k call
    signature: `a` is k-independent geometry, so the EK-aware remainder needs
    nothing per k that the reduced one does not (momwire#249 §5, the
    swept-path row).
    """
    gl_xi, gl_w = leggauss(n_qp)
    t01 = 0.5 * (gl_xi + 1.0)
    w01 = 0.5 * gl_w

    sl = seg_endpoints[:-1]
    h_seg = seg_endpoints[1:] - sl
    N = len(sl)

    s_q = sl[:, None] + t01[None, :] * h_seg[:, None]  # (N, n_qp) global arc
    u_q = (t01[None, :] * h_seg[:, None]) * np.ones((N, 1))  # (N, n_qp) local
    w_q = (w01[None, :] * h_seg[:, None]) * np.ones((N, 1))

    s_flat = s_q.ravel()
    diff = s_flat[:, None] - s_flat[None, :]
    R = np.sqrt(diff * diff + a * a)

    # u^p evaluated at every quadrature node, weight-folded
    u_pow = np.stack([u_q**p for p in range(max_d + 1)], axis=0)  # (max_d+1, N, n_qp)
    wu_pow = w_q[None, :, :] * u_pow

    return {"R": R, "wu_pow": wu_pow, "N": N, "n_qp": n_qp, "a": a, "ek": ek}


def _ek_reg_kernel(R, a, k):
    """`[(e^{-jkR} − 1)·fac + extra] / (4πR)` — the EK smooth-kernel piece.

    At a = 0 this is the reduced `(e^{-jkR} − 1)/(4πR)` term by term, not
    merely to rounding: `fac` is exactly 1.0 and `extra` exactly 0.0.
    """
    # Multi-step (momwire#205), same discipline as `_ek_factor`. The `- 1` is
    # `_expm1_neg_jkR`'s bracket rather than a literal subtraction
    # (momwire#799), which is the reduced branch's rewrite followed one for
    # one — the EK remainder IS that object with a factor on it.
    fac = _ek_factor(R, a, k)
    extra = _ek_reg_extra(R, a, k)
    phase = _expm1_neg_jkR(k, R)
    num = phase * fac
    num = num + extra
    return num / (4 * np.pi * R)


def _seg_seg_reg_moments_from_geometry(geo, k):
    """Per-k smooth-kernel moment block from a `_seg_seg_reg_geometry` dict.

    `k` may be the in-medium complex wavenumber (momwire#553 unit 1); the
    geometry dict is k-independent and so serves both. A complex k takes
    the numpy branches below — the C++ twins are `double`-typed.
    """
    R = geo["R"]
    wu_pow = geo["wu_pow"]
    N = geo["N"]
    n_qp = geo["n_qp"]
    ek = geo.get("ek")
    in_medium = _complex_k(k)
    # Single-k case (the non-swept compute_impedance): the same streaming C++
    # kernel serves it with a length-1 k axis, which we squeeze back off. This
    # is the ~65% of a single d=2 solve that the numpy einsum below otherwise
    # dominates. Bit-close (different reduction order); numpy stays the fallback.
    # The same-edge kernel keeps its OWN ceiling: momwire#762 tiled the six
    # off-edge kernels, not this one, which takes R from a precomputed array
    # and reduces through a nested wu*wu product rather than a flat wuwu row.
    # Without this term `n_qp_pair_same_edge > 8` RAISED rather than falling
    # back — the site momwire#769 missed, because its refusal is worded
    # differently from the off-edge kernels' and a grep for those found it not.
    _se_serves = _serves_n_qp(
        n_qp,
        "same-edge reg",
        eligible=(_HAVE_BSPLINE_REG_SWEPT_ACCEL or _HAVE_BSPLINE_REG_SWEPT_EK_ACCEL)
        and not in_medium,
        cap=_SAME_EDGE_MAX_N_QP,
    )
    if _HAVE_BSPLINE_REG_SWEPT_ACCEL and ek is None and not in_medium and _se_serves:
        return _acc.seg_seg_reg_moments_bspline_swept(
            np.ascontiguousarray(R, dtype=np.float64),
            np.ascontiguousarray(wu_pow, dtype=np.float64),
            np.ascontiguousarray(np.asarray([k], dtype=np.float64)),
        )[0]
    if (
        ek is not None
        and _HAVE_BSPLINE_REG_SWEPT_EK_ACCEL
        and not in_medium
        and _se_serves
    ):
        # The EK twin (momwire#270), same length-1 k axis. Whole-block
        # eligibility, as below — the C++ kernel takes no group labels.
        return _acc.seg_seg_reg_moments_bspline_swept_ek(
            np.ascontiguousarray(R, dtype=np.float64),
            np.ascontiguousarray(wu_pow, dtype=np.float64),
            np.ascontiguousarray(np.asarray([k], dtype=np.float64)),
            float(_ek_radius(ek, geo["a"])),
        )[0]
    if ek is not None:
        # Whole-block eligibility, as in `_seg_seg_static_moments`: a
        # same-edge block is one straight run of one wire at one radius.
        G_ek = _ek_reg_kernel(R, _ek_radius(ek, geo["a"]), k)
        G_ek_block = G_ek.reshape(N, n_qp, N, n_qp)
        return np.einsum("piq,iqjr,Pjr->pPij", wu_pow, G_ek_block, wu_pow)
    # (exp(-jkR) - 1) / (4π R). At R = a small, this is bounded → -jk/(4π) in
    # the a → 0, kR → 0 limit; no quadrature pathology. The remainder is
    # spelled cancellation-free (momwire#799) — the literal subtraction returns
    # its real part to an ABSOLUTE ε, which is 7e-11 relative at kR = 1e-3.
    G_reg = _expm1_neg_jkR(k, R) / (4 * np.pi * R)
    G_block = G_reg.reshape(N, n_qp, N, n_qp)
    # J_reg[p, P, i, j] = sum_{q, r} wu_pow[p, i, q] G[i, q, j, r] wu_pow[P, j, r]
    return np.einsum("piq,iqjr,Pjr->pPij", wu_pow, G_block, wu_pow)


def _seg_seg_reg_moments_from_geometry_swept(geo, k_array, max_chunk_bytes=256 << 20):
    """Batched `_seg_seg_reg_moments_from_geometry` over a vector of k.

    Returns (n_k, max_d+1, max_d+1, N, N). The only k-dependent factor is the
    phase `exp(-jkR)`, so R and the weighted powers are reused across the
    whole sweep and the (q, r) quadrature reduction is done once per edge as a
    single batched einsum instead of n_k small ones.

    The (chunk, N·n_qp, N·n_qp) phase intermediate is the memory hot-spot, so
    k is processed in chunks sized to keep it under `max_chunk_bytes`. The
    returned moment block is n_qp² smaller than that intermediate, so storing
    it for the whole sweep is cheap relative to the per-k full-J the caller
    already materializes.
    """
    R = geo["R"]
    wu_pow = geo["wu_pow"]
    N = geo["N"]
    n_qp = geo["n_qp"]
    n_d = wu_pow.shape[0]
    # NOT `np.asarray(k_array, dtype=float)`: that silently discards the
    # imaginary part of an in-medium sweep (momwire#553 unit 1).
    k_array = _k_array_asarray(k_array)
    n_k = k_array.shape[0]
    ek = geo.get("ek")
    in_medium = _complex_k(k_array)

    # Streaming C++ kernel: evaluates exp(-jkR) once per (iq, jr, k) and
    # accumulates straight into the (n_d, n_d) moment block, so it never
    # materializes the (chunk, N*n_qp, N*n_qp) phase intermediate this numpy
    # path has to chunk under max_chunk_bytes. Bit-close (different reduction
    # order) to the einsum below, which stays as the fallback.
    # The same-edge kernel keeps its OWN ceiling: momwire#762 tiled the six
    # off-edge kernels, not this one, which takes R from a precomputed array
    # and reduces through a nested wu*wu product rather than a flat wuwu row.
    # Without this term `n_qp_pair_same_edge > 8` RAISED rather than falling
    # back — the site momwire#769 missed, because its refusal is worded
    # differently from the off-edge kernels' and a grep for those found it not.
    _se_serves = _serves_n_qp(
        n_qp,
        "same-edge reg",
        eligible=(_HAVE_BSPLINE_REG_SWEPT_ACCEL or _HAVE_BSPLINE_REG_SWEPT_EK_ACCEL)
        and not in_medium,
        cap=_SAME_EDGE_MAX_N_QP,
    )
    if _HAVE_BSPLINE_REG_SWEPT_ACCEL and ek is None and not in_medium and _se_serves:
        return _acc.seg_seg_reg_moments_bspline_swept(
            np.ascontiguousarray(R, dtype=np.float64),
            np.ascontiguousarray(wu_pow, dtype=np.float64),
            np.ascontiguousarray(k_array, dtype=np.float64),
        )
    if (
        ek is not None
        and _HAVE_BSPLINE_REG_SWEPT_EK_ACCEL
        and not in_medium
        and _se_serves
    ):
        # The EK twin (momwire#270). It streams the same way, so the EK sweep
        # no longer materializes the chunked phase intermediate either.
        return _acc.seg_seg_reg_moments_bspline_swept_ek(
            np.ascontiguousarray(R, dtype=np.float64),
            np.ascontiguousarray(wu_pow, dtype=np.float64),
            np.ascontiguousarray(k_array, dtype=np.float64),
            float(_ek_radius(ek, geo["a"])),
        )

    out = np.empty((n_k, n_d, n_d, N, N), dtype=np.complex128)
    bytes_per_k = R.size * 16  # complex128 phase table for one k
    chunk = max(1, int(max_chunk_bytes // max(bytes_per_k, 1)))
    inv4pi_R = 1.0 / (4 * np.pi * R)
    a_ek = None if ek is None else _ek_radius(ek, geo["a"])
    for c0 in range(0, n_k, chunk):
        kk = k_array[c0 : c0 + chunk]
        if ek is None:
            G = _expm1_neg_jkR(kk[:, None, None], R[None, :, :]) * inv4pi_R[None, :, :]
        else:
            G = _ek_reg_kernel(R[None, :, :], a_ek, kk[:, None, None])
        G_block = G.reshape(kk.shape[0], N, n_qp, N, n_qp)
        out[c0 : c0 + chunk] = np.einsum(
            "piq,kiqjr,Pjr->kpPij", wu_pow, G_block, wu_pow, optimize=True
        )
    return out


def _seg_seg_reg_moments(seg_endpoints, a, k, max_d, n_qp, *, ek=None):
    """Smooth-kernel piece (exp(-jkR) - 1)/(4π R) over polynomial moments
    on every same-edge segment pair, via Gauss-Legendre quadrature.

    seg_endpoints: (N+1,) array of arc lengths along a single straight edge.
    a, k: regularization radius and wavenumber.
    max_d: maximum moment degree (inclusive).
    n_qp: Gauss-Legendre nodes per segment per axis.

    Returns J_reg of shape (max_d+1, max_d+1, N, N) complex.
    """
    geo = _seg_seg_reg_geometry(seg_endpoints, a, max_d, n_qp, ek=ek)
    return _seg_seg_reg_moments_from_geometry(geo, k)


def _seg_seg_full_moments_offedge(
    seg_l_i, seg_r_i, seg_l_j, seg_r_j, a, k, max_d, n_qp, *, ek=None
):
    """Full-kernel moment integrals on all segment pairs.

    Returns (max_d+1, max_d+1, N_i, N_j) complex. Uses the C++ accelerator
    when available and `max_d` is in the instantiation set; otherwise falls
    back to the pure-numpy reference.

    The same a² wire-radius regularization handles diagonals (i = j) where
    R could otherwise vanish, and touching segments at kink corners; the
    bspline solver overwrites the same-edge blocks with analytic_static +
    GL_reg afterwards, so the accuracy of this call only needs to be good
    on far / nearly-far pairs.

    `a` is a scalar, or a per-OBSERVER-row (N_i,) array for mixed per-wire
    radii (momwire#147): the regularization represents the boundary
    condition on the observing wire's surface, so each observer row uses
    its own wire's radius — the same observer-side convention the
    sinusoidal solver oracle-validated against NEC's EFLD. A mixed-radius
    array is served by the C++ kernel one constant-radius row-run at a
    time (segments are laid out wire-contiguously, so runs are few).

    `ek`: an `_EK` spec extends the kernel on the pairs its labels declare
    coaxial-and-equal-radius, `group_i[i] == group_j[j]`, and leaves every
    other pair reduced. Eligible pairs have equal radii by definition, so
    the per-observer-row `a` already IS the EK radius and the mixed-radius
    plumbing needs no change. Since momwire#270 unit 2 the C++ paths
    (including the constant-radius run-splitting) serve `ek` too, via the
    `seg_seg_full_moments_bspline_ek` twin — a spec with `group_i`/`group_j`
    unset (whole-block eligibility, the same-edge convention) still falls
    back to numpy here, since the off-edge twin transcribes the PAIR mask
    rather than the whole-block case.
    """
    a = _normalize_row_radius(a, np.asarray(seg_l_i).shape[0])
    in_medium = _complex_k(k)
    gl_xi, gl_w = leggauss(n_qp)
    t01 = 0.5 * (gl_xi + 1.0)
    w01 = 0.5 * gl_w

    # `n_qp` belongs in these predicates for the same reason `max_d` does: it
    # is a shape the kernels do not serve, and the numpy path does
    # (momwire#769). Split from the rest so the warning can tell "the ceiling
    # moved this work" from "it was going to numpy anyway".
    # momwire#778: a complex (in-medium) k is served by the plain off-edge
    # kernel's COMPLEX_K instantiation, which factors exp(-jkR) into the real
    # decay exp(Im(k)*R) times the existing real-lane sincos. Only the PLAIN
    # kernel is widened so far — the EK and swept twins still take the numpy
    # route, which is why `_shape_ok_ek` below keeps its `not in_medium`.
    _cplx_ok = in_medium and _HAVE_BSPLINE_OFFEDGE_CPLX_ACCEL
    _shape_ok = (
        _HAVE_BSPLINE_ACCEL
        and max_d <= _BSPLINE_ACCEL_MAX_D
        and ek is None
        and (not in_medium or _cplx_ok)
    )
    _shape_ok_ek = (
        ek is not None
        and ek.group_i is not None
        and ek.group_j is not None
        and _HAVE_BSPLINE_OFFEDGE_EK_ACCEL
        and not in_medium
        # max_d=0 has no C++ template instantiation (only D in {1, 2} are
        # instantiated for either kernel flavour) — the reduced dispatch
        # above shares this floor too, just never exercises it: `ek is not
        # None` always forced numpy pre-#270-unit-2, so no caller ever hit
        # max_d=0 on the accel-eligible branch until this one.
        and 1 <= max_d <= _BSPLINE_ACCEL_MAX_D
    )
    _serves = _serves_n_qp(n_qp, "off-edge pair", eligible=_shape_ok or _shape_ok_ek)
    accel_ok = _shape_ok and _serves
    accel_ok_ek = _shape_ok_ek and _serves
    if (accel_ok or accel_ok_ek) and not np.ndim(a) == 0:
        # Mixed per-row radii on the C++ path: dispatch one call per
        # contiguous run of equal radius and stitch along the row axis.
        # Under EK, `group_i` is sliced the same way (momwire#270 unit 2):
        # eligible pairs already share a radius by construction, so a run
        # boundary — drawn purely from `a` — never splits a coaxial group.
        bounds = np.flatnonzero(np.diff(a)) + 1
        starts = np.concatenate(([0], bounds))
        stops = np.concatenate((bounds, [a.shape[0]]))
        return np.concatenate(
            [
                _seg_seg_full_moments_offedge(
                    seg_l_i[s:e],
                    seg_r_i[s:e],
                    seg_l_j,
                    seg_r_j,
                    float(a[s]),
                    k,
                    max_d,
                    n_qp,
                    ek=(
                        _EK(a=ek.a, group_i=ek.group_i[s:e], group_j=ek.group_j)
                        if ek is not None
                        else None
                    ),
                )
                for s, e in zip(starts, stops)
            ],
            axis=2,
        )

    if accel_ok:
        # `float(k)` would silently truncate an in-medium wavenumber, which is
        # the whole hazard class momwire#553 U1 named — so the complex branch
        # is a different ENTRY POINT taking `complex(k)`, not a widened arg.
        _fn = (
            _acc.seg_seg_full_moments_bspline_cplx
            if in_medium
            else _acc.seg_seg_full_moments_bspline
        )
        return _fn(
            np.ascontiguousarray(seg_l_i, dtype=np.float64),
            np.ascontiguousarray(seg_r_i, dtype=np.float64),
            np.ascontiguousarray(seg_l_j, dtype=np.float64),
            np.ascontiguousarray(seg_r_j, dtype=np.float64),
            float(a) * float(a),
            complex(k) if in_medium else float(k),
            int(max_d),
            np.ascontiguousarray(t01, dtype=np.float64),
            np.ascontiguousarray(w01, dtype=np.float64),
        )

    if accel_ok_ek:
        # The C++ EK twin (momwire#270 unit 2): the reduced kernel's own
        # geometry contract, plus the per-segment group labels and the
        # plain (unsquared) EK radius — `a` here is already scalar (the
        # mixed-radius branch above recurses one constant-radius run at a
        # time before reaching this point).
        return _acc.seg_seg_full_moments_bspline_ek(
            np.ascontiguousarray(seg_l_i, dtype=np.float64),
            np.ascontiguousarray(seg_r_i, dtype=np.float64),
            np.ascontiguousarray(seg_l_j, dtype=np.float64),
            np.ascontiguousarray(seg_r_j, dtype=np.float64),
            float(a) * float(a),
            float(k),
            int(max_d),
            np.ascontiguousarray(t01, dtype=np.float64),
            np.ascontiguousarray(w01, dtype=np.float64),
            np.ascontiguousarray(ek.group_i, dtype=np.int64),
            np.ascontiguousarray(ek.group_j, dtype=np.int64),
            float(_ek_radius(ek, a)),
        )

    len_i = np.linalg.norm(seg_r_i - seg_l_i, axis=1)
    len_j = np.linalg.norm(seg_r_j - seg_l_j, axis=1)

    pos_i = (1 - t01[None, :, None]) * seg_l_i[:, None, :] + t01[
        None, :, None
    ] * seg_r_i[:, None, :]
    pos_j = (1 - t01[None, :, None]) * seg_l_j[:, None, :] + t01[
        None, :, None
    ] * seg_r_j[:, None, :]

    u_i = t01[None, :] * len_i[:, None]
    u_j = t01[None, :] * len_j[:, None]
    w_i = w01[None, :] * len_i[:, None]
    w_j = w01[None, :] * len_j[:, None]

    diff = pos_i[:, :, None, None, :] - pos_j[None, None, :, :, :]
    a2 = a * a if np.ndim(a) == 0 else (a * a)[:, None, None, None]
    R = np.sqrt((diff * diff).sum(-1) + a2)
    G = np.exp(-1j * k * R) / (4 * np.pi * R)
    if ek is not None:
        a_ek = _ek_radius(ek, a)
        a_b = a_ek if np.ndim(a_ek) == 0 else a_ek[:, None, None, None]
        mask = _ek_pair_mask(ek, R.shape[0], R.shape[2])
        fac = np.where(mask[:, None, :, None], _ek_factor(R, a_b, k), 1.0)
        G = G * fac

    u_pow_i = np.stack([u_i**p for p in range(max_d + 1)], axis=0)
    u_pow_j = np.stack([u_j**p for p in range(max_d + 1)], axis=0)
    wu_i = w_i[None, :, :] * u_pow_i
    wu_j = w_j[None, :, :] * u_pow_j

    return np.einsum("piq,iqjr,Pjr->pPij", wu_i, G, wu_j)


def _seg_seg_full_moments_offedge_swept(
    seg_l_i, seg_r_i, seg_l_j, seg_r_j, a, k_array, max_d, n_qp, *, ek=None
):
    """Batched-over-k `_seg_seg_full_moments_offedge`.

    Returns (n_k, max_d+1, max_d+1, N_i, N_j) complex. The per-(i,j) geometry
    is k-independent, so the C++ kernel builds it once and reuses it across the
    whole sweep — the off-edge analog of the swept reg-moment kernel. Lets
    `compute_impedance_swept` build the all-pairs off-edge moments in one call
    instead of one single-k call per frequency. Falls back to stacking the
    single-k path when the accelerator is unavailable.

    `a` is a scalar or a per-observer-row (N_i,) array — same contract and
    per-run C++ dispatch as `_seg_seg_full_moments_offedge` (axis 3 here).
    `ek` likewise: same spec and the same per-run dispatch; since
    momwire#270 unit 2 a spec with concrete `group_i`/`group_j` reaches the
    batched `seg_seg_full_moments_bspline_swept_ek` twin instead of falling
    back to stacking single-k calls. A spec with unset group labels (or no
    C++ EK twin available) still falls back to that stack.
    """
    # NOT `np.asarray(k_array, dtype=np.float64)` — see `_k_array_asarray`.
    k_array = _k_array_asarray(k_array)
    in_medium = _complex_k(k_array)
    a = _normalize_row_radius(a, np.asarray(seg_l_i).shape[0])
    _shape_ok = (
        _HAVE_BSPLINE_OFFEDGE_SWEPT_ACCEL
        and max_d <= _BSPLINE_ACCEL_MAX_D
        and ek is None
        and not in_medium
    )
    _shape_ok_ek = (
        ek is not None
        and ek.group_i is not None
        and ek.group_j is not None
        and _HAVE_BSPLINE_OFFEDGE_SWEPT_EK_ACCEL
        and not in_medium
        # Same max_d=0 floor as the single-k twin above.
        and 1 <= max_d <= _BSPLINE_ACCEL_MAX_D
    )
    _serves = _serves_n_qp(
        n_qp, "swept off-edge pair", eligible=_shape_ok or _shape_ok_ek
    )
    accel_ok = _shape_ok and _serves
    accel_ok_ek = _shape_ok_ek and _serves
    if (accel_ok or accel_ok_ek) and not np.ndim(a) == 0:
        bounds = np.flatnonzero(np.diff(a)) + 1
        starts = np.concatenate(([0], bounds))
        stops = np.concatenate((bounds, [a.shape[0]]))
        return np.concatenate(
            [
                _seg_seg_full_moments_offedge_swept(
                    seg_l_i[s:e],
                    seg_r_i[s:e],
                    seg_l_j,
                    seg_r_j,
                    float(a[s]),
                    k_array,
                    max_d,
                    n_qp,
                    ek=(
                        _EK(a=ek.a, group_i=ek.group_i[s:e], group_j=ek.group_j)
                        if ek is not None
                        else None
                    ),
                )
                for s, e in zip(starts, stops)
            ],
            axis=3,
        )
    if accel_ok:
        gl_xi, gl_w = leggauss(n_qp)
        t01 = 0.5 * (gl_xi + 1.0)
        w01 = 0.5 * gl_w
        return _acc.seg_seg_full_moments_bspline_swept(
            np.ascontiguousarray(seg_l_i, dtype=np.float64),
            np.ascontiguousarray(seg_r_i, dtype=np.float64),
            np.ascontiguousarray(seg_l_j, dtype=np.float64),
            np.ascontiguousarray(seg_r_j, dtype=np.float64),
            float(a) * float(a),
            np.ascontiguousarray(k_array),
            int(max_d),
            np.ascontiguousarray(t01, dtype=np.float64),
            np.ascontiguousarray(w01, dtype=np.float64),
        )
    if accel_ok_ek:
        gl_xi, gl_w = leggauss(n_qp)
        t01 = 0.5 * (gl_xi + 1.0)
        w01 = 0.5 * gl_w
        return _acc.seg_seg_full_moments_bspline_swept_ek(
            np.ascontiguousarray(seg_l_i, dtype=np.float64),
            np.ascontiguousarray(seg_r_i, dtype=np.float64),
            np.ascontiguousarray(seg_l_j, dtype=np.float64),
            np.ascontiguousarray(seg_r_j, dtype=np.float64),
            float(a) * float(a),
            np.ascontiguousarray(k_array),
            int(max_d),
            np.ascontiguousarray(t01, dtype=np.float64),
            np.ascontiguousarray(w01, dtype=np.float64),
            np.ascontiguousarray(ek.group_i, dtype=np.int64),
            np.ascontiguousarray(ek.group_j, dtype=np.int64),
            float(_ek_radius(ek, a)),
        )
    return np.stack(
        [
            _seg_seg_full_moments_offedge(
                seg_l_i,
                seg_r_i,
                seg_l_j,
                seg_r_j,
                a,
                # `float(k)` pre-#553; `_k_scalar` returns the identical
                # float for a real sweep and a `complex` for an in-medium
                # one, where `float()` would raise TypeError.
                _k_scalar(k),
                max_d,
                n_qp,
                ek=ek,
            )
            for k in k_array
        ],
        axis=0,
    )
