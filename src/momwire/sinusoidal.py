"""NEC2-style sinusoidal-basis MoM for wires (Section III of the NEC2
Theory Manual, Burke & Poggio 1981 — see `docs/sinusoidal_basis_design.md`).

This is an OPTIONAL solver alongside `BSplineSolver` (the default). The
sinusoidal basis is what NEC2 / PyNEC / nec2c use; reproducing it in momwire
lets us isolate which parts of NEC's pulse-basis convergence behaviour are
intrinsic to the basis itself versus its kernel / source / junction
treatment.

Scope (deliberately narrow):
  * Free space, PEC-image ground (`ground_z`), NEC-style reflection-
    coefficient finite ground (`ground_z` + `ground_eps`, matching NEC's
    IPERF=0 approximation), or Sommerfeld/Norton finite ground
    (`ground_model="sommerfeld"`, the same C2-image + interpolated
    smooth-remainder decomposition BSplineSolver uses — see
    docs/sommerfeld-everywhere-plan.md Phase 2).
  * Thin-wire kernel (Eqs 73-79), no extended thin-wire / current-element.
  * Delta-gap "applied-E" source (Eq 187) on a single basis function.
  * Per-wire radius (stevenmburns/momwire#147): `wire_radius` is a scalar
    or a length-n_wires sequence. Mixed radii follow NEC2's per-segment
    convention — the basis end-condition constants use each segment's own
    radius (nec2c TBF), and the thin-wire kernel keeps the source current
    a filament on the source axis while enforcing the boundary condition
    on the OBSERVER segment's surface (EFLD's rh = sqrt(rho² + a_obs²)).
    The scalar-radius C++ field-tensor kernels serve mixed radii one
    constant-radius observer-row run at a time (`_radius_runs`).
  * Free wire ends use the X_i = 0 zero-current condition (the more
    physical J_1/J_0 end-cap condition is negligible for thin wires).
    A wire end lying IN an active ground plane is instead ground-
    connected (#151): NEC's tbf ground path — the plane side's P-sum
    gains the segment's own atom and the image-side extension folds
    back onto the segment with the sin term mirrored, so basis + image
    carry current continuously through the plane. Junctions AT the
    plane follow NEC's conect(): each member is ground-connected
    independently, with no inter-wire junction entries.
"""

import math
from collections import namedtuple
from dataclasses import dataclass


import numpy as np
import scipy.linalg
import scipy.sparse

from . import (
    _feed_snap,
    _field_ground,
    _ground_mirror,
    _ground_refl,
    _ground_spec,
    _sommerfeld,
    _wire_loading,
    _wire_spec,
)
from .bspline import SINGULAR_ENRICHMENT_NEVER
from ._accel import acc as _acc
from ._cancel import _Cancelable
from ._capabilities import Capabilities
from ._element_currents import _ElementCurrents
from ._port_solution import PortSolution, _SweptPortSolutions
from ._stable import asinh_diff
from ._stable import expm1_neg_j as _expm1_neg_j
from ._stable import expm1_neg_j_from_half as _expm1_neg_j_from_half

_HAVE_FIELD_TENSOR = _acc is not None and hasattr(_acc, "sinusoidal_field_tensor")
_HAVE_FIELD_TENSOR_REFL = _acc is not None and hasattr(
    _acc, "sinusoidal_field_tensor_refl"
)
# momwire#258 changed both extended-kernel entry points' signatures: EKSCX's
# IRA is now decided per source-observer PAIR inside the kernel, so they no
# longer take the build-wide `want_swapped` scalar #245 passed. A stale
# extension still EXPORTS both symbols under the old arity, and calling one
# with the new argument list is a TypeError rather than a graceful fallback —
# so presence of the symbol is no longer enough to claim the capability. The
# module-level flag below is what says "this build speaks the per-pair
# signature"; without it the EK paths take the numpy reference, which carries
# the same per-pair fix.
_EK_IRA_PER_PAIR = _acc is not None and bool(getattr(_acc, "ek_ira_per_pair", False))
_HAVE_FIELD_TENSOR_EK = (
    _acc is not None
    and hasattr(_acc, "sinusoidal_field_tensor_ek")
    and _EK_IRA_PER_PAIR
)
_HAVE_FIELD_TENSOR_EK_REFL = (
    _acc is not None
    and hasattr(_acc, "sinusoidal_field_tensor_ek_refl")
    and _EK_IRA_PER_PAIR
)

_EULER_GAMMA = 0.5772156649015329

# Degenerate-geometry guard for the #282 contact-charge kernel: a
# collocation point sitting exactly ON the contact node (a zero-length
# segment) would otherwise divide by zero building p̂.
_CONTACT_TINY = 1e-30

# Threshold for dense vs sparse assembly in `_assemble_Z`. Below this N
# the BLAS overhead on a tiny matrix loses to dense matmul; above it the
# O(N³) zgemm cost on a mostly-zero matrix loses to CSC sparse matmul.
# Measured crossover on Kaby Lake R / OpenBLAS-pthreads ≈ 60.
_DENSE_ASSEMBLY_THRESHOLD = 60

# kΔ below which `_assemble_Z` switches to the well-scaled shape set
# {1, sin kξ, cos kξ − 1} instead of NEC's literal {1, sin kξ, cos kξ}
# (stevenmburns/momwire#606).
#
# The two are the same fill in exact arithmetic. In float64 the literal one
# forms `Φ_c @ M_A + Φ_co @ M_C` where A ≈ −C and Φ_co → Φ_c as k → 0, so an
# O((kΔ)²) answer comes out of O(1) terms: a relative error of ~8ε/(kΔ)²,
# measured at exactly that scaling over four decades on the #606 deck.
#
# The threshold is set where that error crosses ~1e-9, because 1e-9 is where
# the OTHER factor starts to matter: an electrically tiny structure carries
# cond(Z) ≳ 1e7 (measured 8.6e7 on the #606 deck), and 1e-9 × 1e8 is a
# percent. Solving 8ε/(kΔ)² = 1e-9 gives kΔ = 1.3e-3; rounded to 1.5e-3.
#
# Deliberately NOT set at the merely-visible boundary. A half-wave dipole at
# N=801 sits at kΔ = 3.8e-3 and carries the 1.2e-10 fill error #203 measured
# and accepted; its cond is ~1e4, so that error is worth 1e-6 in the answer
# and the literal spelling is fine. Pulling such a deck onto the well-scaled
# path would cost it the C++ kernel — 70 % of a single-k solve at N ≳ 80 —
# to fix digits nothing was reading.
#
# Above the threshold nothing changes: the literal path keeps NEC's spelling
# and the C++ kernels, and is bit-identical to the pre-#606 fill. Below it the
# well-scaled path costs the accelerator (the kernels transcribe literal-cos
# only), which is affordable exactly where it is engaged — electrically tiny
# structures are small-N by construction.
_WELL_SCALED_KD = 1.5e-3

# Complex entries per observer chunk of the Sommerfeld remainder evaluator
# (`_field_tensor_sommerfeld_remainder`): the grid interpolation's block is
# (rows, N·n_qp_sommerfeld), so this caps it at 8 MB whatever the mesh.
# Named rather than inline since momwire#332 unit D gave the evaluator a
# streaming outlet — the chunk is now the granularity a consumer reduces at,
# so a gate has to be able to shrink it. Value unchanged.
_REMAINDER_CHUNK_ELEMS = 1 << 19

# The extended-kernel payload `_field_components_bcast` takes on its GALERKIN
# path (momwire#246), as distinct from the point-matched path's plain
# `(src_a, ind1, ind2)` triple:
#
#   src_a     the source segments' radius — the `a` of NEC Eq 89's factor,
#             broadcastable against the field tables;
#   eligible  a boolean mask, broadcastable the same way, that is True on the
#             pairs the SYMMETRIC pair rule extends (coaxial + equal radius,
#             momwire#249 §4), or None meaning "every pair in this block".
#
# The two payloads are different types on purpose: the point-matched triple
# carries NEC's PER-END IND1/IND2 gating codes, which have no meaning under a
# pair rule (per-end gating is not symmetric in i↔j, and ‖G−Gᵀ‖/‖G‖ is a
# load-bearing invariant of the Galerkin fill). Handing either payload to the
# other path raises rather than silently serving the wrong tables.
#
# `n_panels` picks the delta quadrature's density (see `_ek_delta_rule`): 1,
# the default, is the FAR tier, and the Galerkin near-pair path raises it.
_EKPairs = namedtuple("_EKPairs", "src_a eligible n_panels", defaults=(1,))

# Gauss-Legendre nodes per panel for the EK delta quadrature (momwire#246).
#
# The delta kernel is analytic on the integration path — its nearest
# singularity is the reduced kernel's own ζ = ±jρ, off the real axis by a full
# wire radius, and at ζ = 0 itself it is merely ¼·G_red(a). But "off the real
# axis by ρ" is the whole story only once the path is measured in units of ρ:
# in ξ the feature is a spike of WIDTH ρ inside a segment of half-length H, so
# a plain rule's accuracy is set by ρ/H and collapses on exactly the pairs a
# fill cares most about — the self pair and its neighbours, where the observer
# sits ON the source segment. `_folded_ek_delta_fields` therefore integrates in
# the sinh-mapped variable ζ = ρ·sinh t, in which R = ρ·cosh t and the spike is
# O(1) wide whatever ρ/H is; what is left is a bounded interval whose half
# width grows only like ln(2H/ρ), covered by splitting it into `n_panels`
# equal panels of this many nodes each.
_N_QP_EK_DELTA = 16

# Panels for the NEAR tier — the pairs whose observer can sit on or beside the
# source segment. One panel per unit of mapped half-width is what the measured
# box needs (Δ/a from 1 to 500 at ≤2e-10 of the reduced const field, 8 panels);
# far pairs, whose mapped interval is short and holds no spike at all, are
# converged on one panel and take the default.
_N_PANEL_EK_DELTA_NEAR = 8

# Reused by `_reject_junction_ports` (the raise) and `capabilities.refusals`
# below — one message, not a copy in each.
_JUNCTION_PORTS_REFUSAL = (
    "junction_ports are not supported on SinusoidalSolver: the "
    "sinusoidal basis enforces KCL identically, so a node-current "
    "port is outside its span (momwire#177; same limitation as "
    "NEC-2). Use BSplineSolver, or do what NEC requires: mesh a "
    "short bridge wire across the gap and gap-feed it"
)

# The `buried` cell for both classes in this family (momwire#792).
# `_build_geometry`'s scan raised the geometry line and no reason at all —
# "wire 0 dips below the ground plane (min z = -1 < ground_z = 0)" — which
# told a caller what it had drawn and nothing about why it was refused. The
# sentence is appended there and declared here; the raise's geometry preamble
# is unchanged.
#
# ONE cell and no combination keys, unlike `BSplineSolver`'s and
# `RazorSolver`'s rows: this family's scan is coarser than
# `_medium_spec.wire_media`. It asks only "is any point below the plane", so
# a mid-span CROSSING and a buried wire under a ground with no lower medium
# both arrive at this same sentence — and the sentence is true of all three,
# because nothing in this formulation fills below the interface under any
# ground.
_BURIED_REFUSAL = (
    "{cls} has no buried fill. The momwire#553 buried serve - a direct, an "
    "image and a Sommerfeld-remainder block evaluated in the lower medium at "
    "k_m = k0*sqrt(eps_tilde) - is written for BSplineSolver's testing side "
    "only, and this family has no in-medium kernel at all: every fill here "
    "takes the free-space wavenumber and reaches the ground through an image "
    "or a reflection weight above the interface. A wire below the plane is a "
    "LEGAL deck - solve it with BSplineSolver over ground_model='sommerfeld' "
    "- or raise the wire clear of the plane"
)

# No node_gaps kwarg exists on this solver at all (unlike BSplineSolver /
# SinusoidalGalerkinSolver) — passing one is a plain TypeError, not a
# NotImplementedError, so there is no raise to reuse this from.
_NODE_GAPS_REFUSAL = (
    "node_gaps are not accepted by SinusoidalSolver: the point-matched "
    "basis has no node-gap treatment (see SinusoidalGalerkinSolver or "
    "BSplineSolver, which both do)"
)

# `_build_geometry` resolves a `feeds` arclength to the nearest segment
# CENTRE, so a caller that names a knot is answered half a segment away
# without anything being raised (momwire#611).  There is no constructor
# refusal to reuse this from, because the snap is not an error in this
# family's own terms: the segment gap IS the point gap at the only
# resolution collocation has (`_reject_point_feed_model`, momwire#212).
#
# What it is not is the KNOT gap a node-addressing dialect asks for, and
# under point matching there is nothing to offer instead.  TWO DIFFERENT
# failures live here and it is worth keeping them apart, because only one
# of them is this cell's:
#
#   * a zero-width gap AT a match point — `feed_model="point"` on this
#     class — is δ(s_m − s₀) with s₀ = s_m, so δ·δ and the pairing is
#     undefined.  That is `_reject_point_feed_model`'s (momwire#212).
#   * a zero-width gap at a KNOT is not at a match point at all: the match
#     points ARE the segment centres and a knot is a segment boundary, so
#     E_app(s_m) = V·δ(s_m − s₀) = 0 in EVERY row and the RHS is the
#     unexcited problem.  That is momwire#177's observation — the one
#     `SinusoidalGalerkinSolver._node_cut_vectors` quotes as "point-samples
#     to nothing at any collocation point" — and it is this cell's.
#
# Either way the refusal is PERMANENT for this class, but a reader who is
# told δ·δ will go looking for a regularization, and #212 §17 already ran
# every one of those to its dead end.  Zero rows are not a regularization
# problem; they are the absence of an excitation.
#
# The Galerkin subclass overrides this with its own (momwire#648), because
# its pairing collapses the same delta to −V·f_i(s₀) and only the plumbing
# is missing.
_KNOT_FEEDS_REFUSAL = (
    "knot feeds are not served by SinusoidalSolver: a delta gap in this "
    "family lands on the nearest segment CENTRE, so a source named at a "
    "knot is solved half a segment away from where it was asked for. "
    "Point matching admits no gap at a knot to offer instead — the match "
    "points are the segment centres, so a delta at a knot point-samples to "
    "nothing in every row and the RHS is the unexcited problem "
    "(momwire#177). Use BSplineSolver or RazorSolver, whose gaps land on "
    "the knot named"
)


def _sin_minus_arg(u):
    """sin(u) − u to full relative precision.

    The literal difference loses everything below u²/6 relative, which is the
    whole answer for the segment half-angles this solver works at (u = kΔ/2 ≈
    1.9e-3 at N=801 on the half-wave dipole → ε·6/u² = 3.6e-10 relative). The
    Taylor series has no cancellation at all — every term is smaller than the
    last — so it is exact where the subtraction is worst; above u ≈ 0.1 the
    subtraction costs at most 6ε/u² = 1.3e-13 and the series would need more
    terms, so that is where the two swap.
    """
    u = np.asarray(u, dtype=float)
    u2 = u * u
    series = (
        -(u * u2)
        / 6.0
        * (1.0 - u2 / 20.0 * (1.0 - u2 / 42.0 * (1.0 - u2 / 72.0 * (1.0 - u2 / 110.0))))
    )
    return np.where(np.abs(u) < 0.1, series, np.sin(u) - u)


def _recip_sin_gap(kd):
    """1/sin(kΔ) − 1/(2 sin(kΔ/2)) to full relative precision (momwire#606).

    This difference is the whole of `A + C` on a neighbour entry, and on the
    ground-junction extension that reuses the same shape. Both terms are
    O(1/kΔ) and they agree to O(kΔ), so the literal subtraction returns an
    answer of size ~kΔ/8 out of terms of size 1/kΔ — a relative error of
    ~8ε/(kΔ)², which is 44000 % at kΔ = 1e-5 (issue #606's ladder).

    The half-angle identity sin kΔ = 2 sin(kΔ/2) cos(kΔ/2) collapses it to

        1/sin kΔ − 1/(2 sin(kΔ/2))
            = [1/cos(kΔ/2) − 1] / (2 sin(kΔ/2))
            = sin²(kΔ/4) / (sin(kΔ/2) cos(kΔ/2))

    in which no term is larger than the answer, so it is correct to a
    relative ε at every kΔ this solver meets. Same trick, and same reason, as
    `_sin_minus_arg` and `_asinh_minus_arg` above: never subtract two things
    that are about to agree.
    """
    kd = np.asarray(kd, dtype=float)
    s4 = np.sin(0.25 * kd)
    return s4 * s4 / (np.sin(0.5 * kd) * np.cos(0.5 * kd))


def _asinh_minus_arg(x):
    """asinh(x) − x to full relative precision (momwire#205).

    Substituting x = sinh t turns the answer into −(sinh t − t), whose series
    converges factorially — nine terms cover |t| ≤ 1 to below ε, where the
    asinh series itself would need ~28 — and t = asinh(x) is exactly what
    libm gives to a relative ε. Perturbing t by ε·t moves sinh t − t ≈ t³/6 by
    (t²/2)·εt = 3ε of itself, so routing through t costs nothing.

    Above |t| = 1 the plain subtraction is already accurate (the answer is
    within a factor 6 of sinh t there) and is used instead.
    """
    t = np.asarray(np.arcsinh(x), dtype=float)
    t2 = t * t
    series = (
        (t * t2)
        / 6.0
        * (
            1.0
            + t2
            / 20.0
            * (
                1.0
                + t2
                / 42.0
                * (1.0 + t2 / 72.0 * (1.0 + t2 / 110.0 * (1.0 + t2 / 156.0)))
            )
        )
    )
    return np.where(np.abs(t) < 1.0, -series, -(np.sinh(t) - t))


@dataclass(frozen=True)
class _SegmentBasis:
    """Opaque `PortSolution.basis` payload for the segment-basis families.

    The per-solve context needed to read a coefficient column: the geometry
    tables, the segment view (`starts`/`jbasis`/`sigma`/`A`/`B`/`C`, the thing
    `_feed_segment_current` and the Galerkin readouts index), and the
    wavenumber they were built at. Private on purpose — #232 hands consumers
    an OPAQUE handle, not an interface; nothing outside this family should
    unpack it, and nothing here promises it survives the next solve.
    """

    geom: dict
    seg_view: dict
    k: float


class SinusoidalSolver(_ElementCurrents, _SweptPortSolutions, _Cancelable):
    """NEC2's three-term (const + sin + cos) basis on each segment, with
    end-condition coefficients closed-form per Eqs 25-64.

    Constructor takes the same `wires` / `n_per_edge_per_wire` / `junctions`
    interface as `BSplineSolver` for drop-in comparison.

    `feed_model`
        Which SOURCE a delta-gap feed applies. Accepted for signature parity
        with `SinusoidalGalerkinSolver` (momwire#192), but `"segment"` — NEC's
        E_app = V/Δ_m over the whole feed segment, Eq 187 — is the only value
        this solver can carry, and `"point"` is refused with the derivation in
        `_reject_point_feed_model`. The short form: the feed-model axis is a
        property of the TESTING, not of the basis, because a zero-width gap
        has no collocation RHS at all, and under point matching the segment
        gap already IS the zero-width gap at the mesh's own resolution
        (momwire#212, report §17).
    """

    eps = 8.8541878188e-12
    mu = 1.25663706127e-6

    # momwire#396: every ground, wire loading and the extended kernel are
    # served (this is NEC2's own basis); junction_ports and node_gaps are
    # not (`_reject_junction_ports`; node_gaps has no kwarg here at all).
    # Per-wire radii are served (momwire#147); singular enrichment doesn't
    # exist on this family (no kwarg — same TypeError shape as node_gaps).
    #
    # `knot_feeds` is the axis with no raise behind it (momwire#611): the
    # snap to a segment centre is silent, which is exactly why the row has
    # to carry it. See `_KNOT_FEEDS_REFUSAL`.
    #
    # momwire#282 stage 1 withdrew one combination inside the served ground
    # column: ground CONTACT under `ground_model="refl-coef"`. Same key,
    # same prose and same reasoning as `BSplineSolver`'s and `RazorSolver`'s
    # — contact is not a declared axis anywhere, so it is a combination key.
    # momwire#624 left this the ONLY contact refusal in the tree: razor's
    # broader `"contact+finite_ground"`, which was the pattern this comment
    # used to point at, went when its Sommerfeld half was served.
    capabilities = Capabilities(
        grounds=frozenset({"pec", "refl-coef", "sommerfeld"}),
        wire_loading=True,
        extended_kernel=True,
        junction_ports=False,
        node_gaps=False,
        knot_feeds=False,
        # momwire#673: this family snaps to the segment CENTRE grid, which is
        # exactly the grid `nec2` names -- so the axis it fails on the node
        # side is the one it serves here. The two cells are mirrors, not
        # opposites.
        centre_feeds=True,
        per_wire_radius=True,
        singular_enrichment=False,
        # Contact at a wire END is served (`ground_minus` / `ground_plus`);
        # the refl-coef row inside that column is the combination below.
        # Buried is refused outright — see `_BURIED_REFUSAL`.
        buried=False,
        contact=True,
        refusals={
            "buried": _BURIED_REFUSAL.format(cls="SinusoidalSolver"),
            "junction_ports": _JUNCTION_PORTS_REFUSAL,
            "node_gaps": _NODE_GAPS_REFUSAL,
            "knot_feeds": _KNOT_FEEDS_REFUSAL,
            "contact+refl-coef": _ground_spec.CONTACT_UNDER_REFL_COEF_REFUSAL,
            "singular_enrichment": SINGULAR_ENRICHMENT_NEVER.format(
                cls="SinusoidalSolver"
            ),
        },
    )

    def __init__(
        self,
        *,
        wires,
        n_per_edge_per_wire=None,
        feed_wire_index=0,
        feed_arclength=None,
        feeds=None,
        feed_model="segment",
        wavelength=22,
        halfdriver_factor=0.962,
        wire_radius=0.0005,
        nsegs=101,
        ground_z=None,
        ground_eps=None,
        ground_model="refl-coef",
        n_qp_sommerfeld=3,
        junctions=None,
        junction_ports=None,
        n_qp_const=8,
        extended_kernel=False,
        wire_conductivity=None,
        insulation_radius=None,
        insulation_eps_r=None,
        swept_mem_mb=256,
        cancel=None,
    ):
        if junction_ports:
            self._reject_junction_ports()
        if feed_model not in ("segment", "point"):
            raise ValueError(
                f"feed_model must be 'segment' or 'point', got {feed_model!r}"
            )
        if feed_model == "point":
            self._reject_point_feed_model()
        self.feed_model = feed_model
        # NEC's EK card (momwire#233). False — the default, and what every
        # momwire solver did before #233 — is NEC's reduced ("thin-wire")
        # kernel: the source current is a filament on the wire axis and the
        # only trace of the conductor's girth is the a² regularization of ρ
        # (`rho_eval` below). True is NEC's EXTENDED thin-wire kernel, Eqs
        # 84-98 of the theory manual: the current is a uniform tube of
        # surface current at ρ' = a and the kernel is its circumferential
        # average, expanded to O(a²). The two agree to O((a/R)²), which is
        # invisible at ordinary Δ/a and worth tens of percent below Δ/a ≈ 1.
        # NEC's `EK`/`EK 0` maps to True, `EK -1` to False.
        #
        # Cost: since momwire#245 the point-matched fill has its own C++ entry
        # point for EK (`sinusoidal_field_tensor_ek`, EKSCX), so True costs
        # ~1.2× the EK-OFF fill rather than the ~8.5× the numpy reference did
        # at N=801, and nothing (N, N, n_qp_const)-sized is materialized. The
        # numpy kernel remains the reference and the fallback. momwire#259
        # finished the sweep: the reflection-coefficient image block
        # (`ground_eps` without `ground_model="sommerfeld"`) has its own
        # EKSCX + Fresnel-dyad entry point too
        # (`sinusoidal_field_tensor_ek_refl`), so NO block of a point-matched
        # sinusoidal fill still falls to numpy under EK — free space, the PEC
        # image, the Fresnel image and Sommerfeld's C₂-scaled image all ride a
        # C++ EK kernel. (Sommerfeld's smooth interpolated REMAINDER is numpy
        # + its own kernels either way; it is a grid-dyad term, not a
        # thin-wire kernel, so EK does not apply to it at all.)
        self.extended_kernel = bool(extended_kernel)
        self._cached_ek_gating: tuple | None = None
        self._cancel = cancel
        self.wavelength = wavelength
        self.halfdriver_factor = halfdriver_factor
        self.wire_radius = wire_radius
        self.nsegs = nsegs
        self.ground_z = ground_z
        # Finite ground via NEC-style reflection-coefficient weighting of
        # the image field (docs/refl-coef-ground-plan.md Phase 6). None →
        # PEC image (today's behavior). A complex ε̃ or (eps_r, sigma)
        # tuple → Fresnel-weighted image; needs ground_z. Unlike
        # BSplineSolver there is deliberately NO `ground_phi_mode` knob:
        # the sinusoidal solver is field-based — Eqs 76-79 give the TOTAL
        # E-field of each source shape, vector- and scalar-potential
        # contributions already merged — so NEC's field dyad applies
        # exactly and no image-charge (Φ-term) weighting split exists to
        # approximate. One place the merge is NOT free is a wire end lying
        # IN the plane: the end charge each shape's field carries there is
        # ρ-weighted on the image side and not on the wire's, and the
        # residual is a point charge on the plane. It cannot be reweighted
        # in place for want of that same split, so it is subtracted whole —
        # see `_contact_charge_kernel` (momwire#282). That subtraction now
        # runs only under `ground_model="sommerfeld"`: momwire#282 stage 1
        # refused contact under the reflection-coefficient ground, so the
        # ρ-weighted branch of `_contact_charge_kernel` is reachable from
        # this constructor no longer. The branch stays because the kernel is
        # also the derivation's own statement of what the charge IS.
        if ground_eps is not None and ground_z is None:
            raise ValueError("ground_eps requires ground_z to be set")
        self.ground_eps = ground_eps
        # Finite-ground physics model, mirroring BSplineSolver: the
        # default "refl-coef" keeps every existing result bit-exact;
        # "sommerfeld" swaps the Fresnel field dyad for NEC's exact
        # decomposition — constant C2 = (eps-1)/(eps+1) on the PEC image
        # tensor (the C++ field-tensor kernel keeps serving it, a scalar
        # commutes with the projection) plus the smooth interpolated
        # remainder (`_field_tensor_sommerfeld_remainder`).
        if ground_model not in ("refl-coef", "sommerfeld"):
            raise ValueError(
                "ground_model must be 'refl-coef' or 'sommerfeld', "
                f"got {ground_model!r}"
            )
        if ground_model == "sommerfeld" and ground_eps is None:
            raise ValueError("ground_model='sommerfeld' requires ground_eps")
        self.ground_model = ground_model
        self.n_qp_sommerfeld = n_qp_sommerfeld
        # Fill-transient budget, same name/semantics/default as
        # `BSplineSolver`: MB the assembly may hold in Φ windows on top of
        # Z itself. `_assemble_Z` divides it by the per-observer-row cost
        # of the blocks the ground model switches on (momwire#332); the
        # peak is then O(budget) instead of the 6-12x Z the whole-matrix
        # field-tensor residency cost.
        self.swept_mem_mb = int(swept_mem_mb)
        if self.swept_mem_mb < 1:
            raise ValueError(f"swept_mem_mb must be >= 1, got {swept_mem_mb}")

        self.c = 1 / np.sqrt(self.eps * self.mu)
        self.freq = self.c / self.wavelength
        self.omega = 2 * np.pi * self.freq
        self.k = self.omega / self.c
        self.eta = float(np.sqrt(self.mu / self.eps))
        self.halfdriver = self.halfdriver_factor * self.wavelength / 4
        # Gauss-Legendre nodes for the const-source self-integral are
        # k-independent; cache by n_qp so sweep loops don't pay for
        # repeated leggauss() calls.
        self._leggauss_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        # `compute_impedance(...)` and `currents_at_knots(alpha)` are almost
        # always called as a pair from the UI; both internally rebuild geom
        # and basis-coefs from scratch (~5 ms/step on N=21 hentenna). The
        # geometry is purely a function of (wires, n_per_edge, junctions),
        # which are immutable after __init__; the basis-coefs are a function
        # of (geom, k, wire_radius). Cache both so the second call reuses
        # the work the first call already did. _basis_coefs validates the
        # cache by identity-checking the geom dict + value-comparing k and
        # a radius key (the scalar radius, or the per-wire array's bytes)
        # — so `compute_impedance_swept` (which mutates self.k in a loop)
        # still rebuilds basis-coefs per k, but reuses geom.
        self._cached_geometry: dict | None = None
        self._cached_basis: tuple[dict, float, float | bytes, list] | None = None
        # k-independent specular-ray tables for the `ground_eps` weighted
        # image (a `_field_ground.PairTables`: cos θ, p̂ components,
        # tangent·p̂ projections), cached per geometry object — same
        # identity-check pattern as _cached_basis / bspline's
        # `_image_refl_prep`. ρ_v/ρ_h are NOT cached: they depend on ε̃(ω)
        # and are recomputed per frequency by `PairTables.weights`. Since
        # momwire#332 unit B the point-matched fill no longer reads this
        # cache — it builds its own per-band tables via `_image_refl_band`
        # instead — so what is left resident here is only what
        # `SinusoidalGalerkinSolver` still asks `_image_refl_prep` for.
        self._cached_image_refl_prep: tuple | None = None

        if not wires:
            raise ValueError("wires must be non-empty")
        self.wires_polylines = [np.asarray(w, dtype=float) for w in wires]
        for i, pl in enumerate(self.wires_polylines):
            if pl.ndim != 2 or pl.shape[0] < 2 or pl.shape[1] != 3:
                raise ValueError(f"wire {i}: polyline must be (M, 3) with M >= 2")

        # momwire#282 stage 1: ground CONTACT under the reflection-
        # coefficient ground is refused, at construction. Same condition,
        # same prose and same place in the constructor as `BSplineSolver`'s
        # — this is a statement about the GROUND, which both trunks share,
        # and not about either basis. The #282 contact-charge correction
        # (`_contact_charge_correction`) is what made this row answerable on
        # this trunk at all; it is unaffected, and keeps serving contact
        # under `ground_model="sommerfeld"`.
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
        # (polyline) its own radius, mapped to segments via _wire_of_seg.
        # `_uniform_radius` is the scalar fast path — it keeps the
        # historical scalar code paths (and the single-`a` C++ kernels)
        # bit-identical whenever all wires share one radius, including
        # when that radius arrived as a uniform array.
        self._radius_per_wire, self._uniform_radius = _wire_spec.normalize_wire_radius(
            wire_radius, n_w
        )

        # Distributed series wire impedance (stevenmburns/momwire#131,
        # sinusoidal support #134): finite conductivity and/or a dielectric
        # jacket, same normalize/validate contract as BSplineSolver. This
        # solver is point-matched (NEC collocation), so the loading enters
        # as NEC's impedance boundary condition at the match points —
        # E_scat(n) − Z'_w·I(n) = −E_app(n) — a sparse subtraction of
        # Z'·(basis current at segment centre) in `_assemble_Z`, NOT the
        # Galerkin overlap the BSpline family uses. `wire_loss_power` DOES
        # use the closed-form ∫|I|² overlaps (a physical integral is basis-
        # scheme-independent).
        _wire_loading.configure_loading(
            self, n_w, wire_conductivity, insulation_radius, insulation_eps_r
        )

        if n_per_edge_per_wire is None:
            n_per_edge_per_wire = [None] * n_w
        if len(n_per_edge_per_wire) != n_w:
            raise ValueError(
                f"n_per_edge_per_wire length {len(n_per_edge_per_wire)} "
                f"!= number of wires {n_w}"
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
                    f"wire {i}: n_per_edge length {len(npe)} "
                    f"!= number of edges {n_edges_w}"
                )
            self.n_per_edge_per_wire.append(npe)

        if feeds is None:
            if not (0 <= feed_wire_index < n_w):
                raise ValueError(f"feed_wire_index {feed_wire_index} out of range")
            self.feeds = [(int(feed_wire_index), feed_arclength, 1.0 + 0.0j)]
        else:
            # feeds=[] is legal only when the model is driven entirely through
            # junction ports (issue #172's rule, mirrored from BSplineSolver)
            # or through node ports/gaps (#182/#305) — the Galerkin subclass
            # declares those BEFORE super().__init__ runs, via the attribute,
            # because this base's signature never learns the port kwargs.
            if (
                len(feeds) == 0
                and not junction_ports
                and not getattr(self, "_node_drive_declared", False)
            ):
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

        self.feed_wire_index = self.feeds[0][0] if self.feeds else None
        self.feed_arclength = self.feeds[0][1] if self.feeds else None
        self.n_qp_const = n_qp_const

        # momwire#429 rank 8: the spec, its inference and its validation are
        # `_wire_spec.normalize_junctions` -- one owner, because a node-gap
        # port names a MEMBER of a group and every family has to agree with
        # every other about what the members are.
        self.junctions = _wire_spec.normalize_junctions(
            junctions, self.wires_polylines, self.n_per_edge_per_wire
        )

        self.junction_ports = self._normalize_junction_ports(junction_ports)

    def _reject_junction_ports(self):
        """Refuse `junction_ports=` on the point-matched solver.

        Not a plumbing gap — the basis excludes the port (issue #177, where
        the full derivation lives so it isn't redone). Junction continuity
        here is enforced INSIDE the sinusoidal basis via the N⁻/N⁺ neighbour
        tables, so every basis function satisfies KCL at every junction as an
        algebraic identity (measured residuals ~1e-13 on a 3-way star) — the
        span simply contains no current with nonzero net inflow at a node, at
        any mesh density. Relaxing a junction's neighbour entries yields I = 0
        free ends, and a KCL row added on top would enforce 0 = 0. A real port
        needs Galerkin (segment-integrated) test rows the entire
        point-collocation field stack doesn't provide — NEC-2 has the same
        limitation for the same reason (its EX/NT/TL port is a segment-centre
        delta gap; a node-localized EMF samples to a zero RHS).

        `SinusoidalGalerkinSolver` overrides this to a no-op so the port
        basis column CAN be built and measured there — but its solves refuse
        too, for a second and stronger reason #177 did not anticipate
        (momwire#182 M5: the port basis's node charge is priced at the
        wire radius by this family's field kernel). See that class.
        """
        raise NotImplementedError(_JUNCTION_PORTS_REFUSAL)

    def _reject_point_feed_model(self):
        """Refuse `feed_model="point"` on the point-matched solver.

        Not a plumbing gap and not an RHS formula nobody has written down yet
        — the pairing that RHS *is* does not exist. Collocation tests with
        δ(s − s_m), so row m's drive is ⟨δ_m, E_app⟩ = E_app(s_m), defined
        only where E_app is a FUNCTION. The zero-width gap's applied field is
        the distribution E_app = V·δ(s − s0), and δ·δ is not one: there is no
        number to put in the feed row, and 0 in every other row is the
        unexcited problem. `SinusoidalGalerkinSolver` carries the same source
        only because its pairing ⟨f_i, E_app⟩ tests against a continuous f_i,
        on which the delta collapses to the drive column −V·f_i(s0)
        (momwire#192). The feed-model axis is a property of the TESTING.

        Every regularization of the delta collapses into one of two dead ends.
        Replace δ by a nascent delta g_w of unit integral and width w, and
        sample: v_m = −V·g_w(s_m − s0). Measured on the canonical 0.962 λ/2
        dipole over N = 41…641 (momwire#212, report §17):

          **w ≪ h.** Every match point but the feed's sees ~0 while the feed
          row sees −V·g_w(0) ≈ −V/w, so the RHS is the segment-gap RHS scaled
          by h/w, and Z — homogeneous of degree −1 in the RHS — comes out
          (w/h)·Z_segment. Collocation reads the source's WIDTH off the feed
          row's amplitude and cannot tell a narrow gap from a small voltage,
          so a sub-mesh width is returned as a fraction of a volt. The only
          width this model already owns is the wire radius: the b → a limit of
          the magnetic frill, i.e. a delta ring of magnetic current at ρ = a,
          whose on-axis field E_z = (V a²/2)(1 + jkR)e^{−jkR}/R³ with
          R = √(d² + a²) is smooth, finite everywhere, integrates to V, and
          has no free parameter left (the textbook frill's entire content is
          its b/a — exactly the modeling parameter a zero-width gap may not
          introduce). Evaluated in this solver's own kernel convention — ring
          on the surface, observation on the axis, which reproduces the
          source-filament / BC-on-surface separation √(d² + a²) the fill
          already uses — it has w = 2a, and a/h ≤ 3.0e-2 across the whole
          ladder. Measured: the sampled RHS is the segment-gap RHS times
          h/2a to ≤2.8e-5 relative, and Z DOUBLES at every mesh doubling
          (0.270 → 0.533 → 1.059 → 2.112 → 4.218 Ω at N = 41…641) rather than
          stabilizing. Not the segment gap's log walk — a linear one.

          **w ≳ h.** The source is mesh-resolved and collocation is consistent
          again, but w is then a modeling parameter, and the smallest one the
          mesh resolves is w = h — which IS NEC's segment gap. Confirmed from
          the other side: the exact cell average of the zero-width field above
          reproduces the segment gap's −V/h to 6.0e-8…1.4e-6 of |Z| on the
          same ladder, four decades inside the (a/h)² bound.

        So under collocation the segment gap is not a rival source model to
        the point gap — it is the point gap, rendered at the only resolution
        collocation has. There is nothing left to opt into, which is why this
        is a refusal rather than a second default.

        A Hallén-style particular-solution forcing term is the one remaining
        escape and is rejected on scope: it needs the infinite-wire kernel's
        Fourier inversion to get J_p, J_p satisfies neither this basis's end
        conditions nor its junction tables, and moving the unknown changes
        what "the basis" means — which would break the one-axis-at-a-time
        discipline the whole instrument exists to keep.
        """
        raise NotImplementedError(
            "feed_model='point' is not supported on SinusoidalSolver: a "
            "zero-width gap has no collocation RHS — the drive is E_app "
            "sampled AT a match point and the source is a delta there, so "
            "the pairing is undefined (momwire#212). Under point matching "
            "NEC's segment gap already is the zero-width gap at the mesh's "
            "own resolution. For the zero-width source use "
            "SinusoidalGalerkinSolver(feed_model='point'), whose test "
            "integral is what makes a delta source admissible"
        )

    def _normalize_junction_ports(self, junction_ports):
        """Validate `junction_ports=` into a list of (junction_index, voltage).

        Same rules as `BSplineSolver` (issue #172): entries may be a plain
        int (voltage 0) or a (index, voltage) pair; indices must be in range
        and may not repeat. Returns [] when there are none, which is the only
        outcome the point-matched solver ever reaches — it raises in
        `_reject_junction_ports` first.
        """
        if not junction_ports:
            return []
        out = []
        seen = set()
        for p in junction_ports:
            j_idx, volt = (p, 0.0 + 0.0j) if np.isscalar(p) else p
            j_idx = int(j_idx)
            if not (0 <= j_idx < len(self.junctions)):
                raise ValueError(
                    f"junction_ports: junction index {j_idx} out of range "
                    f"[0, {len(self.junctions)})"
                )
            if j_idx in seen:
                raise ValueError(f"junction_ports: junction {j_idx} listed twice")
            seen.add(j_idx)
            out.append((j_idx, complex(volt)))
        return out

    def _leggauss_cached(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._leggauss_cache.get(n)
        if cached is not None:
            return cached
        gx, gw = np.polynomial.legendre.leggauss(n)
        gx = np.ascontiguousarray(gx, dtype=np.float64)
        gw = np.ascontiguousarray(gw, dtype=np.float64)
        self._leggauss_cache[n] = (gx, gw)
        return gx, gw

    # ------------------------------------------------------------------
    # Geometry build
    # ------------------------------------------------------------------

    def _build_geometry(self):
        """Discretize wires into segments and build the N^-/N^+ neighbor
        tables for every segment, with arc-flip σ signs.

        For a segment n at the head of a wire-segment sequence whose
        natural arc direction matches NEC's convention (segment's `end-2` =
        `seg_r` is at the junction node when treated as an N^- neighbour
        of another basis), σ = +1. When the natural tangent is reversed
        relative to NEC's expected arc, σ = -1.
        """
        if self._cached_geometry is not None:
            return self._cached_geometry
        # Build per-edge chunks (n_e segments per edge) in vectorized form,
        # then concatenate. Inner k_seg Python loop (171 iters × 5 list
        # appends + 4 numpy temporaries) was ~12% of py-spy samples at N=21.
        seg_l_chunks: list[np.ndarray] = []
        seg_r_chunks: list[np.ndarray] = []
        seg_c_chunks: list[np.ndarray] = []
        seg_t_chunks: list[np.ndarray] = []
        seg_h_chunks: list[np.ndarray] = []
        wire_first_seg: list[int] = []
        wire_last_seg: list[int] = []
        running_count = 0

        for w_idx, (pl, npe_list) in enumerate(
            zip(self.wires_polylines, self.n_per_edge_per_wire)
        ):
            wire_first = running_count
            for e_idx in range(pl.shape[0] - 1):
                p0 = pl[e_idx]
                p1 = pl[e_idx + 1]
                vec = p1 - p0
                edge_len = float(np.linalg.norm(vec))
                if edge_len < 1e-15:
                    raise ValueError(f"wire {w_idx} edge {e_idx} has zero length")
                tan = vec / edge_len
                n_e = npe_list[e_idx]
                h_e = edge_len / n_e
                # frac in [0, 1] sampled at n_e+1 points; consecutive points
                # bound each segment.
                frac = np.linspace(0.0, 1.0, n_e + 1)
                pts = p0 + frac[:, None] * vec  # (n_e+1, 3)
                pl_l_arr = pts[:-1]  # (n_e, 3)
                pl_r_arr = pts[1:]  # (n_e, 3)
                seg_l_chunks.append(pl_l_arr)
                seg_r_chunks.append(pl_r_arr)
                seg_c_chunks.append(0.5 * (pl_l_arr + pl_r_arr))
                seg_t_chunks.append(np.broadcast_to(tan, (n_e, 3)).copy())
                seg_h_chunks.append(np.full(n_e, h_e, dtype=np.float64))
                running_count += n_e
            wire_last_seg.append(running_count - 1)
            wire_first_seg.append(wire_first)

        seg_l = np.concatenate(seg_l_chunks, axis=0)
        seg_r = np.concatenate(seg_r_chunks, axis=0)
        seg_c = np.concatenate(seg_c_chunks, axis=0)
        seg_t = np.concatenate(seg_t_chunks, axis=0)
        seg_h = np.concatenate(seg_h_chunks)
        n_segs = seg_l.shape[0]

        # Per-segment N^- / N^+ neighbours — built directly as flat arrays
        # (nm_basis, nm_seg, nm_sigma and the np_ trio), where each entry
        # k is a single (basis i, neighbour seg j, σ) triple. No
        # list-of-lists intermediate, no per-seg Python append loop.
        #
        # nm_basis[k] = i  → basis i has a N⁻ neighbour at seg j
        #   nm_seg[k] = j     ("j's NEC end-2 coincides with i's end-1")
        # nm_sigma[k] = σ.   In-wire connections use σ = +1; junction
        # connections use ±1 per the L/R side rule below.
        nm_basis_chunks: list[np.ndarray] = []
        nm_seg_chunks: list[np.ndarray] = []
        nm_sigma_chunks: list[np.ndarray] = []
        np_basis_chunks: list[np.ndarray] = []
        np_seg_chunks: list[np.ndarray] = []
        np_sigma_chunks: list[np.ndarray] = []

        # In-wire neighbours: per wire, segs [first+1..last] each get nm
        # = (i-1, +1) and segs [first..last-1] each get np = (i+1, +1).
        # Two np.arange pairs per wire instead of an n_segs Python loop.
        for w_idx in range(len(self.wires_polylines)):
            first = wire_first_seg[w_idx]
            last = wire_last_seg[w_idx]
            if last > first:
                m = last - first
                nm_basis_chunks.append(np.arange(first + 1, last + 1, dtype=np.int64))
                nm_seg_chunks.append(np.arange(first, last, dtype=np.int64))
                nm_sigma_chunks.append(np.ones(m, dtype=np.int8))
                np_basis_chunks.append(np.arange(first, last, dtype=np.int64))
                np_seg_chunks.append(np.arange(first + 1, last + 1, dtype=np.int64))
                np_sigma_chunks.append(np.ones(m, dtype=np.int8))

        # Junctions AT the ground plane (#151): NEC's conect() connects a
        # ground-touching end to ground ONLY — it never searches for real
        # partners there (`icon1[i] = i; jump = TRUE` in nec2c). So a
        # grounded junction emits NO inter-wire neighbour entries: each
        # member is independently ground-connected (self-image basis), and
        # the members couple through the plane via their images.
        grounded_junctions = set()
        gz = self.ground_z
        if gz is not None:
            for j_i, jn in enumerate(self.junctions):
                w0, end0 = jn[0]
                pl0 = np.asarray(self.wires_polylines[w0], dtype=np.float64)
                pt = pl0[0] if end0 == "start" else pl0[-1]
                if abs(pt[2] - gz) <= _ground_spec.ground_touch_tol(pl0):
                    grounded_junctions.add(j_i)

        # Junction neighbours: small Python loop (junctions count is O(1)
        # in geometry size — 2-4 junctions on typical antennas, with K=2-6
        # members each producing K(K-1) edges). Append to per-junction
        # Python lists, then convert to numpy once.
        junc_nm_basis: list[int] = []
        junc_nm_seg: list[int] = []
        junc_nm_sigma: list[int] = []
        junc_np_basis: list[int] = []
        junc_np_seg: list[int] = []
        junc_np_sigma: list[int] = []
        for j_i, jn in enumerate(self.junctions):
            if j_i in grounded_junctions:
                continue
            # (segment_idx, which_end_of_segment_is_at_node) for every
            # wire-end at this junction.
            members = []
            for w, end in jn:
                if end == "start":
                    seg_idx = wire_first_seg[w]
                    end_side = "L"  # seg_l of this segment is at node
                else:
                    seg_idx = wire_last_seg[w]
                    end_side = "R"  # seg_r of this segment is at node
                members.append((seg_idx, end_side))
            # Every (i, j) pair with i != j contributes one edge from i's
            # perspective — N^- or N^+ depending on i's side at the node.
            for a_idx in range(len(members)):
                i_seg, i_side = members[a_idx]
                for b_idx in range(len(members)):
                    if b_idx == a_idx:
                        continue
                    j_seg, j_side = members[b_idx]
                    if i_side == "L":
                        # j is in N^-(i). σ = +1 if j's natural seg_r is at
                        # the node (j_side == "R"), else −1.
                        sigma = +1 if j_side == "R" else -1
                        junc_nm_basis.append(i_seg)
                        junc_nm_seg.append(j_seg)
                        junc_nm_sigma.append(sigma)
                    else:
                        # j is in N^+(i). σ = +1 if j's natural seg_l is at
                        # the node (j_side == "L"), else −1.
                        sigma = +1 if j_side == "L" else -1
                        junc_np_basis.append(i_seg)
                        junc_np_seg.append(j_seg)
                        junc_np_sigma.append(sigma)
        if junc_nm_basis:
            nm_basis_chunks.append(np.asarray(junc_nm_basis, dtype=np.int64))
            nm_seg_chunks.append(np.asarray(junc_nm_seg, dtype=np.int64))
            nm_sigma_chunks.append(np.asarray(junc_nm_sigma, dtype=np.int8))
        if junc_np_basis:
            np_basis_chunks.append(np.asarray(junc_np_basis, dtype=np.int64))
            np_seg_chunks.append(np.asarray(junc_np_seg, dtype=np.int64))
            np_sigma_chunks.append(np.asarray(junc_np_sigma, dtype=np.int8))

        # Per-feed segment index: for each feed, the segment on its wire
        # whose center is closest to the requested arclength (default:
        # midpoint of the wire). Primary feed kept as `feed_seg` for
        # back-compat with single-feed callers / fields.
        #
        # `feed_xi` is what is LEFT OVER — the requested arclength's offset
        # from that segment's centre, in metres, signed along the wire's arc.
        # A caller that names a segment centre gets 0 and nothing changes; a
        # caller that names a KNOT gets ±h/2 (momwire#648).
        #
        # The distinction only reaches the fill under `feed_model="point"`,
        # and that is not an implementation limit but what the two source
        # models ARE: a segment gap is E_app spread over one segment, so its
        # position is that segment and there is no sub-segment offset to
        # honour. A zero-width gap is a point, and a point can be anywhere.
        # `SinusoidalGalerkinSolver._drive_columns` reads this; the segment
        # branch and the whole point-matched family ignore it.
        feed_segs = []
        feed_xi = []
        for w_f, arc_req, _v in self.feeds:
            first = wire_first_seg[w_f]
            last = wire_last_seg[w_f]
            feed_h_w = seg_h[first : last + 1]
            feed_arc_centers = np.cumsum(feed_h_w) - 0.5 * feed_h_w
            total_arc = float(np.sum(feed_h_w))
            feed_arc = arc_req if arc_req is not None else 0.5 * total_arc
            feed_arc = min(max(feed_arc, 0.0), total_arc)
            pick, _margin = _feed_snap.snap(
                feed_arc_centers,
                feed_arc,
                total_arc=total_arc,
                family=type(self).__name__,
                wire=w_f,
            )
            feed_segs.append(first + pick)
            feed_xi.append(float(feed_arc - feed_arc_centers[pick]))
        # None with feeds=[] — legal only under junction-port drive (#172).
        feed_seg = feed_segs[0] if feed_segs else None

        # Concatenate the in-wire + junction chunks into the final flat
        # neighbour arrays. Empty geometries (e.g. a single-segment wire
        # with no junctions) get length-zero arrays of the right dtype.
        def _cat(chunks: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
            return np.concatenate(chunks) if chunks else np.empty(0, dtype=dtype)

        nm_basis = _cat(nm_basis_chunks, np.int64)
        nm_seg = _cat(nm_seg_chunks, np.int64)
        nm_sigma = _cat(nm_sigma_chunks, np.int8)
        np_basis = _cat(np_basis_chunks, np.int64)
        np_seg = _cat(np_seg_chunks, np.int64)
        np_sigma = _cat(np_sigma_chunks, np.int8)
        # Per-seg neighbour counts via bincount — no Python loop.
        nm_count = np.bincount(nm_basis, minlength=n_segs).astype(np.int64)
        np_count = np.bincount(np_basis, minlength=n_segs).astype(np.int64)

        # Ground-junction flags (#151): a wire endpoint lying in an active
        # ground plane is electrically connected to its own image — NEC2's
        # "connected to ground" condition. Mark the end segment's plane
        # side so _basis_coefs swaps the free-end X=0 branch for the
        # interior branch with a self-image P-sum atom. end-1 (seg_l at
        # the plane) is the N⁻ side, end-2 (seg_r) the N⁺ side.
        ground_minus = np.zeros(n_segs, dtype=bool)
        ground_plus = np.zeros(n_segs, dtype=bool)
        gz = self.ground_z
        if gz is not None:
            junctioned = set()
            for jn in self.junctions:
                for w, end in jn:
                    junctioned.add((w, end))
            for w_idx, pl in enumerate(self.wires_polylines):
                pl_arr = np.asarray(pl, dtype=np.float64)
                tol = _ground_spec.ground_touch_tol(pl_arr)
                if float(pl_arr[:, 2].min()) < gz - tol:
                    raise ValueError(
                        f"wire {w_idx} dips below the ground plane "
                        f"(min z = {pl_arr[:, 2].min():.6g} < ground_z = "
                        f"{gz:g}): " + _BURIED_REFUSAL.format(cls=type(self).__name__)
                    )
                start_touch = abs(pl_arr[0, 2] - gz) <= tol
                end_touch = abs(pl_arr[-1, 2] - gz) <= tol
                if start_touch and (w_idx, "start") not in junctioned:
                    ground_minus[wire_first_seg[w_idx]] = True
                if end_touch and (w_idx, "end") not in junctioned:
                    ground_plus[wire_last_seg[w_idx]] = True
            # A segment lying IN the plane (both ends at gz) is degenerate
            # over a conducting ground — its image cancels it.
            in_plane = (np.abs(seg_l[:, 2] - gz) + np.abs(seg_r[:, 2] - gz)) <= (
                2e-6 * np.maximum(seg_h, 1e-30)
            )
            if in_plane.any():
                raise ValueError(
                    "segment(s) lying in the ground plane (both endpoints at "
                    "ground_z) — degenerate over a conducting ground"
                )
            # Grounded junction: per NEC's conect(), each member wire-end
            # is ground-connected INSTEAD of inter-connected (its junction
            # entries were skipped above) — the members couple through
            # their images.
            for j_i in grounded_junctions:
                for w, end in self.junctions[j_i]:
                    if end == "start":
                        ground_minus[wire_first_seg[w]] = True
                    else:
                        ground_plus[wire_last_seg[w]] = True

        self._cached_geometry = {
            "seg_l": seg_l,
            "seg_r": seg_r,
            "seg_centers": seg_c,
            "seg_tangents": seg_t,
            "seg_h": seg_h,
            "n_segs": n_segs,
            "nm_basis": nm_basis,
            "nm_seg": nm_seg,
            "nm_sigma": nm_sigma,
            "np_basis": np_basis,
            "np_seg": np_seg,
            "np_sigma": np_sigma,
            "nm_count": nm_count,
            "np_count": np_count,
            "wire_first": wire_first_seg,
            "wire_last": wire_last_seg,
            "feed_seg": feed_seg,
            "feed_segs": feed_segs,
            "feed_xi": feed_xi,
            "ground_minus": ground_minus,
            "ground_plus": ground_plus,
            # Junctions whose node lies in the ground plane: their members are
            # ground-connected INSTEAD of inter-connected, so they emit no
            # neighbour entries. Published because a junction port cannot live
            # on one (its node voltage is pinned by the image) — see
            # `SinusoidalGalerkinSolver._junction_port_view`.
            "grounded_junctions": frozenset(grounded_junctions),
        }
        return self._cached_geometry

    # ------------------------------------------------------------------
    # Basis-function coefficient computation
    # ------------------------------------------------------------------

    def _basis_coefs(self, geom, k):
        """Per-basis closed-form (A, B, C, σ) coefficients on every
        supporting segment, following Eqs 43-64 of the NEC2 Theory Manual.

        Returns a CSR-by-segment `seg_view` dict:

            seg_view["starts"][s:s+2] → range of entries for segment s in
                the flat per-segment arrays below.
            seg_view["jbasis"][k]     → which basis contributes entry k.
            seg_view["A"/"B"/"C"][k]  → that basis's coefficient on seg.
            seg_view["AC"][k]         → A + C, in closed form (see below).
            seg_view["sigma"][k]      → σ sign relative to NEC arc.

        `AC` is A + C — the entry's current at the segment CENTRE, and the
        coefficient the well-scaled shape set {1, sin kξ, cos kξ − 1} puts on
        the constant shape. It is published rather than left to each consumer
        because A ≈ −C to O((kΔ)²) on every entry type (self: A = −1 against
        C → 1; N± neighbour: C/A = −cos(kΔ/2)), so `σA + B·sin kξ + σC·cos kξ`
        evaluates a quantity of size ~(kΔ)²/8 from terms of size 1 — a
        relative error of ~ε·8/(kΔ)², which is 1.2e-10 at N=801 on the
        half-wave dipole and grows like N² (stevenmburns/momwire#203).

        `AC` is built from a per-branch CLOSED FORM, never from `A + C`
        (stevenmburns/momwire#606). Adding the two float64 coefficients is
        correctly rounded *to the sum*, but the sum of two rounded values is
        not the rounded value of the sum: `A` and `C` are each O(1) carrying
        an absolute ~ε, so their float sum carries ~ε absolute against a true
        value of O((kΔ)²). That is a relative error of 1 % at kΔ = 2.1e-4 and
        44000 % at kΔ = 1e-5 — the summed spelling has no correct digits in
        the regime it exists to serve. Each branch instead uses an identity
        in which no term is larger than the answer: `_recip_sin_gap` for the
        neighbour and ground-extension entries, the D-collected numerator for
        the interior self entry, and a sum-to-product `cos(kΔ/2) − cos(kΔ)`
        for the free-end entries. All three are exact to a relative ε.

        Writing the entries directly into seg-major position during the
        main per-basis loop (instead of building a list-of-lists `basis`
        and then transposing it) avoids a second Python pass over ~1700
        entries — the path that was costing ~1.2 ms/step at N=21 hentenna.
        It also lets `_assemble_Z`'s flat-array fill become vectorized
        numpy (np.repeat + element-wise multiply) rather than a Python
        scatter loop.

        Reciprocity lets us compute `starts[]` upfront from the geometry
        alone, without a first pass to count: a segment s appears as a
        support entry of basis s itself (the self entry) plus once per
        basis i in `nm[s] ∪ np_[s]` — i.e. once for every neighbour of s.
        So entries_per_seg[s] = 1 + len(nm[s]) + len(np_[s]). Each basis
        i contributes 1 + len(nm[i]) + len(np_[i]) entries on its own
        support, so the *basis-major* count is the same value indexed
        differently. We only need the seg-major layout for downstream
        consumers, so that's all we materialise.
        """
        # Uniform radius keeps the historical scalar path (bit-exact);
        # mixed radii promote `a` to a per-segment array — each segment
        # inherits its wire's radius. The cache key is the scalar for the
        # uniform case, the radius array's bytes otherwise.
        if self._uniform_radius is not None:
            a = a_key = self._uniform_radius
        else:
            a = self._seg_radius(geom)
            a_key = self._radius_per_wire.tobytes()
        cached = self._cached_basis
        if (
            cached is not None
            and cached[0] is geom
            and cached[1] == k
            and cached[2] == a_key
        ):
            return cached[3]
        seg_h = geom["seg_h"]
        n_segs = geom["n_segs"]
        nm_basis = geom["nm_basis"]
        nm_seg = geom["nm_seg"]
        nm_sigma = geom["nm_sigma"]
        np_basis = geom["np_basis"]
        np_seg = geom["np_seg"]
        np_sigma = geom["np_sigma"]
        nm_count = geom["nm_count"]
        np_count = geom["np_count"]

        ka = k * a
        # a_i± from Eq 25: 1/(ln(2/(k·a)) − γ) per segment (scalar when the
        # radius is uniform). Mixed radii follow NEC2's TBF convention —
        # each segment's log-constant uses that segment's OWN radius: the
        # self formulas below (D, Q±, B_i0, C_i0 and the end branches) take
        # a_const at the basis's segment, while the P sums and the N±
        # neighbour coefficient entries take it at the NEIGHBOUR's segment
        # (nec2c computes aj from bi[jcox] for every connected segment and
        # resets aj = ap = the self constant before the Q/D formulas).
        a_const = 1.0 / (np.log(2.0 / ka) - _EULER_GAMMA)

        # Pre-compute every per-segment trig in one vectorized pass.
        kd_arr = k * np.asarray(seg_h, dtype=np.float64)
        sin_kd = np.sin(kd_arr)
        cos_kd = np.cos(kd_arr)
        sin_kd_2 = np.sin(0.5 * kd_arr)
        cos_kd_2 = np.cos(0.5 * kd_arr)
        sin_kd_4 = np.sin(0.25 * kd_arr)
        # 1/sin(kΔ) − 1/(2 sin(kΔ/2)), cancellation-free (#606). This IS
        # `A + C` on every neighbour entry (up to its a·Q factor) and on the
        # ground-junction extension, which is the same shape folded back.
        recip_gap = _recip_sin_gap(kd_arr)
        # P-sum atoms: (1-cos(kd_j))/sin(kd_j) * a_const for N⁻; flip sign
        # for N⁺. Spelled as its exact identity tan(kΔ/2) (momwire#799): the
        # literal quotient computes a numerator of size kΔ²/2 out of terms of
        # size 1, which is 1.9e-12 relative at the kΔ = 3.8e-3 of an
        # 801-segment half-wave dipole and grows as 1/kΔ². Same argument, and
        # the same regime, as `_recip_sin_gap` (momwire#606) two doors down.
        P_minus_atom = (sin_kd_2 / cos_kd_2) * a_const

        # Per-basis P_minus[i] = Σ_{j ∈ N⁻(i)} atom[j], via scatter-sum on
        # the flat nm arrays. Same for P_plus[i] over N⁺.
        P_minus_arr = np.zeros(n_segs, dtype=np.float64)
        np.add.at(P_minus_arr, nm_basis, P_minus_atom[nm_seg])
        P_plus_arr = np.zeros(n_segs, dtype=np.float64)
        np.add.at(P_plus_arr, np_basis, -P_minus_atom[np_seg])

        # Ground junction (#151): an end at the ground plane is connected
        # to its own image — same length, same radius — so the plane side's
        # P-sum gains the segment's own atom and the segment takes the
        # interior (connected) branch instead of the free-end X=0 one. No
        # entry lands in the nm/np extension tables: the image side's
        # current is supplied by the ground image blocks, which mirror the
        # whole real expansion.
        ground_minus = geom["ground_minus"]
        ground_plus = geom["ground_plus"]
        if ground_minus.any():
            P_minus_arr = P_minus_arr + np.where(ground_minus, P_minus_atom, 0.0)
        if ground_plus.any():
            P_plus_arr = P_plus_arr - np.where(ground_plus, P_minus_atom, 0.0)

        # Per-basis (A_i0, B_i0, C_i0, Q_minus, Q_plus) as N-vectors,
        # following Eqs 43-64. The 4-way branch on (has_minus, has_plus)
        # is masked: compute the interior formula everywhere, then patch
        # the rare end / isolated branches via boolean masks. For a
        # hentenna (closed loop) every segment is interior; for a dipole
        # the wire-tip segments hit only_minus / only_plus.
        has_minus = (nm_count > 0) | ground_minus
        has_plus = (np_count > 0) | ground_plus
        both = has_minus & has_plus

        # Interior branch (Eqs 49-53). Compute everywhere using a_minus =
        # a_plus = a_const; mask out below where the branch doesn't apply.
        D = (P_minus_arr * P_plus_arr + a_const * a_const) * sin_kd + (
            P_minus_arr - P_plus_arr
        ) * a_const * cos_kd
        # Guard the denominator: replace 0 with 1 so the (masked-away)
        # bogus result is finite. Real divide-by-zero in the interior
        # branch would be a degenerate geometry (kd_i = nπ) we don't
        # support anyway.
        D_safe = np.where(D != 0, D, 1.0)
        sin_kd_safe = np.where(sin_kd != 0, sin_kd, 1.0)
        # 1 − cos kΔ = 2 sin²(kΔ/2), exactly (momwire#799). Both numerators
        # are a difference of two O(a·kΔ²) terms, so the O(kΔ²) half arriving
        # with only an ABSOLUTE ε would set the whole quotient's accuracy.
        one_m_cos_kd = 2.0 * sin_kd_2 * sin_kd_2
        Q_minus_arr = (a_const * one_m_cos_kd - P_plus_arr * sin_kd) / D_safe
        Q_plus_arr = (-a_const * one_m_cos_kd - P_minus_arr * sin_kd) / D_safe
        A_i0_arr = np.full(n_segs, -1.0)
        B_i0_arr = a_const * (Q_minus_arr + Q_plus_arr) * sin_kd_2 / sin_kd_safe
        C_i0_arr = a_const * (Q_minus_arr - Q_plus_arr) * cos_kd_2 / sin_kd_safe
        # `A + C` on the interior self entry, cancellation-free (#606).
        # A = −1 and C → 1 as k → 0, so the literal sum is a 1−1 subtraction
        # returning O((kΔ)²) — 44000 % wrong at kΔ = 1e-5. Substituting
        # C = a(Q⁻−Q⁺)/(2 sin(kΔ/2)) and the interior Q's, then collecting
        # over the common denominator D, every surviving term carries an
        # explicit sin²(kΔ/4) or S² factor and the O(1) pieces are gone:
        #
        #   A + C = [4a²S·sin²(kΔ/4) − 2a·ΔP·sin²(kΔ/4)
        #            − 2·S·C₂·P⁻P⁺ + 2a·ΔP·S²] / D
        #
        # with S = sin(kΔ/2), C₂ = cos(kΔ/2), ΔP = P⁻ − P⁺. Numerator and
        # denominator are O((kΔ)³) and O(kΔ), giving the O((kΔ)²) answer at
        # a relative ε instead of an absolute one.
        _dP = P_minus_arr - P_plus_arr
        _s4sq = sin_kd_4 * sin_kd_4
        AC_i0_arr = (
            4.0 * a_const * a_const * sin_kd_2 * _s4sq
            - 2.0 * a_const * _dP * _s4sq
            - 2.0 * sin_kd_2 * cos_kd_2 * P_minus_arr * P_plus_arr
            + 2.0 * a_const * _dP * sin_kd_2 * sin_kd_2
        ) / D_safe

        # Only-plus / only-minus / isolated branches: skip the work
        # entirely when none of them apply (the common case).
        only_plus = has_plus & ~has_minus
        only_minus = has_minus & ~has_plus
        iso = ~has_minus & ~has_plus
        if only_plus.any() or only_minus.any() or iso.any():
            # End segment with free end at end-1 (Eqs 54-57). X = 0.
            denom_x = cos_kd
            denom_x_safe = np.where(denom_x != 0, denom_x, 1.0)
            qpe1_denom = a_const * sin_kd - P_plus_arr * cos_kd
            qpe1_denom_safe = np.where(qpe1_denom != 0, qpe1_denom, 1.0)
            Q_plus_e1 = -one_m_cos_kd / qpe1_denom_safe
            B_e1 = (sin_kd_2 + a_const * Q_plus_e1 * cos_kd_2) / denom_x_safe
            C_e1 = (cos_kd_2 + a_const * Q_plus_e1 * sin_kd_2) / denom_x_safe

            # End segment with free end at end-2 (Eqs 58-61). X = 0.
            qme2_denom = a_const * sin_kd + P_minus_arr * cos_kd
            qme2_denom_safe = np.where(qme2_denom != 0, qme2_denom, 1.0)
            Q_minus_e2 = one_m_cos_kd / qme2_denom_safe
            B_e2 = (-sin_kd_2 + a_const * Q_minus_e2 * cos_kd_2) / denom_x_safe
            C_e2 = (cos_kd_2 - a_const * Q_minus_e2 * sin_kd_2) / denom_x_safe

            # Isolated single segment (Eq 64). X = 0 → A = -1, B = 0,
            # C = 1/cos(kΔ/2).
            cos_kd_2_safe = np.where(cos_kd_2 != 0, cos_kd_2, 1.0)
            C_iso = 1.0 / cos_kd_2_safe

            Q_minus_arr = np.where(only_plus | iso, 0.0, Q_minus_arr)
            Q_minus_arr = np.where(only_minus, Q_minus_e2, Q_minus_arr)
            Q_plus_arr = np.where(only_minus | iso, 0.0, Q_plus_arr)
            Q_plus_arr = np.where(only_plus, Q_plus_e1, Q_plus_arr)
            B_i0_arr = np.where(only_plus, B_e1, B_i0_arr)
            B_i0_arr = np.where(only_minus, B_e2, B_i0_arr)
            B_i0_arr = np.where(iso, 0.0, B_i0_arr)
            C_i0_arr = np.where(only_plus, C_e1, C_i0_arr)
            C_i0_arr = np.where(only_minus, C_e2, C_i0_arr)
            C_i0_arr = np.where(iso, C_iso, C_i0_arr)
            # A_i0 = -1 in every branch — no patch needed.
            #
            # `A + C` on the same three branches, cancellation-free (#606).
            # With A = −1 and C = (cos(kΔ/2) ± a·Q·sin(kΔ/2))/cos(kΔ), the
            # O(1) part of the sum is cos(kΔ/2) − cos(kΔ), which the
            # sum-to-product identity turns into 2·sin(3kΔ/4)·sin(kΔ/4) —
            # explicitly O((kΔ)²), no subtraction. The Q-terms are already
            # O((kΔ)²) themselves (Q_e ~ (1−cos kΔ)/… ), so nothing here is
            # bigger than the answer.
            _cos_gap_e = 2.0 * np.sin(0.75 * kd_arr) * sin_kd_4
            AC_e1 = (_cos_gap_e + a_const * Q_plus_e1 * sin_kd_2) / denom_x_safe
            AC_e2 = (_cos_gap_e - a_const * Q_minus_e2 * sin_kd_2) / denom_x_safe
            # Isolated (Eq 64): A + C = 1/cos(kΔ/2) − 1 = 2sin²(kΔ/4)/cos(kΔ/2).
            AC_iso = 2.0 * _s4sq / cos_kd_2_safe
            AC_i0_arr = np.where(only_plus, AC_e1, AC_i0_arr)
            AC_i0_arr = np.where(only_minus, AC_e2, AC_i0_arr)
            AC_i0_arr = np.where(iso, AC_iso, AC_i0_arr)
        # `both` mask is unused: the interior formula already lives there.
        del both

        # Ground junction (#151), part 2: the connected-side EXTENSION.
        # For a real neighbour the Q-scaled Eqs 43-48 shape lands on the
        # neighbour segment; for a ground connection the "neighbour" is the
        # segment's own image, and nec2c's tbf folds that extension back
        # onto the segment itself with the sin term mirrored (s → −s, so
        # B → −B; `segj.bx[jsnox] = -segj.bx[jsnox]` in tbf). The folded
        # entry targets the same (basis, seg) pair as the self entry — and
        # downstream scatter is assignment, not add — so merge it into the
        # self coefficients. Basis + its image then forms the intended
        # continuous through-plane current.
        # (the ground "neighbour" is the segment itself, so the neighbour
        # a-constant is its own — works for scalar and per-segment alike)
        # The extension's own A/C pair is the neighbour shape, so its
        # contribution to `A + C` is the same `recip_gap` identity (#606) —
        # not the difference of the two patches applied above.
        if ground_minus.any():
            Qg = np.where(ground_minus, Q_minus_arr, 0.0)
            A_i0_arr = A_i0_arr + a_const * Qg / sin_kd_safe
            B_i0_arr = B_i0_arr - a_const * Qg / (2.0 * cos_kd_2)
            C_i0_arr = C_i0_arr - a_const * Qg / (2.0 * sin_kd_2)
            AC_i0_arr = AC_i0_arr + a_const * Qg * recip_gap
        if ground_plus.any():
            Qg = np.where(ground_plus, Q_plus_arr, 0.0)
            A_i0_arr = A_i0_arr - a_const * Qg / sin_kd_safe
            B_i0_arr = B_i0_arr - a_const * Qg / (2.0 * cos_kd_2)
            C_i0_arr = C_i0_arr + a_const * Qg / (2.0 * sin_kd_2)
            AC_i0_arr = AC_i0_arr - a_const * Qg * recip_gap

        # Build the flat entry arrays. Three blocks (self, N⁻, N⁺), each
        # produced with one set of vectorized array ops:
        #
        # Self entries: basis = arange(n_segs), seg = arange(n_segs).
        self_seg = np.arange(n_segs, dtype=np.int64)

        # N⁻ neighbour entries (Eqs 43-45). Basis i contributes at seg j
        # (the j-side neighbour) using Q_minus[i] for the magnitude and the
        # NEIGHBOUR segment's a-constant (a_const[nm_seg], per TBF).
        a_nm = a_const if np.ndim(a_const) == 0 else a_const[nm_seg]
        a_np = a_const if np.ndim(a_const) == 0 else a_const[np_seg]
        nm_Q = Q_minus_arr[nm_basis]
        nm_A = a_nm * nm_Q / sin_kd[nm_seg]
        nm_B = a_nm * nm_Q / (2.0 * cos_kd_2[nm_seg])
        nm_C = -a_nm * nm_Q / (2.0 * sin_kd_2[nm_seg])
        # A + C = a·Q·[1/sin(kΔ) − 1/(2 sin(kΔ/2))] — the identity, not the
        # subtraction (#606).
        nm_AC = a_nm * nm_Q * recip_gap[nm_seg]

        # N⁺ neighbour entries (Eqs 46-48).
        np_Q = Q_plus_arr[np_basis]
        np_A = -a_np * np_Q / sin_kd[np_seg]
        np_B = a_np * np_Q / (2.0 * cos_kd_2[np_seg])
        np_C = a_np * np_Q / (2.0 * sin_kd_2[np_seg])
        np_AC = -a_np * np_Q * recip_gap[np_seg]

        # Concatenate the three blocks and sort by seg-target to get CSR.
        # Within-seg ordering is unconstrained — every downstream consumer
        # (M_{A,B,C} scatter assignment, I_feed reduction,
        # _evaluate_basis_at_points scatter-add) is order-invariant
        # because each (basis, seg) coordinate pair is unique (a basis
        # contributes to a given seg at most once: as self, as N⁻
        # neighbour, or as N⁺ neighbour, never two of those).
        all_seg = np.concatenate([self_seg, nm_seg, np_seg])
        all_basis = np.concatenate([self_seg, nm_basis, np_basis])
        all_A = np.concatenate([A_i0_arr, nm_A, np_A]).astype(np.complex128)
        all_B = np.concatenate([B_i0_arr, nm_B, np_B]).astype(np.complex128)
        all_C = np.concatenate([C_i0_arr, nm_C, np_C]).astype(np.complex128)
        all_AC = np.concatenate([AC_i0_arr, nm_AC, np_AC]).astype(np.complex128)
        # Self entries have σ=+1; neighbour σ comes from geometry.
        self_sigma = np.ones(n_segs, dtype=np.int8)
        all_sigma = np.concatenate([self_sigma, nm_sigma, np_sigma])

        order = np.argsort(all_seg, kind="stable")
        counts = np.ones(n_segs, dtype=np.int64) + nm_count + np_count
        starts = np.empty(n_segs + 1, dtype=np.int64)
        starts[0] = 0
        np.cumsum(counts, out=starts[1:])

        A_ord, C_ord = all_A[order], all_C[order]
        seg_view = {
            "starts": starts,
            "jbasis": all_basis[order],
            "A": A_ord,
            "B": all_B[order],
            "C": C_ord,
            "AC": all_AC[order],
            "sigma": all_sigma[order],
        }
        self._cached_basis = (geom, k, a_key, seg_view)
        return seg_view

    def _evaluate_basis_at_points(self, seg_view, eval_seg, eval_s, alpha):
        """Vectorized evaluation of Σ_j α_j · f_{j, seg}(s_local) at an
        array of (segment, s_local) pairs.

        Replaces the per-knot `eval_at` Python closure: 342 individual
        calls per N=21 hentenna step (1.5 ms/step of Python frame +
        per-segment list iteration) collapse to one sin/cos call over
        n_eval points, one ragged gather, and one scatter-add.

        Parameters
        ----------
        seg_view : dict
            CSR-format inverse index from `_basis_coefs`.
        eval_seg : (n_eval,) int64
            Segment index per evaluation point.
        eval_s : (n_eval,) float64
            Local arc from each segment's centre.
        alpha : (n_basis,) complex128
            Basis amplitudes (from the EFIE solve).

        Returns
        -------
        (n_eval,) complex128
        """
        n_eval = eval_seg.shape[0]
        gathered = self._basis_entry_gather(seg_view, eval_seg)
        if gathered is None:
            return np.zeros(n_eval, dtype=np.complex128)
        entry_eval_idx, entry_global = gathered
        # Precompute trig at each eval point.
        sin_ks = np.sin(self.k * eval_s)
        # cos(kξ) − 1 = −2sin²(kξ/2), to a relative ε rather than an absolute
        # one — the literal subtraction loses everything below (kξ)²/2 (#606).
        _half_ks = np.sin(0.5 * self.k * eval_s)
        cosm1_ks = -2.0 * _half_ks * _half_ks
        jb = seg_view["jbasis"][entry_global]
        AC_e = seg_view["AC"][entry_global]
        B_e = seg_view["B"][entry_global]
        C_e = seg_view["C"][entry_global]
        sigma_e = seg_view["sigma"][entry_global]
        sin_e = sin_ks[entry_eval_idx]
        cosm1_e = cosm1_ks[entry_eval_idx]
        # σ·AC + B·sin kξ + σC·(cos kξ − 1) — the well-scaled spelling of
        # σA + B·sin kξ + σC·cos kξ (#606). Identical in exact arithmetic;
        # here it is the one whose terms are not larger than its answer.
        contrib = alpha[jb] * (sigma_e * AC_e + B_e * sin_e + sigma_e * C_e * cosm1_e)
        I_out = np.zeros(n_eval, dtype=np.complex128)
        np.add.at(I_out, entry_eval_idx, contrib)
        return I_out

    @staticmethod
    def _basis_entry_gather(seg_view, eval_seg):
        """CSR expansion of an evaluation set: which entries each point sums.

        Returns ``(entry_eval_idx, entry_global)`` — one row per (eval point,
        supporting basis entry) pair, mapping back to the point and forward
        into `seg_view`'s flat arrays — or ``None`` when there is nothing to
        sum (no points, or no entry on any of their segments).

        Pure index arithmetic, and shared by the value and the SLOPE
        evaluators because it is the same walk over the same support
        (momwire#611). The two differ only in the shape they put on each
        entry, which is the whole of what a derivative changes; keeping one
        gather means a change to the support cannot reach one reader and
        miss the other.
        """
        n_eval = eval_seg.shape[0]
        if n_eval == 0:
            return None
        starts = seg_view["starts"]
        starts_at = starts[eval_seg]  # (n_eval,)
        lengths = starts[eval_seg + 1] - starts_at  # entries per eval
        n_entries = int(lengths.sum())
        if n_entries == 0:
            return None
        # For each eval i with `lengths[i]` entries, produce that many
        # entry-level rows. `entry_eval_idx` maps each row back to its source
        # eval; `entry_global` gathers from `seg_view`.
        entry_eval_idx = np.repeat(np.arange(n_eval, dtype=np.int64), lengths)
        # within-segment offset of each entry within its eval's block:
        # arange(n_entries) - cumulative-start-per-eval-block
        cum_starts = np.empty(n_eval, dtype=np.int64)
        cum_starts[0] = 0
        if n_eval > 1:
            np.cumsum(lengths[:-1], out=cum_starts[1:])
        within = np.arange(n_entries, dtype=np.int64) - np.repeat(cum_starts, lengths)
        entry_global = np.repeat(starts_at, lengths) + within
        return entry_eval_idx, entry_global

    def _evaluate_basis_slope_at_points(self, seg_view, eval_seg, eval_s, alpha):
        """Vectorized ``Σ_j α_j · f'_{j, seg}(s_local)`` — the derivative twin
        of :meth:`_evaluate_basis_at_points`, same arguments, same shape out.

        Differentiating the well-scaled shape set {1, sin kξ, cos kξ − 1}
        term by term,

            f  = σ(A+C) + B·sin kξ + σC·(cos kξ − 1)
            f' = k·[ B·cos kξ − σC·sin kξ ]

        and the constant shape's coefficient — `AC`, the one momwire#606 had
        to rebuild in a per-branch closed form because A ≈ −C to O((kΔ)²)
        made the float sum worthless below kΔ ≈ 1e-4 — DIFFERENTIATES AWAY.
        So the cancellation that governs the value's accuracy is simply not
        in this expression: no term here is larger than the answer, and no
        well-scaled respelling is needed to keep it. Measured against a
        central difference of :meth:`currents_at_knots` on a dipole at
        N = 11…641 and k·Δ from 2.9e-1 down to 4.9e-8, the agreement is flat
        at ~1e-11 relative — the difference quotient's own floor — with no
        walk in either N or k·Δ (`test_sinusoidal_current_slopes.py`).

        ``cos kξ`` is evaluated literally rather than as ``1 + (cos kξ − 1)``:
        it tends to 1, not to 0, so there is nothing to cancel and the
        rearrangement would only cost an operation.
        """
        n_eval = eval_seg.shape[0]
        gathered = self._basis_entry_gather(seg_view, eval_seg)
        if gathered is None:
            return np.zeros(n_eval, dtype=np.complex128)
        entry_eval_idx, entry_global = gathered
        sin_ks = np.sin(self.k * eval_s)
        cos_ks = np.cos(self.k * eval_s)
        jb = seg_view["jbasis"][entry_global]
        B_e = seg_view["B"][entry_global]
        C_e = seg_view["C"][entry_global]
        sigma_e = seg_view["sigma"][entry_global]
        sin_e = sin_ks[entry_eval_idx]
        cos_e = cos_ks[entry_eval_idx]
        contrib = alpha[jb] * self.k * (B_e * cos_e - sigma_e * C_e * sin_e)
        out = np.zeros(n_eval, dtype=np.complex128)
        np.add.at(out, entry_eval_idx, contrib)
        return out

    # ------------------------------------------------------------------
    # Field of elementary current segments (Eqs 76-79)
    # ------------------------------------------------------------------

    def _field_tensor(
        self,
        geom,
        k,
        src_centers=None,
        src_tangents=None,
        obs_rows=None,
        cos_shape="cos",
    ):
        """Tangential-field tensor Φ of shape (3, N, N) where
        Φ[0, m, n] = ŝ_m · E^const_n(at center of m's surface),
        Φ[1, m, n] = ŝ_m · E^sin_n(at center of m's surface),
        Φ[2, m, n] = ŝ_m · E^cos_n(at center of m's surface).

        `cos_shape` selects the third shape exactly as
        `_field_components_bcast` documents: `"cos"` (the literal NEC shape)
        or `"cos-1"` (the well-scaled `cos kξ − 1`, momwire#205/#606). The
        C++ kernels transcribe the literal-cos closed forms ONLY, so a
        `"cos-1"` request takes the numpy reference path below regardless of
        accelerator availability — stated as a branch here rather than left
        to the kernel, which would otherwise hand back literal-cos tables for
        a folded-shape request (the same silent-wrong-answer trap momwire#246
        closed on the EK payload).

        The source's local frame is centered on segment n with z-axis
        along n's natural tangent. The "sin"/"cos" sources are
        sin(k·z'_local)/cos(k·z'_local) with z'_local measured from n's
        center along n's natural tangent. σ accounting is the caller's
        job — the tensor is in NATURAL-arc convention.

        `src_centers` / `src_tangents` default to the geometry's segment
        centers and tangents (free-space build). The PEC image build
        passes mirrored versions so the same tensor formula computes
        the image-source field at the original observer points.

        `obs_rows = (i0, i1)` restricts the OBSERVER axis to one row band,
        returning (3, i1-i0, N); the source axis is always whole. Every
        kernel below is already rectangular in the observer count — the
        mixed-radius path has dispatched per observer-row run since #147 —
        so a band costs exactly the pairs it names and nothing is
        recomputed across bands. `_assemble_Z` chunks on this to keep the
        fill's peak at one band rather than the whole tensor (#332).

        Hot path uses the C++ accelerator `sinusoidal_field_tensor` (the
        70% bottleneck of single-k solves at N≳80), or
        `sinusoidal_field_tensor_ek` when `extended_kernel` is set — the
        EKSCX transcription of momwire#245, which additionally takes the
        source radius and the per-end gating codes as (N,) tables. The
        pure-numpy formulation (`_field_components` + the tangential
        projection below) is kept as the reference for both, and as the
        fallback when the accelerator isn't available. Either C++ kernel
        takes a single scalar radius —
        the OBSERVER segment's (necpp EFLD) — so mixed per-wire radii
        dispatch one call per constant-radius observer-row run
        (stevenmburns/momwire#147). The
        reflection-coefficient finite-ground image block bypasses this
        method — see `_field_tensor_image_refl`, which applies the Fresnel
        field dyad pre-projection through its own kernel pair
        (`sinusoidal_field_tensor_refl` / `sinusoidal_field_tensor_ek_refl`).
        The PEC and Sommerfeld image blocks do NOT bypass it: both go
        through `_field_tensor_image`, so they ride whichever kernel this
        method picks.
        """
        seg_c = geom["seg_centers"]  # (N, 3) — observer centers
        seg_t = geom["seg_tangents"]  # (N, 3) — observer tangents
        seg_h = geom["seg_h"]  # (N,) full lengths

        src_c = src_centers if src_centers is not None else seg_c
        src_t = src_tangents if src_tangents is not None else seg_t
        # `slice(None)` — not `slice(0, N)` — on the whole-tensor call, so
        # the unchunked marshalling is byte-for-byte the pre-#332 one.
        win = slice(None) if obs_rows is None else slice(*obs_rows)

        # `extended_kernel=True` has had its own C++ entry point since
        # momwire#245 — EKSCX rather than EKSC, taking the SOURCE radius and
        # the per-end gating codes as (N,) tables alongside the observer-side
        # scalar radius. It is dispatched first, and separately, so the EK-OFF
        # marshalling below is untouched by it (which is what keeps the
        # default bit-exact; see the #233 off-path armor).
        literal_cos = cos_shape == "cos"
        if _HAVE_FIELD_TENSOR_EK and self.extended_kernel and literal_cos:
            gx, gw = self._leggauss_cached(self.n_qp_const)
            # Both tables are indexed by SOURCE segment, and the image build
            # mirrors the source geometry without reordering it, so the same
            # (N,) arrays serve both passes — which is also what NEC does,
            # EFLD passing one IND1/IND2 pair through both passes of its
            # KSYMP image loop (nec2-1.2.1.2.f:2914-2971).
            ind1, ind2 = self._ek_gating(geom)
            src_a = np.ascontiguousarray(self._seg_radius(geom), dtype=np.float64)

            def _call_ek(rows, a):
                return _acc.sinusoidal_field_tensor_ek(
                    np.ascontiguousarray(seg_c[rows], dtype=np.float64),
                    np.ascontiguousarray(seg_t[rows], dtype=np.float64),
                    np.ascontiguousarray(src_c, dtype=np.float64),
                    np.ascontiguousarray(src_t, dtype=np.float64),
                    np.ascontiguousarray(seg_h, dtype=np.float64),
                    float(a),
                    float(k),
                    float(self.eta),
                    np.ascontiguousarray(gx, dtype=np.float64),
                    np.ascontiguousarray(gw, dtype=np.float64),
                    src_a,
                    np.ascontiguousarray(ind1, dtype=np.int8),
                    np.ascontiguousarray(ind2, dtype=np.int8),
                    self._cancel_flag,
                )

            # The observer radius is still one scalar per call (EFLD's `ai`),
            # so mixed radii still split into constant-radius observer-row
            # runs. Spelled out rather than shared with the EK-OFF block
            # below so that block's diff stays empty.
            if self._uniform_radius is not None:
                return _call_ek(win, self._uniform_radius)
            parts = [
                _call_ek(slice(s, e), a)
                for s, e, a in self._radius_runs(geom, obs_rows)
            ]
            return tuple(
                np.concatenate([p[i] for p in parts], axis=0) for i in range(3)
            )

        # The reduced-kernel C++ kernels transcribe EKSC only, so an EK-ON
        # solve reaches this branch only when the accelerator is unavailable —
        # in which case it takes the numpy reference path below, same as an
        # EK-OFF solve would.
        if _HAVE_FIELD_TENSOR and not self.extended_kernel and literal_cos:
            gx, gw = self._leggauss_cached(self.n_qp_const)

            def _call(rows, a):
                return _acc.sinusoidal_field_tensor(
                    np.ascontiguousarray(seg_c[rows], dtype=np.float64),
                    np.ascontiguousarray(seg_t[rows], dtype=np.float64),
                    np.ascontiguousarray(src_c, dtype=np.float64),
                    np.ascontiguousarray(src_t, dtype=np.float64),
                    np.ascontiguousarray(seg_h, dtype=np.float64),
                    float(a),
                    float(k),
                    float(self.eta),
                    np.ascontiguousarray(gx, dtype=np.float64),
                    np.ascontiguousarray(gw, dtype=np.float64),
                    self._cancel_flag,
                )

            if self._uniform_radius is not None:
                return _call(win, self._uniform_radius)
            # Mixed per-wire radii: the radius is the OBSERVER segment's
            # (necpp EFLD convention) and the C++ kernel takes one scalar,
            # so dispatch one call per contiguous constant-radius run of
            # observer rows and stitch — segments are wire-contiguous, so
            # runs are few (same pattern as the bspline kernels, #147).
            parts = [
                _call(slice(s, e), a) for s, e, a in self._radius_runs(geom, obs_rows)
            ]
            return tuple(
                np.concatenate([p[i] for p in parts], axis=0) for i in range(3)
            )

        # numpy fallback: unprojected per-shape (E_z, E_ρ) components,
        # then project tangentially onto the observer:
        #   E_t = td · E_z + rho_proj · E_ρ  (NEC's ρ-projection rule).
        cm = self._field_components(
            geom,
            k,
            src_centers=src_c,
            src_tangents=src_t,
            cos_shape=cos_shape,
            **self._obs_window_kwargs(geom, obs_rows),
        )
        td = cm["td"]
        rho_proj_factor = cm["rho_proj_factor"]
        Phi_const = td * cm["Ez_const"] + rho_proj_factor * cm["Erho_const"]
        Phi_sin = td * cm["Ez_sin"] + rho_proj_factor * cm["Erho_sin"]
        Phi_cos = td * cm["Ez_cos"] + rho_proj_factor * cm["Erho_cos"]
        return Phi_const, Phi_sin, Phi_cos

    # ---- NEC's extended thin-wire kernel (EK card), momwire#233 -----------
    #
    # Theory manual Eqs 84-98 (Burke & Poggio Part I, pp. 30-34). The reduced
    # kernel treats the segment current as a filament on the wire axis and
    # keeps the observer on the wire surface, so the source-observer distance
    # never falls below a; that is Eq 84's R = sqrt((z-z')² + a²), and it is
    # what every momwire solver computes. The EXTENDED kernel instead keeps
    # the current as a uniform tube of surface current at ρ' = a and averages
    # the free-space Green's function over the circumference (Eq 85). Eqs
    # 86-88 expand that average in a Taylor series about the axial filament
    # and truncate at second order, which is exact to O(a²/R²) and — crucially
    # — reintroduces the ρ' ≠ 0 terms the reduced kernel drops. Eq 89 is the
    # resulting scalar kernel and Eqs 90-98 are its z- and ρ-derivatives, the
    # six per-end quantities the field expressions consume.
    #
    # NEC's implementation of all of this is `GXX` (nec2-1.2.1.2.f:4857-4897),
    # which stands in for the reduced-kernel `GX` (f.4842-4852) at ONE END of
    # ONE SOURCE SEGMENT at a time, and `EKSCX` (f.3170-3234), which is
    # `EKSC` (f.3124-3166) with GX swapped for GXX per end plus a correction
    # to the constant-current term. momwire's `_field_components_bcast` is
    # algebraically EKSC rearranged into per-endpoint brackets (verified term
    # by term — see `test_extended_kernel_zero_radius_limit_is_the_reduced
    # _kernel`), so the extended kernel drops into the same slots.

    def _ek_gating(self, geom):
        """Per-source-segment extended-kernel end gating — NEC's IND1/IND2.

        NEC does NOT apply the extended kernel unconditionally. Once per
        SOURCE segment it asks, separately for each end, whether the segment's
        sinusoidal current really continues straight through that end into an
        identical conductor; only then is the O(a²) surface-current expansion
        legitimate there. The decision lives in the matrix fill, not in the
        kernel: nec2-1.2.1.2.f lines 2019-2053 (repeated verbatim at
        6197-6224 and 7217-7244 for the other two fill routines), guarded by
        `IF( IEXK.EQ.0) GOTO 16` at f.2019. The three codes it produces are
        consumed by EKSCX at f.3193 (`IF( INX1.EQ.2) GOTO 3`) and f.3199,
        which route the end to the reduced GX for code 2 and to the extended
        GXX for codes 0 and 1:

          IND = 1  Free end — ICONn(J) == 0 (f.2032 / f.2049). Nothing is
                   attached, so nothing can violate the expansion; NEC
                   extends.
          IND = 0  Either (a) a simple two-segment junction whose partner is
                   collinear to |t·t'| >= 0.999999 (f.2040-2041) AND of equal
                   radius to |a'/a - 1| <= 1e-6 (f.2042-2043), or (b) a
                   ground-plane contact, ICONn(J) == J, by a segment
                   perpendicular to the plane — NEC's CABJ²+SABJ² <= 1e-8
                   test at f.2026-2027 / f.2043-2044 — where the image
                   continues the wire straight through.
          IND = 2  Everything else: a bend, a radius step, a non-perpendicular
                   ground contact, or a multi-way junction. The last is
                   implicit: NEC threads 3+ segments meeting at a node onto a
                   circular ICON chain, so the reciprocal test at f.2025
                   (`IF(-ICON1(IPR).NE.J)`) / f.2031 (`IF(ICON2(IPR).NE.J)`)
                   fails for every member as soon as K >= 3.

        momwire's N⁻/N⁺ neighbour tables carry exactly this information. End 1
        is the N⁻ side and end 2 the N⁺ side (`nm_seg[k]` is documented as
        "j's NEC end-2 coincides with i's end-1", so the arc orientation
        matches NEC's ICON1/ICON2 convention). A neighbour count of exactly 1
        is precisely NEC's reciprocal-ICON test: momwire emits K(K-1) edges at
        a K-member junction, so K == 2 leaves count 1 on both members and
        K >= 3 leaves count >= 2 — the case NEC's chain rejects.

        Returns `(ind1, ind2)`, int8 arrays of shape (N,) holding NEC's codes.
        """
        geom_key = self._cached_ek_gating
        if geom_key is not None and geom_key[0] is geom:
            return geom_key[1], geom_key[2]

        n = geom["n_segs"]
        t = geom["seg_tangents"]  # (N, 3) unit tangents, NEC arc order
        rad = self._seg_radius(geom)  # (N,)
        # NEC's CABJ² + SABJ² is the squared HORIZONTAL projection of the
        # segment tangent (f.2026): the ground plane is z = const, so a
        # segment is perpendicular to it exactly when t_x² + t_y² vanishes.
        horiz = t[:, 0] * t[:, 0] + t[:, 1] * t[:, 1]
        perpendicular_to_plane = horiz <= 1e-8

        inds = []
        for count, basis, seg, ground in (
            (geom["nm_count"], geom["nm_basis"], geom["nm_seg"], geom["ground_minus"]),
            (geom["np_count"], geom["np_basis"], geom["np_seg"], geom["ground_plus"]),
        ):
            # Default IND = 2: reduced kernel at this end unless proven safe.
            ind = np.full(n, 2, dtype=np.int8)
            # Free end (f.2032): no neighbour entry AND no ground contact.
            ind[(count == 0) & ~ground] = 1
            # Ground contact (f.2026-2029): NEC's ICON == J self-connection.
            # momwire records it as a `ground_minus`/`ground_plus` flag and
            # emits no neighbour edge, so it never collides with the branch
            # below.
            ind[ground & perpendicular_to_plane] = 0
            # Simple two-segment junction: collinear AND equal radius.
            single = np.flatnonzero(count == 1)
            if single.size:
                # count == 1 means exactly one table entry names this basis,
                # so a last-write-wins scatter resolves the neighbour.
                nbr = np.empty(n, dtype=np.int64)
                nbr[basis] = seg
                j = nbr[single]
                # ABS() at f.2040: σ = ±1 records whether the neighbour's
                # natural tangent is arc-flipped, and NEC likewise compares
                # magnitudes, so an antiparallel collinear pair still counts.
                dot = np.abs(np.einsum("id,id->i", t[single], t[j]))
                ok = (dot >= 0.999999) & (np.abs(rad[j] / rad[single] - 1.0) <= 1e-6)
                ind[single[ok]] = 0
            inds.append(ind)

        self._cached_ek_gating = (geom, inds[0], inds[1])
        return inds[0], inds[1]

    @staticmethod
    def _ek_end_gxx(k, zz, rh, b, want_swapped):
        """NEC's GXX (f.4857-4897): extended-kernel segment-end contributions.

        `zz` is NEC's ZZ — the axial distance from the observer to this END of
        the source segment, signed as NEC signs it (Z2 = H - z, Z1 = -(H + z)).
        `rh` is the radial distance and `b` the radius entering the O(a²)
        terms; EKSCX has already ordered them so that `rh >= b` (see
        `_extended_kernel_fields`). `want_swapped` selects EKSCX's IRA == 1
        arm — the case where the observation point lies INSIDE the source
        conductor's radius and the roles of "distance" and "radius" trade
        places (f.4886-4896). It is PER PAIR: NEC sets IRA inside EKSCX, once
        per segment pair (f.3186-3192), so a boolean array broadcastable
        against `rh` selects the arm elementwise. Before momwire#258 this was
        one scalar for the whole fill, and a single observation point inside a
        source conductor put every other pair on the IRA==1 formula evaluated
        with ITS unswapped (rh, b) — not the physics for those pairs, and
        visible wherever the ρ-projection does not vanish.

        A scalar `False` (or `True`) is still accepted and takes a single arm
        with no `np.where`: `_extended_kernel_fields` passes `False` whenever
        no pair swaps, which is every uniform-radius deck, and that is the
        spelling whose rounding the #233 ladder is frozen against.

        Returns the six quantities EKSCX consumes, in NEC's argument order:
        `(G1, G1P, G2, G2P, G3, GZP)` — respectively the extended scalar
        kernel (Eq 89), its z-derivative, the ρ-kernel and its ρ- and
        z-derivatives (Eqs 90-96), and the reduced-kernel z-derivative that
        only the constant-current correction term uses.
        """
        # Multi-step spelling throughout (momwire#205, momwire#392): a complex
        # product whose RIGHT operand is a dead temporary is elided into an
        # in-place multiply above numpy's 256 KB threshold, and the in-place
        # loop rounds differently — which would make the fill depend on block
        # size. Every such operand is bound to a name first.
        r2 = zz * zz + rh * rh
        r = np.sqrt(r2)
        kr = k * r
        kr2 = kr * kr
        # C1, C2, C3 of f.4869-4871 — the polynomial coefficients of the
        # second-order Taylor terms of Eqs 86-88.
        c1 = 1.0 + 1j * kr
        c2 = 3.0 * c1 - kr2
        c3 = (6.0 + 1j * kr) * kr2 - 15.0 * c1
        a2 = b * b
        # T1, T2 of f.4867-4868: the two O(a²) shape factors of Eq 89.
        rh2 = rh * rh
        r4 = r2 * r2
        t1 = 0.25 * a2 * rh2 / r4
        t2 = 0.5 * a2 / r2
        gz = np.exp(-1j * kr) / r  # reduced kernel e^{-jkR}/R
        # Eq 89: the circumferentially averaged kernel, ρ-flavoured (G2) and
        # z-flavoured (G1, which carries the extra -T2·C1 term).
        f2 = 1.0 + t1 * c2
        g2 = gz * f2
        g1 = g2 - t2 * c1 * gz
        gzr = gz / r2
        f2p = t1 * c3 - c1
        g2p = gzr * f2p
        gzp_t = t2 * c2 * gzr
        g3 = g2p + gzp_t
        g1p = g3 * zz
        # GZP: the plain reduced-kernel z-derivative. It is a-independent and
        # only enters EKSCX's constant-current term, scaled by (ka/2)².
        gzp_out = -zz * c1 * gzr

        def _ira1():
            # IRA == 1 (f.4886-4896): observation point inside the conductor.
            t2b = 0.5 * b
            g2_s = -t2b * c1 * gzr
            g2p_s = t2b * gzr * c2 / r2
            g3_s = rh2 * g2p_s - b * gzr * c1
            return g2_s, g2p_s * zz, g3_s

        def _ira0():
            # IRA == 0 (f.4879-4885), the ordinary case. `rh` is never zero in
            # momwire — it is at least the observer radius — so NEC's
            # RH < 1e-10 guard at f.4881 is unreachable here.
            return g2 / rh, g2p * zz / rh, (g3 + gzp_t) * rh

        if np.ndim(want_swapped) == 0:
            # One arm for the whole call. This is the pre-#258 spelling and it
            # is kept for the (overwhelmingly common) uniform-radius case,
            # where no pair can swap: it is the rounding the #233 ladder is
            # frozen against, and it does not pay for a second arm.
            g2_out, g2p_out, g3_out = _ira1() if want_swapped else _ira0()
        else:
            # Per pair (f.3186-3192). Both arms are finite everywhere on this
            # grid — the IRA==0 divisions are by `rh`, the LARGER of the two
            # lengths, which is never zero — so evaluating both and selecting
            # elementwise is safe; no masked evaluation is needed.
            g2_out, g2p_out, g3_out = tuple(
                np.where(want_swapped, s, o) for s, o in zip(_ira1(), _ira0())
            )
        return g1, g1p, g2_out, g2p_out, g3_out, gzp_out

    @staticmethod
    def _ek_end_gx(k, zz, rhx):
        """NEC's GX (f.4842-4852) repackaged into GXX's six-value contract.

        EKSCX's `INX == 2` arm (f.3195-3201, f.3201-3207) calls the REDUCED
        end routine and then rescales its two outputs into the same six slots
        GXX fills. Note it passes RHX — the ORIGINAL radial distance — not the
        possibly-swapped RH, and it zeroes GZP so the end contributes nothing
        to the constant-current correction.
        """
        r2 = zz * zz + rhx * rhx
        r = np.sqrt(r2)
        kr = k * r
        c1 = 1.0 + 1j * kr
        gz = np.exp(-1j * kr) / r
        gp = -c1 * gz / r2
        g1p = gp * zz
        return gz, g1p, gz / rhx, g1p / rhx, gp * rhx, np.zeros_like(gz)

    def _ek_rho_cos1_gap(self, k, z1, z2, rhx, rh, b, ind1, ind2, swap):
        """`X = Ψ(z₂) − Ψ(z₁)` for EKSCX's ρ-tables, cancellation-free (#614).

        `Ψ(ζ) = ζ·G2P + G2 + G3` on the six values `_ek_end_gxx`/`_ek_end_gx`
        return, and `X` is the combination EKSCX's `Erho_cos` and
        `Erho_const` differ by:

            Erho_cos − Erho_const
                = −con·[ X + D·(cos kH − 1) + (z₂G2₂ + z₁G2₁)·sin(kH)·k ]

        `X` is O((kΔ)²) — it has to be, since cos kξ → 1 makes the cos-shape
        ρ-field the const-shape one — but the literal combination is a sum of
        O(1) terms and bottoms out at ε: measured 8.8e-16 relative at
        kΔ = 1.9e-8, i.e. no correct digits, and floored at ~3e-13 absolute
        from kΔ = 1.9e-6 down. This is its closed form.

        The derivation, per END. With `r² = ζ² + rh²` the three returns group
        so that the `rh` powers cancel:

            Ψ = [r²·g2pᴸ + g2ᴸ + 2rh²·gzp_t] / rh
              = gz·[ (1 − c1) + t1·(c3 + 5·c2) ] / rh

        and both brackets collapse exactly: `1 − c1 = −j·kr`, while

            c3 + 5c2 = (6 + j·kr)kr² − 15c1 + 15c1 − 5kr² = kr²·c1

        leaving, since `gz·kr = k·e^{−jkr}`,

            Ψ_IRA0 = k·e^{−jkr}·(−j + t1·kr·c1) / rh.

        The other two arms fall out of the same algebra:

            Ψ_IRA1 = −½·b·k²·e^{−jkr} / r      (0.5c2 − 1.5c1 = −½kr²)
            Ψ_GX   = −j·k·e^{−jkr} / rhx       (the b → 0 limit)

        Only the `−j·k·e^{−jkr}` term is O(k) rather than O(k²), so only its
        difference between the two ends needs care — and there the exponential
        difference is spelled `e^{−jkr₁}·(e^{−jk(r₂−r₁)} − 1)` through
        `_expm1_neg_j`, with

            r₂ − r₁ = (z₂² − z₁²)/(r₁ + r₂) = −4·H·z/(r₁ + r₂)

        so neither the exponential nor the square-root difference is a
        subtraction of two quantities about to agree.

        That leading term always carries the `rhx` denominator, on every arm
        that has one: `IRA1` replaces `IRA0` exactly when `swap` is set, and
        `swap` is exactly when `rh` and `rhx` differ — so an `IRA0` end always
        has `rh == rhx`, and a `GX` end uses `rhx` by NEC's own choice
        (f.3195-3207). The two ends can therefore share one radius.
        """
        z1 = np.asarray(z1)
        z2 = np.asarray(z2)
        # GX/IRA0 radius (they coincide wherever both can occur; see above).
        r1 = np.sqrt(z1 * z1 + rhx * rhx)
        r2 = np.sqrt(z2 * z2 + rhx * rhx)
        e1 = np.exp(-1j * k * r1)
        e2 = np.exp(-1j * k * r2)

        ext1 = ind1 != 2
        ext2 = ind2 != 2
        # Which ends carry the O(k) leading term at all: everything but IRA1.
        lead1 = ~(ext1 & swap)
        lead2 = ~(ext2 & swap)

        # (z₂² − z₁²)/(r₁ + r₂) — exact, and free of the sqrt subtraction.
        dr = (z2 * z2 - z1 * z1) / (r1 + r2)
        both = lead1 & lead2
        # Paired: one stable expm1. Otherwise at most one term survives, so
        # the plain difference is already at the answer's size.
        d_lead_paired = e1 * _expm1_neg_j(k * dr)
        d_lead_plain = np.where(lead2, e2, 0.0) - np.where(lead1, e1, 0.0)
        d_lead = np.where(both, d_lead_paired, d_lead_plain)
        out = -1j * k * d_lead / rhx

        # The O(k²) remainder, per end, on whichever arm that end took.
        def _q(zz, rr, ee, ext):
            r2_ = rr * rr
            c1 = 1.0 + 1j * k * rr
            t1 = 0.25 * (b * b) * (rh * rh) / (r2_ * r2_)
            q_ira0 = (k * k) * rr * t1 * c1 * ee / rh
            q_ira1 = -0.5 * b * (k * k) * ee / rr
            q = np.where(swap, q_ira1, q_ira0)
            return np.where(ext, q, np.zeros_like(q))

        # IRA1's own radius is built on `rh`, not `rhx`; they differ only
        # where `swap`, which is exactly where IRA1 is the arm.
        r1h = np.where(swap, np.sqrt(z1 * z1 + rh * rh), r1)
        r2h = np.where(swap, np.sqrt(z2 * z2 + rh * rh), r2)
        e1h = np.where(swap, np.exp(-1j * k * r1h), e1)
        e2h = np.where(swap, np.exp(-1j * k * r2h), e2)
        out = out + _q(z2, r2h, e2h, ext2) - _q(z1, r1h, e1h, ext1)
        return out

    def _extended_kernel_fields(
        self, k, H, z_eval, rho_eval, src_a, ind1, ind2, cos_shape="cos"
    ):
        """NEC's EKSCX (f.3170-3234) — the extended-kernel field tables.

        Drop-in replacement for the reduced-kernel body of
        `_field_components_bcast`, returning the same six per-shape tables.

        `cos_shape` picks the third source shape exactly as
        `_field_components_bcast` documents. `"cos-1"` (momwire#614) is NOT a
        second kernel: it is the SAME G-quantities rearranged, because the
        folded shape is `cos kξ − 1` and EKSCX's own `cos` and `const` tables
        already carry both halves. Subtracting them ALGEBRAICALLY leaves only
        terms that are explicitly O((kΔ)²) —

            Ez:   (g1₂+g1₁)·ss·k, (g1p₂−g1p₁)·(cs−1),
                  k²(1−bk2)·int_G0, bk2·d_gzz
            Erho: X, D·(cs−1), (z₂G2₂+z₁G2₁)·ss·k

        — where `cs − 1` is spelled `−2sin²(kH/2)` and `X` comes from
        `_ek_rho_cos1_gap`'s closed form. Subtracting them NUMERICALLY does
        not work and is the trap this route exists to avoid: measured on a
        fat-wire deck, `(|cos|+|const|)/|cos−const|` is 4.3e8 at kΔ = 1.9e-4
        and 1.4e16 at 1.9e-8, so the difference of the two computed tables has
        no digits left exactly where the folded shape is wanted.

        `H` is the source HALF length, `z_eval` the observer's axial
        coordinate in the source frame, `rho_eval` the a-regularized radial
        distance (NEC's RHX out of EFLD), `src_a` the SOURCE segment's radius
        (NEC's BX = BI(J) — not the observer radius the reduced path
        regularizes with), and `ind1`/`ind2` the per-end gating codes from
        `_ek_gating`, broadcastable against the source axis.
        """
        if cos_shape not in ("cos", "cos-1"):
            raise ValueError(f"cos_shape must be 'cos' or 'cos-1', got {cos_shape!r}")
        # f.3186-3192: order the two lengths so RH is the larger. When the
        # observation point falls inside the source conductor the roles trade
        # and IRA is set; every downstream use of RH — including the INTX
        # integration radius and the (kB/2)² correction factor — sees the
        # ordered pair, not the raw one.
        #
        # Both the ordering AND the IRA arm it selects are PER PAIR, which is
        # where NEC sets them. Until momwire#258 only the ordering was: the
        # arm was `np.any(swap)`, one branch for the whole fill, so a single
        # observation point inside a source conductor put every pair on the
        # IRA==1 formula — evaluated with the unswapped (rh, b) of the pairs
        # that did not swap. That is masked on a collinear deck (the arm
        # rewrites only the ρ-flavoured slots and the ρ-projection vanishes
        # there) and worth ~20% of the tensor as soon as a skew member is
        # present.
        rhx = rho_eval
        swap = rhx < src_a
        any_swap = bool(np.any(swap))
        rh = (
            np.where(swap, src_a, rhx) if any_swap else np.broadcast_to(rhx, swap.shape)
        )
        b = (
            np.where(swap, rhx, src_a)
            if any_swap
            else np.broadcast_to(src_a, swap.shape)
        )

        # NEC's Z2 = SH - Z and Z1 = -(SH + Z); momwire's dz_i is -Z_i.
        z2 = H - z_eval
        z1 = -(H + z_eval)
        ss = np.sin(k * H)
        cs = np.cos(k * H)

        ends = []
        for zz, ind in ((z1, ind1), (z2, ind2)):
            zz = np.broadcast_to(zz, swap.shape)
            ext = np.broadcast_to(ind != 2, swap.shape)
            # `False` rather than the all-False mask when nothing swaps: same
            # answer, one arm instead of two, and bit-for-bit the pre-#258
            # spelling on every deck that never reached the arm.
            q_ext = self._ek_end_gxx(k, zz, rh, b, swap if any_swap else False)
            if ext.all():
                ends.append(q_ext)
                continue
            q_red = self._ek_end_gx(k, zz, rhx)
            ends.append(tuple(np.where(ext, e, r) for e, r in zip(q_ext, q_red)))
        (g1_1, g1p_1, g2_1, g2p_1, g3_1, gzz_1) = ends[0]
        (g1_2, g1p_2, g2_2, g2p_2, g3_2, gzz_2) = ends[1]

        # CON of f.3181 — NEC's DATA CONX/0., 4.771341189/ is jη/(4πk) with
        # NEC's k = 2π. Identical to `_field_components_bcast`'s `pref_z`.
        con = 1j * self.eta / (4.0 * np.pi * k)

        # f.3213-3222, verbatim. These are algebraically the same brackets the
        # reduced path spells per endpoint; only the G-quantities changed.
        ez_sin = con * ((g1_2 - g1_1) * cs * k - (g1p_2 + g1p_1) * ss)
        ez_cos = -con * ((g1_2 + g1_1) * ss * k + (g1p_2 - g1p_1) * cs)
        erho_sin = -con * (
            (z2 * g2p_2 + z1 * g2p_1 + g2_2 + g2_1) * ss
            - (z2 * g2_2 - z1 * g2_1) * cs * k
        )
        erho_cos = -con * (
            (z2 * g2p_2 - z1 * g2p_1 + g2_2 - g2_1) * cs
            + (z2 * g2_2 + z1 * g2_1) * ss * k
        )
        erho_const = con * (g3_2 - g3_1)

        # f.3223-3227: the constant-current term. INTX integrates e^{-jkR}/R
        # along the segment at the ORDERED radius RH, and the extended kernel
        # scales that integral by (1 - (kB/2)²) while adding back a (kB/2)²
        # weighted reduced-kernel end difference — the O(a²) piece of Eq 98
        # that the ρ-expansion of the axial term leaves behind. Unlike the
        # per-end substitutions this factor is NOT gated: it applies whenever
        # EK is on, even when both ends fell back to GX.
        # `asinh_diff` on the ORDERED radius (momwire#799), the reduced
        # tensor's treatment verbatim. `rh` is not `rho_eval`, so the two end
        # distances are re-formed here rather than reused.
        u2 = H - z_eval
        u1 = -H - z_eval
        rh2 = rh * rh
        rr2 = np.sqrt(rh2 + u2 * u2)
        rr1 = np.sqrt(rh2 + u1 * u1)
        int_inv_r0 = asinh_diff(u1, u2, rh2, rr1, rr2)
        gx, gw = self._leggauss_cached(self.n_qp_const)
        z_qp = H[..., None] * gx
        dz_qp = z_eval[..., None] - z_qp
        r0_qp = np.sqrt(rh[..., None] ** 2 + dz_qp**2)
        # (e^{-jkr} - 1)/r, NOT G0 - 1/r: the literal difference of two
        # 1/r-sized terms returns its real part to an absolute ε, which is
        # 1.1e-12 relative at kr = 8e-3 and grows as 1/(kr)² (momwire#799).
        # Through the HALF phase, which is the form the C++ twin's phase table
        # already holds — same spelling in both lanes, not merely the same
        # value.
        reg_qp = _expm1_neg_j_from_half(-0.5 * k * r0_qp) / r0_qp
        int_G0 = int_inv_r0 + np.einsum("...q,q->...", reg_qp, gw) * H
        bk = k * b
        bk2 = 0.25 * bk * bk
        d_gzz = gzz_2 - gzz_1  # named: `bk2 * (…)` would otherwise elide (#392)
        ez_const = -con * (g1p_2 - g1p_1 + k * k * (1.0 - bk2) * int_G0 - bk2 * d_gzz)
        if cos_shape == "cos-1":
            # The folded shape, algebraically (momwire#614). `cs - 1` is the
            # only place a subtraction could hide, and it is spelled as its
            # half-angle square; every other term already carries an explicit
            # k², k·ss or bk2 factor.
            half_h = np.sin(0.5 * k * H)
            cs_m1 = -2.0 * half_h * half_h
            ez_cos = -con * (
                (g1_2 + g1_1) * ss * k
                + (g1p_2 - g1p_1) * cs_m1
                - k * k * (1.0 - bk2) * int_G0
                + bk2 * d_gzz
            )
            d_g2p = z2 * g2p_2 - z1 * g2p_1 + g2_2 - g2_1
            x_gap = self._ek_rho_cos1_gap(
                k, z1, z2, rhx, rh, b, ind1, ind2, swap if any_swap else False
            )
            erho_cos = -con * (x_gap + d_g2p * cs_m1 + (z2 * g2_2 + z1 * g2_1) * ss * k)
        return {
            "Erho_const": erho_const,
            "Ez_const": ez_const,
            "Erho_sin": erho_sin,
            "Ez_sin": ez_sin,
            "Erho_cos": erho_cos,
            "Ez_cos": ez_cos,
        }

    def _field_components(
        self,
        geom,
        k,
        src_centers=None,
        src_tangents=None,
        obs_centers=None,
        obs_tangents=None,
        obs_radius=None,
        cos_shape="cos",
    ):
        """Pure-numpy unprojected field tables behind `_field_tensor`'s
        fallback path (Eqs 76-79 of the NEC2 Theory Manual).

        `cos_shape` is forwarded to `_field_components_bcast` unchanged —
        see its docstring for the two shapes and what each is for.

        Observers default to the geometry's segment centres (collocation).
        A Galerkin test caller (`SinusoidalGalerkinSolver`) overrides
        `obs_centers` / `obs_tangents` / `obs_radius` to evaluate the same
        closed-form source fields at Gauss quadrature points ALONG each test
        segment; the source side is unaffected (still the geometry's segments
        unless `src_*` is overridden by an image build). With M observers and
        N sources every returned table is (M, N).

        For each (observer m, source n) pair and each of the three source
        current shapes (const / sin / cos), computes the field scalars in
        the SOURCE's local cylindrical frame: E = E_z·t_src + E_ρ·ρ̂ with
        ρ̂ = rho_vec/rho_eval. Returns a dict of (M, N) tables:

            Erho_const, Ez_const, Erho_sin, Ez_sin, Erho_cos, Ez_cos —
                per-shape field scalars;
            td              : t_obs · t_src (E_z projection factor);
            rho_proj_factor : (rho_vec · t_obs)/rho_eval (E_ρ projection);
            rho_vec         : (M, N, 3) perpendicular from source axis to
                              the observer point;
            rho_eval        : radius-regularized |rho_vec|, ≥ a.

        `_field_tensor` consumes td/rho_proj_factor for the plain
        tangential projection; `_field_tensor_image_refl` also needs
        rho_vec/rho_eval to resolve E·p̂ for the Fresnel field dyad.
        """
        # Thin-wire surface offset: the OBSERVER segment's radius. NEC
        # keeps the source current a filament on the source axis and
        # enforces the boundary condition on the observing segment's
        # surface — nec2c/necpp pass ai = segment_radius[i] (the segment
        # the field is evaluated on) into EFLD, where it regularizes
        # rh = sqrt(rho² + ai²). Self terms therefore use the wire's own
        # radius, and mutual terms between wires of different radii use
        # the OBSERVER wire's radius. (The opposite source-radius
        # convention was tried first and refuted by the PyNEC oracle: the
        # mixed-radius delta grew with mesh refinement near junctions.)
        # Scalar on the uniform path; an (M, 1) per-observer column when
        # mixed. Observers are always the real segments — image builds
        # only mirror the SOURCE side — so this indexes geom directly.
        if obs_radius is not None:
            a = obs_radius
        elif self._uniform_radius is not None:
            a = self._uniform_radius
        else:
            a = self._seg_radius(geom)[:, None]
        seg_c = geom["seg_centers"] if obs_centers is None else obs_centers  # (M,3)
        seg_t = geom["seg_tangents"] if obs_tangents is None else obs_tangents  # (M,3)
        seg_h = geom["seg_h"]  # (N,) SOURCE full lengths
        h_n = 0.5 * seg_h  # (N,) source half-lengths

        # Sources default to the geometry's segments — NOT the observer set,
        # which may be quadrature points; image builds override src_* directly.
        src_c = src_centers if src_centers is not None else geom["seg_centers"]
        src_t = src_tangents if src_tangents is not None else geom["seg_tangents"]

        # Extended thin-wire kernel (#233): the SOURCE segment's radius and
        # the per-end gating codes, both indexed by source segment. Image
        # builds mirror the source geometry without reordering it, so the same
        # (N,) tables apply there — which is also what NEC does, EFLD passing
        # one IND1/IND2 pair through both passes of its KSYMP image loop
        # (nec2-1.2.1.2.f:2914-2971). (N,) broadcasts against (M, N).
        ek = None
        if self.extended_kernel:
            ind1, ind2 = self._ek_gating(geom)
            src_a = (
                self._uniform_radius
                if self._uniform_radius is not None
                else self._seg_radius(geom)
            )
            ek = (src_a, ind1, ind2)

        # Every-observer × every-source: insert the singleton axes that make
        # the shared core broadcast to the (M, N) outer product.
        return self._field_components_bcast(
            k,
            obs_c=seg_c[:, None, :],  # (M, 1, 3)
            obs_t=seg_t[:, None, :],  # (M, 1, 3)
            a=a,  # scalar or (M, 1)
            src_c=src_c[None, :, :],  # (1, N, 3)
            src_t=src_t[None, :, :],  # (1, N, 3)
            src_hh=h_n[None, :],  # (1, N)
            cos_shape=cos_shape,
            ek=ek,
        )

    @staticmethod
    def _pair_geometry(obs_c, obs_t, a, src_c, src_t):
        """The (observer, source) frame every Eqs 76-79 evaluation starts from:
        `(z_eval, rho_eval, rho_vec, td, rho_proj_factor)`, each broadcast to
        whatever shape the position/tangent arguments broadcast to.

        Lifted verbatim out of `_field_components_bcast` (its only caller until
        momwire#299) so that the Galerkin fill's post-fill end-bracket
        correction — which has to evaluate a piece of the SAME kernel at the
        SAME pairs, and whose whole job is to cancel a term the fill already
        added — reads the frame off one spelling instead of a transcription of
        it. Nothing here depends on the source shape or on the kernel choice.
        """
        # Pairwise separation obs - src; shape is whatever the two broadcast to.
        rvec = obs_c - src_c
        t_src = src_t
        t_obs = obs_t

        z_eval = np.einsum("...d,...d->...", rvec, t_src)
        # Perpendicular component:
        rho_vec = rvec - z_eval[..., None] * t_src  # (M, N, 3)
        rho_axis = np.linalg.norm(rho_vec, axis=-1)  # (M, N)
        rho_eval = np.sqrt(rho_axis * rho_axis + a * a)  # (M, N), >= a

        # Tangent dot products; t_obs is shape (M, 1, 3), broadcasting
        # handles the singletons.
        td = (t_obs * t_src).sum(axis=-1)  # (M, N)
        # rho_vec · t_obs at the observer (rho_vec is the perpendicular
        # vector from source axis to obs-axis center)
        rho_dot_tobs = (rho_vec * t_obs).sum(axis=-1)  # (M, N)
        # NEC's prescription: tangential E_ρ component is (ρ·ŝ)/ρ' · E_ρ.
        rho_proj_factor = rho_dot_tobs / rho_eval  # (M, N)
        return z_eval, rho_eval, rho_vec, td, rho_proj_factor

    def _field_components_bcast(
        self, k, obs_c, obs_t, a, src_c, src_t, src_hh, cos_shape="cos", ek=None
    ):
        """Shape-agnostic core of `_field_components` (Eqs 76-79).

        Every argument is broadcast against the others, so the caller — not
        this method — decides the pairing. `_field_components` passes
        (M,1,·) observers against (1,N,·) sources for the full outer product;
        `SinusoidalGalerkinSolver`'s near-singular correction passes (P,G,·)
        quadrature points against (P,1,·) sources to get just the P near
        pairs evaluated, without ever forming the N² table those pairs live
        in. Returns the same dict of field tables, each of the common
        broadcast shape S (`rho_vec` is S + (3,)).

        `obs_c`/`obs_t`/`src_c`/`src_t` are position/tangent arrays whose last
        axis is the 3 spatial components; `src_hh` is the source segment's
        HALF length; `a` is the observer-side thin-wire radius.

        `cos_shape` selects which third SOURCE shape the `*_cos` tables carry
        (momwire#205):

        `"cos"`
            I = cos kξ, the literal NEC shape, and the default. NOT "the
            point-matched contract" any more (momwire#606): the point-matched
            fill takes it above `_WELL_SCALED_KD` and the folded shape below,
            so `cos_shape` is now a per-FILL decision on both families rather
            than a per-family one.
        `"cos-1"`
            I = cos kξ − 1, the shape a coefficient product actually pairs
            with σC once #203's fold is applied. Its field is O((kΔ)²) of the
            const shape's, so taking the difference of the two closed forms in
            float64 leaves ε·‖T_const‖ in it — which is the reciprocity floor
            #205 exists to remove. The branch below gets the same quantity out
            of a spelling in which no term is larger than the answer.

            Both families ask for it, for the same reason at different sizes:
            Galerkin always (its coefficient product is written that way),
            point-matched below the threshold where the literal spelling stops
            having digits (#606).

        `ek` turns on NEC's EXTENDED thin-wire kernel, and its TYPE says which
        of the two kernels' extended paths is meant (momwire#246 §4). The two
        combinations that exist are:

        `(src_a, ind1, ind2)` + `cos_shape="cos"`
            the point-matched contract: NEC's EKSCX closed forms with per-end
            IND gating, served by `_extended_kernel_fields`.
        `_EKPairs(src_a, eligible)` + `cos_shape="cos-1"`
            the Galerkin contract: the #205 folded REDUCED fields, untouched,
            plus `_folded_ek_delta_fields`' quadrature of the smooth delta on
            the pairs `eligible` selects.

        The other two raise. That is the point of dispatching on the payload
        rather than on `ek is not None`: before momwire#246 this method took
        the EKSCX early return whatever `cos_shape` said, so a caller asking
        for the folded shape with `ek=` set was silently handed literal-cos
        EK tables — a wrong answer with no symptom.
        """
        if cos_shape not in ("cos", "cos-1"):
            raise ValueError(f"cos_shape must be 'cos' or 'cos-1', got {cos_shape!r}")
        z_eval, rho_eval, rho_vec, td, rho_proj_factor = self._pair_geometry(
            obs_c, obs_t, a, src_c, src_t
        )

        # Source half-length, materialized at the full broadcast shape so the
        # `H[..., None]` source-quadrature axis below has somewhere to land.
        # Under the outer product that shape is (M, N), M being the observer
        # count: N segment centres (collocation) or N·n_qp Gauss points
        # (Galerkin test integration).
        H = np.broadcast_to(src_hh, z_eval.shape)

        # Extended thin-wire kernel (#233). The geometry above is shared —
        # rho_eval, td and rho_proj_factor are EFLD's, computed before it
        # chooses between EKSC and EKSCX (nec2-1.2.1.2.f:2919-2962), and the
        # ρ̂ projection uses the UNSWAPPED radial distance either way. From
        # here down the two kernels part company, so the extended branch is a
        # clean early return: with `ek=None` (the default, and everything
        # `extended_kernel=False` can reach) not one floating-point operation
        # below changes, which is what makes the option a true no-op when off.
        #
        # The Galerkin payload (momwire#246) does NOT early-return: its EK
        # field is the reduced field below plus a quadrature correction, so it
        # falls through and the delta is added at the return statement. Every
        # reduced closed form it rides on is the one #205 shipped, unchanged
        # and unrounded — the correction is a separate sum at its own size.
        ek_pairs = None
        if ek is not None:
            if isinstance(ek, _EKPairs):
                if cos_shape != "cos-1":
                    raise NotImplementedError(
                        "the pair-rule extended kernel serves the folded "
                        "cos_shape='cos-1' shape only; cos_shape="
                        f"{cos_shape!r} with an _EKPairs payload is not "
                        "wired (momwire#246)"
                    )
                ek_pairs = ek
            else:
                # Both shapes since momwire#614: EKSCX's folded tables are the
                # same G-quantities rearranged, so the per-end IND contract
                # serves `cos-1` without the pair-rule payload. (Before #614
                # this raised — right at the time, since the only folded route
                # then known went through `_EKPairs`.)
                src_a, ind1, ind2 = ek
                tables = self._extended_kernel_fields(
                    k, H, z_eval, rho_eval, src_a, ind1, ind2, cos_shape=cos_shape
                )
                tables["td"] = td
                tables["rho_proj_factor"] = rho_proj_factor
                tables["rho_vec"] = rho_vec
                tables["rho_eval"] = rho_eval
                return tables

        # z values at source ends: z' = +H (z2) and z' = -H (z1).
        # Δz = z_eval - z' at the two endpoints.
        dz2 = z_eval - H  # at z' = +H
        dz1 = z_eval + H  # at z' = -H
        r0_2 = np.sqrt(rho_eval * rho_eval + dz2 * dz2)
        r0_1 = np.sqrt(rho_eval * rho_eval + dz1 * dz1)
        G0_2 = np.exp(-1j * k * r0_2) / r0_2
        G0_1 = np.exp(-1j * k * r0_1) / r0_1

        # Evaluation-order discipline for every complex product below
        # (momwire#392). numpy ELIDES a temporary that is the RIGHT operand of
        # an array product into an in-place multiply once it passes 256 KB
        # (`NPY_MIN_ELIDE_BYTES`), and for complex128 the in-place loop is not
        # the one that runs out of place — the two round differently in the
        # last bit. Measured: `named * (temp)` moves at 625 KB and does not at
        # 16 KB, while `(temp) * named` and a complex-by-real divide move at
        # neither. `G0_e * (bracket)` and `pref_rho * (difference)` are exactly
        # the shape that moves, so each such right-hand operand is BOUND TO A
        # NAME first — a named array carries a second reference and cannot be
        # elided. Nothing is reassociated: same expression, same order, and
        # the value is the one every batch shape UNDER the boundary already
        # produced. Without it these tables are a function of the caller's
        # block size (`_near_block`, `_fill_block`), which is a scheduling
        # decision and not arithmetic. `_folded_cos_fields` and
        # `_folded_ek_delta_fields` name their steps for the same reason.
        #
        # `pref_z` and `pref_rho_const` are python scalars, not arrays, so
        # `pref_z * (…)` is a scalar-array product and takes the same loop
        # either way — measured not to move at any shape, which is why those
        # brackets are left as one expression. `pref_rho` IS an array (it
        # carries 1/ρ_eval), and it is the one prefactor that had to be split.

        # Common scalar prefactors. λ = 2π/k → k²λ = 2πk → jη/(2k²λ) = jη/(4πk).
        # Eqs 76-79 carry a "-I_0/λ · jη/(2k²ρ)" (E_ρ) or "I_0/λ · jη/(2k²)"
        # (E_z) prefactor. For unit I_0 = 1, factor pulled out:
        #   E_ρ prefactor = -jη/(4πk·ρ_eval)
        #   E_z prefactor = +jη/(4πk)
        pref_rho = -1j * self.eta / (4.0 * np.pi * k * rho_eval)
        pref_z = 1j * self.eta / (4.0 * np.pi * k)

        # ---- Constant source (I = 1): Eqs 78, 79 ----
        # E_ρ^f = -I/λ · jη/(2k²) · [(1+jkr_0) ρ G_0 / r_0²]_{z1}^{z2}
        #       = pref_rho · ρ_eval · [(1+jkr_0) ρ_eval G_0 / r_0²]_{z1}^{z2}
        #         (pref_rho already has 1/ρ_eval; multiply back by ρ_eval to
        #          recover the form -jη·ρ_eval/(4πk·ρ_eval)= -jη/(4πk))
        # Reorganize for clarity:
        pref_rho_const = -1j * self.eta / (4.0 * np.pi * k)
        Erho_const = pref_rho_const * (
            (1.0 + 1j * k * r0_2) * rho_eval * G0_2 / (r0_2 * r0_2)
            - (1.0 + 1j * k * r0_1) * rho_eval * G0_1 / (r0_1 * r0_1)
        )
        # E_z^f = -I/λ · jη/(2k²) · { [(1+jkr_0)(z-z') G_0 / r_0²]_{z1}^{z2}
        #                              + k² ∫_{z1}^{z2} G_0 dz' }
        # Note sign: -I/λ · jη/(2k²) = -jη/(4πk) for our prefactor convention.
        # We have pref_z = +jη/(4πk); so we use -pref_z.
        # ∫G_0 dz' via singularity extraction. The plain integrand has a
        # 1/r_0 spike at z' = z_eval when ρ_eval is small (self / near-self
        # pairs in thin wires), and Gauss-Legendre with a small node count
        # under-resolves it. Split into closed-form singular + smooth regular:
        #   ∫G_0 dz' = ∫ 1/r_0 dz'              (closed form, arcsinh)
        #            + ∫ (G_0 - 1/r_0) dz'      (regular: tends to -jk as r_0→0)
        # 1/r_0 closed form: ∫_{-H}^{+H} 1/√(ρ²+(z-z')²) dz' = arcsinh((H-z)/ρ) - arcsinh((-H-z)/ρ).
        # Through `asinh_diff` (momwire#799): on a collinear deck ρ is the wire
        # radius and every non-neighbour pair has (H-z) and (-H-z) of the same
        # sign and agreeing to 2H/z, so the literal difference of two O(log)
        # asinh values costs ε·|asinh|·z/(2H) — 1.8e-13 at the far end of a
        # 201-segment dipole, 8.4e-13 at 801. Same object, same helper, as
        # razor's `_static_axis_moments` m0.
        int_inv_r0 = asinh_diff(-dz1, -dz2, rho_eval * rho_eval, r0_1, r0_2)
        gx, gw = self._leggauss_cached(self.n_qp_const)
        z_qp = H[..., None] * gx  # S + (n_qp,)
        dz_qp = z_eval[..., None] - z_qp
        r0_qp = np.sqrt(rho_eval[..., None] ** 2 + dz_qp**2)
        # (e^{-jkr} - 1)/r, NOT G0 - 1/r: the literal difference of two
        # 1/r-sized terms returns its real part to an absolute ε, which is
        # 1.1e-12 relative at kr = 8e-3 and grows as 1/(kr)² (momwire#799).
        # Through the HALF phase, which is the form the C++ twin's phase table
        # already holds — same spelling in both lanes, not merely the same
        # value.
        reg_qp = _expm1_neg_j_from_half(-0.5 * k * r0_qp) / r0_qp
        int_reg = np.einsum("...q,q->...", reg_qp, gw) * H
        int_G0 = int_inv_r0 + int_reg  # (M, N)
        Ez_const_boundary = (1.0 + 1j * k * r0_2) * dz2 * G0_2 / (r0_2 * r0_2) - (
            1.0 + 1j * k * r0_1
        ) * dz1 * G0_1 / (r0_1 * r0_1)
        Ez_const = -pref_z * (Ez_const_boundary + k * k * int_G0)

        # ---- Sine source (I = sin(k·z'_local)): Eq 76, Eq 77 ----
        # Eq 71 (sinusoidal general) factored: E_ρ^f = pref_rho · G_0 ·
        # {k(z-z') cos(kz') + [1 - (z-z')²(1+jkr_0)/r_0²] sin(kz')} |_{z1}^{z2}
        # — note G_0 is an overall factor on the WHOLE bracket, evaluated
        # at each endpoint with its own r_0.
        sin2 = np.sin(k * H)  # sin(k · z'_2) where z'_2 = +H, so sin(kH)
        cos2 = np.cos(k * H)
        sin1 = np.sin(-k * H)  # at z'_1 = -H
        cos1 = np.cos(-k * H)
        b_sin_2 = (
            k * dz2 * cos2
            + (1.0 - dz2 * dz2 * (1.0 + 1j * k * r0_2) / (r0_2 * r0_2)) * sin2
        )
        bracket_sin_2 = G0_2 * b_sin_2
        b_sin_1 = (
            k * dz1 * cos1
            + (1.0 - dz1 * dz1 * (1.0 + 1j * k * r0_1) / (r0_1 * r0_1)) * sin1
        )
        bracket_sin_1 = G0_1 * b_sin_1
        d_sin = bracket_sin_2 - bracket_sin_1
        Erho_sin = pref_rho * d_sin
        # E_z^f = pref_z · G_0 · {k cos(kz') - (1+jkr_0)(z-z')/r_0² sin(kz')}_{z1}^{z2}
        b_sin_z_2 = k * cos2 - (1.0 + 1j * k * r0_2) * dz2 / (r0_2 * r0_2) * sin2
        bracket_sin_z_2 = G0_2 * b_sin_z_2
        b_sin_z_1 = k * cos1 - (1.0 + 1j * k * r0_1) * dz1 / (r0_1 * r0_1) * sin1
        bracket_sin_z_1 = G0_1 * b_sin_z_1
        Ez_sin = pref_z * (bracket_sin_z_2 - bracket_sin_z_1)

        if cos_shape == "cos":
            # ---- Cosine source (I = cos(k·z'_local)): same as Eqs 76, 77 with
            #      the "(cos kz'/-sin kz')" toggle picking the lower row, i.e.
            #      swap sin↔cos and negate the (sin-row → -sin) term.
            b_cos_2 = (
                -k * dz2 * sin2
                + (1.0 - dz2 * dz2 * (1.0 + 1j * k * r0_2) / (r0_2 * r0_2)) * cos2
            )
            bracket_cos_2 = G0_2 * b_cos_2
            b_cos_1 = (
                -k * dz1 * sin1
                + (1.0 - dz1 * dz1 * (1.0 + 1j * k * r0_1) / (r0_1 * r0_1)) * cos1
            )
            bracket_cos_1 = G0_1 * b_cos_1
            d_cos = bracket_cos_2 - bracket_cos_1
            Erho_cos = pref_rho * d_cos
            b_cos_z_2 = -k * sin2 - (1.0 + 1j * k * r0_2) * dz2 / (r0_2 * r0_2) * cos2
            bracket_cos_z_2 = G0_2 * b_cos_z_2
            b_cos_z_1 = -k * sin1 - (1.0 + 1j * k * r0_1) * dz1 / (r0_1 * r0_1) * cos1
            bracket_cos_z_1 = G0_1 * b_cos_z_1
            Ez_cos = pref_z * (bracket_cos_z_2 - bracket_cos_z_1)
        else:
            Erho_cos, Ez_cos = self._folded_cos_fields(
                k,
                H,
                z_eval,
                rho_eval,
                dz1,
                dz2,
                r0_1,
                r0_2,
                G0_1,
                G0_2,
                sin2,
                dz_qp,
                r0_qp,
                gx,
                gw,
                pref_z,
                pref_rho,
            )

        tables = {
            "Erho_const": Erho_const,
            "Ez_const": Ez_const,
            "Erho_sin": Erho_sin,
            "Ez_sin": Ez_sin,
            "Erho_cos": Erho_cos,
            "Ez_cos": Ez_cos,
        }
        if ek_pairs is not None:
            # Extended kernel, Galerkin path (momwire#246): reduced + delta.
            # The delta is computed for the whole block and selected off the
            # eligible set rather than gathered — the block is already
            # materialized, and a mask keeps the arithmetic on the pairs that
            # DO get it identical however the caller slices them.
            delta = self._folded_ek_delta_fields(
                k,
                H,
                z_eval,
                rho_eval,
                ek_pairs.src_a,
                cos_shape="cos-1",
                n_panels=ek_pairs.n_panels,
            )
            for key, base in tables.items():
                summed = base + delta[key]
                if ek_pairs.eligible is not None:
                    # Selecting the SUM rather than zeroing the delta: adding
                    # a zero is not the identity on a signed zero, and this
                    # branch has to give the reduced table back to the bit.
                    summed = np.where(ek_pairs.eligible, summed, base)
                tables[key] = summed
        tables["td"] = td
        tables["rho_proj_factor"] = rho_proj_factor
        tables["rho_vec"] = rho_vec
        tables["rho_eval"] = rho_eval
        return tables

    @staticmethod
    def _folded_cos_fields(
        k,
        H,
        z,
        rho,
        dz1,
        dz2,
        r1,
        r2,
        G1,
        G2,
        sH,
        dz_qp,
        r_qp,
        gx,
        gw,
        pref_z,
        pref_rho,
    ):
        """(E_ρ, E_z) of the source shape I = cos kξ − 1 (momwire#205).

        Same quantity as `Ez_cos − Ez_const` on the same source quadrature —
        exactly, not approximately: nothing here changes the rule, only which
        differences are taken before which sums. What changes is the error.
        The literal subtraction leaves ε·|const-shape field| in an answer that
        is O((kH)²) of it, so it is ε·6/(kH)² relative and it lands on G undiluted
        — 3.9e-12 of the const scale at N=2401 on the half-wave dipole, which
        is exactly the reciprocity floor #203 measured and could not reach.

        Taking the two closed forms apart term by term (the endpoint pairs
        cancel against each other, and the const shape's ∫G₀ against the cos
        shape's endpoint terms) leaves three quantities, each ALREADY the size
        of the answer:

            E_z:  pref_z·[ k²·(∫G₀ − (sin kH/k)(G₁+G₂)) − (cos kH − 1)(F₂−F₁) ]
            E_ρ:  pref_ρ·[ −k·W + ρ²(cos kH − 1)(T₂−T₁) ]

        with F_e = Δz_e·T_e, T_e = (1+jkr_e)G_e/r_e², and W the closed form of
        kρ²∫sin(kξ)T(ξ)dξ. The three cancellations that spelling still hides
        are removed one at a time:

        * **∫G₀ against its endpoint pair.** Split at the singularity
          extraction the const shape already does. The 1/r half is the exact
          trapezoid remainder ∫(1/r) − H(1/r₁+1/r₂), which rationalizes to
          `asinh(X) − X` plus a manifestly positive term (`X` is the argument
          of the arcsinh difference, itself rationalized so its numerator does
          not cancel). The regular half is a quadrature sum whose nodes are
          taken relative to the NEAREST endpoint, through r-differences that
          are exact by construction (Δz_q − Δz_e = ±H(1 ∓ t_q): the observer
          drops out) and an `expm1` of the resulting phase.
        * **H − sin(kH)/k**, from `_sin_minus_arg`.
        * **W**, whose two endpoint groups cancel to O((kH)³). Referring both
          endpoints to the pair's MEAN radius turns them into one even and one
          odd combination, and the even one telescopes to
          A·(sin B − B) − B·(sin A − A) with A, B = kH ± k(r₁−r₂)/2 — two more
          `_sin_minus_arg`s — plus a term in H(c₁+c₂) − (r₁−r₂), which has its
          own cancellation-free closed form (`d_lin` below).

        What is left is the regular quadrature sum, where the node terms are
        O(kH) and their sum is O((kH)²): one power of kH, not two, so the
        residual is ε·(kH)·|const| rather than ε·|const| — measured 1.6e-15 of
        the const scale at N=2401 against a 50-digit reference, a factor 2400.
        Removing the last power needs the SECOND difference of the regular
        integrand (node pair against endpoint pair) in closed form; at the
        meshes this solver runs, that residual is already below the fill's
        structural asymmetry, so it is left to #205's follow-up.
        """
        kH = k * H
        cm1 = -2.0 * np.sin(0.5 * kH) ** 2  # cos kH − 1, without the subtraction
        T2 = (1.0 + 1j * k * r2) * G2 / (r2 * r2)
        T1 = (1.0 + 1j * k * r1) * G1 / (r1 * r1)

        # X = sinh of the arcsinh difference ∫(1/r) dξ, i.e. asinh(X) = that
        # integral. Its numerator Δz₁r₂ − Δz₂r₁ cancels when the observer is
        # off the segment's span and Δz₁r₂ + Δz₂r₁ cancels when it is inside,
        # so each branch takes the sum that does not: rationalizing
        # (Δz₁r₂)² − (Δz₂r₁)² = ρ²(Δz₁² − Δz₂²) swaps one for the other.
        outside = dz1 * dz2 >= 0.0
        s_den = np.where(outside, dz1 * r2 + dz2 * r1, 1.0)
        X = np.where(
            outside,
            2.0 * H * (dz1 + dz2) / s_den,
            (dz1 * r2 - dz2 * r1) / (rho * rho),
        )
        # ∫(1/r)dξ − H(1/r₁ + 1/r₂). The rationalized second term is
        # X − H(r₁+r₂)/(r₁r₂) = Hρ²X²/((r₁+r₂)r₁r₂) — all positive, no
        # cancellation — leaving asinh(X) − X for `_asinh_minus_arg`. Above
        # |X| = 1 the pair no longer cancels at all (this is the self and
        # near-neighbour band, where the answer IS the arcsinh), and the two
        # rationalized terms would instead cancel against each other, so the
        # literal difference is the accurate spelling there.
        t_sing = np.where(
            np.abs(X) < 1.0,
            _asinh_minus_arg(X) + H * rho * rho * X * X / ((r1 + r2) * r1 * r2),
            np.arcsinh(X) - H * (1.0 / r1 + 1.0 / r2),
        )

        # Σ_q gw_q·g(ξ_q) − (g₁ + g₂) for the regular part g = G₀ − 1/r, with
        # every node referred to the endpoint on its own side of the segment.
        g1 = _expm1_neg_j(k * r1) / r1
        g2 = _expm1_neg_j(k * r2) / r2
        near2 = gx >= 0.0
        r_ref = np.where(near2, r2[..., None], r1[..., None])
        dz_ref = np.where(near2, dz2[..., None], dz1[..., None])
        g_ref = np.where(near2, g2[..., None], g1[..., None])
        # e^{-jkr} at the reference endpoint, from the Green's value already in
        # hand rather than from a second exp.
        e_ref = np.where(near2, (G2 * r2)[..., None], (G1 * r1)[..., None])
        # Δz_q − Δz_ref is ±H(1 ∓ t_q) exactly: z_eval cancels, so the node
        # radius difference below is exact however far away the observer is.
        d_step = np.where(near2, H[..., None] * (1.0 - gx), -H[..., None] * (1.0 + gx))
        delta = d_step * (dz_qp + dz_ref) / (r_qp + r_ref)
        # g(r_ref + δ) − g(r_ref), rearranged so both terms are O(kδ). Spelled
        # in named steps, not as one expression: above numpy's 256 KB temporary
        # threshold a complex product with a dead operand is evaluated in place
        # by a different loop, which would make this (…, n_qp)-sized array's
        # rounding depend on the fill's BLOCK size — the one thing
        # `test_far_fill_blocking_is_ulp_exact` bounds (a few ULPs, #236;
        # the named-steps discipline is what keeps it that tight).
        dg_phase = _expm1_neg_j(k * delta)
        dg_a = e_ref * dg_phase
        dg_b = g_ref * delta
        dg_num = dg_a - dg_b
        dg = dg_num / r_qp
        # The rule's weights sum to 2 and its two halves to 1 each only to
        # within ε, and those ε's are part of the value the const shape
        # already carries, so they are kept rather than idealized away.
        m_reg = (
            np.einsum("...q,q->...", dg, gw)
            + (gw[near2].sum() - 1.0) * g2
            + (gw[~near2].sum() - 1.0) * g1
        )

        d_int = t_sing + H * m_reg - (_sin_minus_arg(kH) / k) * (G1 + G2)
        Ez = pref_z * (k * k * d_int - cm1 * (dz2 * T2 - dz1 * T1))

        # E_ρ: W = kρ²∫sin(kξ)·(1+jkr)G₀/r² dξ in closed form, with both
        # endpoint terms referred to the pair's mean radius (r₁+r₂)/2 — which
        # costs no new exponential, since e^{-jk(r₁+r₂)/2} = e^{-jkr₂}·e^{-jkφ}
        # and φ = k(r₁−r₂)/2 is needed anyway.
        dr = 4.0 * H * z / (r1 + r2)  # r₁ − r₂, exactly
        phi = 0.5 * k * dr
        A = kH + phi
        B = kH - phi
        # H(c₁+c₂) − (r₁−r₂), with c_e = Δz_e/r_e: the same rationalization
        # twice over, ending on
        # ρ² + Δz₁Δz₂ − r₁r₂ = −4ρ²H²/(ρ² + Δz₁Δz₂ + r₁r₂).
        d_lin = (
            -8.0
            * H**3
            * z
            * rho
            * rho
            / ((rho * rho + dz1 * dz2 + r1 * r2) * (r1 + r2) * r1 * r2)
        )
        cph = np.cos(phi)
        sph = np.sin(phi)
        w_even = (A * _sin_minus_arg(B) - B * _sin_minus_arg(A)) / kH + (
            d_lin / H
        ) * sH * cph
        # c₂ − c₁ = −ρ²X/(r₁r₂), from the same rationalized numerator as X.
        w_odd = 1j * sH * (-(rho * rho * X) / (r1 * r2)) * sph
        W = (G2 * r2) * (cph - 1j * sph) * (w_even + w_odd)
        # `pref_rho` is an ARRAY (it carries 1/ρ), so the bracket is named
        # before the product rather than left as a dead temporary on the
        # right of it — the one place in this routine that had been left as
        # one expression, and the shape `_field_components_bcast` fixed in
        # momwire#392. Inert on every table measured there; named so it
        # cannot become the exception the rule is written against.
        e_rho_brk = -k * W + rho * rho * cm1 * (T2 - T1)
        Erho = pref_rho * e_rho_brk
        return Erho, Ez

    def _ek_delta_rule(self, n_gl, n_panels):
        """Composite Gauss-Legendre rule on [−1, 1]: `n_panels` equal panels of
        `n_gl` nodes each, Σw = 2. Cached on the solver's own node cache.

        The delta quadrature runs in the sinh-mapped variable, where the
        integrand is analytic with its nearest pole a fixed distance π/2 off
        the real axis. Convergence is therefore set by the mapped interval's
        half width per panel, not by ρ/H — so one rule serves every wire
        thickness, and refining means adding panels rather than nodes.
        """
        gx, gw = self._leggauss_cached(n_gl)
        if n_panels == 1:
            return gx, gw
        edges = np.linspace(-1.0, 1.0, n_panels + 1)
        mid = 0.5 * (edges[:-1] + edges[1:])
        half = 0.5 * (edges[1:] - edges[:-1])
        return (
            (mid[:, None] + half[:, None] * gx[None, :]).ravel(),
            (half[:, None] * gw[None, :]).ravel(),
        )

    def _folded_ek_delta_fields(
        self, k, H, z, rho, src_a, cos_shape="cos-1", n_gl=None, n_panels=1
    ):
        """The EK-minus-reduced field correction, by direct quadrature
        (momwire#246).

        Returns the same six per-shape tables `_field_components_bcast`
        returns — `Erho_const`, `Ez_const`, `Erho_sin`, `Ez_sin`,
        `Erho_cos`, `Ez_cos` — but holding only the DIFFERENCE between the
        extended and reduced kernels' fields, so the caller adds them to the
        reduced tables it already has. Every argument broadcasts: `H` (source
        HALF length), `z` (observer's axial coordinate in the source frame),
        `rho` (the a-regularized radial distance `rho_eval`) and `src_a` (the
        source segments' radius) share the field tables' shape S, and every
        returned table has shape S.

        The delta kernel
        ----------------
        Write the reduced kernel as a function of u = R² = ρ² + ζ²,

            g(u) = e^{−jkR}/R,   g⁽ⁿ⁾(u) = (−½)ⁿ·e^{−jkR}·Aₙ(jkR)/R^{2n+1}

        with Aₙ the reverse Bessel polynomials (A₁ = 1+x, A₂ = 3+3x+x²,
        A₃ = 15+15x+6x²+x³, A₄ = 105+105x+45x²+10x³+x⁴). The extended kernel
        is the source tube's circumferential average, R(φ)² = u + a² −
        2aρ·cos φ; averaging the Taylor expansion in (R² − u) — ⟨R²−u⟩ = a²,
        ⟨(R²−u)²⟩ = a⁴ + 2a²ρ² — gives the delta kernel this integrates:

            W(ρ, ζ) = a²·g′(u) + a²ρ²·g″(u)                              (*)

        At ρ = a — which is what eligibility MEANS, coaxial and equal radius,
        so the observer sits on its own wire's surface on the source axis —
        (*) is NEC Eq 89's factor minus one times the reduced kernel, term for
        term: a²g′ = −(a²/2R²)·C1·G and a⁴g″ = (a⁴/4R⁴)·C2·G, i.e. exactly
        `_bspline_kernels._ek_factor`'s `(fac − 1)·G_red` (momwire#249 §1.2).
        Keeping the ρ² of (*) rather than substituting a² for it changes
        NOTHING on the pairs this serves — but it is the difference between a
        right and a wrong E_ρ, because E_ρ differentiates the kernel in ρ and
        Eq 89's factor form, being a function of R and a alone, has no honest
        ρ-derivative to give. (Measured: substituting a² first lands E_ρ at
        HALF the value of the exact circumferential average and of NEC's own
        EKSCX; (*) reproduces both. E_z is untouched by the choice, since
        ∂/∂z reaches u only through ζ.)

        Why a quadrature and not a closed form
        --------------------------------------
        `W` is bounded — no singularity survives at ζ = 0, where it tends to a
        finite multiple of G_red(a) — and analytic along the whole source
        segment, its nearest pole (the reduced kernel's own ζ = ±jρ) a full
        wire radius off the real axis. That is what Gauss-Legendre is for. And
        — the point — the folded source shape can be evaluated POINTWISE as
        −2·sin²(kξ/2). There is no folded-versus-literal spelling question to
        answer, because there is no subtraction: the cancellation discipline
        `_folded_cos_fields` exists to enforce never arises here. Differencing
        the EKSCX closed forms instead would inherit their internal
        ~1e-13·‖T_const‖ noise, the very floor #205 removed.

        The variable, and why it is not ξ
        ---------------------------------
        "A pole a wire radius off the axis" is a statement about convergence
        only once the path is measured in radii. Integrated in ξ, the delta's
        feature is a spike of width ρ inside a segment of half-length H, so a
        fixed rule's accuracy is governed by ρ/H — fine for a fat wire, and
        useless exactly where a fill spends its most important pairs: the self
        pair and its neighbours, where the observer sits ON the source segment
        (|z| ≤ H) and ρ/H is the reciprocal of Δ/a. Measured, a plain 16-node
        ξ rule at the self pair is wrong by 5× the answer at Δ/a = 6 and by
        1e6× at Δ/a = 122.

        So the integration variable is the sinh-mapped one:

            ζ = ρ·sinh t,  R = ρ·cosh t,  dζ = ρ·cosh t dt,
            t ∈ [asinh((z−H)/ρ), asinh((z+H)/ρ)]

        — the classical near-singular substitution, and here an unusually
        clean one because R comes out in closed form rather than as a square
        root. In t the spike is O(1) wide however thin the wire is, the
        kernel's poles sit at t = ±jπ/2 independent of ρ, and the interval's
        half width grows only like ln(2H/ρ). `n_panels` splits that interval
        into equal panels, which is what makes one rule serve every thickness.

        There IS a floor underneath all of this, and it is not the rule's: the
        delta's whole-line integral very nearly vanishes (extended and reduced
        kernels agree away from the wire), so the sum carries ~(H/ρ)² of
        cancellation and float64 leaves ~ε·(H/ρ)² of the delta's own peak in
        it. In the units that reach G that is ~ε·const, because the delta is
        itself O((ρ/H)²) of the field — measured ≤2e-10 of the reduced const
        field out to Δ/a = 500 on the near tier, and falling to 1e-13 for fat
        wire. It does bound how thin a wire this decomposition can resolve an
        EK correction for, which is a property of reduced-plus-delta and not
        of the quadrature.

        The field operators are the reduced path's own, re-read off its closed
        forms and proved against them in `scripts/derive_galerkin_ek_delta.py`
        §1:

            E_z[s] = −pref_z·∫ s(ξ)·(k² + ∂²/∂z²) W dξ
            E_ρ[s] = −pref_z·∫ s(ξ)·(∂²/∂ρ∂z)   W dξ,  pref_z = jη/(4πk)

        (`pref_rho = −pref_z/ρ` is a factoring of the reduced brackets, not a
        second normalization.) Applied to (*) with ∂u/∂z = 2ζ, ∂u/∂ρ = 2ρ they
        collapse onto the four derivatives above — §2 of the same script:

            (k² + ∂²_z)W = a²·[ k²(g′ + ρ²g″) + 2(g″ + ρ²g‴)
                                + 4ζ²(g‴ + ρ²g⁗) ]
            ∂²_{ρz} W    = a²·ρζ·[ 8g‴ + 4ρ²g⁗ ]

        Eligibility is NOT decided here — this computes the delta for whatever
        pairs it is handed, and `_field_components_bcast` (with
        `_EKPairs.eligible`) or a caller that pre-selects a sub-block decides
        which pairs get it.

        `cos_shape` picks the third source shape exactly as
        `_field_components_bcast` does: `"cos-1"` (the default here) is the
        Galerkin fill's folded shape −2·sin²(kξ/2); `"cos"` is the literal NEC
        shape, which exists so the delta can be measured against the
        point-matched `_extended_kernel_fields` on the same three shapes.
        """
        if cos_shape not in ("cos", "cos-1"):
            raise ValueError(f"cos_shape must be 'cos' or 'cos-1', got {cos_shape!r}")
        gx, gw = self._ek_delta_rule(_N_QP_EK_DELTA if n_gl is None else n_gl, n_panels)

        # Source quadrature over ξ ∈ [−H, H], reparametrized by ζ = z − ξ =
        # ρ·sinh t. `rho` is the shared regularized distance the reduced path
        # is already using at this pair — taking it from there rather than
        # rebuilding it from `src_a` is what makes reduced + delta the extended
        # kernel's field and not something near it. On an eligible pair the two
        # agree anyway (ρ IS the radius).
        hh = np.asarray(H)[..., None]
        zz = np.asarray(z)[..., None]
        rr = np.asarray(rho)[..., None]
        t_lo = np.arcsinh((zz - hh) / rr)
        t_hi = np.arcsinh((zz + hh) / rr)
        t_mid = 0.5 * (t_hi + t_lo)
        t_half = 0.5 * (t_hi - t_lo)
        t = t_mid + t_half * gx
        cosh_t = np.cosh(t)
        # R = ρ·cosh t exactly, so the near-singular denominator never goes
        # through a difference of squares. ξ = z − ζ recovers the source arc
        # the shape functions are evaluated on.
        R = rr * cosh_t
        zeta = rr * np.sinh(t)
        xi = zz - zeta
        w = (t_half * gw) * (rr * cosh_t)  # dζ = ρ·cosh t dt
        r2 = R * R

        # g′ … g⁗ of the reduced kernel with respect to u = R², through the
        # reverse Bessel polynomials. Named steps throughout, per the rule at
        # `_folded_cos_fields`: above numpy's 256 KB temporary threshold a
        # one-expression complex product with a dead operand is evaluated by a
        # different loop, which would make these (…, n_gl)-sized arrays'
        # rounding depend on the fill's BLOCK size.
        x = 1j * (k * R)
        x2 = x * x
        x3 = x2 * x
        x4 = x2 * x2
        a1 = 1.0 + x
        a2 = 3.0 + 3.0 * x + x2
        a3 = 15.0 + 15.0 * x + 6.0 * x2 + x3
        a4 = 105.0 + 105.0 * x + 45.0 * x2 + 10.0 * x3 + x4
        inv2 = 1.0 / r2
        phase = np.exp(-1j * (k * R))
        base = phase / R
        g1 = base * a1
        g1 = g1 * inv2
        g1 = -0.5 * g1
        g2 = base * a2
        g2 = g2 * (inv2 * inv2)
        g2 = 0.25 * g2
        g3 = base * a3
        g3 = g3 * (inv2 * inv2 * inv2)
        g3 = -0.125 * g3
        g4 = base * a4
        g4 = g4 * (inv2 * inv2 * inv2 * inv2)
        g4 = 0.0625 * g4

        # The two operators of (*), with the a² held back to the end.
        rho2 = rr * rr
        t_k = (k * k) * (g1 + rho2 * g2)
        t_c = 2.0 * (g2 + rho2 * g3)
        t_z = (4.0 * zeta * zeta) * (g3 + rho2 * g4)
        l_z = t_k + t_c
        l_z = l_z + t_z
        l_r = 8.0 * g3 + 4.0 * rho2 * g4
        l_r = l_r * rr
        l_r = l_r * zeta

        # −pref_z, times the a² that makes the whole correction exactly 0.0 at
        # a = 0 in IEEE and not merely in the limit: (*) is linear in a², so
        # one multiplication at the end carries the collapse. `src_a` is the
        # SOURCE segment's radius, the `a` of NEC Eq 89 (and of
        # `_extended_kernel_fields`' BX).
        pref = -1j * self.eta / (4.0 * np.pi * k)
        gain = pref * (src_a * src_a)

        kxi = k * xi
        if cos_shape == "cos":
            s_cos = np.cos(kxi)
        else:
            # The folded shape POINTWISE — never cos − 1 by subtraction.
            s_cos = -2.0 * np.sin(0.5 * kxi) ** 2
        shapes = {"const": None, "sin": np.sin(kxi), "cos": s_cos}
        out = {}
        for name, s in shapes.items():
            sw = w if s is None else s * w
            out[f"Ez_{name}"] = gain * np.einsum("...q,...q->...", sw, l_z)
            out[f"Erho_{name}"] = gain * np.einsum("...q,...q->...", sw, l_r)
        return out

    def _ek_end_bracket_fields(self, k, H, z, rho, src_a, sign, cos_shape="cos-1"):
        """One END BRACKET of `_folded_ek_delta_fields`' operator, in closed
        form — the boundary term of its integration by parts in ξ
        (momwire#299).

        Same six per-shape tables, same broadcasting contract and same
        arguments as `_folded_ek_delta_fields`, except for `sign`: +1 selects
        the source segment's ξ = +H end, −1 its ξ = −H end, and the returned
        tables carry that end's share of the bracket ALONE (no quadrature, no
        smooth remainder). Adding both ends' tables to the smooth integrals
        below reproduces the full delta to 3e-13 relative on every table.

        The decomposition
        -----------------
        ζ = z − ξ, so ∂/∂ξ = −∂/∂ζ = −∂/∂z on W, and integrating the delta's
        two field operators by parts once in ξ gives

            E_z[s] = −pref_z ( [ s·∂_ξW − s′·W ]_{−H}^{+H}
                               + ∫ (k²s + s″) W dξ )
            E_ρ[s] = −pref_z ( −[ s·∂_ρW ]_{−H}^{+H} + ∫ s′·∂_ρW dξ )

        with W = a²(g′ + ρ²g″) the delta kernel (*) of
        `_folded_ek_delta_fields` and

            ∂_ξW = −2ζ·a²(g″ + ρ²g‴),    ∂_ρW = 2ρ·a²(2g″ + ρ²g‴).

        Why the bracket is worth naming
        -------------------------------
        It is the whole divergence. At the observer radii a matrix fill uses
        (ρ = a on an eligible pair) the s·∂_ξW cap integrates over an adjacent
        test segment to a_src²/ρ³ = **O(1/a)**, while W itself integrates to
        O(1) — measured 1.19e4 against 23.9 at a = 0.002, and 99.8 % of the
        cap comes from within 10 a of the segment end. Two such caps meeting
        at an interior node cancel identically (the source current and its
        derivative are continuous there and the two ends carry opposite
        `sign`), which is why a fill that extends BOTH sides of every node is
        healthy and one that extends only one side is not — momwire#299's
        defect, and `SinusoidalGalerkinSolver._ek_bracket_correction`'s reason
        for calling this.

        `cos_shape` picks the third source shape exactly as
        `_folded_ek_delta_fields` does; the shape values are evaluated AT the
        end, with the folded one spelled pointwise as −2·sin²(kξ/2).
        """
        if cos_shape not in ("cos", "cos-1"):
            raise ValueError(f"cos_shape must be 'cos' or 'cos-1', got {cos_shape!r}")
        xi_end = sign * np.asarray(H, dtype=float)
        zeta = np.asarray(z) - xi_end
        rr = np.asarray(rho)
        R = np.sqrt(rr * rr + zeta * zeta)

        # g′ … g‴ of the reduced kernel in u = R², named step by step exactly
        # as in `_folded_ek_delta_fields` (same reverse Bessel polynomials,
        # same association) so the two spellings of the same kernel cannot
        # drift apart.
        x = 1j * (k * R)
        x2 = x * x
        x3 = x2 * x
        a1 = 1.0 + x
        a2 = 3.0 + 3.0 * x + x2
        a3 = 15.0 + 15.0 * x + 6.0 * x2 + x3
        inv2 = 1.0 / (R * R)
        base = np.exp(-1j * (k * R)) / R
        g1 = -0.5 * (base * a1) * inv2
        g2 = 0.25 * (base * a2) * (inv2 * inv2)
        g3 = -0.125 * (base * a3) * (inv2 * inv2 * inv2)

        rho2 = rr * rr
        w_val = g1 + rho2 * g2  # W / a²
        dw_dxi = (-2.0 * zeta) * (g2 + rho2 * g3)  # ∂_ξW / a²
        dw_drho = (2.0 * rr) * (2.0 * g2 + rho2 * g3)  # ∂_ρW / a²

        # −pref_z·a², the same one multiplication that carries
        # `_folded_ek_delta_fields`' exact collapse at a = 0.
        pref = -1j * self.eta / (4.0 * np.pi * k)
        gain = pref * (src_a * src_a) * sign

        kxi = k * xi_end
        s_sin = np.sin(kxi)
        c_cos = np.cos(kxi)
        if cos_shape == "cos":
            s_cos, sp_cos = c_cos, -k * s_sin
        else:
            # The folded shape POINTWISE — never cos − 1 by subtraction.
            s_cos, sp_cos = -2.0 * np.sin(0.5 * kxi) ** 2, -k * s_sin
        shapes = {
            "const": (np.ones_like(kxi), None),
            "sin": (s_sin, k * c_cos),
            "cos": (s_cos, sp_cos),
        }
        out = {}
        for name, (s, sp) in shapes.items():
            ez = s * dw_dxi if sp is None else s * dw_dxi - sp * w_val
            out[f"Ez_{name}"] = gain * ez
            out[f"Erho_{name}"] = -gain * (s * dw_drho)
        return out

    # ------------------------------------------------------------------
    # Matrix assembly and solve
    # ------------------------------------------------------------------

    def _image_source_centers_tangents(self, geom):
        """Mirror source segments across z = ground_z and flip their tangent
        z-components, mirroring the convention the Galerkin solvers use for the
        PEC image build. Same shape ((N, 3), (N, 3)) as the originals.
        """
        seg_c = geom["seg_centers"]
        seg_t = geom["seg_tangents"]
        src_c_img = _ground_mirror.mirror_positions(seg_c, self.ground_z)
        src_t_img = _ground_mirror.mirror_tangents(seg_t)
        return src_c_img, src_t_img

    def _field_tensor_image(self, geom, k, obs_rows=None, cos_shape="cos"):
        """Field tensor for image sources at PEC ground. The image keeps the
        same per-segment half-length and basis shape; only the source center
        is mirrored and the source tangent z-component is flipped.

        `obs_rows` forwards `_field_tensor`'s observer band: the mirror is a
        SOURCE-side transform, so the band means the same rows here.
        `cos_shape` likewise forwards unchanged — the mirror does not touch
        the source SHAPE, so the image block must be built in whichever shape
        set the direct block used or the two cannot be added (#606).
        """
        src_c_img, src_t_img = self._image_source_centers_tangents(geom)
        return self._field_tensor(
            geom,
            k,
            src_centers=src_c_img,
            src_tangents=src_t_img,
            obs_rows=obs_rows,
            cos_shape=cos_shape,
        )

    def _image_refl_prep(self, geom):
        """The whole-geometry `_field_ground.PairTables` for the `ground_eps`
        weighted image, cached per geometry object (identity check, same
        pattern as `_cached_basis`) so swept callers pay the O(N²) build
        once, not per frequency; ρ_v/ρ_h depend on ε̃(ω) and are NOT cached
        — they come from `PairTables.weights` per ω.

        Delegates: the tables themselves are built by
        `_field_ground.specular_pair_prep`, which is the ONE spelling of
        this geometry in the field trunk since momwire#397 unit 1. What
        this method still owns is the SCHEDULE — the per-geometry cache.

        Point-matched does not call this — `_field_tensor_image_refl` takes
        `_image_refl_band`'s per-band tables instead (#332 unit B). What
        remains here is `SinusoidalGalerkinSolver._refl_projection`, which
        fancy-indexes the whole table with per-quadrature-point (test
        segment, source segment) pairs rather than a contiguous observer
        slice, so it is not a band consumer the same way; its own residency
        is issue #332's separate Galerkin section.
        """
        cached = self._cached_image_refl_prep
        if cached is not None and cached[0] is geom:
            return cached[1]
        # Square case: sources = observers. The specular geometry is taken
        # from the REAL (unmirrored) source centers — `specular_pair_prep`
        # mirrors internally (dz = z_m + z_n − 2·ground_z).
        prep = _field_ground.specular_pair_prep(
            geom["seg_centers"], geom["seg_tangents"], self.ground_z
        )
        self._cached_image_refl_prep = (geom, prep)
        return prep

    def _image_refl_band(self, geom, obs_rows=None):
        """Per-BAND `_field_ground.PairTables` for the `ground_eps` weighted
        image: the same cos θ / p̂ / tangent-projection quintet
        `_image_refl_prep` builds, but shaped (band, N_src) for one observer
        band rather than (N_obs, N_src) for the whole geometry, and never
        cached — the point-matched fill's #332 unit B transplant of #323's
        window-producer trade (bspline's `_image_weight_window_fn` retired
        the same tables the same way).

        Delegates to `_field_ground.specular_pair_prep`, the shared builder
        (momwire#397 unit 1); the band-vs-whole choice this method makes IS
        the schedule, and is all that distinguishes it from
        `_image_refl_prep`.

        The quintet is k-INDEPENDENT, so a sweep rebuilds the same numbers
        at every frequency: the residency #332 retired was, seen from the
        k loop, a cache. It is not coming back and it cannot be replayed
        either — replay would need the k loop innermost, which means every
        k's Z live at once (20 x 16N² = 115 MB at the N = 600, 20-point
        sweep momwire#357 measures) against the 5 x 8N² = 14.4 MB the cache
        cost. So #357 item 2 bought the recompute down instead of buying it
        back: `specular_ray_tables` now tiles its own elementwise build so
        it runs out of cache, and the two tangent projections hold one
        scratch buffer between them rather than four temporaries. Same
        floats, ~0.6x the per-band cost, no residency at all.

        `obs_rows = (i0, i1)` restricts the OBSERVER axis, same contract as
        `_field_tensor`'s; `None` means the whole geometry (one band).
        Sources stay the full, REAL (unmirrored) centers/tangents.
        """
        return _field_ground.specular_pair_prep(
            geom["seg_centers"], geom["seg_tangents"], self.ground_z, obs_rows
        )

    def _field_tensor_image_refl(
        self, geom, k, obs_rows=None, ground=None, cos_shape="cos"
    ):
        """Fresnel-weighted image field tensor for the `ground_eps` finite
        ground (NEC IPERF=0 reflection-coefficient approximation).

        Applies NEC's field dyad D = ρ_v·(I − p̂p̂) − ρ_h·p̂p̂ to the IMAGE
        source's field vector at each observer before the tangential
        projection, with p̂ the horizontal unit normal to the plane of
        incidence of the specular ray (image midpoint → observer midpoint)
        and ρ_v/ρ_h the Fresnel coefficients at that ray's incidence angle
        — per-pair constants, NEC's approximation. The algebra is
        `_field_ground.PairWeights.project`'s, which is where it lives for
        the whole field trunk since momwire#397 unit 1; what this method
        owns is the SCHEDULE — the observer band, the mixed-radius runs,
        and the choice of backend.

        PEC limit ε̃ → ∞: ρ_v → +1, ρ_h → −1, so the p̂ correction vanishes
        and this reduces exactly to `_field_tensor_image` — the ε̃=1e16
        collapse test rides on that. The returned tensors are SUBTRACTED
        in `_assemble_Z` with the same single global minus sign as the PEC
        image. Hot path is the fused C++ `sinusoidal_field_tensor_refl`
        kernel (Eqs 76-79 + the dyad projection in one pass; ρ_v/ρ_h
        computed in-kernel per pair), or `sinusoidal_field_tensor_ek_refl`
        — the same dyad tail over EKSCX's tables — when
        `extended_kernel` is set (momwire#259). The numpy formulation
        below is the bit-close reference / fallback for both.

        `obs_rows = (i0, i1)` restricts the observer axis, exactly as in
        `_field_tensor`. `_image_refl_band` builds the specular tables
        directly at (band, N_src) for that SAME band — never the whole
        (N_obs, N_src) table `_image_refl_prep` cached before #332's unit
        B — so the geometry slice and the specular tables share one index
        space (0 .. band size) by construction rather than by both taking
        the same absolute row slice; a stale pairing between them is no
        longer reachable here. `i0` below re-expresses the mixed-radius
        runs (absolute geometry indices) in that same band-relative space.

        `ground` is the `_field_ground.FieldGround` this band belongs to —
        the fill passes its own, so the mirror map, the per-pair weights
        and ε̃ all come from the one object (momwire#397 unit 2). Called
        without one (the direct-evaluator tests, and any caller outside a
        fill) it builds the solver's own, which is the same ground by
        construction. What the object decides here is the BACKEND: the
        fused C++ kernels compute ρ_v/ρ_h in-kernel, so they serve
        `standard_fresnel` grounds only, and a coefficient-modified ground
        (the radial-wire screen) falls to the numpy `project` path below
        without this method being edited for it.
        """
        if ground is None:
            ground = _field_ground.field_ground_for(self, geom, k, self.omega)
        src_c_img, src_t_img = ground.image_sources()
        seg_c = geom["seg_centers"]
        seg_t = geom["seg_tangents"]
        win = slice(None) if obs_rows is None else slice(*obs_rows)
        obs_c = seg_c[win]
        obs_t = seg_t[win]
        tabs = ground.pair_weights(obs_rows)
        # The five tables the fused kernels take positionally, unpacked so
        # their call sites read as they did before the shared builder.
        cos_th, px, py = tabs.cos_th, tabs.px, tabs.py
        tm_p, tn_p = tabs.tm_p, tabs.tn_p
        i0 = 0 if obs_rows is None else obs_rows[0]
        # ε̃(ω) — per-frequency (the swept loops update self.omega
        # alongside k before assembling), hoisted onto the ground object
        # so a banded fill pays it once rather than once per band.
        eps_t = ground.eps_tilde

        # `extended_kernel=True` has its own fused Fresnel kernel since
        # momwire#259 — EKSCX's E_z/E_ρ tables under the SAME dyad tail, which
        # is all the extended variant ever was: the Fresnel weighting rides on
        # top of the unprojected components and does not know which kernel
        # produced them. It is dispatched first, and separately, so the EK-OFF
        # marshalling below is untouched by it (the #233 off-path armor), and
        # it closes the last numpy-speed EK path on this family — PEC and
        # Sommerfeld image blocks already ride #245's kernel through
        # `_field_tensor_image` (Sommerfeld's is that PEC tensor times the
        # scalar C₂; see `_assemble_Z`).
        # The fused kernels transcribe the literal-cos closed forms, so a
        # `cos-1` request takes the numpy reference below (#606) — same rule,
        # and same reason, as `_field_tensor`'s `literal_cos` gate.
        literal_cos = cos_shape == "cos"
        if (
            _HAVE_FIELD_TENSOR_EK_REFL
            and self.extended_kernel
            and ground.standard_fresnel
            and literal_cos
        ):
            seg_h = geom["seg_h"]
            gx, gw = self._leggauss_cached(self.n_qp_const)
            # Source-indexed EK tables, exactly as `_field_tensor` passes
            # them: the mirror reorders nothing, so one IND1/IND2 pair serves
            # both KSYMP passes (nec2-1.2.1.2.f:2914-2971). EKSCX's IRA is not
            # among them: since momwire#258 it is resolved per PAIR inside the
            # kernel, off the MIRRORED sources' own rho_eval, so this build
            # reaches its own answer pair by pair without being told — and the
            # image build's answer can differ from free space's, pair for pair.
            ind1, ind2 = self._ek_gating(geom)
            src_a = np.ascontiguousarray(self._seg_radius(geom), dtype=np.float64)

            def _call_ek_refl(rel_rows, a):
                # `rel_rows` indexes the shared band-relative space (0 ..
                # band size) that `obs_c`/`obs_t` and the `_image_refl_band`
                # tables above are already built in — one slice serves both.
                return _acc.sinusoidal_field_tensor_ek_refl(
                    np.ascontiguousarray(obs_c[rel_rows], dtype=np.float64),
                    np.ascontiguousarray(obs_t[rel_rows], dtype=np.float64),
                    np.ascontiguousarray(src_c_img, dtype=np.float64),
                    np.ascontiguousarray(src_t_img, dtype=np.float64),
                    np.ascontiguousarray(seg_h, dtype=np.float64),
                    float(a),
                    float(k),
                    float(self.eta),
                    np.ascontiguousarray(gx, dtype=np.float64),
                    np.ascontiguousarray(gw, dtype=np.float64),
                    src_a,
                    np.ascontiguousarray(ind1, dtype=np.int8),
                    np.ascontiguousarray(ind2, dtype=np.int8),
                    np.ascontiguousarray(cos_th[rel_rows], dtype=np.float64),
                    np.ascontiguousarray(px[rel_rows], dtype=np.float64),
                    np.ascontiguousarray(py[rel_rows], dtype=np.float64),
                    np.ascontiguousarray(tm_p[rel_rows], dtype=np.float64),
                    np.ascontiguousarray(tn_p[rel_rows], dtype=np.float64),
                    complex(eps_t),
                    self._cancel_flag,
                )

            # Observer radius is still one scalar per call (EFLD's `ai`), so
            # mixed radii still split into constant-radius observer-row runs.
            # Spelled out rather than shared with the EK-OFF block below so
            # that block's diff stays empty (same reason as `_field_tensor`).
            if self._uniform_radius is not None:
                return _call_ek_refl(slice(None), self._uniform_radius)
            parts = [
                _call_ek_refl(slice(s - i0, e - i0), a)
                for s, e, a in self._radius_runs(geom, obs_rows)
            ]
            return tuple(
                np.concatenate([p[i] for p in parts], axis=0) for i in range(3)
            )

        # The reduced-kernel fused Fresnel kernel transcribes EKSC only, so an
        # EK-ON solve reaches this branch only when that accelerator is
        # unavailable — in which case it takes the numpy reference below, same
        # as an EK-OFF solve would.
        if (
            _HAVE_FIELD_TENSOR_REFL
            and not self.extended_kernel
            and ground.standard_fresnel
            and literal_cos
        ):
            # Fused C++ path: Eqs 76-79 field components + the Fresnel
            # dyad projection in one pass, with rho_v/rho_h computed
            # in-kernel per pair from eps_t and cos_th (same principal-
            # branch sqrt as _ground_refl.fresnel_rho). The numpy path
            # below is the bit-close reference / fallback.
            seg_h = geom["seg_h"]
            gx, gw = self._leggauss_cached(self.n_qp_const)

            def _call(rel_rows, a):
                # `rel_rows` indexes the shared band-relative space (0 ..
                # band size) that `obs_c`/`obs_t` and the `_image_refl_band`
                # tables above are already built in — one slice serves both.
                return _acc.sinusoidal_field_tensor_refl(
                    np.ascontiguousarray(obs_c[rel_rows], dtype=np.float64),
                    np.ascontiguousarray(obs_t[rel_rows], dtype=np.float64),
                    np.ascontiguousarray(src_c_img, dtype=np.float64),
                    np.ascontiguousarray(src_t_img, dtype=np.float64),
                    np.ascontiguousarray(seg_h, dtype=np.float64),
                    float(a),
                    float(k),
                    float(self.eta),
                    np.ascontiguousarray(gx, dtype=np.float64),
                    np.ascontiguousarray(gw, dtype=np.float64),
                    np.ascontiguousarray(cos_th[rel_rows], dtype=np.float64),
                    np.ascontiguousarray(px[rel_rows], dtype=np.float64),
                    np.ascontiguousarray(py[rel_rows], dtype=np.float64),
                    np.ascontiguousarray(tm_p[rel_rows], dtype=np.float64),
                    np.ascontiguousarray(tn_p[rel_rows], dtype=np.float64),
                    complex(eps_t),
                    self._cancel_flag,
                )

            if self._uniform_radius is not None:
                return _call(slice(None), self._uniform_radius)
            # Mixed per-wire radii: one call per constant-radius run of
            # observer rows (see `_field_tensor` / `_radius_runs`).
            parts = [
                _call(slice(s - i0, e - i0), a)
                for s, e, a in self._radius_runs(geom, obs_rows)
            ]
            return tuple(
                np.concatenate([p[i] for p in parts], axis=0) for i in range(3)
            )

        cm = self._field_components(
            geom,
            k,
            src_centers=src_c_img,
            src_tangents=src_t_img,
            cos_shape=cos_shape,
            **self._obs_window_kwargs(geom, obs_rows),
        )
        # `_image_refl_band` already built the specular tables at this
        # band's size — no row-slice, and no index arrays, because the
        # weights are ALREADY at the pairing `cm` is in (contrast the pre-
        # #332 unit B build, which sliced them out of a whole-geometry
        # cache, and the Galerkin consumer, which names its pairing).
        return tabs.weights(eps_t).project(cm)

    def _field_tensor_sommerfeld_remainder(
        self,
        geom,
        k,
        eps_t,
        obs_centers=None,
        obs_tangents=None,
        cos_shape="cos",
        consume=None,
        row_group=1,
        r1_max=None,
    ):
        """Smooth Sommerfeld-remainder tensor S[3, M, N]: the tangential
        remainder field of segment n's three source shapes (`cos_shape`) at the
        center of segment m, from the interpolated SommerfeldGrid F dyad
        (theory manual eqs 143-147 azimuth combination — the same algebra
        as `BSplineSolver._Z_sommerfeld_remainder`, minus the observer
        quadrature: this solver point-matches at segment centers, so only
        the SOURCE side integrates, GL with `n_qp_sommerfeld` nodes).

        Observers default to the geometry's segment centres/tangents
        (collocation). `SinusoidalGalerkinSolver` overrides them with the
        Gauss points along each test segment — the same reuse-the-evaluator
        pattern as `_field_components`'s `obs_*` overrides — so M is then
        the number of quadrature points rather than the number of segments.
        The grid extent below is derived from segment ENDPOINTS, which
        bounds any interior quadrature point, so it serves both callers.

        The grid's surfaces are the E-field of a unit current MOMENT
        (Il = 1, eq 123 normalization), so integrating shape(z')·F dz'
        along the segment gives the remainder field of the full source
        distribution — endpoint-charge contributions are inherent in the
        element superposition, matching the field-form Eqs 76-79 blocks.
        ADDED in `_assemble_Z` (`Phi_free - C2·Phi_img + S`): the sign is
        pinned by the cross-solver test against bspline-sommerfeld on a
        0.05-wl dipole (where the remainder is ~20 ohm, so a sign error
        is a ~2x miss), not derived here — same discipline as the
        ground-plan's PEC-limit sign pinning.

        `consume` is the streaming outlet (momwire#332 unit D). The evaluator
        has always walked its observers in chunks, sized so the grid
        interpolation's own (rows, N·q) block stays bounded; with a consumer
        given, each chunk's `(3, rows, N)` piece is handed straight to
        `consume(i0, i1, block)` and NOTHING is returned, so the full S never
        exists. That is the whole of unit D on the tested side: the Galerkin
        caller's observers are the test quadrature points, M = N·n_qp_test,
        and its S is `n_qp_test` times the matrix — several times the
        (nnz, N) triple it reduces to. The point-matched callers pass no
        consumer and get the tensor back exactly as before, chunk boundaries
        included.

        `row_group` is the consumer's alignment: chunks are rounded DOWN to a
        multiple of it (never below one group), so a caller whose reduction
        groups observers — the Galerkin one's `n_qp_test` nodes per test
        segment — sees every group whole inside one chunk and can reduce it
        without carrying a partial sum across the boundary. The default 1
        leaves the chunk arithmetic untouched.

        `r1_max` is the grid-sizing radius (`max_image_distance` over the
        geometry's endpoints, momwire#367): band- and k-invariant, so a
        caller invoking this evaluator once per observer band — the
        point-matched `_assemble_Z` fill — hoists the scan and passes it,
        the same once-per-fill discipline as its eps_t/C2 hoist. Left at
        None (the Galerkin caller, which calls once per solve), it is
        computed here exactly as before.

        A thin composition of `_sommerfeld_remainder_prepare` and
        `_replay_sommerfeld_remainder` (momwire#357 item 1) — the one-shot
        spelling, for callers that evaluate one observer set per k. The
        banded fill calls the two halves itself so the source-side working
        set is built once per fill instead of once per band.
        """
        prepared = self._sommerfeld_remainder_prepare(
            geom, k, eps_t, cos_shape=cos_shape, r1_max=r1_max
        )
        return self._replay_sommerfeld_remainder(
            prepared,
            obs_centers=obs_centers,
            obs_tangents=obs_tangents,
            consume=consume,
            row_group=row_group,
        )

    def _sommerfeld_remainder_prepare(
        self, geom, k, eps_t, cos_shape="cos", r1_max=None
    ):
        """Observer-INDEPENDENT half of the remainder evaluator (#357 item 1).

        Everything `_replay_sommerfeld_remainder` needs that does not depend
        on which observer rows it is asked for: the source-side quadrature
        nodes and tangents, the k-weighted source shapes, the interpolation
        grid, and the submerged-geometry refusal. A caller that walks the
        observer axis in bands — the point-matched `_assemble_Z` fill —
        calls this ONCE per fill and replays it per band, instead of
        rebuilding it per band the way a plain loop over one-shot calls
        does. Same shape as `RazorSolver._assemble_Z_prepare`: a plain
        dict, held by the caller for exactly as long as the loop it feeds,
        with no instance cache and so nothing to invalidate.

        Residency is O(N), never O(N²): the three tables are (N·q, 3),
        (N·q, 3) and (3, N, q) float64 — 0.09 MB each at N = 1200, q = 3.
        The grid is the module-level cached object, borrowed not copied.

        The state is keyed by its arguments alone (geometry, k, ε̃,
        `cos_shape`), so replaying it against a k the caller has since
        stepped away from is the one misuse available; `_assemble_Z` builds
        it inside the same scope that fixes k, and the Galerkin caller
        composes the two in one expression.
        """
        gz = self.ground_z
        seg_c = geom["seg_centers"]
        seg_t = geom["seg_tangents"]
        seg_l = geom["seg_l"]
        seg_r = geom["seg_r"]
        h_half = 0.5 * geom["seg_h"]  # (N,)
        N = geom["n_segs"]
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

        q = self.n_qp_sommerfeld
        gx, gw = self._leggauss_cached(q)
        zloc = h_half[:, None] * gx[None, :]  # (N, q) local arc offsets
        src = seg_c[:, None, :] + zloc[..., None] * seg_t[:, None, :]
        w_node = h_half[:, None] * gw[None, :]  # (N, q) dz' weights
        # Source shapes at the nodes, NATURAL-arc convention like the
        # free-space tensor (sigma accounting stays the caller's job). The
        # third shape follows the free-space tensor's `cos_shape` contract
        # (momwire#205) — this block is SUBTRACTED from one built there, so
        # both have to be on the same shape set. Here the fold is only a
        # better-scaled weight inside a quadrature (no cancelling closed
        # forms), and −2sin²(kξ/2) keeps it accurate at its own size.
        shp_cos = (
            np.cos(k * zloc)
            if cos_shape == "cos"
            else -2.0 * np.sin(0.5 * k * zloc) ** 2
        )
        shp = np.stack([np.ones_like(zloc), np.sin(k * zloc), shp_cos])  # (3, N, q)

        # Grid extent: obs-to-image-point distance is convex in the two
        # segment parameters, so its max is attained at endpoint pairs.
        if r1_max is None:
            r1_max = _sommerfeld.max_image_distance(seg_l, seg_r, gz)
        grid = _sommerfeld.get_grid(
            eps_t,
            k,
            r1_max,
            omega=self.omega,
            mu=self.mu,
            cancel_flag=self._cancel_flag,
        )

        n_src = N * q
        return {
            "k": k,
            "gz": gz,
            "N": N,
            "q": q,
            "n_src": n_src,
            "grid": grid,
            "srcf": src.reshape(n_src, 3),
            "t_src": np.repeat(seg_t, q, axis=0),
            # `shp * w_node[None]`, the one k-dependent table, folded here
            # rather than in the replay so the per-band pass owns nothing
            # that outlives its own chunk.
            "shp_w": shp * w_node[None],
            "seg_c": seg_c,
            "seg_t": seg_t,
        }

    def _replay_sommerfeld_remainder(
        self,
        prepared,
        obs_centers=None,
        obs_tangents=None,
        consume=None,
        row_group=1,
    ):
        """Observer-DEPENDENT half: evaluate `prepared` at one observer set.

        Chunking, streaming (`consume`), and `row_group` alignment are
        exactly as `_field_tensor_sommerfeld_remainder` documents them —
        this IS that loop, with the source-side build lifted out. Every
        expression the block is built from is untouched, and the chunk
        boundaries are still derived from `_REMAINDER_CHUNK_ELEMS // n_src`,
        so the floats are the one-shot spelling's floats bit for bit.
        """
        N = prepared["N"]
        q = prepared["q"]
        gz = prepared["gz"]
        k = prepared["k"]
        grid = prepared["grid"]
        srcf = prepared["srcf"]
        t_src = prepared["t_src"]
        shp_w = prepared["shp_w"]

        obs_c = (
            prepared["seg_c"]
            if obs_centers is None
            else np.asarray(obs_centers, dtype=float)
        )
        obs_t = (
            prepared["seg_t"]
            if obs_tangents is None
            else np.asarray(obs_tangents, dtype=float)
        )
        M = obs_c.shape[0]

        S = None if consume is not None else np.empty((3, M, N), dtype=np.complex128)
        chunk = max(1, _REMAINDER_CHUNK_ELEMS // max(prepared["n_src"], 1))
        if row_group > 1:
            if M % row_group:
                # The last chunk would be a partial group, which is the one
                # thing the alignment exists to prevent. Raise rather than
                # hand a consumer a group it cannot reduce whole.
                raise ValueError(
                    f"observer count {M} is not a multiple of row_group {row_group}"
                )
            chunk = max(row_group, (chunk // row_group) * row_group)
        for i0 in range(0, M, chunk):
            self._checkpoint()  # per observer chunk of the eval block
            i1 = min(i0 + chunk, M)
            proj = _sommerfeld.remainder_field_proj(
                obs_c[i0:i1], obs_t[i0:i1], srcf, t_src, gz, k, grid
            )
            fq = proj.reshape(i1 - i0, N, q)
            # Per output element this is a sum over the q source nodes and
            # nothing else, so it does not see the chunk it is in: the block
            # a consumer gets is bit-identical to the same rows of the whole
            # S, at any chunk size.
            block = np.einsum("snq,mnq->smn", shp_w, fq)
            del proj, fq
            if consume is None:
                S[:, i0:i1, :] = block
            else:
                consume(i0, i1, block)
            del block
        return S

    # ------------------------------------------------------------------
    # The ground-contact node charge over a FINITE ground (#282)
    # ------------------------------------------------------------------

    def _contact_nodes(self, geom):
        """Every wire end that LIES IN the ground plane, as
        `(segment, sign, node_point)` — sign = −1 when the plane is at the
        segment's end-1 (N⁻ side) and +1 at its end-2, so the node sits at
        local arc `sign·h/2`. Empty unless there is a ground.
        """
        if self.ground_z is None:
            return []
        gm, gp = geom["ground_minus"], geom["ground_plus"]
        seg_c, seg_t, seg_h = (
            geom["seg_centers"],
            geom["seg_tangents"],
            geom["seg_h"],
        )
        out = []
        for mask, sgn in ((gm, -1.0), (gp, +1.0)):
            for i in np.flatnonzero(mask):
                out.append((int(i), sgn, seg_c[i] + sgn * 0.5 * seg_h[i] * seg_t[i]))
        return out

    def _contact_node_values(self, geom, k, seg_view, i, sgn):
        """`(bases, values)`: what every basis with support on segment `i`
        is worth AT the contact node — the current each one delivers into
        the plane, and so the charge each one would leave there.

        Away from a junction only the segment's own basis is nonzero here
        (a neighbour's extension vanishes at the node it does not touch),
        but the whole CSR block is evaluated so a multi-wire junction
        standing on the plane is covered without a special case.
        """
        blk = slice(seg_view["starts"][i], seg_view["starts"][i + 1])
        sig = seg_view["sigma"][blk]
        s = sgn * 0.5 * geom["seg_h"][i]
        # Well-scaled spelling (#606): σA + B·sin kξ + σC·cos kξ rewritten as
        # σ·AC + B·sin kξ + σC·(cos kξ − 1), with cos kξ − 1 = −2sin²(kξ/2) so
        # neither the coefficient sum nor the trig term is a subtraction of
        # two quantities about to agree.
        half = np.sin(0.5 * k * s)
        val = (
            sig * seg_view["AC"][blk]
            + seg_view["B"][blk] * np.sin(k * s)
            - sig * seg_view["C"][blk] * 2.0 * half * half
        )
        return seg_view["jbasis"][blk], val

    def _contact_charge_fresnel(self, d, r2, obs_c, obs_t, eps_t):
        """The reflection-coefficient ground's specular geometry at a contact
        node, shared by the reduced and extended charge kernels.

        The node IS its own mirror image, so the specular ray is simply
        (node → observer): cos θ is the observer's height over the plane
        divided by that distance, and p̂ — the horizontal unit vector the
        Fresnel dyad resolves the horizontal polarization on — is
        perpendicular to the ray's horizontal projection. Returns
        `(ρ_v, ρ_h, t·p̂, p̂_x, p̂_y, p̂·d)`; the last is the reduced kernel's
        own E·p̂ up to its prefactor (its E is parallel to `d`), while the
        extended kernel resolves E·p̂ from p̂_x/p̂_y itself because its E is
        not.
        """
        dz = obs_c[..., 2] - self.ground_z
        rmag = np.sqrt(np.maximum(r2, _CONTACT_TINY))
        rho_v, rho_h = _ground_refl.fresnel_rho(eps_t, dz / rmag)
        hyp = np.hypot(d[..., 0], d[..., 1])
        safe = hyp > _CONTACT_TINY
        inv_hyp = np.where(safe, 1.0 / np.where(safe, hyp, 1.0), 1.0)
        px = np.where(safe, -d[..., 1] * inv_hyp, 1.0)
        py = np.where(safe, d[..., 0] * inv_hyp, 0.0)
        t_p = obs_t[..., 0] * px + obs_t[..., 1] * py
        p_d = px * d[..., 0] + py * d[..., 1]
        return rho_v, rho_h, t_p, px, py, p_d

    def _contact_charge_kernel(self, geom, k, node, obs_c, obs_t, a_obs):
        """Per-observer field of the SPURIOUS charge a unit contact current
        leaves at `node`, tangentially projected — the #282 correction's
        whole content, on the REDUCED kernel's end-charge bracket. When the
        fill extended that end, `_contact_charge_ek_delta` (#292) adds the
        difference to EKSCX's bracket on top of what this returns.

        Over a PEC plane a wire end in the plane is charge-free: the end
        charge the wire's own field carries (+I/jω, the boundary term every
        one of Eqs 76-79 puts at a segment end) is cancelled exactly by the
        image's end charge at the SAME point, because the image current
        equals the wire current there. Over a finite ground the image is
        scaled by ρ, so the cancellation leaves (1−ρ)·I/jω sitting ON the
        plane — a point charge whose potential at the nearest collocation
        point grows like 1/Δ. That is #282: |Z| walks away under mesh
        refinement instead of settling, in proportion to |1−ρ|, on both
        sinusoidal solvers and in nec2c itself.

        The charge is not physics; it is double-counting. ρ < 1 is already
        the model's statement that the earth takes the current the plane
        does not reflect — piling that same current up as a point charge on
        the wire end charges the earth twice. The mixed-potential solvers
        never had it: `BSplineSolver` builds its charge term from the basis
        DERIVATIVE over the support, so a ground-contact basis simply has
        no end charge, which is why it converges here and is the reference
        this correction restores agreement with.

        So subtract the residual charge's field. For a unit terminating
        current the deposited charge is I/(jω), and with the kernel's own
        thin-wire regularization r₀ = √(|d|² + a²) its field is

            E = −jη/(4πk) · (1 + jkr₀)·e^{−jkr₀}/r₀³ · d,   d = r_obs − node

        which is exactly the (1+jkr₀)(z−z')G₀/r₀² boundary term the closed
        forms carry, in the same normalization — the cancellation is
        algebraic, not a fit. The image's copy of it sits at the SAME point
        (the node is its own mirror image) and is weighted by the ground:
        the scalar C₂ under Sommerfeld, the Fresnel dyad under the
        reflection-coefficient ground. What is returned is the residual,
        real minus image-weighted, which is identically zero in the PEC
        limit (ρ_v → 1, ρ_v + ρ_h → 0) and so leaves elevated and PEC
        geometries untouched.
        """
        d = obs_c - node
        r2 = np.einsum("...d,...d->...", d, d)
        a2 = np.asarray(a_obs, dtype=float)
        a2 = a2.reshape(a2.shape[:1]) if a2.ndim == 2 else a2
        r0 = np.sqrt(r2 + a2 * a2)
        pref = (
            -1j
            * self.eta
            / (4.0 * np.pi * k)
            * (1.0 + 1j * k * r0)
            * np.exp(-1j * k * r0)
            / (r0 * r0 * r0)
        )
        t_d = np.einsum("...d,...d->...", obs_t, d)
        eps_t = _ground_refl.eps_tilde(self.ground_eps, self.omega, self.eps)
        if self.ground_model == "sommerfeld":
            # The image term the C₂ scalar multiplies is the whole of the
            # Sommerfeld model's point-charge behaviour at the contact; the
            # interpolated remainder is a smooth correction on the REAL
            # source's ground response, with no 1/R there and so no share
            # of the charge being cancelled.
            return pref * (1.0 - (eps_t - 1.0) / (eps_t + 1.0)) * t_d
        # Reflection-coefficient ground: the same dyad the image tensor
        # applies, evaluated on the specular ray from the node — which IS
        # its own image, so the ray is (node → observer) and cos θ = the
        # observer's height over the plane divided by that distance.
        # t·D·E = ρ_v(t·E) − (ρ_v + ρ_h)(t·p̂)(E·p̂) with E ∝ d.
        rho_v, rho_h, t_p, _px, _py, p_d = self._contact_charge_fresnel(
            d, r2, obs_c, obs_t, eps_t
        )
        return pref * ((1.0 - rho_v) * t_d + (rho_v + rho_h) * t_p * p_d)

    def _contact_charge_end_gradient(self, geom, k, i, sgn, obs_c, obs_t, a_obs):
        """∇G_ext at the contact end of segment `i`, in the two components
        the fill resolves fields on: `(E_z, E_ρ/ρ_eval, t_src, rho_vec)`.

        This is the extended kernel's answer to the same question
        `_contact_charge_kernel`'s prefactor answers for the reduced one —
        what field does the charge sitting at this end produce, per unit
        terminating current — and it is read straight off `_ek_end_gxx`
        rather than re-derived, because the point of #292 is that the
        subtraction must be the SAME bracket the fill carries.

        NEC's GXX returns exactly the gradient of Eq 89's circumferentially
        averaged kernel: `G1P = ∂G_ext/∂z` and `G3 = ∂G_ext/∂ρ` — the
        ρ-derivative needing both the explicit ρ² inside T1 and the chain
        rule through R, which is precisely why GXX's IRA == 0 arm carries
        `(G3 + GZP_T)·RH` and not `G3·RH`, and the identity being gated by
        `test_292_gxx_returns_the_gradient_of_the_extended_kernel`. EKSCX spends
        them exactly there: an end's share of `Ez_const` is −CON·G1P and of
        `Erho_const` is +CON·G3, per unit current terminating at that end,
        with the sin and cos shapes carrying the same pair scaled by their
        own end value. So

            E_end = CON·∇G_ext = −CON·G1P·t̂_src + CON·G3·ρ̂

        with ρ̂ = rho_vec/ρ_eval — NEC's regularized ρ̂, the same one
        `_field_components_bcast`'s `rho_proj_factor` divides by, so the
        projection matches the fill's term for term. In the a → 0 limit
        G1P → G1P_red and G3 → G3_red·ρ and the pair collapses onto
        `_contact_charge_kernel`'s isotropic (1+jkr₀)e^{−jkr₀}d/r₀³.

        The O(a²/R²) difference between the two is ~10 % of the charge at
        the meshes #292 measured, and — being a difference of the SUBTRACTED
        term rather than of the fill — it does not shrink under refinement:
        it is the whole of the EK-on residue #282 left behind. On the G16c
        deck the EK-on ladder ran spread 1.56 over NS = 11 → 41 against
        EK-off's 0.13, and lands at 0.127 with this bracket in.

        NOT gated here: the caller has already established that this end is
        EK-eligible. `(rh, b)` ordering and the IRA arm are per pair exactly
        as in `_extended_kernel_fields` (#258) — the observation point can
        sit inside the source conductor here as anywhere else.
        """
        src_c = geom["seg_centers"][i]
        src_t = geom["seg_tangents"][i]
        src_hh = 0.5 * float(geom["seg_h"][i])
        src_a = (
            self._uniform_radius
            if self._uniform_radius is not None
            else float(self._seg_radius(geom)[i])
        )
        rvec = obs_c - src_c
        z_eval = rvec @ src_t
        rho_vec = rvec - z_eval[..., None] * src_t
        rho_axis = np.linalg.norm(rho_vec, axis=-1)
        a = np.asarray(a_obs, dtype=float)
        a = a.reshape(a.shape[:1]) if a.ndim == 2 else a
        rho_eval = np.sqrt(rho_axis * rho_axis + a * a)
        # NEC's Z2 = SH − Z at end 2, Z1 = −(SH + Z) at end 1; `sgn` is +1
        # for a `ground_plus` contact (end 2) and −1 for `ground_minus`.
        zz = (src_hh - z_eval) if sgn > 0 else -(src_hh + z_eval)
        swap = rho_eval < src_a
        any_swap = bool(np.any(swap))
        rh = np.where(swap, src_a, rho_eval) if any_swap else rho_eval
        b = (
            np.where(swap, rho_eval, src_a)
            if any_swap
            else np.full_like(rho_eval, src_a)
        )
        _g1, g1p, _g2, _g2p, g3, _gzp = self._ek_end_gxx(
            k, zz, rh, b, swap if any_swap else False
        )
        con = 1j * self.eta / (4.0 * np.pi * k)
        return -con * g1p, con * g3 / rho_eval, src_t, rho_vec

    def _contact_charge_ek_delta(
        self, geom, k, i, sgn, node, obs_c, obs_t, a_obs, use_real, use_image
    ):
        """How much `_contact_charge_kernel`'s residual changes when the
        fill's end-charge bracket is EKSCX's rather than the reduced
        kernel's — the whole of momwire#292, added on top of #282 rather
        than replacing it so that an EK-off fill runs not one extra
        floating-point operation.

        Where the reduced kernel's terminating charge is a POINT on the
        axis, radiating isotropically so that `t·E ∝ t·d` and `p̂·E ∝ p̂·d`,
        the extended kernel's is the same charge smeared over the source
        tube's circumference. Its field is anisotropic — axial and radial
        parts carrying different O(a²) corrections
        (`_contact_charge_end_gradient`) — so both projections have to be
        resolved from the decomposition instead of from `d`:

            t·E = E_z·(t_obs·t̂_src) + (E_ρ/ρ_eval)·(rho_vec·t_obs)
            p̂·E = E_z·(p̂·t̂_src)    + (E_ρ/ρ_eval)·(rho_vec·p̂)

        The residual #282 subtracts is `E_real − W[E_image]` with W the
        ground's weighting (the scalar C₂ under Sommerfeld, the Fresnel
        dyad under the reflection-coefficient ground), and both sides of it
        are linear in the bracket. So the amendment is

            Δ = use_real·(t·δE) − use_image·W[δE],   δE = E_ext − E_red

        with `use_real` / `use_image` the per-observer masks saying where
        the fill actually carried the extended bracket on that side. They
        are separate because they CAN differ: `SinusoidalGalerkinSolver`
        scores eligibility per pair against the mirrored sources
        (`_ek_axis_labels(mirror=True)`), so a slanted wire standing on the
        plane is coaxial with itself but not with its image. On the
        point-matched solver they never differ — NEC threads one IND1/IND2
        pair through both passes of its KSYMP image loop, and a contact is
        EK-gated at all only when the segment is PERPENDICULAR to the
        plane, whose mirror is its own axis with t̂ → −t̂; that flips the
        sign of both `zz` and `t̂_src` and so leaves the end-charge VECTOR,
        rho_vec included, exactly equal to the real one.

        Identically zero in the PEC limit, like the term it amends: both
        `use_*` branches then carry the same δE and W is the identity.
        """
        ez, erho_hat, src_t, rho_vec = self._contact_charge_end_gradient(
            geom, k, i, sgn, obs_c, obs_t, a_obs
        )
        d = obs_c - node
        r2 = np.einsum("...d,...d->...", d, d)
        a2 = np.asarray(a_obs, dtype=float)
        a2 = a2.reshape(a2.shape[:1]) if a2.ndim == 2 else a2
        r0 = np.sqrt(r2 + a2 * a2)
        pref = (
            -1j
            * self.eta
            / (4.0 * np.pi * k)
            * (1.0 + 1j * k * r0)
            * np.exp(-1j * k * r0)
            / (r0 * r0 * r0)
        )
        rho_t = np.einsum("...d,...d->...", rho_vec, obs_t)
        t_d = np.einsum("...d,...d->...", obs_t, d)
        dt = ez * (obs_t @ src_t) + erho_hat * rho_t - pref * t_d
        zero = np.zeros((), dtype=np.complex128)
        eps_t = _ground_refl.eps_tilde(self.ground_eps, self.omega, self.eps)
        if self.ground_model == "sommerfeld":
            # As in the reduced kernel: C₂ carries the whole of the
            # Sommerfeld image's point-charge behaviour at the contact and
            # the interpolated remainder takes no share of it.
            c2 = (eps_t - 1.0) / (eps_t + 1.0)
            return np.where(use_real, dt, zero) - c2 * np.where(use_image, dt, zero)
        rho_v, rho_h, t_p, px, py, p_d = self._contact_charge_fresnel(
            d, r2, obs_c, obs_t, eps_t
        )
        dp = (
            ez * (src_t[0] * px + src_t[1] * py)
            + erho_hat * (rho_vec[..., 0] * px + rho_vec[..., 1] * py)
            - pref * p_d
        )
        img = rho_v * dt - (rho_v + rho_h) * t_p * dp
        return np.where(use_real, dt, zero) - np.where(use_image, img, zero)

    def _contact_ek_masks(self, geom, i, sgn, obs_seg):
        """`(use_real, use_image)` for `_contact_charge_ek_delta` at the
        contact end `(i, sgn)`, or None when this end kept the REDUCED
        bracket and #282's correction already matches the fill.

        The point-matched fill decides per source-segment END, with NEC's
        IND codes (`_ek_gating`): code 2 routes the end to the reduced GX,
        codes 0 and 1 to GXX. A ground contact earns code 0 only when the
        segment is perpendicular to the plane, so a SLANTED contact keeps
        the reduced bracket even with EK on. The decision does not depend
        on the observer, and it is the same on the image pass, hence the
        two scalar `True`s. `SinusoidalGalerkinSolver` overrides this: its
        extended kernel is #246's per-PAIR rule, not NEC's per-end one, so
        its masks are per observer and its two sides can disagree.
        """
        if not self.extended_kernel:
            return None
        ind1, ind2 = self._ek_gating(geom)
        if int((ind2 if sgn > 0 else ind1)[i]) == 2:
            return None
        return True, True

    def _contact_charge_correction(self, G, geom, k, seg_view):
        """Subtract the #282 residual contact charge from a collocated fill,
        in place. No-op without a finite ground or without a contact, so
        PEC and elevated geometries keep their base arithmetic exactly.

        Each contact end takes the bracket its own fill carried: the
        reduced kernel's always, plus #292's `_contact_charge_ek_delta`
        wherever the extended kernel reached that end. With EK off the
        second term is never built and not one floating-point operation
        here changes.
        """
        if self.ground_eps is None:
            return G
        nodes = self._contact_nodes(geom)
        if not nodes:
            return G
        obs_c, obs_t = geom["seg_centers"], geom["seg_tangents"]
        a_obs = self._seg_radius(geom)
        obs_seg = np.arange(geom["n_segs"], dtype=np.int64)
        for i, sgn, node in nodes:
            R = self._contact_charge_kernel(geom, k, node, obs_c, obs_t, a_obs)
            masks = self._contact_ek_masks(geom, i, sgn, obs_seg)
            if masks is not None:
                R = R + self._contact_charge_ek_delta(
                    geom, k, i, sgn, node, obs_c, obs_t, a_obs, *masks
                )
            jb, val = self._contact_node_values(geom, k, seg_view, i, sgn)
            G[:, jb] -= sgn * R[:, None] * val[None, :]
        return G

    def _sommerfeld_ground(self):
        """True when the fill carries NEC's exact-image + interpolated
        remainder decomposition rather than a PEC or Fresnel image.

        Since momwire#429 unit 2 this predicate has exactly ONE reader:
        `_fill_row_bytes`, which is residency arithmetic rather than
        physics and counts blocks with it. The factory that used to be the
        other reader now asks `_ground_spec.ground_config`, whose
        `ground_model == "sommerfeld"` is this test with the two leading
        conjuncts already established by the branches above it — the same
        answer, spelled once for both trunks. The fill itself branches on
        the ground object and reads no strings.
        """
        return (
            self.ground_z is not None
            and self.ground_eps is not None
            and self.ground_model == "sommerfeld"
        )

    def _fill_row_bytes(self, N):
        """Bytes one observer row of `_assemble_Z`'s band costs, in
        complex128 rows of N sources.

        The band's high-water is whichever of its two phases is taller:

          * the FILL — one row per shape of every block the ground model
            switches on: free space always (3), the image under any ground
            (3 more), the Sommerfeld remainder on top of that (3 more).
            They coexist because the image is folded into the free-space
            band and the remainder into the image band, so 3·blocks;
          * the REDUCTION — the surviving free-space band (3) plus TWO.
            One is the matmul product; the three products are accumulated
            into Z one at a time rather than summed as an expression, so
            only one is ever live. The other is scipy's: `dense @ sparse`
            runs as `(M.T @ Φ.T).T`, and the sparse matmul needs its dense
            operand C-contiguous, so Φ.T is copied. Measured at N = 1200,
            single band: 6x Z peak free space, which is Z + 3 + 2 exactly.

        Nothing else in the loop exceeds O(N), so dividing `swept_mem_mb`
        by this bounds the whole fill transient. It is a bound on the
        BANDS, not a guarantee about the fill: marshalling overhead runs
        it ~1.7x over at small band heights, the same overshoot #338
        tracks on the bspline side."""
        blocks = 1 + (self.ground_z is not None) + self._sommerfeld_ground()
        return max(3 * blocks, 3 + 2) * N * 16

    def _assemble_Z(self, geom, k):
        """Point-matched Z, filled in observer-row chunks (momwire#332).

        The row band is the natural chunk: every Φ block below is
        rectangular in the observer count already (the mixed-radius
        dispatch has cut the kernels on that axis since #147), and the
        reduction into Z is a matmul on the SOURCE axis, so one output row
        of Z depends on exactly one input row of Φ. Row banding therefore
        leaves each output element's reduction order untouched — this is a
        residency change, not a reassociation, and the answer is bit-equal
        to the whole-tensor build (`test_sinusoidal_banded_assembly_is_
        bit_equal`).

        What that retires, measured at N = 1200 above Z: 110 MB free space
        (the Φ triple plus the matmul temporaries), 176 MB under a ground
        (the image triple on top of that), 242 MB under sommerfeld (the
        remainder triple and the C2-scaled differences on top of THAT) —
        5x, 8x and 11x Z. Peak is now Z plus one band, bounded by
        `swept_mem_mb`.
        """
        seg_view = self._basis_coefs(geom, k)
        N = geom["n_segs"]
        # Build (N, N) coefficient matrices M_{A,B,C}[n, j] = effective
        # coefficient that basis j contributes at source segment n. With
        #   A_eff = σ * A, B_eff = B, C_eff = σ * C
        # (see docs/sinusoidal_basis_design.md), the per-basis loop reduces
        # to three N×N matmuls G = Phi_c @ M_A + Phi_s @ M_B + Phi_co @ M_C.
        #
        # seg_view is already CSR-by-segment, so the row coordinate n_idx
        # is just np.repeat(arange(N), counts) and σA / σC are element-wise
        # numpy products. No Python scatter loop here — the per-basis fill
        # already happened directly into seg_view inside _basis_coefs.
        #
        # Each basis has only ~3 entries (self + N⁻ + N⁺ neighbour), so M
        # is very sparse (~3N nonzeros). Two regimes:
        #   N < _DENSE_ASSEMBLY_THRESHOLD: dense matmul wins because the
        #     scipy.sparse constructor overhead dominates the BLAS call.
        #   N ≥ threshold: sparse matmul wins because BLAS zgemm pays the
        #     full O(N³) cost on a mostly-zero matrix, while CSC matmul
        #     pays O(N · 3N) = O(N²).
        # Crossover measured ≈ N=60 on Kaby Lake R / OpenBLAS-pthreads.
        starts = seg_view["starts"]
        n_idx_arr = np.repeat(np.arange(N, dtype=np.int64), starts[1:] - starts[:-1])
        j_idx_arr = seg_view["jbasis"]
        sigma_arr = seg_view["sigma"]
        # Which shape set this fill is written in (#606). The switch is one
        # decision made here and carried to every block — direct, image and
        # remainder — because the image tensor is SUBTRACTED from the direct
        # one and the remainder is added to it: mixing spellings between them
        # would not be a loss of accuracy but a different operator.
        #
        #   literal :  Φ_c @ σA  + Φ_s @ B + Φ_co @ σC
        #   scaled  :  Φ_c @ σAC + Φ_s @ B + Φ_d  @ σC ,  Φ_d = Φ(cos kξ − 1)
        #
        # identical in exact arithmetic (AC = A + C, Φ_d = Φ_co − Φ_c), and
        # the second is the one whose terms are not larger than its answer.
        # `Φ_d` has to come from its own closed form, not from the difference
        # of two computed tensors — that subtraction reintroduces the same
        # ε·‖Φ_c‖ the reformulation exists to remove, and `Φ_d @ σC` is 84 %
        # of the fill at kΔ = 2.1e-4, not a correction term.
        #
        # The EXTENDED kernel is included since momwire#614: EKSCX's folded
        # tables are the same G-quantities rearranged, so the per-end IND
        # contract serves `cos-1` too and the switch is purely about kΔ.
        # (#606 shipped with an EK carve-out and a warning; both went away
        # with the limit they described.)
        kd_min = float(np.min(k * np.asarray(geom["seg_h"], dtype=np.float64)))
        cos_shape = "cos-1" if kd_min < _WELL_SCALED_KD else "cos"
        A_eff = sigma_arr * (seg_view["A"] if cos_shape == "cos" else seg_view["AC"])
        B_eff = seg_view["B"]
        C_eff = sigma_arr * seg_view["C"]
        if N < _DENSE_ASSEMBLY_THRESHOLD:
            M_A = np.zeros((N, N), dtype=np.complex128)
            M_B = np.zeros((N, N), dtype=np.complex128)
            M_C = np.zeros((N, N), dtype=np.complex128)
            M_A[n_idx_arr, j_idx_arr] = A_eff
            M_B[n_idx_arr, j_idx_arr] = B_eff
            M_C[n_idx_arr, j_idx_arr] = C_eff
        else:
            M_A = scipy.sparse.csc_matrix((A_eff, (n_idx_arr, j_idx_arr)), shape=(N, N))
            M_B = scipy.sparse.csc_matrix((B_eff, (n_idx_arr, j_idx_arr)), shape=(N, N))
            M_C = scipy.sparse.csc_matrix((C_eff, (n_idx_arr, j_idx_arr)), shape=(N, N))

        # The ground, as ONE object (momwire#397 unit 2): which per-pair
        # weight, which image coefficient, which composition. `None` is
        # free space, and the band branch below is then structurally
        # absent rather than skipped. Nothing here reads `ground_z`,
        # `ground_eps` or `ground_model` anymore — the strings are the
        # factory's, and what this fill branches on is `mode`.
        #
        # Everything per-fill and observer-independent is paid at
        # construction or on the `remainder()` call: ε̃ and C2 (per
        # frequency, not per chunk — swept loops update self.omega
        # alongside k), the grid-sizing endpoint scan (band-invariant, and
        # at ~46 bands x O(N^2) the dominant cost of the banded fill before
        # momwire#367), and the remainder's whole SOURCE side — quadrature
        # nodes, their tangents, the k-weighted shapes, the grid handle —
        # which the bands replay instead of rebuilding (momwire#357 item
        # 1). O(N), and it dies with `fg` at the end of this fill.
        fg = _field_ground.field_ground_for(self, geom, k, self.omega)
        somm_rem = None if fg is None else fg.remainder(cos_shape=cos_shape)

        # Below the dense-M threshold the whole fill is one chunk, budget or
        # no budget. Two reasons, and the first alone would settle it:
        #   * there is nothing to save — at N < 60 the entire whole-tensor
        #     residency is under 700 KB, so no reachable `swept_mem_mb` is
        #     violated by taking it in one bite;
        #   * the reduction there is a DENSE zgemm, and BLAS picks its
        #     k-blocking off the operand shape, so the same row's dot
        #     product reassociates when the row count changes. Measured on
        #     OpenBLAS at N=25: chunked-vs-whole deltas of 8.6e-15 relative
        #     at chunk=1 and 1.7e-17 at chunk=24, i.e. real reassociation,
        #     not a slicing bug. The sparse-M regime has no such freedom —
        #     each output element sums that column's ~3 nonzeros in CSC
        #     order — and is bit-equal at every chunk size.
        chunk = (
            N
            if N < _DENSE_ASSEMBLY_THRESHOLD
            else max(1, int(self.swept_mem_mb * 1024 * 1024 // self._fill_row_bytes(N)))
        )

        G = np.zeros((N, N), dtype=np.complex128)
        seg_c = geom["seg_centers"]
        seg_t = geom["seg_tangents"]
        for i0 in range(0, N, chunk):
            self._checkpoint()  # per observer chunk of the fill
            i1 = min(i0 + chunk, N)
            rows = (i0, i1)
            Phi_c, Phi_s, Phi_co = self._field_tensor(
                geom, k, obs_rows=rows, cos_shape=cos_shape
            )
            if fg is not None:
                # Image ground: subtract the sub-assembly built from the
                # image field tensor. The image source's mirrored geometry +
                # flipped z-tangent already encode both the anti-parallel
                # horizontal image current and the parallel vertical image
                # current; the combined image-current + image-charge sign
                # flip reduces to a single minus sign on the image-Z block
                # (same as BSpline). Which per-pair weight rides on that
                # image — none for PEC, the Fresnel dyad under `ground_eps`
                # (NEC IPERF=0) — is `fg.image_field`'s decision, not this
                # loop's; the subtraction sign is unchanged either way
                # because the weighted tensor reduces to the PEC one in the
                # ε̃ → ∞ limit.
                if fg.mode == "compose":
                    # `coef·img − remainder` has to be associated BEFORE the
                    # single minus below, because `free − (c2·img − S)` is
                    # not `(free − c2·img) + S` in float64. That is the
                    # whole of what "compose" declares, and it is the ONE
                    # ground fact this schedule needs (sommerfeld today).
                    #
                    # NEC's decomposition (theory manual eqs 136-147):
                    # exact image scaled by the constant C2, which absorbs
                    # all the singular behavior — a plain scalar on the
                    # projected PEC-image tensor, so the C++ kernel keeps
                    # serving it — plus the smooth interpolated remainder,
                    # which ADDS (see the remainder method's docstring for
                    # the sign pinning). eps->inf: C2 -> 1, S -> 0, PEC
                    # image exactly; eps -> 1: both vanish, free space.
                    #
                    # The remainder's replay half already takes observer
                    # centres/tangents, so the band is a plain argument
                    # there; it is the only block that has to be told which
                    # rows it is on rather than shown them. Its
                    # observer-independent half was prepared once above
                    # (#357 item 1).
                    Phi_i = list(fg.image_field(obs_rows=rows, cos_shape=cos_shape))
                    S = somm_rem.replay(
                        obs_centers=seg_c[i0:i1],
                        obs_tangents=seg_t[i0:i1],
                    )
                    # `c2 * Φ_img − S` in place, and with C2 on the LEFT.
                    # Not cosmetic: complex128 multiply evaluates the
                    # imaginary part as x.re*y.im + x.im*y.re, so swapping
                    # the operands reorders that sum and moves the last bit
                    # (measured: 1e-17 relative on the assembled Z, i.e. one
                    # ULP, on every Sommerfeld config). `Pi *= c2` is the
                    # swapped order; `np.multiply(c2, Pi, out=Pi)` is the
                    # whole-matrix build's order, in place — and it is the
                    # coefficient's own interface contract, which is why the
                    # left operand is read off the object here.
                    c2 = fg.image_coefficient
                    for Pi, Si in zip(Phi_i, S):
                        np.multiply(c2, Pi, out=Pi)
                        Pi -= Si
                    del S
                else:
                    Phi_i = fg.image_field(obs_rows=rows, cos_shape=cos_shape)
                Phi_c -= Phi_i[0]
                Phi_s -= Phi_i[1]
                Phi_co -= Phi_i[2]
                del Phi_i
            # One product at a time, released as it lands, rather than
            # `(Φ_c@M_A) + (Φ_s@M_B) + (Φ_co@M_C)` — that expression holds
            # two products plus their sum at once, which would make the
            # reduction, not the fill, the band's high-water in free space.
            # Same association: `x = A; x += B; x += C` is `(A + B) + C`.
            G[i0:i1] = Phi_c @ M_A
            del Phi_c
            G[i0:i1] += Phi_s @ M_B
            del Phi_s
            G[i0:i1] += Phi_co @ M_C
            del Phi_co
        self._contact_charge_correction(G, geom, k, seg_view)
        self._apply_loading(G, geom, seg_view, k)
        return G, seg_view

    @staticmethod
    def _wire_of_seg(geom):
        """(n_segs,) int array mapping segment index → wire index."""
        firsts = np.asarray(geom["wire_first"], dtype=np.int64)
        lasts = np.asarray(geom["wire_last"], dtype=np.int64)
        return np.repeat(np.arange(firsts.shape[0]), lasts - firsts + 1)

    def _seg_radius(self, geom):
        """(n_segs,) per-segment radius — each segment inherits its wire's."""
        return self._radius_per_wire[self._wire_of_seg(geom)]

    def _obs_window_kwargs(self, geom, obs_rows):
        """`obs_*` overrides that restrict `_field_components` to observer
        rows `obs_rows = (i0, i1)`; `{}` when unwindowed, so the numpy
        reference path's unchunked call keeps its exact pre-#332 argument
        list. `obs_radius` has to travel with the centres: it defaults to
        the FULL (N, 1) per-observer column, which would broadcast against
        a windowed observer axis as a shape error (or, at N == chunk,
        silently against the wrong rows)."""
        if obs_rows is None:
            return {}
        win = slice(*obs_rows)
        return {
            "obs_centers": geom["seg_centers"][win],
            "obs_tangents": geom["seg_tangents"][win],
            "obs_radius": (
                None
                if self._uniform_radius is not None
                else self._seg_radius(geom)[win][:, None]
            ),
        }

    def _radius_runs(self, geom, obs_rows=None):
        """Contiguous constant-radius observer-row runs, as (start, stop,
        radius) triples — the per-run dispatch unit that serves mixed
        per-wire radii through the scalar-radius C++ field kernels
        (stevenmburns/momwire#147). Segments are wire-contiguous, so the
        number of runs is at most the number of wires.

        `obs_rows = (i0, i1)` clips the runs to one observer-row chunk of
        the #332 fill. Runs and chunks are both contiguous, so the
        intersection is again a set of runs; the kernel still sees every
        (row, source) pair exactly once, under the same scalar radius,
        so splitting a run across chunk boundaries changes no arithmetic."""
        a_seg = self._seg_radius(geom)
        bounds = np.flatnonzero(np.diff(a_seg)) + 1
        starts = np.concatenate(([0], bounds))
        stops = np.concatenate((bounds, [a_seg.shape[0]]))
        runs = [(int(s), int(e), float(a_seg[s])) for s, e in zip(starts, stops)]
        if obs_rows is None:
            return runs
        i0, i1 = obs_rows
        return [(max(s, i0), min(e, i1), a) for s, e, a in runs if s < i1 and e > i0]

    def _apply_loading(self, G, geom, seg_view, k):
        """NEC's impedance boundary condition, in place; no-op when loading
        is off. The system point-matches E_scat(n) = −E_app(n) at segment
        centres; a distributed series impedance changes the wire surface
        condition to E_scat(n) − Z'_w(n)·I(n) = −E_app(n), so subtract
        Z'·(current of basis j at segment n's centre) = Z'·σ(A+C) from
        G[n, j] over the basis-support entries. (Contrast the Galerkin
        overlap loading in `BSplineSolver._apply_loading` — same physics,
        each entering through its own testing scheme.)"""
        if not self._loading_active:
            return G
        omega = k * self.c
        # (n_segs,) Z_s(ω), zeros where switched off — the shared spec layer
        # (momwire#428); this method's share is the boundary condition below.
        z_seg = _wire_loading.loading_for(self, omega, geom).z_seg
        starts = seg_view["starts"]
        n_segs = geom["n_segs"]
        rows = np.repeat(np.arange(n_segs, dtype=np.int64), starts[1:] - starts[:-1])
        cols = seg_view["jbasis"]
        i_center = seg_view["sigma"] * seg_view["AC"]  # not A + C (#606)
        # Each (seg, basis) pair is unique in seg_view (see _basis_coefs),
        # so plain fancy-index subtraction is exact.
        G[rows, cols] -= z_seg[rows] * i_center
        return G

    def wire_loss_power(self, coeffs, omega=None):
        """Ohmic power dissipated in the wire metal, from a solve's coeffs.

        P_wire = ½ Σ_w Re[Z'_w(ω)] · Σ_{s∈w} ∫|I_s(ξ)|² dξ — the physical
        ∫R'(l)·|I(l)|² dl readout, same contract as the BSpline family's.
        On each segment the current is the single aggregated three-term
        shape I_s(ξ) = P + Q·sin(kξ) + R·cos(kξ) (α-weighted sums of the
        per-basis σA/B/σC entries), so the integral over ξ ∈ [−h/2, h/2]
        is closed-form:

            ∫|I|² = |P|²h + 2Re(P·R̄)·(2/k)sin(kh/2)
                    + |Q|²(h/2 − sin(kh)/2k) + |R|²(h/2 + sin(kh)/2k)

        (P–Q and Q–R cross terms vanish by parity). Insulation loading is
        purely reactive and contributes nothing.

        Returns (total_watts, per_wire_watts ndarray (n_wires,)).
        """
        n_w = len(self.wires_polylines)
        per_wire = np.zeros(n_w, dtype=np.float64)
        if not self._loading_active:
            return 0.0, per_wire
        if omega is None:
            omega = self.omega
        k = omega / self.c
        geom = self._build_geometry()
        seg_view = self._basis_coefs(geom, k)
        n_segs = geom["n_segs"]
        h = np.asarray(geom["seg_h"], dtype=np.float64)

        starts = seg_view["starts"]
        rows = np.repeat(np.arange(n_segs, dtype=np.int64), starts[1:] - starts[:-1])
        alpha_e = np.asarray(coeffs)[seg_view["jbasis"]]
        P = np.zeros(n_segs, dtype=np.complex128)
        Q = np.zeros(n_segs, dtype=np.complex128)
        R = np.zeros(n_segs, dtype=np.complex128)
        np.add.at(P, rows, alpha_e * seg_view["sigma"] * seg_view["A"])
        np.add.at(Q, rows, alpha_e * seg_view["B"])
        np.add.at(R, rows, alpha_e * seg_view["sigma"] * seg_view["C"])

        sin_kh = np.sin(k * h)
        sin_kh_2 = np.sin(0.5 * k * h)
        int_abs_i2 = (
            np.abs(P) ** 2 * h
            + 2.0 * np.real(P * np.conj(R)) * (2.0 / k) * sin_kh_2
            + np.abs(Q) ** 2 * (0.5 * h - sin_kh / (2.0 * k))
            + np.abs(R) ** 2 * (0.5 * h + sin_kh / (2.0 * k))
        )
        # zeros where switched off
        r_w = np.real(_wire_loading.loading_for(self, omega).z_wire)
        wire_of = self._wire_of_seg(geom)
        np.add.at(per_wire, wire_of, 0.5 * r_w[wire_of] * int_abs_i2)
        return float(per_wire.sum()), per_wire

    def _feed_segment_current(self, alpha, seg_view, feed_seg, xi=0.0):
        """Current at a point on a feed segment, `xi` metres from its centre.

        At the centre (``xi = 0``, every caller before momwire#648) this is
        I(s_local=0) = Σ_j α_j · σ_j · (A_jn + C_jn) over bases j whose
        support includes `feed_seg` — sin(k·0) = 0 drops B and cos(k·0) − 1
        drops the third shape, leaving the one coefficient. Off the centre the
        other two shapes come back and the full three-term value is evaluated.

        `xi` exists so the readout can follow a point gap that does not sit at
        a segment centre: under ``feed_model="point"`` the drive column IS this
        functional, and Y is symmetric only while the two stay the same point.
        """
        s = seg_view["starts"][feed_seg]
        e = seg_view["starts"][feed_seg + 1]
        # `AC`, not `A + C`: the sum of the two rounded coefficients has no
        # correct digits below kΔ ≈ 1e-4 (#606), and this current IS the
        # admittance the port algebra reads — the fill being well-scaled buys
        # nothing if the readout throws the digits away again.
        sig = seg_view["sigma"][s:e]
        amp = alpha[seg_view["jbasis"][s:e]]
        if xi == 0.0:
            return complex((amp * sig * seg_view["AC"][s:e]).sum())
        # cos(kξ) − 1 spelled as −2sin²(kξ/2), the same well-scaled shape set
        # the fill and `_basis_value` use (#203/#606).
        half = math.sin(0.5 * self.k * xi)
        return complex(
            (
                amp
                * (
                    sig * seg_view["AC"][s:e]
                    + seg_view["B"][s:e] * math.sin(self.k * xi)
                    - 2.0 * sig * seg_view["C"][s:e] * half * half
                )
            ).sum()
        )

    def _port_count(self):
        """Ports `compute_port_solution` returns — the configured gap feeds
        and nothing else (this family refuses `junction_ports`, #177)."""
        return len(self.feeds)

    def _feed_drive_vector(self, geom):
        """`(v, voltages)`: the RHS for the configured feed voltages, Eq 187's
        Σ_i V_i · (-1/h_i) · e_{feed_i}, and the voltages themselves.

        k-independent — it only needs segment lengths — which is why the
        sweeps hoist it out of their loops. Shared by `compute_impedance` and
        `compute_impedance_swept` so the drive convention has one home (#252).
        """
        voltages = np.array([v for _, _, v in self.feeds], dtype=np.complex128)
        v = np.zeros(geom["n_segs"], dtype=np.complex128)
        for fi, V_i in zip(geom["feed_segs"], voltages):
            v[fi] += -V_i / geom["seg_h"][fi]
        return v, voltages

    def _feed_impedances(self, alpha, geom, seg_view, voltages):
        """Per-feed driving-point impedance V_i / I_i off a solved `alpha`,
        scalar at a single feed (back-compat). The readout half of the port
        algebra `compute_impedance` and its sweep share (#252)."""
        feed_currents = np.array(
            [
                self._feed_segment_current(alpha, seg_view, fi)
                for fi in geom["feed_segs"]
            ],
            dtype=np.complex128,
        )
        z_per_feed = voltages / feed_currents
        return z_per_feed[0] if len(self.feeds) == 1 else z_per_feed

    def compute_impedance(self):
        """Return (Z_drive, alpha). With a single feed, Z_drive is a scalar
        V/I (back-compat). With N feeds, Z_drive is a length-N complex
        array of per-feed driving-point impedances V_i / I_i — the RHS is
        built as Σ_i V_i · (-1/h_i) · e_{feed_i} (linear superposition of
        Eq 187 delta-gap sources).
        """
        geom = self._build_geometry()
        self._checkpoint()  # after geometry, before the field-tensor fill
        G, seg_view = self._assemble_Z(geom, self.k)
        v, voltages = self._feed_drive_vector(geom)
        self._checkpoint()  # after assembly, before the dense LU solve
        # Factor Gᵀ in place: G.T is an F-ordered view of the C-ordered G,
        # so LAPACK's getrf overwrites it with zero copies (factoring G
        # directly would silently F-copy the whole matrix). trans=1 on the
        # factors then solves G·α = v. The factors — not the raw matrix —
        # are what gets stashed: the collocation G is not symmetric, so the
        # wire-loading adjoint oracle needs Gᵀw = r, which is the same
        # factorization at trans=0. One getrf serves both.
        lu_piv = scipy.linalg.lu_factor(G.T, overwrite_a=True)
        alpha = scipy.linalg.lu_solve(lu_piv, v, trans=1)

        Z_drive = self._feed_impedances(alpha, geom, seg_view, voltages)
        self.Z_factors = lu_piv  # LU of Gᵀ; adjoint solve = lu_solve(·, trans=0)
        return Z_drive, alpha

    def compute_y_matrix(self) -> np.ndarray:
        """Short-circuit admittance matrix [Y_sc] at the configured feeds.

        Y_sc[i, j] is the current flowing out of port i when port j is
        driven with V_j = 1 and every other port held at V_k = 0; invert
        to recover the open-circuit Z matrix for network analysis.
        Implementation mirrors `compute_impedance`: build G once, but
        solve with an N-column RHS where column j has unit excitation
        at port j's feed segment and zeros elsewhere. SinusoidalSolver
        uses Eq 187's delta-gap source which scales by `-1/h_i`; the
        Y matrix readout via `_feed_segment_current` already accounts
        for the basis arithmetic.

        This is the `y` field of `compute_port_solution()` and nothing
        else — see there for the per-port solution columns this throws
        away (#232).
        """
        return self.compute_port_solution().y

    def compute_port_solution(self) -> PortSolution:
        """Solve every port from ONE fill and ONE factorisation.

        Returns a `PortSolution` carrying `y` (identical to
        `compute_y_matrix()`), `coeffs` — column j is the segment-amplitude
        solution for a 1 V drive at port j with the others shorted — the
        per-port current readout `port_currents`, and the opaque `basis`
        handle. Ports are the configured gap feeds, in order; this solver
        rejects `junction_ports` at construction (#177: its basis enforces
        KCL identically, so a node-current port is outside its span), so
        there is nothing after the feeds.

        Drive convention is Eq 187's delta gap, which is why column j's
        right-hand side is `-1/h_j` at port j's feed segment rather than a
        bare 1: that scaling is INSIDE here, and a caller who wants the
        coefficients for an arbitrary excitation just forms `coeffs @ V`
        with V in volts. Ground models, wire loading and the extended
        thin-wire kernel all ride exactly as they do for
        `compute_y_matrix`; there are no extra preconditions.

        `basis` is stable across the ports of this one solution and NOT
        across solves — re-solve and the previous handle is stale.
        """
        geom = self._build_geometry()
        G, seg_view = self._assemble_Z(geom, self.k)
        feed_segs = geom["feed_segs"]
        n_ports = len(feed_segs)
        n_segs = geom["n_segs"]

        # Column j: drive port j with V = 1 (RHS = -1/h_j at port j's
        # segment, 0 elsewhere). Stack all N columns and let LAPACK
        # do one LU + N back-subs.
        B = np.zeros((n_segs, n_ports), dtype=np.complex128)
        for j, fi in enumerate(feed_segs):
            B[fi, j] = -1.0 / geom["seg_h"][fi]
        alphas = scipy.linalg.solve(G, B)  # (n_segs, n_ports)

        Y = np.zeros((n_ports, n_ports), dtype=np.complex128)
        for j in range(n_ports):
            for i, fi in enumerate(feed_segs):
                Y[i, j] = self._feed_segment_current(alphas[:, j], seg_view, fi)
        return PortSolution(
            y=Y,
            coeffs=alphas,
            port_currents=Y,  # the same object: the readout IS the Y matrix
            basis=_SegmentBasis(geom=geom, seg_view=seg_view, k=self.k),
        )

    def _port_solutions_swept(self, k_array):
        """Per-k `PortSolution` generator behind `compute_y_matrix_swept` and
        `compute_port_solution_swept` (#252).

        A plain loop over `compute_port_solution` — there is no batched
        assembly on this family, so the per-k core IS the single-k entry
        point and the swept Y cannot drift from the stacked single-k Y by so
        much as an ulp. The only work worth hoisting (the geometry build) is
        already cached on the instance.
        """
        with self._k_restored():
            for kk in np.asarray(k_array, dtype=float):
                self._checkpoint()  # top of each frequency iteration
                self._set_k(kk)
                yield self.compute_port_solution()

    def compute_impedance_swept(self, k_array):
        """Loop over wavenumbers. Per-call work that doesn't depend on k
        (geometry build, source-vector index, the set of bases that touch
        the feed segment) is lifted out of the loop so the per-k cost
        reduces to field-tensor + basis-coefs + assembly + solve. Together
        with the assemble_Z vectorization and the C++ field-tensor
        accelerator, this brings the n=21 sweep from ~70 ms to ~30 ms.

        Drive and readout come from `compute_impedance`'s own
        `_feed_drive_vector` / `_feed_impedances`, so the sweep carries no
        second copy of the port algebra (#252). What it does NOT do is call
        `compute_impedance` per k: that entry point factors Gᵀ in place and
        stashes the factors for the wire-loading adjoint oracle, and a sweep
        must not leave `Z_factors` pointing at a frequency the solver has
        already stepped away from.
        """
        k_array = np.asarray(k_array, dtype=float)
        n_feeds = len(self.feeds)
        if n_feeds == 1:
            z_out = np.zeros(k_array.shape[0], dtype=np.complex128)
        else:
            z_out = np.zeros((k_array.shape[0], n_feeds), dtype=np.complex128)
        geom = self._build_geometry()
        v, voltages = self._feed_drive_vector(geom)
        with self._k_restored():
            for i, kk in enumerate(k_array):
                self._checkpoint()  # top of each frequency iteration
                self._set_k(kk)
                G, seg_view = self._assemble_Z(geom, self.k)
                alpha = scipy.linalg.solve(G, v)
                z_out[i] = self._feed_impedances(alpha, geom, seg_view, voltages)
        return z_out

    def currents_at_knots(self, alpha, s_array=None):
        """Per-wire complex current sampled at every mesh knot.

        Each basis j contributes (A_jn + B_jn sin(k·s_local) +
        C_jn cos(k·s_local))·σ_jn on every segment n in its support, with
        s_local measured from segment n's centre. The current at a knot
        between adjacent segments is the average of the right-edge value
        of the segment to its left and the left-edge value of the segment
        to its right (continuity makes them equal up to round-off; the
        average is the symmetric pick). Wire-endpoint knots use only the
        adjacent segment.

        When `s_array` is provided as a list of 1D arc-length arrays (one per
        wire), evaluates the basis sum at those arc positions instead of the
        mesh knots. Arc is measured from the wire's start (s=0) to its end
        (s=Σ h_seg). Samples that fall exactly on an interior knot return the
        symmetric average of the two adjacent segments (same as the default
        knot path); samples in the interior of a segment evaluate the basis
        directly on that segment.
        """
        alpha = np.asarray(alpha)
        geom = self._build_geometry()
        seg_view = self._basis_coefs(geom, self.k)
        seg_h = geom["seg_h"]
        n_wires = len(self.wires_polylines)

        # Pattern shared between both paths: each evaluation produces a
        # (segment, s_local) pair, a destination index in the flat output,
        # and a weight (1.0 for endpoints / interior samples, 0.5 each side
        # for interior-knot symmetric-average pairs). We batch them all up,
        # call `_evaluate_basis_at_points` once, then scatter-add into the
        # output. In the natural-tangent frame I = σA + B·sin(ks) + σC·cos(ks);
        # this lives inside `_evaluate_basis_at_points`. The earlier `σ` bug
        # (fixed 2026-06: multiplying the whole bracket by σ added a spurious
        # 2·B·sin(ks) at σ=−1 junction neighbours) is gone by construction.

        if s_array is None:
            # Per wire of M segs: M+1 knots. Each segment contributes two
            # edge evaluations — its left edge to the adjacent left knot,
            # its right edge to the adjacent right knot. The first seg's
            # left-edge and the last seg's right-edge hit wire endpoints
            # (weight 1.0); every other edge hits an interior knot shared
            # with another segment's edge, so each contributes weight 0.5.
            eval_seg_parts = []
            eval_s_parts = []
            eval_target_parts = []
            eval_weight_parts = []
            wire_knot_offsets = [0]
            for w_idx in range(n_wires):
                first = geom["wire_first"][w_idx]
                last = geom["wire_last"][w_idx]
                n_w_segs = last - first + 1
                h_w = seg_h[first : last + 1]
                base = wire_knot_offsets[-1]
                seg_arr = np.arange(first, last + 1, dtype=np.int64)
                # Left edges (s = -h/2) feed knot 0..M-1
                left_target = base + np.arange(n_w_segs, dtype=np.int64)
                left_weight = np.full(n_w_segs, 0.5)
                left_weight[0] = 1.0
                # Right edges (s = +h/2) feed knot 1..M
                right_target = base + np.arange(1, n_w_segs + 1, dtype=np.int64)
                right_weight = np.full(n_w_segs, 0.5)
                right_weight[-1] = 1.0
                eval_seg_parts.append(np.concatenate([seg_arr, seg_arr]))
                eval_s_parts.append(np.concatenate([-0.5 * h_w, +0.5 * h_w]))
                eval_target_parts.append(np.concatenate([left_target, right_target]))
                eval_weight_parts.append(np.concatenate([left_weight, right_weight]))
                wire_knot_offsets.append(base + n_w_segs + 1)

            eval_seg = np.concatenate(eval_seg_parts)
            eval_s = np.concatenate(eval_s_parts)
            eval_target = np.concatenate(eval_target_parts)
            eval_weight = np.concatenate(eval_weight_parts)

            I_evals = self._evaluate_basis_at_points(seg_view, eval_seg, eval_s, alpha)
            I_flat = np.zeros(wire_knot_offsets[-1], dtype=np.complex128)
            np.add.at(I_flat, eval_target, eval_weight * I_evals)
            return [
                I_flat[wire_knot_offsets[i] : wire_knot_offsets[i + 1]]
                for i in range(n_wires)
            ]

        # s_array path: per-wire, vectorized over the sample positions.
        # Each sample is either right on an interior knot (→ two evals,
        # weight 0.5 each, for the continuity-average) or interior to a
        # segment (→ one eval, weight 1.0).
        sampled = []
        for w_idx, sv in enumerate(s_array):
            sv = np.asarray(sv, dtype=np.float64)
            first = geom["wire_first"][w_idx]
            last = geom["wire_last"][w_idx]
            n_w_segs = last - first + 1
            wire_h = seg_h[first : last + 1]
            arc_at_knot = np.concatenate([[0.0], np.cumsum(wire_h)])
            wire_arc = float(arc_at_knot[-1])
            n_s = sv.shape[0]
            if n_s == 0:
                sampled.append(np.zeros(0, dtype=np.complex128))
                continue
            s_clip = np.clip(sv, 0.0, wire_arc)
            eps = 1e-12 * max(wire_arc, 1.0)
            knot_hit = np.searchsorted(arc_at_knot, s_clip)
            on_interior_knot = (
                (knot_hit > 0)
                & (knot_hit < n_w_segs)
                & (np.abs(s_clip - arc_at_knot[np.clip(knot_hit, 0, n_w_segs)]) <= eps)
            )
            non_knot_mask = ~on_interior_knot

            # Containing-segment evaluation for non-knot samples.
            seg_in_wire = np.searchsorted(arc_at_knot, s_clip, side="right") - 1
            seg_in_wire = np.clip(seg_in_wire, 0, n_w_segs - 1)
            nk_target = np.nonzero(non_knot_mask)[0]
            nk_seg_in_wire = seg_in_wire[non_knot_mask]
            nk_seg = first + nk_seg_in_wire
            nk_s = (s_clip[non_knot_mask] - arc_at_knot[nk_seg_in_wire]) - 0.5 * wire_h[
                nk_seg_in_wire
            ]
            nk_weight = np.ones(nk_seg.shape[0])

            # Two-sided evaluation for samples on an interior knot.
            k_target = np.nonzero(on_interior_knot)[0]
            k_idx = knot_hit[on_interior_knot]
            k_left_seg = first + k_idx - 1
            k_right_seg = first + k_idx
            k_left_s = +0.5 * seg_h[k_left_seg]
            k_right_s = -0.5 * seg_h[k_right_seg]
            k_weight = np.full(2 * k_target.shape[0], 0.5)

            all_seg = np.concatenate([nk_seg, k_left_seg, k_right_seg])
            all_s = np.concatenate([nk_s, k_left_s, k_right_s])
            all_target = np.concatenate([nk_target, k_target, k_target])
            all_weight = np.concatenate([nk_weight, k_weight])

            I_evals = self._evaluate_basis_at_points(seg_view, all_seg, all_s, alpha)
            I_out = np.zeros(n_s, dtype=np.complex128)
            np.add.at(I_out, all_target, all_weight * I_evals)
            sampled.append(I_out)
        return sampled

    def _slope_eval_points(self, geom, w_idx, sv):
        """``(segment, ξ)`` per sample on wire `w_idx`, with the knot
        tie-break :meth:`current_slopes` documents.

        Arc is clipped into the wire's own range and the containing span is
        `searchsorted(..., side="right") - 1`, so a sample sitting exactly on
        an interior knot takes the span to its RIGHT and the final knot falls
        back to the span on its left. `RazorSolver.current_slopes` picks the
        same two, and at ``degree == 1`` so does scipy's `BSpline` derivative
        — which is what lets the three families' readouts be compared without
        first asking which side each one chose.
        """
        seg_h = geom["seg_h"]
        first = geom["wire_first"][w_idx]
        last = geom["wire_last"][w_idx]
        n_w_segs = last - first + 1
        wire_h = seg_h[first : last + 1]
        arc_at_knot = np.concatenate([[0.0], np.cumsum(wire_h)])
        sv = np.clip(np.asarray(sv, dtype=np.float64), 0.0, float(arc_at_knot[-1]))
        seg_in_wire = np.clip(
            np.searchsorted(arc_at_knot, sv, side="right") - 1, 0, n_w_segs - 1
        )
        xi = (sv - arc_at_knot[seg_in_wire]) - 0.5 * wire_h[seg_in_wire]
        return first + seg_in_wire, xi

    def current_slopes(self, coeffs, s_array=None):
        """Per-wire ``dI/ds`` — the solved current's arc-length derivative.

        The twin of :meth:`currents_at_knots`, with the same signature and
        the same two calling conventions as
        :meth:`~momwire.bspline.BSplineSolver.current_slopes` and
        :meth:`~momwire.razor.RazorSolver.current_slopes`: a list of 1-D
        complex arrays, one per wire in ``wires_polylines`` order, at the mesh
        knots (``s_array=None``) or at the per-wire arc positions given,
        clipped into the wire's own arc range.

        **Why it exists** (momwire#497, and momwire#611 for this family): the
        linear charge density a NEC printout reports is ``q = -(1/jω)·dI/ds``
        at each element's centre, and the EZNEC seam reads it through this
        method rather than differencing two current samples.  Until it existed
        here, every family in this branch of the roster refused that dialect
        on a missing attribute.

        Here the derivative is CLOSED FORM and exact: the basis is
        ``{1, sin kξ, cos kξ}`` per segment, so ``dI/ds`` is another
        combination of the same three coefficients rather than anything
        differenced — see :meth:`_evaluate_basis_slope_at_points` for the
        expression and for why it is better conditioned at small k·Δ than the
        value it differentiates.

        ``ξ`` is measured from a segment's centre and increases with the
        wire's own arc, so ``dI/dξ`` IS ``dI/ds`` — no chain rule and no
        re-signing.  A junction neighbour's σ = −1 is a coefficient sign
        inside the shape, not a reversal of the coordinate, and it is carried
        by the same term that carries it in the value (momwire#203).

        A sample taken exactly on a knot has to pick a side, and this picks
        the span to the RIGHT — the left span at the final knot — which is
        `RazorSolver.current_slopes`' tie-break and scipy's at ``degree ==
        1``, so a caller cannot tell the three implementations apart by their
        choice.  On THIS family the choice is unobservable anyway, and that
        is worth saying because it is not true of either twin: NEC-2's basis
        matches the current AND its derivative at every segment junction
        (Eqs 43-64), so ``dI/ds`` is CONTINUOUS across an interior knot here
        — measured at 5e-10 of the peak slope, on meshes from N = 11 to 161
        and across a two-wire junction whose segment lengths differ 3:7.  A
        razor tent expansion jumps 27-53 % at the same knots, because
        piecewise-linear current has piecewise-constant slope.  Same readout,
        same contract, genuinely different smoothness: the charge density
        this family reports is continuous where razor's is a staircase, and
        a consumer comparing two printouts is comparing that too.

        The seam asks for element CENTRES regardless, which is the honest
        thing to ask for on any of the three.
        """
        alpha = np.asarray(coeffs)
        geom = self._build_geometry()
        seg_view = self._basis_coefs(geom, self.k)
        seg_h = geom["seg_h"]
        n_wires = len(self.wires_polylines)

        if s_array is None:
            # Every knot of every wire, in the same per-wire layout
            # `currents_at_knots` returns.
            s_array = []
            for w_idx in range(n_wires):
                first = geom["wire_first"][w_idx]
                last = geom["wire_last"][w_idx]
                s_array.append(
                    np.concatenate([[0.0], np.cumsum(seg_h[first : last + 1])])
                )

        # One batched evaluation for the whole structure: the per-wire walk
        # only decides WHERE, which keeps the trig and the scatter-add single
        # calls however many wires there are.
        segs, xis, offsets = [], [], [0]
        for w_idx in range(n_wires):
            seg, xi = self._slope_eval_points(geom, w_idx, s_array[w_idx])
            segs.append(seg)
            xis.append(xi)
            offsets.append(offsets[-1] + seg.shape[0])
        if offsets[-1] == 0:
            return [np.zeros(0, dtype=np.complex128) for _ in range(n_wires)]
        flat = self._evaluate_basis_slope_at_points(
            seg_view,
            np.concatenate(segs).astype(np.int64),
            np.concatenate(xis),
            alpha,
        )
        return [
            np.asarray(flat[offsets[i] : offsets[i + 1]], dtype=np.complex128)
            for i in range(n_wires)
        ]
