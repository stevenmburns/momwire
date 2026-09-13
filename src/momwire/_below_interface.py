"""The below-interface family's plumbing, as functions of DATA (momwire#980
step C, part 1).

Everything a solver needs to decide *whether* and *how* to fill a deck with
wires below the ground plane, and nothing about *what* it fills with: the
crossing-junction scope, the node-mesh advisory's arms, the field-form
quadrature nodes, the grid extents and the named refusals. `BSplineSolver`
grew all of this for the momwire#553 serve and `RazorSolver` copied the
scope half of it for #813; the sinusoidal-Galerkin serve (#980) would have
been the third copy. It is one module instead, and each solver keeps thin
wrappers that hand in its own state.

Two rules, both gated:

* **No solver here.** Every function takes polylines, geometry columns,
  flags and scalars. `tests/test_below_interface_980.py` collects every
  attribute access in this module from the AST and asserts none is a solver
  method or field — the same gate `_crossing_fill` has carried since #801.
* **Bit identity.** The bodies are the solver methods' own, moved, not
  rewritten: the buried, crossing, transmitted and razor suites pin every
  number they pinned before the move.

What deliberately stays with the solvers: the `_wire_media` cache (the memo
is per instance and the labels come from `_medium_spec` already), the
`CrossingContext` adapter (it names the basis — polynomials for bspline and
razor, a sampler for SG), `_pair_extents_below` (an accelerated kernel with a
numpy twin, monkeypatched by its own tests through `bspline`), and the
moment-shaped fills.
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple

import numpy as np

from . import (
    _crossing_fill,
    _ground_spec,
    _medium_spec,
    _sommerfeld,
    _sommerfeld_below,
    _sommerfeld_transmitted,
)
from ._quadrature import leggauss

# Gauss order for the buried fill's THREE field-form blocks. See
# `n_qp_buried_field` for the measurement that set it (momwire#553's fifth
# inversion, measured at ε̃ = 1 against the free-space block).
N_QP_BURIED_FIELD = 6

# The named refusals a buried deck may draw. Every one names the geometry AND
# the limit in the same sentence, because the two buried Sommerfeld families
# refuse rather than clamp — there is no negligible tail to freeze below the
# interface — and a serve-time refusal with no numbers in it is not
# actionable. `bspline` re-exports them under its historical `_BURIED_*`
# names for `_couplings` and the tests.
BURIED_ENRICHMENT_REFUSAL = (
    "use_singular_enrichment=True + a wire below the ground plane is not "
    "served: the enrichment DOFs are a SECOND kernel implementation with "
    "their own quadrature over the s^(-1/2) shapes, in C++ with a `double k` "
    "and in a numpy twin, and neither carries an in-medium wavenumber or the "
    "buried image and remainder blocks (momwire#553 U5 widens the polynomial "
    "moment fill only). Solve the buried deck without singular enrichment"
)
BURIED_EXTENDED_KERNEL_REFUSAL = (
    "extended_kernel=True + a wire below the ground plane is not served: the "
    "extended kernel's eligibility is a COAXIAL-AND-EQUAL-RADIUS grouping "
    "scored across the whole geometry, and momwire#553 measured neither what "
    "that grouping means for a pair spanning two media (the tube expansion's "
    "O(a^2) term is written at one wavenumber) nor what the mirror labels "
    "mean when the image of a buried source lands in the OTHER medium. "
    "Solve the buried deck with extended_kernel=False, which is the default"
)
BURIED_DENSE_BUDGET_REFUSAL = (
    "a deck with buried wires is filled through the DENSE moment tensor on "
    "this build and this one does not fit: {n} segments need about {need:.0f} "
    "MB per tensor against a swept_mem_mb budget of {budget} MB. The chunked "
    "fill+assemble route (momwire#915) needs the windowed C++ assemblers' "
    "complex-eps twins, which this accelerator build does not export, so the "
    "fill refuses rather than silently truncating the medium. Rebuild the "
    "accelerators, raise swept_mem_mb, or mesh the deck coarser"
)
BURIED_PAST_CAP_REFUSAL = (
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
BURIED_GRAZING_REFUSAL = (
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
BURIED_CROSS_RANGE_REFUSAL = (
    "this deck's cross-medium pairs reach an observer radius of R = {r:.6g} m "
    "({wl:.3g} free-space wavelengths) about a buried source's ground "
    "projection, past the {cap:.6g} m ({capwl:g} free-space wavelengths) the "
    "transmitted family is tabulated to. The transmitted integral is the "
    "WHOLE field above a buried source, not a remainder, so there is no "
    "negligible tail to freeze and no honest clamp. Extending the log-R axis "
    "is cheap and honest work (7 nodes per doubling of range); extrapolating "
    "past it is not"
)
BURIED_DEPTH_REFUSAL = (
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
BURIED_CROSS_GRAZING_REFUSAL = (
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


# ---------------------------------------------------------------------------
# media and junction scope
# ---------------------------------------------------------------------------


def lower_medium(ground_eps, ground_model):
    """Whether this ground has a HALF-SPACE below the interface — a medium
    with a wavenumber of its own, not just a boundary condition. Exactly the
    Sommerfeld ground. `_ground_spec.ground_config` answers the same
    question in its own vocabulary (`mode == "compose"`), but it needs an ω
    to fold ε̃ at and this one is asked at geometry time."""
    return ground_eps is not None and ground_model == "sommerfeld"


def grounded_junction_ends(polylines, ground_z, groups):
    """The `(wire, "start"|"end")` pairs that participate in a junction
    whose shared point lies IN the ground plane — the crossing-junction
    exemption `_medium_spec.wire_media` keys on (momwire#524 phase 2).

    Pure geometry: no media labels yet (the labels are what this feeds).
    Which of those junctions actually SPAN the interface, and whether the
    deck is inside the crossing serve's scope, is `crossing_junctions`'
    question, asked after the labels exist. The two conditions
    (momwire#698, #700 — at least two members, at least one reaching above
    the plane) live in `_medium_spec.grounded_crossing_exemption`; the two
    trunks differ only in where `groups` comes from — declared (bspline) or
    detected (razor) — and momwire#700 is what two copies of the geometry
    cost.
    """
    if ground_z is None:
        return frozenset()
    return _medium_spec.grounded_crossing_exemption(polylines, ground_z, groups)


def grounded_junctions(polylines, ground_z, groups):
    """Indices of junctions whose shared point lies in the ground plane.

    Their KCL row (Σ signed outflows = 0) is dropped: at a grounded node
    current may flow into the ground stake (completed by the image), so
    enforcing closure among the real wires alone would be wrong physics."""
    if ground_z is None or not groups:
        return frozenset()
    grounded = set()
    for j_idx, jw in enumerate(groups):
        w, end = jw[0]
        pl = polylines[w]
        pt = pl[0] if end == "start" else pl[-1]
        if abs(pt[2] - ground_z) <= _ground_spec.ground_touch_tol(pl):
            grounded.add(j_idx)
    return frozenset(grounded)


def crossing_junctions(
    media, groups, grounded, polylines, ground_z, radii, *, two_radius=False
):
    """Indices of junctions that CROSS the interface — grounded junctions
    joining an ABOVE wire to a BELOW wire — after checking the deck against
    the crossing serve's scope (momwire#524 phase 2).

    `groups[j]` is junction `j`'s member list of `(wire, end)` pairs;
    `grounded` the set of `j` whose shared point lies in the plane; `radii`
    the per-wire radius. The scope is what the phase-2 adjudication
    validated, refused by name past it:

    * exactly ONE above member per crossing junction, N ≥ 1 below members —
      the node fan (a monopole over a buried radial screen risen to the node,
      momwire#524 fan widening). Multiple above members share the interface
      corner between above tents, a pair class no adjudicator has measured;
    * ONE wire radius across the deck — the radius rule ρ_eff = √(ρ² + a²) is
      the corner's regularization and a per-pair radius has no pinned
      convention — unless the caller passes `two_radius=True` (BSpline, U5):
      then every above wire at one radius and every below wire at another is
      served (`crossing_side_radii`), and a spread WITHIN a side is refused;
    * OTHER junctions only wholly BELOW and off the plane — the buried hub
      (one rise + N radials joined at depth, the screen's other spelling).
      Its by-parts end terms cancel through the hub's own KCL row (probe35:
      fan M+hub ≡ M to the digit), and the hub ≡ N-rises gate holds the two
      spellings of the same screen together. An above-side or in-plane other
      junction stays refused: only the below axis's completion machinery has
      that cancellation measured.

    The exemption audit (momwire#698) runs before the empty-crossing return
    because the escape it closes is exactly the empty case:
    `grounded_junction_ends` grants its exemption on GEOMETRY — a junction
    whose shared point lies in the plane — and that exemption silences
    `_medium_spec.wire_media`'s contact+buried refusal. Only a junction that
    actually CROSSES earns the silence: the crossing fill is what carries
    the contact end's current into the lower medium, and a grounded junction
    that turned out not to span the interface leaves the deck on the
    field-form transmitted block with the contact basis's O(1) boundary term
    unaccounted for. So the refusal predicate is re-asked here with the
    exemption narrowed from "touches the plane" to "validated as crossing".
    """
    if _medium_spec.BELOW not in media:
        return ()
    crossing = []
    for j_idx, jw in enumerate(groups):
        if j_idx not in grounded:
            continue
        sides = {media[w] for w, _e in jw}
        if len(sides) == 2:
            crossing.append(j_idx)

    earned = {tuple(m) for j_idx in crossing for m in groups[j_idx]}
    stranded = [
        c for c in _ground_spec.contact_ends(polylines, ground_z) if c not in earned
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
        n_above = sum(1 for w, _e in groups[j_idx] if media[w] == _medium_spec.ABOVE)
        if n_above != 1:
            raise NotImplementedError(
                "crossing junction with more than one above member: the "
                "crossing serve joins ONE above wire to N below wires "
                "at the interface (momwire#524 fan widening); the "
                "above-tent x above-tent interface corner has no "
                "measured convention"
            )
    for j_idx, jw in enumerate(groups):
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
    radii = np.asarray(radii, dtype=float)
    if float(radii.max()) - float(radii.min()) > 0.0:
        if not two_radius:
            raise NotImplementedError(
                "crossing serve with per-wire radii: the radius rule "
                "rho_eff = sqrt(rho^2 + a^2) regularizes the corner with "
                "ONE wire radius, and a mixed-radius convention is not "
                "pinned (momwire#524 phase 2)"
            )
        crossing_side_radii(media, radii)
    return tuple(crossing)


def crossing_side_radii(media, radii):
    """`(a_above, a_below)` for a TWO-RADIUS crossing deck: every above wire
    at one radius, every below wire at another (antennaknobs plan U5).

    That is the class the two-radius rule was measured on
    (`_crossing_fill.cross_complete_blocks_two_radius`): line tests at their
    observer's radius, the node's point tests at the buried radius. A spread
    WITHIN one side, a fan of buried members of different radii included, has
    no measured node radius, so it is refused by name.
    """
    radii = np.asarray(radii, dtype=float)
    out = []
    for side in (_medium_spec.ABOVE, _medium_spec.BELOW):
        r = radii[[w for w, m in enumerate(media) if m == side]]
        if float(r.max()) - float(r.min()) > 0.0:
            raise NotImplementedError(
                f"crossing serve with per-wire radii that differ within the "
                f"{side} wires: the two-radius rule serves one radius per side "
                "of the interface (the node's point tests at the buried "
                "radius, antennaknobs plan U5); a spread within one side has "
                "no measured node radius"
            )
        out.append(float(r[0]))
    return tuple(out)


def crossing_node_members(
    crossing, media, groups, polylines, n_per_edge_per_wire, stand_off_floor
):
    """A `_crossing_fill.NodeArm` per member of every crossing junction.

    Each arm is walked OUTWARD from the shared point, edge by edge, until
    the walk passes `_crossing_fill.NODE_REACH`. Two lengths come back per
    arm: the finest segment length seen inside that reach (`h_resolved`,
    what the advisory gates) and the length of the segment actually touching
    the node (`h_adjacent`). They are gated separately because the touching
    segment does not predict the error — a 50 mm feed gap in front of a
    chain graded to 6 mm is a resolved node, while a single 667 mm tent is
    not, and node-adjacent h reads those 13x apart when their errors are
    ~200x apart. `_crossing_fill`'s constants carry the measurement.

    The first edge always counts, however long it is: an arm made of one
    coarse edge must not read as "unmeasured" and escape.

    `stand_off_floor(w)` is the caller's: the height wire `w` must clear
    above the interface (jacket radius or the bare-wire floor), asked only
    for above-side arms (momwire#926).
    """
    reach = _crossing_fill.NODE_REACH
    out = []
    for j_idx in crossing:
        for w, end in groups[j_idx]:
            pl = polylines[w]
            npe = n_per_edge_per_wire[w]
            # Edge indices ordered from the node outward.
            order = range(len(npe)) if end == "start" else range(len(npe) - 1, -1, -1)
            h_resolved = h_adjacent = None
            slope = 0.0
            walked = 0.0
            for e in order:
                if walked > 0.0 and walked >= reach:
                    break
                edge = np.asarray(pl[e + 1], dtype=float) - np.asarray(
                    pl[e], dtype=float
                )
                length = float(np.linalg.norm(edge))
                h = length / int(npe[e])
                if h_adjacent is None:
                    h_adjacent = h
                    # momwire#926: the NODE-ADJACENT edge's rise per unit
                    # arclength. This is what prices the stand-off floor
                    # against node grading — the arm leaves the interface
                    # at this slope, so a vertex at arclength l sits at
                    # h = slope * l. Taken from the first edge rather than
                    # end to end because grading only ever puts vertices
                    # inside it.
                    slope = abs(float(edge[2])) / length if length > 0 else 0.0
                h_resolved = h if h_resolved is None else min(h_resolved, h)
                walked += length
            side = "above" if media[w] == _medium_spec.ABOVE else "below"
            out.append(
                _crossing_fill.NodeArm(
                    h_resolved,
                    h_adjacent,
                    w,
                    end,
                    side,
                    slope,
                    stand_off_floor(w) if side == "above" else None,
                )
            )
    return out


# ---------------------------------------------------------------------------
# the fill's quadrature, grids and scope
# ---------------------------------------------------------------------------


# The most the separation rule may multiply the base order by (momwire#1004).
# Measured factors are 4 and 10 (see `n_qp_buried_field`); 16 is the ceiling,
# not a measurement. Past it the order saturates and the collapse residual
# rises again — `test_near_plane_collapse_1004` records where that starts, so a
# deck beyond the cap gets a worse answer rather than an unaffordable fill. The
# cost is linear in q per source node on a table the grid fill dwarfs, but
# q = 6 * (h/sep) is unbounded as sep -> 0 and something has to stop it.
_MAX_NEAR_Q_FACTOR = 16


def cross_pair_separation(seg_l, seg_r, a_idx, b_idx):
    """`(separation, h)` for the closest ABOVE/BELOW segment pair.

    `separation` is the smallest centre-to-centre distance between a segment
    above the interface and one below it; `h` is the longest segment length
    among those two classes. Their ratio is what
    `n_qp_buried_field` reads.

    Centres rather than true segment-to-segment distance: the quantity being
    resolved is the pair integrand's feature width, which scales with the pair
    separation, and the centres are what the fill's own pair geometry uses.
    Cheap — O(n_a x n_b) on centres, against a fill that is O(N^2 q) with a
    Sommerfeld grid behind it.

    Returns `(None, None)` when either class is empty, which is every
    single-medium deck and is how the caller keeps today's order.
    """
    if a_idx.size == 0 or b_idx.size == 0:
        return None, None
    c = 0.5 * (np.asarray(seg_l) + np.asarray(seg_r))
    ca = c[a_idx]
    cb = c[b_idx]
    d = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=-1)
    h = float(
        max(
            np.linalg.norm(
                np.asarray(seg_r)[a_idx] - np.asarray(seg_l)[a_idx], axis=-1
            ).max(),
            np.linalg.norm(
                np.asarray(seg_r)[b_idx] - np.asarray(seg_l)[b_idx], axis=-1
            ).max(),
        )
    )
    return float(d.min()), h


def near_q_factor(separation, seg_h):
    """How many times the base Gauss order the cross block needs (momwire#1004).

    `1` unless the closest above/below pair is nearer than a segment — which is
    every single-medium deck and every deck in the suite today — and past that
    `ceil(seg_h / separation)`, capped at `_MAX_NEAR_Q_FACTOR`. `separation`
    and `seg_h` come from `cross_pair_separation`; either one `None` or
    non-positive means "no cross pair to resolve" and keeps the base order.

    The rule, its two measured ratios and what it cannot go below are all in
    `n_qp_buried_field`'s docstring. It is exposed as a FACTOR because the two
    trunks hold different halves of the order: the sinusoidal-Galerkin plan has
    `n_qp_sommerfeld` and wants the whole order, while
    `compute_Z_operator_buried` never sees that knob and only needs to raise an
    order the solver already chose. One rule, two spellings of the same answer,
    and neither trunk owns a second copy of the arithmetic.

    The comparison is TOLERANCED because a deck sitting exactly a segment apart
    is not unusual — it is what a uniform mesh either side of the plane
    produces — and a bare `ceil` turns h/sep = 1+1e-16 into factor 2, doubling
    that deck's order for a rounding.
    """
    if separation is None or seg_h is None or separation <= 0.0 or seg_h <= 0.0:
        return 1
    ratio = seg_h / separation
    if ratio <= 1.0 + 1e-9:
        return 1  # a segment or more apart: today's decks, unchanged
    return min(int(math.ceil(ratio - 1e-9)), _MAX_NEAR_Q_FACTOR)


def n_qp_buried_field(n_qp_sommerfeld, *, separation=None, seg_h=None):
    """Gauss order for the buried fill's THREE field-form blocks.

    **This is momwire#553's fifth inversion, and it is a tolerance inherited
    from "the remainder is small".** `n_qp_sommerfeld` is 3 because the ±=+
    remainder is a smooth correction over a segment — no near zone, nothing
    to resolve. Two of the three blocks here are remainders and 3 would
    serve them. The CROSS block is not a remainder: the transmitted integral
    is the WHOLE field, near zone and all, so the quantity a cross pair
    integrates over a segment falls like 1/R³ and 3-point Gauss
    under-resolves it exactly where an above wire and a buried wire come
    close — which on a buried radial screen is every pair that matters.

    Measured at ε̃ = 1, where the whole cross block must reproduce the
    free-space mixed-potential block over the same pairs and the two
    disagree only by quadrature and interpolation (worst entry, relative to
    the block's largest):

        deck                       q=3      q=4      q=6      q=8
        10 m mono / 5 m radial,
        3 + 2 segments           6.6e-3   3.1e-3   6.8e-4   6.8e-4
        15 + 10 segments         3.9e-4   1.1e-5   1.0e-5   1.0e-5

    — the floor in each row is the GRID's own interpolation error, which the
    quadrature cannot go below, and q = 6 reaches it on both. Cost is q² per
    pair on a table the grid fill already dwarfs, so the order is set at the
    measurement rather than at the cheapest passing rung.

    **SUB-SEGMENT SEPARATION NEEDS MORE (momwire#1004).** The table above was
    taken on a deck whose above/below pairs are never closer than a segment.
    When they ARE closer the same 1/R^3 argument bites again, and 6 is short:
    on two 0.6 m wires with 15 segments each (h = 0.04 m) sitting 5 mm either
    side of the plane, the cross block misses the free-space block it must
    reproduce at eps-tilde = 1 by 5.5e-02, thirteen times its own floor.

    The order therefore scales with `seg_h / separation` when that exceeds 1,
    and the rule is MEASURED at two ratios rather than fitted to one — it
    predicts the convergence point at both:

        h/separation    rule q    residual at q=6   at the rule's q   converged
        4  (5 mm gap)      24          5.489e-02        4.118e-03      yes, 24
        10 (2 mm gap)      60          1.492e+00        1.616e-02      yes, 60

    Raising q past the rule's value buys nothing (q = 48 and 96 at ratio 4 read
    4.120e-03 and 4.119e-03), and the observer rule is already converged —
    `n_qp_test` 8 / 32 / 128 at q = 48 reads 4.120e-03 / 4.118e-03 / 4.119e-03.
    What remains at the rule's q is the GRID's own near-plane interpolation
    floor, which this knob cannot go below and which grows as the plane is
    approached: 5.4e-06 / 3.8e-05 / 5.6e-04 / 4.1e-03 at gaps of 0.5 / 0.1 /
    0.02 / 0.005 m.

    All of those are on the corrected metric — `|G_mixed + G_free| / |G_free|`
    over the cross block. The mixed cross block is stored NEGATED relative to
    the free-space one, so a naive `|G_mixed - G_free|` reads a constant 2.0
    and no knob appears to move anything.

    `separation` / `seg_h` come from `cross_pair_separation`; omitting them
    keeps the pre-#1004 order exactly, which is what every single-medium deck
    gets.

    `n_qp_sommerfeld` still raises it if a caller asked for more: the knob
    keeps meaning "at least this". Since momwire#692 the CROSSING fill's axes
    no longer route through this knob — its density ladder banked its own
    `_NEAR_Q`/`_FAR_Q` in `_crossing_fill`. The q = 6 measurement above
    stays authoritative for the three grid field-form blocks.
    """
    return max(int(n_qp_sommerfeld), N_QP_BURIED_FIELD) * near_q_factor(
        separation, seg_h
    )


def field_nodes(seg_l, seg_r, tangents, h, q):
    """`(points, tangents, u_phys, w_node)` for the field-form quadrature
    over a set of segments — `_Z_sommerfeld_remainder`'s own node rule at
    order `q`. The nodes are strictly interior to each segment, which is
    what keeps a wire ending IN the plane off its own singularity and what
    bounds every grid extent the plan sizes.

    Basis-agnostic: `u_phys[i, q]` is each node's arc from its segment's
    start and `w_node[i, q]` its weight. A polynomial caller folds them into
    its moment weights `w·u^p`; a sampler caller multiplies its samples by
    `w_node` directly.
    """
    xg, wg = leggauss(q)
    tq = 0.5 * (xg + 1.0)
    nodes = seg_l[:, None, :] + tq[None, :, None] * (seg_r - seg_l)[:, None, :]
    u_phys = h[:, None] * tq[None, :]
    w_node = 0.5 * h[:, None] * wg[None, :]
    n = seg_l.shape[0]
    return nodes.reshape(n * q, 3), np.repeat(tangents, q, axis=0), u_phys, w_node


def refuse_out_of_scope(
    *,
    use_singular_enrichment,
    extended_kernel,
    n,
    degree,
    dense_fits,
    chunked_serves,
    swept_mem_mb,
):
    """The three solver configurations a buried deck may not reach.

    Each one is a SECOND kernel that has no medium, not a tolerance:
    refusing by name is the same discipline U1 applied one level down to the
    fills it did not widen.
    """
    if use_singular_enrichment:
        raise NotImplementedError(BURIED_ENRICHMENT_REFUSAL)
    if extended_kernel:
        raise NotImplementedError(BURIED_EXTENDED_KERNEL_REFUSAL)
    if not dense_fits and not chunked_serves:
        raise NotImplementedError(
            BURIED_DENSE_BUDGET_REFUSAL.format(
                n=n,
                need=(degree + 1) ** 2 * n * n * 16 / (1 << 20),
                budget=swept_mem_mb,
            )
        )


def somm_grid(eps_t, k, r1_max, omega, mu, cancel_flag):
    """The above/above Sommerfeld remainder grid, sized to `r1_max`."""
    return _sommerfeld.get_grid(
        eps_t,
        k,
        r1_max,
        omega=omega,
        mu=mu,
        cancel_flag=cancel_flag,
    )


def serve_plan(
    ground_z, seg_l, seg_r, a_idx, obs_a, obs_b, k_p, k_m, *, crossing, pair_extents
):
    """Grid extents for the three field-form blocks, or a named refusal.

    Every extent is measured on the QUADRATURE NODES the fill will actually
    query (`obs_a`, `obs_b`), not on segment endpoints. That is not an
    optimisation: a wire standing in the plane has an endpoint AT the
    interface, whose transmitted-grid radius about a buried source's ground
    projection is zero, and the nodes are strictly interior, so the node set
    is both the honest domain and a smaller one.

    Nothing here clamps. Both buried families' `eval` already refuse rather
    than freeze an amplitude — their remainder is not a negligible tail and
    the transmitted surface is the whole field — so a geometry past a cap
    has no answer to give, and the refusal is raised HERE, before an
    80-second grid fill, with the deck's own numbers and the limit in the
    same sentence.

    `crossing=True` skips the cross-medium section entirely: a crossing
    deck's cross pair is `_crossing_fill`'s designed DIRECT evaluation — no
    transmitted grid is ever built — so its θ-floor cost law must not refuse
    the deck (a node-graded crossing mesh routinely puts quadrature nodes
    fractions of a millimetre below the plane, which is exactly the grazing
    geometry the grid can't pay for and the designed evaluator doesn't care
    about).

    `pair_extents(x, y, d_b) -> (r1_max, th_min)` is the caller's — bspline's
    `_pair_extents_below`, an accelerated kernel with a numpy twin whose own
    tests monkeypatch it through `bspline`.

    **`obs_a` / `obs_b` must be the union of EVERY rule the caller will query
    with, not just its field rule** (momwire#980 D2). bspline never meets
    this: both axes of a transmitted pair use the same buried field rule, so
    its plan and its fill query the same points. The Galerkin trunk tests on
    its own quadrature (`n_qp_test`, 8) while sourcing on the field rule (6),
    and the higher rule's outermost node sits CLOSER to a segment end —
    0.150945 m against 0.151608 m on a 21-segment radial — so a plan sized on
    the field nodes alone builds a z' ladder the fill then queries outside of,
    and the grid refuses. Pass both rules' points.
    """
    gz = ground_z
    lam_p = 2.0 * np.pi / k_p
    lam_m = 2.0 * np.pi / abs(k_m)
    plan = {}

    # --- below/below: R1 = |two depths added|, theta = atan2(h, rho) ---
    d_b = gz - obs_b[:, 2]
    r1_max, th_min = pair_extents(obs_b[:, 0], obs_b[:, 1], d_b)
    cap = _sommerfeld_below._SOMM_BELOW_R1_CAP_LAMBDA_M * lam_m
    if r1_max > cap:
        raise ValueError(
            BURIED_PAST_CAP_REFUSAL.format(
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
            BURIED_GRAZING_REFUSAL.format(
                th=math.degrees(th_min),
                floor=_sommerfeld_below._SOMM_BELOW_TH_MIN_DEG,
                depth=float(np.min(d_b)) + float(np.min(d_b)),
            )
        )
    plan["r1_below"] = r1_max

    if a_idx.size == 0:
        return plan

    # --- above/above: the shipped sizing, over the above segments -----
    plan["r1_above"] = _sommerfeld.max_image_distance(seg_l[a_idx], seg_r[a_idx], gz)

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
            BURIED_CROSS_RANGE_REFUSAL.format(
                r=r_hi,
                wl=r_hi / lam_p,
                cap=r_cap,
                capwl=_sommerfeld_transmitted._R_CAP_LAMBDA_P,
            )
        )
    zp_cap = _sommerfeld_transmitted._ZPRIME_MAX_LAMBDA_M * lam_m
    if zp_hi > zp_cap:
        raise ValueError(
            BURIED_DEPTH_REFUSAL.format(
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
            BURIED_CROSS_GRAZING_REFUSAL.format(
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


class BuriedFills(NamedTuple):
    """The moment-shaped fills, handed in rather than imported.

    Everything here is basis- or solver-shaped: the assemblers own the
    quadrature and the moment tensors, `crossing_context` names the basis
    (polynomials for bspline and razor, a sampler for SG), and `nodes` and
    `wire_media` read per-instance caches. `compute_Z_operator_buried` owns
    only the ROUTING between the three pair classes, which is the part every
    formulation shares -- momwire#980 step C part 2.
    """

    checkpoint: Callable
    nodes: Callable
    wire_media: Callable
    crossing_context: Callable
    assemble_Z: Callable
    build_J_blocks_subset: Callable
    accumulate_Z_subset_chunked: Callable
    image_Z_weighted: Callable
    image_tangent_dot: Callable
    field_galerkin_block: Callable
    apply_loading: Callable


def compute_Z_operator_buried(
    geom,
    supp_seg,
    polys,
    *,
    f,
    below_segments,
    buried_medium,
    refuse_out_of_scope_fn,
    crossing_junctions_fn,
    serve_plan_fn,
    somm_grid_fn,
    crossing_node_members_fn,
    ground_z,
    eps,
    omega,
    mu,
    cancel_flag,
    chunked,
):
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
      the reciprocity gate. It is also the only block on this fill whose
      quadrature order depends on the GEOMETRY: `near_q_factor` raises it
      when the two media come closer than a segment, which is exactly where
      the calibrated order under-resolves the 1/R^3 integrand (momwire#1004).

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

        Moved verbatim from `BSplineSolver._compute_Z_operator_buried` (momwire
        #980 step C part 2). The bodies are the solver method's own -- the
        buried, crossing, transmitted and razor suites pin every number they
        pinned before the move -- with `self.X` replaced by the data or the
        callable it stands for and nothing else changed.
    """
    below = below_segments(geom)
    b_idx = np.nonzero(below)[0]
    a_idx = np.nonzero(~below)[0]
    eps_t, eps_m, k_p, k_m, c2, a_m = buried_medium()
    gz = ground_z

    refuse_out_of_scope_fn(geom)
    obs_a, t_a, W_a = f.nodes(geom, a_idx)
    obs_b, t_b, W_b = f.nodes(geom, b_idx)
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
    crossing_j = crossing_junctions_fn()
    crossing_plan = bool(a_idx.size and crossing_j)

    # --- the cross block's own Gauss order (momwire#1004) ----------------
    # PER PAIR CLASS, not per fill: the raised order is for the transmitted
    # blocks, whose two media can come closer than a segment. The two
    # same-medium remainders above have no such pair — the closest above/above
    # distance is a mesh spacing, not a clearance — so raising them would
    # multiply the most expensive blocks in the fill by up to 16 for nothing.
    # A crossing deck skips the transmitted grid entirely (`_crossing_fill`'s
    # designed direct evaluation), so it skips this too.
    #
    # The nodes carry the order; nothing downstream is told it. Both
    # `f.field_galerkin_block` calls below read it back off the arrays' own
    # shapes, which is what makes "the order the nodes were built at" and "the
    # order the table is reshaped by" one fact instead of two that can part.
    q_factor = 1
    if a_idx.size and not crossing_plan:
        q_factor = near_q_factor(
            *cross_pair_separation(geom["seg_l"], geom["seg_r"], a_idx, b_idx)
        )
    if q_factor > 1:
        obs_ax, t_ax, W_ax = f.nodes(geom, a_idx, q_factor=q_factor)
        obs_bx, t_bx, W_bx = f.nodes(geom, b_idx, q_factor=q_factor)
        # `serve_plan` sizes every extent on "the union of EVERY rule the
        # caller will query with" — its own D2 contract, and this is the
        # second trunk to have two rules. A ladder built on the base nodes
        # alone is one the raised-order fill then queries outside of, and the
        # grid refuses rather than clamps.
        plan_a = np.vstack([obs_a, obs_ax])
        plan_b = np.vstack([obs_b, obs_bx])
    else:
        obs_ax, t_ax, W_ax = obs_a, t_a, W_a
        obs_bx, t_bx, W_bx = obs_b, t_b, W_b
        plan_a, plan_b = obs_a, obs_b

    plan = serve_plan_fn(
        geom,
        a_idx,
        plan_a,
        plan_b,
        k_p,
        k_m,
        crossing=crossing_plan,
    )

    # --- the two direct blocks and the two image blocks, each in its
    #     own medium. Chunked whenever the windowed assemblers' complex-
    #     eps~ twins are built (momwire#915): the same four terms with the
    #     same signs, never a (d+1, d+1, N, N) tensor, and faster even
    #     where the tensor fits (12 radials: 3.5 -> 2.7 s) because the
    #     zero-padded scatter of `_build_J_blocks_subset` is gone. The
    #     dense route below is the REFERENCE every buried gate was pinned
    #     on and the chunked route is gated against it at 1e-12.
    if not chunked:
        f.checkpoint()
        Z = f.assemble_Z(
            f.build_J_blocks_subset(geom, k_m, b_idx),
            supp_seg,
            polys,
            geom,
            eps=eps_m,
        )
        td_img = f.image_tangent_dot(geom["tangents"])
        if a_idx.size:
            f.checkpoint()
            Z += f.assemble_Z(
                f.build_J_blocks_subset(geom, k_p, a_idx), supp_seg, polys, geom
            )
            f.checkpoint()
            Z -= f.image_Z_weighted(
                f.build_J_blocks_subset(geom, k_p, a_idx, mirror_sources=True),
                supp_seg,
                polys,
                c2 * td_img.astype(np.complex128),
                np.full(td_img.shape, c2, dtype=np.complex128),
            )
        f.checkpoint()
        Z -= f.image_Z_weighted(
            f.build_J_blocks_subset(geom, k_m, b_idx, mirror_sources=True),
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
        f.accumulate_Z_subset_chunked(
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
            f.accumulate_Z_subset_chunked(
                Z,
                geom,
                k_p,
                a_idx,
                supp_seg,
                polys,
                mirror_sources=False,
                eps=eps,
                scale=1.0,
            )
            f.accumulate_Z_subset_chunked(
                Z,
                geom,
                k_p,
                a_idx,
                supp_seg,
                polys,
                mirror_sources=True,
                eps=eps,
                scale=-1.0,
                weight=complex(c2),
            )
        f.accumulate_Z_subset_chunked(
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
        grid_above = somm_grid_fn(eps_t, plan["r1_above"])

        def proj_aa(o, to, s, ts):
            return _sommerfeld.remainder_field_proj(
                o, to, s, ts, gz, k_p, grid_above, cancel_flag=cancel_flag
            )

        Z -= f.field_galerkin_block(
            supp_seg, polys, proj_aa, a_idx, a_idx, obs_a, t_a, W_a, obs_a, t_a, W_a
        )

    grid_below = _sommerfeld_below.get_grid_below(
        eps_t, k_p, plan["r1_below"], omega, mu=mu
    )

    def proj_bb(o, to, s, ts):
        return _sommerfeld_below.remainder_field_proj_below(
            o, to, s, ts, gz, k_p, k_m, grid_below
        )

    Z -= f.field_galerkin_block(
        supp_seg, polys, proj_bb, b_idx, b_idx, obs_b, t_b, W_b, obs_b, t_b, W_b
    )

    crossing = crossing_j if a_idx.size else ()
    if crossing:
        # The node-mesh advisory (momwire#696) goes first, because
        # this is the one place per fill where the crossing serve
        # actually engages: the plan site above asks the same question
        # before the deck is committed to the crossing path.
        _crossing_fill.warn_coarse_node(
            crossing_node_members_fn(crossing, f.wire_media())
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
        f.checkpoint()
        ctx = f.crossing_context(geom, supp_seg, polys)
        ax_a = _crossing_fill.axis_data(ctx, a_idx)
        ax_b = _crossing_fill.axis_data(ctx, b_idx)
        if ctx.a_below is not None:
            # A TWO-RADIUS node (antennaknobs plan U5): the two cross blocks
            # are no longer transposes — above rows test lines at the above
            # radius, below rows at the buried one, and the node's point tests
            # share one radius — and continuity is closed by the crossing
            # junction's KCL row rather than left to emerge.
            t_above, t_below = _crossing_fill.cross_complete_blocks_two_radius(
                ctx, a_idx, b_idx, ax_a, ax_b
            )
            Z -= t_above
            Z -= t_below.T
            Z += _crossing_fill.self_completions_two_radius(ctx, ax_b, ax_a)
        else:
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
            omega,
            mu=mu,
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

        Z -= f.field_galerkin_block(
            supp_seg,
            polys,
            proj_ab,
            a_idx,
            b_idx,
            obs_ax,
            t_ax,
            W_ax,
            obs_bx,
            t_bx,
            W_bx,
        )
        Z -= f.field_galerkin_block(
            supp_seg,
            polys,
            proj_ba,
            b_idx,
            a_idx,
            obs_bx,
            t_bx,
            W_bx,
            obs_ax,
            t_ax,
            W_ax,
        )

    return f.apply_loading(Z)
