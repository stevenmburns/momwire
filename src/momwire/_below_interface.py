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


def crossing_junctions(media, groups, grounded, polylines, ground_z, radii):
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
      convention;
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
        raise NotImplementedError(
            "crossing serve with per-wire radii: the radius rule "
            "rho_eff = sqrt(rho^2 + a^2) regularizes the corner with "
            "ONE wire radius, and a mixed-radius convention is not "
            "pinned (momwire#524 phase 2)"
        )
    return tuple(crossing)


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


def n_qp_buried_field(n_qp_sommerfeld):
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

    `n_qp_sommerfeld` still raises it if a caller asked for more: the knob
    keeps meaning "at least this". Since momwire#692 the CROSSING fill's axes
    no longer route through this knob — its density ladder banked its own
    `_NEAR_Q`/`_FAR_Q` in `_crossing_fill`. The q = 6 measurement above
    stays authoritative for the three grid field-form blocks.
    """
    return max(int(n_qp_sommerfeld), N_QP_BURIED_FIELD)


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
