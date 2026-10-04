"""Constructor-side normalisation of the wire spec, shared by every solver.

The formulations disagree about basis, testing rule, kernel and ground —
they do not disagree about what `wire_radius=[…]` means. The audit
momwire#429 measured 464 of 749 constructor code lines sitting in verbatim
clones across the four roots; this module is where those land as they are
extracted, starting with the radius (rank 3, with momwire#425) and the
junction spec (rank 8).

Nothing here knows a formulation. A function that would have to branch on
one belongs in the solver.
"""

import numpy as np

from ._junction_rule import coincident_end_groups


def normalize_wire_radius(value, n_wires, *, per_wire_refusal=None):
    """`wire_radius` → ``((n_wires,) float array, uniform value or None)``.

    A scalar applies to every wire; a length-n_wires sequence gives each
    wire (polyline) its own conductor radius (stevenmburns/momwire#147).
    Every entry must be positive and finite — a radius of zero is not a
    thin wire, it is the singular kernel the reduced form exists to avoid.

    The second return value is the SCALAR FAST PATH: the common radius when
    every wire shares one (including a uniform array), else None. It is
    what keeps the historical scalar code paths — and the single-`a` C++
    kernels — bit-identical whenever a model is uniform, however the caller
    spelled it.

    `per_wire_refusal` is for a formulation whose kernel takes exactly one
    `a`: pass the refusal prose and a non-scalar value raises
    `NotImplementedError` with it instead of being accepted. That is one
    spelling of the refusal rather than one per solver, and dropping the
    argument is the whole of what "this formulation gained per-wire radii"
    means at the constructor.
    """
    if per_wire_refusal is not None and not np.isscalar(value):
        raise NotImplementedError(per_wire_refusal)
    radius = np.asarray(value, dtype=float)
    if radius.ndim == 0:
        radius = np.full(n_wires, float(radius))
    elif radius.shape != (n_wires,):
        raise ValueError(
            f"wire_radius: expected a scalar or a length-{n_wires} sequence "
            f"(one entry per wire), got shape {radius.shape}"
        )
    if not np.all(np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError(
            f"wire_radius entries must be positive and finite, got {radius}"
        )
    uniform = float(radius[0]) if np.all(radius == radius[0]) else None
    return radius, uniform


# The validation floor: the deck front fuses span endpoints onto its
# `deck/_polylines._NODE_EPS` = 1e-6 m grid, so two ends it calls one node
# can differ by up to ~1.8e-6 m Euclidean. 1e-5 m accepts everything that
# grid can produce with a 5x margin, whatever the mesh scale.
_JUNCTION_COINCIDENCE_FLOOR = 1e-5


def seg_radius(radius_per_wire, segs_per_wire):
    """`(n_segs,)` per-segment radius: each segment inherits its wire's
    (stevenmburns/momwire#147). `segs_per_wire` is each wire's segment count
    in segment order — the solver reads it off its own geometry, which is the
    only part that differs between them. The kernel helpers collapse a
    uniform array back to the scalar fast path, so passing this everywhere
    keeps scalar-radius solves bit-identical."""
    return np.repeat(radius_per_wire, segs_per_wire)


def refuse_duplicated_wires(wires_polylines):
    """Refuse a deck that lists the same wire twice (momwire#1042).

    Two wires whose polylines coincide vertex for vertex, in either
    direction, put the same conductor in the model twice. Every basis a
    family hangs on one has an identical twin on the other, sign-flipped
    when the copy runs backwards, so the operator has two equal columns and
    is singular at any mesh, in any medium and under any quadrature —
    NEC-5's column-order fill reports `Singular matrix` on
    qantenna/airplane.nec (GW 116 and GW 117), while row-order fills pivot on
    rounding residue and answer differently build to build. It is a
    modelling slip, so it is refused at construction, naming both wires.

    The test is exact rather than tolerant, as razor's bundle check is: the
    vertices rounded to nanometres. A duplicate is authored, not arrived at
    by drift, and a tolerant test would start refusing merely CLOSE
    conductors, which are served. It keys on WHOLE wires: a bundle that
    shares a segment or two with other wires (a radial screen's coincident
    rises, momwire#524) is a different geometry, served by BSplineSolver
    and refused segment by segment on razor.
    """
    seen = {}
    for i, pl in enumerate(wires_polylines):
        pl = np.asarray(pl, dtype=float)
        if pl.ndim != 2:
            continue  # malformed; the caller's own shape check names it
        verts = tuple(tuple(v) for v in np.round(pl, 9))
        key = min(verts, verts[::-1])
        if key in seen:
            first, first_verts = seen[key]
            raise ValueError(
                f"wires {first} and {i} are the same wire listed twice: "
                "their vertices coincide"
                + (" in reverse order" if verts != first_verts else "")
                + ", so the model holds one conductor twice and its matrix "
                "is singular at any mesh. Delete one of them - momwire#1042"
            )
        seen[key] = (i, verts)


def check_junction_coincidence(wires_polylines, n_per_edge_per_wire, junctions):
    """Refuse junction groups whose member wire-ends do not coincide.

    momwire#522, the #518 postmortem's guardrail: an explicit ``junctions=``
    spec with a wrong wire index welds ends that sit nowhere near each other
    (KCL between non-coincident points), and the member whose entry was lost
    is silently zeroed instead — both produce a well-posed WRONG model that
    converges cleanly, which is why the mistake must refuse at construction
    rather than surface as physics.

    ``junctions`` is the solver's already-normalized list of
    ``[(wire_idx, "start"|"end"), ...]`` groups. Each group's first member is
    the reference; every other member must lie within tolerance of it. The
    tolerance is scale-aware — 1e-3 of the shortest terminal segment among
    the group's members — floored at ``_JUNCTION_COINCIDENCE_FLOOR`` so the
    deck front's node-grid quantization can never fire it. Raises
    ``ValueError`` naming the group, both ends, the distance and the
    tolerance; returns None on success.
    """
    for j, group in enumerate(junctions):
        anchors = []
        seg_lens = []
        for w, end in group:
            pl = np.asarray(wires_polylines[w], dtype=float)
            npe = n_per_edge_per_wire[w]
            if end == "start":
                anchor, edge, count = pl[0], pl[1] - pl[0], npe[0]
            else:
                anchor, edge, count = pl[-1], pl[-1] - pl[-2], npe[-1]
            anchors.append(anchor)
            seg_lens.append(float(np.linalg.norm(edge)) / max(int(count), 1))
        tol = max(_JUNCTION_COINCIDENCE_FLOOR, 1e-3 * min(seg_lens))
        w0, end0 = group[0]
        for (w, end), anchor in zip(group[1:], anchors[1:]):
            dist = float(np.linalg.norm(anchor - anchors[0]))
            if dist > tol:
                raise ValueError(
                    f"junction {j}: members do not coincide - wire {w} {end} at "
                    f"{tuple(anchor)} is {dist:.6g} m from wire {w0} {end0} at "
                    f"{tuple(anchors[0])} (tolerance {tol:.3g} m, 1e-3 of the "
                    f"shortest terminal segment; a junction group must name "
                    f"ONE point - momwire#522)"
                )


def normalize_junctions(junctions, wires_polylines, n_per_edge_per_wire):
    """The `junctions=` spec → a validated list of `(wire, "start"|"end")`.

    momwire#429 rank 8, and the audit's own note on why it is ranked where
    it is: *"prerequisite for razor node gaps"*.  A node-gap port names a
    MEMBER of a junction group, so a family that wants one first has to
    agree with every other family about what a group is and which members
    are in it.  Two constructors carried this and 16 of their 18 lines were
    identical; the two that were not are a comment and one error string
    that said ``got 0`` where the other did not.

    ``None`` means INFER from the geometry (momwire#590 step 3): coincident
    wire ends ARE a junction unless the caller says otherwise, because that
    is what the geometry means, what NEC does, and what ``RazorSolver`` and
    ``HarringtonSolver`` already did — solving the same wires apart silently
    is a wrong answer rather than a coarse one.  An EMPTY list is the
    escape, and a different statement: these wires really are disconnected.

    Wire-to-wire connectivity only.  A lone end resting in the ground plane
    is NOT inferred into a one-member grounded junction, because ground
    contact is a separate question (momwire#151) with its own tolerance and
    inferring it here would change grounded decks that read correctly today.
    Razor DOES infer it, and that asymmetry survives step 3 deliberately —
    which is also why a caller must never hand a detected-junction list back
    to a family that infers its own (``momwire.deck._solver.port_kwargs``).

    A one-member group is legal (momwire#172).  As a plain junction its KCL
    row pins ``I_end = 0``, which is numerically a free end; as a junction
    PORT it is the natural form of a lone-conductor-end attachment.
    """
    n_w = len(wires_polylines)
    if junctions is None:
        junctions = coincident_end_groups(wires_polylines)
    out = []
    for j, group in enumerate(junctions):
        if len(group) < 1:
            raise ValueError(f"junction {j}: need >= 1 wire-end")
        members = []
        for wire, end in group:
            if not (0 <= wire < n_w):
                raise ValueError(
                    f"junction {j}: wire_idx {wire} out of range [0, {n_w})"
                )
            if end not in ("start", "end"):
                raise ValueError(
                    f"junction {j}: end must be 'start' or 'end', got {end!r}"
                )
            members.append((int(wire), end))
        out.append(members)
    check_junction_coincidence(wires_polylines, n_per_edge_per_wire, out)
    return out


def normalize_node_gaps(node_gaps, junctions, n_wires, *, junction_ports=()):
    """The `node_gaps=` spec → a validated `[(wire, "start"|"end", volts)]`.

    A series EMF at a node, addressed by ONE member of the junction group it
    sits at: the gap is between that member and every other member of the
    group, which is the shape NEC's ``EX`` card writes and the only shape it
    can write (momwire#315 proposes a two-set partition for the fan-dipole
    feed that ``EX`` cannot express; this is the degenerate case of it).

    Every rule below is formulation-independent — it is about the SPEC and
    the topology, not about any basis — which is why it is here rather than
    in a solver (momwire#603 U4, on momwire#429 rank 8's normalised
    ``junctions``).  What a family cannot share is the port's COLUMN: the
    named member's current expressed in that family's own junction unknowns.
    Give that a method on the solver and everything else is this function.

    ``junction_ports`` is the junction indices already carrying a shunt port
    (momwire#172); a family with no such port passes nothing.

    How many gaps one junction takes depends on how many members it has
    (momwire#1300).  At K = 2 a gap named through either member is the SAME
    cut — one through-current, the two EMFs in series — so a second gap there
    is not a second port and is refused; a caller folds the two into one
    (the EZNEC seam's ``_assign_columns`` does exactly that).  At K >= 3 a gap
    named through member m sits in m's own branch, and gaps named through
    DIFFERENT members are different cuts with different currents: licensed
    NEC-5 serves an object on each named wire at one such node, each in its
    own wire's branch.  So K >= 3 takes one gap per member, up to all K of
    them.  With all K gapped the K port currents still sum to zero (KCL at
    the node), so the K-port admittance has rank K - 1 — adding one EMF to
    every branch only shifts the node's potential, which is physics rather
    than a defect for a caller to guard.
    """
    out = []
    if node_gaps is None:
        return out
    end_to_junction = {}
    for j, group in enumerate(junctions):
        for member in group:
            end_to_junction[member] = j
    seen_members, seen_junctions = set(), set()
    ported = set(junction_ports)
    for i, entry in enumerate(node_gaps):
        if len(entry) != 3:
            raise ValueError(
                f"node_gaps[{i}]: expected (wire_index, 'start'|'end',"
                f" voltage), got {entry!r}"
            )
        wire, end, volts = entry
        if not (0 <= wire < n_wires):
            raise ValueError(
                f"node_gaps[{i}]: wire_index {wire} out of range [0, {n_wires})"
            )
        if end not in ("start", "end"):
            raise ValueError(
                f"node_gaps[{i}]: end must be 'start' or 'end', got {end!r}"
            )
        member = (int(wire), end)
        j_idx = end_to_junction.get(member)
        if j_idx is None:
            raise ValueError(
                f"node_gaps[{i}]: wire {wire} {end!r} is not a member "
                "of any junction group — a series node gap lives at a "
                "junction; for a feed inside a wire use feeds="
            )
        if len(junctions[j_idx]) < 2:
            raise ValueError(
                f"node_gaps[{i}]: junction {j_idx} has a single member "
                "— there is no through-current path to be in series "
                "with (a lone-end attachment is junction_ports=)"
            )
        if member in seen_members:
            raise ValueError(f"node_gaps[{i}]: wire {wire} {end!r} listed twice")
        if j_idx in seen_junctions and len(junctions[j_idx]) == 2:
            raise ValueError(
                f"node_gaps[{i}]: junction {j_idx} already carries a "
                "node gap and joins only two wire ends, so a gap through "
                "the other member is the far side of the same cut — one "
                "series gap per two-wire junction (sum the EMFs into one)"
            )
        if j_idx in ported:
            raise ValueError(
                f"node_gaps[{i}]: junction {j_idx} is also a junction "
                "port — the shunt (#172) and series (#305) ports of "
                "one node cannot be driven together yet"
            )
        seen_members.add(member)
        seen_junctions.add(j_idx)
        out.append((int(wire), end, complex(volts)))
    return out
