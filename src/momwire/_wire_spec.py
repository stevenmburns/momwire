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

import functools
import warnings
from typing import NamedTuple

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


# --------------------------------------------------------------------------
# A wire listed twice (momwire#1042)
# --------------------------------------------------------------------------


class DuplicateWire(UserWarning):
    """A deck lists the same wire twice; it was solved ONCE (momwire#1042).

    Advisory: the answer is the antenna with the conductor present once,
    which is what NEC-2 returns for such a deck to within its own rounding
    (qantenna/airplane.nec: 68.5-91.06j as written, 68.457-91.179j with GW
    117 deleted). Two copies cannot be solved as two: every basis on one has
    an identical twin on the other, so the operator is singular and the
    split of the current between them is undefined (NEC-5 reports `Singular
    matrix`; row-order fills answer from rounding residue)."""


class WireDedup(NamedTuple):
    """How a constructor's INPUT wires map onto the wires it solves.

    ``kept[i]`` is the input index of solved wire ``i``. ``removed`` holds
    one ``(input index, twin input index, reversed)`` per dropped copy.
    """

    n_input: int
    kept: tuple
    removed: tuple

    def solver_index(self, w):
        """The solved index of input wire `w` (its twin's, for a copy)."""
        for r, twin, _rev in self.removed:
            if w == r:
                w = twin
                break
        return self.kept.index(w)

    def to_input(self, per_solved, fill):
        """A per-solved-wire list re-expanded to input order; each dropped
        copy gets ``fill(copy index, twin_value, reversed)``."""
        out = [None] * self.n_input
        for i, w in enumerate(self.kept):
            out[w] = per_solved[i]
        for r, twin, rev in self.removed:
            out[r] = fill(r, out[twin], rev)
        return out

    def to_solved(self, per_input):
        """A per-input-wire list restricted to the solved wires."""
        return [per_input[w] for w in self.kept]


def _vertex_key(pl):
    return tuple(tuple(v) for v in np.round(np.asarray(pl, dtype=float), 9))


def find_duplicated_wires(wires_polylines):
    """``[(input index, twin input index, reversed)]`` for every wire whose
    polyline repeats an earlier one's, vertex for vertex, in either
    direction.

    EQUAL, not close: every vertex within 1e-12 of the deck's coordinate
    scale of its twin's, i.e. the same numbers up to the rounding of a
    transform that produced them (GM copies). A duplicate is authored, not
    arrived at by drift, and a looser test would start merging merely CLOSE
    conductors, which are served — or which another rule refuses: wires
    0.6 nm apart sit in Harrington's near-coincident window and must reach
    that refusal (`test_a_chained_tolerance_is_refused...`). Candidates are
    bucketed on nanometre-rounded vertices and then compared at that
    tolerance; a pair straddling a rounding boundary is not merged, which is
    the old behaviour. It keys on WHOLE wires, so a bundle that shares a
    segment or two with other wires (a radial screen's coincident rises,
    momwire#524) is untouched. Coinciding is necessary, not sufficient:
    `drop_duplicated_wires` merges a copy only when it is also joined like
    its twin (momwire#1333).
    """
    pls = [np.asarray(pl, dtype=float) for pl in wires_polylines]
    good = [pl for pl in pls if pl.ndim == 2 and pl.size]
    if not good:
        return []
    tol = 1e-12 * max(1.0, max(float(np.max(np.abs(pl))) for pl in good))
    seen, out = {}, []
    for i, pl in enumerate(pls):
        if pl.ndim != 2:
            continue  # malformed; the caller's own shape check names it
        verts = _vertex_key(pl)
        key = min(verts, verts[::-1])
        if key in seen:
            first = seen[key]
            twin = pls[first]
            if twin.shape == pl.shape and np.max(np.abs(twin - pl)) <= tol:
                out.append((i, first, False))
                continue
            if twin.shape == pl.shape and np.max(np.abs(twin - pl[::-1])) <= tol:
                out.append((i, first, True))
                continue
        else:
            seen[key] = i
    return out


def _same(a, b):
    if a is None or b is None:
        return a is b
    try:
        if np.isnan(a) and np.isnan(b):
            return True
    except TypeError:
        pass
    return a == b


def _same_counts(a, b, rev):
    """Whether two raw `n_per_edge_per_wire` entries mesh one polyline alike
    (`b` read backwards when its copy runs the other way)."""
    if a is None or b is None:
        return a is b
    la = [int(x) for x in np.atleast_1d(a)]
    lb = [int(x) for x in np.atleast_1d(b)]
    return la == (lb[::-1] if rev else lb)


def _per_wire_entries(value, n):
    """`value` as a per-wire list, or None when it is not one (None, a
    scalar, or one spec meaning every wire)."""
    if value is None or np.ndim(value) == 0:
        return None
    if not isinstance(value, (list, tuple, np.ndarray)):
        return None
    entries = list(value)
    return entries if len(entries) == n else None


def _flip(end):
    return "end" if end == "start" else "start"


def _merge_connected_alike(wires, dups, junctions):
    """Which coincident copies are DUPLICATES (momwire#1333).

    ``dups`` is `find_duplicated_wires`'s list, every copy paired with the
    first wire on its path. A copy merges into the first earlier wire on
    that path that is joined exactly as it is: at each end, the same
    junction as that wire's coincident end, or no junction at both. Returns
    ``({copy: (kept twin, reversed)}, [(copy, nearest unmerged twin)])``;
    the second list holds the coincident wires left in place because their
    connectivity differs.

    Keyed on the junction, not on the set of other wires in it: two
    coincident ends in one junction reach the same wires by construction,
    and two in different junctions do not, even when each junction holds
    nothing but other coincident copies.
    """
    if junctions is not None:
        groups = [[(int(w), end) for w, end in g] for g in junctions]
    else:
        groups = coincident_end_groups(wires)
    node = {}
    for j, group in enumerate(groups):
        for member in group:
            node[member] = j

    def joined(w, rev):
        # Ends named in the class's FIRST wire's direction.
        return tuple(node.get((w, _flip(e) if rev else e)) for e in ("start", "end"))

    classes = {}
    for r, first, rev in dups:
        classes.setdefault(first, [(first, False)]).append((r, rev))
    removed, kept_coincident = {}, []
    for members in classes.values():
        reps = []
        for w, rev in members:
            key = joined(w, rev)
            match = next(((p, prev) for p, prev, pkey in reps if pkey == key), None)
            if match is None:
                if reps:
                    kept_coincident.append((w, reps[-1][0]))
                reps.append((w, rev, key))
            else:
                p, prev = match
                removed[w] = (p, rev != prev)
    return removed, kept_coincident


def drop_duplicated_wires(
    family,
    wires,
    n_per_edge_per_wire,
    *,
    per_wire=None,
    sites=None,
    junctions=None,
    junction_refs=(),
    refuse_reason=None,
    coincident_refusal=None,
):
    """Solve a wire listed twice ONCE (momwire#1042), or refuse by name.

    The duplicate is DROPPED before anything is built and every
    wire-indexed argument is remapped onto the shorter list, so the solve
    never sees it; the solver re-expands its per-wire READOUTS to the
    caller's numbering (`WireDedup.to_input`), reporting the dropped copy as
    carrying no current — the kept twin carries the conductor's whole
    current, since the split between two coincident copies is exactly the
    undetermined part.

    A copy is a duplicate only when it is also CONNECTED like its twin
    (momwire#1333): each of its ends sits in the same junction as the twin's
    coincident end, or both ends are free. A coincident wire joined to
    different wires than its twin (rise k to radial k alone, under explicit
    `junctions`) is a structure the caller wrote, not a repeat: it is left in
    place and solved as written, as before #1042, and a family that cannot
    solve coincident wires refuses it by name (`coincident_refusal`, or the
    family's own refusal later). Inferred junctions group every coincident
    end, so without `junctions` every copy is connected like its twin.

    Merged only when the two copies are the same conductor in every respect
    the physics reads. Refused, naming both, when they differ in segment
    count or in any `per_wire` argument (radius, conductivity, insulation,
    distributed RLC), when a feed, load or node gap in `sites` sits on the
    copy, when dropping it would empty a junction that a junction-indexed
    argument (`junction_refs`) may address, or when the family says so
    (`refuse_reason`).

    `per_wire` maps argument name to value; `sites` maps argument name to a
    list of tuples whose first entry is a wire index. Returns
    ``(wires, n_per_edge_per_wire, per_wire, sites, junctions, dedup)`` with
    ``dedup`` None and everything returned unchanged when there is no copy.
    """
    per_wire = dict(per_wire or {})
    sites = dict(sites or {})
    dups = find_duplicated_wires(wires)
    if not dups:
        return wires, n_per_edge_per_wire, per_wire, sites, junctions, None
    n = len(wires)

    def refuse(r, twin, why):
        raise ValueError(
            f"{family}: wires {twin} and {r} are the same wire listed twice, "
            f"and {why}, so they cannot be merged into one conductor. Delete "
            f"wire {r}, or make the two copies identical - momwire#1042"
        )

    removed, kept_coincident = _merge_connected_alike(wires, dups, junctions)
    if kept_coincident and coincident_refusal:
        r, twin = kept_coincident[0]
        raise ValueError(
            f"{family}: wires {twin} and {r} coincide, vertex for vertex, but "
            f"are joined to different wires, so they are two conductors "
            f"written on one path rather than a wire listed twice, and "
            f"{coincident_refusal} - momwire#1333"
        )
    if not removed:
        return wires, n_per_edge_per_wire, per_wire, sites, junctions, None
    for r, (twin, rev) in removed.items():
        if refuse_reason:
            refuse(r, twin, refuse_reason)
        if n_per_edge_per_wire is not None and not _same_counts(
            n_per_edge_per_wire[twin], n_per_edge_per_wire[r], rev
        ):
            refuse(r, twin, "their segment counts differ")
        for name, value in per_wire.items():
            entries = _per_wire_entries(value, n)
            if entries is not None and not _same(entries[r], entries[twin]):
                refuse(r, twin, f"their {name} entries differ")
        for name, entries in sites.items():
            for entry in entries or ():
                # On the twin as much as on the copy: a gap or load on one
                # of two coincident conductors is bridged by the other.
                if int(entry[0]) in (r, twin):
                    refuse(r, twin, f"{name} places a site on wire {int(entry[0])}")

    # Explicit junctions: a merged copy's end stands beside its twin's
    # coincident end (`_merge_connected_alike` merges nothing else), so
    # dropping the member changes no connectivity.
    if junctions is not None:
        new_junctions = []
        for j, group in enumerate(junctions):
            members = [(int(w), end) for w, end in group]
            kept_members = [(w, end) for w, end in members if w not in removed]
            if not kept_members:
                continue
            if (
                len(kept_members) < len(members)
                and len(kept_members) < 2
                and junction_refs
            ):
                w = next(w for w, _e in members if w in removed)
                refuse(
                    w,
                    removed[w][0],
                    f"junction {j} would be left with "
                    "one member while junction-indexed ports are declared",
                )
            new_junctions.append(kept_members)
        if len(new_junctions) != len(junctions) and junction_refs:
            w = next(iter(removed))
            refuse(
                w,
                removed[w][0],
                "dropping it would renumber the junctions "
                "that the junction-indexed ports address",
            )
        junctions = new_junctions
    elif junction_refs:
        # Inferred junctions: a group of only the copy and its twin vanishes
        # from the reduced geometry, renumbering every later group.
        reduced = [wires[w] for w in range(n) if w not in removed]
        if len(coincident_end_groups(reduced)) != len(coincident_end_groups(wires)):
            w = next(iter(removed))
            refuse(
                w,
                removed[w][0],
                "dropping it would renumber the junctions "
                "that the junction-indexed ports address",
            )

    kept = tuple(i for i in range(n) if i not in removed)
    new_index = {w: i for i, w in enumerate(kept)}
    dedup = WireDedup(
        n, kept, tuple((r, t, v) for r, (t, v) in sorted(removed.items()))
    )

    wires = [wires[w] for w in kept]
    if n_per_edge_per_wire is not None:
        n_per_edge_per_wire = [n_per_edge_per_wire[w] for w in kept]
    for name, value in per_wire.items():
        entries = _per_wire_entries(value, n)
        if entries is not None:
            per_wire[name] = [entries[w] for w in kept]
    for name, entries in sites.items():
        if entries:
            sites[name] = [(new_index[int(e[0])],) + tuple(e[1:]) for e in entries]
    if junctions is not None:
        junctions = [[(new_index[w], end) for w, end in g] for g in junctions]

    # ONE advisory per construction, however many copies: a GM-built wire
    # grid can repeat hundreds (cebik 6-6-nec4.nec: 1372).
    pairs = [f"{twin} and {r}" for r, twin, _rev in dedup.removed]
    named = "; ".join(pairs[:6]) + (
        f"; and {len(pairs) - 6} more pairs" if len(pairs) > 6 else ""
    )
    warnings.warn(
        f"{family}: {len(pairs)} wire(s) listed twice (their vertices "
        f"coincide), each solved ONCE as the first of its pair: wires "
        f"{named}. Two coincident copies have no answer of their own - the "
        f"split of current between them is undetermined - and the conductor "
        f"present once is what NEC-2 returns for such a deck. Each later "
        f"copy's current reads as 0; the first carries it all. Delete the "
        f"copies to silence this - momwire#1042",
        DuplicateWire,
        stacklevel=3,
    )
    return wires, n_per_edge_per_wire, per_wire, sites, junctions, dedup


def reads_input_wires(kind):
    """Decorate a per-wire READOUT so it speaks the caller's wire numbering
    when a duplicate was dropped (momwire#1042); a no-op otherwise.

    ``kind`` is how the readout is shaped:

    * ``"per_wire"`` — ``(coeffs, s_array=None) -> [per-wire array]``
      (`currents_at_knots`, `current_slopes`). `s_array` arrives in input
      numbering and is restricted to the solved wires; the result is
      re-expanded with ZEROS at each dropped copy, sized like its own
      `s_array` entry, else like its twin's.
    * ``"loss"`` — ``-> (total, per_wire ndarray)`` (`wire_loss_power`); the
      copy dissipates 0 W.
    * ``"placements"`` — ``-> [FeedPlacement]``; each `wire` is mapped back.

    Internal callers that index the result by SOLVED wire pass
    ``_solved=True``.
    """

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, _solved=False, **kwargs):
            dedup = getattr(self, "_wire_dedup", None)
            if dedup is None or _solved:
                return fn(self, *args, **kwargs)
            if kind == "placements":
                return [p._replace(wire=dedup.kept[p.wire]) for p in fn(self)]
            if kind == "loss":
                total, per_wire = fn(self, *args, **kwargs)
                full = dedup.to_input(list(per_wire), lambda r, v, rev: 0.0)
                return total, np.asarray(full, dtype=np.asarray(per_wire).dtype)
            args = list(args)
            s_in = kwargs.pop("s_array", args.pop(1) if len(args) > 1 else None)
            s_solved = None if s_in is None else dedup.to_solved(list(s_in))
            out = fn(self, *args, s_array=s_solved, **kwargs)

            def zeros(r, twin_value, _rev):
                n = len(twin_value) if s_in is None else len(s_in[r])
                return np.zeros(n, dtype=np.asarray(twin_value).dtype)

            return dedup.to_input(list(out), zeros)

        wrapper.reads_input_wires = kind
        return wrapper

    return decorate


# --------------------------------------------------------------------------
# The mesh floor at a gap (momwire#959) — a STOPGAP advisory
# --------------------------------------------------------------------------


class GapMeshFloor(UserWarning):
    """Segments shorter than the wire's radius at a gap (momwire#959).

    A STOPGAP. The thin-wire model this tree solves (a reduced or extended
    kernel with a delta-gap source) is unreliable where the segments at a gap
    are shorter than the wire radius: the answer drifts toward 0 ohm as they
    shrink instead of converging. The fix is an exact kernel with a
    finite-width feed (momwire#1330); until then this says so. Advisory only:
    nothing is remeshed or refused.

    Measured on a free-space 600 MHz dipole (L = 0.24 m, a = 3.175 mm, a 2 a
    mesh elsewhere), relative to the same deck with a 2 a gap segment:

    * The floor is the mesh AT A GAP. A run of 0.2 a segments ten radii long
      away from every gap moves Z by <= 1e-4, and so does one at a two-wire
      junction node.
    * FLOORS, from a uniform ladder against the delta/a ~ 2 mesh: the reduced
      kernel is 16 % off at delta/a = 0.85 (`GAP_FLOOR_REDUCED`), the
      extended kernel 9 % off at 0.6 (`GAP_FLOOR_EXTENDED`); both collapse
      below (reduced ~0 ohm by 0.35, extended by 0.3).
    * The SPAN in radii is not the variable; the number of sub-floor
      segments around the gap is, and it depends on the gap model
      (`GAP_DEPTH`): three 0.1 a segments (a 0.3 a span) collapse bspline's
      point gap, two 0.6 a ones (1.2 a) move it 2.7 %.

      - A point gap (bspline, Sin-Galerkin, either feed model) survives a
        short gap segment alone (13 % at 0.1 a, bounded as it shrinks) and a
        knot between two short ones (12 %), and fails once a sub-floor
        segment lies BEYOND those on both sides: 0.91 (0.3 a), 0.27 (0.6 a)
        with three; 1.0 / 0.49 / 0.095 with four at a knot.
      - Razor's knot gap fails with the two segments that meet at it:
        1.0 (0.1 a), 0.87 (0.3 a), 0.23 (0.6 a).
      - A point-matched segment gap fails with its own segment: 0.83
        (0.1 a), 0.61 (0.3 a), 0.40 (0.6 a). Pulse and Harrington rows are
        segment gaps too, and are treated so (not separately measured).
    * LOADS fail like feeds: a 50 ohm port with five 0.2 a segments around it
      vanishes from the answer (28 % on every family, razor's lumped loads
      included), so loads, node gaps and ports are gaps here.
    * JUNCTIONS: a fine span at a two-wire node is harmless (<= 2e-5), and
      at a T node on the reduced kernel (<= 9e-3); under the EXTENDED kernel
      the sinusoidal families fail at a T (point-matched 0.52-0.69,
      Sin-Galerkin 0.27-0.31) while bspline and razor do not (<= 3e-4). So a
      junction of three or more members counts as a gap there only.
    """


GAP_FLOOR_REDUCED = 0.85
GAP_FLOOR_EXTENDED = 0.6
# Sub-floor segments needed BEYOND the gap's own on each side, per gap model.
GAP_DEPTH = {"point": 1, "knot": 0, "segment": 0}


def _segment_lengths(polyline, npe):
    pl = np.asarray(polyline, dtype=float)
    lengths = np.linalg.norm(np.diff(pl, axis=0), axis=1)
    return np.repeat(lengths / np.asarray(npe, dtype=float), np.asarray(npe))


def gap_past_floor(polyline, npe, a, where, floor, depth):
    """``(fires, delta/a, span/a)`` for a gap at `where` on one wire: an
    arclength from the first vertex (None: the midpoint), or "start"/"end"
    for a gap at a wire end.

    The gap's own segments are the one containing it, or both meeting at it
    when it sits on a knot (to the snaps' rounding). It fires when all of
    them are shorter than ``floor * a`` and the contiguous sub-floor run
    reaches `depth` more segments beyond them on each side the wire has.
    """
    h = _segment_lengths(polyline, npe)
    knots = np.concatenate([[0.0], np.cumsum(h)])
    if where == "start":
        own = [0]
    elif where == "end":
        own = [h.size - 1]
    else:
        t = knots[-1] / 2.0 if where is None else float(where)
        k = int(np.clip(np.searchsorted(knots, t, side="right") - 1, 0, h.size - 1))
        own = [k]
        tol = 1e-11 * knots[-1]
        if k > 0 and abs(t - knots[k]) <= tol:
            own = [k - 1, k]
        elif k + 1 < h.size and abs(t - knots[k + 1]) <= tol:
            own = [k, k + 1]
    short = h < floor * a
    ratio = float(min(h[i] for i in own) / a)
    if not all(short[i] for i in own):
        return False, ratio, 0.0
    i, j = min(own), max(own) + 1
    while i > 0 and short[i - 1]:
        i -= 1
    while j < h.size and short[j]:
        j += 1
    span = float((knots[j] - knots[i]) / a)
    left = min(own) - i if min(own) > 0 else None  # None: the wire ends here
    right = j - max(own) - 1 if max(own) < h.size - 1 else None
    sides = [n for n in (left, right) if n is not None]
    return all(n >= depth for n in sides), ratio, span


def advise_gap_mesh_floor(
    family,
    wires_polylines,
    n_per_edge_per_wire,
    radius_per_wire,
    extended_kernel,
    gaps,
    *,
    gap_model,
):
    """Warn `GapMeshFloor` once, naming every gap past the floor
    (`gap_past_floor` with this family's `GAP_DEPTH[gap_model]`).

    `gaps` is ``[(label, [(wire, where), ...])]``: a feed, load, port or
    junction and the wire position(s) it sits at. Deduped per gap; the worst
    member speaks for it.
    """
    floor = GAP_FLOOR_EXTENDED if extended_kernel else GAP_FLOOR_REDUCED
    depth = GAP_DEPTH[gap_model]
    found = []
    for label, sites in gaps:
        worst = None
        for w, where in sites:
            a = float(radius_per_wire[w])
            if a <= 0.0:
                continue
            fires, ratio, span = gap_past_floor(
                wires_polylines[w],
                n_per_edge_per_wire[w],
                a,
                where,
                floor,
                # A node's members each see it as a wire end, with one side
                # to reach along. Not separately measured: one sub-floor
                # segment beyond the end one, as a point gap needs.
                depth if where not in ("start", "end") else max(depth, 1),
            )
            if fires and (worst is None or ratio < worst[0]):
                worst = (ratio, span)
        if worst is not None:
            found.append((label, *worst))
    if not found:
        return
    parts = []
    for label, ratio, span in found[:6]:
        part = (
            f"{label}, where the segments are down to {ratio:.3g} of the "
            f"radius over {span:.3g} radii"
        )
        if not extended_kernel and ratio >= GAP_FLOOR_EXTENDED:
            part += (
                f" (extended_kernel=True holds to delta/a ~ "
                f"{GAP_FLOOR_EXTENDED:g}, which covers this)"
            )
        parts.append(part)
    more = f"; and {len(found) - 6} more" if len(found) > 6 else ""
    warnings.warn(
        f"{family}: the impedance is unreliable at "
        + "; ".join(parts)
        + more
        + ". This solver's thin-wire model with a delta-gap source does not "
        "converge where the segments at a gap are shorter than the wire "
        "radius: the answer drifts toward 0 ohm as they shrink. The fix is "
        "an exact kernel with a finite-width feed, momwire#1330. Advisory: "
        "nothing is remeshed. See stevenmburns/momwire#959.",
        GapMeshFloor,
        stacklevel=3,
    )


def solver_gaps(solver, *, junctions_are_gaps=False):
    """Every gap a constructed solver carries, as `advise_gap_mesh_floor`
    takes them: feeds, lumped loads, node gaps, junction ports and node
    ports — and, when `junctions_are_gaps`, every junction of three or more
    members — each labelled with its wire and position."""
    gaps = []
    for i, (w, arc, _v) in enumerate(getattr(solver, "feeds", None) or []):
        pos = "its midpoint" if arc is None else f"{float(arc):.4g} m"
        gaps.append((f"source {i} (wire {w} at {pos})", [(int(w), arc)]))
    for i, (w, arc, _z) in enumerate(getattr(solver, "lumped_loads", None) or []):
        pos = "its midpoint" if arc is None else f"{float(arc):.4g} m"
        gaps.append((f"load {i} (wire {w} at {pos})", [(int(w), arc)]))
    for i, (w, end, _v) in enumerate(getattr(solver, "node_gaps", None) or []):
        gaps.append((f"node gap {i} (wire {w} {end})", [(int(w), end)]))
    junctions = getattr(solver, "junctions", None) or []
    ported = set()
    for kind, ports in (
        ("junction port", getattr(solver, "junction_ports", None) or []),
        ("node port", getattr(solver, "node_ports", None) or []),
    ):
        for i, entry in enumerate(ports):
            j = int(entry[0])
            if 0 <= j < len(junctions):
                ported.add(j)
                members = [(int(w), end) for w, end in junctions[j]]
                gaps.append((f"{kind} {i} (junction {j})", members))
    if junctions_are_gaps:
        for j, group in enumerate(junctions):
            if j not in ported and len(group) >= 3:
                members = [(int(w), end) for w, end in group]
                gaps.append((f"junction {j} ({len(group)} wires)", members))
    return gaps
