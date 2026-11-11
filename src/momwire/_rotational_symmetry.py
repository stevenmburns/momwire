"""The rotational-symmetry rule for the sector (block-circulant) solve route,
and its refusals (momwire#1029 phase 1).

A radial screen is N copies of one sector about a vertical axis, so its Z is
block-circulant in the sector index and one sector's rows carry the whole
operator (`scratch/1029-block-circulant/PLAN.md` §1 registers the rule and
measures the premise). This module decides whether a deck IS that shape, and
says why not when it is not. It never fills, solves or touches a basis: every
condition here is a geometry, material or drive fact frozen at construction,
which is what lets the route refuse before a solve rather than after one.

WHAT THE GROUPING MEANS, because the refusals only read right once it is
fixed. The wires split in two:

* the **axial group** — wires the rotation leaves alone. Seeded by every wire
  whose vertices share one vertical line. When every such wire shares ONE
  line, it is the axis.
* the **sectors** — everything else, in orbits of size N under rotation by
  2*pi/N about the axis.

A deck whose vertical wires stand on several lines (momwire#1897: a whip on
the axis inside a cage of off-axis verticals) is read the same way with ONE
of those lines as the axis: its wires are the axial group, and every other
vertical wire is an ordinary sector wire that must find its orbit like any
radial. The lines are tried nearest the deck's vertex centroid first, which
is where any rotation axis of a finite structure has to pass. "Exactly one
axis" then means what it always meant: the axis-invariant wires share one
vertical line, and everything else lies in an orbit of size N.

That reading is tried FIRST and, when what is left of the deck is a screen,
it is the answer. Only when it is not does the axial group GROW across
junctions with exactly two members — a degree-2 junction is one conductor
continuing rather than a place sectors attach — and the grown group is then
held to condition 1. The growth exists to make condition 1 measurable: a
tilted mast is still the mast, so it joins the group and fails the group's
own rule, instead of reading as a spare sector and failing somebody else's.
Ordering it as a fallback is what keeps a one-radial deck — whose hub is
degree 2 — refused for having fewer than two sectors rather than for a
horizontal "axial" wire.

The sector LABELLING is free, and the route depends on that. Any transversal
of the orbits is a fundamental domain: Lambda_0 sums a row over a source
dof's whole orbit, so which member is called "sector 0" cannot change it.
What is not free is the POSITION of a dof inside its sector, which is what
pairs a row with its N images.

Order of the checks is `PLAN-phase1.md` §3's order — axis, sectors,
discretisation and material, ports, loads, ground — and the refusal names the
FIRST condition that fails.
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from typing import NamedTuple

import numpy as np
from ._vec3 import norm3
from . import _lapack

# 1e-9 of the deck's largest extent. The registered tolerance (§3.1): the
# structure probe measures rotated radials landing 3.9e-16 m (4 radials) to
# 6.4e-15 m (48) apart on 6.33 m arms, four to seven orders inside it.
TOL_REL = 1e-9

_WAY_OUT = "Drop rotational_symmetry=True to solve this deck densely."

# momwire#1132: the route fills a ROW-COMPACT Z (only the rows the harmonic-0
# block reads). False restores the square `rows=` destination; it exists so the
# gates can hold the two to the bit, not as a user knob.
COMPACT_Z = True

# The grounds that are invariant under rotation about a vertical axis. Any
# ground a solver can name outside this set breaks the route, and a solver
# family that grows a terrain or two-media ground says so by returning a name
# that is not here — see `BSplineSolver._rotational_ground_kind`.
AXISYMMETRIC_GROUNDS = ("free", "pec", "refl-coef", "sommerfeld")


class RotationalSymmetryRefused(NotImplementedError):
    """Raised by `rotational_symmetry=True` when a deck is not N copies of one
    sector about a vertical axis. The message names the first failing
    condition of `PLAN-phase1.md` §3 and the way out."""


class SectorMap(NamedTuple):
    """The grouping the route fills and solves through.

    `sectors[s]` is the wire indices of the s-th image, `sectors[s][c]` and
    `sectors[t][c]` being the same wire of orbit `c` — so position `c` is the
    pairing and `s` is only a label. `axial` is the rotation's fixed wires.
    """

    axis: tuple[float, float]
    n_sectors: int
    sectors: tuple[tuple[int, ...], ...]
    axial: tuple[int, ...]
    tol: float
    extent: float
    # `reversed[s][c]`: sector s's wire of orbit c runs opposite to sector
    # 0's (momwire#1897). () when none does, which is every deck the route
    # served before the reversed reading existed.
    reversed: tuple[tuple[bool, ...], ...] = ()


def _refuse(sentence):
    return RotationalSymmetryRefused(f"rotational symmetry: {sentence} {_WAY_OUT}")


def _rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rho(pl, axis):
    return np.hypot(pl[:, 0] - axis[0], pl[:, 1] - axis[1])


def _arc_length(pl):
    return float(np.sum(norm3(np.diff(pl, axis=0))))


def _worst_z_angle(pl):
    """(degrees off z, of the worst edge). 0 for a purely vertical wire."""
    d = np.diff(pl, axis=0)
    n = norm3(d)
    cos = np.abs(d[:, 2]) / np.maximum(n, 1e-300)
    return float(np.max(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))))


def _junction_of_end(junctions, n_wires):
    """{(wire, end): junction index} and the per-junction member count."""
    of_end = {}
    for j, members in enumerate(junctions):
        for w, end in members:
            of_end[(int(w), str(end))] = j
    sizes = [len(m) for m in junctions]
    return of_end, sizes


def _junction_point(polylines, junctions, j):
    w, end = junctions[j][0]
    pl = polylines[int(w)]
    return pl[0] if end == "start" else pl[-1]


def _vertical_seeds(polylines, tol):
    """`[(wire, xy centre)]` for every wire whose vertices share one vertical
    line. Raises condition 1's first out-of-scope form: no such wire at all."""
    seeds = []
    for w, pl in enumerate(polylines):
        xy = pl[:, :2]
        centre = xy.mean(axis=0)
        if float(np.max(np.linalg.norm(xy - centre, axis=1))) <= tol:
            seeds.append((w, centre))
    if not seeds:
        raise _refuse(
            "no wire lies on a single vertical line, so the deck has no axial "
            "group and no axis. The route serves a radial screen about a "
            "vertical axis."
        )
    return seeds


def _one_line(seeds, tol):
    """`(axis, seed wires)` when every seed stands on ONE vertical line, else
    None — the reading every deck before momwire#1897 had to satisfy, kept
    exactly: the axis is the mean of the seed centres, and one centre more
    than `tol` from it means more than one line."""
    axis = np.mean([c for _w, c in seeds], axis=0)
    for _w, centre in seeds:
        if float(np.linalg.norm(centre - axis)) > tol:
            return None
    return (float(axis[0]), float(axis[1])), tuple(sorted(w for w, _c in seeds))


def _lines(polylines, seeds, tol):
    """The seeds grouped by vertical line, as `[(axis, wires)]`, nearest the
    deck's vertex centroid first (momwire#1897).

    A rotation that maps a finite structure onto itself fixes the centroid of
    its vertices, so the axis of a C_N deck is the line through it; trying
    that line first makes the right reading the first one tried, and the
    others only cost a failed partition each. Grouping is greedy, in wire
    order, against each line's first centre."""
    groups = []
    for w, centre in seeds:
        for g in groups:
            if float(np.linalg.norm(centre - g[0][1])) <= tol:
                g.append((w, centre))
                break
        else:
            groups.append([(w, centre)])
    centroid = np.mean(np.concatenate([p[:, :2] for p in polylines]), axis=0)
    lines = []
    for g in groups:
        axis = np.mean([c for _w, c in g], axis=0)
        lines.append(
            (
                float(np.linalg.norm(axis - centroid)),
                (float(axis[0]), float(axis[1])),
                tuple(sorted(w for w, _c in g)),
            )
        )
    lines.sort(key=lambda t: (t[0], t[2]))
    return [(axis, wires) for _d, axis, wires in lines]


def _grown(polylines, junctions, seeds):
    """The seeds plus every wire reachable across a junction with exactly TWO
    members: a degree-2 junction is one conductor continuing, not a place
    sectors attach. A branch point of degree 3 or more STOPS the walk, which
    is what keeps the hub's radials out of the axial group."""
    of_end, sizes = _junction_of_end(junctions, len(polylines))
    by_junction = {}
    for (w, end), j in of_end.items():
        by_junction.setdefault(j, []).append((w, end))
    axial = set(seeds)
    frontier = list(axial)
    while frontier:
        w = frontier.pop()
        for end in ("start", "end"):
            j = of_end.get((w, end))
            if j is None or sizes[j] != 2:
                continue
            for u, _e in by_junction[j]:
                if u not in axial:
                    axial.add(u)
                    frontier.append(u)
    return tuple(sorted(axial))


def _condition_1(polylines, axial, axis, tol):
    """Every wire of the axial group runs along the axis (§3.1)."""
    for w in axial:
        pl = polylines[w]
        if float(np.max(_rho(pl, axis))) > tol:
            raise _refuse(
                f"the axial group's wires are not parallel to z (wire {w} runs "
                f"{_worst_z_angle(pl):.3f} deg off). The symmetry axis must be "
                f"normal to the interface."
            )


def _first_vertex_index(polylines, wires):
    """A k-d tree over the first vertex of each of `wires` (positions in the
    list, not wire numbers). Imported here: only a route deck pays for
    scipy.spatial."""
    from scipy.spatial import cKDTree

    return cKDTree(np.array([polylines[u][0] for u in wires], dtype=float))


def _image_of(polylines, wires, index, a, tol, taken):
    """The first wire of `wires`, in list order, that is not `taken` and lies
    within `tol` of the rotated polyline `a` vertex for vertex, else None.

    The k-d tree only narrows the candidates: any wire within `tol` vertex
    for vertex has its FIRST vertex within `tol`, so the ball (drawn a little
    wider, for the tree's own rounding of the distance) holds every wire the
    full scan would accept, and visiting it in list order picks the same one.
    That keeps the answer the brute-force scan's (momwire#1029) at a cost
    that does not grow as wires squared — a 3,799-wire ground grid
    (momwire#1897) was the case that needed it."""
    for i in sorted(index.query_ball_point(a[0], r=2.0 * tol)):
        u = wires[i]
        b = polylines[u]
        if b.shape != a.shape:
            continue
        d = float(np.max(norm3(a - b)))
        if d <= tol and u not in taken:
            return u
    return None


def _image(polylines, wires, index, a, tol, taken):
    """`(wire, reversed)` for the image of rotated polyline `a`, else None.

    The same-direction copy is looked for FIRST, and only when there is none
    the copy that runs the other way (momwire#1897): a deck built edge by
    edge — sym4's ground lattice is 3,796 unit cells — has no reason to draw
    a rotated cell in the rotated direction, and 1,746 of its wires are
    reversed images. The preference is what keeps every deck the route read
    before reading the same: each of those found its forward copy."""
    u = _image_of(polylines, wires, index, a, tol, taken)
    if u is not None:
        return u, False
    u = _image_of(polylines, wires, index, a[::-1], tol, taken)
    if u is not None:
        return u, True
    return None


def _match_under(polylines, wires, axis, theta, tol, index=None):
    """`({w: its image}, {w: reversed})` when rotation by `theta` maps `wires`
    onto itself, else None. Vertex for vertex (§3.2), in the same vertex
    order or, since momwire#1897, the opposite one."""
    R = _rot(theta)
    centre = np.array([axis[0], axis[1], 0.0])
    if index is None:
        index = _first_vertex_index(polylines, wires)
    image = {}
    flip = {}
    taken = set()
    for w in wires:
        a = (polylines[w] - centre) @ R.T + centre
        found = _image(polylines, wires, index, a, tol, taken)
        if found is None:
            return None
        image[w], flip[w] = found
        taken.add(image[w])
    return image, flip


def _orbits(image, n):
    """The cycles of `image`, each required to have length exactly `n`."""
    seen = set()
    cycles = []
    for w in sorted(image):
        if w in seen:
            continue
        cyc = [w]
        seen.add(w)
        u = image[w]
        while u != w:
            if u in seen:
                return None
            cyc.append(u)
            seen.add(u)
            u = image[u]
        if len(cyc) != n:
            return None
        cycles.append(cyc)
    return cycles


def _far_azimuth(pl, axis):
    rho = _rho(pl, axis)
    v = pl[int(np.argmax(rho))]
    return math.atan2(v[1] - axis[1], v[0] - axis[0]) % (2 * math.pi)


def _diagnose_sectors(polylines, off, axis, tol, extent):
    """No rotation maps the off-axis wires onto themselves. Say which one
    breaks it, reading the deck as the registered class: one wire per sector,
    ordered by the azimuth of its farthest vertex."""
    n = len(off)
    order = sorted(off, key=lambda w: _far_azimuth(polylines[w], axis))
    w0 = order[0]
    pl0 = polylines[w0]
    len0 = _arc_length(pl0)
    az0 = _far_azimuth(pl0, axis)
    step = 2 * math.pi / n
    for s in range(1, n):
        w = order[s]
        pl = polylines[w]
        if pl.shape != pl0.shape:
            raise _refuse(
                f"sector {s}'s wire has {pl.shape[0]} vertices against sector "
                f"0's {pl0.shape[0]}. Every sector must map onto the next "
                f"under rotation by 2*pi/N, vertex for vertex."
            )
        ls = _arc_length(pl)
        if abs(ls - len0) > tol:
            raise _refuse(
                f"sector {s}'s wire is {100.0 * (ls - len0) / len0:+.2f} % "
                f"longer than sector 0's ({ls:.3f} m against {len0:.3f} m). "
                f"Every sector must map onto the next under rotation by "
                f"2*pi/N."
            )
        want = (az0 + s * step) % (2 * math.pi)
        got = _far_azimuth(pl, axis)
        rho = float(np.max(_rho(pl, axis)))
        if abs((got - want + math.pi) % (2 * math.pi) - math.pi) * rho > tol:
            raise _refuse(
                f"sector {s} sits at {math.degrees(got):.3f} deg, where {n} "
                f"sectors require {math.degrees(want):.3f} deg (tolerance "
                f"{TOL_REL:.0e} x {extent:.1f} m). Every sector must map onto "
                f"the next under rotation by 2*pi/N."
            )
    raise _refuse(
        f"the {n} off-axis wires do not map onto themselves under rotation by "
        f"2*pi/{n} about ({axis[0]:.3f}, {axis[1]:.3f}), and no coarser "
        f"rotation serves either. Every sector must map onto the next."
    )


def _try_partition(polylines, off, axis, tol):
    """`(N, sectors, reversed)` when the off-axis wires ARE a screen, else
    None. `reversed` is `SectorMap.reversed`: () when every member runs the
    way its sector-0 wire does.

    N is the LARGEST divisor of the off-axis wire count whose rotation maps
    them onto themselves: the symmetry group is cyclic, so a rotation by
    2*pi/n is a symmetry exactly when n divides the true sector count, and the
    largest such n IS that count.
    """
    if len(off) < 2:
        return None
    index = _first_vertex_index(polylines, off)
    for n in sorted(
        (d for d in range(2, len(off) + 1) if len(off) % d == 0), reverse=True
    ):
        matched = _match_under(polylines, off, axis, 2 * math.pi / n, tol, index)
        if matched is None:
            continue
        image, flip = matched
        cycles = _orbits(image, n)
        if cycles is None:
            continue
        # Each member's direction against its orbit's sector-0 wire: the
        # rotation's flips, accumulated along the cycle.
        runs_back = {}
        for cyc in cycles:
            o = False
            for w in cyc:
                runs_back[w] = o
                o ^= flip[w]
        sectors = tuple(tuple(cyc[s] for cyc in cycles) for s in range(n))
        rev = tuple(tuple(runs_back[w] for w in sec) for sec in sectors)
        return n, sectors, (rev if any(any(r) for r in rev) else ())
    return None


def _check_mesh_and_material(solver, sectors, polylines, axis, tol, rev=()):
    """§3.3: the same mesh, the same metal, the same jacket, and junction
    membership that rotates with the wire. A member that runs the other way
    (`rev`) is compared end for opposite end, its per-edge counts read
    backwards."""
    n = len(sectors)
    of_end, sizes = _junction_of_end(solver.junctions, len(polylines))
    a_cond = solver._conductor_radius_per_wire
    per_wire_specs = (
        ("conductivity", solver.wire_conductivity, "S/m", 1.0),
        ("jacket radius", solver.insulation_radius, "mm", 1e3),
        ("jacket eps_r", solver.insulation_eps_r, "", 1.0),
        ("jacket loss tangent", solver.insulation_loss_tangent, "", 1.0),
    )
    for c in range(len(sectors[0])):
        w0 = sectors[0][c]
        for s in range(1, n):
            w = sectors[s][c]
            back = bool(rev) and rev[s][c]
            npe_w = list(solver.n_per_edge_per_wire[w])
            if (npe_w[::-1] if back else npe_w) != list(solver.n_per_edge_per_wire[w0]):
                raise _refuse(
                    f"sector {s}'s wire is meshed "
                    f"{list(solver.n_per_edge_per_wire[w])} against sector 0's "
                    f"{list(solver.n_per_edge_per_wire[w0])}. Every sector must "
                    f"carry the same per-edge segment counts and grading."
                )
            if float(a_cond[w]) != float(a_cond[w0]):
                raise _refuse(
                    f"sector {s}'s conductor radius is "
                    f"{1e3 * float(a_cond[w]):.3f} mm against sector 0's "
                    f"{1e3 * float(a_cond[w0]):.3f} mm. Every sector must carry "
                    f"the same radius, conductivity and jacket."
                )
            for name, arr, unit, scale in per_wire_specs:
                if arr is None:
                    continue
                x, y = float(arr[w]), float(arr[w0])
                same = (math.isnan(x) and math.isnan(y)) or x == y
                if not same:
                    raise _refuse(
                        f"sector {s}'s {name} is {scale * x:.6g} {unit} against "
                        f"sector 0's {scale * y:.6g} {unit}. Every sector must "
                        f"carry the same radius, conductivity and jacket."
                    )
            if solver.distributed_rlc is not None and (
                solver.distributed_rlc[w] != solver.distributed_rlc[w0]
            ):
                raise _refuse(
                    f"sector {s}'s distributed RLC is "
                    f"{solver.distributed_rlc[w]!r} against sector 0's "
                    f"{solver.distributed_rlc[w0]!r}. Every sector must carry "
                    f"the same radius, conductivity and jacket."
                )
            for end in ("start", "end"):
                end_w = {"start": "end", "end": "start"}[end] if back else end
                j0, j = of_end.get((w0, end)), of_end.get((w, end_w))
                if (j0 is None) != (j is None):
                    raise _refuse(
                        f"sector {s}'s wire is junctioned at its {end} where "
                        f"sector 0's is not (or the reverse). A junction must "
                        f"map to a junction under the rotation."
                    )
                if j is None:
                    continue
                p0 = _junction_point(polylines, solver.junctions, j0)
                on_axis = math.hypot(p0[0] - axis[0], p0[1] - axis[1]) <= tol
                if on_axis and j != j0:
                    raise _refuse(
                        f"sector {s}'s wire meets junction {j} at its {end} "
                        f"where sector 0's meets junction {j0}, and that node "
                        f"is ON the axis. An on-axis junction must be shared "
                        f"by every sector."
                    )
                if not on_axis and sizes[j] != sizes[j0]:
                    raise _refuse(
                        f"sector {s}'s {end} junction joins {sizes[j]} wire "
                        f"ends against sector 0's {sizes[j0]}. Junction "
                        f"membership must rotate with the wire."
                    )


def _port_sites(solver, polylines):
    """(name, 3-vector) for every drive/readout site, in the house port order
    [gap feeds..., junction ports..., node gaps...]."""
    sites = []
    for i, (w, arc, _v) in enumerate(solver.feeds):
        pl = polylines[int(w)]
        seg = norm3(np.diff(pl, axis=0))
        total = float(np.sum(seg))
        s = 0.5 * total if arc is None else float(arc)
        s = min(max(s, 0.0), total)
        # walk to the edge holding `s`; the site is the point at that arc
        acc = 0.0
        point = pl[-1]
        for e in range(len(seg)):
            if s <= acc + seg[e] or e == len(seg) - 1:
                t = 0.0 if seg[e] == 0.0 else (s - acc) / seg[e]
                point = pl[e] + t * (pl[e + 1] - pl[e])
                break
            acc += seg[e]
        sites.append((f"feed {i}", np.asarray(point, dtype=float)))
    for i, (j, _v) in enumerate(solver.junction_ports):
        sites.append(
            (
                f"junction port {i}",
                np.asarray(
                    _junction_point(polylines, solver.junctions, int(j)), dtype=float
                ),
            )
        )
    for i, (w, end, _v) in enumerate(solver.node_gaps):
        pl = polylines[int(w)]
        sites.append(
            (f"node gap {i}", np.asarray(pl[0] if end == "start" else pl[-1], float))
        )
    return sites


def sector_map(solver, tol_rel=TOL_REL) -> SectorMap:
    """The deck's sector grouping, or a refusal naming the first condition of
    `PLAN-phase1.md` §3 that fails.

    Reads only what is frozen at construction — the polylines, the mesh, the
    per-wire material, the junctions, the ports and the ground kind — so the
    refusal lands before any fill.
    """
    polylines = [np.asarray(p, dtype=float) for p in solver.wires_polylines]
    extent = max(float(np.max(np.abs(p))) for p in polylines)
    tol = tol_rel * extent

    # 1 and 2 are read TOGETHER, and the order is what makes the refusals
    # land on the right condition. The wires on the axis are tried as the
    # axial group first: when what is left IS a screen, that reading is the
    # answer and nothing is grown. Only when it is not does the group grow
    # across degree-2 junctions — which is how a tilted mast becomes an axial
    # wire that fails condition 1 (its own sentence) rather than a spare
    # sector that fails condition 2 (somebody else's).
    every = set(range(len(polylines)))
    seed_centres = _vertical_seeds(polylines, tol)
    one = _one_line(seed_centres, tol)
    if one is None:
        # Vertical wires on more than one line (momwire#1897): one line is the
        # axis and the others are sector wires.
        axis, axial, n, sectors, rev = _off_axis_verticals(
            polylines, seed_centres, every, tol
        )
        return _rest_of_the_rule(
            solver, polylines, axis, axial, n, sectors, rev, tol, extent
        )
    axis, seeds = one
    off = sorted(every - set(seeds))
    # Growth can only ever take wires OUT of `off`, so too few of them here is
    # already the whole answer, and saying so beats a diagnosis of the one
    # arm's shape. A one-radial screen reaches this: its hub is degree 2, so
    # the growth below would swallow the arm and refuse it for running 90 deg
    # off the axis, which is true and useless.
    if len(off) < 2:
        raise _refuse(
            f"the deck has {len(off)} wire(s) off the axis, and the route "
            f"needs N >= 2 sectors. A single arm is not a screen."
        )
    found = _try_partition(polylines, off, axis, tol)
    if found is not None:
        axial, (n, sectors, rev) = seeds, found
    else:
        axial = _grown(polylines, solver.junctions, seeds)
        _condition_1(polylines, axial, axis, tol)
        off = sorted(every - set(axial))
        if len(off) < 2:
            raise _refuse(
                f"the deck has {len(off)} wire(s) off the axis, and the route "
                f"needs N >= 2 sectors. A single arm is not a screen."
            )
        _diagnose_sectors(polylines, off, axis, tol, extent)
        raise AssertionError("unreachable: _diagnose_sectors always refuses")
    return _rest_of_the_rule(
        solver, polylines, axis, axial, n, sectors, rev, tol, extent
    )


def _off_axis_verticals(polylines, seed_centres, every, tol):
    """`(axis, axial, N, sectors, reversed)` for a deck whose vertical wires stand on
    more than one line (momwire#1897), or a refusal saying which wire has no
    orbit.

    Each line is tried as the axis, nearest the vertex centroid first: its
    wires are the axial group, and every other wire — the other verticals
    included — must partition into orbits of size N, exactly as a screen's
    radials do. The first line that partitions is the answer; a finite
    structure has at most one vertical rotation axis, so at most one can."""
    lines = _lines(polylines, seed_centres, tol)
    for axis, axial in lines:
        off = sorted(every - set(axial))
        found = _try_partition(polylines, off, axis, tol)
        if found is not None:
            n, sectors, rev = found
            return axis, axial, n, sectors, rev
    axis, axial = lines[0]
    off = sorted(every - set(axial))
    vertical = {w: c for w, c in seed_centres}
    _diagnose_off_axis(polylines, off, axis, tol, vertical)
    raise AssertionError("unreachable: _diagnose_off_axis always refuses")


def _diagnose_off_axis(polylines, off, axis, tol, vertical):
    """No line's partition worked. Name the wires the rotation strands, under
    the N that strands the fewest: when a deck is a C_N structure with one
    wire moved, that N is the deck's own and the wire named is one of the two
    the move broke (the one moved, or the one whose image it was)."""
    index = _first_vertex_index(polylines, off)
    centre = np.array([axis[0], axis[1], 0.0])
    best = None
    for n in sorted(
        (d for d in range(2, len(off) + 1) if len(off) % d == 0), reverse=True
    ):
        R = _rot(2 * math.pi / n)
        stranded = [
            w
            for w in off
            if _image(
                polylines, off, index, (polylines[w] - centre) @ R.T + centre, tol, ()
            )
            is None
        ]
        if best is None or len(stranded) < len(best[1]):
            best = (n, stranded)
    if best is None:
        raise _refuse(
            f"the deck has {len(off)} wire(s) off the axis "
            f"({axis[0]:.3f}, {axis[1]:.3f}), and the route needs N >= 2 "
            f"sectors. A single arm is not a screen."
        )
    n, stranded = best
    if not stranded:
        raise _refuse(
            f"every one of the {len(off)} wires off the axis "
            f"({axis[0]:.3f}, {axis[1]:.3f}) has a copy under rotation by "
            f"2*pi/{n}, but they do not close into orbits of {n}. Every wire "
            f"off the axis must lie in an orbit of N rotated copies."
        )
    w = stranded[0]
    where = (
        f", which stands on a vertical line at "
        f"({vertical[w][0]:.3f}, {vertical[w][1]:.3f})"
        if w in vertical
        else ""
    )
    raise _refuse(
        f"under rotation by 2*pi/{n} about the axis "
        f"({axis[0]:.3f}, {axis[1]:.3f}), {len(stranded)} of the {len(off)} "
        f"wires off it have no rotated copy; the first is wire {w}{where}. "
        f"Every wire off the axis, a vertical one included, must be one of N "
        f"rotated copies."
    )


def _rotated_copies_are_ports(point, sites, axis, n, tol):
    """Whether every image of `point` under rotation by 2*pi*t/n, t = 1..n-1,
    is itself a port site."""
    centre = np.array([axis[0], axis[1], 0.0])
    for t in range(1, n):
        q = (np.asarray(point) - centre) @ _rot(2 * math.pi * t / n).T + centre
        if not any(float(np.linalg.norm(q - p)) <= tol for _nm, p in sites):
            return False
    return True


def _rest_of_the_rule(solver, polylines, axis, axial, n, sectors, rev, tol, extent):
    """Conditions 3 to 6, once the axis and the sectors are read."""
    # 3. identical discretisation and material, junctions that rotate
    _check_mesh_and_material(solver, sectors, polylines, axis, tol, rev)

    # 4. the port SET maps onto itself. A port on the axis is its own image;
    #    one off it needs a port at each of its N-1 rotated images
    #    (momwire#1897: sym4's four loaded posts). A lone off-axis port is
    #    refused as it always was: no port set containing it is symmetric.
    sites = _port_sites(solver, polylines)
    for name, point in sites:
        rho = math.hypot(point[0] - axis[0], point[1] - axis[1])
        if rho > tol and not _rotated_copies_are_ports(point, sites, axis, n, tol):
            raise _refuse(
                f"port '{name}' sits at "
                f"({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}), off the "
                f"symmetry axis ({axis[0]:.3f}, {axis[1]:.3f}), and its "
                f"rotated images are not all ports. An off-axis port needs a "
                f"copy in every sector; alone, its drive excites every "
                f"harmonic, and this route serves the axis-symmetric drive "
                f"only."
            )

    # 5. lumped loads. The house spelling is per-wire and DISTRIBUTED, checked
    #    at 3 above. A formulation that serves discrete sites (razor's
    #    `lumped_loads`; bspline reaches the same physics as a node-gap port,
    #    so this loop is empty there) names one by (wire, arclength): a load on
    #    an axial wire is on the axis by condition 1, and a load on a sector
    #    wire needs its N rotated copies.
    of_sector = {w: (s, c) for s, sec in enumerate(sectors) for c, w in enumerate(sec)}
    copies = {}
    for w, arc, z in getattr(solver, "lumped_loads", ()) or ():
        if int(w) in set(axial):
            continue
        s, c = of_sector[int(w)]
        copies.setdefault((c, None if arc is None else float(arc), complex(z)), set())
        copies[(c, None if arc is None else float(arc), complex(z))].add(s)
    for (c, arc, z), seen in sorted(copies.items(), key=lambda kv: kv[0][0]):
        if len(seen) != n:
            raise _refuse(
                f"the lumped load {z} at arclength {arc} on sector position "
                f"{c} is carried by {len(seen)} of {n} sectors. A load must sit "
                f"on the axis or come as N identical rotated copies."
            )

    # 6. a ground invariant under rotation about the axis
    kind = solver._rotational_ground_kind()
    if kind not in AXISYMMETRIC_GROUNDS:
        raise _refuse(
            f"the ground model {kind!r} is not invariant under rotation about "
            f"the axis. Free space, PEC, a reflection-coefficient ground and a "
            f"Sommerfeld half-space qualify."
        )

    return SectorMap(
        axis=axis,
        n_sectors=n,
        sectors=sectors,
        axial=axial,
        tol=tol,
        extent=extent,
        reversed=rev,
    )


# ---------------------------------------------------------------------------
# The route itself — momwire#1029 phase 2 unit D
#
# Moved here from `BSplineSolver` verbatim, `self` renamed to `solver` and
# nothing else changed; the four methods stay on the solver as one-line
# delegations, so every phase-1 test and every scratch harness keeps its call.
#
# It belongs beside the CHECK, not inside a 6,900-line solver, for the reason
# the check does: none of it is basis-shaped. `dof_groups` reads the sector
# map and the solver's own wire->basis table, `observer_rows` reads the
# geometry, and `harmonic_zero_block` and `solve` are linear algebra on the
# assembled Z. What is solver-shaped — the fills, the drive, the readout — is
# reached through the solver that is handed in, which is also what keeps this
# module free of an import back into `bspline`.
#
# Phase 2 grows exactly this code: phase 3's general drive builds the other
# N-1 harmonic blocks beside `harmonic_zero_block`, and an off-axis port
# picks the sectors apart in `dof_groups`. That is the reason to give it a
# module now rather than after. momwire#1897 is that growth:
# `harmonic_blocks` and `solve_ports` serve the port entries, harmonic by
# harmonic, and `dof_signs` lets a sector wire run either way round.
# ---------------------------------------------------------------------------


# The bar on "the drive is rotation-invariant", checked per right-hand side:
# a vector with content in another harmonic would need the N-1 blocks this
# route does not build, so it is refused as a drive that needs every harmonic.
# Relative to the vector's largest entry. The two right-hand sides this route
# ever sees are EXACTLY invariant, not approximately: the gap source vector is
# zero on every sector (phase 0's F2 measured 0), and the hub's KCL row
# carries the same sign on each sector's directional basis (S-0 measured the
# permuted row's largest change at exactly 0.0). The bar is loose against that
# so a port column built by a different route is judged by roundoff rather
# than by bit equality, and it is still 4 orders under anything physical.
DRIVE_TOL = 1e-12


def refuse_if_not_invariant(rhs, sectors):
    """Refuse a right-hand side that is not the same on every sector.

    `rhs` is (n_basis, n_columns); every column is judged together, against
    the whole array's largest entry, which is how the Schur step applies
    them. A vector with content in another harmonic would need the N-1
    blocks this route does not build, so it is refused rather than solved
    through the one block that exists.
    """
    r0 = rhs[sectors[0]]
    scale = max(float(np.max(np.abs(rhs))), 1e-300)
    for t in range(1, len(sectors)):
        spread = float(np.max(np.abs(rhs[sectors[t]] - r0))) / scale
        if spread > DRIVE_TOL:
            raise RotationalSymmetryRefused(
                f"rotational symmetry: the right-hand side differs "
                f"between sector 0 and sector {t} by {spread:.3e} of "
                f"its largest entry, so the drive is not "
                f"rotation-invariant and needs every harmonic. This "
                f"route serves the axis-symmetric drive only. Drop "
                f"rotational_symmetry=True to solve this deck densely."
            )


def check_drive(v, kcl_A, sectors):
    """The drive guard, BEFORE the fill (momwire#1029 phase 2b).

    Every other refusal on this route is frozen at construction, so a ticked
    box on the wrong design costs nothing. This one is about the right-hand
    side rather than the deck, and it used to sit inside the Schur step —
    after a fill on `compute_impedance`, and after k fills on a sweep.

    It does not have to. `_feed_drive_and_readout` never reads Z and the
    fill never reads the drive, so the two commute; checking here judges
    EXACTLY the two objects `solve` will apply Z^-1 to (the source vector,
    and the constraint rows), and there is nothing else the solve could
    reach that this does not.
    """
    refuse_if_not_invariant(np.asarray(v, dtype=np.complex128)[:, None], sectors)
    if kcl_A.shape[0]:
        refuse_if_not_invariant(kcl_A.T.astype(np.complex128), sectors)


def _invariant_rows(kcl_A, sectors):
    """Per row: True when it is the same on every sector, bit for bit."""
    r0 = kcl_A[:, sectors[0]]
    ok = np.ones(kcl_A.shape[0], dtype=bool)
    for t in range(1, len(sectors)):
        ok &= np.all(kcl_A[:, sectors[t]] == r0, axis=1)
    return ok


def _sparse_key(row):
    nz = np.flatnonzero(row)
    return nz.tobytes() + row[nz].tobytes()


def _row_orbits(kcl_A, sectors):
    """`(invariant, reps)`: per row, whether it is the same on every sector,
    and one representative row index per orbit of the others — each orbit
    checked to be exactly the N rotated copies of its representative.

    Every row is a +-1 pattern on directional bases, so a rotated copy is
    found by exact comparison."""
    invariant = _invariant_rows(kcl_A, sectors)
    if bool(np.all(invariant)):
        return invariant, []
    n = len(sectors)
    # dof -> its image one sector on; dofs outside every sector (the axial
    # group) are fixed by the rotation.
    step = np.arange(kcl_A.shape[1], dtype=np.int64)
    for s in range(n):
        step[sectors[s]] = sectors[(s + 1) % n]
    by_key = {}
    for r in np.flatnonzero(~invariant):
        by_key.setdefault(_sparse_key(kcl_A[r]), []).append(int(r))
    used = set()
    reps = []
    for r in np.flatnonzero(~invariant):
        r = int(r)
        if r in used:
            continue
        used.add(r)
        reps.append(r)
        image = kcl_A[r]
        for t in range(1, n):
            rotated = np.zeros_like(image)
            rotated[step] = image
            image = rotated
            hit = [q for q in by_key.get(_sparse_key(image), ()) if q not in used]
            if not hit:
                raise RotationalSymmetryRefused(
                    f"rotational symmetry: junction constraint {r} has no "
                    f"rotated copy among the deck's junctions (rotation by "
                    f"{t} of {n} sectors), so the junctions do not rotate "
                    f"with the wires. {_WAY_OUT}"
                )
            used.add(hit[0])
    return invariant, reps


def orbit_constraints(kcl_A, sectors):
    """The constraint rows an INVARIANT solution has to satisfy, one per
    orbit (momwire#1897).

    A KCL row closes one junction. A junction ON the axis is shared by every
    sector (condition 3), so its row is already the same on every sector and
    is kept as it is. A junction OFF the axis — a cage vertical meeting its
    ring — has N rotated copies, and each copy's row lives on one sector's
    dofs only, so no single row is rotation-invariant and the drive guard
    would refuse it.

    For a rotation-invariant solution x the N rows of one orbit say the same
    thing (row t applied to x equals row 0 applied to x), and the dense
    solve's multipliers on them are equal too, so the orbit's N constraints
    and their N multiplier columns are exactly one constraint, the orbit SUM,
    and one column, the same sum. That row is invariant by construction, so
    the harmonic-0 solve serves the deck unchanged. It is the harmonic-0 row
    of `_harmonic_constraints`.

    A matrix whose rows are all invariant is returned AS IS — the same
    object — which is what keeps every deck the route served before
    byte-identical.
    """
    if kcl_A.shape[0] == 0:
        return kcl_A
    invariant, reps = _row_orbits(kcl_A, sectors)
    if not reps:
        return kcl_A
    n = len(sectors)
    rep_of = {r: r for r in reps}
    out = []
    for r in range(kcl_A.shape[0]):
        if invariant[r]:
            out.append(kcl_A[r])
        elif r in rep_of:
            row = kcl_A[r]
            # the axial part, fixed by the rotation, is counted N times
            total = n * row
            block = sum(row[sectors[u]] for u in range(n))
            for u in range(n):
                total[sectors[u]] = block
            out.append(total)
    return np.vstack(out)


def _invariant(rhs, sectors):
    """`refuse_if_not_invariant`'s test, as a bool."""
    try:
        refuse_if_not_invariant(rhs, sectors)
    except RotationalSymmetryRefused:
        return False
    return True


def harmonics_needed(B, kcl_A, sectors):
    """`(0,)` when every port column is rotation-invariant (every port on the
    axis), else every harmonic: a port off the axis, one of N rotated
    copies, drives each of them. The constraints never add one: their orbit
    sums are harmonic-0 rows (`orbit_constraints`), and a harmonic no column
    excites has the zero solution under homogeneous constraints."""
    if B.shape[1] == 0 or _invariant(B, sectors):
        return (0,)
    return tuple(range(len(sectors)))


def harmonic_blocks(solver, Z, sectors, axial, harmonics, row_of=None):
    """`{h: LU}`: the factorised block of each harmonic in `harmonics`, from
    the sector-0 rows of the (row-compact) fill (momwire#1897).

    In the DFT over sectors the radial-radial part of Z is block-diagonal,
    Lambda_h = sum_d C_d exp(+2 pi j h d / N), C_d being sector 0's rows
    against sector d's columns, and only harmonic 0 couples to the axis: the
    axial rows read the SUM of a harmonic's sectors, which vanishes for
    h != 0. So harmonic 0 is the (m + p) block `harmonic_zero_block` builds,
    and every other harmonic is one m x m block — N factorisations of size m
    against one of size N m + p.

    Each Lambda_h is accumulated straight into its own Fortran-ordered
    array and factorised in place, so the route never holds a stack of the
    N blocks next to the fill. The caller drops Z after this returns; the
    solves read only the factors.
    """
    n = len(sectors)
    out = {}
    if 0 in harmonics:
        K0, _m, _p = harmonic_zero_block(solver, Z, sectors, axial, row_of=row_of)
        out[0] = _lapack.lu_factor(K0, overwrite_a=True)
        del K0
    s0 = sectors[0]
    r_s0 = s0 if row_of is None else row_of[s0]
    m = int(s0.size)
    hs = [h for h in harmonics if h != 0]
    if not hs:
        return out
    # Sector t's block C_t is gathered ONCE and folded into every harmonic
    # (momwire#1902): the h-outer form gathered it once per harmonic, N - 1
    # fancy-index copies of an m x m block where one does, and that was 46 %
    # of a 13-frequency C6 sweep. Each Lambda_h still receives its terms in
    # ascending t as `w * C_t` then an add, so the sums are bit-identical to
    # the h-outer form. The N - 1 accumulators are the blocks the factors
    # replace in place, so the peak is one gathered block above it.
    lams = {h: np.zeros((m, m), dtype=np.complex128, order="F") for h in hs}
    term = np.empty((m, m), dtype=np.complex128, order="F")
    for t in range(n):
        C_t = Z[np.ix_(r_s0, sectors[t])]
        for h in hs:
            np.multiply(np.exp(2j * math.pi * h * t / n), C_t, out=term)
            lams[h] += term
        del C_t
    for h in hs:
        out[h] = _lapack.lu_factor(lams.pop(h), overwrite_a=True)
    return out


# `_harmonic_constraints` by the digest of its inputs (momwire#1902): the
# rows are geometry only (the KCL rows and the sector map), yet a frequency
# sweep on the route rebuilt them, orbit search and all, at every k.
_CONSTRAINTS_CACHE: OrderedDict = OrderedDict()
_CONSTRAINTS_CACHE_SIZE = 4


def _constraints_key(kcl_A, sectors, axial):
    """A digest of exactly what `_harmonic_constraints` reads: the shape,
    dtype, nonzero positions and values of `kcl_A` (a +-1 pattern, so its
    sparse form is small), and the sector map. One `flatnonzero` pass,
    against the orbit search it stands in for."""
    a = np.ascontiguousarray(kcl_A)
    nz = np.flatnonzero(a)
    h = hashlib.blake2b(digest_size=20)
    h.update(repr((a.shape, a.dtype.str, len(sectors))).encode())
    h.update(nz.astype(np.int64).tobytes())
    h.update(a.ravel()[nz].tobytes())
    for sec in sectors:
        h.update(np.asarray(sec, dtype=np.int64).tobytes())
        h.update(b"|")
    h.update(np.asarray(axial, dtype=np.int64).tobytes())
    return h.digest()


def _harmonic_constraints(kcl_A, sectors, axial):
    """The constraint rows harmonic by harmonic, `(rows0, rows_h)`: the
    rows `_harmonic_constraints_parts` derives, memoised on
    `_constraints_key`. The arrays handed back are read-only, since the
    cache shares them between calls."""
    key = _constraints_key(kcl_A, sectors, axial)
    hit = _CONSTRAINTS_CACHE.get(key)
    if hit is None:
        rows0, a_hat, m = _harmonic_constraints_parts(kcl_A, sectors, axial)
        rows0.setflags(write=False)
        if a_hat is not None:
            a_hat.setflags(write=False)
        hit = (rows0, a_hat, m)
        _CONSTRAINTS_CACHE[key] = hit
        while len(_CONSTRAINTS_CACHE) > _CONSTRAINTS_CACHE_SIZE:
            _CONSTRAINTS_CACHE.popitem(last=False)
    else:
        _CONSTRAINTS_CACHE.move_to_end(key)
    rows0, a_hat, m = hit

    def rows_h(h):
        return a_hat[h] if a_hat is not None else np.zeros((0, m), np.complex128)

    return rows0, rows_h


def _harmonic_constraints_parts(kcl_A, sectors, axial):
    """`(rows0, a_hat, m)`: the parts `_harmonic_constraints` serves.

    `rows0` is harmonic 0's, on its (m + p) unknowns [x_hat_0, x_axial]: an
    invariant row r reads sqrt(N) r_sector . x_hat_0 + r_axial . x_axial,
    and each orbit of the others reads its orbit sum (`orbit_constraints`).

    `a_hat[h]` is harmonic h's rows, on its m unknowns: an orbit whose
    representative row has sector blocks a_u enforces, for its N rotated
    copies together, a_hat_h . x_hat_h = 0 with a_hat_h = sum_u exp(+2 pi j
    h u / N) a_u — the N rows of an orbit ARE its N harmonic rows, under the
    same DFT that block-diagonalises Z. The multiplier columns are the
    conjugate (a_hat_{-h}), since the rows are real.
    """
    n = len(sectors)
    sq = math.sqrt(n)
    m = int(sectors[0].size)
    p = int(axial.size)
    invariant, reps = _row_orbits(kcl_A, sectors)
    inv = np.flatnonzero(invariant)
    rows0 = np.zeros((inv.size + len(reps), m + p), dtype=np.complex128)
    for i, r in enumerate(inv):
        rows0[i, :m] = sq * kcl_A[r, sectors[0]]
        rows0[i, m:] = kcl_A[r, axial]
    blocks = np.zeros((n, len(reps), m), dtype=np.float64)
    for i, r in enumerate(reps):
        if np.any(kcl_A[r, axial]):
            raise RotationalSymmetryRefused(
                f"rotational symmetry: junction constraint {r} joins the "
                f"axial group to the sectors without being shared by every "
                f"sector. {_WAY_OUT}"
            )
        for u in range(n):
            blocks[u, i] = kcl_A[r, sectors[u]]
    rows0[inv.size :, :m] = blocks.sum(axis=0)
    # a_hat[h] = sum_u exp(+2 pi j h u / N) a_u: numpy's inverse DFT, times N.
    a_hat = np.fft.ifft(blocks, axis=0) * n if reps else None
    return rows0, a_hat, m


def _constrained(lu, rhs, rows, cols):
    """`_solve_with_kcl_ports`'s Schur step on one harmonic's factor."""
    if rows.shape[0] == 0:
        return _lapack.lu_solve(lu, rhs)
    sol = _lapack.lu_solve(lu, np.hstack([rhs, cols]))
    W, X = sol[:, : rhs.shape[1]], sol[:, rhs.shape[1] :]
    lam = _lapack.solve(rows @ X, rows @ W)
    return W - X @ lam


def solve_ports(blocks, B, kcl_A, sectors, axial):
    """`_solve_with_kcl_ports` on the sector route (momwire#1897): every port
    column of `B` under the constraints `kcl_A`, through the harmonics
    `blocks` holds (`harmonic_blocks`, chosen by `harmonics_needed`).

    The columns are DFT'd over the sectors, each harmonic is solved under
    its own constraint rows (`_harmonic_constraints`) — the orbits of the
    deck's junctions block-diagonalise the Schur step exactly as the
    rotation block-diagonalises Z, so harmonic h carries one row per orbit
    rather than N — and the harmonics are summed back. Harmonic 0 carries
    the axial unknowns and the invariant rows too.
    """
    n = len(sectors)
    sq = math.sqrt(n)
    m = int(sectors[0].size)
    n_p = B.shape[1]
    rows0, rows_h = _harmonic_constraints(kcl_A, sectors, axial)
    b = np.empty((n, m, n_p), dtype=np.complex128)
    for t in range(n):
        b[t] = B[sectors[t]]
    # b_hat[h] = (1/sqrt N) sum_t exp(-2 pi j h t / N) b_t
    bh = np.fft.fft(b, axis=0) / sq
    xh = np.zeros_like(bh)
    sol0 = _constrained(
        blocks[0], np.vstack([bh[0], B[axial]]), rows0, rows0.T.astype(np.complex128)
    )
    xh[0] = sol0[:m]
    for h in blocks:
        if h == 0:
            continue
        rows = rows_h(h)
        xh[h] = _constrained(blocks[h], bh[h], rows, np.conj(rows).T)
    # x_t = (1/sqrt N) sum_h exp(+2 pi j h t / N) x_hat[h]
    x = np.fft.ifft(xh, axis=0) * sq
    X = np.empty((B.shape[0], n_p), dtype=np.complex128)
    for t in range(n):
        X[sectors[t]] = x[t]
    X[axial] = sol0[m:]
    return X


def dof_groups(solver, wire_basis_global, n_basis_total):
    """`(sectors, axial)`: one global-index array per sector image, and
    the axial group's.

    `sectors[s][i]` and `sectors[t][i]` are the SAME basis, of the same
    wire of the same orbit, under the rotation — gate S-0 pins that
    bijection dof by dof (kind, local index, end position and junction all
    match at equal position), and it is what lets row `i` of sector 0
    stand for its N images. `s` is only a label: the route sums a row over
    a source dof's whole orbit, so relabelling the sectors cannot change
    anything it computes.

    A member that runs opposite to its sector-0 wire (`SectorMap.reversed`,
    momwire#1897) lists its bases backwards: on a mesh that is the mirror
    of sector 0's, its basis i from the far end is the image of sector 0's
    basis i. Its current runs the other way too, which is `dof_signs`.
    """
    smap = solver._rotational_map
    rev = smap.reversed
    sectors = []
    for s in range(smap.n_sectors):
        idx = []
        for c, w in enumerate(smap.sectors[s]):
            kept, l2g = wire_basis_global[w]
            local = [int(l2g[i]) for i in range(len(kept))]
            idx.extend(local[::-1] if rev and rev[s][c] else local)
        sectors.append(np.asarray(idx, dtype=np.int64))
    on_sector = np.zeros(n_basis_total, dtype=bool)
    for idx in sectors:
        on_sector[idx] = True
    return sectors, np.nonzero(~on_sector)[0].astype(np.int64)


def dof_signs(solver, wire_basis_global, n_basis_total):
    """`D`, the +-1 per dof that turns the deck's bases into the ROTATED
    ones, or None when no member runs backwards (every deck before
    momwire#1897, which then sees none of this).

    A wire drawn the other way carries the same physical current with the
    opposite sign, so rotating sector 0's basis onto it gives MINUS its own
    basis. In the signed bases f~ = D f the operator is block-circulant
    again: Z~ = D Z D, the drive D v, the constraint rows A D, and the
    coefficients x = D x~. The route solves in those bases and signs back,
    every step exact, since D is a sign. Sector 0 and the axial group are
    never flipped, so the rows the fill holds keep their sign and only the
    columns of Z need it.
    """
    smap = solver._rotational_map
    if not smap.reversed:
        return None
    D = np.ones(n_basis_total, dtype=np.float64)
    for s, row in enumerate(smap.reversed):
        for c, back in enumerate(row):
            if back:
                kept, l2g = wire_basis_global[smap.sectors[s][c]]
                D[[int(l2g[i]) for i in range(len(kept))]] = -1.0
    return D


def observer_rows(solver, geom):
    """The observer segments one sector's fill needs: sector 0's wires and
    every axial wire. No basis straddles two wires (S-0's S0a), so a
    whole-wire segment set is exactly a whole-dof row set."""
    smap = solver._rotational_map
    seg_off = geom["seg_offsets"]
    per_wire = geom["per_wire"]
    runs = [
        np.arange(seg_off[w], seg_off[w] + per_wire[w]["n_total"])
        for w in (*smap.sectors[0], *smap.axial)
    ]
    return np.sort(np.concatenate(runs))


def harmonic_zero_block(solver, Z, sectors, axial, row_of=None):
    """The harmonic-0 block and its consistency reading.

    `Z` is the square fill, or with `row_of` the ROW-COMPACT one
    (momwire#1132): basis row m held at row `row_of[m]`. Only sector 0's rows
    and the axial rows are read either way, so the gathers below are the
    same numbers from either shape.

    An on-axis drive is rotation-invariant, so the solution is too and
    only harmonic 0 is ever excited (`PLAN.md` §2, measured at F4). In the
    DFT over sectors the radial-radial part is block-diagonal with
    Lambda_h = sum_d C_d exp(+2 pi j h d / N), and harmonic 0 is the only
    block that couples to the axis:

        K_0 = [[Lambda_0, sqrt(N) B], [sqrt(N) C, Z_MM]]

    with c_hat_0 = sqrt(N) c_sector. Lambda_0 = sum_d C_d is the ROW SUM
    of sector 0's rows over every sector's columns, so it needs no
    labelling; B is sector 0's axial columns. The axis-against-sector
    block is the SUM of the N copies scaled by 1/sqrt(N) — using the sum
    rather than one copy averages their roundoff, and their spread is a
    free check on the sector assignment (they are not bit-identical:
    rotated copies round differently).
    """
    n = len(sectors)
    m = int(sectors[0].size)
    p = int(axial.size)
    s0 = sectors[0]
    sq = math.sqrt(n)
    # The rows the gathers read, where they live in `Z`.
    r_s0 = s0 if row_of is None else row_of[s0]
    r_ax = axial if row_of is None else row_of[axial]
    K0 = np.empty((m + p, m + p), dtype=np.complex128)
    lam0 = Z[np.ix_(r_s0, sectors[0])].copy()
    for t in range(1, n):
        lam0 += Z[np.ix_(r_s0, sectors[t])]
    K0[:m, :m] = lam0
    K0[:m, m:] = sq * Z[np.ix_(r_s0, axial)]
    c_0 = Z[np.ix_(r_ax, sectors[0])]
    c_sum = c_0.copy()
    worst = 0.0
    scale = float(np.max(np.abs(c_0))) if c_0.size else 1.0
    for t in range(1, n):
        c_t = Z[np.ix_(r_ax, sectors[t])]
        worst = max(worst, float(np.max(np.abs(c_t - c_0))))
        c_sum += c_t
    K0[m:, :m] = c_sum / sq
    K0[m:, m:] = Z[np.ix_(r_ax, axial)]
    solver._rotational_copy_spread = worst / max(scale, 1e-300)
    return K0, m, p


def solve(solver, Z, v, kcl_A, sectors, axial, row_of=None):
    """The constrained solve through K_0 — `_solve_with_kcl`'s Schur step
    with the sector inverse in place of the dense one.

    The Schur algebra only ever applies Z^-1 to `v` and to the constraint
    rows, and BOTH are rotation-invariant for an on-axis drive: the gap
    source vector is supported on the feed's own axial wire, and the hub's
    KCL row carries the same +1 on every sector's directional basis. That
    is asserted rather than assumed — a right-hand side with content in
    another harmonic would need the other N-1 blocks, which this route
    does not build.

    `row_of` is `harmonic_zero_block`'s: set when `Z` is row-compact.
    """
    n = len(sectors)
    sq = math.sqrt(n)
    K0, m, _p = harmonic_zero_block(solver, Z, sectors, axial, row_of=row_of)
    lu = _lapack.lu_factor(K0, overwrite_a=True)

    def z_inv(rhs):
        rhs = np.asarray(rhs, dtype=np.complex128).reshape(rhs.shape[0], -1)
        # Kept here as well as in `check_drive` (momwire#1029 phase 2b), and
        # not as belt and braces: this is the guard for every caller that
        # reaches the solve directly — `BSplineSolver._rotational_solve` and
        # the scratch harnesses that drive it — where no `compute_impedance`
        # has run the early one.
        refuse_if_not_invariant(rhs, sectors)
        r0 = rhs[sectors[0]]
        top = np.vstack([sq * r0, rhs[axial]])
        sol = _lapack.lu_solve(lu, top)
        out = np.empty((rhs.shape[0], rhs.shape[1]), dtype=np.complex128)
        per_sector = sol[:m] / sq
        for t in range(n):
            out[sectors[t]] = per_sector
        out[axial] = sol[m:]
        return out

    w = z_inv(v[:, None])[:, 0]
    if kcl_A.shape[0] == 0:
        return w
    X = z_inv(kcl_A.T.astype(np.complex128))
    lam = _lapack.solve(kcl_A @ X, kcl_A @ w)
    return w - X @ lam


def sector_fill(solver, geom, supp_seg, polys, sectors, axial, n_basis_total):
    """`(Z, row_of)`: the fill of sector 0's rows and the axial rows against
    every source — the route's one heavy call, shared by `compute_impedance`
    and `compute_port_solution`. `row_of` is None for the square
    destination."""
    rows = observer_rows(solver, geom)
    # `_compute_Z_operator` takes `rows=` on both of its fills: the buried one
    # (momwire#1029) and, since momwire#1131, the above-ground one -- so the
    # route serves an elevated or surface screen through the same call.
    if not COMPACT_Z:
        Z = solver._compute_Z_operator(geom, supp_seg, polys, rows=rows)
        row_of = None
    else:
        # momwire#1132: only the rows the solve reads are ever allocated. The
        # square destination this replaced was fully resident for a 90-row
        # write (huge pages, column-major), 59 % of the 150-radial peak.
        Z, held = solver._compute_Z_operator(
            geom, supp_seg, polys, rows=rows, compact=True
        )
        want = np.union1d(sectors[0], axial)
        if not np.array_equal(held, want):
            raise AssertionError(
                "the compact fill holds a different row set than the sector "
                f"solve reads ({held.size} rows against {want.size})"
            )
        row_of = np.full(n_basis_total, -1, dtype=np.int64)
        row_of[held] = np.arange(held.size, dtype=np.int64)
    return Z, row_of


def compute_impedance(solver):
    """`compute_impedance` on the sector route (momwire#1029).

    One fill of sector 0's rows plus the axial rows against EVERY source,
    one (m + p) solve whatever N is, and the coefficients reassembled in
    the dense ordering — so `element_currents` and the far readout need no
    new code, and the caller cannot tell which route produced the vector.
    """
    geom = solver._build_geometry()
    supp_seg, polys, kcl_A, wire_knots, wire_basis_global = (
        solver._build_basis_polynomials(geom)
    )
    n_basis_total = supp_seg.shape[0]
    sectors, axial = dof_groups(solver, wire_basis_global, n_basis_total)
    # The drive is built BEFORE the fill so its refusal lands before one
    # (momwire#1029 phase 2b). Neither reads the other, and the fill is the
    # expensive half, so this is the order that makes every refusal on the
    # route cost milliseconds — on a sweep as much as on one solve.
    v, port_vectors, _vpf_T, all_voltages, kcl_con = solver._feed_drive_and_readout(
        geom, wire_knots, wire_basis_global, n_basis_total, kcl_A
    )
    # Members drawn backwards (momwire#1897) are solved in the signed bases
    # `dof_signs` describes; None, and nothing below is touched.
    D = dof_signs(solver, wire_basis_global, n_basis_total)
    if D is not None:
        v = D * v
        kcl_con = kcl_con * D[None, :]
    # An off-axis junction's N rows become their one invariant orbit sum
    # (momwire#1897); a deck with every junction on the axis gets its own
    # matrix back unchanged.
    kcl_con = orbit_constraints(kcl_con, sectors)
    check_drive(v, kcl_con, sectors)
    solver._checkpoint()  # after geometry/basis/drive, before the one-sector fill
    Z, row_of = sector_fill(
        solver, geom, supp_seg, polys, sectors, axial, n_basis_total
    )
    if D is not None:
        Z *= D[None, :]
    solver._checkpoint()  # before the harmonic-0 solve
    coeffs = solve(solver, Z, v, kcl_con, sectors, axial, row_of=row_of)
    del Z
    if D is not None:
        coeffs = D * coeffs
    return solver._per_feed_z(coeffs, port_vectors, all_voltages), coeffs


def compute_impedance_swept(solver, k_array, z_out):
    """`compute_impedance_swept` on the sector route (momwire#1029 phase 2b).

    A LOOP, and that is the honest shape rather than a shortcut: the route's
    fill is `_compute_Z_operator` under `rows=`, which has no k axis to
    batch over, so the sweep is one sector fill and one (m + p) solve per
    frequency. On a buried deck nothing is given up by looping — the dense
    sweep is a loop there too (`_swept_batched_available` needs
    `ground_eps is None`). On a free-space or PEC screen (momwire#1131) the
    dense sweep CAN batch, and the route still loops: its per-k fill is the
    (m + p)-row one, which is the saving the route exists for.

    THE CONTRACT IS THE DENSE PATH'S, not a second spelling of it: `z_out` is
    the caller's own allocation — (n_k,) for one port, (n_k, n_ports)
    otherwise, complex128 — and this fills it through the same `z_out[i] = z`
    line. So does the frequency triple: `_k_restored` puts k, omega and
    wavelength back the way the dense sweep does, including on a refusal part
    way through.

    No `same_edge_prep`. `_compute_Z_operator` ignores the hoist on its
    buried branch, so building it would be work no fill can read — measured
    as pure cost on the dense sweep of a buried deck too, which is a separate
    finding and not this route's to fix. The above-ground `rows=` fill
    (momwire#1131) would read one; it is not built here yet, so an
    above-ground route sweep rebuilds each same-edge block per k.
    """
    with solver._k_restored():
        for i, kk in enumerate(k_array):
            solver._checkpoint()  # top of each frequency iteration
            solver._set_k(kk)
            z, _ = compute_impedance(solver)
            z_out[i] = z
    return z_out


def compute_port_solution(solver):
    """`compute_port_solution` on the sector route (momwire#1897).

    Every port solved from ONE row-compact fill, as the dense entry does from
    one square one: `(Y, X, pieces)` with `Y = B^T X` and `X` in the dense
    ordering, so the caller wraps them in the same `PortSolution` the dense
    path returns and nothing downstream (the readout, a load stamped on a
    port, `element_currents`) can tell the routes apart. `pieces` is the
    basis build the wrapper needs.

    This is what a loaded deck needs: its loads are ports, stamped onto Y by
    the consumer's port algebra, so a deck with N identical loaded posts is
    a 1 + N port solve whose posts are off the axis. Condition 4 has already
    held the port set to the rotation, and `solve_ports` takes each column
    through the harmonics it needs.
    """
    geom = solver._build_geometry()
    supp_seg, polys, kcl_A, wire_knots, wire_basis_global = (
        solver._build_basis_polynomials(geom)
    )
    n_basis_total = supp_seg.shape[0]
    sectors, axial = dof_groups(solver, wire_basis_global, n_basis_total)
    B, kcl_con = solver._port_columns(
        geom, wire_knots, wire_basis_global, n_basis_total, kcl_A
    )
    D = dof_signs(solver, wire_basis_global, n_basis_total)
    Bt, kcl_t = (B, kcl_con) if D is None else (D[:, None] * B, kcl_con * D[None, :])
    harmonics = harmonics_needed(Bt, kcl_t, sectors)
    solver._checkpoint()  # after geometry/basis/ports, before the one-sector fill
    Z, row_of = sector_fill(
        solver, geom, supp_seg, polys, sectors, axial, n_basis_total
    )
    if D is not None:
        Z *= D[None, :]
    blocks = harmonic_blocks(solver, Z, sectors, axial, harmonics, row_of=row_of)
    del Z
    solver._checkpoint()  # before the harmonic solves
    X = solve_ports(blocks, Bt, kcl_t, sectors, axial)
    if D is not None:
        X = D[:, None] * X
    Y = _lapack.matmul(B.T, X)
    pieces = (geom, supp_seg, polys, wire_knots, wire_basis_global, n_basis_total)
    return Y, X, pieces
