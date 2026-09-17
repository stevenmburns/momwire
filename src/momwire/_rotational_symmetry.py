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
  whose vertices share one vertical line (there must be exactly one such
  line: it is the axis).
* the **sectors** — everything else, in orbits of size N under rotation by
  2*pi/N about the axis.

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

import math
from typing import NamedTuple

import numpy as np

# 1e-9 of the deck's largest extent. The registered tolerance (§3.1): the
# structure probe measures rotated radials landing 3.9e-16 m (4 radials) to
# 6.4e-15 m (48) apart on 6.33 m arms, four to seven orders inside it.
TOL_REL = 1e-9

_WAY_OUT = "Drop rotational_symmetry=True to solve this deck densely."

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


def _refuse(sentence):
    return RotationalSymmetryRefused(f"rotational symmetry: {sentence} {_WAY_OUT}")


def _rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rho(pl, axis):
    return np.hypot(pl[:, 0] - axis[0], pl[:, 1] - axis[1])


def _arc_length(pl):
    return float(np.sum(np.linalg.norm(np.diff(pl, axis=0), axis=1)))


def _worst_z_angle(pl):
    """(degrees off z, of the worst edge). 0 for a purely vertical wire."""
    d = np.diff(pl, axis=0)
    n = np.linalg.norm(d, axis=1)
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


def _axis_and_seeds(polylines, junctions, tol):
    """The axis, and the wires that lie ON it. Raises condition 1's
    out-of-scope forms: no axial wire at all, or more than one axis."""
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
    axis = np.mean([c for _w, c in seeds], axis=0)
    for w, centre in seeds:
        if float(np.linalg.norm(centre - axis)) > tol:
            raise _refuse(
                f"wire {w} lies on a vertical line at "
                f"({centre[0]:.3f}, {centre[1]:.3f}), not on the deck's axis "
                f"({axis[0]:.3f}, {axis[1]:.3f}). There must be exactly one "
                f"axis."
            )
    return (float(axis[0]), float(axis[1])), tuple(sorted(w for w, _c in seeds))


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


def _match_under(polylines, wires, axis, theta, tol):
    """{w: its image} when rotation by `theta` maps `wires` onto itself, else
    None. Vertex for vertex, in the same vertex order (§3.2)."""
    R = _rot(theta)
    centre = np.array([axis[0], axis[1], 0.0])
    image = {}
    taken = set()
    for w in wires:
        a = (polylines[w] - centre) @ R.T + centre
        best = None
        for u in wires:
            b = polylines[u]
            if b.shape != a.shape:
                continue
            d = float(np.max(np.linalg.norm(a - b, axis=1)))
            if d <= tol and u not in taken:
                best = u
                break
        if best is None:
            return None
        image[w] = best
        taken.add(best)
    return image


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
    """`(N, sectors)` when the off-axis wires ARE a screen, else None.

    N is the LARGEST divisor of the off-axis wire count whose rotation maps
    them onto themselves: the symmetry group is cyclic, so a rotation by
    2*pi/n is a symmetry exactly when n divides the true sector count, and the
    largest such n IS that count.
    """
    if len(off) < 2:
        return None
    for n in sorted(
        (d for d in range(2, len(off) + 1) if len(off) % d == 0), reverse=True
    ):
        image = _match_under(polylines, off, axis, 2 * math.pi / n, tol)
        if image is None:
            continue
        cycles = _orbits(image, n)
        if cycles is None:
            continue
        return n, tuple(tuple(cyc[s] for cyc in cycles) for s in range(n))
    return None


def _check_mesh_and_material(solver, sectors, polylines, axis, tol):
    """§3.3: the same mesh, the same metal, the same jacket, and junction
    membership that rotates with the wire."""
    n = len(sectors)
    of_end, sizes = _junction_of_end(solver.junctions, len(polylines))
    a_cond = solver._conductor_radius_per_wire
    per_wire_specs = (
        ("conductivity", solver.wire_conductivity, "S/m", 1.0),
        ("jacket radius", solver.insulation_radius, "mm", 1e3),
        ("jacket eps_r", solver.insulation_eps_r, "", 1.0),
    )
    for c in range(len(sectors[0])):
        w0 = sectors[0][c]
        for s in range(1, n):
            w = sectors[s][c]
            if list(solver.n_per_edge_per_wire[w]) != list(
                solver.n_per_edge_per_wire[w0]
            ):
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
                j0, j = of_end.get((w0, end)), of_end.get((w, end))
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
        seg = np.linalg.norm(np.diff(pl, axis=0), axis=1)
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
    axis, seeds = _axis_and_seeds(polylines, solver.junctions, tol)
    every = set(range(len(polylines)))
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
        axial, (n, sectors) = seeds, found
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

    # 3. identical discretisation and material, junctions that rotate
    _check_mesh_and_material(solver, sectors, polylines, axis, tol)

    # 4. every port on the axis — a multiport solve drives each in turn, and
    #    an off-axis drive excites every harmonic
    for name, point in _port_sites(solver, polylines):
        rho = math.hypot(point[0] - axis[0], point[1] - axis[1])
        if rho > tol:
            raise _refuse(
                f"port '{name}' sits at "
                f"({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}), off the "
                f"symmetry axis ({axis[0]:.3f}, {axis[1]:.3f}). An off-axis "
                f"drive excites every harmonic, and this route serves the "
                f"axis-symmetric drive only."
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
    )
