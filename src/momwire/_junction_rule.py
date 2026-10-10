"""The one rule for deciding which wire ends are the same node.

Two solvers disagreeing about the same deck's CONNECTIVITY is a worse failure
than either answer being wrong on its own, so the rule lives here once and
every caller walks it rather than re-spelling it. `HarringtonSolver` used to
carry a hand-written copy with a comment saying exactly that; this module is
that comment taken at its word (momwire#590 step 1).

The rule, in full:

    An end joins the FIRST existing group whose REPRESENTATIVE point it lies
    within `tol` of, or else starts its own group.

Comparing against representatives rather than against every member is what
makes the rule **non-transitive**, and that is load-bearing, not an accident
of implementation: with A~B, B~C but A≁C, a transitive union-find merges all
three, where this gives {A, B} and a separate {C}. Changing it would silently
re-wire decks that sit in that window.

`JUNCTION_TOL` is an absolute 1e-9 m under the Euclidean norm. It agrees with
nothing else in the tree, and in particular NOT with the deck front end, which
fuses span endpoints onto a `deck/_polylines._NODE_EPS` = 1e-6 m grid — a
thousand times looser, a different algorithm (grid quantisation, not first
match) and a different norm (per-coordinate rounding). What keeps the gap from
biting is the deck's own invariant, not an agreement between the numbers: by
the time a model exists its coincident ends are already exactly equal, so the
coarse grid absorbs transform ulps rather than deciding connectivity. A caller
assembling geometry by hand, or a future front end that relaxes that
invariant, lands in the window — which is why `HarringtonSolver` refuses ends
that fall in it rather than quietly disconnecting them.

Unifying the tree's several "same point" tolerances is a deliberate future
decision and is NOT what this module does. It moves one rule to one place at
its existing value; it changes no number anywhere.
"""

from __future__ import annotations

import numpy as np

# Two wire endpoints this close are a junction, not a coincidence.
#
# The comment that used to sit on this constant claimed it was "the same
# tolerance the caller-facing geometry helpers use for 'same point'". That was
# false — momwire#429 correction 2 caught it, and the module docstring above
# says what is actually true.
JUNCTION_TOL = 1e-9


def coincident_groups(points, tol: float = JUNCTION_TOL) -> list[int]:
    """Group points by the rule above; return each one's representative INDEX.

    `points` is a sequence of (3,) coordinates in the caller's own order —
    that order is the rule's input, since "first existing group" is defined by
    it. Returns a list `rep` with `rep[i]` the index of the point that
    represents `i`'s group. A point that starts its own group is its own
    representative, so `rep[i] == i`, and `rep[i] <= i` always.

    Callers key their own labels off the result: bucket by `rep[i]` to get
    groups (dict insertion order reproduces group-creation order, because a
    representative `r` is first seen at `i == r`), or assign `label[rep[i]]`
    to merge node ids.
    """
    pts = [np.asarray(p, dtype=float) for p in points]
    cells = _cells(pts, tol)
    if cells is None:
        return _first_match_walk(pts, tol)
    # momwire#1420: the same rule, asked of the representatives NEAR each
    # point instead of all of them. Points are binned on a grid of side
    # 2 * tol; any representative within `tol` of a point lies in the point's
    # own cell or one of its 26 neighbours (`_cells` says why, and when it
    # declines to bin). The candidates are then tried in ascending index —
    # which is creation order, since a representative's index is where its
    # group started — with the walk's own distance expression, so the first
    # one that passes is the one the walk would have stopped at. Memory is
    # O(P) and time O(P) for well-spread points, where the walk is
    # O(P * groups): a deck of thousands of wire ends used to pay millions
    # of `norm` calls here.
    rep: list[int] = []
    buckets: dict[tuple[int, int, int], list[int]] = {}
    get = buckets.get
    for i, (cx, cy, cz) in enumerate(cells):
        candidates: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    found = get((cx + dx, cy + dy, cz + dz))
                    if found:
                        candidates += found
        hit = -1
        if candidates:
            candidates.sort()
            p = pts[i]
            for j in candidates:
                if float(np.linalg.norm(p - pts[j])) <= tol:
                    hit = j
                    break
        if hit < 0:
            rep.append(i)
            buckets.setdefault((cx, cy, cz), []).append(i)
        else:
            rep.append(hit)
    return rep


# Bin indices stay exact, and two points within `tol` of each other stay in
# adjacent bins, while |coordinate| / (2 * tol) stays below 2**48: each
# quotient then carries under 2**48 * 2**-53 = 1/32 of a bin of rounding, so
# two of them differ by at most 1/2 + 1/16 bins and their floors by at most
# one. Past it (|x| > 560 km at the 1e-9 m default) the walk answers.
_MAX_CELL = 2.0**48


def _cells(pts, tol) -> list | None:
    """Each point's grid cell for :func:`coincident_groups`, or ``None`` when
    the grid cannot promise the walk's answer and the walk must run instead.

    The promise: if the walk's ``norm(p - q) <= tol`` holds then every
    coordinate differs by at most ``tol`` (to a few ulps, since a norm is
    never below one of its components by more than rounding), so on a grid
    of side ``2 * tol`` the two cells differ by at most one in each axis —
    with room to spare for the rounding of ``x / (2 * tol)`` (``_MAX_CELL``). It is declined for a non-finite or
    non-positive ``tol``, a non-finite coordinate (NaN never matches in the
    walk, which the walk itself then spells), points that are not 3-vectors,
    and coordinates past ``_MAX_CELL`` bins from the origin.
    """
    if not pts:
        return []
    if not (np.isfinite(tol) and tol > 0.0):
        return None
    if any(p.shape != (3,) for p in pts):
        return None
    q = np.floor(np.array(pts) / (2.0 * tol))
    if not np.all(np.abs(q) < _MAX_CELL):  # also False for NaN and inf
        return None
    return q.astype(np.int64).tolist()


def _first_match_walk(pts, tol) -> list[int]:
    """The rule as a walk over every representative, the spelling
    :func:`coincident_groups` used before momwire#1420 and still uses
    wherever the grid declines (:func:`_cells`)."""
    rep: list[int] = []
    reps: list[tuple[int, np.ndarray]] = []
    for i, p in enumerate(pts):
        for j, q in reps:
            if float(np.linalg.norm(p - q)) <= tol:
                rep.append(j)
                break
        else:
            rep.append(i)
            reps.append((i, p))
    return rep


def coincident_end_groups(wires_polylines, tol: float = JUNCTION_TOL) -> list[list]:
    """Groups of TWO OR MORE coincident wire ends, as `(wire, end)` labels.

    The question "does this geometry contain a junction?", asked of bare
    polylines. Lone ends are dropped, so an empty result means every wire end
    stands alone and there is no connectivity to declare.

    Ends are enumerated in razor's order — first wire first, `start` before
    `end` within a wire — so a group here is spelled exactly as the
    corresponding `junctions=` entry would be, and can be handed straight back
    to the caller in a message.
    """
    labels, points = [], []
    for i, pl in enumerate(wires_polylines):
        labels.append((i, "start"))
        points.append(pl[0])
        labels.append((i, "end"))
        points.append(pl[-1])
    return [g for g in grouped(labels, points, tol) if len(g) >= 2]


def canonical_groups(declared) -> list[list]:
    """A caller's `junctions=` put into the order detection would have used.

    Ends sort first-wire-first with `start` before `end`, and groups sort by
    their earliest member — which is exactly the order `coincident_end_groups`
    produces, because a group's representative is its earliest end.

    That equivalence is the point: declaring the junctions a geometry already
    has must give byte-identical output to letting them be inferred, so the
    override is a way to DISAGREE with the geometry (usually by declaring
    fewer), never an accidental change of answer when you agree with it.
    """
    key = lambda e: (int(e[0]), 0 if e[1] == "start" else 1)  # noqa: E731
    groups = [sorted((int(w), str(end)) for w, end in g) for g in declared]
    groups = [sorted(g, key=key) for g in groups]
    return sorted(groups, key=lambda g: key(g[0]) if g else (-1, -1))


def grouped(labels, points, tol: float = JUNCTION_TOL) -> list[list]:
    """`coincident_groups` bucketed into label groups, in creation order.

    The convenience shape for callers that want the groups themselves rather
    than a representative map. `labels[i]` belongs to `points[i]`.
    """
    rep = coincident_groups(points, tol)
    out: dict[int, list] = {}
    for i, label in enumerate(labels):
        out.setdefault(rep[i], []).append(label)
    return list(out.values())
