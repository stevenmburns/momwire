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
    rep: list[int] = []
    reps: list[tuple[int, np.ndarray]] = []
    for i, p in enumerate(points):
        p = np.asarray(p, dtype=float)
        for j, q in reps:
            if float(np.linalg.norm(p - q)) <= tol:
                rep.append(j)
                break
        else:
            rep.append(i)
            reps.append((i, p))
    return rep


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
