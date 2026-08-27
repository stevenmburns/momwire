"""Where a named arclength lands on a family's own grid, and the one case
where the answer does not exist (stevenmburns/momwire#623).

Three call sites resolve a site by snapping to the nearest point of their own
grid — segment centres in `sinusoidal.py` and `pulse.py`, knots in
`razor.py`'s `_snap_to_knot`, which serves feeds AND lumped loads
(momwire#427). Snapping is the feature: a caller who names 3.7 m on a mesh
with no port at 3.7 m wants the nearest port and gets it.

The case this module is about is narrower — a target sitting EXACTLY BETWEEN
two grid points. There the family cannot represent what was asked and cannot
choose either, and the candidates are not close: the port moves half a cell,
worth 8-24 % of the driving-point impedance on an asymmetric deck
(`tests/test_feed_snap_623.py` measures it per family). Which one wins is
then decided by the last bits of the `cumsum` that built the grid — on an
equal-armed inverted-V ladder the margin between the two candidate distances
runs 48 to 19 200 ULPs and its SIGN flips with N, so the feed walks across
the apex as the mesh refines.

It is not an exotic request. The midpoint of a symmetric wire is a knot
whenever the segment count is even, so `feed_arclength=None` on an even-count
mesh IS this case for every centre-snapping family. antennaknobs never
reaches it (`engines/momwire.py:_parity_for_solver` coerces the mesh per
solver); nothing makes a direct caller do that.

## What step 1 measured, and what it settles

Tapping every snap the quick suite makes: **530 of 3341 are exact ties**, in
99 distinct (family, arclength, site-count, target) shapes, across five
families — `SinusoidalSolver`, its Galerkin subclass, `RazorSolver`,
`PulseSolver` and `HarringtonSolver`. 523 of the 530 are on the WRONG
PARITY for their family, which is the mechanism stated as a count.

So a refusal is **not** free, and the earlier guess that it might be is dead:
it would break a sixth of the snaps this tree's own tests make. The suite is
green through all 530 because those decks are symmetric about the site, where
the two candidates are mirror images — the same thing that hid #623 from
#421's ladder in the first place.

What step 2 now has to answer is therefore not "warn or refuse" but: **of the
ties, how many are on decks where the choice changes the answer?** That needs
each tie re-solved from the other side, which this module's tap makes
possible and which is deliberately not done here.

The warning is a DIAGNOSTIC, not a contract: it fires only under
`MOMWIRE_623_TALLY`, because a warning that cries on 530 harmless symmetric
decks is a worse warning than none, and turning it on by default would be
step 2's decision made quietly rather than measured.
"""

from __future__ import annotations

import atexit
import json
import os
import warnings

import numpy as np


class AmbiguousSite(UserWarning):
    """A named arclength that falls between two of a family's grid points."""


# Fraction of the wire's arclength within which the two candidate distances
# count as equal. DERIVED, the way momwire#578's dust bars were, from the void
# the corpus itself leaves. Over the 3341 snaps the quick suite makes:
#
#     largest margin that is rounding     1.301e-14   (the 530 ties)
#     smallest margin that is a request   1.248e-03   (the other 2791)
#
# — eleven empty decades. The bar sits three decades above the rounding and
# eight below the smallest real separation. It is deliberately kept under
# 2e-10, which is where `tests/test_feed_snap_623.py` straddles a tie on
# purpose with a 1 nm nudge on a 10 m wire; that test is asking for both
# sides and should not be told they are ambiguous.
AMBIGUITY_TOL_FRAC = 1e-11

_TAP = "MOMWIRE_623_TALLY"


def snap(grid, target, *, total_arc, family, what="feed", wire=None):
    """``(pick, margin)`` — the nearest grid point, and by how much it won.

    `grid` is the family's own site arclengths along one wire; `margin` is the
    gap between the best and second-best distances in metres, `inf` when the
    grid holds one point. The pick is `argmin` and is NEVER changed by this
    module — it reports, and #623 step 2 decides what the contract does.
    """
    arcs = np.asarray(grid, dtype=float)
    if arcs.size == 0:
        raise ValueError(f"{family}: no {what} sites on wire {wire}")
    dist = np.abs(arcs - target)
    pick = int(np.argmin(dist))
    if arcs.size == 1:
        return pick, float("inf")
    best, second = np.partition(dist, 1)[:2]
    margin = float(second - best)
    if os.environ.get(_TAP):
        _record(family, what, wire, target, margin, total_arc, arcs.size)
        if margin <= AMBIGUITY_TOL_FRAC * total_arc:
            where = "" if wire is None else f" on wire {wire}"
            warnings.warn(
                f"{family}: the {what} arclength {target:.12g}{where} falls "
                f"between two sites this basis can carry. The pick is decided "
                f"by rounding in the grid, and the two answers put the site "
                f"half a cell apart. See stevenmburns/momwire#623.",
                AmbiguousSite,
                stacklevel=3,
            )
    return pick, margin


# --------------------------------------------------------------------------
# Diagnostic tap, modelled on `MOMWIRE_532_DUMP` and `MOMWIRE_403_DUMP`.
#
# Point `MOMWIRE_623_TALLY` at a path and every snap drops one JSON line
# there. The margin as a FRACTION of the arc is the axis step 2 has to reason
# on, and it is not visible from the warning alone: the warning fires on the
# ties, and the derivation needs everything that did NOT fire to know there is
# a void between them. One file per process — the suite runs under xdist and a
# shared handle would interleave.
# --------------------------------------------------------------------------
_TALLY: list[dict] = []


def _record(family, what, wire, target, margin, total_arc, sites):
    _TALLY.append(
        {
            "family": family,
            "what": what,
            "wire": wire,
            "target": target,
            "margin": margin,
            "total_arc": total_arc,
            "frac": (margin / total_arc) if total_arc else None,
            "sites": int(sites),
        }
    )


@atexit.register
def _flush_tally():
    root = os.environ.get(_TAP)
    if not root or not _TALLY:
        return
    with open(f"{root}.{os.getpid()}.jsonl", "w") as handle:
        for row in _TALLY:
            handle.write(json.dumps(row) + "\n")
