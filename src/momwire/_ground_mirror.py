"""The mirror map: the reflection through `z = ground_z`, once.

momwire#429 unit 3 (the sharing audit's rank 6). The reflection is
M = diag(1, 1, −1), applied to POSITIONS about the plane and to
DIRECTIONS about the origin, and those two are the whole of it — but the
audit counted **14 spellings across 6 files**, twelve of them the ad-hoc
literal `np.array([1.0, 1.0, -1.0])` and the rest `z → 2g − z` written
out at the call site. `_potential_ground.ImageGeometry` had already named
the two operations for the razor pilot (momwire#398 unit 2); this module
is that naming promoted to where every consumer can reach it, and
`ImageGeometry` now delegates to it.

**Why here and not in `_ground_spec`.** This sits BELOW `_ground_refl` —
`specular_pair_tables` reflects the source tangents itself, and was one
of the fourteen — whereas `_ground_spec` sits above it (it asks that
layer for ε̃). Two tiny modules, each with one import direction, rather
than one module with a cycle in it.

**The two operations are not interchangeable**, which is the whole reason
to name them. A position picks up the plane's offset and a direction does
not: `2g − z` versus `−z`. Both spellings existed in the tree and a site
whose arithmetic is neither was left alone — see the unit's report.

Nothing here is frequency-, basis- or formulation-dependent, so both
trunks import it and neither owns it. numpy is its only dependency.
"""

from __future__ import annotations

import numpy as np

# M = diag(1, 1, −1) as the row it multiplies a `(..., 3)` array by. One
# array, so every z-reflection in the tree is visibly the same reflection.
MIRROR = np.array([1.0, 1.0, -1.0])


def mirror_positions(positions, ground_z):
    """Reflect `(..., 3)` POINTS through the plane `z = ground_z`.

    Returns a fresh array; `positions` is never written through.
    """
    out = positions.copy()
    out[..., 2] = 2 * ground_z - out[..., 2]
    return out


def mirror_tangents(tangents):
    """Reflect `(..., 3)` DIRECTIONS: `t → (t_x, t_y, −t_z)`.

    The plane's offset does not enter — a direction is a difference of
    positions, and `2·ground_z` cancels — so this is a pure z-flip and
    there is deliberately no `ground_z` argument to get wrong.
    """
    return tangents * MIRROR
