"""razor-2p's far-mesh accuracy class, as a measured table and one advisory.

`RazorSolver`'s path (razor-blade) testing rule is NEC-5's rule, reproduced by
design — momwire#603 measures razor-2p equal to licensed NEC-5 at EQUAL mesh.
That agreement says the two share a convergence class; it does not say either
is converged. momwire#845 measured the class: **razor-2p is first order in the
far mesh on every deck family**, where `BSplineSolver` at degree 2 is already
converged at the same segment count.

So a razor-2p answer at a catalog-default mesh is systematically off, by an
amount nobody can see from the answer itself. This module states that once, at
construction.

WHY THERE IS NO TRIGGER. The obvious design — warn when the far segment length
puts a deck outside a stated class — was measured and does not work. Over the
antennaknobs catalog (100 decks, both grounds, at the shipped mesh and 2x/4x):

    corr(lambda/delta, |dZ|) = +0.021        corr(delta/a,   |dZ|) = +0.135
    corr(n_wires,      |dZ|) = -0.087        corr(Sigma-seg, |dZ|) = -0.056

Far segment length carries essentially NO information about the error it would
be guarding. 74 of those decks share lambda/delta in 80-90 while their |dZ|
runs 0.17 to 32.19 ohm — a 190x spread at one segment density. The cause is
benign and structural: antennaknobs' `auto_mesh` is a DENSITY (N segments per
quarter-wave), so it has already equalised lambda/delta across the catalog to
~83, and an equalised axis has nothing left in it to discriminate with. A
threshold there would look principled and fire at random with respect to
accuracy. No other solve-free property in the census predicts it either
(against relative error: lambda/delta +0.092, n_wires -0.066, |Z| -0.170), so
this advisory is unconditional rather than badly conditioned.

WHY THE NUMBERS ARE RELATIVE. #845's own tables are in ohms, and in ohms the
ranking is dominated by impedance magnitude rather than by accuracy —
corr(|Z|, |dZ|) = +0.69. `wire.lazy_h` is the catalog's worst deck at 32.19 ohm
and that is **0.63 %** of its 5092 ohm |Z|, while `loops.skyloop_lmatch` at
16.11 ohm is **33.6 %** of its 48 ohm. Ranked in ohms the 0.6 % deck leads and
the three ~33 % decks — skyloop_lmatch, verticals.rectangle, dipoles.koch_dipole,
all at ordinary ~48 ohm feedpoints — are invisible. A class stated in ohms
would therefore be unreadable on precisely the decks it matters for.

The figures live in `FAR_MESH_CLASS` rather than in the message text so a
re-sweep updates them without editing prose, and so a test can pin the message
to the row instead of to a literal.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import NamedTuple


class RazorFarMeshClass(UserWarning):
    """razor-2p is first order in the far mesh; its default-mesh answer is not
    converged. Advisory only — nothing is remeshed and nothing is refused."""


class FarMeshClass(NamedTuple):
    """One measured statement of razor-2p's far-mesh class.

    Relative figures are |dZ| against a converged `BSplineSolver` degree-2
    reference, as a percentage of |Z| at the same deck and mesh.
    """

    median_rel_pct: float  # median over the catalog at its shipped density
    tail_rel_pct: float  # worst deck measured
    median_abs_ohm: float  # the same median in ohms, for continuity with #845
    order: int  # 1 = halves per mesh doubling
    ground_spread_pct: float  # how far free and Sommerfeld medians differ
    n_decks: int  # decks carrying a reportable number
    issue: str
    provenance: str


# Measured 2026-09-03 on antennaknobs main fc0e5c68c with momwire 9eda56f, over
# the 100 razor-2p-served catalog decks (88 free / 83 Sommerfeld carry a
# reportable number; the rest are cost-guarded, refused, or lack a bounded
# reference). Medians halve per mesh doubling: 2.29 -> 1.27 -> 0.70 ohm.
FAR_MESH_CLASS = FarMeshClass(
    median_rel_pct=3.3,
    tail_rel_pct=34.0,
    median_abs_ohm=2.29,
    order=1,
    ground_spread_pct=2.0,
    n_decks=88,
    issue="stevenmburns/momwire#845",
    provenance="antennaknobs scratch/845-mesh-policy, commit 6bafd991d",
)

# Point the warning at the caller's construction site rather than inside this
# package, exactly as `_crossing_fill` does for `CoarseCrossingNode`.
_WARN_TARGET = (
    {"skip_file_prefixes": (os.path.dirname(os.path.abspath(__file__)),)}
    if sys.version_info >= (3, 12)
    else {"stacklevel": 4}
)


def far_mesh_class_message(cls: FarMeshClass = FAR_MESH_CLASS) -> str:
    """The advisory text, composed from the measured row.

    Separate from the warning so a test can pin the message to
    `FAR_MESH_CLASS` rather than to a literal, and so a caller that wants the
    sentence for a report can have it without raising a warning.
    """
    return (
        f"razor-2p is first order in the far mesh: its path (razor-blade) "
        f"testing rule is NEC-5's, which converges more slowly than a "
        f"Galerkin-tested basis, so a coarse answer is systematically off "
        f"rather than noisy. Measured over {cls.n_decks} antennaknobs catalog "
        f"decks at their shipped mesh density, the driving-point impedance is "
        f"a median {cls.median_rel_pct:.1f} % from converged, with a tail to "
        f"{cls.tail_rel_pct:.0f} % ({cls.median_abs_ohm:.2f} ohm median in "
        f"absolute terms, though the relative figure is the readable one: in "
        f"ohms the ranking follows |Z| rather than accuracy). The ground "
        f"barely matters — free-space and Sommerfeld medians agree to "
        f"{cls.ground_spread_pct:.0f} %. It is order {cls.order} in the mesh, "
        f"so DOUBLING the segment count on the razor path roughly HALVES the "
        f"error; there is no mesh at which it is free. For a converged answer "
        f"use BSplineSolver (degree 2), which is already converged at these "
        f"same segment counts. Advisory: nothing is remeshed and nothing is "
        f"refused — a coarse mesh is a legitimate thing to ask for, and every "
        f"rung of a convergence ladder but the last is one. "
        f"See {cls.issue} ({cls.provenance})."
    )


def warn_far_mesh_class(cls: FarMeshClass = FAR_MESH_CLASS) -> str:
    """Emit the advisory once, and return the text that was emitted.

    Returning it is deliberate: it lets a test read back exactly what a caller
    saw, and it is the seam a diagnostic or a report generator uses without
    having to catch a warning.

    UNCONDITIONAL by measurement, not by laziness — see the module docstring:
    every solve-free predictor of the error was measured and none correlates
    with it, so there is no honest threshold to put here.
    """
    text = far_mesh_class_message(cls)
    warnings.warn(text, RazorFarMeshClass, **_WARN_TARGET)
    return text
