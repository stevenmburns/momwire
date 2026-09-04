"""Which axis values cannot be combined, and momwire's own reason for each.

`_capabilities.AXIS_VALUES` says what values an axis has; each solver's row
says which of them that class can be configured to. Neither says that some
combinations are refused — and they are, which is what a panel rendering the
axes as independent controls would get wrong. antennaknobs#1006 asks "which
cells of the product space are reachable, and why not the rest"; this is the
answer, as data.

WHY NOT IN `_capabilities.py`. That module's contract is one NamedTuple plus
one method and no machinery, and half these reasons live on solver classes
(`HMatrixSolver.capabilities.refusals["buried"]`) — importing those from
`_capabilities` is a cycle, since every solver module imports it. So this is
its own module that imports the solvers, with nothing importing it back.

PROSE IS REFERENCED, NEVER RETYPED. Every `reason` below is the identical
string object the refusal itself raises. That is the whole point: a coupling
whose description drifts from its refusal is worse than one that is
undocumented, because it reads as authoritative. Two of the five needed their
literal hoisted out of the `raise` first; three were already module constants.

`inspect.signature` IS NOT THE CHECK for whether a class takes a keyword —
kwargs reach the parent, so the signature reports `SinusoidalGalerkinSolver`
as not taking `extended_kernel` while it plainly does. Construct the cell;
that is what `applies_to` records and what the gate verifies.

THE TABLE IS THE INVENTORY, NOT THE PANEL'S MENU. Two entries name a
non-axis keyword (`near_correction`, and a stepped radius at a junction), so
nothing can render them as a greyed-out cell today. They are here anyway,
marked `b_is_axis=False`, because a table trimmed to what the UI can draw
would be the panel's limitation published as the engine's.
"""

from __future__ import annotations

from typing import NamedTuple

from ._ground_spec import CONTACT_UNDER_REFL_COEF_REFUSAL
from .bspline import (
    _BURIED_EXTENDED_KERNEL_REFUSAL,
    _ENRICHMENT_EXTENDED_KERNEL_REFUSAL,
    _ENRICHMENT_PER_WIRE_RADIUS_REFUSAL,
    _ENRICHMENT_WIRE_LOADING_REFUSAL,
)
from .hmatrix import HMatrixSolver
from .sinusoidal import _POINT_FEED_MODEL_REFUSAL
from .sinusoidal_galerkin import (
    _EK_NEAR_CORRECTION_REFUSAL,
    _EK_STEPPED_RADIUS_JUNCTION_REFUSAL,
)


class Coupling(NamedTuple):
    """One refused combination of axis values.

    `axis_a`/`value_a` is the choice; `axis_b`/`value_b` is what that choice
    puts out of reach. The direction is how a panel reads it — pick A, lose B —
    and not a claim about causation: several of these are symmetric in the
    formulation and asymmetric only in how a user meets them.

    `b_is_axis` is False when `axis_b` names a constructor keyword rather than
    a compositional axis, so a consumer can skip what it cannot render without
    keeping a second list of exceptions. `a_is_axis` says the same about the
    A side, and it exists because the original shape assumed A was ALWAYS an
    axis — true of the first six rows and false of two added in momwire#888,
    where NEITHER side is compositional (`per_wire_radius`, `wire_loading`).
    Those are still inventory: the table records what momwire refuses, and
    which of it a panel can draw is the panel's question, not this module's.

    `condition` is None for a flat refusal and a short phrase when the
    combination is refused only in a narrower case — carried verbatim to the
    consumer, because "refused" and "refused when X" are different sentences
    and collapsing them overstates the first.

    `issue` is None when the refusal's own prose cites no issue — the
    `wire_loading+singular_enrichment` row is the case. Inventing a plausible
    citation there would be the same drift this table exists to stop, one
    field over: a reader would follow the number to an issue that never
    discussed the refusal.

    `applies_to` names the solver class(es) that actually raise this. It is
    NOT decoration: a coupling is per-class, and a consumer filtering by "can
    this backend be configured to `value_a`" mis-attributes three of the six
    rows below — it would tell a `bspline` user that the extended kernel
    forbids `near_correction=False`, a keyword `BSplineSolver` does not have
    (measured: TypeError, not a refusal). Filter on this field, never on
    `value_a` reachability.
    """

    axis_a: str
    value_a: str
    axis_b: str
    value_b: str
    reason: str
    issue: str | None
    a_is_axis: bool = True
    b_is_axis: bool = True
    condition: str | None = None
    applies_to: tuple[str, ...] = ()


COUPLINGS: tuple[Coupling, ...] = (
    # Point matching cannot take a zero-width gap: the drive is E_app sampled
    # AT a match point and the source is a delta there, so the pairing is
    # undefined. This is why `SinusoidalSolver` and `SinusoidalGalerkinSolver`
    # differ in TWO axes rather than the one the pair exists to isolate — the
    # feed model is dragged along by the testing.
    Coupling(
        axis_a="testing",
        value_a="point-matching",
        axis_b="feed_model",
        value_b="point-gap",
        reason=_POINT_FEED_MODEL_REFUSAL,
        issue="momwire#212",
        # The point-matched class only. Its Galerkin subclass SERVES the point
        # gap — that is the whole reason the pair exists.
        applies_to=("SinusoidalSolver",),
    ),
    # An accelerated assembly has no per-segment medium, so choosing it gives
    # up buried geometry. Two entries rather than one with a value list: the
    # schema is one value per side, and `ArrayBlockSolver` inherits the
    # refusal through `HMatrixSolver` but is a different cell of the space.
    Coupling(
        axis_a="solve_strategy",
        value_a="aca",
        axis_b="wire_position",
        value_b="buried",
        reason=HMatrixSolver.capabilities.refusals["buried"],
        issue="momwire#553",
        applies_to=("HMatrixSolver",),
    ),
    Coupling(
        axis_a="solve_strategy",
        value_a="element-block",
        axis_b="wire_position",
        value_b="buried",
        reason=HMatrixSolver.capabilities.refusals["buried"],
        issue="momwire#553",
        # The two rows exist BECAUSE the cells differ; naming both classes on
        # each would undo that.
        applies_to=("ArrayBlockSolver",),
    ),
    # The extended kernel's eligibility is a coaxial-and-equal-radius grouping
    # scored across the whole geometry, and nobody has measured what that
    # means for a pair spanning two media.
    Coupling(
        axis_a="kernel",
        value_a="extended",
        axis_b="wire_position",
        value_b="buried",
        reason=_BURIED_EXTENDED_KERNEL_REFUSAL,
        issue="momwire#553",
        applies_to=("BSplineSolver",),
    ),
    # Not an axis pair: `near_correction` is a constructor keyword. Kept
    # because it is a real refused combination and the inventory is the point.
    Coupling(
        axis_a="kernel",
        value_a="extended",
        axis_b="near_correction",
        value_b="False",
        reason=_EK_NEAR_CORRECTION_REFUSAL,
        issue="momwire#246",
        b_is_axis=False,
        applies_to=("SinusoidalGalerkinSolver",),
    ),
    # Conditional, and the condition is load-bearing: uniform-radius junctions
    # are untouched, which is the overwhelmingly common case. Stating this one
    # flat would tell a user the extended kernel refuses junctions, which is
    # false and would send them to the wrong workaround.
    Coupling(
        axis_a="kernel",
        value_a="extended",
        axis_b="junction_ports",
        value_b="True",
        reason=_EK_STEPPED_RADIUS_JUNCTION_REFUSAL,
        issue="momwire#398",
        b_is_axis=False,
        condition="a radius step at the junction",
        applies_to=("SinusoidalGalerkinSolver",),
    ),
    # ---- singular enrichment, three rows (momwire#888) -------------------
    #
    # All three are BSplineSolver's alone: `use_singular_enrichment` is not a
    # keyword the other families take at all, so an unattributed row would
    # advise users about a control they do not have — the same
    # mis-attribution `applies_to` was introduced for.
    #
    # The first of these is the row antennaknobs was missing, and the cost of
    # its absence is on momwire#888: with no row to reference, that frontend
    # hand-wrote its own sentence, citing momwire#271 where the refusal below
    # cites momwire#249 follow-up C, and giving one reason where it gives
    # three. The copy was authoritative-looking and wrong about its own
    # source. That is the argument for this table, observed rather than
    # predicted.
    Coupling(
        axis_a="kernel",
        value_a="extended",
        axis_b="singular_enrichment",
        value_b="True",
        reason=_ENRICHMENT_EXTENDED_KERNEL_REFUSAL,
        issue="momwire#249",
        # Not an axis: `use_singular_enrichment` is a constructor keyword, so
        # no panel can draw this as a cell of the product space.
        b_is_axis=False,
        applies_to=("BSplineSolver",),
    ),
    Coupling(
        axis_a="per_wire_radius",
        value_a="True",
        # Neither side is an axis — see `a_is_axis`.
        axis_b="singular_enrichment",
        value_b="True",
        reason=_ENRICHMENT_PER_WIRE_RADIUS_REFUSAL,
        issue="momwire#147",
        a_is_axis=False,
        b_is_axis=False,
        applies_to=("BSplineSolver",),
    ),
    Coupling(
        axis_a="wire_loading",
        value_a="True",
        # Neither side is an axis — see `a_is_axis`.
        axis_b="singular_enrichment",
        value_b="True",
        reason=_ENRICHMENT_WIRE_LOADING_REFUSAL,
        # The refusal's prose cites nothing, so neither does this. See the
        # `issue` docstring: a plausible-looking number would be worse than
        # none, because it would be followed.
        issue=None,
        a_is_axis=False,
        b_is_axis=False,
        applies_to=("BSplineSolver",),
    ),
    # ---- ground contact under the reflection-coefficient model -----------
    #
    # The first row whose BOTH sides are DERIVED axes (`wire_position` and
    # `ground_model` are computed by `axes_for`, not declared), so it belongs
    # to a ground panel rather than a solver panel. Recorded here anyway: the
    # table is the inventory, and where a consumer draws it is the consumer's
    # question.
    #
    # `applies_to` MEASURED, not guessed, and the measurement changed the
    # answer twice. It is declared in four modules but reaches SIX classes
    # through inheritance (HMatrix and ArrayBlock get BSpline's). And it is
    # NOT universal, though all seven classes refuse the pair: HarringtonSolver
    # refuses `contact` OUTRIGHT, under every ground model, so for it this is a
    # single-cell refusal and not a coupling at all. Listing it here would tell
    # a `pulse` user the PAIRING is the problem and imply that contact over
    # Sommerfeld would work, which is false. The six below all serve `contact`
    # alone and refuse only this combination — verified by asking each class
    # for `refusal("contact")` and `refusal("contact", "sommerfeld")`.
    Coupling(
        axis_a="wire_position",
        value_a="contact",
        axis_b="ground_model",
        value_b="refl-coef",
        reason=CONTACT_UNDER_REFL_COEF_REFUSAL,
        issue="momwire#282",
        applies_to=(
            "BSplineSolver",
            "HMatrixSolver",
            "ArrayBlockSolver",
            "SinusoidalSolver",
            "SinusoidalGalerkinSolver",
            "RazorSolver",
        ),
    ),
)
