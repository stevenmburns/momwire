"""The ground SPEC layer: what ground is this, shared by both trunks.

momwire#429 unit 2 (the sharing audit's rank 5). `_potential_ground` and
`_field_ground` each opened with the same 37-line decision tree — free
space, PEC, the reflection-coefficient fold, the Sommerfeld composition —
82 % of it literally identical, differing only in which object the branch
constructed. The audit's finding is that the DECISION is not a piece of
either formulation: it reads four solver strings and answers three
numbers, and the two trunks then build their own objects out of that
answer. So the decision lives here, once, and each factory keeps only its
trunk-specific construction.

**Where this sits.** Above `_ground_refl` (it asks that layer for ε̃(ω))
and below both trunk modules, which import it and not each other. That
one-directional shape is the point of the extraction: the field trunk
must never acquire the potential trunk's Sommerfeld-grid and quadrature
dependencies to learn what ground it is filling, nor the reverse. The
mirror map is deliberately NOT here — it sits BELOW `_ground_refl`, which
reflects tangents itself, and so has its own module (`_ground_mirror`).

**Why the two "is this Sommerfeld?" tests were already the same test.**
`_field_ground` asked `solver._sommerfeld_ground()`, which is
`ground_z is not None and ground_eps is not None and ground_model ==
"sommerfeld"`; `_potential_ground` asked `ground_model == "sommerfeld"`.
Inside the branch both leading conjuncts are already established, so the
two agree entry for entry, and `ground_config` spells the shorter one.
`_sommerfeld_ground` survives for `SinusoidalSolver._fill_row_bytes`,
which counts blocks with it — residency arithmetic, not physics.

**What the fifth ground buys with this.** The radial-wire screen
(architecture doc §6.1) is a `weighted=True, standard_fresnel=False` row
plus a modified coefficient function in `_ground_refl`. Before this unit
it had to be added to two factories; now it is one row here and both
trunks see it, which is the audit's "the screen lands once".
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from . import _ground_refl


class GroundConfig(NamedTuple):
    """One solve's ground decision, in the vocabulary both trunks share.

    `mode` — `"fold"` (PEC, refl-coef, and the screen when it lands) or
    `"compose"` (sommerfeld). The two trunks' `mode` docstrings spell out
    what the distinction costs in float64; the DECISION is this one.

    `eps_tilde` — the complex relative permittivity ε̃(ω), or `None` over
    PEC (and the tell for "PEC" on both trunks). Hoisted here so a banded
    or chunked fill pays `_ground_refl.eps_tilde` once per fill rather
    than once per band — same value either way, it is a pure function of
    arguments the fill holds fixed.

    `image_coefficient` — 1.0, or C₂ = (ε̃−1)/(ε̃+1) for sommerfeld. How it
    is APPLIED is per trunk and deliberately different (the field trunk
    multiplies a finished block under a measured operand-order contract;
    the potential trunk folds it into the constant weight tables), which
    is exactly why the value is decided here and the application is not.

    `standard_fresnel` — True when the per-pair weight is the stock
    Fresnel one, so each trunk's fused C++ kernels are eligible. A
    coefficient-modified ground sets it False and is served by the numpy
    projector / weight producers with zero solver edits.
    """

    mode: str
    eps_tilde: complex | None
    image_coefficient: complex | float
    standard_fresnel: bool = True

    @property
    def weighted(self) -> bool:
        """True for the reflection-coefficient ground alone: the one
        ground whose image carries a per-PAIR weight.

        PEC has no weight, and the Sommerfeld image's weighting is the
        scalar `image_coefficient` rather than a pair table. Both trunks
        asked this question in their own spelling — `weighted=` on the
        field trunk's constructor, "is `phi_mode` carried" on the
        potential trunk's — and it is the same question.
        """
        return self.eps_tilde is not None and self.mode == "fold"


def ground_config(solver, omega) -> GroundConfig | None:
    """`solver`'s ground decision at `omega`, or `None` for free space.

    `None` is not a null object and must not become one on either trunk:
    the free-space fill takes the path it always took, with the ground
    branch structurally absent rather than skipped, so not one float
    operation differs. Same standard as `extended_kernel=False`.

    | ground     | mode    | eps_tilde | coef   | weighted |
    |------------|---------|-----------|--------|----------|
    | free       | *(None — no config)*                     |
    | PEC        | fold    | None      | 1      | False    |
    | refl-coef  | fold    | ε̃(ω)      | 1      | True     |
    | sommerfeld | compose | ε̃(ω)      | C₂(ε̃)  | False    |

    Every expression below is the two factories' own, moved and not
    rewritten — the discipline that makes this a bit-identical migration
    rather than an equivalent one (momwire#392, architecture doc §3.4).
    """
    if solver.ground_z is None:
        return None
    if solver.ground_eps is None:
        return GroundConfig(mode="fold", eps_tilde=None, image_coefficient=1.0)
    eps_t = _ground_refl.eps_tilde(solver.ground_eps, omega, solver.eps)
    if solver.ground_model == "sommerfeld":
        # NEC's decomposition (theory manual eqs 136-147): the exact image
        # scaled by C₂, which absorbs all the singular behaviour, plus the
        # smooth remainder. ε̃ → ∞: C₂ → 1 and the remainder → 0, PEC image
        # exactly; ε̃ → 1: both vanish, free space.
        return GroundConfig(
            mode="compose",
            eps_tilde=eps_t,
            image_coefficient=(eps_t - 1.0) / (eps_t + 1.0),
        )
    return GroundConfig(mode="fold", eps_tilde=eps_t, image_coefficient=1.0)


def ground_touch_tol(polyline):
    """Snap distance for "this endpoint touches the ground plane": 1e-6 of
    the wire's own polyline length.

    Loose enough for deck-import float noise at z = 0, far tighter than
    any deliberate clearance (a 1 mm stand-off on a 10 m vertical is 100×
    the tolerance). It scales with each wire's length rather than being
    absolute, which is why it can disagree with an ABSOLUTE grouping
    tolerance about a plane two coincident ends share — see
    `RazorSolver._find_junctions`, which handles that disagreement on
    purpose.

    momwire#429 unit 4 (the audit's rank 7): five byte-identical spellings
    of these two lines lived in `bspline`, `razor`, `pulse` and
    `sinusoidal` (twice), the B-spline one already cited BY NAME from the
    other three's docstrings. The VALUE is unchanged and this unit changed
    no tolerance anywhere — the tree carries six disagreeing tolerances
    across three algorithms and two norms, and unifying those is a
    separate decision (momwire#429's correction 2).

    It lives in the ground spec layer because "does this wire end touch
    the plane" is a question about the GROUND, asked identically by every
    solver, and answered before any formulation is chosen.
    """
    pl = np.asarray(polyline, dtype=np.float64)
    length = float(np.sum(np.linalg.norm(np.diff(pl, axis=0), axis=1)))
    return 1e-6 * max(length, 1e-30)


def contact_ends(polylines, ground_z):
    """Which wire ENDS lie in the ground plane: `(wire_index, "start"|"end")`.

    A pure geometry question — no formulation, no ground model — answered
    with `ground_touch_tol`'s per-wire tolerance, so it agrees end for end
    with `BSplineSolver._wire_endpoint_status`'s `"ground"` tagging,
    `SinusoidalSolver`'s `ground_minus`/`ground_plus` marking and
    `RazorSolver._ground_ends`. Returns a sorted tuple, empty in free space
    and over a plane nothing touches.

    It answers ONLY about the two anchors of each polyline, because that is
    what "ground contact" means everywhere in this tree: a wire END in the
    plane, whose current the image continues. An INTERIOR anchor in the
    plane is mid-span touchdown, which is a different (and refused, or
    unmodelled) thing on every solver, and this helper deliberately does not
    speak for it. Geometry that is outright wrong — a wire dipping BELOW the
    plane, or an edge lying IN it — is not diagnosed here either: each
    solver already raises its own `ValueError` for those, with its own
    wording, and duplicating the check would only make two error messages
    race.
    """
    if ground_z is None:
        return ()
    gz = float(ground_z)
    ends = []
    for i, pl in enumerate(polylines):
        pl_arr = np.asarray(pl, dtype=np.float64)
        tol = ground_touch_tol(pl_arr)
        if abs(float(pl_arr[0, 2]) - gz) <= tol:
            ends.append((i, "start"))
        if abs(float(pl_arr[-1, 2]) - gz) <= tol:
            ends.append((i, "end"))
    return tuple(sorted(ends))


# The one geometry the reflection-coefficient ground served and should not
# have. Raised at construction by `BSplineSolver` (and its `HMatrixSolver` /
# `ArrayBlockSolver` subclasses) and by `SinusoidalSolver` (and its
# `SinusoidalGalerkinSolver` subclass), and quoted by each one's
# `capabilities.refusals["contact+refl-coef"]`.
#
# This WITHDRAWS a capability that shipped. momwire#151 gave the mixed-
# potential trunk a grounded-end basis and momwire#282 gave the direct-field
# trunk its contact-charge correction, and both then answered a `refl-coef`
# contact deck — the DEFAULT ground model — without complaint and without
# any reference behind them. `docs/design/contact-over-finite-ground.md`
# put a reference behind them (the licensed NEC-5 binary's printed
# impedances) and the refl-coef row failed it by 27 Ω, which is the measured
# fact this prose states. The maintainer's decision to withdraw rather than
# warn is dated 2026-08-18 in that document's decision record (D3).
#
# The MODEL is what fails, not momwire's implementation of it, and the
# refusal says so because the alternative reading — "momwire cannot do what
# other codes can" — is the wrong lesson to leave a user with. NEC-2's own
# reflection-coefficient ground was run on the same deck class as a
# post-study cross-check (stock nec2c, `GN 0`, 5.35 m contact monopole at
# 14 MHz): 175 − 779j Ω over average soil and 155 − 1248j Ω over poor soil,
# against a sane 39.4 + 22.1j Ω from the same binary over `GN 1`. Hundreds
# of ohms of spurious reactance, in the reference implementation of the
# model. A reflection coefficient is a plane-wave object evaluated on a
# specular ray, and at zero clearance there is no such ray to evaluate it
# on; no code has a valid story there.
CONTACT_UNDER_REFL_COEF_REFUSAL = (
    "ground CONTACT under ground_model='refl-coef' is refused "
    "(momwire#282 stage 1, 2026-08-18): the reflection-coefficient "
    "ground's Phi-term weight is a specular-angle approximation with no "
    "validity at zero clearance (momwire#153), and at contact it does not "
    "approximate the answer at all. Measured on a base-fed quarter-wave "
    "vertical over average soil (eps_r 13, sigma 0.005 S/m) at 14 MHz, "
    "N=41: refl-coef gives 27.0+12.6j ohm against this solver's own "
    "sommerfeld answer of 51.5+23.2j ohm on the same deck and the NEC-5 "
    "binary's printed 52.4+22.7j ohm — 27 ohm out, and on the WRONG SIDE "
    "of the 40.7+23.3j ohm PEC answer, so it is not a degraded "
    "approximation but a different number. This row was served and "
    "silently wrong; docs/design/contact-over-finite-ground.md 3.6 has "
    "the measurement. The MODEL is what has no story at the plane, not "
    "momwire's implementation of it: NEC-2's own reflection-coefficient "
    "ground (nec2c, GN 0) prints 175-779j ohm over average soil and "
    "155-1248j ohm over poor soil on the same contact monopole, against a "
    "sane 39.4+22.1j ohm over GN 1 — a reflection coefficient is a "
    "plane-wave object evaluated on a specular ray, and at zero clearance "
    "there is no such ray. Use ground_model='sommerfeld', which is "
    "contact-capable (momwire#151) and gated there against the binary, or "
    "raise the wire clear of the plane — refl-coef stays served, and stays "
    "the default, for wires standing clear in its 0.1-0.5 lambda validity "
    "window"
)
