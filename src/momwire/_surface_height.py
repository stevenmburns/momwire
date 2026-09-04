"""Radials ON the ground: the height is the model, and it is ill-conditioned.

momwire#865 asked what to do about a wire lying in the ground plane. The
answer is that there is nothing to do at z = 0 — a conductor ON the interface
is not a physical configuration and the kernel has no answer there, so that
refusal stands unchanged. What a real "surface" radial is, is the ELEVATED
family taken to h ~ the conductor's own centre height: an insulated No. 18
wire lying in grass has its conductor about 1 mm above the soil (0.5 mm radius
plus ~0.4 mm of jacket), and the above-ground fill answers that today with no
new kernel.

That is a serve by SPELLING, not by physics, and it comes with a sharp edge
this module exists to say out loud: **at these heights the answer depends
strongly on h**, because a wire lying on a lossy dielectric is a slow-wave
line. The radial becomes electrically longer than its free-space length, and a
sparse screen detunes through a resonance as h falls toward the wire radius.
Measured on the Severns deck below, R swings 53 -> 74 -> 126 -> 175 -> 124 ohm
between h = 5 mm and h = 1 mm at four radials. A user who lays the same wire in
slightly deeper grass gets a different antenna, and that is a fact about the
installation rather than an artefact of the model.

## Why this advisory does not measure the deck's own slope

It would cost a second solve, and this is the class where that is least
affordable: the 64-radial Severns deck takes **449 s** on a 2026 Linux dev box,
so an unconditional two-height slope would make it ~900 s. The advisory follows
`_razor_class.warn_far_mesh_class` instead — unconditional, solve-free, quoting
a committed measured row plus the facts that are free to read off the geometry
(this deck's h, h/a and radial count). `surface_height_slope()` below is the
opt-in that pays for the real per-deck number when a caller wants it.

## The validity floor is a refusal, not part of this advisory

Below h/a = 2 the fill stops being trustworthy (the mesh check moves ~4 % at
h/a = 2 and worse beneath it), so that is refused by name at geometry time in
each solver rather than warned about here. This module is for decks that ARE
served and are merely sensitive.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import NamedTuple


class SurfaceRadialHeight(UserWarning):
    """A conductor is within a few radii of the interface, where the answer is
    a strong function of the stand-off. Advisory only — nothing is moved and
    nothing is refused."""


class SurfaceHeightClass(NamedTuple):
    """One measured statement of the low-height class's conditioning.

    Slopes are local `dR/dh` in ohm per millimetre on the reference deck, which
    is what a user comparing two installations of the same antenna feels.
    """

    slope_sparse_ohm_per_mm: float  # at N = 4, around h = 1.5 mm
    slope_dense_ohm_per_mm: float  # at N >= 16, same neighbourhood
    sparse_n: int  # the radial count the sparse figure is for
    dense_n: int  # the count at which it becomes quotable
    default_h_mm: float  # radius + jacket for No. 18, the recommended default
    floor_h_over_a: float  # the validity floor, refused below
    advisory_h_over_a: float  # above this the class is insensitive; no advisory
    mesh_move_pct_at_floor: float  # how much the mesh check moves at that floor
    issue: str
    provenance: str


# Measured 2026-09-03/04 on the Severns part 3 deck (7.2 MHz, 33.5 ft mast,
# 33 ft No. 18 radials, soil 30 / 0.020, n_rad 10), momwire main 80217bd.
#
# The sparse slope is read off R at N = 4: 175.28 ohm at h = 1.5 mm against
# 125.96 at 2.0 mm and 123.66 at 1.0 mm — about 50 ohm across the 1 mm either
# side of the peak, and it changes SIGN through it, which is why the number is
# quoted as a magnitude and the message says "swings" rather than "rises".
# The dense figure is the same neighbourhood at N = 16: 43.19 / 47.65 / 51.85
# at 2.0 / 1.5 / 1.0 mm, i.e. a few ohms per millimetre and monotone.
#
# `advisory_h_over_a` is where the class stops being sensitive, read off the
# same N = 4 column rather than chosen: |dR/dh| against h/a runs
#
#     h/a    294    98     39     20     10     5.9    3.9
#     ohm/mm 0.02   0.06   0.27   1.56   10.1   52.4   98.6
#
# so 20 is the decade where it crosses ~1 ohm/mm. Above it a millimetre of
# grass is worth less than the bar on any gate we have, and an advisory
# would be noise on every ordinary elevated deck.
SURFACE_HEIGHT_CLASS = SurfaceHeightClass(
    slope_sparse_ohm_per_mm=50.0,
    slope_dense_ohm_per_mm=4.0,
    sparse_n=4,
    dense_n=16,
    default_h_mm=1.0,
    floor_h_over_a=2.0,
    advisory_h_over_a=20.0,
    mesh_move_pct_at_floor=4.0,
    issue="stevenmburns/momwire#865",
    provenance="antennaknobs scratch/buried-flow/unit5-surface-radials.md",
)

# Point the warning at the caller's construction site rather than inside this
# package, exactly as `_crossing_fill` does for `CoarseCrossingNode` and
# `_razor_class` for the far-mesh advisory.
_WARN_TARGET = (
    {"skip_file_prefixes": (os.path.dirname(os.path.abspath(__file__)),)}
    if sys.version_info >= (3, 12)
    else {"stacklevel": 4}
)


def surface_height_message(
    h_m: float,
    a_m: float,
    n_low: int,
    cls: SurfaceHeightClass = SURFACE_HEIGHT_CLASS,
) -> str:
    """The advisory text, composed from the measured row and this deck's facts.

    Separate from the warning so a test can pin the message to
    `SURFACE_HEIGHT_CLASS` rather than to a literal, and so a caller that wants
    the sentence for a report can have it without raising a warning.

    THE FIRST LINE IS LOAD-BEARING, for the same reason it is in
    `_razor_class.far_mesh_class_message`: `pyproject.toml`'s pytest
    `filterwarnings` silences this advisory by matching the opening words
    ("a conductor lies within"), NOT by category — matching by category makes
    pytest import momwire while parsing filters, ahead of the conftest's
    per-worker `OMP_NUM_THREADS` pin. Reword freely after that phrase; if the
    opening must change, change the filter in the same commit.
    """
    h_mm = h_m * 1e3
    ratio = h_m / a_m if a_m > 0 else float("inf")
    sparse = n_low <= cls.sparse_n
    return (
        f"a conductor lies within a few radii of the ground: {n_low} "
        f"near-ground wire(s) at h = {h_mm:.2f} mm, h/a = {ratio:.1f}. At this "
        f"stand-off the driving-point impedance is a STRONG function of h — a "
        f"wire on a lossy dielectric is a slow-wave line, so the conductor is "
        f"electrically longer than its free-space length and a sparse screen "
        f"detunes as h falls. Measured on the reference deck, |dR/dh| is about "
        f"{cls.slope_sparse_ohm_per_mm:.0f} ohm per MILLIMETRE at "
        f"N = {cls.sparse_n} near 1.5 mm (it swings through a resonance and "
        f"changes sign), against roughly {cls.slope_dense_ohm_per_mm:.0f} "
        f"ohm/mm at N >= {cls.dense_n}, where the class becomes quotable. "
        + (
            "This deck is in the sparse regime, so treat its impedance as "
            "indicative rather than predictive: the same wire in deeper grass "
            "is a measurably different antenna. "
            if sparse
            else f"This deck is at or above N = {cls.dense_n}, the quotable "
            f"end of the class. "
        )
        + f"The height IS the model here — there is no coating model, so h "
        f"stands in for radius plus jacket ({cls.default_h_mm:.1f} mm for a "
        f"No. 18 insulated wire lying on soil) and for however the wire sits "
        f"in the grass. Below h/a = {cls.floor_h_over_a:.0f} the fill is "
        f"refused by name, because the mesh check already moves "
        f"{cls.mesh_move_pct_at_floor:.0f} % there. For the deck's OWN slope "
        f"rather than the class figure, call "
        f"`momwire.surface_height_slope()` — it costs a second solve, which "
        f"is why it is not done here. Advisory: nothing is moved and nothing "
        f"is refused. See {cls.issue} ({cls.provenance})."
    )


def warn_surface_height(
    h_m: float,
    a_m: float,
    n_low: int,
    cls: SurfaceHeightClass = SURFACE_HEIGHT_CLASS,
) -> str:
    """Emit the advisory once, and return the text that was emitted.

    Returning it is deliberate: it lets a test read back exactly what a caller
    saw, and it is the seam a report generator uses without catching a warning.
    """
    text = surface_height_message(h_m, a_m, n_low, cls)
    warnings.warn(text, SurfaceRadialHeight, **_WARN_TARGET)
    return text


def surface_height_slope(solver_factory, h_m: float, step: float = 0.25):
    """This deck's OWN local dR/dh, in ohm per millimetre. Opt-in, and it
    costs TWO solves.

    `solver_factory(h)` must build the same deck at stand-off `h` in metres.
    The slope is a forward difference over `h` and `(1 + step) * h`, which is a
    ratio rather than an absolute step so it stays meaningful across the two
    decades of height this class spans.

    Not called by the advisory, and that is the whole point: on the 64-radial
    Severns deck one solve is 449 s, so making this unconditional would put a
    second 449 s on precisely the decks that can least afford it. Callers who
    need the real number for one deck can pay for it here knowingly.

    Returns `(slope_ohm_per_mm, z_lo, z_hi)` so the caller can see the two
    impedances the slope came from rather than trusting a single number.
    """
    z_lo, _ = solver_factory(h_m).compute_impedance()
    h_hi = h_m * (1.0 + step)
    z_hi, _ = solver_factory(h_hi).compute_impedance()
    slope = (z_hi.real - z_lo.real) / ((h_hi - h_m) * 1e3)
    return slope, z_lo, z_hi
