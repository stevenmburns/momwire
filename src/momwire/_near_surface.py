"""Bare wire running alongside the ground surface, BELOW it (momwire#1946).

momwire#865 refuses a bare wire ABOVE the interface closer than h = 2a
(`BSplineSolver._wire_endpoint_status`). Nothing said anything about the same
zone on the other side: a buried wire within about two radii of the surface,
or the first stretch of a radial that leaves a crossing node (which has to
start AT the surface, z = 0) on a shallow slope.

The crossing-node study (momwire#1149, comment of 11-14; #1687) measured that
zone. Wherever a wire's axis lies within about 2a of z = 0 for a stretch, on
either side, razor2p and nec5-bs2 converge TOGETHER to a mesh-converged
answer about 0.5 ohm in X away from NEC-5's; move the wire out of the zone and
the gap goes. There every thin-wire model, NEC-5's included, is outside its
own validity (the surface of a wire within one radius of the interface
pierces it), so no engine can say which answer is right.

This is an ADVISORY, never a refusal: a sloping radial has to start at the
surface, so refusing it would refuse a shape people build. It is pure
geometry, read once per solver from facts every basis already holds (the
polylines, the per-wire media labels, the conductor radii, the wavelength),
so it is ONE check shared by razor2p, the B-spline family and the
sinusoidal family, and it cannot move a number.

## The rule, and why

For every bare wire labelled BELOW (wholly buried, which includes a crossing
node's buried arm), take the length of its axis whose depth below the
interface is under ``2a`` (``a`` the conductor radius). Pieces of one wire
that a deck spells as several GW cards — a graded node, a kink — are joined
into one stretch where exactly two of them meet end to end (a node where three
or more meet, such as the crossing node itself, is a branch and joins
nothing). The advisory fires when a stretch is

* at least ``λ/500`` long (λ the free-space wavelength), AND
* at least ``10a`` long.

The first condition is what separates the study's decks, and it is a length
rather than a count of radii because the measured gap tracks the absolute
length of wire in the zone, not its length in radii (the study's table; λ =
21.4 m at 14 MHz, so λ/500 = 43 mm):

    deck (momwire#1149)              per-radial stretch   in λ      ΔX at 2560
    slant14 (#1687's shape)          0.42 m               λ/51      -0.55 ohm
    slant14, x5 slope                0.08 m               λ/268     -0.09 ohm
    slant14, 0.2 mm radials          0.04 m               λ/535     -0.04 ohm
    slant7 (7 MHz, 1 mm)             0.07 m               λ/611     -0.02 ohm
    slant14 KINKED to -1 cm          0.02 m               λ/1070    -0.01 ohm
    -0.5 mm buried control           0.36 m               λ/59      -1.2 ohm

so λ/500 fires on every deck with a gap of 0.09 ohm or more and is silent on
every one at 0.04 ohm or less, the kinked −1 cm control (the acceptance
control) among them with a factor of two to spare. Counted in radii the 0.2 mm
radials (200 radii, −0.04 ohm) would sit ABOVE slant14 (210 radii, −0.55
ohm) — the wrong order.

The issue sketched the rule as "within 2a for more than a segment length".
That is deliberately not used: the gap is MESH-CONVERGED (it settles under
refinement, uniform or node-local), so a rule keyed on the segment length
would switch the advisory on as the user refines toward the right answer,
which is backwards.

The second condition makes the advisory about a wire lying ALONGSIDE the
interface rather than one that crosses it: a wire leaving the surface at angle
θ spends 2a/sin θ within the zone, so 10a means θ below about 11°. A vertical
ground rod spends exactly 2a there whatever its radius and never fires, even
a fat one at a high frequency where 2a alone could pass λ/500.

Out of scope, on purpose:

* ABOVE-ground wire. momwire#865 owns it: the B-spline family refuses an edge
  below h = 2a and `SurfaceRadialHeight` advises up to h = 20a, so a second
  message there would be a double one. (razor2p and the sinusoidal family
  have no above-ground floor today; the study's +0.5 mm control shows the
  same class of gap there. That is a separate item.)
* Jacketed wire. The question for a buried insulated wire is the jacket's
  radius against the depth, a different statement than this measured one.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import NamedTuple

import numpy as np

from . import _medium_spec


class NearSurfaceWire(UserWarning):
    """A bare buried wire runs within about two radii of the ground surface,
    where no thin-wire model is reliable. Advisory only — nothing is moved and
    nothing is refused."""


# The zone, in conductor radii: the depth under which a wire counts as AT the
# surface. The same 2 as momwire#865's above-ground floor, so the two sides of
# the interface read the same number.
ZONE_RADII = 2.0
# The shortest stretch that fires, in free-space wavelengths (see the module
# docstring's table for why 1/500).
MIN_STRETCH_WAVELENGTHS = 1.0 / 500.0
# ...and in radii: a stretch this long is lying alongside the surface rather
# than passing through it (θ below ~11°).
MIN_STRETCH_RADII = 10.0
# The study's figure, quoted in the message (momwire#1149 / #1687, slant14).
STUDY_DX_OHM = 0.5

# Point the warning at the caller's construction site, as `_surface_height`
# and `_crossing_fill` do.
_WARN_TARGET = (
    {"skip_file_prefixes": (os.path.dirname(os.path.abspath(__file__)),)}
    if sys.version_info >= (3, 12)
    else {"stacklevel": 4}
)


class Stretch(NamedTuple):
    """One run of buried wire within the zone."""

    wires: tuple  # the caller's wire indices it is made of, in deck order
    length_m: float  # axis length within ZONE_RADII radii of the interface
    closest_radii: float  # least depth / a along it (0: touches the surface)
    ends: tuple  # its two outer ends, (x, y, z) in metres
    radius_m: float  # its widest piece's conductor radius


def _in_zone(pl, gz, band):
    """(axis length with depth < band, least depth) for one polyline.

    The depth is linear along each straight edge, so the in-zone part of an
    edge is closed-form and the least depth sits at a vertex.
    """
    depth = np.maximum(gz - pl[:, 2], 0.0)
    total = 0.0
    for i in range(len(pl) - 1):
        d0, d1 = float(depth[i]), float(depth[i + 1])
        ln = float(np.linalg.norm(pl[i + 1] - pl[i]))
        lo, hi = min(d0, d1), max(d0, d1)
        if hi < band:
            total += ln
        elif lo < band:
            total += ln * (band - lo) / (hi - lo)
    return total, float(depth.min())


def near_surface_stretches(
    polylines, media, conductor_radii, ground_z, insulation_radius=None
):
    """Every stretch of bare BELOW wire within the zone, longest first.

    Pure geometry. `media` is the solver's `_wire_media()` labels;
    `insulation_radius` is the per-wire jacket array (NaN for a bare wire) or
    None. Returns every stretch with a positive in-zone length — the fire
    threshold is `stretches_that_fire`'s, so a caller (a test, a census) can
    read the near misses too.
    """
    if ground_z is None or _medium_spec.BELOW not in media:
        return []
    gz = float(ground_z)
    radii = np.asarray(conductor_radii, dtype=float)
    per_wire = {}
    for w, label in enumerate(media):
        if label != _medium_spec.BELOW:
            continue
        if insulation_radius is not None and np.isfinite(insulation_radius[w]):
            continue
        a = float(radii[w])
        pl = np.asarray(polylines[w], dtype=np.float64)
        length, closest = _in_zone(pl, gz, ZONE_RADII * a)
        if length > 0.0:
            per_wire[w] = (pl, a, length, closest)
    if not per_wire:
        return []

    # Join pieces of ONE wire that a deck split into several cards: where
    # exactly two in-zone wire ends meet. The key is the end's position on a
    # grid far finer than any radius (a thousandth of the smallest), so float
    # noise from a deck's own text cannot split a point in two.
    q = 1e-3 * min(a for _, a, _, _ in per_wire.values())
    meets = {}
    for w, (pl, _a, _ln, _c) in per_wire.items():
        for end, p in (("start", pl[0]), ("end", pl[-1])):
            key = tuple(int(round(float(c) / q)) for c in p)
            meets.setdefault(key, []).append((w, end))
    parent = {w: w for w in per_wire}

    def find(w):
        while parent[w] != w:
            parent[w] = parent[parent[w]]
            w = parent[w]
        return w

    for members in meets.values():
        if len(members) == 2 and members[0][0] != members[1][0]:
            ra, rb = find(members[0][0]), find(members[1][0])
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

    groups = {}
    for w in per_wire:
        groups.setdefault(find(w), []).append(w)
    out = []
    for ws in groups.values():
        ws = sorted(ws)
        # Outer ends: those not shared with another member of the stretch.
        counts = {}
        for w in ws:
            pl = per_wire[w][0]
            for p in (pl[0], pl[-1]):
                key = tuple(int(round(float(c) / q)) for c in p)
                counts.setdefault(key, [0, p])[0] += 1
        outer = [p for n, p in counts.values() if n == 1] or [
            per_wire[ws[0]][0][0],
            per_wire[ws[-1]][0][-1],
        ]
        out.append(
            Stretch(
                wires=tuple(ws),
                length_m=sum(per_wire[w][2] for w in ws),
                closest_radii=min(per_wire[w][3] / per_wire[w][1] for w in ws),
                ends=tuple(tuple(float(c) for c in p) for p in outer[:2]),
                radius_m=max(per_wire[w][1] for w in ws),
            )
        )
    out.sort(key=lambda s: -s.length_m)
    return out


def stretches_that_fire(stretches, wavelength):
    """The stretches long enough to advise on (the module docstring's rule)."""
    floor = MIN_STRETCH_WAVELENGTHS * float(wavelength)
    return [
        s
        for s in stretches
        if s.length_m >= floor and s.length_m >= MIN_STRETCH_RADII * s.radius_m
    ]


def _num(v):
    """A coordinate in metres to 4 significant figures, never "-0": a wire
    0.2 mm deep must read as -0.0002, not as a fixed-point -0.000."""
    if abs(v) < 1e-12:
        return "0"
    return f"{v:.4g}"


def _xyz(p):
    return "(" + ", ".join(_num(float(v)) for v in p) + ")"


def near_surface_message(fired):
    """The advisory text for the stretches that fire (longest first).

    The opening words ("buried wire runs within 2 radii of the ground
    surface") are what the tests match on; reword after them freely, and
    change `tests/test_near_surface_advisory_1946.py` with the opening.
    """
    s = fired[0]
    where = (
        f"from {_xyz(s.ends[0])} to {_xyz(s.ends[1])} m"
        if len(s.ends) == 2
        else f"at {_xyz(s.ends[0])} m"
    )
    names = ", ".join(str(w) for w in s.wires)
    closest = (
        "it touches the surface"
        if s.closest_radii < 0.05
        else f"its closest approach is {s.closest_radii:.1f} radii"
    )
    others = ""
    if len(fired) > 1:
        others = (
            f" {len(fired) - 1} other buried wire"
            f"{'s do' if len(fired) > 2 else ' does'} the same."
        )
    return (
        f"buried wire runs within {ZONE_RADII:.0f} radii of the ground "
        f"surface: the wire {where} (momwire wire {names}) stays that close for "
        f"{s.length_m:.2f} m, and {closest}.{others} So close to the surface "
        f"no thin-wire model, NEC's included, describes the wire properly, and "
        f"the programs disagree: on a test antenna whose four radials each ran "
        f"0.42 m this close, the reactance X came out about "
        f"{STUDY_DX_OHM:.1f} ohm different between momwire and NEC-5 (the "
        f"resistance R about 0.1 ohm), and neither can be shown right. More "
        f"wire this close means a larger doubt. Burying the wire deeper, or "
        f"sloping it away from the surface more steeply where it leaves the "
        f"feed point, removes it. Advisory only: nothing is changed or "
        f"refused (momwire#1946)."
    )


def warn_near_surface(
    polylines, media, conductor_radii, ground_z, wavelength, insulation_radius=None
):
    """Emit the advisory if any stretch fires; return its text, else None.

    Called once per solver, where its `_wire_media()` labels are first
    computed (they are cached per instance, so this cannot repeat within one
    solver). Reads only; cannot change a fill.
    """
    stretches = near_surface_stretches(
        polylines, media, conductor_radii, ground_z, insulation_radius
    )
    fired = stretches_that_fire(stretches, wavelength)
    if not fired:
        return None
    text = near_surface_message(fired)
    warnings.warn(text, NearSurfaceWire, **_WARN_TARGET)
    return text


def warn_for_solver(solver, media):
    """`warn_near_surface` with a solver's own attributes, once per solver.

    Called from each basis's `_wire_media()` where its labels are first
    computed. razor2p computes them inside `__init__`, BEFORE
    `_wire_loading.configure_loading` has set the jacket and the conductor
    radius this rule reads, so until that has run the labels are parked on the
    solver and `configure_loading` calls `flush` as its last step. Either way
    it reads only, and cannot change a fill.
    """
    if not hasattr(solver, "_conductor_radius_per_wire"):
        solver._near_surface_pending = media
        return None
    solver._near_surface_pending = None
    return warn_near_surface(
        solver.wires_polylines,
        media,
        solver._conductor_radius_per_wire,
        solver.ground_z,
        solver.wavelength,
        getattr(solver, "insulation_radius", None),
    )


def flush(solver):
    """Emit a parked advisory (see `warn_for_solver`); a no-op otherwise."""
    media = getattr(solver, "_near_surface_pending", None)
    if media is not None:
        return warn_for_solver(solver, media)
    return None
