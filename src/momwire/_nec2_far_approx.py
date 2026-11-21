"""NEC-2's interaction approximation on the point-matched sinusoidal fill:
an opt-in, APPROXIMATE feature (`nec2_far_approx`), off by default
everywhere.

NEC-2 fills a matrix entry whose two segment centres are more than RKH
wavelengths apart from an "approximate integration" (its printout's words)
rather than the full one, and sets RKH with the KH card. The user's guide
gives the card and its purpose, a time saving on large structures; it does
not give the form, so the form here was fixed by black-box experiment on
nec2c 1.3.1 (below), against the NEC-2 theory manual's three-term current
(Burke and Poggio, NEC Part I, 1981).

THE FORM (`_sinusoidal_far_core.cpp`): the source segment's three-term
current A + B sin(k s) + C cos(k s) over [-H, H] is lumped into an
infinitesimal dipole at the segment's centre, along its tangent, whose
moment is the current's integral: 2H for the constant shape, zero for the
sine (odd), 2 sin(kH)/k for the cosine. Its full field (the 1/R, 1/R^2 and
1/R^3 terms) is evaluated at the observer point the exact kernel uses and
projected as the exact kernel projects, or through the same Fresnel dyad on
the reflection-coefficient image. Against the exact kernel it differs by
O((kH)^2) and O((H/R)^2) relative, so at RKH = 1 wavelength and ordinary
meshes each approximated entry moves by about 1e-3 of itself.

WHAT NEC-2 DOES, measured on nec2c 1.3.1 as a black box (2026-11-20; the
probes and their numbers are in the tests,
`tests/test_nec2_far_approx.py`):

* With no KH card the approximation is ON at 1 wavelength: every printout
  says "APPROXIMATE INTEGRATION EMPLOYED FOR SEGMENTS THAT ARE MORE THAN
  1.000 WAVELENGTHS APART", and the answer equals `KH 0 0 0 0 1`'s. So the
  default range here, `DEFAULT_RANGE`, is 1 wavelength.
* KH's range is its first floating field, the fifth field of the card
  (`KH 0 0 0 0 0.5`); the four integer fields are unused. `KH 0 0 0 0 1000`
  turns the approximation off in effect (every pair nearer than that).
* The image interactions take it too, decided on the observer's distance to
  the IMAGE segment's centre: a short horizontal dipole whose own pairs are
  all inside the range and whose image pairs are all outside it moves when
  the range is crossed, over PEC, a reflection-coefficient and a Sommerfeld
  ground.
* The extended kernel (EK) takes it too.

THE DECISION is ``|c_m - c_n| > range * wavelength`` on the centres, in
both the direct and the image block. A pair whose distance equals the range
to rounding is on a side decided by rounding, in NEC-2 as here; with
`pair_reuse` on, such a deck declines the reuse (`ties`), because a reused
class could otherwise serve a member the other side's value.

SCOPE. The NEC-2 dialect's kernels on the real-k fill: free space, the PEC
image, the reflection-coefficient image (Fresnel's), and the Sommerfeld
ground's C2-scaled image, whose interpolated remainder stays exact; the
reduced kernel has its own far-pair kernel (the speed), the extended kernel
and the folded cos-1 shapes take the shipped tensors with the far pairs
overwritten (the answer, at the shipped cost). It refuses a NEC-4 junction
rule, a buried or ground-crossing wire, and a reflection coefficient other
than Fresnel's (`REFUSALS`).
"""

from __future__ import annotations

import math

import numpy as np

# NEC-2's range when a deck sends no KH card, in wavelengths (measured on
# nec2c 1.3.1; see the module docstring).
DEFAULT_RANGE = 1.0

# The relative width, |d - R| <= TIE_REL * R, inside which a pair's side of
# the range is a matter of rounding for `pair_reuse`'s classes: their
# members differ from the representative by 2^-(KEY_BITS + 1) relative
# (KEY_BITS = 48, `_sin_pair_reuse`) and the canonical frame adds a few ulp,
# so 2^-30 leaves a margin of about 2^17.
TIE_REL = 2.0**-30

NEC4_RULE_REFUSAL = (
    "nec2_far_approx is NEC-2's interaction approximation and acts on the "
    "NEC-2 dialect's kernel only; this solve takes the NEC-4 junction rule, "
    "so turn it off for this deck"
)
BELOW_INTERFACE_REFUSAL = (
    "nec2_far_approx is NEC-2's interaction approximation and NEC-2 has no "
    "buried or ground-crossing wires; turn it off for this deck"
)
NONSTANDARD_GROUND_REFUSAL = (
    "nec2_far_approx serves Fresnel's reflection coefficient only; turn it "
    "off for this ground"
)
REFUSALS = (NEC4_RULE_REFUSAL, BELOW_INTERFACE_REFUSAL, NONSTANDARD_GROUND_REFUSAL)

TIES_DECLINE = (
    "a pair whose distance equals the nec2_far_approx range to rounding, "
    "whose side a reused class could not keep"
)


def check_range(value) -> float:
    """The range in wavelengths, as a float: finite and not negative."""
    r = float(value)
    if not (math.isfinite(r) and r >= 0.0):
        raise ValueError(
            f"nec2_far_approx_range must be a finite, non-negative number of "
            f"wavelengths, got {value!r}"
        )
    return r


def range_metres(range_wl: float, k: float) -> float:
    """The range in metres at free-space wavenumber `k`."""
    return float(range_wl) * 2.0 * math.pi / float(k)


def ties(acc, obs_centers, src_centers, far_r: float) -> int:
    """How many (observer, source) pairs lie within `TIE_REL` of `far_r`."""
    return int(
        acc.far_range_ties(
            np.ascontiguousarray(obs_centers, dtype=np.float64),
            np.ascontiguousarray(src_centers, dtype=np.float64),
            float(far_r),
            TIE_REL,
        )
    )
