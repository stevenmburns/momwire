"""Ground field evaluation at a POINT IN SPACE: the field trunk's
point-observer arm (momwire#545 U1).

`_sommerfeld.remainder_field_proj` already computes the whole reflected-field
vector per (observer, source) pair — the theory-manual eqs 143–147 azimuth
combination of the four interpolation surfaces I_ρ^V, I_z^V, I_ρ^H, I_φ^H —
and then throws away everything but the observer-TANGENT projection, because
its one consumer is a matrix fill. This module is that same combination kept
as a vector, so an observer that is a point in space rather than a wire
element can be asked the same question. One dyad, two consumers.

What lives here is the ground REMAINDER at a point (and its curl). The
composition is the consumer's, exactly as `_field_ground.FieldGround`
documents it for the fill:

    E(point) = E_direct + C₂·E_image + E_remainder,   C₂ = (ε̃−1)/(ε̃+1)

with the direct and image terms coming from whichever readout owns them (the
eznec seam and the nec2 portal both read them out of
`portal._portal._element_fields`, one owner per readout) and the association
discipline of `FieldGround.mode == "compose"` applying unchanged: the
coefficient goes on the LEFT of the block, and `coef·img + rem` is associated
before any outer sum.

The measured story (momwire#545, and the sign is measurement rather than
derivation)
---------------------------------------------------------------------------
Composed against the licensed engine's captured `NEAR ELECTRIC FIELDS` /
`NEAR MAGNETIC FIELDS` grids over `GN 0`, worst LIVE cell of each table
against that table's own scale:

    capture  card  antenna                 worst |cell|   worst phase
    0110     NE    10.3 m vertical             4.25 %        1.26°
    0113     NE    elevated 20 m dipole        1.80 %        0.30°
    0111     NH    10.3 m vertical (FD curl)   1.01 %        0.33°

The remainder ADDS. The opposite sign reads 7.7 % / 20.6° in `EZ` on 0110 and
15 % / 68.8° in `EY` on 0113, so the convention is pinned by the captures and
by a standing anti-regression gate (`tests/test_field_point.py`), not by an
argument about which way a manual's minus sign points.

The regimes
-----------
The evaluator is keyed by (source medium, observer medium) from its first
commit, so the buried-wire arc extends a dict instead of forcing a second
evaluator. Today exactly one key is implemented:

    (source, observer)      surface family                      state
    ("above", "above")      reflected wave over (ρ, h = z+z′)   implemented
    ("below", "below")      transmitted, both media             momwire#524
    ("below", "above")      transmitted, upward                 momwire#524
    ("above", "below")      transmitted, downward               momwire#524

The three unimplemented regimes raise `ValueError` naming themselves: their
surfaces are the TRANSMITTED-field Sommerfeld families of momwire#524
(its phases 1/3), a different integral family through different contours, not
a sign flip of this one. A regime entry owns its whole answer — the grid it
needs, the surfaces it samples, and the combination that turns them into a
vector — which is what makes "add a regime" an addition rather than a
refactor.

Clean-room note
---------------
The ("above", "above") formulation is NEC-2 theory-manual material
(docs/nec2_theory_manual.pdf §IV.1–IV.2), already implemented in
`_sommerfeld` from those public-domain equations. The licensed NEC-5 engine
enters only as a DATA oracle: its printed near-field tables are the captures
the envelopes above are measured against. No NEC-5 internals were consulted
and none are cited — conclusions only.

Byte-stability
--------------
`remainder_field_proj` and every solve path keep their bytes: this module
lands beside the fill and edits nothing. That the two spellings are ONE
algebra is a GATE — the vector projected on an observer tangent reproduces
the fill's own numbers at the same pairs — and not a refactor.
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple

import numpy as np

from . import _sommerfeld

# One owner for μ₀ (`_sommerfeld`'s, which is what `get_grid` normalizes the
# surfaces with); named here so the curl's −jωμ₀ and the grid fill cannot
# drift apart.
_MU0 = _sommerfeld._MU0

# Central-difference step for the magnetic evaluator, in wavelengths. See
# `reflected_h_field_at` for the error budget this number sits at the bottom
# of; it is a measured compromise and not a round number chosen for looks.
CURL_STEP_WAVELENGTHS = 1e-3


class _Regime(NamedTuple):
    """One (source medium, observer medium) pair's whole answer.

    `label` names the regime in refusals. `grid_for` sizes and fetches the
    interpolation family this regime reads; `surfaces` samples it at the
    (observer, source) pairs and returns the per-pair geometry the
    combination needs; `combine` weights those by a complex source moment and
    returns the 3-vector. `pending` is `None` for an implemented regime and
    otherwise the sentence that says whose work its surfaces are.

    Three callables rather than one function so that a new regime supplies a
    surface FAMILY and a COMBINATION separately: momwire#524's transmitted
    fields share neither with the reflected wave, but a later reflected-wave
    variant (a coefficient-modified ground, say) would share the combination
    and replace only the family.
    """

    label: str
    grid_for: Callable | None
    surfaces: Callable | None
    combine: Callable | None
    pending: str | None


def _reflected_wave_grid(points, src_mid, eps_t, ground_z, k, omega, mu):
    """The cached `SommerfeldGrid` sized for these observer/source pairs.

    Observer points size `r1_max` exactly as segment endpoints do — the
    obs-to-source-IMAGE distance — so `max_image_distance(ex, ex, ground_z)`
    over `ex = concat(points, src_mid)` is used unchanged. That is an UPPER
    BOUND rather than the exact max over obs-to-image pairs (it also ranges
    over obs-obs and src-src pairs), which is the direction an interpolation
    radius is allowed to err in: a grid tabulated further out is valid for
    every pair inside it, and `get_grid` buckets the radius up in ~25 %
    geometric steps anyway, so the bound almost always keys the same cached
    fill the exact max would.

    The issue #157 15 λ cap and `SommerfeldGrid.eval`'s r1 → r1_max clamp
    apply unchanged: past the cap the true distance stays in `g` and only the
    slowly-varying surface amplitude freezes.
    """
    ex = np.concatenate([np.asarray(points, dtype=float), np.asarray(src_mid)])
    r1_max = _sommerfeld.max_image_distance(ex, ex, ground_z)
    return _sommerfeld.get_grid(eps_t, k, r1_max, omega, mu=mu)


def _reflected_wave_surfaces(points, src_mid, ground_z, k, grid):
    """Per-(observer, source) reflected-wave geometry and surface values.

    ρ = horizontal separation, h = (z − z_g) + (z′ − z_g) — the distance to
    the IMAGE point is R₁ = √(ρ² + h²) and θ = atan2(h, ρ), the pair of
    coordinates the four surfaces are tabulated over. Returns the four
    interpolated surfaces, the free-space factor g = e^{−jkR₁}/R₁ they
    multiply, and the horizontal unit vector d̂ from the source's ground
    projection to the observer's, all at shape (N_points, N_src).
    """
    points = np.asarray(points, dtype=float)
    src_mid = np.asarray(src_mid, dtype=float)

    dx = points[:, 0][:, None] - src_mid[None, :, 0]
    dy = points[:, 1][:, None] - src_mid[None, :, 1]
    rho = np.hypot(dx, dy)
    hh = (points[:, 2] - ground_z)[:, None] + (src_mid[None, :, 2] - ground_z)
    r1 = np.sqrt(rho * rho + hh * hh)
    surf = grid.eval(r1, np.arctan2(hh, rho))
    g = np.exp(-1j * k * r1) / r1

    tiny = 1e-12 * grid.r1_max
    safe_r = rho > tiny
    inv_rho = np.where(safe_r, 1.0 / np.where(safe_r, rho, 1.0), 0.0)
    # rho -> 0: the incidence azimuth degenerates and any horizontal unit
    # vector serves, because I_rho^H(90 deg) = -I_phi^H there (the identity
    # `remainder_field_proj` relies on for the same cell, unit-tested in the
    # engine suite). The fill has a source tangent handy and uses its
    # horizontal direction; a point observer's source is a complex MOMENT
    # with no meaningful direction, so (1, 0) serves instead.
    dhx = np.where(safe_r, dx * inv_rho, 1.0)
    dhy = np.where(safe_r, dy * inv_rho, 0.0)
    return {"surf": surf, "g": g, "dhx": dhx, "dhy": dhy}


def _reflected_wave_combination(family, src_moment):
    """The eqs 143–147 azimuth combination, kept as a VECTOR and summed over
    sources: (N_points, 3) complex.

    The dyad is LINEAR in the source, so a complex current moment
    p = I·dl enters through two complex bilinears

        cph = p_x·d̂x + p_y·d̂y        (the unit-tangent form's |p_h|·cos φ)
        sph = p_x·d̂y − p_y·d̂x        (               ...  |p_h|·sin φ)

    and p_z, with no magnitude/direction split anywhere. That split is what
    `remainder_field_proj` does — it factors a REAL unit tangent into
    |t_h|·(cos φ, sin φ) and carries the magnitude separately — and it is
    available to it only because its sources are unit tangents. A complex
    horizontal moment has no direction to normalize (|p_h| is not a modulus
    that factors out of a complex pair, and choosing a phase reference for it
    would be inventing one), and it needs none: linearity means the two
    bilinears ARE the combination's only dependence on the source.
    """
    src_moment = np.asarray(src_moment, dtype=complex)
    surf = family["surf"]
    g = family["g"]
    dhx = family["dhx"]
    dhy = family["dhy"]

    p_x = src_moment[:, 0]
    p_y = src_moment[:, 1]
    p_z = src_moment[:, 2]

    cph = p_x[None, :] * dhx + p_y[None, :] * dhy
    sph = p_x[None, :] * dhy - p_y[None, :] * dhx

    e_rho = g * (p_z[None, :] * surf["IrhoV"] + cph * surf["IrhoH"])
    e_phi = g * sph * surf["IphiH"]
    e_z = g * (p_z[None, :] * surf["IzV"] - cph * surf["IrhoV"])

    out = np.empty((dhx.shape[0], 3), dtype=complex)
    out[:, 0] = np.sum(dhx * e_rho - dhy * e_phi, axis=1)
    out[:, 1] = np.sum(dhy * e_rho + dhx * e_phi, axis=1)
    out[:, 2] = np.sum(e_z, axis=1)
    return out


_REGIMES: dict[tuple[str, str], _Regime] = {
    ("above", "above"): _Regime(
        label="a source above the interface read at an observer above it",
        grid_for=_reflected_wave_grid,
        surfaces=_reflected_wave_surfaces,
        combine=_reflected_wave_combination,
        pending=None,
    ),
    ("below", "below"): _Regime(
        label="a source below the interface read at an observer below it",
        grid_for=None,
        surfaces=None,
        combine=None,
        pending=(
            "its field is carried by the TRANSMITTED-field Sommerfeld surfaces "
            "of momwire#524 (phases 1/3), a different integral family through "
            "different contours"
        ),
    ),
    ("below", "above"): _Regime(
        label="a source below the interface read at an observer above it",
        grid_for=None,
        surfaces=None,
        combine=None,
        pending=(
            "its field is carried by the TRANSMITTED-field Sommerfeld surfaces "
            "of momwire#524 (phases 1/3), a different integral family through "
            "different contours"
        ),
    ),
    ("above", "below"): _Regime(
        label="a source above the interface read at an observer below it",
        grid_for=None,
        surfaces=None,
        combine=None,
        pending=(
            "its field is carried by the TRANSMITTED-field Sommerfeld surfaces "
            "of momwire#524 (phases 1/3), a different integral family through "
            "different contours"
        ),
    ),
}


def _regime_for(source_medium: str, observer_medium: str) -> _Regime:
    """The implemented `_Regime` for one (source, observer) key, or a
    `ValueError` that names the regime it was asked for.

    The dispatch shape, not a docstring promise: the buried-wire arc lands by
    filling in three entries of `_REGIMES`, and no caller of this module
    changes.
    """
    key = (source_medium, observer_medium)
    regime = _REGIMES.get(key)
    if regime is None:
        raise ValueError(
            f"unknown ground regime {key!r}: source_medium and observer_medium "
            "must each be 'above' or 'below'"
        )
    if regime.pending is not None:
        raise ValueError(
            f"the {key!r} ground regime — {regime.label} — has no field "
            f"evaluator here: {regime.pending}. This module implements the "
            "('above', 'above') reflected-wave regime only."
        )
    return regime


def reflected_field_at(
    points,
    src_mid,
    src_moment,
    eps_t,
    ground_z,
    k,
    omega,
    *,
    source_medium: str = "above",
    observer_medium: str = "above",
    mu: float = _MU0,
    grid=None,
):
    """The ground REMAINDER of E at `points`: (N_points, 3) complex.

    `src_mid` (S, 3) are source element midpoints and `src_moment` (S, 3)
    their complex current moments p = I·dl — the same pair
    `portal._portal._element_fields` reads the direct field out of, so a
    consumer composes the two without resampling anything. `eps_t` is the
    complex relative permittivity ε̃ = εr − jσ/(ωε₀) (Im ≤ 0 for a passive
    ground, the e^{+jωt} convention this tree uses throughout), `ground_z` the
    interface height, `k` the free-space wavenumber and `omega` the radian
    frequency the surfaces are normalized at.

    This is the REMAINDER ALONE. It is not the ground's field and it is not
    an answer on its own: the direct term and C₂·(the image term) are the
    consumer's, and composing them is what `_field_ground`'s `"compose"` mode
    is about (see the module docstring). Returning only the remainder is the
    same split the fill uses, and it keeps one owner per readout — this
    module never learns what a mixed-potential element field is.

    `grid` lets a caller supply an already-built `SommerfeldGrid` so a batch
    of related evaluations provably shares ONE fill; omitted, the regime
    sizes and fetches its own through the module-level cache.
    """
    regime = _regime_for(source_medium, observer_medium)
    points = np.asarray(points, dtype=float)
    src_mid = np.asarray(src_mid, dtype=float)
    if grid is None:
        grid = regime.grid_for(points, src_mid, eps_t, ground_z, k, omega, mu)
    family = regime.surfaces(points, src_mid, ground_z, k, grid)
    return regime.combine(family, src_moment)


def reflected_h_field_at(
    points,
    src_mid,
    src_moment,
    eps_t,
    ground_z,
    k,
    omega,
    *,
    source_medium: str = "above",
    observer_medium: str = "above",
    mu: float = _MU0,
    step: float | None = None,
):
    """The ground REMAINDER of H at `points`, by numerical curl of E:
    (N_points, 3) complex.

    H_rem = ∇×E_rem/(−jωμ₀), the curl taken by central differences at
    `step` (default `CURL_STEP_WAVELENGTHS`·λ = 1e-3 λ) — six evaluations of
    `reflected_field_at`, two per axis.

    **Why differences and not a fifth surface.** The analytic route is a
    first-derivative λ-integral family through the same contours: real work,
    a new tabulation, and a new set of R₁ → 0 limits. The corpus's one
    magnetic oracle decides whether that work is needed, and it says no —
    0111 reads 1.01 % / 0.33° on its live column, inside the same envelope
    class the electric captures sit in. The analytic surfaces stay available
    as a later optimization if a gate ever demands one.

    **The error budget**, which is the whole of why this step and not another:

    * *FD truncation* — the central difference drops a term of order
      (k·step)²/6 relative to the derivative, which at step = 1e-3 λ is
      (2π·1e-3)²/6 ≈ 7e-6 of scale. Shrinking the step shrinks this
      quadratically.
    * *Interpolation noise amplification* — the grid's own interpolation
      error is ~1e-4 of surface scale and is NOT smooth in the observer
      coordinate, so a difference of two evaluations amplifies it by
      2/(k·step) in units of the derivative scale: ~2·(1e-4)/(2π·1e-3)
      ≈ 3e-2 worst-case. Shrinking the step GROWS this, linearly.

    The two cross near 1e-3 λ, which is where the step sits, and the
    amplification bound is a worst case rather than a measurement: the
    interpolation error of two points a thousandth of a wavelength apart is
    strongly correlated (the same 4×4 Lagrange stencil, usually the same
    cell), so most of it differences away. Measured on the oracle the total
    is 1.0 %, thirty times inside the bound.

    One grid serves all six evaluations, sized over the union of the offset
    points: the offsets differ from `points` by a thousandth of a wavelength
    and would key the same `get_grid` bucket anyway, but "would" is not
    "does", and a rebuild mid-curl would difference two DIFFERENT
    interpolants — the one case where the noise above does not correlate at
    all.
    """
    regime = _regime_for(source_medium, observer_medium)
    points = np.asarray(points, dtype=float)
    src_mid = np.asarray(src_mid, dtype=float)
    if step is None:
        step = CURL_STEP_WAVELENGTHS * (2.0 * math.pi / k)

    offsets = []
    for axis in range(3):
        for sign in (+1.0, -1.0):
            shift = np.zeros(3)
            shift[axis] = sign * step
            offsets.append(points + shift)

    grid = regime.grid_for(
        np.concatenate(offsets), src_mid, eps_t, ground_z, k, omega, mu
    )
    fields = [
        reflected_field_at(
            offset,
            src_mid,
            src_moment,
            eps_t,
            ground_z,
            k,
            omega,
            source_medium=source_medium,
            observer_medium=observer_medium,
            mu=mu,
            grid=grid,
        )
        for offset in offsets
    ]
    d_e = [(fields[2 * a] - fields[2 * a + 1]) / (2.0 * step) for a in range(3)]
    curl = np.stack(
        [
            d_e[1][:, 2] - d_e[2][:, 1],
            d_e[2][:, 0] - d_e[0][:, 2],
            d_e[0][:, 1] - d_e[1][:, 0],
        ],
        axis=-1,
    )
    return curl / (-1j * omega * mu)
