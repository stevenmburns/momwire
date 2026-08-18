"""Closed-form segment moments of the reduced thin-wire kernel.

The reduced kernel g(r, r') = exp(−jkR)/(4πR) with
R = sqrt(|r − r'|² + a²) has an axis frame in which its static part
integrates in closed form over a straight segment: project the observation
point onto the segment's axis (:func:`_axis_frame`) and the distance to a
source at local arc τ is R² = (τ − u_r)² + ρ², so ∫dτ/R and ∫τ dτ/R are
`asinh` and `sqrt` (:func:`_static_axis_moments`). What is left —
(exp(−jkR) − 1)/(4πR) — is smooth everywhere the reduced kernel is defined
and takes plain Gauss-Legendre.

This is not a formulation. It is the arithmetic any point-tested or
path-tested row wants once it has decided to expand on the wire AXIS, which
is why it lives here rather than inside one of them: `RazorSolver` (tent
basis, razor-blade testing) and `PulseSolver` (pulse basis, point matching)
both consume it, and momwire#425 is the record of pulse having reached into
razor's module-level privates to do so.

The two names keep their leading underscores (they are private to the
package, not to a module) so that the consumers' import lines are a pure
path change.
"""

import numpy as np


def _axis_frame(obs, seg_p0, seg_t, a):
    """Project observation points onto every segment's axis.

    Returns ``(u_r, rho2)`` of shape ``(n_obs, n_seg)``: the signed axial
    coordinate of the projection, measured from the segment's start point,
    and the squared perpendicular distance plus a². In that frame the
    reduced kernel's distance to a source at local arc τ is simply
    R² = (τ − u_r)² + ρ², which is what both the closed-form static
    moments and the quadratured remainder consume.

    `a` is the reduced kernel's regularising radius. It may be a scalar —
    one radius for the whole model, the historical case and still the fast
    path — or any array that broadcasts against ``(n_obs, n_seg)``:
    ``(n_seg,)`` for a radius carried by the SOURCE segment, ``(n_obs, 1)``
    for one carried by the OBSERVER. Only ``a * a`` is touched, so the
    scalar case is bit-for-bit what it always was and a uniform array is
    bit-for-bit the scalar (momwire#425).
    """
    d = obs[:, None, :] - seg_p0[None, :, :]
    u_r = np.einsum("psc,sc->ps", d, seg_t)
    # The perpendicular part can go a few ulps negative for a truly
    # collinear observation point; a² dominates it either way.
    rho2 = np.maximum(np.einsum("psc,psc->ps", d, d) - u_r * u_r, 0.0) + a * a
    return u_r, rho2


def _static_axis_moments(u_r, rho2, seg_h):
    """Closed-form static moments of the reduced kernel over each segment.

    Given the axis frame from :func:`_axis_frame`, returns ``(m0, m1)``:

        m0 = ∫₀^h dτ / R,   m1 = ∫₀^h τ dτ / R

    with τ the source's local arc length from the segment start and
    R = sqrt(|r − r'|² + a²). In the axis frame these are the collinear
    forms with u = τ − u_r: ∫du/R = asinh(u/ρ) and ∫u du/R = sqrt(u² + ρ²),
    then m1 = u_r·m0 + [sqrt(u² + ρ²)]. ρ ≥ a > 0 always, which is exactly
    what the reduced kernel's a² buys — the formula never sees a bare axis.

    The radius enters only through `rho2`, so a per-segment or per-observer
    radius needs nothing here.
    """
    rho = np.sqrt(rho2)
    u0 = -u_r
    u1 = seg_h[None, :] - u_r
    m0 = np.arcsinh(u1 / rho) - np.arcsinh(u0 / rho)
    m1 = u_r * m0 + (np.sqrt(u1 * u1 + rho2) - np.sqrt(u0 * u0 + rho2))
    return m0, m1
