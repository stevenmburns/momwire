"""Closed-form segment moments of the thin-wire kernels, reduced and extended.

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

:func:`_static_axis_moments_ek` is the same two integrals under the EXTENDED
(tubular) kernel, for the coaxial equal-radius pairs `_ek_axis_groups`
declares eligible. It is a third function rather than a flag on the second
because the two are different closed forms, and because a caller that never
asks for it must not enter it (the EK-off bit-identity standard).

The names keep their leading underscores (they are private to the package,
not to a module) so that the consumers' import lines are a pure path change.
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


def _static_axis_moments_ek(u_r, rho2, seg_h, a):
    """Closed-form static moments of the EXTENDED (tubular) kernel.

    The same ``m0 = ∫₀^h dτ/…``, ``m1 = ∫₀^h τ dτ/…`` as
    :func:`_static_axis_moments`, but of NEC's extended-kernel static
    integrand rather than of 1/R. The extended kernel is the reduced one
    times `_bspline_kernels._ek_factor`, whose k → 0 limit is

        f_static(R) = 1/R − a²/(2R³) + 3a⁴/(4R⁵)

    (the coaxial equal-radius specialisation of NEC Eq 89 — ρ_eval = b = a,
    so the factor is a function of R alone; see `_bspline_kernels`' module
    docstring for the specialisation and `_ek_axis_groups` for who is
    eligible for it). R is the SAME R the reduced kernel already uses,
    R² = u² + ρ² with `u = τ − u_r` and ρ² from :func:`_axis_frame`, which is
    what makes this a drop-in replacement for the reduced statics on the
    eligible pairs and nothing else.

    Both are elementary. With A_n(u) = ∫du/R^n:

        A_1 = asinh(u/ρ)    A_3 = u/(ρ²R)    A_5 = u/(3ρ²R³) + 2u/(3ρ⁴R)

    so, collecting the two 1/R terms and using ρ² − a² = the squared
    PERPENDICULAR offset of the observer from the source axis,

        m0 = [ asinh(u/ρ) + a⁴u/(4ρ²R³) − a²(ρ²−a²)u/(2ρ⁴R) ]₀^h
        m1 = u_r·m0 + [ R + a²/(2R) − a⁴/(4R³) ]₀^h

    The 1/R terms are collected rather than left as −a²/(2ρ²)·(u/R) +
    a⁴/(2ρ⁴)·(u/R) deliberately: on an eligible pair the observer sits on
    the source's own axis, so ρ = a and those two cancel EXACTLY. Written
    separately the cancellation would be catastrophic (two O(1) terms
    summing to zero); written as ρ²−a² it is exactly 0.0 in IEEE, and the
    surviving a⁴u/(4ρ²R³) is the whole O(a²) correction. Both expressions
    also collapse to :func:`_static_axis_moments` term by term at a = 0
    rather than merely rounding to it.

    `a` is the pair's common radius — eligibility REQUIRES equal radii, so
    the source column's radius (`RazorSolver._kernel_radius`) is that number
    on every pair this is called for. Scalar or anything broadcasting
    against ``(n_obs, n_seg)``, exactly as :func:`_axis_frame` takes it.
    """
    rho2_ = rho2
    a2 = a * a
    a4 = a2 * a2
    perp2 = rho2_ - a2
    rho = np.sqrt(rho2_)
    u0 = -u_r
    u1 = seg_h[None, :] - u_r
    r0 = np.sqrt(u0 * u0 + rho2_)
    r1 = np.sqrt(u1 * u1 + rho2_)
    # P(u) = asinh(u/ρ) + a⁴u/(4ρ²R³) − a²·perp²·u/(2ρ⁴R)
    c3 = 0.25 * a4 / rho2_
    c1 = 0.5 * a2 * perp2 / (rho2_ * rho2_)
    p1 = np.arcsinh(u1 / rho) + c3 * u1 / (r1 * r1 * r1) - c1 * u1 / r1
    p0 = np.arcsinh(u0 / rho) + c3 * u0 / (r0 * r0 * r0) - c1 * u0 / r0
    m0 = p1 - p0
    # Q(u) = R + a²/(2R) − a⁴/(4R³)
    q1 = r1 + 0.5 * a2 / r1 - 0.25 * a4 / (r1 * r1 * r1)
    q0 = r0 + 0.5 * a2 / r0 - 0.25 * a4 / (r0 * r0 * r0)
    m1 = u_r * m0 + (q1 - q0)
    return m0, m1
