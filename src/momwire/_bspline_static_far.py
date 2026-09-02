"""Far-field spelling of the same-edge static moments (momwire#808).

`_bspline_static_moments.J_static_moment` is sympy's closed form for

    J_pq = ∫_α^β ∫_A^B (s−α)^p (s'−A)^q / √((s−s')² + a²) ds' ds

and it is excellent where the two segments are close — measured 1e-14 at the
self pair and 2.6e-13 at the neighbour, for all nine moments at every mesh
this repo runs. It falls apart as they separate, and not gently: at 401
segments on a 9.7 m edge the (2, 2) moment between the two ends is **2.94e+01
relative**, i.e. no correct digits at all.

**That is not a grouping sympy could have chosen better.** The value decays
like h^5/D while any closed form assembled from antiderivatives evaluated at
the ξ = s − s' window's corners carries terms growing like D^5, so a
four-corner formula has to cancel D^6/h^5 whatever basis it is written in.
Local coordinates do not help — translation invariance removes α and A but
leaves the separation — and there is no pair of terms to collect over a
common denominator, because each carries a different polynomial coefficient.
The cancellation is a property of the closed form, not of this one.

So this module is the other regime rather than a replacement. With

    ξ = ξ₀ + (u − h₁/2) − (v − h₂/2),   ξ₀ = (α+β)/2 − (A+B)/2

expanding f(ξ) = (ξ² + a²)^{-1/2} about the pair's centroid ξ₀ gives

    J_pq = Σ_n f⁽ⁿ⁾(ξ₀)/n! · Σ_m C(n,m) (−1)^{n−m} · M_p[m] · M_q[n−m]

with M_p[m] = ∫₀^{h₁} u^p (u − h₁/2)^m du the polynomial moments taken about
each segment's OWN midpoint. Those are sums of positive terms with the odd
ones identically zero (see `_centred_moments`), so nothing in the coefficient
chain cancels; and the series is in powers of the window half-width over the
distance to the centroid, so it converges geometrically at exactly the rate
`series_ratio` reports.

WHERE THE SWITCH IS, AND WHY IT IS NOT TUNED
--------------------------------------------
f's poles sit at ξ = ±ja, so the expansion about ξ₀ converges for window
half-widths below √(ξ₀² + a²) — which is precisely

    r = (h₁ + h₂)/2 / √(ξ₀² + a²)

so `r < 1` IS the convergence condition, read off the function rather than
fitted. `FAR_RATIO = 0.5` takes the series a factor of two inside it, which
buys 53 terms of headroom at the boundary and lands where both spellings are
already accurate:

    r      1.000 (adjacent)   series 1e-5..1e-2   closed form 2.6e-13
    r      0.500 (next but one) series 3.0e-14    closed form 1.1e-12

The regimes overlap rather than meet at a cliff, which is the property that
makes a threshold safe to have. Measured across p, q ∈ {0,1,2}², N ∈ {21, 81,
201, 401} and separations 0 … N−1, against a Gauss-Legendre oracle on the
integral's own integrand under the ξ = a·sinh t substitution (which maps the
a-wide peak to unit width and so is trustworthy at the self pair too — plain
GL on the raw integrand is NOT, and reports the self pair 1e-2 wrong).
"""

import numpy as np
from math import comb

# Series terms. At the switch ratio each term is half the last, so 64 is
# 2^-64 of the first — comfortably under an ulp with room for the coefficient
# growth. Fixed rather than adaptive: this is evaluated over whole arrays, and
# a per-element stopping test would cost more than the terms it saves.
N_TERMS = 64

# Take the series a factor of two inside its radius of convergence. See the
# module docstring for why this is derived and not tuned.
FAR_RATIO = 0.5

MAX_P = 2  # what `_bspline_static_moments` is generated for


def _centred_moments(p, h, nmax):
    """M[m] = ∫₀^h u^p (u − h/2)^m du, for m = 0 … nmax.

    Substituting û = u − h/2 and expanding (û + h/2)^p makes every surviving
    term positive, and kills the odd ones outright:

        M[m] = Σ_j C(p,j) (h/2)^{p−j} · h^{m+j+1} / (2^{m+j} (m+j+1))

    over the j with m + j even. No subtraction appears anywhere in it, which
    is the point — these coefficients multiply a derivative chain that is
    already alternating, so any cancellation here would compound.
    """
    out = [np.zeros_like(np.asarray(h, dtype=float)) for _ in range(nmax + 1)]
    for m in range(nmax + 1):
        acc = np.zeros_like(np.asarray(h, dtype=float))
        for j in range(p + 1):
            k = m + j
            if k % 2:
                continue
            acc = acc + comb(p, j) * (h / 2.0) ** (p - j) * h ** (k + 1) / (
                2.0**k * (k + 1)
            )
        out[m] = acc
    return out


def _kernel_derivatives_over_factorial(xi0, a, nmax):
    """g[n] = f⁽ⁿ⁾(ξ₀)/n! for f(ξ) = (ξ² + a²)^{-1/2}, n = 0 … nmax.

    From f's own ODE, (ξ² + a²) f′ + ξ f = 0, differentiated n times by
    Leibniz:

        (ξ² + a²) f⁽ⁿ⁺¹⁾ + (2n+1) ξ f⁽ⁿ⁾ + n² f⁽ⁿ⁻¹⁾ = 0

    Dividing through by (n+1)! carries the factorial INSIDE the recurrence:

        g[n+1] = −[ (2n+1) ξ₀ g[n] + n g[n−1] ] / ((n+1)(ξ₀² + a²))

    which is the same three-term step and is what this returns. Not a
    refinement — f⁽ⁿ⁾ on its own reaches 1e174 at 64 terms on a 401-segment
    edge and about 1e265 at 10⁴ segments, and dividing by 64! afterwards
    would be doing the arithmetic in the one place it can overflow. g[n]
    stays near 1/ξ₀ⁿ⁺¹ throughout.

    Evaluated only where ξ₀² + a² is bounded away from zero by the switch,
    which is exactly the condition that makes the forward recurrence stable:
    the solution being followed is the growing one.
    """
    den = xi0 * xi0 + a * a
    g = [1.0 / np.sqrt(den)]
    g.append(-xi0 / (den * np.sqrt(den)))
    for n in range(1, nmax + 1):
        g.append(-((2 * n + 1) * xi0 * g[n] + n * g[n - 1]) / (den * (n + 1)))
    return g


def series_ratio(alpha, beta, A, B, a):
    """The expansion's convergence ratio r — see the module docstring.

    Below 1 the series converges; `far_mask` takes it at half that.
    """
    h1 = beta - alpha
    h2 = B - A
    xi0 = 0.5 * (alpha + beta) - 0.5 * (A + B)
    return 0.5 * (h1 + h2) / np.sqrt(xi0 * xi0 + a * a)


def far_mask(alpha, beta, A, B, a):
    """Where the series is the right spelling. Elementwise over arrays."""
    return series_ratio(alpha, beta, A, B, a) <= FAR_RATIO


def J_static_far(p, q, alpha, beta, A, B, a):
    """`J_static_moment`'s far-field twin. Same signature, same broadcast.

    Correct only where `far_mask` holds; the caller is expected to have
    checked. Evaluated unconditionally over an array is harmless — the series
    simply stops converging — but it is not an answer there.
    """
    if not (0 <= p <= MAX_P and 0 <= q <= MAX_P):
        raise ValueError(f"(p, q) = ({p}, {q}) not in [0, {MAX_P}]²")
    h1 = beta - alpha
    h2 = B - A
    xi0 = 0.5 * (alpha + beta) - 0.5 * (A + B)
    g = _kernel_derivatives_over_factorial(xi0, a, N_TERMS)
    Mp = _centred_moments(p, h1, N_TERMS)
    Mq = _centred_moments(q, h2, N_TERMS)
    total = np.zeros_like(np.asarray(g[0], dtype=float))
    for n in range(N_TERMS + 1):
        coef = np.zeros_like(total)
        # C(n, m) stepped rather than called, so the C++ twin can do the same
        # thing: `math.comb` is an exact integer and `std::` has no equivalent,
        # and the two lanes have to truncate the same way to agree.
        c_nm = 1.0
        for m in range(n + 1):
            if m > 0:
                c_nm = c_nm * (n - m + 1) / m
            term = c_nm * Mp[m] * Mq[n - m]
            coef = coef - term if (n - m) % 2 else coef + term
        total = total + g[n] * coef
    return total


def J_static_stable(p, q, alpha, beta, A, B, a):
    """The moment at any separation: closed form near, series far.

    This is what callers want. `_bspline_static_moments.J_static_moment` on
    its own is the near-field half and is wrong by 2900 % at the far corner
    of a 401-segment edge.

    The far branch is GATHERED rather than selected with `np.where`, which
    would evaluate it everywhere. Outside its radius the series does not
    merely return a wrong number — the derivative chain grows like
    1/(ξ₀²+a²)^{n/2}, so on the self pair (ξ₀ = 0, so the radius is the wire
    radius) it reaches 1e214 at 64 terms for a = 5e-4 and overflows outright
    below about a = 1e-5. `np.where` would discard those, but only after
    computing them and warning about it. The C++ twin's `if` short-circuits
    for the same reason, so this also keeps the two lanes the same shape.
    """
    from ._bspline_static_moments import J_static_moment

    out = np.asarray(J_static_moment(p, q, alpha, beta, A, B, a), dtype=float)
    mask = np.asarray(far_mask(alpha, beta, A, B, a))
    if not mask.any():
        return out
    out = np.array(out, copy=True)
    shape = out.shape
    if mask.shape != shape:
        mask = np.broadcast_to(mask, shape)

    def _at(x):
        x = np.asarray(x, dtype=float)
        return np.broadcast_to(x, shape)[mask] if x.shape != () else x

    out[mask] = J_static_far(p, q, _at(alpha), _at(beta), _at(A), _at(B), _at(a))
    return out
