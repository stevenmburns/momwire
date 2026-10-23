"""The NEC-4 condition on charge at a junction, for the point-matched
sinusoidal basis on the NEC-4 dialect (momwire#1644).

Public source: G. J. Burke, *Numerical Electromagnetics Code - NEC-4, Method
of Moments, Part II: Program Description - Theory*, UCRL-MA-109338 Pt. II
(January 1992), Section 3.4 and Appendix A. In paraphrase:

NEC-2 (and NEC-3) fix the ratio of the charge densities the members of a
junction carry with the Wu-King log rule, the theory manual's (3-41), which
is `_basis_coefs`' `a_const = 1/(ln(2/(k a)) - gamma)` per SEGMENT. NEC-4
replaces that rule at a junction whose charge is not equal by symmetry: it
isolates the junction, extends each member wire to infinity along its own
outward direction, and solves a small moment problem for continuity of the
scalar potential there, (3-42). The charge on member l is a half triangle
over its junction segment plus a constant running to infinity (n = 2 in
(3-43)), matched at distances Delta_i / 2 and Delta_i from the node on
member i's axis. The density at the node, Q_l = (q_l1 + q_l2) Delta_l, is
then the a^- / a^+ the basis functions take at that end.

Only the RATIOS of the Q_l at one node matter: every basis function's
coefficients are invariant when all the factors at one node are scaled
together (that node's charge unknown absorbs it). `junction_charge_factors`
therefore returns the Q_l scaled so the first member keeps the factor the
log rule gives it, which changes no answer and keeps the numbers near the
NEC-2 ones in size.

Three readings the public text leaves open, each decided against licensed
NEC-4.2 run as a black box (decks in, printouts out; momwire#1644 has the
tables):

* the source of each member is a tube at that member's own radius around
  its axis, the observer on the observing member's axis, so
  R^2 = a_l^2 + |x_obs - x_src|^2 (Appendix A's rho);
* the charge ratios are REAL. The solve below takes the kernel's real part,
  cos(kR)/R; taking the real part of the complex solve's ratios instead is
  the same to 1e-3 ohm on every deck measured, and keeping the imaginary
  part puts the input resistance 0.01 to 0.1 ohm further from NEC-4.2 (on
  a feed away from a step, -0.03 against -0.001);
* a junction of two members of equal radius keeps the uniform (log-rule)
  factors; a node of three or more members is solved even at one radius (a
  symmetric node solves to equal factors anyway).

Evaluation. Appendix A's own reduction is used: the ramp part of the
triangle integrates in closed form, (j/k)(e^{-jkR2} - e^{-jkR1}), and the
semi-infinite constant is the finite integral up to the observer's foot plus
half the infinite line, -(j pi / 2) H0^(2)(k b), (A-5) -- whose real part is
-(pi/2) Y0(k b). The one remaining integral, Phi0 = int cos(kR)/R, is its
static part 1/R in closed form (asinh) plus the smooth remainder
(cos kR - 1)/R = -2 sin^2(kR/2)/R by a fixed Gauss-Legendre rule. Appendix A
evaluates the same integrals by a short series; the two agree to the
series' own accuracy.

Every operation below is scalar float arithmetic on libm functions (math's
sqrt, sin, log, asinh), so the C++ copy in `_sinusoidal_front_core.cpp`
repeats it bit for bit under `-ffp-contract=off`.
`tests/test_nec4_junction_1644.py` holds the two equal.
"""

from __future__ import annotations

import math

from ._quadrature import leggauss

__all__ = ["junction_charge_factors", "junction_densities", "JUNCTION_GL_ORDER"]

# The Gauss-Legendre order on every smooth-remainder interval. The remainder
# -2 sin^2(kR/2)/R is smooth apart from a |u| kink of size k^2 b at the
# observer's foot, which every interval below has at an END, and the
# intervals are at most a segment long, so 16 nodes integrate it far below
# the solve's own sensitivity (`tests/test_nec4_junction_1644.py` measures
# the elements against mpmath to 1e-10).
JUNCTION_GL_ORDER = 16

_EULER_GAMMA = 0.5772156649015329
_HALF_PI = 0.5 * math.pi
_TWO_OVER_PI = 2.0 / math.pi
# Terms of the J0 / Y0 power series. k b here is at most a segment's
# electrical length, far inside the series' fast range; the cap is a guard.
_BESSEL_TERMS = 40
# Above this argument the series loses digits; `junction_densities` refuses
# a node whose members' geometry would need it.
BESSEL_MAX_ARG = 8.0

JUNCTION_TOO_WIDE = (
    "a junction member lies too far from another member's axis for the "
    "NEC-4 junction-charge solve (k b = {kb:.3g} > {cap:g}); shorten the "
    "segments at the junction (momwire#1644)"
)


def _gl_rule():
    xs, ws = leggauss(JUNCTION_GL_ORDER)
    return [float(x) for x in xs], [float(w) for w in ws]


_GL_X, _GL_W = _gl_rule()


def _phi0(k, b, z, lo, hi):
    """int_lo^hi cos(kR)/R ds, R = sqrt(b^2 + (s - z)^2): the static part in
    closed form, the remainder by the fixed rule on each side of the
    observer's foot s = z when it falls inside, so each piece is smooth."""
    b2 = b * b
    total = math.asinh((hi - z) / b) - math.asinh((lo - z) / b)
    pieces = ((lo, z), (z, hi)) if lo < z < hi else ((lo, hi),)
    for p, q in pieces:
        half = 0.5 * (q - p)
        mid = 0.5 * (q + p)
        acc = 0.0
        for x, w in zip(_GL_X, _GL_W, strict=True):
            u = mid + half * x - z
            r = math.sqrt(b2 + u * u)
            sh = math.sin(0.5 * k * r)
            acc = acc + w * (-2.0 * sh * sh / r)
        total = total + acc * half
    return total


def _bessel_y0(x):
    """Y0(x) by its power series, x > 0."""
    q = 0.25 * x * x
    term = 1.0
    j0 = 1.0
    s = 0.0  # sum of (-1)^{m+1} H_m q^m / (m!)^2
    harmonic = 0.0
    for m in range(1, _BESSEL_TERMS + 1):
        term = -term * q / (m * m)
        harmonic = harmonic + 1.0 / m
        j0 = j0 + term
        s = s - harmonic * term
        if abs(term) < 1e-18:
            break
    return _TWO_OVER_PI * ((math.log(0.5 * x) + _EULER_GAMMA) * j0 + s)


def _triangle(k, b, z, d):
    """int_0^d (d - s) cos(kR)/R ds -- the real part of Appendix A's ramp
    form, (d - z) Phi0 - (j/k)(e^{-jkR(d)} - e^{-jkR(0)})."""
    r_hi = math.sqrt(b * b + (d - z) * (d - z))
    r_lo = math.sqrt(b * b + z * z)
    # Re[-(j/k)(e^{-jkR2} - e^{-jkR1})] = (sin kR1 - sin kR2)/k
    return (d - z) * _phi0(k, b, z, 0.0, d) + (
        math.sin(k * r_lo) - math.sin(k * r_hi)
    ) / k


def _semi_infinite(k, b, z):
    """int_0^inf cos(kR)/R ds, by (A-5): the finite integral over [0, z]
    (signed) plus -(pi/2) Y0(k b) for the line's far half."""
    out = -_HALF_PI * _bessel_y0(k * b)
    if z > 0.0:
        out = out + _phi0(k, b, z, 0.0, z)
    elif z < 0.0:
        out = out - _phi0(k, b, z, z, 0.0)
    return out


def _solve(mat, rhs):
    """Gaussian elimination with partial pivoting (largest |.|, first wins)
    on a small real system, in place."""
    n = len(rhs)
    for c in range(n):
        p = c
        best = abs(mat[c][c])
        for r in range(c + 1, n):
            v = abs(mat[r][c])
            if v > best:
                best = v
                p = r
        if best == 0.0:
            raise ZeroDivisionError("singular junction-charge system")
        if p != c:
            mat[c], mat[p] = mat[p], mat[c]
            rhs[c], rhs[p] = rhs[p], rhs[c]
        piv = mat[c][c]
        top = mat[c]
        for r in range(c + 1, n):
            f = mat[r][c] / piv
            row = mat[r]
            for cc in range(c, n):
                row[cc] = row[cc] - f * top[cc]
            rhs[r] = rhs[r] - f * rhs[c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        acc = rhs[r]
        for cc in range(r + 1, n):
            acc = acc - mat[r][cc] * x[cc]
        x[r] = acc / mat[r][r]
    return x


def junction_densities(k, dirs, radii, lengths):
    """Q_l, the charge density each member carries at the node, from the
    n = 2 solve of (3-43) with C = 1. `dirs` are unit vectors pointing AWAY
    from the node along each member, `radii` and `lengths` each member's
    radius and junction-segment length."""
    m = len(radii)
    mat = []
    for i in range(m):
        for s in (0.5 * lengths[i], lengths[i]):
            p0 = s * dirs[i][0]
            p1 = s * dirs[i][1]
            p2 = s * dirs[i][2]
            row = []
            for ell in range(m):
                d = dirs[ell]
                z = p0 * d[0] + p1 * d[1] + p2 * d[2]
                x0 = p0 - z * d[0]
                x1 = p1 - z * d[1]
                x2 = p2 - z * d[2]
                a = radii[ell]
                b = math.sqrt(a * a + (x0 * x0 + x1 * x1 + x2 * x2))
                if k * b > BESSEL_MAX_ARG:
                    raise ValueError(
                        JUNCTION_TOO_WIDE.format(kb=k * b, cap=BESSEL_MAX_ARG)
                    )
                delta = lengths[ell]
                row.append(_triangle(k, b, z, delta))
                row.append(delta * _semi_infinite(k, b, z))
            mat.append(row)
    q = _solve(mat, [1.0] * (2 * m))
    return [(q[2 * ell] + q[2 * ell + 1]) * lengths[ell] for ell in range(m)]


def junction_charge_factors(k, dirs, radii, lengths, first_factor):
    """The basis functions' a^-/a^+ for each member of one node, scaled so
    member 0 carries `first_factor` (its log-rule value)."""
    dens = junction_densities(k, dirs, radii, lengths)
    scale = first_factor / dens[0]
    return [first_factor] + [qd * scale for qd in dens[1:]]
