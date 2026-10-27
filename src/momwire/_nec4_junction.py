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
scalar potential there, (3-42). The charge on member l is expanded in n
functions: a half triangle over the junction segment, n - 2 full triangles
beyond it, and a constant running to infinity, (3-43), matched at
Delta_i / 2, Delta_i, 2 Delta_i, ... from the node on member i's axis. The
manual's NEC-4 sets n = 2 and reads the density at the node,
Q_l = (q_l1 + q_ln) Delta_l, as the a^- / a^+ the basis functions take at
that end.

Only the RATIOS of the Q_l at one node matter: every basis function's
coefficients are invariant when all the factors at one node are scaled
together (that node's charge unknown absorbs it). `junction_charge_factors`
therefore returns the Q_l scaled so the first member keeps the factor the
log rule gives it, which changes no answer and keeps the numbers near the
NEC-2 ones in size.

The readings below are licensed NEC-4.2's, each decided against it run as
a black box (decks in, printouts out). The first three are momwire#1644's:

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

and the last three momwire#1668's, which closed the two cases #1644 left
open (a feed beside a radius step, an equal-radius T), both of which were
the members' charge RATIO alone: one scale on one member's factor brings
every feed position on a deck to NEC-4.2 at once, to 1e-4 of the factor.

* n = 3: the half triangle, one full triangle (0 at the node, peak at
  Delta, 0 at 2 Delta) and the constant, matched at Delta/2, Delta and
  2 Delta. It moves a step's ratio by under 1 %, which is the whole
  feed-beside-step residual: n = 2 leaves 0.2 to 4 ohm of X on the
  ladder, n = 3 none above the uniform-wire floor, n = 4 and up too
  little again;
* ONE length for every member: Delta = the mean of the members'
  junction-segment lengths, for the expansion and the match points alike.
  At a step meshed 20 / 40 per half the fitted common length is the mean
  to 1e-4; each member's own length leaves 3 to 4 ohm. (An equal-radius
  two-member node with unequal segments is then symmetric, so its solve
  is the uniform factors the third reading gives it anyway.)
* where Q_l is read: at the node, (q_l1 + q_l3) Delta, for a node of two
  members; at the member's own junction-segment centre, s = Delta_l / 2
  (the main solve's match point), for a node of three or more. Read at the
  node, a T lands 3 to 11 ohm of X from NEC-4.2 and the miss grows with
  the mesh; read at the centre, every three- to five-member node measured
  (a T at five meshes and three radii, arms at 30 to 150 degrees, unequal
  meshes per arm, radii differing at the node) is within the floor. A
  two-member step read at the centre lands 6 to 60 ohm of X off instead,
  so the two readings are not one rule written twice.

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
sqrt, sin, log, asinh), so the C++ copy in `_sinusoidal_nec4_core.cpp`
repeats it bit for bit under `-ffp-contract=off`.
`tests/test_nec4_radius_step_1644.py` holds the two equal.
"""

from __future__ import annotations

import math

from ._quadrature import leggauss

__all__ = ["junction_charge_factors", "junction_densities", "JUNCTION_GL_ORDER"]

# The Gauss-Legendre order on every smooth-remainder interval. The remainder
# -2 sin^2(kR/2)/R is smooth apart from a |u| kink of size k^2 b at the
# observer's foot, which every interval below has at an END, and the
# intervals are at most two segments long, so 16 nodes integrate it far below
# the solve's own sensitivity (`tests/test_nec4_radius_step_1644.py` measures
# the elements against adaptive quadrature: 1e-10 within a segment of the
# node, 5e-9 at the third match point two segments out).
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


def _full_triangle(k, b, z, d, tri):
    """int_0^{2d} f2(s) cos(kR)/R ds for the full triangle f2 that rises
    from 0 at the node to d at s = d and falls back to 0 at s = 2d: the
    rising half is d Phi0 less the half triangle `tri` already computed for
    this element, the falling half is the half triangle moved out by d."""
    return d * _phi0(k, b, z, 0.0, d) - tri + _triangle(k, b, z - d, d)


def _density(q, d, s, at_node):
    """The charge density the n = 3 expansion of one member gives at
    distance s from the node: q1 (d - s) + q2 f2(s) + q3 d, or at the node
    itself, (q1 + q3) d (f2 vanishes there)."""
    if at_node:
        return (q[0] + q[2]) * d
    f1 = d - s if s < d else 0.0
    if s <= d:
        f2 = s
    elif s < 2.0 * d:
        f2 = 2.0 * d - s
    else:
        f2 = 0.0
    return q[0] * f1 + q[1] * f2 + q[2] * d


def junction_densities(k, dirs, radii, lengths):
    """Q_l, the charge density each member carries at the node, from the
    n = 3 solve of (3-43) with C = 1. `dirs` are unit vectors pointing AWAY
    from the node along each member, `radii` and `lengths` each member's
    radius and junction-segment length.

    The solve takes ONE length for every member, the mean of the members'
    junction-segment lengths, for the charge basis and the match points
    alike. A node of two members reads its density at the node; a node of
    three or more reads each member's at its own junction segment's centre
    (the main solve's match point), s = Delta_l / 2. Both readings, the
    order n = 3 and the common length were measured against licensed
    NEC-4.2 run as a black box (momwire#1668; the module docstring)."""
    m = len(radii)
    # A plain left-to-right sum, as the C++ copy adds (Python's own `sum`
    # compensates float rounding since 3.12).
    total = 0.0
    for length in lengths:
        total = total + length
    d = total / m
    mat = []
    for i in range(m):
        for s in (0.5 * d, d, 2.0 * d):
            p0 = s * dirs[i][0]
            p1 = s * dirs[i][1]
            p2 = s * dirs[i][2]
            row = []
            for ell in range(m):
                u = dirs[ell]
                z = p0 * u[0] + p1 * u[1] + p2 * u[2]
                x0 = p0 - z * u[0]
                x1 = p1 - z * u[1]
                x2 = p2 - z * u[2]
                a = radii[ell]
                b = math.sqrt(a * a + (x0 * x0 + x1 * x1 + x2 * x2))
                if k * b > BESSEL_MAX_ARG:
                    raise ValueError(
                        JUNCTION_TOO_WIDE.format(kb=k * b, cap=BESSEL_MAX_ARG)
                    )
                tri = _triangle(k, b, z, d)
                row.append(tri)
                row.append(_full_triangle(k, b, z, d, tri))
                row.append(d * _semi_infinite(k, b, z))
            mat.append(row)
    q = _solve(mat, [1.0] * (3 * m))
    at_node = m == 2
    return [
        _density(q[3 * ell : 3 * ell + 3], d, 0.5 * lengths[ell], at_node)
        for ell in range(m)
    ]


def junction_charge_factors(k, dirs, radii, lengths, first_factor):
    """The basis functions' a^-/a^+ for each member of one node, scaled so
    member 0 carries `first_factor` (its log-rule value)."""
    dens = junction_densities(k, dirs, radii, lengths)
    scale = first_factor / dens[0]
    return [first_factor] + [qd * scale for qd in dens[1:]]
