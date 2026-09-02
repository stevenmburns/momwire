"""Cancellation-free spellings of the differences the kernels form.

Every kernel here integrates a *remainder* — the full Green's function with
its static part taken analytically — and the subtraction that forms it is a
difference of near-equal numbers wherever the phase is small. Computed
literally, `exp(-jkR) - 1` at kR = 1e-3 returns its real part to an absolute
epsilon rather than a relative one: the answer is -kR^2/2 ~ 6e-7 out of terms
of size 1, so the relative error is eps/(kR^2/2) ~ 7e-11, and it grows as
1/kR^2 as the mesh refines.

The remedy is always the same and never a tolerance: spell the difference so
that no term in it is larger than the answer. The siblings, which is the
table to consult before writing a new kernel:

    unstable                      stable
    --------                      ------
    exp(x) - 1                    expm1(x)                (std::/np./math.)
    log(1 + x)                    log1p(x)                        "
    cos(y) - 1                    -2 sin^2(y/2)           `expm1_neg_j` below
    exp(-jkR) - 1                 the bracket below       `expm1_neg_jkR`
    sin(u) - u                    series      `sinusoidal._sin_minus_arg`
    asinh(x) - x                  series      `sinusoidal._asinh_minus_arg`
    1/sin(kd) - 1/(2 sin(kd/2))   half-angle  `sinusoidal._recip_sin_gap`
    (1 - cos kd)/sin kd           tan(kd/2)               exact identity
    asinh(x1) - asinh(x0)         log1p of the ratio      `asinh_diff` below
    sqrt(u1^2+p^2) - sqrt(u0^2+p^2)   (u1^2-u0^2)/(r1+r0) `sqrt_diff` below
    sqrt(1 + x) - 1               x/(sqrt(1+x) + 1)
    cosh(x) - 1                   2 sinh^2(x/2)
    |d|^2 - (d.t)^2               |d - (d.t)t|^2  `_kernel_moments._axis_frame`

**The last row is the one that bit.** It is the only entry whose subtraction
is EXACTLY zero rather than merely small -- a collinear pair has |d| = |d.t|
-- so it is invisible to every check that compares one implementation against
itself, and shows up only where two implementations round it differently. It
cost 1e-7 relative in the `rho2` every static moment takes a logarithm of,
and it was the whole of the 8.2e-13 cross-lane gap momwire#798 measured on
macOS and attributed to the row at the top of this table. Removing the top
row's cancellation did not move that number by a digit; removing this one did.

**The complex bracket.** `expm1` alone only fixes the decay half of the
remainder. With k = k_re + j*k_im, Im k <= 0, a = k_im*R and y = k_re*R,

    exp(-jkR) - 1 = [ expm1(a)*cos(y) - 2 sin^2(y/2) ]  -  j*[ exp(a)*sin(y) ]

Both terms of the real bracket are cancellation-free, and the imaginary part
never cancelled. At real k (a = 0) it collapses to -2 sin^2(y/2) - j sin(y),
which is what `expm1_neg_j` computes directly.

`np.expm1` accepts a complex dtype, and it is NOT used here. Its real part is
`exp(a)cos(y) - 1` in some builds, i.e. exactly the unstable form, so trusting
it would make the numpy lane's stability a property of the numpy build rather
than of this repo. It measured stable on numpy 2.5.2 / glibc 2.35 and that is
not the claim these kernels need. The C++ twins spell the bracket out; so do
these, term for term, so the two lanes share ONE form.

Multi-step spelling throughout (momwire#205): a single expression with a dead
operand changes rounding above numpy's temporary-elision threshold, which
makes a fill depend on its block size.
"""

import numpy as np


def expm1_neg_j(w):
    """e^{-jw} - 1 for a REAL angle w, neither part cancelling.

    The real part is cos w - 1 = -2 sin^2(w/2), which the literal subtraction
    would compute to an absolute epsilon rather than a relative one; the
    imaginary part is -sin w, which never cancelled.
    """
    half = np.sin(0.5 * w)
    re = -2.0 * half * half
    im = np.sin(w)
    return re - 1j * im


def expm1_neg_j_from_half(half):
    """e^{-jw} - 1 given `half` = -w/2, the HALF phase.

    Same object as :func:`expm1_neg_j`, reached from the half angle instead of
    from w. With h = sin(-w/2) and c = cos(-w/2),

        cos w - 1 = -2 h^2       sin w = -2 h c

    so both parts come out of the same two transcendental calls. This is the
    spelling a kernel wants when it already holds a table of half phases --
    `_accel_sinusoidal.cpp` builds one, because r_q = r_ref + delta_q makes
    the half phase of r_q the SUM of two stored halves and no new sincos is
    needed for it at all. The numpy lane spells it the same way so the two
    share one form (momwire#799 rule 1); against :func:`expm1_neg_j` the
    imaginary part differs by ~2 ulp, which is why this is a separate entry
    point rather than a reimplementation of that one.
    """
    h = np.sin(half)
    c = np.cos(half)
    re = -2.0 * h * h
    im = 2.0 * h * c
    return re + 1j * im


def expm1_neg_jkR(k, R):
    """e^{-jkR} - 1 for a real or in-medium (complex, Im k <= 0) `k`.

    The module docstring's bracket. The complex branch is an OPTIMISATION and
    not a second definition: at a = 0 `expm1(a)*cos(y)` is exactly 0.0 and
    `exp(a)*sin(y)` exactly `sin(y)`, so a complex-dtype k whose imaginary
    part happens to be zero returns the real branch's bits. What the branch
    buys is not evaluating an exp and a cos on the hot real-k path.
    """
    if not np.iscomplexobj(k):
        return expm1_neg_j(k * R)
    a = np.imag(k) * R
    y = np.real(k) * R
    half = np.sin(0.5 * y)
    re = np.expm1(a) * np.cos(y)
    re = re - 2.0 * half * half
    im = np.exp(a) * np.sin(y)
    return re - 1j * im


def sqrt_diff(u0, u1, rho2, r0, r1):
    """sqrt(u1^2 + rho^2) - sqrt(u0^2 + rho^2), rationalised.

    `(u1^2 - u0^2)/(r1 + r0)`, with the numerator formed as
    `(u1 - u0)*(u1 + u0)` so that neither factor is a difference of terms
    larger than itself: `u1 - u0` is the segment length, exact to the
    subtraction that formed the two, and `u1 + u0` never cancels against
    anything. `r1 + r0` is a sum of positives.

    An exact rationalisation, not a series: it is the identity
    (a-b) = (a^2-b^2)/(a+b) with a, b > 0.
    """
    du = u1 - u0
    su = u1 + u0
    return du * su / (r1 + r0)


def asinh_diff(u0, u1, rho2, r0, r1):
    """asinh(u1/rho) - asinh(u0/rho), through `log1p` of the ratio.

    asinh(x) = log(x + sqrt(1 + x^2)), so with p_i = u_i + r_i (both scaled
    by rho, which cancels) the difference is log(p1/p0) = log1p((p1-p0)/p0),
    and p1 - p0 = (u1 - u0) + (r1 - r0) = du*(1 + (u1+u0)/(r1+r0)), whose
    bracket cancels for a far observer where u1 and u0 have the same sign and
    r_i -> |u_i|. Collecting it over r1 + r0 removes that too:

        p1 - p0 = du * (p1 + p0) / (r1 + r0)

    so the whole difference is `log1p(du*(p1+p0)/((r1+r0)*p0))` and no term
    in it is larger than the answer.

    `p_i = u_i + r_i` is itself a cancelling subtraction when u_i < 0 (there
    r_i -> |u_i|), and is rationalised as rho^2/(r_i - u_i) exactly then --
    the same identity as :func:`sqrt_diff`, applied to (r_i - u_i)(r_i + u_i)
    = rho^2. That branch is on the SIGN of u_i, so it is exact rather than a
    threshold.

    The divisor is written `r_i + |u_i|`, which IS `r_i - u_i` wherever that
    arm is selected (u_i < 0, and negation of a float is exact). `np.where`
    evaluates both arms, and on the arm it discards `r_i - u_i` underflows to
    0.0 once rho/|u_i| drops below sqrt(eps) -- a 1e-6 radius seen from 100 m
    reaches it -- so the literal spelling raises a divide-by-zero warning for
    a value that is then thrown away, and would raise for real under
    `np.seterr(divide="raise")`. `r_i + |u_i|` is never zero. The C++ twin's
    ternary short-circuits and so has no such arm, but it is spelled with
    `fabs` too so the two lanes read the same.
    """
    p0 = np.where(u0 >= 0.0, u0 + r0, rho2 / (r0 + np.abs(u0)))
    p1 = np.where(u1 >= 0.0, u1 + r1, rho2 / (r1 + np.abs(u1)))
    du = u1 - u0
    num = du * (p1 + p0)
    return np.log1p(num / ((r1 + r0) * p0))
