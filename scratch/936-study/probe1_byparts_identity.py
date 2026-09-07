"""momwire#936 study: what exactly does the by-parts W move drop when a
segment is tilted, and how big is it?

THE MOVE UNDER TEST. `_crossing_fill` replaces a derivative-of-basis integral
against W by a derivative-of-KERNEL integral plus an end term
(`scratch/813-node-derivations/probe1_sw_by_parts.py` states it):

    sum_u w_u W F'(u)  -  [W F]_ends   ==   - sum_u w_u F(u) (dW/dl)

That is exact by parts for ANY orientation. What the fill then substitutes is

    dW/dl  ->  t_z * dW/dz                                    (the code)

which is the whole content of "t_perp . grad_perp = d/dl". For a general unit
tangent t = t_perp + t_z z_hat the true chain rule is

    dW/dl  =  t_perp . grad_perp W  +  t_z dW/dz
           =  -(t_perp . rho_hat) dW/drho  +  t_z dW/dz

because W depends on the horizontal coordinates only through
rho = |r_perp - r'_perp|. So the dropped piece is exactly

    -(t_perp . rho_hat) * dW/drho ,

which vanishes when t_perp = 0 (vertical) and, in the terms that carry it,
when t_z = 0 (horizontal) -- the two orientations the scope names.

THIS PROBE MEASURES THAT, without assuming it. Three quantities on one
tilted pair, all from the shipped tables:

    LHS   the direct spelling  : sum w W F'  -  [W F]_ends
    CODE  the substituted move : - sum w F t_z dW/dz
    TRUE  the honest move      : - sum w F dW/dl, with dW/dl obtained by
                                 CENTRED FINITE DIFFERENCE of W along the
                                 segment direction -- no analytic claim

LHS == TRUE for every alpha is by-parts working. CODE == TRUE only at
alpha = 0 and 90 is the defect, and (LHS - CODE) is its size.
"""

import numpy as np

from momwire._near_interface import designed_tables

SOIL, K2 = (13.0, 0.005), 2.0 * np.pi * 7.1e6 / 299792458.0
EPS0 = 8.8541878128e-12


def eps_tilde():
    from momwire import _ground_refl

    return _ground_refl.eps_tilde(SOIL, 2.0 * np.pi * 7.1e6, EPS0)


ET = eps_tilde()


def W_at(p_above, p_below):
    """W(rho, z, z') for one above point and one below point."""
    d = np.asarray(p_above, float) - np.asarray(p_below, float)
    rho = float(np.hypot(d[0], d[1]))
    return complex(
        designed_tables(ET, K2, rho, float(p_above[2]), float(p_below[2]))["W"]
    )


def run(alpha_deg, h=0.30, n=64, src=(0.7, 0.0, -0.15), eps_fd=1e-6):
    """One tilted ABOVE segment of length h leaving (0,0,z0) at `alpha_deg`
    from the interface normal, against a fixed below point."""
    a = np.radians(alpha_deg)
    t = np.array([np.sin(a), 0.0, np.cos(a)])  # unit tangent
    p0 = np.array([0.0, 0.0, 0.05])
    u = (np.arange(n) + 0.5) * h / n
    w = np.full(n, h / n)
    pts = p0[None, :] + u[:, None] * t[None, :]

    F = u / h  # a rising tent on this segment
    Fd = np.full(n, 1.0 / h)

    Wv = np.array([W_at(p, src) for p in pts])
    lhs = float("nan")
    end_hi = W_at(p0 + h * t, src) * 1.0  # F(h) = 1
    end_lo = W_at(p0, src) * 0.0  # F(0) = 0
    lhs = np.sum(w * Wv * Fd) - (end_hi - end_lo)

    # CODE: dW/dl -> t_z dW/dz, with dW/dz by the same finite difference so
    # the comparison is about the SUBSTITUTION, not about differentiation.
    dz = np.array([0.0, 0.0, eps_fd])
    dWdz = np.array(
        [(W_at(p + dz, src) - W_at(p - dz, src)) / (2 * eps_fd) for p in pts]
    )
    code = -np.sum(w * F * t[2] * dWdz)

    # TRUE: dW/dl along the segment direction itself.
    dl = eps_fd * t
    dWdl = np.array(
        [(W_at(p + dl, src) - W_at(p - dl, src)) / (2 * eps_fd) for p in pts]
    )
    true = -np.sum(w * F * dWdl)

    # and the piece the substitution drops, measured directly
    drho = -np.sum(w * F * (dWdl - t[2] * dWdz))
    return lhs, code, true, drho


print(f"soil {SOIL}, 7.1 MHz, eps_tilde = {ET:.4f}")
print("alpha: 0 = normal to the interface (vertical), 90 = in-plane\n")
print(
    f"{'alpha':>6s} {'|LHS-TRUE|':>12s} {'|LHS-CODE|':>12s} "
    f"{'rel drop':>10s}   {'dropped term':>26s}"
)
for al in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0):
    lhs, code, true, drho = run(al)
    rel = abs(lhs - code) / max(abs(lhs), 1e-300)
    print(
        f"{al:6.1f} {abs(lhs - true):12.3e} {abs(lhs - code):12.3e} "
        f"{rel:10.3%}   {drho.real:+11.4e}{drho.imag:+11.4e}j"
    )


# ===========================================================================
# MEASURED 2026-09-07, soil (13.0, 0.005), 7.1 MHz, eps_tilde = 13.0000-12.6585j
#
#  alpha   |LHS-TRUE|   |LHS-CODE|   rel drop          dropped term
#    0.0    1.936e-06    1.936e-06     0.001%   -0.0000e+00-0.0000e+00j
#   15.0    3.786e-07    2.769e-02    21.415%   -2.7288e-02+4.6917e-03j
#   30.0    1.580e-06    5.621e-02    57.914%   -5.5479e-02+9.0602e-03j
#   45.0    3.066e-06    8.590e-02   161.057%   -8.4892e-02+1.3118e-02j
#   60.0    3.561e-06    1.171e-01   465.898%   -1.1587e-01+1.6924e-02j
#   75.0    1.935e-06    1.493e-01   161.487%   -1.4788e-01+2.0483e-02j
#   90.0    2.938e-06    1.791e-01   100.000%   -1.7755e-01+2.3544e-02j
#
# READ IT AS: |LHS-TRUE| ~ 1e-6 at EVERY alpha is the finite-difference
# truncation and nothing else -- by parts is exact for any orientation, so the
# defect is not in the by-parts step. |LHS-CODE| is zero at alpha = 0 and
# grows with alpha: the SUBSTITUTION dW/dl -> t_z dW/dz is what fails, exactly
# as "t_perp . grad_perp = d/dl" says it must.
#
# The dropped column IS -(t_perp . rho_hat) dW/drho, computed here as the
# difference of two finite differences rather than assumed. Its alpha
# dependence is the geometric factor alone -- which is why the fix is not a
# table in alpha.
#
# CAVEAT this probe does NOT settle: it tests the move in ISOLATION. Which of
# the assembled sandwich's terms actually make the substitution is a separate
# question, answered by probe2.
