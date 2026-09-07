"""momwire#883, step 2 — the survey, starting with the generator itself.

`scripts/derive_bspline_static_moments.py` is the named blocker, and the
issue's shape says "re-run it with a larger MAX_D". Reading it first, that
would silently emit WRONG moments, because the EK family's derivation route
is only valid for q <= 2 and the script does not know it.

The route (module docstring) is by parts:

    int_A^B Q(t) H''(s-t) dt = -[Q H']_A^B - [Q' H]_A^B + int_A^B Q''(t) H(s-t) dt

with Q(t) = (t-A)^q. The docstring writes the last term as
`q(q-1) * int H(s-t) dt` and the code (line 214) emits `-q(q-1) * J_p0`. That
is right ONLY because Q'' = q(q-1) is a CONSTANT for q <= 2. In general

    Q''(t) = q(q-1) (t-A)^(q-2)

so the term is `-q(q-1) * J_{p, q-2}`, which coincides with J_{p,0} at q = 2
and at q = 0,1 (where the factor is zero and the call is dead) -- every value
the script has ever been run at. At q = 3 it should be J_{p,1} and the script
would emit J_{p,0}.

This probe does not argue that; it measures it, against direct 2-D numerical
quadrature of the ONE integrand Dg -- never as two families (the module
docstring's own invariant: /R^3 and /R^5 are individually O(1) while their sum
is O(a^2), so differencing them would ship a catastrophic cancellation).

Three columns per row:
    numeric     adaptive 2-D quadrature of the true integrand
    as-written  the by-parts expression the script emits today
    corrected   the same with J_{p,0} -> J_{p,q-2}

If the reading is right, the three agree for q <= 2 and only `corrected`
agrees at q = 3. If `as-written` also matches at q = 3, the reading is wrong
and MAX_D = 3 is safe as it stands.
"""

import sys
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE.parents[2] / "scripts"))

from derive_bspline_static_moments import J_CALL  # noqa: E402
from momwire._bspline_static_moments import J_static_moment  # noqa: E402

# A concrete same-edge pair. Distinct corners, a well away from zero, and
# alpha != A so no accidental symmetry can make a wrong formula look right.
ALPHA, BETA = 0.30, 0.95
A_LO, B_HI = 0.10, 0.70
RAD = 0.023


def dg(xi, a):
    """The ONE extended-kernel static correction integrand."""
    r2 = xi * xi + a * a
    r = np.sqrt(r2)
    return -(a**2) / (2.0 * r**3) + 3.0 * a**4 / (4.0 * r**5)


def numeric(p, q):
    def inner(t, s):
        return (s - ALPHA) ** p * (t - A_LO) ** q * dg(s - t, RAD)

    val, err = integrate.dblquad(
        inner,
        ALPHA,
        BETA,
        lambda _s: A_LO,
        lambda _s: B_HI,
        epsabs=1e-14,
        epsrel=1e-13,
    )
    return val, err


def _byparts(p, q, j_index_q):
    """The script's own construction, with the J call's q index made a knob.

    `j_index_q = 0` reproduces the emitted code exactly; `q - 2` is the
    correction. Everything else is copied structurally from `derive_ek_all`
    so this measures the SCRIPT, not a re-derivation that might differ for
    unrelated reasons.
    """
    alpha, beta, A, B, a = sp.symbols("alpha beta A B a", real=True, positive=True)
    xi, d = sp.symbols("xi d", real=True)
    R = sp.sqrt(xi**2 + a**2)
    poly = sp.expand((xi + d) ** p)
    f0 = sp.simplify(sp.integrate(poly / R, xi))
    f1 = sp.simplify(sp.integrate(poly * xi / R**3, xi))

    def corner(F, c):
        shift = {d: c - alpha}
        return F.subs({xi: beta - c, **shift}) - F.subs({xi: alpha - c, **shift})

    q_at_b = (B - A) ** q
    q_at_a = sp.Integer(1 if q == 0 else 0)
    qp_at_b = q * (B - A) ** (q - 1) if q >= 1 else sp.Integer(0)
    qp_at_a = sp.Integer(1 if q == 1 else 0)
    bracket = -q_at_b * corner(f1, B) + q_at_a * corner(f1, A)
    bracket += qp_at_b * corner(f0, B) - qp_at_a * corner(f0, A)
    bracket += -sp.Integer(q * (q - 1)) * J_CALL
    expr = sp.Rational(1, 4) * a**2 * bracket

    j_val = J_static_moment(p, j_index_q, ALPHA, BETA, A_LO, B_HI, RAD)
    return float(
        sp.re(
            sp.N(
                expr.subs(
                    {
                        alpha: ALPHA,
                        beta: BETA,
                        A: A_LO,
                        B: B_HI,
                        a: RAD,
                        J_CALL: sp.Float(j_val),
                    }
                )
            )
        )
    )


def main():
    print("momwire#883 — is the EK by-parts route valid at q = 3?")
    print(f"pair: alpha={ALPHA} beta={BETA} A={A_LO} B={B_HI} a={RAD}\n")
    print(
        f"  {'p':>2} {'q':>2} {'numeric':>16} {'as-written':>16} {'corrected':>16}"
        f" {'rel(as)':>10} {'rel(corr)':>10}"
    )
    verdict = {}
    # p only to 2: the correction term calls J_{p,q-2}, and the SHIPPED J
    # family is [0,2]^2. p = 3 cannot be checked until J is regenerated too,
    # which is the other half of the work and not what this probe is about.
    for p in range(3):
        for q in range(4):
            num, err = numeric(p, q)
            assert abs(err) < 1e-10 * max(abs(num), 1e-12) + 1e-13, (
                f"quadrature did not converge for p={p} q={q}: err={err:.3e} "
                f"-- the reference is not a reference"
            )
            as_written = _byparts(p, q, 0)
            corrected = _byparts(p, q, max(0, q - 2))
            scale = max(abs(num), 1e-30)
            r_as = abs(as_written - num) / scale
            r_co = abs(corrected - num) / scale
            print(
                f"  {p:>2} {q:>2} {num:16.9e} {as_written:16.9e} {corrected:16.9e}"
                f" {r_as:10.2e} {r_co:10.2e}"
            )
            verdict[(p, q)] = (r_as, r_co)
    print()
    bad_as = [k for k, (a_, _) in verdict.items() if a_ > 1e-9]
    bad_co = [k for k, (_, c_) in verdict.items() if c_ > 1e-9]
    print(f"  as-written disagrees at: {sorted(bad_as)}")
    print(f"  corrected  disagrees at: {sorted(bad_co)}")
    if not bad_as:
        print("\n  READING WRONG: the emitted route already agrees at q = 3.")
    elif all(q == 3 for _, q in bad_as) and not bad_co:
        print("\n  CONFIRMED: the emitted route is wrong exactly at q = 3, and")
        print("  J_{p,0} -> J_{p,q-2} fixes it with q <= 2 unchanged.")


if __name__ == "__main__":
    main()
