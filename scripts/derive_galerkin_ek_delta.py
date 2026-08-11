"""Derive the E_ρ/E_z field operators applied to the extended-kernel DELTA,
and check the transcription that landed in `sinusoidal._folded_ek_delta_fields`
(momwire#246 unit A).

What this derives
-----------------
§1 pins the two field operators and their shared prefactor by reading the
SHIPPED reduced closed forms of `_field_components_bcast` back through their
own derivation (NEC2 Theory Manual Eqs 76-79) — no constant is fitted, every
sign is decided by a symbolic identity against the code as written:

    E_z[s] = −pref_z · ∫ s(ξ)·(k² + ∂²/∂z²) K dξ
    E_ρ[s] = −pref_z · ∫ s(ξ)·(∂²/∂ρ∂z)   K dξ,     pref_z = jη/(4πk)

(the `pref_rho = −pref_z/ρ` the reduced path carries is a factoring of its own
closed forms, not a second normalization — its 1/ρ is paid back by the ρ²
inside its brackets).

§2 derives the kernel those operators are applied to. The extended thin-wire
kernel is the source tube's circumferential average; writing the reduced
kernel as a function of u = R² = ρ² + ζ²,

    g(u) = e^{−jk√u}/√u,   R(φ)² = u + a² − 2aρ·cos φ

and averaging the Taylor expansion in (R² − u) — ⟨R²−u⟩ = a², ⟨(R²−u)²⟩ =
a⁴ + 2a²ρ² — gives the DELTA kernel through O(a²):

    W(ρ, ζ) = a²·g′(u) + a²ρ²·g″(u)                                     (*)

At ρ = a, which is what EK eligibility MEANS (coaxial, equal radius, so the
observer sits on its own wire's surface on the source axis), (*) is NEC Eq
89's `(fac − 1)·G_red` term for term — a²g′ = −(a²/2R²)C1·G and a⁴g″ =
(a⁴/4R⁴)C2·G, i.e. exactly `_bspline_kernels._ek_factor` (momwire#249 §1.2).
That identity is checked below.

Keeping the ρ² of (*) rather than substituting a² for it is invisible to E_z
(∂/∂z reaches u only through ζ) and decisive for E_ρ: Eq 89's factor is a
function of R and a alone and has no honest ρ-derivative to give. Measured on
the coaxial box the fill visits, differentiating the factor form in ρ lands
E_ρ at HALF the exact circumferential average — and half of NEC's own EKSCX,
which agrees with the exact average. (*) reproduces both.

Both derivatives collapse onto the reverse Bessel polynomials, since

    g⁽ⁿ⁾(u) = (−½)ⁿ·e^{−jkR}·Aₙ(jkR)/R^{2n+1}
    A₁ = 1+x, A₂ = 3+3x+x², A₃ = 15+15x+6x²+x³,
    A₄ = 105+105x+45x²+10x³+x⁴

so the two transcribed brackets are

    (k² + ∂²_z)W = a²·[ k²(g′ + ρ²g″) + 2(g″ + ρ²g‴) + 4ζ²(g‴ + ρ²g⁗) ]
    ∂²_{ρz} W    = a²·ρζ·[ 8g‴ + 4ρ²g⁗ ]

`W` is linear in a², so pulling that factor out to the very end makes the
whole delta EXACTLY 0.0 at a = 0 in IEEE rather than merely in the limit —
the structural collapse gate (G-A3). `W` is also bounded at ζ = 0, which is
what lets the fill integrate it against the folded source shape by plain
Gauss–Legendre with the shape evaluated POINTWISE as −2·sin²(kξ/2) (momwire
#246 design §1, experiment `e1_fold_poison.py`).

Method note
-----------
§1 works in the (R, ζ) chart at fixed ρ, where d/dξ = ∂_ξ − ∂_ζ − (ζ/R)∂_R.
That keeps every expression rational in (R, ζ, e^{−jkR}) instead of nested
under a radical, which is the difference between seconds and never for
`sp.simplify` on these.

Run it::

    python scripts/derive_galerkin_ek_delta.py

It prints the derivation and asserts, so a broken transcription fails loudly.
Nothing is generated: the numpy is short enough to write by hand in named
steps (the rounding rule at `_folded_cos_fields`), and this script is the
record that what was written is what was derived.
"""

import numpy as np
import sympy as sp

# R and ζ are §1's chart; ρ is a parameter (constant along the source
# integration), tied to them by ρ² = R² − ζ². ξ is the source coordinate,
# ζ = z − ξ. `u` is §2's variable, u = R².
R, rho, a, k, u = sp.symbols("R rho a k u", positive=True)
zeta, xi = sp.symbols("zeta xi", real=True)
J = sp.I
E = sp.exp(-J * k * R)  # the one transcendental; everything else is rational


def d_dxi(expr):
    """d/dξ of a function of (R, ζ, ξ) at fixed ρ and z.

    ζ = z − ξ so ∂ζ/∂ξ = −1, and R = √(ρ²+ζ²) so ∂R/∂ξ = −ζ/R.
    """
    return sp.diff(expr, xi) - sp.diff(expr, zeta) - (zeta / R) * sp.diff(expr, R)


def chain_z2(f):
    """(∂²/∂z²) of f(R), in the chart: f″·ζ²/R² + f′·(1/R − ζ²/R³)."""
    return sp.diff(f, R, 2) * zeta**2 / R**2 + sp.diff(f, R) * (1 / R - zeta**2 / R**3)


def chain_rho_z(f):
    """(∂²/∂ρ∂z) of f(R), in the chart: ζρ·(f″/R² − f′/R³)."""
    return zeta * rho * (sp.diff(f, R, 2) / R**2 - sp.diff(f, R) / R**3)


def rationalize(expr):
    """Kill ρ in favour of (R, ζ) and cancel."""
    e = sp.expand(expr).subs(rho, sp.sqrt(R**2 - zeta**2))
    return sp.simplify(sp.cancel(sp.expand(e)))


# ---------------------------------------------------------------------------
# §1 — pin the field operators against the SHIPPED reduced closed forms.
#
# `_field_components_bcast` spells, for the source shape s over ξ ∈ [−H, H]
# with Δz = z − ξ = ζ, r = R, P = (1+jkR)/R²:
#
#   Ez_const   = −pref_z·( [P·G·ζ]_{ξ=−H}^{+H} + k²∫G dξ )
#   Erho_const = −pref_z·( [P·G·ρ]_{ξ=−H}^{+H} )
#   Ez_sin     = +pref_z·( [G·(k·cos kξ − P·ζ·sin kξ)]_{−H}^{+H} )
#   Erho_sin   = −pref_z/ρ·( [G·(kζ·cos kξ + (1 − ζ²P)·sin kξ)]_{−H}^{+H} )
#
# The claim under test is that each equals −pref_z·∫ s·L dξ with L the
# operator above — i.e. that d/dξ of the operator antiderivative reproduces
# the integrand. The brackets are read off the shipped source and the sign of
# every one of them is decided here.
# ---------------------------------------------------------------------------
G = E / R
P = (1 + J * k * R) / R**2  # the shipped `(1 + 1j*k*r0)/(r0*r0)`, G not folded in

_SHAPES = {"const": sp.Integer(1), "sin": sp.sin(k * xi)}

_ANTIDERIV = {
    ("const", "z"): P * G * zeta,
    ("const", "rho"): P * G * rho,
    ("sin", "z"): -(G * (k * sp.cos(k * xi) - P * zeta * sp.sin(k * xi))),
    ("sin", "rho"): (
        G * (k * zeta * sp.cos(k * xi) + (1 - zeta**2 * P) * sp.sin(k * xi))
    )
    / rho,
}


def check_field_operators():
    for name, s in _SHAPES.items():
        for which in ("z", "rho"):
            if which == "z":
                integrand = s * (k**2 * G + chain_z2(G))
                if name == "const":
                    # this one closed form keeps k²∫G explicitly
                    integrand = integrand - k**2 * s * G
            else:
                integrand = s * chain_rho_z(G)
            resid = rationalize(d_dxi(_ANTIDERIV[(name, which)]) - integrand)
            assert resid == 0, (name, which, resid)
            print(f"  ok  E_{which}[{name}] = -pref_z * int s*L_{which}[G] dxi")
    print("      (pref_rho = -pref_z/rho is a factoring, not a normalization)")


# ---------------------------------------------------------------------------
# §2 — the delta kernel W, its Eq 89 identity, and the two operator brackets.
# ---------------------------------------------------------------------------
def g_of_u():
    """The reduced kernel as a function of u = R²."""
    return sp.exp(-J * k * sp.sqrt(u)) / sp.sqrt(u)


def bessel_poly(n):
    """Reverse Bessel polynomial Aₙ(x), from the recursion the u-derivatives
    of g obey: A_{n+1} = (2n+1)·Aₙ + x·Aₙ − x·Aₙ′."""
    x = sp.Symbol("x")
    A = sp.Integer(1)
    for m in range(n):
        A = sp.expand((2 * m + 1) * A + x * A - x * sp.diff(A, x))
    return A


def check_derivative_closed_form():
    g = g_of_u()
    x = sp.Symbol("x")
    for n in range(1, 5):
        lhs = sp.diff(g, u, n).subs(u, R**2)
        An = bessel_poly(n).subs(x, J * k * R)
        rhs = sp.Rational(-1, 2) ** n * sp.exp(-J * k * R) * An / R ** (2 * n + 1)
        assert sp.simplify(sp.expand(lhs - rhs)) == 0, n
        print(f"  ok  d^{n}g/du^{n} = (-1/2)^{n} e^(-jkR) A{n}(jkR) / R^{2 * n + 1}")


def check_eq89_agreement():
    """W|_{ρ=a} == (fac − 1)·G_red, NEC Eq 89's factor (`_ek_factor`)."""
    g = g_of_u()
    W = a**2 * sp.diff(g, u) + a**2 * rho**2 * sp.diff(g, u, 2)
    W_a = W.subs(rho, a).subs(u, R**2)
    c1 = 1 + J * k * R
    c2 = 3 * c1 - (k * R) ** 2
    eq89 = (a**4 / (4 * R**4) * c2 - a**2 / (2 * R**2) * c1) * (sp.exp(-J * k * R) / R)
    assert sp.simplify(sp.expand(W_a - eq89)) == 0
    print("  ok  W at rho=a equals (fac-1)*G_red   (NEC Eq 89 / _ek_factor)")


def _W_and_derivs():
    """W and g′..g⁗ in the (ρ, ζ) chart, u spelled out as ρ²+ζ²."""
    g = g_of_u()
    uu = rho**2 + zeta**2
    gd = {n: sp.diff(g, u, n).subs(u, uu) for n in range(1, 5)}
    return a**2 * gd[1] + a**2 * rho**2 * gd[2], gd


def check_operator_forms():
    W, gd = _W_and_derivs()
    lhs_z = k**2 * W + sp.diff(W, zeta, 2)  # ∂/∂z = ∂/∂ζ at fixed ξ
    lhs_r = sp.diff(W, rho, 1, zeta, 1)
    rhs_z = a**2 * (
        k**2 * (gd[1] + rho**2 * gd[2])
        + 2 * (gd[2] + rho**2 * gd[3])
        + 4 * zeta**2 * (gd[3] + rho**2 * gd[4])
    )
    rhs_r = a**2 * rho * zeta * (8 * gd[3] + 4 * rho**2 * gd[4])
    assert sp.simplify(sp.expand(lhs_z - rhs_z)) == 0
    assert sp.simplify(sp.expand(lhs_r - rhs_r)) == 0
    print("  ok  (k^2 + d2_z)W  = a^2 [k^2(g1 + p^2 g2) + 2(g2 + p^2 g3)")
    print("                          + 4 z^2 (g3 + p^2 g4)]")
    print("  ok  d2_(rho,z) W   = a^2 rho z [8 g3 + 4 p^2 g4]")


# ---------------------------------------------------------------------------
# §3 — the transcription, replayed exactly as `_folded_ek_delta_fields` spells
# it, against a 40-digit evaluation of the symbolic forms.
# ---------------------------------------------------------------------------
def _transcribed(R_, zeta_, rho_, a_, k_):
    """The named-step numpy of `sinusoidal._folded_ek_delta_fields`, verbatim
    (a² pulled to the end, exactly as there)."""
    r2 = R_ * R_
    x = 1j * (k_ * R_)
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    a1 = 1.0 + x
    a2 = 3.0 + 3.0 * x + x2
    a3 = 15.0 + 15.0 * x + 6.0 * x2 + x3
    a4 = 105.0 + 105.0 * x + 45.0 * x2 + 10.0 * x3 + x4
    inv2 = 1.0 / r2
    phase = np.exp(-1j * (k_ * R_))
    base = phase / R_
    g1 = -0.5 * (base * a1 * inv2)
    g2 = 0.25 * (base * a2 * (inv2 * inv2))
    g3 = -0.125 * (base * a3 * (inv2 * inv2 * inv2))
    g4 = 0.0625 * (base * a4 * (inv2 * inv2 * inv2 * inv2))
    rho2 = rho_ * rho_
    t_k = (k_ * k_) * (g1 + rho2 * g2)
    t_c = 2.0 * (g2 + rho2 * g3)
    t_z = (4.0 * zeta_ * zeta_) * (g3 + rho2 * g4)
    l_z = (t_k + t_c) + t_z
    l_r = ((8.0 * g3 + 4.0 * rho2 * g4) * rho_) * zeta_
    a2c = a_ * a_
    return l_z * a2c, l_r * a2c


def check_transcription():
    import mpmath

    mpmath.mp.dps = 40
    W, _ = _W_and_derivs()
    f_lz = sp.lambdify((zeta, rho, a, k), k**2 * W + sp.diff(W, zeta, 2), "mpmath")
    f_lr = sp.lambdify((zeta, rho, a, k), sp.diff(W, rho, 1, zeta, 1), "mpmath")
    rng = np.random.default_rng(2461)
    worst = 0.0
    for _ in range(80):
        a_ = float(10 ** rng.uniform(-5, -2))
        rho_ = a_ * float(10 ** rng.uniform(0, 1.5))
        zeta_ = float(10 ** rng.uniform(-4, 0)) * float(rng.choice([-1.0, 1.0]))
        k_ = float(10 ** rng.uniform(-1, 1))
        R_ = float(np.hypot(rho_, zeta_))
        lz, lr = _transcribed(R_, zeta_, rho_, a_, k_)
        ez = complex(f_lz(zeta_, rho_, a_, k_))
        er = complex(f_lr(zeta_, rho_, a_, k_))
        scale = max(abs(ez), abs(er))
        worst = max(worst, abs(lz - ez) / scale, abs(lr - er) / scale)
    print(f"\n  transcription vs 40-digit reference: worst rel {worst:.2e}")
    assert worst < 1e-13, worst


def main():
    print("1. field operators, against the shipped reduced closed forms")
    check_field_operators()
    print("\n2. the delta kernel and its operator forms")
    check_derivative_closed_form()
    check_eq89_agreement()
    check_operator_forms()
    print("\n3. the numpy transcription")
    check_transcription()
    print("\nall checks pass")


if __name__ == "__main__":
    main()
