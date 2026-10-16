# Large-λ extraction for the below/below remainder: the derivation

Sources: the Sommerfeld identity and elementary integrals only (any table,
e.g. Gradshteyn & Ryzhik §6.611, §6.621; Watson). No NEC source or manual was
opened.

## 1. The integrands as series times a static kernel

`_integrand_six_below` evaluates, at one node (ρ, h), h = |z| + |z′|,

| i | quantity | integrand |
|---|---|---|
| 0 | ∂²V/∂ρ² | D₂ e^{−γ_m h} λ³ (J₁(x)/x − J₀(x)) |
| 1 | ∂²V/∂z² | D₂ γ_m² e^{−γ_m h} λ J₀(x) |
| 2 | ∂²V/∂ρ∂z | −D₂ γ_m e^{−γ_m h} λ² J₁(x) |
| 3 | (1/ρ)∂V/∂ρ | −D₂ e^{−γ_m h} λ³ J₁(x)/x |
| 4 | V | D₂ e^{−γ_m h} λ J₀(x) |
| 5 | U | D₁ e^{−γ_m h} λ J₀(x) |

with x = λρ, γ_s = (λ² − k_s²)^{1/2}, and the swapped `_d12` pair

    D₁ = 2/(γ_p + γ_m) − 2k_m²/(γ_m S),
    D₂ = 2/(k_p² γ_m + k_m² γ_p) − 2/(γ_m S),        S = k_p² + k_m².

Put u = 1/λ and γ_s = λ s_s(u), where

    s_s(u) = (1 − k_s² u²)^{1/2} = Σ_j C(½, j) (−k_s² u²)^j

on the branch with Re γ > 0 at large real λ, which is the branch `_gamma`
picks. Then D₁ = u·d₁(u) and D₂ = u·d₂(u), with

    d₁(u) = 2/(s_p + s_m) − (2k_m²/S)/s_m,
    d₂(u) = 2/(k_p² s_m + k_m² s_p) − (2/S)/s_m.

At u = 0, d₁ = 1 − 2k_m²/S = (k_p² − k_m²)/S = A_m, the image coefficient.
d₂ cancels at orders u⁰ and u¹: to first order the two terms are both
2/S·(1 + …). The first surviving order is

    d₂(u) = (k_m² A_m / S) u² + O(u⁴),    i.e.  D₂ = (k_m² A_m/S) λ⁻³ + O(λ⁻⁵).

The exponential splits as

    e^{−γ_m h} = e^{−λh} · E(u),    E(u) = exp(−h (s_m(u) − 1)/u)
               = 1 + (k_m² h/2) u + (k_m⁴ h²/8) u² + …

E carries all of h's influence on the asymptotics. It is a series in u whose
coefficients are polynomials in h, so h → 0 is regular term by term.

Each integrand is therefore

    f_i(λ) = g_i(u) · e^{−λh} · B_i(λρ),

with B_i ∈ {J₁/x − J₀, J₀, J₁, J₁/x, J₀, J₀} and

    g₀ = u⁻² d₂ E,  g₁ = u⁻² d₂ s_m² E,  g₂ = −u⁻² d₂ s_m E,
    g₃ = −u⁻² d₂ E, g₄ = d₂ E,           g₅ = d₁ E.

All six are ordinary power series in u. g₄ starts at u², and the other five
start at u⁰. So every integrand except V decays only as fast as
e^{−λh}·(Bessel amplitude ∝ λ^{−1/2}). At grazing h → 0, that is the whole
problem. The tail has to reach λ ≈ ln(1/rtol)/h, and with one panel per π/ρ
that takes ~(ρ/h)·ln(1/rtol)/π ≈ 8/tan θ panels at rtol = 1e-11. The
measured law is ~6.4/tan θ, because the quiet test fires a little earlier
than full e-folding: it also gains the Bessel amplitude's λ^{−1/2}.

The study computes the series coefficients numerically per node, with
truncated power-series arithmetic (`extract.PS`: products, reciprocal, the
binomial square root, and exp of a series with zero constant term). For
Dan's soil at R1 = 1 λ_m and θ = 0.5°, the residual f_i − Σ_{n<n₀+N} falls
as λ^{−N}: from λ = 20|k_m| to 320|k_m| it shrinks 36×, 260× and 8,400× for
N = 1, 2, 3.

## 2. The extracted terms and their closed forms

Subtract, for each i, the first N terms of g_i:

    T_i(λ) = Σ_{n = n₀ᵢ}^{n₀ᵢ+N−1} c_{i,n} λ^{−n} (1 − e^{−λb})^{M} e^{−λh} B_i(λρ).

The regulariser (1 − e^{−λb})^M (b > 0, M ≥ 0 per term, below) does not
change T at large λ beyond an exponentially small amount. It makes each term
integrable at λ = 0 and analytic everywhere. Because T_i is entire, its
integral along the head's detour equals its integral along the real axis, so
one subtraction works on the whole contour:

    I_i = ∫_C (f_i − T_i) dλ  +  ∫_0^∞ T_i dλ,

with the second piece in closed form. The building block is the Sommerfeld
identity at zero wavenumber,

    Φ₀(t) := ∫_0^∞ e^{−λt} J₀(λρ) dλ = 1/R,    R = (ρ² + t²)^{1/2}.

Write λ^{−m} for m-fold integration in t. Define Φ_m by
dΦ_m/dt = −Φ_{m−1}, so that Φ_m(t) equals the formally divergent
∫ λ^{−m} e^{−λt} J₀ dλ up to a polynomial in t of degree ≤ m − 1. With
L(t) = ln(t + R):

    Φ₁ = −L
    Φ₂ = tL − R
    Φ₃ = −[(t²/2 − ρ²/4) L − (3/4) t R]
    Φ₄ = (t³/6 − ρ²t/4) L + (5/12) ρ² R − (11/36) R³
    Φ₅ = −[(t⁴/24 − ρ²t²/8 + ρ⁴/64) L − (25t³/288 − 55ρ²t/576) R]

Each was checked to 1e-41 with 40-digit mpmath (dΦ_m/dt + Φ_{m−1} at three
(t, ρ) points).

Writing (1 − e^{−λb})^M = Σ_j C(M,j)(−1)^j e^{−jλb} turns the regularised
integral into an M-th forward difference in t:

    ∫_0^∞ λ^{−m} (1 − e^{−λb})^M e^{−λh} J₀(λρ) dλ = Σ_{j=0}^{M} (−1)^j C(M,j) Φ_m(h + jb).

This is exact once M ≥ m, because the M-th difference annihilates Φ's
polynomial ambiguity. The other Bessel forms follow from
J₁(λρ) = −λ⁻¹ ∂_ρ J₀(λρ):

    ∫ λ^{−n} e^{−λt} J₁(λρ)    ↔  −∂_ρ Φ_{n+1}(t),
    ∫ λ^{−n} e^{−λt} J₁(λρ)/(λρ) ↔ −(1/ρ) ∂_ρ Φ_{n+2}(t),

with ∂_ρ L = ρ/(R(t + R)) and ∂_ρ R = ρ/R (`extract.dphi_drho`). The
lowest orders reproduce the classical results:

    ∫ e^{−λh} J₁ dλ = (R − h)/(ρR),    ∫ e^{−λh} J₁(λρ)/(λρ) dλ = (R − h)/ρ² = 1/(R + h).

M is the smallest value that makes the regularised integral converge
(`extract.reg_order`): M = 0 when the term already converges at λ = 0
(J₀ with n = 0; J₁ with n ≤ 1; J₁/x with n = 0), and otherwise M equals
the index of the Φ used. Every regularised closed form was checked against
direct 25-digit mpmath quadrature of the same regularised integrand at two
(h, ρ, b) points, for every form and order N ≤ 3 needs. The worst
disagreement was 1.2e-13, which is the finite difference's cancellation at
M = 5.

### The h → 0 and ρ → 0 ends

Every closed form stays finite as h → 0, because Φ_m(t) is smooth at t = 0
for ρ > 0. The regularised differences sample t = h + jb ≥ jb, so they never
see t = 0 except through j = 0. At ρ → 0, the expressions with an explicit
1/ρ (the J₁/x forms) are taken as written. The study does not need ρ → 0,
since the grid's smallest ρ is R1·cos 80°. A production version would use
the cancellation-free spellings, e.g. (R − h)/ρ² = 1/(R + h).

### Choice of b

b sets where the regulariser switches on. It must be large enough that
T matches f in the tail: there the residual carries an extra
c·M·e^{−λb}·λ^{−n} term. It must be small enough that the finite
differences do not cancel, and that T is not huge against f on the head:
for n ≥ 1, ∫λ^{−n}… over λ < 1/b grows like (kb)^n. The study uses
b = 1/a with a = 1.1 max(k_p, |k_m|), the head/tail boundary. The extra
e^{−λ/a} term then decays to 1e-11 by λ ≈ 25a, well inside the panels the
tail walks anyway. `b_scan` in the results tests b = a⁻¹·{0.5, 1, 2, 4}.

## 3. The tail accelerator: weighted averages (Mosig–Michalski)

The current tail sums panels between consecutive J₀ zeros (lattice π/ρ, also
capped at 1.5/h), stops after two quiet panels, and falls back to Wynn ε when
the 24,000-panel budget runs out. The partial sums S_n at the lattice points
ξ_n form an alternating sequence whose remainder has a known form:

    S − S_n ≈ ω_n Σ_{i≥0} c_i ξ_n^{−i},    ω_n = (−1)^n ξ_n^{α−1/2} e^{−ξ_n h},

where α is the power of λ multiplying the Bessel factor in the integrand
(α − 1/2 includes J's λ^{−1/2} amplitude, and J₁/x gives a further −1).
This is the Mosig–Michalski weighted-averages model (Michalski 1998;
Mosig & Michalski 2021). With analytic ω_n the unknowns c_i are eliminated
exactly by Sidi's W-algorithm, which is equivalent to the MM recursion:

    M_n⁽⁰⁾ = S_n/ω_n,  N_n⁽⁰⁾ = 1/ω_n,
    M_n⁽ᵏ⁾ = (M_{n+1}⁽ᵏ⁻¹⁾ − M_n⁽ᵏ⁻¹⁾)/(ξ_{n+k}⁻¹ − ξ_n⁻¹)   (same for N),
    S ≈ M_n⁽ᵏ⁾ / N_n⁽ᵏ⁾.

It runs per component (each of the six integrals has its own α, and under
extraction α drops by N). It uses only the last ≤ 11 lattice-aligned sums,
and stops when two successive estimates agree to rtol × (answer scale). The
current quiet test stays in force as well, so a tail that goes quiet by
itself (steep nodes, where the 1.5/h cap makes panels shorter than the
lattice) stops exactly as it does today. Where WA engages, it replaces the
walk to λ ≈ ln(1/rtol)/h with a dozen panels past the head.
