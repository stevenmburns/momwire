# The exact ring kernel as a series in k (momwire#1454, step 3)

Public mathematics only: the binomial and exponential series, the arithmetic-geometric mean, elementary antiderivatives, Gauss-Legendre and product integration, and the trapezoid rule on a periodic integrand. Public context for the kernel is King (1956), Werner (IEEE TAP 1998), Wilton and co-workers on singularity extraction, Davies, Duncan and Funken on the cylindrical kernel's static part, and Fikioris and Valagiannopoulos (PIER 2005). Nothing here comes from NEC-4.2 or NEC-5 material.

The code is `src/momwire/_exact_kernel_series.py` (the reference) and `src/momwire/_accel_exact_kernel_series.cpp` (the production route). The harnesses that produced the numbers are in `scratch/1454-split/`.

## 1. What is being computed

A point- or path-tested solver (razor-2p) needs, for an observer point on the surface of a straight wire of radius a and a source segment on the same wire, the two moments

    N_q(c, h; k) = ∫₀^h v^q K(c + v) dv,   q = 0, 1,

of the exact ring kernel

    K(z) = (1/2π) ∮ e^{-jkR} / (4πR) dφ,   R² = z² + β²,   β = 2a sin θ,   θ = φ/2.

Here c is the segment's start relative to the observer along the direction the segment runs and h is its length (`_exact_kernel.point_moments`). Before this step every wavenumber integrated K again from scratch: about 100 kernel evaluations per pair near the observer and 16 per pair elsewhere, each with one or two AGMs and a series in k.

## 2. The split

Write ⟨·⟩ for the ring average (1/2π)∮ dφ = (2/π)∫₀^{π/2} dθ. For any real reference distance R_c,

    e^{-jkR} / R = e^{-jkR_c} Σ_{n≥0} (-jk)^n (R - R_c)^n / (n! R),

so

    N_q(c, h; k) = e^{-jkR_c} Σ_{n≥0} (-jk)^n / n! · μ_{n,q}(c, h),          (1)
    μ_{n,q}(c, h) = (1/4π) ∫₀^h v^q ⟨(R - R_c)^n / R⟩(c + v) dv.              (2)

With R_c = 0 this is the split of the issue: the n = 0 term is the static kernel 1/R, which carries the log singularity, and n ≥ 1 is the dynamic remainder (e^{-jkR} - 1)/R, which is bounded.

**Every μ is real and depends on the pair's geometry alone.** That is the whole lever. A geometry's coefficients are integrated once, and a new frequency is one Horner evaluation of (1) per unique pair, plus one complex exponential where R_c ≠ 0. So the cache holds all the μ, not only the static n = 0.

## 3. How many terms

Let x = |k| · max |R - R_c| over the pair and the ring. Since |(R - R_c)^n / R| ≤ (x/|k|)^n / R,

    | Σ_{n>N} (-jk)^n μ_{n,q} / n! |  ≤  M_q · Σ_{n>N} x^n / n!  ≤  M_q · (x^N / N!) · (x/(N+1)) / (1 - x/(N+2)),

with M_q = (1/4π)∫ v^q ⟨1/R⟩ dv, the pair's own static moment. The series stops at the first N with x^N / N! ≤ 10⁻¹⁷ (`SERIES_TOL`, the truncation `_exact_kernel._near_terms` already used), which puts the tail below 10⁻¹⁷ of the static moment for every x served.

N is fixed when a pair is built, for the largest x the pair's rule is ever used at:

| rule | R_c | bound on x | N |
|---|---|---|---|
| NEAR, PLAIN FAR | 0 | `X_MAX` = 4 | 34 |
| CENTRED | middle of the pair's range of R | `X_CENTRED` = 1 | 19 |

The second budget is rounding. The terms of e^{-jx} grow to about e^x / √(2πx) before they fall, so the sum loses log₁₀ of that in digits: at x = 4 the largest term is 10.7, one digit. That is why `X_MAX` is 4 and not larger, and why a pair far along a wire must not use R_c = 0: its kR is the wire's electrical length, not the segment's.

`serves(k, group_reach(h, a))` is the matching test on a whole group: with h_max the group's longest segment, a NEAR pair has R < hypot(2 h_max, 2a), a PLAIN FAR one R < hypot(6a + h_max, 2a), and a CENTRED one |R - R_c| ≤ h_max/2 + a/6 (§6). A group passes when k times the first two is at most 4 and k (h_max/2 + a) is at most 1. In practice that means segments shorter than about 0.3 λ. A group that fails at some frequency keeps the direct rule at that frequency, whole. The choice reads geometry and k only, never the state of a cache, so a frequency's answer does not depend on what was solved before it.

## 4. Which rule a pair takes

Let d be the distance from the observer to the nearer end of the segment (0 when the observer is on it).

- **NEAR**: d < h. The pair is closer than its own length. R_c = 0, closed form in z (§5).
- **CENTRED**: d ≥ h and d ≥ 6a. Gauss-Legendre in v about the pair's own phase centre (§6).
- **PLAIN FAR**: d ≥ h and d < 6a (a fat wire's neighbours). R_c = 0, Gauss-Legendre in v on the ring averages (§7).

**Why the closed form is not used everywhere.** Its result is a difference of an antiderivative at the two ends. For a pair at distance c ≫ h the two values agree to c/h digits, and the first moment loses (c/h)² because it subtracts c times the zeroth. On a 59-segment wire that is 10⁻¹³ of the largest entry of the row from rounding alone. For d < h the factor is at most 3. In the other direction, a Gauss-Legendre rule in v needs the kernel analytic in a neighbourhood of the interval, and K has its singularities at z = ±iβ, at distance ≥ d from it. With d ≥ h the Bernstein ellipse of the interval has parameter ρ ≥ 3 + √8 = 5.83, and the 16-point rule's error is of order ρ⁻³² = 3·10⁻²⁵. With d < h that guarantee is gone, which is exactly where the existing code switched to graded chunks and the log split. So d = h is the switch in both directions, and it is the one `_exact_kernel.point_moments` already used between its one-rule and graded branches.

## 5. NEAR: closed form in z, a short integral in θ

R_c = 0, so the integrand of (2) is v^q R^{n-1}. Put p = n - 1.

**Even p (odd n).** ⟨R^{2j}⟩ is a polynomial of degree 2j in z, so the 16-point Gauss-Legendre rule in v integrates v^q ⟨R^{2j}⟩ with no singularity involved. It is exact through degree 31; beyond that (n ≥ 33) the terms carry a weight x^n/n! below 10⁻¹⁷.

**Odd p (even n).** Exchange the two integrals and do the z-integral first. With z₀ = c, z₁ = c + h and R_i = √(z_i² + β²),

    ∫ R⁻¹ dz = asinh(z/β),
    ∫ R^p dz = [ z R^p + p β² ∫ R^{p-2} dz ] / (p + 1),          p = 1, 3, 5, ...
    ∫ z R^p dz = R^{p+2} / (p + 2).

The first is the standard integral; the third is immediate from d(R²) = 2z dz; the second follows from d(z R^p)/dz = R^p + p z² R^{p-2} and z² = R² - β². The moments about the segment's start are ∫ (z - z₀)^q R^p dz.

Every endpoint difference is taken in a form that adds terms of one sign, because on a fat wire (h ≪ a) R₁ and R₀ agree to (a/h)² digits:

    R₁² - R₀² = h (z₁ + z₀),                       exactly a product
    E₁ = R₁ - R₀ = h (z₁ + z₀) / (R₁ + R₀),
    E_{p+2} = R₁^{p+2} - R₀^{p+2} = R₁² E_p + R₀^p (R₁² - R₀²),
    z₁ R₁^p - z₀ R₀^p = h R₁^p + z₀ E_p,

and, with both ends on one side of the observer (0 < z₀ < z₁),

    asinh(z₁/β) - asinh(z₀/β) = log1p( h (1 + (z₁ + z₀)/(R₁ + R₀)) / (z₀ + R₀) ).

What is left is one integral over θ ∈ [0, π/2] of a function of θ that is analytic on the interval.

**The log singularity.** When the observer is on the segment (z₀ ≤ 0 ≤ z₁) the asinh terms no longer cancel each other's singular parts: asinh(z/β) = sgn z · [ln(|z| + R) - ln β], and β → 0 at θ = 0. The integrand then has the form A(θ) + B(θ) ln θ with A and B analytic, B = -(sgn z₁ - sgn z₀) c_j β^{2j} from the recurrence (c_j = Π_{i≤j} (2i-1)/(2i)), and

    A = Σ ± ln(|z_i| + R_i) - (sgn z₁ - sgn z₀) ln(β/θ)      at p = -1.

The chunk of θ that holds 0 is integrated by the product rule that is exact for (polynomial) · ln θ on Gauss-Legendre nodes, the weights `_exact_kernel._gl01_log` builds for the z-space log split. On every other chunk asinh is evaluated directly.

The first version of this subtracted ln β over the whole interval and added its closed form back, ⟨β^{2j} ln β⟩ = (2a)^{2j} w_j (ln a + H_{2j} - H_j) with w_j = C(2j, j)/4^j. That is correct and it was measured: on a fat wire's self term (h/a = 0.05) it cost 1.9·10⁻¹⁴, because away from θ = 0 the two halves are each of order ln a and their sum is of order h/a. Treating the singularity only where it is removes the cancellation (measured below, same case: at the reference's own noise).

**The θ rule.** ln(|z| + R) and R are analytic in θ except at the branch points of R, sin θ = ± i|z|/2a, that is θ = ± i asinh t with t = |z|/2a, and their images under θ → θ + π. Take t from the nearer non-zero end. The rule is Gauss-Legendre on the chunks [0, t], [t, 3t], [3t, 9t], ... up to π/2 (16 points; 20 on a chunk using the log rule). For [0, t] the ellipse through i·asinh t has ρ ≥ 3.8, for every later chunk ρ ≥ 2 + √3 = 3.73, so each chunk's error is of order 3.73⁻³² = 5·10⁻¹⁹. A thin wire's self term (t = h/4a ≥ π/2) is a single chunk. A fat wire's (t = 0.0125 at h/a = 0.05) is six.

This is the "closed-form z-integration plus a short φ integral" of the issue. It evaluates no elliptic integral and no kernel: per θ node it is two square roots, one logarithm, and a recurrence.

## 6. CENTRED: a far pair about its own phase centre

For d ≥ h and d ≥ 6a take

    R_c = ( z_min + hypot(z_max, 2a) ) / 2,

the middle of the range of R over the pair and the ring (z_min, z_max the nearer and farther end). Then

    |R - R_c| ≤ ( hypot(z_max, 2a) - z_min ) / 2 ≤ h/2 + a²/z_max ≤ h/2 + a/6,

whatever the distance. The series (1) runs in k (R - R_c): the last pair of a 59 m wire needs the same 19 terms as the second, and no term of its sum exceeds its first. The one thing that does grow with distance is e^{-jkR_c}, whose phase kR_c is known to kR_c units in the last place. That is the conditioning of the kernel itself, the same in the direct rule.

**The ring average** is the 12-point trapezoid rule, at φ_i = (i + ½)·2π/12, of which six are distinct by symmetry. The integrand is a function of cos φ through R² = s₀ - 2a² cos φ, s₀ = z² + 2a², analytic until 2a² cos φ = s₀. With u = 2a²/s₀ ≤ 1/19 (z ≥ 6a) the trapezoid rule's aliasing error is of order ((1 + √(1 - u²))/u)⁻¹² ≤ 38⁻¹² = 10⁻¹⁹. It replaces the ring-mean expansion to ε⁸ that `_exact_kernel._far` uses, whose own truncation at z = 6a is about 7·10⁻¹⁵; the two therefore differ by that much on pairs just beyond 6a.

**The rule in v** is the 16-point Gauss-Legendre rule, by the argument of §4.

## 7. PLAIN FAR: a fat wire's neighbours

d ≥ h but d < 6a. Here R is not large against the ring's spread, so there is no useful phase centre; R_c = 0 and the rule is Gauss-Legendre in v on ⟨R^{n-1}⟩. Those are what `_exact_kernel._near` already computes inside its k-series: for |z| < 6a, ⟨R^p⟩ = (2/π) ρ^p J_{p/2}(m) with ρ² = z² + 4a², m = 4a²/ρ², the half-integer orders from one AGM and the three-term recurrence, the integer ones from the same recurrence. A node of such a pair that falls beyond 6a takes the ring-mean expansion power by power. `ring_powers` returns them one power of k at a time instead of summed. x ≤ k · hypot(6a + h, 2a) here, which is small because a is.

## 8. What assumes real k

The coefficients are real and geometry-only, and `evaluate` (and the C++ `rows`) take a complex k and do the arithmetic of (1) with it; `test_complex_k_is_served_by_the_arithmetic` holds a lossy k against a brute-force ring average. What assumes real k:

- the **bounds** of §3. They are sized for |k|·|R - R_c| with no growth from e^{Im(k)·R}. For a lossy medium the term counts stand (they depend on |k| only), but the phase factor e^{-jkR_c} of a CENTRED pair then decays or grows in magnitude, which is correct arithmetic and wants its own look at conditioning;
- **`serves` and `accel_serves`**, which follow the direct route in declining a complex k;
- **the callers**: razor refuses a lossy or buried exact-kernel deck before it reaches the kernel layer.

## 9. The cache, and its bound

`_razor_exact_plan.Plan` (momwire#1455) keeps one set of series sources per geometry, beside the per-wavenumber row objects it already kept for the direct rule. A source holds, per unique pair, R_c and 2(N + 1) doubles: at most 560 B for NEAR and PLAIN FAR pairs, 320 B for CENTRED ones, plus its key and its value at the current k. A source keeps at most 4n + 1024 pairs for a group of n segments, as the direct route's does; a pair met after that is integrated again each call. `_group_nbytes` counts `PAIR_NBYTES` = 736 B per kept pair toward the plan's 64 MB cap, so a geometry too large for the cap is computed and not stored, as before.

The key of a plan is unchanged (it is the geometry). The key of its series sources is everything else the coefficients depend on: the quadrature orders, the grading ratio, the 6a switch, `X_MAX`, `X_CENTRED`, the ring rule's order and the tolerance. Neither k nor the ground is in it, because no coefficient reads either.

A source also remembers its pairs' values at the last k it was asked for, so the repeat of one deck evaluates nothing.

## 10. Measured

All on one Skylake box (i7-6700K), branch against `main` at 7025747b built in a second tree, with the harnesses in `scratch/1454-split/`.

### 10.1 Against 30-digit arithmetic, on both sides of each switch

`reference.py`: the series (numpy route) and the direct rule against mpmath at 30 digits. Errors are relative to the pair's static moment, the first moment's in units of h. The c/h = 0.999 and 1 rows straddle the near/far switch; the c = 5.999a/6a and 1.999h/2h rows straddle the 6a switch.

         a   h/a     c/h     k rule         series    direct
      0.01  0.05    -0.5  0.15 near        6.1e-16   6.1e-16
      0.01  0.05  -0.999  0.15 near        1.1e-19   6.6e-16
      0.01   0.2    -0.5  0.15 near        6.4e-16   5.1e-16
      0.01     1    -0.5     1 near        1.0e-15   1.3e-15
      0.01     1       0   0.5 near        1.3e-15   1.3e-15
      0.01     1   1e-06   0.5 near        7.0e-19   1.1e-15
     0.001   100    -0.5     1 near        7.6e-16   4.5e-16
     0.001   100     0.5     1 near        5.0e-17   1.6e-16
      0.01     3     0.5     1 near        9.4e-17   5.4e-16
      0.01     7     0.3     2 near        1.3e-16   5.1e-16
      0.01   0.1   0.999     1 near        1.7e-16   1.7e-15
      0.01   0.1       1     1 plain-far   1.7e-16   8.4e-17
      0.01     1   0.999     1 near        1.7e-16   2.6e-15
      0.01     1       1     1 plain-far   1.7e-16   2.6e-18
     0.001    30   0.999     1 near        2.5e-16   2.5e-16
     0.001    30       1     1 centred     0.0e+00   6.6e-17
      0.01     3   1.999     1 plain-far   8.8e-16   8.8e-16
      0.01     3       2     1 centred     5.7e-17   8.8e-16
      0.01     1   5.999     1 plain-far   2.3e-15   2.3e-15
      0.01     1       6     1 centred     1.4e-16   2.3e-15
     0.001   100      30     1 centred     1.7e-16   4.2e-17
     0.001   100     300     1 centred     6.0e-16   3.3e-16

Both rules sit at the reference's own noise, about 10⁻¹⁵, on every row; neither switch shows. The 2.3·10⁻¹⁵ at c = 5.999a is the ring-mean expansion's truncation at 6a, which the PLAIN FAR rule shares with the direct rule and the CENTRED rule (next row) does not have.

### 10.2 Series against the direct rule, matrix by matrix

`dz.py`: every deck of the corpus through razor-2p with `exact_kernel` forced on, each correction applied both ways to the same matrix.

- Population: 395 decks unique by content (158 momwire fixtures, 206 antennaknobs captures not already among them, 31 of Ward's bench decks not already among them; the three sources hold 158, 242 and 120 files). 11 are refused under a forced exact kernel (buried wires) and solve nothing.
- 348 decks take the correction (349 corrections).
- **max |ΔZ| / max |Z| = 3.0·10⁻¹⁵** (`p10_fat_dipole.nec`).
- The largest entry-by-entry ratio |ΔZ_ij| / |Z_ij| is 6.1·10⁻⁹ (`bowtie2x4_n126.nec`), on an entry that is 10⁻⁹ of its matrix's largest: a remainder of the two potentials cancelling, so its relative change says how many digits that entry carries under either rule, not how far the rules are apart.
- Routes: all 2151 groups took the series, none the direct rule. Unique pairs by rule: 4915 NEAR, 278 607 CENTRED, 9 PLAIN FAR.
- Over the exact-kernel test files (`dz_plugin.py`, 39 corrections, including the slow lane's pins): max |ΔZ| / max |Z| = 3.9·10⁻¹⁵, entry-by-entry 4.9·10⁻¹³.

### 10.3 Printouts

`byte_gate.py` and `compare.py`: both EZNEC dialects, each deck rendered twice in one process, TIME and SEC. lines masked, 1580 printouts per arm.

| binding | arm | decks taking the correction | identical to main | differing |
|---|---|---|---|---|
| `auto` (production) | series off | 4 | 1580 | 0 |
| `auto` (production) | series on | 4 | 1580 | 0 |
| forced on | series off | 348 | 1580 | 0 |
| forced on | series on | 348 | 1420 | 160 |

Of the 160, 154 differ only in a printed numerical zero (a quantity at least 10⁹ below its row's scale changing sign or residue, for example `-7.8505E-17` against `7.8505E-17` beside `1.4142E+00`). The other 6 printouts, in 3 decks, move a last printed digit of a real quantity:

- `bowtie4x4_n42.nec`, 2 lines: a segment current's sixth figure, `3.94651E-04` to `3.94652E-04`;
- `bowtie4x4_n84.nec`, 1 line: `3.19027E-08` to `3.19028E-08`;
- `cat_broadband.discone_gnd.nec`, 21 lines: a pattern null near −188 dB, `-187.92` to `-187.93`.

That is why the series is off by default: under the production binding nothing moves, but a forced exact kernel reaches printouts where a last digit does.

With `MOMWIRE_EXACT_KERNEL_SERIES=1` the default, integration and slow tests that name the exact kernel, razor, EZNEC or SimNEC (3817 tests) pass unchanged.

bs2 does not read this module: its exact-kernel moments are the coaxial pair moments of `_accel_exact_kernel.cpp`, which this step leaves alone. Held anyway: 158 fixture decks through `bspline` with the exact kernel forced on, 632 printouts, all identical to main.

### 10.4 Proof the series ran, and what it costs

Dan's UR0GT 7.3 MHz deck without RP (108 segments, 3 XQ runs, 6 groups, 487 unique pairs: 68 NEAR, 417 CENTRED, 2 PLAIN FAR), five frequencies from an empty cache:

| counter | value |
|---|---|
| series sources: miss / hit | 1 / 4 |
| pair coefficient sets integrated (static-cache misses) | 487 |
| pairs found already built (static-cache hits) | 1948 |
| series evaluations | 2435 |
| direct-rule row objects built | 0 |

`interleave.py`, warm in-process, one thread, five rounds of base and change alternated, median ms per call. "Correction" is the time inside the exact-kernel correction; "render" is the whole solve path (`momwire.eznec._shell.render`). The full output is `scratch/1454-split/timing-skylake.txt`.

| deck | pattern | render, main | render, series | correction, main | correction, series |
|---|---|---|---|---|---|
| 7.3 MHz, no RP | same deck repeated | 14.90 | 14.83 | 1.21 | 1.26 |
| 7.3 MHz, no RP | 5-frequency sweep | 17.34 | 14.93 | 3.60 | 1.34 |
| 7.3 MHz, no RP | 103-point sweep | 17.34 | 14.93 | 3.60 | 1.35 |
| 7.3 MHz, with RP | same deck repeated | 42.17 | 42.07 | 1.24 | 1.29 |
| 7.3 MHz, with RP | 5-frequency sweep | 52.64 | 50.31 | 3.64 | 1.37 |
| 7.3 MHz, with RP | 103-point sweep | 52.75 | 50.31 | 3.64 | 1.37 |
| 1.8 MHz, with RP | same deck repeated | 33.26 | 33.12 | 0.71 | 0.76 |
| 1.8 MHz, with RP | 5-frequency sweep | 39.34 | 37.30 | 2.61 | 0.82 |
| 1.8 MHz, with RP | 103-point sweep | 39.29 | 37.37 | 2.61 | 0.82 |

A sweep's correction now costs what a repeat's does. What is left of it (about 1.3 ms on the 7.3 MHz deck) is the base moments it subtracts, the key lookups and the Python around them, none of which this step touches.

A cold call is not slower either: for a 59-segment group seen from its centroids, building the series and evaluating it once takes 180 µs against 337 µs for the direct rule (one thread, median of 300).
