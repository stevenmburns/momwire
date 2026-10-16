# Below/below Sommerfeld tail: extraction vs weighted averages

Prototype study requested by the Rocketlake-builder session. **Scratch only:
nothing in `src/` changes, there is no PR, and nothing is merged.** Run on the
laptop with numpy only, one process at a time, against the installed momwire
at a76771d7, using `_sommerfeld_below`'s numpy path. No NEC source or manual
was opened; the sources are the Sommerfeld identity, elementary integral
tables and the published weighted-averages papers.

## Verdict

**Yes. With weighted averages (WA), a grazing node costs about what a steep
node costs, at ≤ 1e-9 componentwise.** WA is the change worth making;
extraction is not.

- **WA alone:**
  - On the main grid (3 soils × R1 ∈ {0.2, 1, 2} λ_m × θ ∈ {0.05, 0.1, 0.2,
    0.5, 1, 10, 45, 80}°), a grazing node takes 19–26 tail panels.
  - The current engine takes 7,409 at 0.05° and ~3,500 at 0.1°.
  - Integrand points: grazing averages **2,830**, steep averages **2,391**,
    a ratio of **1.18**. Today the ratio is 66.
  - Worst componentwise error vs the tight reference is **2.7e-10**; the
    current engine's is 9.1e-10 at the same nodes.
  - Numpy wall per grazing node: **13.7 ms** mean, against 710 ms today.
- **Extraction alone** (N = 1, 2, 3) only shortens the walk, from 7,409 to
  3,197, 2,662 and 601 panels. At steep, large-R1 lossy nodes it gets
  *worse*: ext3 reaches 1.3e-7. The reason is in §2.4.
- **ext + WA** reaches 11–17 panels, a modest gain over WA's 19–26. To be
  safe it needs a |k_m|h gate. It earns its place only if a production
  kernel can evaluate the subtracted terms for free from the same Bessel
  calls.
- **Under the floor (θ < 0.016667°, down to h = 0):**
  - WA converges in 20–26 panels at every node tested, and ext1+wa in 14–17.
  - They agree with an independent algebraic-tail reference to ≤ 3e-10
    (lossy soils, R1 = 2) and ≤ 4e-11 elsewhere.
  - The current walk runs out its 24,000-panel budget there and returns
    garbage (up to 1e+302, a Wynn blow-up). That is why the floor exists.

## 1. Problem

The below/below remainder is computed one node (ρ, h) at a time. Here
h = |z| + |z′| and θ = atan(h/ρ). Its six integrands carry e^{−γ_m h}, and as
λ → ∞ that tends to e^{−λh} times an amplitude of order λ^{−1/2}. The tail
walks panels between J₀ zeros (lattice π/ρ) until two panels in a row are
quiet at rtol·scale. At grazing angles that means walking to
λ ≈ ln(1/rtol)/h. The measured cost is ~6.4/tan θ panels: 7,000 at 0.05°
and 22,239 at today's floor of 0.016667°. A steep node costs 12–42 panels.

## 2. Method 1: analytic extraction of large-λ terms

The full derivation is in `DERIVATION.md`. In outline:

### 2.1 Series

Put u = 1/λ and γ_s = λ·s_s(u) with s_s = (1 − k_s²u²)^{1/2}. Then:

- D₁ = u·d₁(u), with d₁(0) = A_m = (k_p² − k_m²)/S and S = k_p² + k_m².
- D₂ cancels at orders u⁰ and u¹, so D₂ = (k_m² A_m/S) λ⁻³ + O(λ⁻⁵).
- The exponential factors as e^{−γ_m h} = e^{−λh} E(u), with
  E = exp(−h(s_m − 1)/u) = 1 + (k_m²h/2)u + (k_m⁴h²/8)u² + …

Each integrand is then f_i = g_i(u)·e^{−λh}·B_i(λρ), where
B_i ∈ {J₁/x − J₀, J₀, J₁, J₁/x, J₀, J₀} and each g_i is an ordinary power
series in u. The coefficients are computed per node with truncated
power-series arithmetic (`extract.PS`). The residual f − T falls as λ^{−N} as
it should: on Dan's soil at R1 = 1, θ = 0.5°, it falls 36×, 260× and 8,400×
for N = 1, 2, 3 between λ = 20|k_m| and 320|k_m|.

### 2.2 Closed forms

The building block is the zero-wavenumber Sommerfeld identity,
Φ₀(t) = ∫₀^∞ e^{−λt} J₀(λρ) dλ = 1/R, with R = (ρ² + t²)^{1/2}. A factor
λ^{−m} is m-fold integration in t, so dΦ_m/dt = −Φ_{m−1}. With
L = ln(t + R):

    Φ₁ = −L,   Φ₂ = tL − R,   Φ₃ = −[(t²/2 − ρ²/4)L − (3/4)tR],
    Φ₄ = (t³/6 − ρ²t/4)L + (5/12)ρ²R − (11/36)R³,
    Φ₅ = −[(t⁴/24 − ρ²t²/8 + ρ⁴/64)L − (25t³/288 − 55ρ²t/576)R].

The J₁ and J₁/x kernels follow from J₁(λρ) = −λ⁻¹ ∂_ρ J₀(λρ):

- ∫λ^{−n} e^{−λt} J₁ ↔ −∂_ρ Φ_{n+1}
- ∫λ^{−n} e^{−λt} J₁/x ↔ −(1/ρ) ∂_ρ Φ_{n+2}

Checks:

- Φ₁..Φ₅ satisfy dΦ_m/dt = −Φ_{m−1} to 1e-41 in 40-digit mpmath.
- N ≤ 3 uses Φ up to Φ₅. The J₁/x term at n = 4 would need Φ₆.

### 2.3 Regulariser and contour

Each subtracted term is multiplied by (1 − e^{−λb})^M. That makes it entire
and integrable at λ = 0, so one subtraction serves both the head's detour
and the tail. Its integral over (0, ∞) is the M-th forward difference
Σ_j (−1)^j C(M,j) Φ_m(h + jb). This is exact for M ≥ m, because the
difference annihilates Φ's polynomial ambiguity.

- Every regularised closed form matches 25-digit mpmath quadrature to
  ≤ 1.2e-13. The worst case is the M = 5 difference's cancellation.
- The study uses b = 1/a, with a = 1.1·max(k_p, |k_m|) the head/tail
  boundary.
- **b scan:** b·a ∈ {0.5, 1, 2, 4} changes ext3's panel count by under 1 %
  and its error not at all. ext3+wa moves from 13–19 panels down to 10–13,
  with the same error. b is not a sensitive choice.

### 2.4 Why extraction fails at steep nodes

E(u) is exp(−h(s_m − 1)/u). Its series in u has terms (k_m²h/2)^j/j!·u^j.
Once |k_m|h is O(10) or more (steep nodes at R1 = 2 λ_m, where |k_m|h ≈ 12
at 80°), the truncated series is huge where it is used (λ ≳ a) and cancels
badly. Its tail also no longer represents f there.

Measured:

| node | ext3 error |
|---|---|
| Dan R1 = 2, 45° | 2e-9 |
| Dan R1 = 2, 80° | 6e-9 |
| salt1.8 R1 = 2, 80° | 1.3e-7 |

Those nodes do not need extraction at all: e^{−λh} already kills their tail
in a dozen panels. **Extraction belongs behind a |k_m|h gate, applied at
grazing only.**

## 3. Method 2: Mosig–Michalski weighted averages

The partial sums S_n at the lattice points ξ_n = (n − ¼)π/ρ have remainders
of a known form:

    S − S_n ≈ ω_n Σ_i c_i ξ_n^{−i},    ω_n = (−1)^n ξ_n^{α−½} e^{−ξ_n h},

where α is the power of λ multiplying each Bessel factor (−2 for V, 0 for
the rest; J₁/x adds a further −1; extraction lowers α by N). Sidi's
W-algorithm eliminates the c_i exactly; it is equivalent to the
Mosig–Michalski WA recursion.

How it is run (`methods.tail`, mode "wa"):

- per component;
- on the last ≤ 11 lattice-aligned sums;
- it stops when two successive estimates in a row agree to rtol·scale.

The current quiet test and Wynn fallback stay in place unchanged. So a tail
that goes quiet by itself stops exactly as today: at steep nodes the 1.5/h
decay cap makes panels shorter than the lattice. With mode "quiet" this
walk reproduces `_tail_below` **bit for bit** (checked).

## 4. Results

Settings:

- rtol 1e-11, the engine default.
- **Reference:** the current engine at rtol 1e-12 with a 400,000-panel cap.
  An independent check, `ref2`, is ext3 at 1e-12.
- **Points** counts integrand evaluations: one point is all six integrands
  at one λ.
- **Wall** is numpy on the laptop. It is a ratio between methods only, not
  a C++ forecast.
- Soils:
  - **dan:** ε̃ = 13 − j14.96 at 6.0078 MHz.
  - **salt1.8:** εr 81, σ 5 S/m at 1.8 MHz (very lossy).
  - **fresh28:** εr 80, σ 0.001 at 28 MHz (nearly lossless; AK's
    `TERRAIN_WATER`).

### 4.1 Aggregate (main grid, 72 nodes)

| method | grazing tail panels (max) | grazing points (mean) | steep points (mean) | grazing/steep | worst comp. vs ref | worst vec-norm vs ref | numpy wall, grazing mean |
|---|---|---|---|---|---|---|---|
| base | 7409 | 201305 | 3034 | 66.35 | 9.1e-10 | 2.1e-11 | 710 ms |
| **wa** | **26** | **2830** | **2391** | **1.18** | **2.7e-10** | 3.7e-11 | **13.7 ms** |
| ext1 | 3197 | 82001 | 2811 | 29.17 | 8.0e-10 | 2.1e-11 | 958 ms |
| ext2 | 2662 | 68503 | 2668 | 25.67 | 7.4e-09 | 1.0e-10 | 945 ms |
| ext3 | 601 | 17457 | 2492 | 7.01 | 1.3e-07 | 5.6e-09 | 290 ms |
| ext1+wa | 17 | 2267 | 2296 | 0.99 | 1.4e-10 | 3.9e-11 | 34 ms |
| ext2+wa | 17 | 2371 | 2234 | 1.06 | 7.4e-09 | 1.0e-10 | 41 ms |
| ext3+wa | 17 | 2267 | 2174 | 1.04 | 1.3e-07 | 5.6e-09 | 46 ms |

How to read the table:

- The ext2/ext3 error columns come from the steep R1 = 2 nodes (§2.4). At
  grazing, ext3+wa's worst error is 9e-11.
- The extraction wall in numpy is Python series arithmetic per call. A
  production kernel would reuse the Bessel values it already computes, so
  compare extraction methods on *points*, not ms.
- **WA vs the current engine:** at most 1.2e-9 componentwise at grazing
  (4.2e-10 steep), with the reference closer to WA. That gap is the
  current engine's own error at its rtol, not a defect in WA.
- **Reference self-check:** at grazing nodes, ref vs ref2 agree to
  1.6e-10 componentwise. At steep R1 = 2 nodes ref2 is the bad one, for
  the §2.4 reason.

### 4.2 Dan's soil, per node

Each cell is tail panels / points / wall / worst componentwise vs ref.

| R1/λ_m | θ° | base | wa | ext3+wa |
|---|---|---|---|---|
| 0.2 | 0.05 | 7032 / 507456 / 1794 ms / 4e-11 | 24 / 2880 / 13 ms / 6e-11 | 11 / 1944 / 4e-11 |
| 0.2 | 0.1 | 3579 / 258840 / 875 ms / 3e-11 | 24 / 2880 / 16 ms / 2e-11 | 11 / 1944 / 1e-11 |
| 0.2 | 1 | 381 / 28584 / 102 ms / 4e-11 | 23 / 2808 / 13 ms / 1e-11 | 11 / 1944 / 4e-12 |
| 0.2 | 45 | 22 / 2736 / 11 ms / 1e-11 | 22 / 2736 / 11 ms / 1e-11 | 14 / 2160 / 4e-12 |
| 1 | 0.05 | 6883 / 496728 / 1763 ms / 1e-10 | 22 / 2736 / 13 ms / 2e-11 | 16 / 2304 / 1e-11 |
| 1 | 0.1 | 3504 / 253440 / 911 ms / 1e-10 | 22 / 2736 / 13 ms / 4e-11 | 16 / 2304 / 1e-11 |
| 1 | 10 | 40 / 4032 / 17 ms / 3e-11 | 14 / 2160 / 10 ms / 3e-12 | 12 / 2016 / 4e-12 |
| 1 | 80 | 15 / 2232 / 9 ms / 9e-12 | 15 / 2232 / 9 ms / 9e-12 | 14 / 2160 / 7e-11 |
| 2 | 0.05 | 6943 / 501048 / 1911 ms / 8e-10 | 20 / 2592 / 17 ms / 1e-10 | 17 / 2376 / 6e-11 |
| 2 | 0.1 | 3533 / 255528 / 1021 ms / 6e-10 | 20 / 2592 / 14 ms / 3e-10 | 17 / 2376 / 7e-11 |
| 2 | 45 | 17 / 2376 / 11 ms / 5e-11 | 17 / 2376 / 11 ms / 5e-11 | 17 / 2376 / **2e-9** |
| 2 | 80 | 12 / 2016 / 9 ms / 1e-11 | 12 / 2016 / 16 ms / 1e-11 | 12 / 2016 / **6e-9** |

Every node, soil and method is in `tables.md`, which `summarize.py`
generates.

On Dan's grid, a grazing node under WA costs about 2,600–2,900 points. Most
of that (~2,000) is the head, which WA does not touch. Steep nodes cost
2,000–2,700 points.

### 4.3 Under the floor (θ = 0.01, 0.005, 0.001, 0°)

- Reference: ext3 on the plain quiet tail at rtol 1e-12. Its residual decays
  algebraically (λ^{−3.5}), so it reaches h = 0 in 130–1,200 panels.
- WA (no extraction) is an independent route, and it agrees with this
  reference:

| soil | wa panels | wa worst comp. | ext1+wa panels | ext1+wa worst comp. | base |
|---|---|---|---|---|---|
| dan | 20–24 | 2e-10 | 14–17 | 7e-11 | 24,000 (cap), 2e-7 … 7e+301 |
| salt1.8 | 22–26 | 3e-10 | 14–15 | 2e-10 | 24,000 (cap), 6e-7 … 1e+302 |
| fresh28 | 22–26 | 1e-11 | 14–15 | 2e-11 | 24,000 (cap), 2e-8 … 6e+301 |

The worst WA errors are at R1 = 2 on the lossy soils. That is the same place
the two references disagree by ~1e-10 in §4.1, so it is the floor of what
this study can measure, not a demonstrated WA error.

### 4.4 Robustness sweep (360 nodes)

The sweep covers 3 soils × R1 ∈ {0.02, 0.1, 0.5, 1.5, 3} λ_m × 24 θ spaced
geometrically from 0.03° to 60°. It was looking for false early stops.

- **Cost:** WA's maximum is 27 tail panels, against 12,508 for base. Total
  points: 1.06 M for WA vs 171 M for base, 161× fewer.
- **Against the long-walk reference (base at 1e-12):**
  - Worst WA vec-norm error is 4.6e-9; worst componentwise is 6.3e-8.
  - 17 of the 360 nodes exceed 1e-9 componentwise. **All 17 are at
    R1 = 0.02 λ_m and θ ≤ 0.16°.** Base exceeds 1e-9 at most of the same
    nodes (up to 2.0e-8).
- **The corner was re-refereed with routes that share no tail machinery**
  (`probe_r002b.py`, output in `probe_r002b.log`):
  - The four routes: WA alone, ext3 on the plain quiet tail (no WA),
    ext1+wa and ext3+wa.
  - At 5 corner nodes they agree pairwise to **≤ 1.4e-10**, using 7–27
    panels.
  - The long walks disagree with all four, and with each other: base at
    1e-12 vs ext3 at 1e-12 differs by 2.2e-9 on Dan at θ = 0.03° (a one-off
    probe, not kept).
  - At ρ = 0.02 λ_m, cos θ is ≈ 1, so the long walks sum 5,000–14,000
    alternating panels of a component (∂²V/∂ρ², |·| = 0.34 against ~4
    for the others) whose λ³-weighted panels are much larger than their
    sum. The likely reading is that **the long-walk references are
    roundoff-limited in this corner**, and the short routes are right.
    This explanation is an inference; it was not proven by
    extended-precision arithmetic.
- **No WA false stop was found anywhere else in the sweep.**

## 5. Caveats and what production would need

1. **The reference's own accuracy** is ~1e-10 at R1 = 2 lossy grazing nodes,
   and roundoff-limited at the R1 = 0.02 grazing corner (§4.4). An
   extended-precision reference (mpmath on a handful of nodes) is the honest
   next check. It is too heavy for the laptop.
2. **The WA stopping rule** is heuristic: two agreements in a row at
   rtol·scale, after ≥ 4 lattice sums. It made no false stop in 360 + 72 +
   36 nodes, but a production version should keep the quiet test and the
   Wynn fallback as they are here, and add a cap on WA order.
3. **Componentwise vs vec-norm.** The tolerance is relative to the largest
   component, so components 0 and 5 carry the ~1e-10 errors. That is true of
   the current engine too.
4. **A C++ twin.** `_tail_below` has a C++ kernel; WA adds an 11-point
   divided-difference table per component (66 complex values) and a pow/exp
   per lattice point. Production also needs the bit-identity and eval-order
   gates (cross-CPU tolerance per the BLAS-order and codegen-drift traps).
5. **The floor and grid bands.** If WA holds, the 0.016667° floor and the
   graze-band cell machinery in `_sommerfeld_below` exist to work around a
   cost that WA removes. Revisiting them is a separate decision, and was not
   measured here.
6. **Extraction** is not recommended unless the point savings (~20 % over
   WA) survive a C++ cost model. If it is used, gate it on |k_m|h ≲ 1, use
   N = 1 (ext1+wa already reaches 14–17 panels), and keep b = 1/a.

## Files

| file | role |
|---|---|
| `extract.py` | power series, Φ closed forms, `Extraction` |
| `methods.py` | instrumented base / wa / extN / extN+wa |
| `study.py` | the main grid → `results.jsonl`, `study.log` |
| `extra.py` | b scan, under-floor, 360-node sweep → `extra.jsonl`, `extra.log` |
| `probe_r002b.py` | the R1 = 0.02 corner, cross-referee → `probe_r002b.log` |
| `summarize.py` | → `tables.md` |
| `DERIVATION.md` | the full derivation |

To reproduce, from this directory and with the AK venv's python:
`python study.py && python extra.py && python probe_r002b.py && python summarize.py > tables.md`.
