# momwire#1156 — SG wire loading on buried decks

Measured 2026-09-23 on `fix/1156-sg-buried-loading` (off `8555ada`), soil A
(13, 5 mS/m) at 7 MHz unless stated. Every probe prints to the `.out` of the
same number.

## What was wrong, per call site

| site | route | before | now |
|---|---|---|---|
| `SinusoidalGalerkinSolver._apply_loading` | wholly buried (`_operating_medium` swaps `self.k` to complex k_m) | `loading_for(self, k * self.c)` with complex k → unnamed `TypeError` in `series_impedance_per_wire` | spec read at `medium.k_p * c` (real ω); shapes stay at k_m |
| same | mixed / detached (#980 D2) | never reached: `_MIXED_WIRE_LOADING_REFUSAL`, an undeclared sentence ("applied at a single wavenumber") | applied after the fill, each pair's shapes at its segment's `k_entry` |
| same | crossing (#980 D3) | same refusal (a crossing deck is mixed) | same, the wing columns carry `k_entry` too |
| same, charge side | any buried jacket | `zq_seg` (#1154) never read by SG | `G −= zq_s ∫ f_i′ f_j′` |
| `wire_loss_power` (inherited from `SinusoidalSolver`) | any buried deck | newly reachable once loading solves; rebuilt the shapes at ω/c with the real-k closed form: 2.8e-8 W against bspline's 9.40e-6 W on a buried dipole (340x low), 13 % low on the crossing deck (probe9) | rebuilds the fill's own view (k_m, or stitched + wings) and integrates \|I\|² at complex k |
| `_wire_loading.loading_for` | shared | a complex ω failed inside a float conversion | refused by name (`TypeError`, "REAL angular frequency") |

Swept: SG's `compute_impedance_swept` is a per-k loop over
`compute_impedance`, so the one `_apply_loading` covers it; `_set_k` keeps
`omega = k * c`, and `medium.k_p = float(self.k)`.

Above-ground/free space: `medium is None`, so ω is `k * self.c` exactly as
before and `vals *= z_seg[...]` is the same statement; `wire_loss_power`
takes `super()` whenever no segment is below. probe14: 139 records (Z,
coefficients, per-wire loss, swept, Y) over free / PEC / refl-coef /
Sommerfeld-above / surface-at-b / two-wire decks, bare and loaded four ways,
plus the BARE buried, mixed and crossing decks — **all bit-identical** to
origin/main run from its own source (`probe14_main.log` records the package
that imported; same compiled accelerators, no C++ differs).

## The SG charge Gram

bspline tests the buried jacket's local potential ΔS′·q, q = −(1/jω) dI/dl,
by parts: `zq · ∫ f_m′ f_n′ dl` with zq = ΔS′/(jω), boundary terms dropped.
SG's basis on segment s, per support entry e, is

    f_e(ξ) = P_e + Q_e sin kξ + R_e cos kξ,   ξ ∈ [−h/2, h/2],
    (P, Q, R) = (σA, B, σC)

so f_e′ = k (Q_e cos kξ − R_e sin kξ), and over the symmetric interval
∫ sin·cos = 0 (odd), ∫ cos² = h/2 + sin(kh)/2k, ∫ sin² = h/2 − sin(kh)/2k:

    D_s[e, f] = k² [ Q_eQ_f (h/2 + sin(kh)/2k) + R_eR_f (h/2 − sin(kh)/2k) ]

Bilinear (unconjugated), valid at complex k. Sign: SG loads with
`G −= Z′ ∫ f f` where bspline does `Z += Z′ ∫ f f` (SG's G is −Z for these
local terms), so the charge term is `G −= zq D`, the same relation.

The weak form is the WHOLE term only if f has no jump inside a wire (a jump
would put a delta in f′). probe1: every basis is continuous across every
interior segment boundary to 6e-16 of its scale, at real and complex k, and
exactly 0 at free ends — so the dropped boundary terms vanish there; at
junctions they are dropped as bspline drops them.

Conditioning (probe13, 40-digit mpmath of the same closed forms from the
same float64 coefficients): the new charge Gram loses ≤1.5e-12 relative;
the SHIPPED series overlap loses up to **5.4e-7** on the crossing deck's
graded node segments (P ≈ −R, the #203 cancellation, never folded for the
loading term). Pre-existing and shared with every above-ground deck, so it
is left alone (bit-identity), but it shows up twice:

* swept vs single on the loaded crossing deck: 5.2e-7 Ω when the sweep is
  handed 2π/λ rather than the single solve's own k — exactly the one-ulp
  sensitivity of the loaded deck (probe11: 5.2e-7; bare 5.9e-10; bare at the
  jacket's a′ 1.6e-10; probe12 puts the move in the loading term, 6.5e-11
  relative, the fill moving 7e-12 absolute). The gate hands the sweep the
  single solves' k and holds 1e-9.
* the loaded − unloaded gate is measured against the terms' size BEFORE
  cancellation (1e-12 of Σ|z| h (|P|+|Q|+|R|)², plus 16 ulp of G for the
  subtraction), not against |L|.

## Gates (tests/test_sg_buried_loading_1156.py)

| gate | result | bar |
|---|---|---|
| issue repro, buried dipole: R′, L′, σ, jacket | all serve, all move Z | finite, >0.1 Ω |
| loaded − unloaded vs QUADRATURE of the shapes at real ω (dipole / mixed / crossing; σ and σ+jacket) | within rounding everywhere | 1e-12·M + 16 ulp·\|G\| |
| same gate, red controls: reference at 1.001ω; charge term dropped | both fail | must fail |
| every `loading_for` on a buried solve sees a real ω == k·c | yes, all three routes | — |
| reciprocity: term symmetric; G's asymmetry | term exactly symmetric, asymmetry unchanged | 1e-14 / 1 % |
| #1154 exact in-medium pair, ε̃ = 4, εr = 10 (probe4) | dipole 0.0061 Ω, mixed (buried port) 0.031 Ω; detached 0.031, detached_hub 0.022 | 0.1 Ω |
| red control, charge term dropped | 28.1 / 77.5 Ω (detached_hub 90.3) | >20 / >25 Ω |
| ε̃ = 1 collapse | assembled G `array_equal` with and without the term | exact |
| swept == single (σ + jacket, three decks) | < 1e-9 Ω, and the sweep carries the term | 1e-9 / >1 Ω |
| buried loss readout vs quadrature of the solved current (crossing) | 1e-12 | rtol 1e-12 |
| buried loss readout vs bspline, dipole n=40 | 6e-6 rel (probe9: 3.6e-5 → 6.0e-6 → 1.1e-6 over n = 20/40/80) | 1e-4 |
| complex ω into `loading_for` | named `TypeError` | — |
| slow: SG shift onto bspline, m = 1, 2, 4 | dipole L′ 2.57→1.50→0.87, jacket 0.78→0.43→0.25; mixed L′ 9.34→4.66→2.54, jacket 2.20→1.20→0.69; crossing L′ 0.018→0.009→0.003, jacket 0.0067→0.0051→0.0024 Ω | ratio ≤0.75 (crossing 0.8) |

Razor, same decks (probe3, probe8): SG−razor gaps shrink too (dipole jacket
3.29→1.93→1.18, crossing jacket 0.74→0.33→0.16, mixed jacket 2.20→1.31→0.80).

The crossing and hub4 exact-pair gates cannot run on SG: the exact pair puts
the two media's wires at different radii, and SG's crossing serve refuses
per-wire radii (`crossing_junctions`; SG passes no `two_radius`). The
crossing jacket is gated cross-trunk instead: SG's jacket shift sits 0.007 Ω
from bspline's at m = 1, and bspline passed that exact gate in #1154.

## Found on the way (NOT fixed here — out of scope)

1. **SG's mixed-deck transmitted coupling is sign-inverted** (probe6,
   probe7). At ε̃ = 1 the mixed deck is free space, and SG's Y12 there is
   −5.30e-7+1.62e-5j against its own free-space Y12 of +5.30e-7−1.62e-5j;
   bspline and razor both match free space. On soil A at m = 4, |Y12|
   agrees with bspline to 0.05 %, Y22 to 1e-5 and Y11 to 0.8 %. Negating
   both off-diagonal class blocks is D·G·D with D = ±1 per medium, so EVERY
   single-port Z (every D2 gate, every one-feed deck) is exactly invariant
   and blind to it; what sees it is a multi-port Y / both-ports-driven Z
   (probe5's port 1: 250 Ω off), and the sign of the current on the unfed
   class — so, presumably, the far field of a detached deck (not measured). probe3's `detached` and
   `detached_hub` port-0 "non-convergence" (two ports driven) is this, not
   loading: probe15 feeds `detached_hub` one port at a time and SG meets
   bspline (mast feed: bare to 0.02 Ω, shifts to 0.007 Ω at m = 1; radial
   feed: shift gaps halve per doubling, jacket 11.7 → 4.8 Ω). Crossing decks are not affected (no transmitted grid). Wants its
   own issue (#980 D2 follow-up); "a round trip cannot catch an inversion".
2. **SG `compute_port_solution` / `compute_y_matrix(_swept)` on a
   WHOLLY-buried deck raise** "ground_model='sommerfeld' requires every wire
   at or above ground_z" (probe0): unlike `compute_impedance` they never
   enter `_operating_medium`, so the fill runs in air and the above-family
   remainder refuses. Named, but misleading; independent of loading.
3. The series overlap's closed form is ill-conditioned on graded segments
   (above). A folded spelling (P + R and cos kξ − 1, as #203 did for the
   fill) would fix it but moves every above-ground loaded deck's bits.
