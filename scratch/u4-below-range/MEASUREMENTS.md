# U4 — the below/below range: the prototype screen (registered before the run)

Plan unit U4 (antennaknobs `docs/plan-buried-scope-closure.md`). momwire tabulates
the below/below remainder to `_SOMM_BELOW_R1_CAP_LAMBDA_M` = 4 in-medium
wavelengths and refuses past it (`_below_interface.BURIED_PAST_CAP_REFUSAL`).
The plan asks whether to serve beyond the cap with the remainder set to zero,
or to extend the table.

Registered 2026-09-13, before any run.

## Base, and what is already known

- **Base.** The branch `scratch/u4-below-range` starts at momwire `origin/main` =
  `b284b17`. That tip contains #1052 (merge commit `553d671`, 05:15Z), #1049
  (`b284b17`, 05:22Z) and #1050 (`b8232e5`). Each was checked with `git
  merge-base --is-ancestor` against the merge-commit OIDs GitHub reports.
- **The cap is 4, not 2.** Commit `006bf28` (#838 part 2, 2026-09-02) raised it
  from 2 λ_m to 4 λ_m with a finer θ lattice in the far annulus (worst error
  3.6e-5 against the 2e-4 gate). The comment above the constant still argues
  "2, not 4". It is stale, and gets fixed in any src PR this unit produces.
- **The remainder is not a small tail below the interface.** momwire's refusal
  text records it growing relative to direct plus image with range (12× and
  168× at working range, momwire#553 U2). The phase-0 RESULTS.md measured
  |E_rem|/|E_direct| from 0.10 at ρ = 1 m to 2.3 at ρ = 10 m, soil A.
  - The direct term carries e^{−Im(k_m)R}: about 9.5 nepers at 4 λ_m in soil A
    at 7 MHz.
  - The lateral-wave part of the remainder decays only algebraically along the
    interface.
  - So the plan's literal criterion, remainder against the pair's direct term,
    will pick extension every time. It is reported, but it is not the screen.
- **Only one corpus deck is really out of range.** The census lists three
  range decks, but two are phantoms: both copies of `2lsloper.nec` are an
  above-ground 80 m sloper whose tab fields (`-68 ft`) the tokenizers split,
  confirmed on antennaknobs and in the corpus translation. The one real range
  deck is the Cebik LPDA with buried radials
  (`lpma3r5-4-6el86ft75o-buriedradials.nec`, sha256 `f653f79da804fc55…`).
  It has 920 horizontal buried wires at 8 depths, 0.152–0.686 m; the median
  segment is 0.61 m; the screen reaches 50.6 m from the origin; and
  R1 = 74.7 m = 4.68 λ_m at 3.5 MHz, soil (13, 0.005), λ_m = 15.97 m.

## The screen and the verdict

The screen is on the #524 phase-0 prototype (antennaknobs
`scratch/524-phase0/proto/buried_proto.py`, regime 2: direct + A_m·image +
remainder, verified against empymod in phase 0). It needs neither a corpus
translation nor a momwire change.

**The screen is not the verdict.** The verdict is what zeroing does to Z, on
the LPDA: zero-beyond-cap against an extended table, both harness-side, through
`antennaknobs ladder`, with NEC-5 x13-static as the outside reference. A field
floor can pass while the summed contribution of many beyond-cap pairs on a
large screen still moves Z. The Z gate is registered separately, before its
runs.

### Geometry and cases

Source dipole at depth d_s (z′ = −d_s) and observer at depth d_o. The
horizontal separation ρ is set from R1 = √(ρ² + (d_s + d_o)²), momwire's
image-distance coordinate. Kinds: HED observed at φ = 0 (colinear pairs, as
along a radial), HED at φ = 90° (side-by-side pairs), and VED.

| case | soil (ε_r, σ) | f (MHz) | (d_s, d_o) m | Δ (m) |
|---|---|---|---|---|
| LPDA-shallow | (13, 0.005) | 3.5 | (0.1524, 0.1524) | 0.6096 |
| LPDA-deep | (13, 0.005) | 3.5 | (0.6858, 0.6858) | 0.6096 |
| LPDA-cross | (13, 0.005) | 3.5 | (0.1524, 0.6858) | 0.6096 |
| SPEC A / B / C × 7, 21 MHz | SPEC.md soils | 7, 21 | (0.02, 0.02) and (0.15, 0.15) | λ_m / 26 |

Δ is the pair's self scale: the LPDA's median buried segment, or the same
electrical density, λ_m/26, for the SPEC cases. R1 ∈ {0.1, 0.25, 0.5, 1, 2, 3,
4, 5, 6, 8} λ_m, plus the self-scale point ρ = Δ.

### Metrics

- **M1, the plan's literal ratio:** |E_rem| / |E_direct| at R1 = 4 λ_m. Norms are
  of the Cartesian field vector.
- **M2, the absolute screen:** |E_rem(R1)| / |E_total(ρ = Δ)| at R1 = 4, 5, 6 and
  8 λ_m, same depths and kind. Zeroing the remainder past the cap errs by
  |E_rem| itself. The screen compares that error with the fill's own tolerance,
  2e-4 (the below grid's gate), applied to a self-scale field.
- **M3, the decay:** the log-log slope of |E_rem| between R1 = 4 and 8 λ_m.

### Guards (a miss stops the run as a harness or integration error)

- **G-a, the ε̃ = 1 collapse at range:** at ε_r = 1, σ = 0, |E_rem| ≤ 1e-9·|E_direct|
  at R1 = 4 and 8 λ_0, for every kind. The phase-0 G2 collapse, carried out to
  the far range this unit reads.
- **G-b, quadrature self-convergence:** the prototype's own `rel` estimate is
  ≤ 1e-6 at every point. The worst is reported.
- **`HalfSpace.assert_decay()` holds** for every soil.

### Predictions

| id | prediction | verdict |
|---|---|---|
| PS1 | **informed** (momwire#553 U2's refusal record; phase-0 RESULTS §2): M1 > 10 in every case and kind |  |
| PS2 | **derived** (the lateral wave decays as ρ⁻² along the interface once the direct term is attenuated away): M3 ∈ [−2.5, −1.5] in every case and kind |  |
| PS3 | **blind, low confidence**: in both LPDA cases, M2 at R1 = 4 λ_m exceeds 2e-4 for HED at φ = 0, so the screen fails at the cap. The estimate is quasi-static: E_direct(Δ) ∝ 1/(ε_m Δ³) and E_lateral(ρ) ∝ 1/(ε₀ ρ²) × k_p²/k_m², giving M2 ~ Δ³/ρ² ≈ 2.4e-4 at Δ = 0.61 m and ρ = 64 m, within a decade of the threshold either way |  |

How the outcomes will be read, fixed now:

- **A guard misses** → the far-range integrals are not trustworthy on this
  prototype. Stop, and report which guard and where.
- **PS1 hits** → the plan's literal criterion cannot choose zeroing on any deck,
  and is retired in favour of M2 and the Z gate.
- **M2 ≤ 2e-4 at 4 λ_m in both LPDA cases** → zeroing is the candidate to take to
  the Z gate.
- **M2 > 2e-4** → extension is the candidate. The Z gate still compares
  zero-beyond-cap with the extended table on the LPDA, because the screen is
  field-level and the verdict is Z.
- **PS2 misses** → the remainder past the cap is not the algebraic lateral-wave
  tail. Report its measured form before choosing a branch.

## Amendment before the main run: G-a was vacuous [2026-09-13]

**G-a passed at exactly 0 in 0.0 s, with `rel` = 0 at every point
(`screen_guard.log`).** That was not a measurement. At ε̃ = 1 the prototype
short-circuits `k_s == k_o` (`buried_proto.py`, `integrals_R`), because
D₁ = D₂ ≡ 0 there and a relative tail test on noise never goes quiet (phase-0
RESULTS §5, trap 7). So G-a never exercised the far-range integrator, and
it is withdrawn as a guard.

**G-c replaces it: small-contrast linearity at range.** The remainder is
first order in (ε̃ − 1). At 7 MHz, depths (0.15, 0.15) m, R1 = 4 and 8 λ₀,
HED at φ = 0 and VED, the bar is |E_rem(ε_r = 1.02)| / |E_rem(ε_r = 1.01)|
∈ [1.9, 2.1], with σ = 0. A miss stops the run.

**G-c is informed.** Before this amendment an exploratory run measured
three of its four points at 1.9811, 1.9842 and 1.9724, with `rel` 3e-8 to
6e-8. Those points are outside every predicted case, so no prediction was
touched.

**Cost.** 75–154 s per point at ρ = 171–343 m and a depth sum of 0.3 m. The
screen's cases are lossy soils with λ_m several times shorter, so ρ is
shorter, but the shallow SPEC depths slow the tail. The run uses parallel
workers.

## The Z gate, part 1: harness and guards (registered before any Z run)

`z_gate.py` runs with the antennaknobs venv and this worktree's `src` first on
the path. The worktree's accelerators are built (`make build` exit 0), and the
import resolves to `momwire-wt-u4/src`. No src change is involved.

### Spellings

| spelling | what changes |
|---|---|
| `shipped` | nothing; a deck past the cap refuses by name |
| `extended` | `_SOMM_BELOW_R1_CAP_LAMBDA_M` 4 → 5. The grid, the serve plan's check, antennaknobs' construction preflight (`momwire.below_reach_refusal`, which reads the constant at call time) and the grid-cache bucket all follow. The far annulus's existing lattice (Δr = `_SOMM_BELOW_DR_LAMBDA_M`·λ_m, and #838's θ cell counts) continues to the deck's R1 |
| `zeroed` | the `extended` grid, plus a wrapper on `_sommerfeld_below.remainder_field_proj_below`. The wrapper calls the original, then zeroes every observer/source pair whose image distance R1 exceeds 4 λ_m. So `extended` and `zeroed` differ by exactly the beyond-cap remainder, through the same fill |

**Plumbing assertions, recorded with every run:**

- the grids the fill built, with their resolved `r1_max`;
- the wrapper's call count, pair count and zeroed-pair count.

A zeroing run that shows no wrapper calls, or no zeroed pairs on the LPDA,
is not a result. That is the silent-failure mode of a flag that changes
nothing.

### Guards (bars fixed now; a miss stops the Z gate)

| id | check | bar |
|---|---|---|
| G-Z1 | **the extended band's own accuracy.** A grid built at the LPDA's soil and frequency (13, 0.005; 3.5 MHz) with the cap at 5 and `r1_max` = 4.7 λ_m, against `iv_surfaces_direct_below` at 28 × 25 off-node points: R1 ∈ (4, 4.7] λ_m, θ ∈ [0.1°, 2.0°]. That covers the LPDA's beyond-cap pairs, whose depth sums of 0.30–1.37 m over R1 ≥ 64 m put θ at about 0.2°–1.2°. The error is worst relative over the four surfaces, as #838's probe measures it | ≤ 2e-4, the in-domain gate |
| G-Z2 | **`extended` does not move an in-range deck.** The catalog `buried_radial_vertical` through antennaknobs' momwire engine on soil A: Z under `extended` equals `shipped` | bit for bit (`repr`) |
| G-Z3 | **`zeroed` does not move an in-range deck,** and the wrapper is plumbed. The same design: Z under `zeroed` equals `shipped`, with at least one wrapper call and zero pairs zeroed | bit for bit, `proj_calls` > 0, `zeroed_pairs` = 0 |

### Run plan

1. Guards G-Z1 to G-Z3.
2. **Part 2 of the registration: the Z predictions.** Written after the prototype
   screen's results and before any ladder or NEC-5 run on the LPDA.
3. The LPDA (`lpma3r5-4-6el86ft75o-buriedradials.nec`, sha256
   `f653f79da804fc55…`) through `antennaknobs ladder`:
   - momwire under `shipped` (records the refusal), `extended` and `zeroed`;
   - NEC-5 x13-static on the raw deck (`GN 2`) as the outside reference.

   Rung r = 1 (2,690 segments) is mandatory. r = 3 runs only if rung 1's
   momwire wall time and memory make it feasible under the 24 GB cap; that
   decision is recorded, not assumed.

## The Z gate: guards measured

| id | verdict |
|---|---|
| G-Z1 | **HIT, non-vacuous on the second run.** First run (`gz1_band.*`), θ 0.1°–2°: worst 1.4e-8. A second run over the steep band (`gz1_band_steep.*`, θ 30°–90°) gave 1.8e-7, but on inspection **every θ sample was a lattice node**: 2.5° steps against the far annulus's steep Δθ = 60°/72 = 0.833°. About half the R1 samples were nodes too, so that check measured almost nothing along θ. The sampling now uses fractional offsets, (k + 0.37) in R1 and (k + 0.61) in θ, and was re-run (`gz1_offnode_*`): grazing band worst **1.16e-8** (at 4.03 λ_m, 1.97°); steep band worst **7.5e-6** (at 4.01 λ_m, 38.7°). Off-node, the steep band is 40× its on-node figure, so the check discriminates. Both bands are far under 2e-4 over R1 ∈ (4, 4.7] λ_m at the LPDA's soil and frequency |
| G-Z2 | **HIT.** Catalog `buried_radial_vertical`, soil A: `extended` Z = `shipped` Z = 78.1320604078555+46.337676999899195j, bit for bit. Its below grid is in range (r1_max 15.49 m = 1.56 λ_m) and unchanged by the cap patch (`gz23_identity.*`) |
| G-Z3 | **HIT.** `zeroed` Z is the same, bit for bit. The wrapper was reached (4 calls, 1,774,224 pairs) and zeroed none |

## The Z gate, part 2: predictions (registered before any Z-producing run)

**Informed by the screen's three LPDA cases**, which completed before this was
written. The SPEC rows were still running and are not used here.

- **M2 at 4 λ_m is 4.6e-6 to 2.8e-5.** The beyond-cap remainder is at most
  2.8e-5 of a self-scale field.
- **M3 is −1.10 (HED colinear), −1.16 (VED) and −2.13 (HED side-by-side).**
  The slow ~1/R decay on colinear and vertical pairs is why a field floor
  alone cannot decide.
- **Rough Z-level estimate.** Beyond-cap pairs outnumber self terms by about
  130 to 1 on a screen of this size, so the summed effect is bounded near
  130 × 1e-5 ≈ 1e-3 of |Z| before weighting by the (weaker) outer-radial
  currents and by the phase cancellation of e^{−jk_pρ} across the screen,
  both of which shrink it.

| id | prediction | verdict |
|---|---|---|
| PZ1 | **the U4 verdict**, informed by the screen's LPDA rows: at r = 1, \|Z(extended) − Z(zeroed)\| ≤ 1e-3·\|Z(extended)\| |  |
| PZ2 | **informational, blind, low confidence**: \|Z(extended) − Z(NEC-5)\| ≤ 0.10·\|Z(NEC-5)\| at r = 1. The LPDA's crossing node is coarse by #674's measure, which is worth several ohms on the fan decks, and NEC-5 is the reference engine, not the truth. A miss is a separate finding, not a U4 verdict |  |

**Plumbing assertions for these runs** (a failure means the result isn't one):

- **`zeroed`:** `proj_calls` > 0 and `zeroed_pairs` > 0.
- **`extended` and `zeroed`:** every recorded below grid has `r1_max` ≥ 4.68 λ_m.
- **`shipped`:** refuses with momwire's past-the-cap sentence, at R1 ≈ 74.7 m.

**r = 3.** Run only if r = 1's `extended` wall time is under 1 h. The rung
triples the segment count (8,070) and the below/below pair count scales with
its square. The decision is recorded either way.

How the outcomes will be read, fixed now:

- **PZ1 hits** → **serve beyond the cap with the remainder zeroed.** The src PR:
  - replaces the refusal with the documented bound (field M2 ≤ 3e-5 at 4 λ_m,
    and the Z-level δ measured here);
  - fixes the stale "2, not 4" comment;
  - adds a test that pins zeroed against extended on a synthetic long-screen
    deck.
- **PZ1 misses** → **extend the table.** The far annulus continues past 4 λ_m,
  with G-Z1's accuracy check extended to the new cap in the src PR's tests.
- **PZ2**, hit or miss, is reported beside the verdict and does not choose the
  branch.

## The Z gate on the LPDA: blocked by a second scope limit

**Neither registered Z prediction was measured.** PZ1 and PZ2 are not
runnable on this deck.

| run | outcome |
|---|---|
| `shipped` | refused at construction (0.36 s) by antennaknobs' preflight with momwire's past-the-cap sentence: R1 = 74.87 m, 4.69 λ_m. The preflight measures vertices, which is conservative against the census's node-based 74.7 m (`lpda_shipped_r1.*`) |
| `extended` | **refused, on the grazing floor.** antennaknobs' vertex preflight reads a below/below pair elevation of θ = 0° (`lpda_extended_r1.*`): the deck's 8 crossing nodes are distinct vertices at z = 0, so a pair of them has a depth sum of 0 at nonzero separation. The wrapper was never reached, and no grid was built |
| `zeroed` | the same refusal (`lpda_zeroed_r1.*`) |
| diagnostic: `extended` with antennaknobs' vertex preflight patched out | **momwire's own serve plan refuses too**, before any fill (1.0 s): θ = 0.0239° on quadrature nodes, under the 0.05° floor, for a depth sum of 0.01029 m over about 24.7 m. Those are shallow nodes on rises at two different crossing nodes (`lpda_grazing_nodes.log`). The vertex preflight is not merely over-refusing; the deck is genuinely below the floor |

So the census's one real range deck is also outside momwire's grazing floor.
That floor is the recorded follow-up for a log-spaced grazing band (momwire#553
U2), a separate scope limit. The census recorded only each deck's FIRST
refusal, so this second one went uncounted.

**Status of U4's evidence:**

- **The screen's LPDA rows stand.** The beyond-cap remainder is at most 2.8e-5
  of a self-scale field, with ~1/R decay on colinear and vertical pairs.
- **No corpus deck can carry the Z gate** while the grazing floor holds.
- **What to do next** is a decision for the plan owner: a synthetic deck past
  the range cap and above the floor, the grazing-floor unit first, or deferring
  U4's src change.

**NEC-5 reference on the LPDA (informational).** x13-static (sha256
`7ebf343d…`), raw deck (`GE -1`, `GN 2` 13/0.005, 3.5 MHz), through `antennaknobs
ladder` at r = 1: **53.0711 − 3.5390j** (2690 segments, fed segment 152.4 mm,
424 s; `lpda_nec5_r1.*`). That agrees with the census's published NEC-5 value
(53.0700 − 3.5386j) to about 1 mΩ. With no momwire answer on this deck, it
stands as the reference any future serve will be read against.

## The screen, measured [`screen.json`, `screen_rerun.log`]

- **The first full run crashed** in SPEC-B-21-0.15 (`screen.log`). There the
  smallest image distance was shorter than the depth sum, so ρ clamped to 0
  and put the observer on the source. `tee` masked the exit code, and no
  JSON was saved.
- **The harness now skips such points** (3 skipped: SPEC-B-21-0.15 at
  R1 = 0.1 λ_m, one per kind) and saves after every case (`5163c80`). The
  metrics read R1 ≥ 4 λ_m only, so no metric or prediction moved.
- **The rerun** completed with 492 rows over 15 cases.

| id | verdict |
|---|---|
| G-c | **HIT**: 1.9811 / 1.9842 / 1.9724 / 1.9768 |
| G-b | **HIT**: the worst prototype quadrature estimate over every point is 6.5e-8 |
| PS1 | **MISSED**. M1 > 10 fails in 8 of 45 case-kind rows, all at 21 MHz on soils A and C, for HED φ = 90° and VED: M1 = 0.46–3.57. At 21 MHz those soils attenuate the direct term far less per in-medium wavelength, so at 4 λ_m it is still comparable with the remainder. The practical reading is unchanged: M1 ≥ 0.46 everywhere, nowhere near a floor, so the literal remainder-against-direct criterion can never choose zeroing, and it is retired in favour of M2 and the Z gate |
| PS2 | **MISSED, and the reading rule asks for the measured form.** Over 4 → 8 λ_m, \|E_rem\| falls as **R^(−0.98 … −1.63)** for colinear HED (φ = 0) and VED, and as **R^(−2.01 … −3.98)** for side-by-side HED (φ = 90°). The colinear and vertical pairs, the ones along a radial, decay about as 1/R, not as the ρ⁻² I derived for the lateral wave. So the beyond-cap remainder is a slow tail on exactly the pairs a radial screen is made of |
| PS3 | **MISSED, in the direction that favours zeroing.** M2 at 4 λ_m on the LPDA, HED φ = 0, is 9.8e-6 (shallow), 1.2e-5 (deep) and 2.8e-5 (cross): 7–20× under the 2e-4 floor. Across all 45 case-kind rows the largest M2 at 4 λ_m is 1.04e-4, still under the floor |

**Reading, as registered:**

- **The screen passes everywhere.** Its field floor would put zeroing up as the
  candidate for the Z gate.
- **The slow 1/R tail is exactly what the Z gate exists to test.** It decides
  whether the summed beyond-cap pairs on a large screen stay small in Z.
- **The Z gate cannot run on any corpus deck:** the LPDA is also below the
  grazing floor.
- **The next step is the plan owner's decision.** The options are a synthetic
  deck, the grazing-floor unit first, or deferral.

## The Z gate on a synthetic deck (Steve's option (a); registered before any run)

The LPDA cannot carry the Z gate, so this deck is built to. `synthetic_u4.nec`:

- **Above:** a 21.4 m quarter-wave mast (`GW 1`, 11 segments, base-fed `EX 0 1 1`).
- **Below:** ONE crossing node at the origin, a 0.15 m rise (`GW 2`) to a buried
  hub, and four radials of 38 m at 0.15 m depth (`GW 3`–`6`, 19 segments each).
- **Ground and flags:** `GE -1`, `GN 2` soil A (13, 0.005), 3.5 MHz, λ_m =
  15.966 m. Every radius is 1 mm, so the U5 two-radius rule is not engaged.
- **Range and grazing, by construction:**
  - tip to tip across the screen, R1 = hypot(76, 0.30) = 76.0 m = **4.76 λ_m**:
    past the 4 λ_m cap (63.86 m), inside an extended cap of 5 (79.8 m);
  - the shallowest below/below pair is a rise node near z = 0 against a radial
    tip, a depth sum ≥ 0.15 m over ≤ 38 m, so θ ≥ 0.23°;
  - tip-to-tip pairs sit at θ = 0.23° too;
  - with one crossing node there are no rise-to-rise pairs at separation, so
    the whole screen is above the 0.05° floor at any refinement.
- **Negative control:** `synthetic_u4_shallow_control.nec` is the same deck at
  0.02 m depth. Its tip-to-tip θ is 0.030° and its rise-to-tip θ ≈ 0.030°, both
  under the floor, so a θ guard must fire on it.

**A quirk noted for the record** (the peer is filing it). antennaknobs'
construction preflight (`engines.momwire._below_reach_refusal`) measures polyline
VERTICES, so the z = 0 vertices of two DIFFERENT crossing nodes form a pair with
a depth sum of 0 at nonzero separation, which it reads as θ = 0. On the LPDA, 8
crossing nodes made it refuse at θ = 0°. momwire's node-level plan read 0.0239°
for the same deck and refused too, so there it was conservative rather than
wrong. On a deck whose nodes all clear the floor it would over-refuse. The
synthetic deck has one crossing node and does not trip it.

### Guards (all asserted BEFORE any Z is read; a miss stops the gate)

| id | check | bar |
|---|---|---|
| G-S0 | **the θ guard can fire.** `extents` mode on the shallow control | θ_min < 0.05° is reported, and the run refuses |
| G-S1 | **vertex extents at every rung** (r = 1, 3, 9): `_pair_extents_below` over the refined deck's buried vertices; momwire's `below_reach_refusal` at cap 4 and at cap 5 | 4 < R1/λ_m < 5 and θ_min ≥ 0.05°; cap 4 → the range refusal; cap 5 → served (None) |
| G-S2 | **node extents inside every `extended` and `zeroed` solve.** A wrapper on `serve_plan`'s `pair_extents` records the quadrature-node (R1, θ) and raises before the fill if either bound fails | the same bounds, with at least one recorded call per rung |
| G-S3 | **zeroing plumbing at every rung** | `proj_calls` > 0 and `zeroed_pairs` > 0 |
| G-S4 | **the extended band's accuracy to this deck's R1.** G-Z1 re-run off-node to `r1_max` = 4.8 λ_m, in the grazing band (0.1°–2°) and the steep band (30°–90°) | ≤ 2e-4 in both |

### Predictions

| id | prediction | verdict |
|---|---|---|
| PZ-S1 | **the U4 verdict**, informed by the screen (beyond-cap M2 ≤ 1.04e-4 everywhere, with a ~1/R tail on radial pairs; the beyond-cap pairs here are only the outer ~6 m of opposite radials): at every rung, \|Z(zeroed) − Z(extended)\| ≤ b, where b is the extended table's own last ladder step \|Z_ext(9) − Z_ext(3)\|, or \|Z_ext(3) − Z_ext(1)\| if r = 9 does not run |  |
| PZ-S2 | **blind**: `shipped` refuses at every rung, and on RANGE (the past-the-cap sentence), not on grazing |  |
| PZ-S3 | **informational, blind**: the extended ladder's Richardson estimate lies within 5 % of NEC-5's Richardson estimate on the same rungs. The crossing node is ungraded, which is worth ohms by #674's measure, and NEC-5 is the reference engine, not the truth; a miss does not choose U4's branch |  |

**Ladder:** `antennaknobs ladder --refine 1 3 9` for `extended`, `zeroed` and
NEC-5 (x13-static). r = 9 runs only if `extended` at r = 3 finishes in under 30
min; the decision is recorded either way.

How the outcomes will be read, fixed now:

- **Any guard misses** → stop. There is no Z reading.
- **PZ-S1 hits** → **serve beyond the cap with the remainder zeroed.** The src PR:
  - replaces the range refusal with the documented bound (field M2 at 4 λ_m
    ≤ 1.04e-4 on the screen, and the Z-level δ measured here against b);
  - fixes the stale cap comment;
  - pins zeroed against extended on this deck in a test.
- **PZ-S1 misses** → **extend the table past 4 λ_m.** The far annulus's lattice
  continues, and G-S4's accuracy check becomes the src test.

### Guards measured (before any Z)

| id | verdict |
|---|---|
| G-S0 | **HIT, the guard fires.** On the shallow control, vertex θ_min is 0.0302° at r = 1, 3 and 9, so `within_bounds` is false on every rung. momwire's preflight at cap 5 refuses on the grazing floor ("θ = 0.03016 deg"). At cap 4 it refuses on range first (`gs0_extents_control.*`) |
| G-S1 | **HIT.** On `synthetic_u4.nec`, vertex R1 is 4.7602 λ_m and θ_min is 0.2262° at r = 1, 3 and 9 (88 / 264 / 792 segments). momwire's preflight at cap 4 gives the range refusal (R1 = 76.0006 m); at cap 5 it serves (`gs1_extents.*`) |
| G-S2 | recorded inside every `extended` and `zeroed` solve, below |
| G-S4 | **HIT.** The extended band against direct evaluation off-node, R1 ∈ (4.01, 4.78] λ_m: grazing worst 9.8e-9, steep worst 7.5e-6 (`gs4_band_*`) |

The G-S2 wrapper matches momwire's call: `BSplineSolver._buried_serve_plan`
passes `k_m` positionally (argument 8) and `pair_extents` as a keyword, both of
which the wrapper reads.

### The synthetic deck, measured [`syn_*`]

**Every rung ran with its guards asserted before any Z was read:**

- **Node extents (G-S2),** recorded inside every solve:
  - R1 = 4.752 λ_m (r = 1), 4.757 (r = 3), 4.759 (r = 9);
  - θ = 0.2266°, 0.2263°, 0.2262°;
  - all inside (4, 5) λ_m and above 0.05°.
- **Extended grids** were built to r1_max = 4.768 λ_m.
- **r = 9 was run under the registered rule.** `extended` at r = 1 and 3 together
  took 14 s, under the 30 min threshold. At r = 9 each spelling took 18 s and
  about 350 MB.

| id | verdict |
|---|---|
| G-S2 | **HIT** at every rung (above) |
| G-S3 | **HIT.** The `zeroed` wrapper was reached and zeroed pairs at every rung: r = 1 and 3 together, 5 calls over 2,134,440 pairs with 26,608 zeroed; r = 9, 33 calls over 17,288,964 pairs with 214,736 zeroed (the `delta` run's per-rung counts: 2,692 / 23,916 / 214,736) |
| PZ-S2 | **HIT, with one caveat.** `shipped` refused on RANGE (R1 = 76.0006 m, 4.76 λ_m, past 63.86 m). The ladder stops at its first rung's refusal, so r = 3 and 9 were not attempted through a solve; G-S1's cap-4 preflight verdict gives the same range refusal there |
| PZ-S1 | **HIT, by more than three orders of magnitude.** The bar b is the `extended` ladder's last step, \|Z_ext(9) − Z_ext(3)\| = \|0.0611 + 0.0421j\| = **0.0742 Ω**. At full precision (`z_gate.py delta`, the same engine call the ladder makes), \|Z(zeroed) − Z(extended)\| = **3.14e-5 / 3.08e-5 / 3.11e-5 Ω** at r = 1 / 3 / 9. That is δ/b ≈ 4e-4, and about 4e-7 of \|Z\|. It holds constant under refinement, so it is the converged beyond-cap remainder's contribution and not mesh noise |
| PZ-S3 | **HIT (informational).** `extended` momwire ran 69.6229+40.6985j, 69.7544+40.7702j and 69.8155+40.8123j at r = 1 / 3 / 9, giving a Richardson estimate of 69.8460+40.8334j. NEC-5 x13-static ran 69.3290+36.5550j, 69.6560+39.2460j and 69.7690+40.2250j, giving 69.8255+40.7145j. The two estimates differ by 0.0205 + 0.1189j, so \|Δ\| = 0.121 Ω, 0.15 % of \|Z\|, inside the 5 % bar (corrected, see below). (NEC-5's own r = 3→9 step, 0.99 Ω, is 13× momwire's.) |

**Reading, as registered: PZ-S1 hits, so the branch is to serve beyond the cap
with the remainder zeroed.**

- **Why that is safe.** Across a screen 4.76 λ_m wide, the summed 1/R tail of
  every beyond-cap pair moves Z by 3.1e-5 Ω, while the table's own ladder moves
  it by 0.074 Ω.
- **What the eventual src PR contains:**
  - it replaces `BURIED_PAST_CAP_REFUSAL` with a served zeroing of the
    below/below remainder past the tabulated range;
  - the bound is documented where the refusal stood: a field-level M2 at 4 λ_m
    ≤ 1.04e-4 over the SPEC and LPDA cases, and a Z-level δ = 3.1e-5 Ω
    against a ladder step of 0.074 Ω on this deck;
  - it fixes the stale "2, not 4" cap comment;
  - a test pins zeroed against extended on this deck, as a bound rather than a
    literal.
- **What does not change.** The grazing floor still refuses every pair below
  0.05°. The LPDA stays refused until that separate unit lands.
- **Still to decide.** Whether the src PR is worth doing now, given that no corpus
  deck is unblocked by it alone, is the plan owner's call.

**Correction to PZ-S3 [2026-09-13; caught by the peer].** The momwire
Richardson estimate first recorded, 69.8766+40.8544j with \|Δ\| = 0.149 Ω
(0.18 %), came from **2·Z9 − Z3**. That is the first-order estimator for a ×2
ladder, reused from step (b′)'s r = 1, 2, 4 ladder and applied by hand to this
×3 ladder (r = 3 → 9). For a refinement ratio q, the first-order estimate is
Z_hi + (Z_hi − Z_lo)/(q − 1), antennaknobs' `ladder_estimate`, which for q = 3
is **Z9 + (Z9 − Z3)/2** = 69.8460+40.8334j. The NEC-5 figure,
69.8255+40.7145j, was already that form, printed by the ladder, so the first
comparison mixed two estimators. With both consistent, \|Δ\| = 0.121 Ω
(0.149 %), recomputed from the full-precision Z in `syn_delta.json`. **PZ-S3
stays a HIT, closer than first recorded.** PZ-S1 and PZ-S2 use only the last
ladder step and are unaffected. Estimates in this record come from
`ladder_estimate`'s form from now on, never a hand-written ×2 formula.

## The src PR: serve past the cap with the remainder zeroed (registered before any src edit) [2026-09-13]

Steve said go. This section registers the design, the tests, and the predictions
before any src edit and before any run that produces a number for them. The
src branch is `u4-below-range-zeroing`, from momwire origin/main b284b17. It
stays unmerged for Steve.

### Design

- **D1. One copy, in the projection.** `remainder_field_proj_below` returns
  exactly 0 for every pair with R1 > `grid.r1_cap`, which is
  `_SOMM_BELOW_R1_CAP_LAMBDA_M`·λ_m at grid construction. Every other pair is
  today's number from today's arithmetic.
  - The numpy path evaluates the surfaces at min(R1, r1_cap) and zeroes the
    masked entries.
  - The C++ path keeps its kernel untouched. It hands `grid.eval` the extremes
    with R1 clamped to r1_cap. Only when the reported max R1 is past r1_cap
    does it build the per-pair mask and zero those entries.
  - All three consumers get the same rule:
    - bspline, via `_below_interface.compute_Z_operator_buried`;
    - SG, via `sinusoidal_galerkin` (live: `buried=True`);
    - razor, via `_potential_ground.BelowMediumGround` (dormant behind
      `_SERVE_BURIED = False`).
- **D2. Keyed on the cap, not on `r1_max`.** A grid tabulated SHORT of the cap
  and then queried past its own `r1_max` is a sizing bug, not a domain edge.
  `SommerfeldGridBelow.eval` keeps refusing it, because the clamp is to r1_cap
  and not to r1_max.
- **D3. Still refused, unchanged.**
  - **`remainder_field_below`**, the field-point vector. No product code calls
    it. A lone observer past the cap has no self-scale term for the remainder
    to be small against: this screen's M1 is 0.46–3.57 against direct at
    4 λ_m, so there it is not negligible.
  - **The grazing floor, over ALL pairs, zeroed ones included.** This holds in
    `serve_plan`, in `below_reach_refusal`, in razor's plan, and in the
    projection's extremes. The LPDA stays refused.
  - The tail-cap, z′-ladder depth, cross-range, and cross-grazing refusals.
- **D4. Refusal sites removed, and the bound documented where they stood.**
  - The range branch of `_below_interface.serve_plan`.
  - The range branch of `bspline.below_reach_refusal`.
  - The range branch of `razor._assemble_Z_below_plane`.
  - `BURIED_PAST_CAP_REFUSAL`, and its bspline re-export, are deleted.
- **D5. Stale prose.**
  - The "2, not 4" cap comment is rewritten. It is history plus what 4 λ_m
    means now: where the tabulation is gated accurate, and where the bound
    below takes over.
  - `get_grid_below`'s "at the 2 λ_m cap" becomes 4.
  - The `eval` docstring is reconciled. Relative to direct+image the remainder
    still GROWS with range (#553 U2's 12× and 168× stand). Relative to the
    self-scale terms that set each row of Z it is ≤ 1.04e-4 at 4 λ_m. That is
    why the fill may zero it while a point query may not.
- **D6. The bound, as it will be written at the removed refusal:**
  - field-level M2 at 4 λ_m ≤ 1.04e-4 over the SPEC and LPDA cases;
  - Z-level δ = 3.1e-5 Ω on this record's synthetic deck, against its own
    ladder step of 0.074 Ω (4e-7 of \|Z\|, constant under refinement);
  - plus the extension rows below, once run.
  - A pointer to this record.

### Tests

- **T1 (new, `slow`): the bspline bound.** The synthetic deck is built
  natively from `syn_kwargs_r1.json` / `syn_kwargs_r3.json`. Those are the
  exact `BSplineSolver` kwargs antennaknobs' engine builds, captured at
  construction with no fill (`capture_kwargs.py`). Three solves:
  - `shipped` at r = 1;
  - `extended` at r = 1 and r = 3, with the cap monkeypatched to 5.
  - Assert \|Z_ship(1) − Z_ext(1)\| ≤ 1e-2 · \|Z_ext(3) − Z_ext(1)\|.
  - Guards, all read before δ:
    - **G-T1:** the shipped plan's R1 is past 4 λ_m and its θ is above the
      floor (a spy on `serve_plan`).
    - **G-T2:** the extended grid is tabulated past 4 λ_m.
    - **G-T3:** δ > 1e-9·\|Z\|. A bit-identical pair would mean the zeroing
      never ran.
- **T2 (new, `slow`): the SG bound.** The same deck, bar, and guards, on
  `SinusoidalGalerkinSolver`.
- **T3 (changed): the projection past the cap.**
  - The `past_cap` row of `test_g5689_the_refusals_survive_the_dispatch`
    becomes: a past-cap pair returns exactly 0 on both paths.
  - New: a call mixing an in-range pair with a past-cap pair returns the
    in-range entry bit-identical to that pair queried alone, on each path.
  - `test_gu2_4_past_the_cap_refuses_instead_of_clamping`, the grid's `eval`,
    is unchanged.
- **T4 (changed):** `test_gu5_6_a_buried_structure_past_the_below_cap_refuses`
  becomes "is served": the 1.5×cap radial solves to a finite Z. The 40 m row
  stays.
- **T5 (changed):**
  - `test_g1135_2`: the far pair is no longer a refusal.
  - `test_g1135_4` re-points its sweep across the one bound left, the grazing
    floor, by depth. The "both halves present" scoring rule is unchanged.
- **T6 (changed):** in `test_on_the_fan_it_is_the_r1_cap_that_soil_moves`, the
  wet-soil fan past the cap no longer refuses on range. Asserted on the serve
  plan without solving, keeping the lane fast.
- **T7 (prose):**
  - `test_ble_1937_838`'s crossgate: the ε_r 30 / 135 ft screen now serves,
    and the comments saying "refused" are updated.
  - `test_g980c_3`'s re-export list drops `PAST_CAP`.
- **Downstream, not in this PR.** antennaknobs `tests/test_below_reach_preflight_1135.py`
  asserts the range refusal in three tests:
  - `test_the_issues_own_case_refuses_by_name`;
  - `test_the_threshold_sits_where_the_issue_measured_it`;
  - `test_the_preflight_agrees_with_the_solve`.
  They run against antennaknobs' submodule pointer, so they change only when
  the pointer moves past this PR.

### Guards on the src change (local, before the PR opens)

- **G-B: in-range decks are bit-identical.** `fan_rise_deck()` at soil A
  (inside the cap) gives the same Z bit-for-bit on origin/main and on the
  branch, on both the C++ and the numpy projection. An empty mask must mean
  untouched arithmetic.
- **G-N: the native deck is the measured deck.** Native `extended` Z at r = 1
  reproduces antennaknobs' `extended` 69.62293392355063+40.69854829336737j to
  ≤ 1e-6 Ω.
- **G-X: the branch reproduces the harness.** Branch `shipped` Z at r = 1
  reproduces `z_gate.py`'s `zeroed` spelling at r = 1 to ≤ 1e-9 Ω. That
  spelling zeroed the same pairs by R1 > 4 λ_m on an extended grid, so this
  also tests D1's claim that a grid tabulated further interpolates identically
  inside 4 λ_m.

### The screen extension (registered; run before the PR opens)

antennaknobs serves soil presets this screen never reached, at the HF band
ends. The screen's worst row was its low-loss, shallow, low-frequency corner
(SPEC-C-7-0.02 VED, M2 1.04e-4), so the extension pushes along those axes:

- very poor (5, 0.001) at 1.8 MHz, depths 0.02 and 0.15;
- fresh water (80, 0.001) at 1.8 MHz, depths 0.02 and 0.15, and at 28 MHz,
  depth 0.02;
- salt water (81, 5) at 1.8 MHz, depths 0.02 and 0.15, and at 28 MHz,
  depth 0.02.

The command is `screen_remainder.py --extension`. The G-c guard runs first, as
before. Rows the prototype cannot finish inside the run's wall-clock budget
are recorded as not run.

### Predictions

| id | prediction | result |
|---|---|---|
| PX1 | **informed** (the screen's M2 ≤ 1.04e-4 and the decay rates it measured): M2 at 4 λ_m ≤ 2e-4 on every extension row that runs | |
| PT1 | **informed** (antennaknobs' engine measured δ/b₁₃ = 3.14e-5 / 0.1497 = 2.1e-4): T1 holds at 1e-2·b₁₃, and G-N and G-X hit | |
| PT2 | **blind**: SG serves the synthetic deck, and T2 holds at the same bar | |
| PT3 | **blind**: `make test`, `make slow` and `make crossgate` pass locally, with only the tests named in T3–T7 edited | |
| PB | **blind**: G-B is bit-identical on both paths | |

### Reading rules, fixed now

- **A PX1 miss on any row** stops the PR from opening. That soil and frequency
  first gets the synthetic Z gate, with the deck rescaled to the same
  4.76 λ_m. The zeroing is either kept uniform on that measurement, or the
  miss is scoped and recorded.
- **A PT2 miss** (SG does not serve the deck) is recorded. T2 then moves to a
  deck SG does serve past the cap, which must be named in this record before
  it runs. **A PT2 bound miss** stops the PR, exactly as a PT1 miss would.
- **A G-B miss** is a defect in D1's implementation, not a finding; fix it
  before anything else is read.

### What still blocks the LPDA, after this PR

1. **The vertex preflight quirk in antennaknobs.** It pairs z = 0 vertices on
   different crossing nodes as θ = 0, and the peer is fixing it.
2. **The node-level grazing floor.** The pairs sit at 0.0239°, against a 0.05°
   floor.
3. **The one crossing node.** `_crossing_fill._ends_and_corner` asserts a
   single crossing node, and the LPDA has eight. The peer measured it on
   momwire b8232e5: two base-fed verticals, each with its own node, pass the
   serve plan and die with an AssertionError at `_crossing_fill.py:1218`.

This PR removes none of the three.

### The src PR guards, measured [`src_guards.py`, `src_guards.log`; 2026-09-13]

Each guard ran in its own process, with one momwire tree first on PYTHONPATH
and antennaknobs' venv interpreter throughout.

| guard | result |
|---|---|
| G-B, C++ projection | **HIT.** `fan_rise_deck()` (soil A, inside the cap) reads 148.62141279270838−29.791702667735215j on main and on the branch, bit-identical. "Main" here is the branch's src tree with origin/main's four edited `.py` files restored, so both runs share one set of compiled accelerators and differ only in the Python this PR edits |
| G-B, numpy projection | **HIT.** 148.6214127927059−29.79170266773472j on both, bit-identical (89 s and 85 s) |
| G-N | **HIT.** Native `extended` at r = 1 reads 69.62293392355062+40.698548293367374j, against antennaknobs' 69.62293392355063+40.69854829336737j: \|Δ\| = 1.6e-14 Ω, bar 1e-6. The captured kwargs are the measured deck, and the feed amplitude (1 here, 0j at antennaknobs' construction) does not move Z beyond roundoff |
| G-X | **HIT.** Branch `shipped` at r = 1 reads 69.62292450000115+40.69857828142006j, against `z_gate.py`'s `zeroed` 69.62292450000115+40.698578281420055j: \|Δ\| = 7.1e-15 Ω, bar 1e-9. A grid tabulated only to 4 λ_m and zeroing past it reproduces the zeroed-on-an-extended-grid answer, so D1's claim that a grid tabulated further interpolates identically inside 4 λ_m holds |

**PB is a HIT.**

**A harness defect in T1/T2, caught by T1's own guard before δ was read.**
- **The symptom.** The first run failed the extended-grid guard on both
  trunks. Under a cap patched to 5, the `extended` spelling's grid read
  4.0 λ_m, and its Z was bit-identical to `shipped`.
- **The cause.** `get_grid_below` keys its cache on the 1.25ⁿ R1 bucket, not
  on the cap. 4.0 and 4.75 λ_m both round up to 4.768, so the cache handed
  the extended solve the shipped grid.
- **The fix.** The test now evicts the shipped grids before the extended
  spelling, and drops the extended grids at teardown. The fixture docstring
  said no shipped plan could ask for that bucket; that was wrong, and it is
  corrected.
- **The record's numbers are unaffected.** `z_gate.py delta` ran both
  spellings on the extended cap, since its `zeroed` spelling zeroes on an
  extended grid, and every ladder rung ran in its own process. G-X above
  also reproduces the zeroed number from a fresh process at the shipped cap.

T4 (`test_gu5_6_a_buried_structure_past_the_below_cap_is_served`) and the
re-pointed `test_g1135_4` both PASS, at 19 s and 30 s of call time.

**Amendment: the screen extension's wall-clock budget, fixed before any
extension row reported.** The registration said unfinished rows are recorded
as not run, but named no budget. Only the G-c guard has printed so far, and it
hit: rem(1.02)/rem(1.01) = 1.972–1.984. The budget is **4 h from the run's
start at 12:47:38Z**. Rows unfinished at 16:47Z are recorded as not run.

### T1/T2 and the fast lane, measured [2026-09-13]

| id | result |
|---|---|
| PT1 | **HIT.** bspline: \|Z_ship − Z_ext\| at r = 1 is 3.143e-5 Ω against the extended table's r = 1 → 3 step of 0.1497 Ω, which is 2.10e-4 of the step (bar 1e-2). Extended at r = 1 and r = 3 reproduce antennaknobs' extended prints to ≤ 2e-14 Ω |
| PT2 | **HIT.** SG serves the deck. Shipped reads 70.03600738983259+40.93569496905351j and extended 70.03601631488755+40.93566597359472j, so δ = 3.034e-5 Ω against SG's own step of 0.3550 Ω, 8.5e-5 of it (bar 1e-2). The two trunks' δ agree to 3.5 % while their ladder steps differ by 2.4×. That is consistent with the zeroed contribution being a property of the geometry rather than of the basis |

Serially T1 takes 28.7 s and T2 23.3 s. Both exceed the 20 s ceiling, so the
`slow` mark is required, not a choice.

**PT3 is in progress, and one clause has already missed.** The prediction said
only the tests named in T3–T7 would be edited.
- **The missed test.** `make test` failed a test the registration did not name:
  `test_buried_assembly_910.py::test_g910_2b_the_refusals_still_fire`.
  - It pinned the range refusal with `match="R1"`.
  - The registration's greps searched for the refusal sentence and the constant,
    so none of them matched it.
  - Its 200 m hub deck now passes the range check and refuses on the grazing
    floor instead (θ = 0.043°). The range half is retired in src commit e9153d0.
- **The ceiling breaches are load, not the change.** The same run tripped the
  20 s hard ceiling on five unmarked tests outside this change: the point-field
  envelope and four extended-kernel rows.
  - Run serially, they take 5.3 / 3.8 / 3.3 / 0.4 / 0.2 s.
  - The breaching run overlapped two of this PR's own normal-priority jobs
    (numpy G-B and T4), at a load average near 11.
- **Next.** The lane is being rerun alone. `make slow` and `make crossgate`
  follow.

**PT3, the fast lane, second run (alone).**
- **Result.** 5113 passed, 0 failed, 17 skipped, 4 xfailed. The lane still
  exited 2, because five unmarked tests went over the 20 s hard ceiling.
  - Four were the same above-ground rows as the first run.
  - The fifth was `test_contact_lane_decays_on_the_high_eps_grounds[monopole-vgood]`.
- **Why.** No job of this PR was running, but the screen extension was: six
  prototype workers at nice 15, each still taking about 65 % CPU on this
  4-core / 8-thread box. The load average was 10–14. Niceness does not protect
  eight xdist workers on four physical cores.
- **So the lane is run a third time with the screen paused.** SIGSTOP goes to
  every process in its systemd scope, and SIGCONT is guaranteed by a trap.
- **Amendment to the screen's budget, fixed before any extension row
  reported.** Time the screen spends paused for this PR's local lanes extends
  its 16:47Z deadline by exactly that interval.
  - The pause and resume times are logged in `screen_extension.pauses`.
  - The prototype's verdict metrics are fields, not timings. Only the
    per-point `seconds` of points in flight during a pause are inflated.

**Incident: this PR's own pause killed the screen extension's first run
(13:23:36Z).**
- **What happened.** SIGSTOP to the run's systemd scope made the wrapping task
  shell report the job Stopped (exit 147). The background task then ended, and
  its process tree was torn down. The prototype workers were terminated rather
  than paused: the scope no longer exists, and `resource_tracker` reports
  leaked semaphores.
- **What was lost.** The run had printed only its G-c guard, which hit
  (1.972–1.984), and no extension row. The first row, EXT-very-poor-1.8-0.02,
  had run about 33 min without finishing. Its log is kept as
  `screen_extension_killed.log`, with `screen_extension_killed.pauses`.
- **Budget, re-fixed before any row reports.** No extension result exists, so
  this is still possible. The restart runs as a transient systemd service,
  not as a child of a task shell, and nothing pauses it. Its wall-clock budget
  is **4 h from the restart's own start**; rows unfinished by then are
  recorded as not run.
- **The pause-extension rule above is withdrawn.** Nothing pauses the restart,
  so there is no pause to extend for.
- **The fast lane's third run** proceeds without the screen, on an unloaded
  box.
