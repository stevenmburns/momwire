# U5 (b) — the two-radius crossing rod against NEC-5

Haswell (i7-4770K). Scratch only; no `src/` change. The candidate rule of
`DERIVATION-MIXED-RADIUS.md` (observer-side, pinned at the row level by
check 2) is applied harness-side. Every prediction below is committed before
the run that tests it, and every miss is reported.

## Setup (registered before the run)

- momwire `src/` at `42112b9` (the AK venv's editable checkout, `1dbd384`, is
  src-identical); antennaknobs `1b3a6a920`; NEC-5 reference
  `~/nec5-timing/nec5cl-x13-static`, sha256 `7ebf343d…`.
- **Deck:** #931's crossing rod (`scratch/931-study/probe_rod_ladder.py`
  `RodBuilder`) at soil A, 7.1 MHz: a graded rise from the hub at z = −L up to
  the node, a 50 mm fed wire from the node up with its source at the centre,
  and a graded radiator to λ/4. Radii per wire through `WireSpec`: the rise
  a_B; the fed wire and radiator a_A. The above member at the node is the fed
  wire.
- **Ladder:** uniform refinement r = 1, 2, 4. Every segment is divided by r
  with the graded panel boundaries held: `per_panel` = 2r on both graded wires,
  the fed wire at 2r segments, the rise's far panel at 25 mm / r and the
  radiator's at its design segment / r. The source region refines with
  everything else. First-order Richardson from r = 2 → 4 on both engines.
- **Configurations (a_A, a_B)**: (0.25, 0.25) mm control; (0.25, 0.125) ratio 2;
  (0.25, 0.0625) ratio 4; (0.125, 0.25) ratio 0.5. The largest radius, 0.25 mm,
  keeps Δ/a ≥ 6.25 at r = 4.
- **Lengths:** L = 0.30, 0.60, 1.20 m (the #956 rod lengths; 0.15 m is left out
  because NEC-5's ladder there was not first order).
- **Ports matched:** momwire's feed parity patched to even, so both engines
  have the same mesh and the same knot source, asserted at every rung. U7 found
  the port split immaterial on this class; matching removes it as a variable.
- **momwire spellings:** observer-side at every rung (check 2's harness
  patches); the naive one-radius spelling `single_B` (the whole fill at the
  rise's radius, which is `_radius_per_wire[0]`) at r = 2 only, as a reading.
- **Resolved settings asserted at every momwire rung:** `extended_kernel` False
  on the solver; momwire's segment count equal to NEC-5's; the feed on a knot;
  the crossing fill entered exactly once. The equal-radius control through the
  patched path must reproduce the unpatched solver bit for bit at each L, or
  the run stops.
- **Read at every rung, beside Z:** momwire's node KCL deficit and its slope
  ratio against 1/ε̃.

## Predictions (registered before the run)

| id | prediction | verdict |
|---|---|---|
| Pb.1 | observer-side momwire keeps the node KCL deficit below 1e-5 at every rung of every configuration and length | pending |
| Pb.2 | **the U5 gate**: at each L, the mixed-radius Richardson residual ΔZ∞ = NEC-5 − momwire lies within the equal-radius control's residual band, \|ΔZ∞(mixed) − ΔZ∞(control)\| ≤ b(L), where b(L) is the control's own last step \|ΔZ(r = 4) − ΔZ(r = 2)\| at that L, floored at 0.5 Ω | pending |
| Pb.3 | **informed by check 2**: the observer-side slope ratio is within 1e-3 of 1/ε̃ at every rung | pending |
| Pb.4 | **blind**: the naive `single_B` residual at r = 2 lies outside that band at ratios 2 and 4, at every L | pending |

Competing outcome, registered with them: if NEC-5's own ladder is not first
order at a radius step (its successive steps not shrinking by about 2×), then
Pb.2 cannot be read against NEC-5 in that configuration. That is reported, and
momwire's self-consistency (Pb.1, Pb.3) stands alone there. momwire's docs
record NEC-2 failing to converge at an in-line radius step; NEC-5's behaviour
at the node's radius step is measured here, not assumed.

## Step (b) measured

`b_rod_ladder.py`; `b_control.json`, `b_ratio2.json`, `b_ratio4.json`,
`b_ratio05.json` with their logs. The control's patched path reproduced the
unpatched solver bit for bit at each L. Ports matched, segment counts equal,
`extended_kernel` False and the crossing fill entered once were asserted at
every momwire rung.

### The physics the gate had to see: R's response to the radius change, r = 4

| L (m) | NEC-5 R: control → ratio2 / ratio4 / ratio05 | momwire observer R: control → ratio2 / ratio4 / ratio05 |
|---|---|---|
| 0.30 | 424.97 → +35.40 / +70.68 / +0.28 | 424.77 → +1.07 / +1.97 / +34.72 |
| 0.60 | 255.59 → +17.84 / +35.65 / -0.02 | 255.57 → +0.25 / +0.49 / +17.62 |
| 1.20 | 163.43 → +9.02 / +18.03 / -0.14 | 163.46 → -0.06 / -0.09 / +8.97 |

Halving a buried rod's radius should raise its grounding resistance by about
ln(ratio) / L. **NEC-5 does that:** +35 / +71 Ω at L = 0.30 m for ratios 2 / 4,
scaling as 1/L, and it barely notices a thinner *above* wire (ratio 0.5,
+0.3 Ω). **Observer-side momwire does not:** it moves +1 / +2 Ω for a thinner
rise, and +35 Ω for a thinner *above* wire — as though the above member's
radius were the rise's.

### Verdicts

| id | verdict |
|---|---|
| Pb.1 | **MISSED** — the KCL deficit reaches 7.2e-05 at r = 1, the control included (the source sits 25 mm above the node on this deck); at r = 4 every rung is ≤ 3.2e-06 |
| Pb.2 | **MISSED** — outside the band at every mixed configuration and every length (table below) |
| Pb.3 | **MISSED, and mis-specified**: the slope ratio is up to 1.00 off 1/ε̃ at r = 1 and 0.057 at r = 4, on the equal-radius control too. With the source 25 mm above the node the slope just above the node is the source's, not the AGARD condition; check 2's rod had its feed 4.33 m away |
| Pb.4 | **literal: HIT; like-for-like: MISSED** — see below |

| config | L (m) | NEC-5 Z (r = 4) | momwire Z (r = 4) | ΔZ∞ (mixed) | ΔZ∞ (control) | \|diff\| | band b(L) | Pb.2 |
|---|---|---|---|---|---|---|---|---|
| ratio2 | 0.30 | 460.37-397.74j | 425.842-362.62j | 33.75-34.52j | 0.08822-0.2245j | 48.049 | 0.642 | **outside** |
| ratio2 | 0.60 | 273.43-191.82j | 255.825-175.186j | 17.26-16.29j | 0.04389-0.07162j | 23.652 | 0.500 | **outside** |
| ratio2 | 1.20 | 172.45-65.74j | 163.399-59.3828j | 8.917-6.058j | 0.031-0.0002168j | 10.754 | 0.500 | **outside** |
| ratio4 | 0.30 | 495.65-434.07j | 426.738-364.32j | 67.71-69.19j | 0.08822-0.2245j | 96.584 | 0.642 | **outside** |
| ratio4 | 0.60 | 291.24-209.06j | 256.06-176.082j | 34.6-32.68j | 0.04389-0.07162j | 47.513 | 0.500 | **outside** |
| ratio4 | 1.20 | 181.46-72.455j | 163.369-59.9957j | 17.84-12.21j | 0.031-0.0002168j | 21.590 | 0.500 | **outside** |
| ratio05 | 0.30 | 425.25-361.21j | 459.492-394.481j | -33.19+33.9j | 0.08822-0.2245j | 47.666 | 0.642 | **outside** |
| ratio05 | 0.60 | 255.57-174.45j | 273.19-190.069j | -16.97+16.08j | 0.04389-0.07162j | 23.461 | 0.500 | **outside** |
| ratio05 | 1.20 | 163.29-58.936j | 172.424-64.4892j | -8.76+6.053j | 0.031-0.0002168j | 10.673 | 0.500 | **outside** |

**Pb.4 was worded badly, so it is recorded both ways.** "The naive r = 2 residual
outside that band" compares a one-mesh residual with the control's Richardson
ΔZ∞. Read literally that is the verdict above. Compared like for like, against the
control's own r = 2 residual, the naive spelling sits within 0.05–0.4 Ω of it
at every length and ratio. **The naive single-radius spelling tracks NEC-5 and
the equal-radius residual; the candidate does not.**

| config | L (m) | naive single_B ΔZ (r = 2) | literal: \|naive − ΔZ∞(control)\| vs b(L) | like-for-like: \|naive − ΔZ(control, r = 2)\| |
|---|---|---|---|---|
| ratio2 | 0.30 | 0.176-1.434j | 1.213 vs 0.642 → outside | 0.143 |
| ratio2 | 0.60 | -0.03992-0.8568j | 0.790 vs 0.500 → outside | 0.039 |
| ratio2 | 1.20 | -0.08736-0.7551j | 0.764 vs 0.500 → outside | 0.005 |
| ratio4 | 0.30 | 0.05322-1.388j | 1.164 vs 0.642 → outside | 0.274 |
| ratio4 | 0.60 | -0.06781-0.8462j | 0.783 vs 0.500 → outside | 0.069 |
| ratio4 | 1.20 | -0.0928-0.7579j | 0.768 vs 0.500 → outside | 0.012 |

**NEC-5 is first order on every configuration** (step ratios), so the registered
competing outcome — NEC-5 non-convergent at a radius step — did not occur:

| config | L | NEC-5 steps r1→2, r2→4 (\|ΔZ\|) | ratio |
|---|---|---|---|
| control | 0.30 | 1.640, 0.811 | 0.49 |
| control | 0.60 | 1.020, 0.424 | 0.42 |
| control | 1.20 | 1.085, 0.391 | 0.36 |
| ratio2 | 0.30 | 1.599, 0.780 | 0.49 |
| ratio2 | 0.60 | 1.000, 0.414 | 0.41 |
| ratio2 | 1.20 | 1.099, 0.391 | 0.36 |
| ratio4 | 0.30 | 1.584, 0.749 | 0.47 |
| ratio4 | 0.60 | 0.990, 0.404 | 0.41 |
| ratio4 | 1.20 | 1.113, 0.391 | 0.35 |
| ratio05 | 0.30 | 1.657, 0.810 | 0.49 |
| ratio05 | 0.60 | 1.050, 0.424 | 0.40 |
| ratio05 | 1.20 | 1.114, 0.392 | 0.35 |

### What this does and does not say

It does **not** make `single_B` the rule. That would be a fit, and check 2
measured continuity breaking under it on the momwire-native rod. What it says
is narrower: **with the observer-side crossing rule applied, momwire's R is
insensitive to the buried rise's radius and sensitive to the above wire's**,
against the physics and against NEC-5. The two spellings differ only in the
above-observer rows of the crossing fill, so either those rows carry the rise's
node physics in a way the derivation missed, or something upstream (radius
plumbing, which member the harness labels "above") is not what it appears.
Diagnosis comes before any change to the rule; it is registered next.

## Diagnosis of step (b) — registered before the run

`diag_radius.py`. No rule is changed by this diagnosis, and `single_B` is not
adopted by fit (check 2 measured continuity breaking under it). Each test below
separates one candidate cause.

| id | test | prediction | verdict |
|---|---|---|---|
| D0 | **resolved settings**, on step (b)'s ratio-2 deck, L = 0.30 m, r = 1, observer-side: record `_radius_per_wire` with each wire's z-range; the per-row radii every call to `_build_J_blocks_subset` / `_accumulate_Z_subset_chunked` receives, with its segments' z-range; and the z-range of the `a_idx` segments the crossing fill is handed | **blind**: wire 0 is the rise at a_rise (0.125 mm), wire 1 the fed wire and radiator at a_top (0.25 mm); every same-medium call's per-row radii equal the owning wire's radius; every `a_idx` segment has z ≥ 0, so the harness's "above" block is the above block | pending |
| D1 | **no crossing** (raised on review): a wholly buried two-wire rod, #1027's geometry (L = 0.60 m, top at −0.20 m, soil A, centre feed), with the lower half's radius a_low = 0.25 / 0.125 / 0.0625 mm and the fed wire and upper half at 0.25 mm; ports matched; both engines, refine 8 → 16 Richardson | **blind**: momwire's R response to the lower half's radius (R∞(a_low) − R∞(0.25 mm)) is within 15 % of NEC-5's at both reduced radii | pending |
| D2i | **step below the node**: step (b)'s crossing rod at L = 0.30 m with the rise split at z = −0.10 m; the lower 0.20 m at a_s = 0.25 / 0.125 / 0.0625 mm, everything from −0.10 m up at 0.25 mm, so the crossing node has one radius and the shipped fill needs no rule; the scope check handed uniform radii; ports matched; r = 2 → 4 Richardson | **blind**: momwire's R response to a_s is within 15 % of NEC-5's at both reduced radii | pending |
| D2ii | **step above the node, in the air**: the rise and the fed wire at a_s = 0.25 / 0.125 mm, the radiator from 0.05 m up at 0.25 mm; the node again has one radius (a_s) | **blind**: momwire's R response to a_s is within 15 % of NEC-5's | pending |

How the outcomes will be read, fixed now:

- **D0 fails** → a harness or plumbing error, and step (b)'s observer rows were not what the record says. Re-run (b) once fixed, before anything else.
- **D1 fails** → the buried same-medium family does not carry mixed radii correctly on its own. The corner rule is not implicated, and step (b) cannot test it until that is fixed.
- **D1 passes, D2i fails** → a radius change near a crossing is mishandled even where the node has one radius, so the problem is outside the corner rule but inside the crossing serve.
- **D0, D1 and D2 all pass** → the plumbing is sound, and **the observer-side corner rule itself is wrong for the rise's node physics**. Back to (a).

## Diagnosis measured

`diag_radius.py` produced `diag_d0.json`, `diag_d1.json` and `diag_d2below.json`,
each with its log. `diag_d2above.log` holds D2ii's refusal. The NEC-5 binary is
the one step (b) used (sha256 7ebf343d…).

| id | verdict |
|---|---|
| D0 | **HIT**. `_radius_per_wire`: wire 0 is the rise, z ∈ [−0.30, 0], at 0.125 mm; wire 1 is the fed wire and radiator, z ∈ [0, 10.556], at 0.25 mm. The same-medium fill ran as four chunked calls, with no `_build_J_blocks_subset` call on this deck: direct and image for the 10 below segments (midpoints z ∈ [−0.2875, −0.0031]) at 0.125 mm, and for the 41 above segments (z ∈ [0.0125, 10.430]) at 0.25 mm. `a_idx` z ∈ [0.0125, 10.430] and `b_idx` z ∈ [−0.2875, −0.0031]. The crossing fill ran once, and Z = 428.8675 − 362.6035j, matching step (b)'s ratio-2, L = 0.30, r = 1 row |
| D1 | **HIT**: ratios 0.988 and 1.003 (table below) |
| D2i | **HIT**: ratios 1.003 and 1.002 (table below) |
| D2ii | **NOT RUN**. momwire refuses the deck. The radius step at z = 0.05 m is an above-side junction, and the serve refuses a crossing deck with an above-side or in-plane OTHER junction ("… only the below axis's completions (the crossing node and the buried hub) are measured (momwire#524 phase 2)"). This is not a verdict. Running it would need a second scope patch into a region nothing has measured, so the run was not forced |

| test | a (mm) | momwire R∞ | NEC-5 R∞ | momwire response | NEC-5 response | ratio |
|---|---|---|---|---|---|---|
| D1 (no crossing) | 0.25 | 624.976 | 618.81 | — | — | — |
| D1 | 0.125 | 658.739 | 652.98 | +33.763 | +34.170 | 0.988 |
| D1 | 0.0625 | 693.074 | 686.71 | +68.097 | +67.900 | 1.003 |
| D2i (step below a one-radius node) | 0.25 | 424.472 | 424.58 | — | — | — |
| D2i | 0.125 | 448.178 | 448.22 | +23.707 | +23.640 | 1.003 |
| D2i | 0.0625 | 469.944 | 469.97 | +45.472 | +45.390 | 1.002 |

**Reading, as registered.** The reading needs D0, D1 and D2 to pass. D0, D1 and
D2i passed, and D2ii could not run. The reading is applied with that gap
stated: **the plumbing is sound, and the observer-side corner rule itself is
wrong for the node physics. U5 returns to (a).** momwire carries mixed radii
correctly in the buried same-medium family and near a crossing whose node has
one radius. The one case not measured is a radius step in the air at a
one-radius node, so an air-side fault is not excluded by measurement. Item 1
below points at the node instead.

### Observations for the re-derivation (not verdicts)

1. **Under observer-side, momwire's radius response is NEC-5's with the two
   radii exchanged.** All at r = 4, for L = 0.30 / 0.60 / 1.20 m:

   | engine | a_top, a_rise (mm) | R response (Ω) |
   |---|---|---|
   | momwire | 0.125, 0.25 | +34.72 / +17.62 / +8.97 |
   | NEC-5 | 0.25, 0.125 | +35.40 / +17.84 / +9.02 |
   | momwire | 0.25, 0.125 | +1.07 / +0.25 / −0.06 |
   | NEC-5 | 0.125, 0.25 | +0.28 / −0.02 / −0.14 |

   Put another way, the shift that observer-side adds over the physics
   (momwire's response minus NEC-5's) is:

   | ratio | shift at L = 0.30 / 0.60 / 1.20 m (Ω) |
   |---|---|
   | 2 | −34.3 / −17.6 / −9.1 |
   | 4 | −68.7 / −35.2 / −18.1 |
   | 0.5 | +34.4 / +17.6 / +9.1 |

   That is proportional to ln(a_A/a_B) (1 : 2 : −1) and falls with L as the
   physics does. It carries the full weight of the rise's own radius term, with
   the sign that cancels it. A complete ln a term is being exchanged between
   the two families somewhere in the fill.
2. **The one-radius spelling at a_B tracks NEC-5 on this rod.** `single_B` was
   run at r = 2 only. There it sits within 0.28 Ω of the control's residual at
   every ratio and length (Pb.4's table).
3. **A uniform fill radius hardly matters on either rod. The mixing is what
   moves Z.** On check 2's rod, `single_A` and `single_B` differ by 4 mΩ in R at
   ratio 2 and 13 mΩ at ratio 4, while observer-side sits 8 and 16 Ω below
   both. On step (b)'s rod NEC-5 says that move is wrong. Check 2's "the naive
   fix is wrong by ≈ 8 Ω" was measured against observer-side, not against a
   reference, and is withdrawn as a claim about correctness.
4. **Continuity does not discriminate the rules across decks.** On check 2's
   rod (feed 4.33 m from the node), observer-side's KCL deficit is 3e-8 to
   7e-8 and `single_B`'s is 1.4e-5 to 2.3e-5. On step (b)'s rod (feed 25 mm
   above the node, r = 2, L = 0.30 m), `single_B` reads 1.4e-6 against
   observer-side's 1.5e-5 at ratio 2, but 4.5e-5 against 2.1e-5 at ratio 0.5.
   Check 2's "observer-side is unique" is a statement about check 2's rod.

None of this makes a uniform spelling the rule. That would be a fit, and item 4
says continuity cannot settle it by itself. The next unit of (a) is
DERIVATION-MIXED-RADIUS.md §8.

## Step (b′): the ladder under one node potential (registered before the run)

DERIVATION-MIXED-RADIUS.md §8.8 identified observer-side's error. Check 3′ put
the one-potential spelling against NEC-5 at a single rung (r = 2,
L = 0.30 m). This re-runs step (b)'s whole ladder under that spelling.
`b_ladder_onepot.py`.

**Setup, changed from step (b) only where stated:**

- **The same decks.** Same builder, configurations, lengths, refinements,
  matched ports and asserted settings.
- **Two momwire spellings at every rung, from one solve each:**
  - `node_B`: line tests at their own wire's radius, and both node rows' point
    tests at the buried member's radius.
  - VC_keep(`node_B`): the same captured matrix with the crossing junction's
    KCL multiplier added and every point term kept.
- **NEC-5.** Its rungs are the ones step (b) banked. Before any momwire rung,
  NEC-5 is re-run at L = 0.30 m, r = 4 in each configuration, and the run stops
  unless the value reproduces the banked one exactly.
- **Asserted at every rung:** momwire's segment count equals the banked NEC-5
  count. On the control, `node_B` reproduces step (b)'s banked momwire Z to
  ≤ 1e-9 relative, because at equal radii it is the shipped fill. The control's
  ΔZ∞ and band b(L) are therefore step (b)'s.
- **Pb.2's definitions, unchanged:**
  - ΔZ = NEC-5 − momwire at a rung.
  - ΔZ∞ = 2·ΔZ(r = 4) − ΔZ(r = 2).
  - b(L) = the control's \|ΔZ(r = 4) − ΔZ(r = 2)\|, floored at 0.5 Ω, which
    gives 0.642 / 0.500 / 0.500 Ω.
- **The slope is not read on this rod.** The source sits 25 mm above the node
  (Pb.3's lesson). The slope target stays open for a separate unit on check 2's
  rod, where the feed is 4.33 m away: a derived target, or a secant readout over
  a fixed physical length refined in r.

| id | prediction | verdict |
|---|---|---|
| Pq.1 | **the U5 gate** (Pb.2 under `node_B`), **informed** by check 3′'s r = 2, L = 0.30 m rows: \|ΔZ∞(mixed) − ΔZ∞(control)\| ≤ b(L) at every mixed configuration and length |  |
| Pq.1k | the same gate for VC_keep(`node_B`), **informed** likewise |  |
| Pq.2 | split ≡ merged, **informed** by P3′.2: \|Z(node_B) − Z(VC_keep(node_B))\| ≤ 0.1 Ω at every rung |  |
| Pq.3 | the split's continuity, **blind**: `node_B`'s KCL deficit is ≤ 1e-5 at r = 4 in every configuration and length, and falls from r = 1 to r = 4 in each |  |
| Pq.4 | refinement, **blind**: `node_B`'s momwire step ratio \|Z(4) − Z(2)\| / \|Z(2) − Z(1)\| lies within ±0.15 of the control's momwire ratio at the same L, so the one-potential spelling refines like the equal-radius fill |  |

How the outcomes will be read, fixed now:

- **Pq.1 hits everywhere** → §7's gate item 2 is met on step (b)'s rod. Step (c)
  is still held, for U3's GE −1 scope and Steve's decision.
- **Pq.1 misses where Pq.1k hits** → on the ladder the multiplier matters for Z,
  so the src rule carries it.
- **Pq.1 and Pq.1k both miss** → one node potential is not enough on the full
  ladder, and the r = 2 agreement was a single-rung coincidence. Back to §8.
- **Pq.3 misses** → the split's continuity does not converge under a_n = a_B.
  §7's KCL gate then chooses the multiplier, whose KCL is exact by construction.
- **Pq.3 hits** → the split can stand. §7's KCL gate gets a bar derived from this
  ladder's r = 4 values before any src test is written.
- **Pq.4 misses** → the one-potential spelling refines differently from the
  equal-radius fill. That is reported beside Pq.1, and Pq.1 is also read at
  r = 4 alone.

## Step (b′) measured

- **Code and outputs.** `b_ladder_onepot.py` wrote
  `bq_{control,ratio2,ratio4,ratio05}.json` with their logs, and
  `bq_verdicts.py` wrote `bq_verdicts.log`.
- **NEC-5 spot re-run.** It reproduced the banked value exactly in all four
  configurations.
- **Control.** `node_B` reproduced step (b)'s banked momwire Z at every rung.
- **Segment counts** matched at every rung.

| id | verdict |
|---|---|
| Pq.1 | **HIT**. \|ΔZ∞(mixed) − ΔZ∞(control)\| at L = 0.30 / 0.60 / 1.20 m is 0.074 / 0.027 / 0.008 Ω at ratio 2, 0.105 / 0.035 / 0.013 Ω at ratio 4, and 0.037 / 0.014 / 0.017 Ω at ratio 0.5. The bands are 0.642 / 0.500 / 0.500 Ω. Observer-side's differences were 10.7–96.6 Ω |
| Pq.1k | **HIT**, with the same numbers to ≤ 1 mΩ |
| Pq.2 | **HIT**. The worst \|Z(node_B) − Z(VC_keep)\| over every rung is 0.3 mΩ |
| Pq.3 | **MISSED**. At r = 4 the split's KCL deficit is 2.1e-5 / 1.0e-5 / 4.5e-6 at ratio 2, 2.4e-5 / 1.2e-5 / 5.2e-6 at ratio 4, and 3.6e-5 / 1.8e-5 / 8.2e-6 at ratio 0.5. That is above 1e-5 at L = 0.30 and 0.60 m in every mixed configuration. It also does not fall steadily: at ratios 2 and 4 it dips at r = 2 (1.1e-6 to 1.0e-5) and rises again at r = 4, and at ratio 0.5 it barely moves (3.5e-5 → 4.6e-5 → 3.6e-5 at L = 0.30 m). The control falls from 7.2e-5 to 3.2e-6 as before |
| Pq.4 | **HIT**. momwire's step ratios are 0.68–0.91 against the control's 0.69–0.79, all within ±0.15 |

**One caveat on Pq.1.** momwire's own ladder is slower than first order on
the control too (step ratios 0.69–0.79, where first order would give 0.5), so a
first-order Richardson extrapolation of the residual is not exact here. Read at
r = 4 alone, the mixed-minus-control residual is 0.005–0.186 Ω, which is also
inside every band.

**Reading, as registered:**

- **Pq.1 hits everywhere** → §7's gate item 2 is met on step (b)'s rod, under both
  one-potential spellings. Across the whole ladder, the two-radius crossing node
  now sits inside NEC-5's equal-radius residual band, where observer-side was
  21–150 times its band outside. Step (c) is still held, behind U3's GE −1 scope and
  Steve's decision.
- **Pq.3 misses** → the split's continuity does not converge under a_n = a_B, so
  §7's KCL gate chooses the multiplier. VC_keep's KCL is exact by construction
  (asserted ≤ 1e-9 at every rung), and its Z equals the split's to 0.3 mΩ.
