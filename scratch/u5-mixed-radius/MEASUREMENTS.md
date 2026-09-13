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
