# U5 — the mixed-radius crossing corner: which radius a two-radius node uses

Session artifact, 2026-09-13 (Haswell). Plan unit U5
(antennaknobs `docs/plan-buried-scope-closure.md`): the crossing serve refuses
a node where the members' radii differ — "rho_eff = sqrt(rho^2 + a^2)
regularizes the corner with ONE wire radius, and a mixed-radius convention is
not pinned" (`_below_interface.crossing_junctions`, momwire#524 phase 2). Real
radial screens are thinner than the mast, so this is the refusal users meet
first.

Nothing in `src/momwire` is touched. Status marks as in the phase-2 records:
**[derived]** = algebra on this page, **[pinned]** = verified numerically by a
check in this directory, **[open]** = awaiting one. NEC-5 appears only as
conclusions verified against our licensed materials.

## 0. The result so far in one paragraph (updated after check 2)

Every place the crossing fill uses a radius reads one scalar,
`CrossingContext.a_wire` = `_radius_per_wire[0]`: the cross tables' pair
distance ρ_eff = √(ρ² + a²), the corner V(a), the self completions' closed-form
G at R = √(‖Δr‖² + a²), and the log-grading scale of the node-adjacent
quadrature. The candidate rule is **observer-side**, the convention momwire
already ships for same-medium mixed radii (momwire#147): each evaluation uses
the radius of the member its observer lies on. At a two-radius node the node's
four corner contributions then keep their O(1/a) static parts cancelling
exactly [pinned, check 1], and each block's by-parts identity keeps one radius
throughout [derived]. The price is that the two cross blocks stop being
transposes of each other: t_ba = t_ab(a_B)ᵀ, not t_ab(a_A)ᵀ. At equal radii the
rule is the shipped spelling. **Check 2 pins it at the row level** (§4′): observer-side is
the only rule under which continuity still emerges at a two-radius node
(KCL 3e-8); source-side and the harmonic mean break it, and the naive
one-radius spelling is off by ≈ 8 Ω.

## 1. Where the one radius enters [derived from src, 1dbd384 ≡ 42112b9 in src/]

`_crossing_fill`, all reading `ctx.a_wire`:

| use | function | role |
|---|---|---|
| cross main sandwich + ends | `_main_split`, `_ends_and_corner` via `_near_interface.radius_tables` | ρ_eff = hypot(ρ, a) on every cross pair |
| cross corner | `_ends_and_corner` | −σ_a σ_b c₁ V(ρ = a, z = z′ = 0) |
| self completions | `self_completions` → `_bnd_and_corner` | G = e^{−jkR}/R at R = √(‖Δr‖² + a²), row, column and corner terms |
| node-adjacent quadrature | `axis_data` → `_graded_u` | log-grading starts at the a-scale |

and the call site (`_below_interface.compute_Z_operator_buried`) composes
`Z −= t_ab; Z −= t_ab.T; Z += self_completions(ctx, ax_b, ax_a)`. The refusal
is the radius spread check at the end of `crossing_junctions`. Nothing else on
the buried path reads a radius: the below/below and above/above remainders and
images sit at R₁ ≫ a, and the same-medium MP blocks already take per-observer-
row radii.

## 2. The existing convention for two radii [cited from momwire's docs]

`docs/sinusoidal_basis_design.md` §"Per-wire radius (momwire#147)": the source
current is a filament on the source axis and the boundary condition is enforced
on the OBSERVER segment's surface, ρ′ = √(ρ² + a_obs²). Self terms use the
wire's own radius; mutual terms between wires of different radii use the
observer's. **The opposite (source-side) convention was tried first and refuted
against PyNEC**: on a two-radius dipole the delta grew from ~0.8 Ω at N = 21 to
~11.6 Ω at N = 41, at exactly the in-line near-junction pairs where the two
conventions diverge. With the observer convention it was ~0.3 Ω and stable
under refinement. The BSpline family applies the same convention per observer
row. (momwire#147's issue text *proposed* the source-side convention; the
shipped and validated one is observer-side.)

## 3. The corner at a two-radius node [derived; pinned by check 1]

Above member A (radius a_A) starts at the node (σ_a = −1); below member B
(radius a_B) ends there (σ_b = +1). The node tents na, nb are value-1 there.
The fill adds

    Z[na,na] += c_aa(a_aa)          Z[na,nb] −= c₁V(a_x1)
    Z[nb,nb] += c_bb(a_bb)          Z[nb,na] −= c₁V(a_x2)

(`Z += self_completions`; the corner in t_ab is −σ_aσ_b c₁V = +c₁V, and
`Z −= t_ab`, `Z −= t_abᵀ`). With current continuous through the node the two
node tents carry one coefficient, so the node's corner energy is

    S = c_aa(a_aa) + c_bb(a_bb) − c₁V(a_x1) − c₁V(a_x2).

Each piece is static-dominated at small a: piece ≈ w/a with the SAME weight w
for all three kernels. That equality is the phase-2 record's
DERIVATION-SAME-MEDIUM §3, and check 1 measures a·piece = 14.5416 − 15.8561j for
all three at a = 10⁻⁶ m. Hence

    S = w (1/a_aa + 1/a_bb − 1/a_x1 − 1/a_x2) + O(finite),

and with the self corners at their own radii (a_aa = a_A, a_bb = a_B) the O(1/a)
content cancels **if and only if 1/a_x1 + 1/a_x2 = 1/a_A + 1/a_B**.

Rules that satisfy it: observer-side (a_x1, a_x2) = (a_A, a_B); source-side
(a_B, a_A); the harmonic mean a_x = 2a_Aa_B/(a_A + a_B) on both. Rules that do
not: geometric mean, arithmetic mean, max, min.

**Check 1** (`check1_corner_statics.py`, `check1.log`): a_A·|S| on a soil-A
crossing deck, both radii shrunk together at fixed ratio.

| a_A / a_B | a_A | observer | source | harmonic | geometric | arithmetic | max | min |
|---|---|---|---|---|---|---|---|---|
| 1 (control) | 1e-3 → 1e-6 | 9.18e-3 → 9.18e-6 | same | same | same | same | same | same |
| 2 | 1e-3 → 1e-6 | 9.18e-3 → 9.18e-6 | same | 9.18e-3 → 9.18e-6 | 3.69 | 7.17 | 21.5 | 21.5 |
| 4 | 1e-3 → 1e-6 | 9.18e-3 → 9.18e-6 | same | 9.18e-3 → 9.18e-6 | 21.5 | 38.7 | 64.5 | 64.5 |
| 10 | 1e-3 → 1e-6 | 9.18e-3 → 9.18e-6 | same | 9.18e-3 → 9.18e-6 | 100.6 | 158.4 | 193.6 | 193.6 |

a·|S| falling in proportion to a means S itself is bounded (≈ 9.18, the same
finite dynamic residue as at equal radii). A constant a·|S| is O(1/a) residue.
**Pinned:** observer, source and harmonic keep the corner bounded at every
ratio; the other four do not.

## 4. Narrowing to observer-side [derived — partly WRONG, corrected by check 2 in §4′]

The corner alone leaves three rules. Two arguments separate them.

**Per-block identity (rules out the harmonic mean) [derived, open].** The fill
imposes no continuity row: continuity through the node EMERGES from each
block's field-form equivalence, which rests on the §4 by-parts identity of
DERIVATION-NEAR-INTERFACE, exact at any ONE fixed ρ for the whole block (main
term, test and source end columns, corner). The harmonic mean puts the corner
at a_h while the block's end columns and main term sit at the observer's a.
Each row's identity is then off by c₁(V(a_h) − V(a_obs)) ~ w(1/a_h − 1/a_obs),
O(1/a), and the two rows' errors cancel only in the summed form S. That is only
under a continuity the fill no longer guarantees once each row is wrong by
O(1/a). Observer-side and source-side keep one radius through a whole block, so
each block's identity stays exact. To be checked: emergent continuity on a
mixed-radius crossing rod under each rule (check 2).

**Physics convention (observer over source) [cited, §2].** Both are
block-consistent. momwire's own oracle work refuted source-side on in-line
near-junction pairs, which is exactly the crossing node's geometry. The
observer-side rule is also the only one that matches what the same-medium
blocks of the SAME deck already do, so a crossing deck would carry one
convention throughout rather than two.

## 5. The candidate rule, spelled for the fill [derived; not implemented in src]

For each crossing node, with a_X the radius of member X:

1. **Cross tables and ends**: the above-observer block t_ab uses
   ρ_eff = √(ρ² + a_A²); the below-observer block uses a_B. By reciprocity at a
   fixed radius, t_ba(a_B) = t_ab(a_B)ᵀ, so the fill computes t_ab twice (at a_A
   and at a_B) and composes `Z −= t_ab(a_A); Z −= t_ab(a_B)ᵀ`.
2. **Corner**: in t_ab(a) it is c₁V(a) — carried by rule 1 automatically.
3. **Self completions**: each family at its own member's radius (the above
   family at a_A, the below family at a_B; with N below members of different
   radii, per member).
4. **Grading**: `_graded_u` is a quadrature choice, not physics; grade both axes
   from min(a_A, a_B) so the smaller radius's ln-class end integrals resolve on
   both blocks.
5. **Equal radii**: every rule above collapses to the shipped spelling; keep the
   single-transpose path so the catalog BRV stays bit-identical (the U5 gate).

## 6. Still open before (b)

- **Check 2, registered before the run** (`check2_continuity.py`). A
  momwire-native crossing rod: below arm 2 m, above arm 10 m, fed at 4.33 m,
  soil A, 7 MHz, probe18's g2 grading at refine 1 and 2 (uniform, so the source
  region refines with the rest), per-wire radii [a_B, a_A] with a_A = 1 mm and
  a_A / a_B = 2 and 4. Node segments are 12.5 / 6.25 mm, so Δ/a ≥ 25. The
  spellings are harness-side, as the script's docstring states.

  | id | prediction | verdict |
  |---|---|---|
  | C2.0 | control: at a_A = a_B all five spellings (observer, source, harmonic, single_A, single_B) reproduce the unpatched solver's Z bit for bit; asserted, and the run stops otherwise | **MET** — all five 169.7772371 − 82.27633482j, KCL 6.39e-8 |
  | P2.1 | observer and source: the node's KCL deficit \|I(0⁺) − I(0⁻)\| / \|I(0⁺)\| stays below 1e-5 at both ratios and both refines (the equal-radius crossing rod reads ~1e-7) | **MISSED** — observer 6.8e-8 / 4.4e-8 / 2.9e-8 / 2.9e-8 (hit); source 0.63 / 0.70 / 0.85 / 0.89 (miss) |
  | P2.2 | harmonic mean: the deficit exceeds 1e-3 at both ratios, from the per-row O(1/a) identity error of §4 | **HIT** — 0.33 / 0.38 / 0.45 / 0.49, and Z collapses (−129 − 1091j at ratio 2) |
  | P2.3 | observer and source slope ratios I′(0⁺)/I′(0⁻) agree with each other within 1 % at every rung. No prediction against 1/ε̃ at mixed radii: the AGARD slope condition is a single-radius statement | **MISSED** — observer is within 2.3e-4 … 4.9e-4 of 1/ε̃; source is 56 % … 193 % off |

  Readings, not verdicts: `single_A` and `single_B`, the whole fill at one
  radius.
- Whether N below members of differing radii need anything beyond per-member
  self completions (the fan).
EOF
## §4′ — check 2 measured, and the correction it forces [pinned]

`check2_continuity.py`, `check2.log`, `check2.json`. Crossing rod, a_A = 1 mm,
node KCL deficit \|I(0⁺) − I(0⁻)\| / \|I(0⁺)\| and slope ratio against 1/ε̃:

| a_A / a_B | refine | spelling | Z | KCL deficit | slope vs 1/ε̃ |
|---|---|---|---|---|---|
| 1 | 1 | all five | 169.7772371 − 82.27633482j (bit-identical to unpatched) | 6.39e-8 | — |
| 2 | 1 | observer | 169.6059 − 83.1016j | 6.77e-8 | 2.3e-4 |
| 2 | 1 | source | 169.6059 − 83.1016j | 0.628 | 0.558 |
| 2 | 1 | harmonic | −129.39 − 1090.57j | 0.328 | 2.37 |
| 2 | 1 | single_A | 177.5867 − 86.3522j | 4.18e-4 | 0.652 |
| 2 | 1 | single_B | 177.5909 − 86.3474j | 1.35e-5 | 0.549 |
| 2 | 2 | observer | 169.2843 − 83.1924j | 4.44e-8 | 4.8e-4 |
| 2 | 2 | source | 169.2843 − 83.1924j | 0.701 | 0.875 |
| 2 | 2 | harmonic | −114.13 − 1069.68j | 0.378 | 1.85 |
| 4 | 1 | observer | 169.4695 − 83.7803j | 2.92e-8 | 2.6e-4 |
| 4 | 1 | source | 169.4695 − 83.7803j | 0.845 | 1.17 |
| 4 | 1 | harmonic | −0.82 − 1001.40j | 0.450 | 7.89 |
| 4 | 1 | single_A | 185.2740 − 90.5313j | 7.50e-4 | 1.30 |
| 4 | 1 | single_B | 185.2873 − 90.5231j | 1.63e-5 | 1.15 |
| 4 | 2 | observer | 169.1463 − 83.8802j | 2.85e-8 | 4.9e-4 |
| 4 | 2 | source | 169.1463 − 83.8802j | 0.891 | 1.93 |
| 4 | 2 | harmonic | 0.69 − 988.16j | 0.494 | 6.33 |

(The refine-2 single_A / single_B rows are in the log.)

**The correction.** §4 claimed that any rule keeping ONE radius through each
BLOCK keeps continuity emerging, and so kept source-side as a candidate. Check 2
refutes that. The constraint is per ROW, across blocks. Each node row also holds
its member's self completion, whose corner sits at the member's own radius, so
the static O(1/a) content of row na is w(1/a_A − 1/a_x1) and of row nb
w(1/a_B − 1/a_x2). Both vanish only when a_x1 = a_A and a_x2 = a_B:
**observer-side is the unique rule for which every node row cancels its own
corner singularity.** Source-side cancels only in check 1's summed form, like
the harmonic mean, and its currents break continuity by 63–89 %. Check 1 was
necessary but not sufficient. Check 2 is the row-level test, and observer-side
alone passes it, with continuity at 3e-8 and the AGARD slope within 5e-4 of
1/ε̃ at radius ratios 2 and 4.

**Z alone cannot tell observer-side from source-side.** Their Z agree to every
printed digit, because the source-side matrix is the observer-side matrix
transposed: its cross blocks are exchanged transposes and its self blocks are
symmetric. A driving-point impedance vᵀZ⁻¹v is the same scalar for Z and Zᵀ.
The currents differ, and only they discriminate. **Step (b) must therefore read
node continuity and the slope, not only Z against NEC-5**: an engine
comparison of Z cannot see a transposed rule.

**The naive fix is wrong by about 8 Ω.** Removing the refusal and reading one
radius (`single_A`, `single_B`) gives Z ≈ 177.6 Ω against observer-side's 169.6
at ratio 2 (≈ 185.3 against 169.5 at ratio 4), and breaks continuity by 1e-5 to
1e-3.

**Status of the candidate rule:** observer-side, as spelled in §5, now pinned at
the row level on a momwire-native two-radius crossing rod. It goes to (b), the
two-engine ladder, with the source region refined as its own axis and currents
read beside Z.

## §7 — the gate for the eventual src PR [registered 2026-09-13, before any src change]

The momwire PR that lifts the refusal and implements §5 is gated on all three:

1. **No regression at equal radii:** the catalog `buried_radial_vertical` is
   bit-identical to the shipped fill (the single-transpose path is kept when
   the radii are equal).
2. **The mixed-radius ladder inside the equal-radius band:** step (b)'s Pb.2
   on the two-radius crossing rod, and (c)'s four corpus decks once U2's
   imported-deck refinement lands.
3. **Tests pin node continuity and the slope, not only Z** (raised on review).
   A transposed (source-side) rule gives the same driving-point Z to every
   digit (§4′), so a Z-only test would pass it silently. The PR's tests must
   assert the KCL deficit at a two-radius crossing node (observer-side reads
   ~3e-8, source-side ~0.6–0.9) and the slope ratio against 1/ε̃ (observer-side
   within ~5e-4), so that a future transpose regression fails a test.
