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

## 0. The result so far in one paragraph (updated after check 2; SUPERSEDED by step (b), see the note that follows)

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

**Note after step (b) and its diagnosis (MEASUREMENTS.md).** Against NEC-5,
observer-side is wrong. Its R responds to the radius above the node the way the
physics responds to the rise's radius, through a shift ∝ −ln(a_A/a_B). The
plumbing is sound (D0, D1 and D2i hit), so the rule itself is at fault. Check
2's uniqueness holds on check 2's rod only, because continuity does not
discriminate the rules on step (b)'s rod. The "≈ 8 Ω" was measured against
observer-side, not against a reference. The derivation is reopened in §8.

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
## §4′ — check 2 measured, and the correction it forces [pinned on check 2's rod only; the rule is refuted against NEC-5 by step (b)]

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

**Status after (b): refuted against NEC-5.** See the §0 note and §8.

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

## §8 — reopened after step (b) [2026-09-13; timeboxed]

**What is established.** The buried same-medium family carries mixed radii
correctly: D1 matches NEC-5 to 1.2 %, D2i to 0.3 %, and D0 confirms the per-row
radii reach the kernels. A uniform radius through the whole crossing fill hardly
moves Z (4–13 mΩ), and on step (b)'s rod it tracks NEC-5. Observer-side's mixing
moves Z by −k·ln(a_A/a_B), where k is the physics' own coefficient for the
rise's radius. In effect, the rise's radius term is replaced by the above wire's.

**The question.** Each completion term in §1's list exists to complete a
same-medium block that was filled with a thin-wire kernel at that block's
per-row radius. So:

- which radius does each term need so that it completes its own block rather
  than a neighbouring one?
- which term, under observer-side, removes a ln a_B part and restores a ln a_A
  part, or the reverse?

The candidates are the cross tables' ρ_eff = √(ρ² + a²), the self completions'
closed-form G at √(‖Δr‖² + a²), and the corner V(a). The node grading depends
on the radius only through the quadrature, not through the integrand. §3's
static cancellation fixes only the O(1/a) content. An O(ln a) mismatch passes
check 1 by construction, and item 1 of the diagnosis is O(ln a).

**Rules for this unit.**

- Derive first; register an identity check before measuring any spelling.
- Adopt nothing because it tracks NEC-5. `single_B` doing so is an observation.
- A derived rule must still explain why a uniform fill radius hardly moves Z on
  both rods.

**Timebox.** If the source of the ln(a_A/a_B) term is not identified within
about two working days of 2026-09-13, stop and write up what is known instead.
Step (c) stays parked until then.

### §8.1 The node's point tests evaluate one potential on two surfaces [derived]

By parts on a half-tent leaves f(0)·Φ(0) at the node, so part of a node row's
test is a POINT evaluation of the potential there. Per row, those point tests
are:

- the self completion's node row term and node corner;
- in t_ab, the test-side end terms at the above node (BT on V, TW on W) and
  the corner;
- in the below row, by transpose, the end terms at the below node (SQ on V,
  SW on W) and the corner.

Everything else in a node row is a line test.

With the current continuous through the node (c_na = c_nb), the merged row
(na + nb) is the physical test ∫ f E over the whole node tent. By parts over
the whole tent it has no point term: it is −∫ f′Φ. The potential's jump across
the node, J = Φ_A(0) − Φ_B(0), enters that form as the field J·δ(s). Split
into two half-tents, the point terms f(0)Φ(0) of rows na and nb carry opposite
signs. **They reproduce the merged test if and only if both evaluate the same
Φ(0).**

Observer-side evaluates row na's point test at a_A and row nb's at a_B. The
thin-wire potential at the node on surface X, from the local line charge q, is
≈ (q/2πε)·ln(2h/a_X). The merged row therefore carries an extra
(q(0)/2πε)·ln(a_B/a_A): it drops the field of the jump. The solution can then
hold Φ_A(0) ≠ Φ_B(0) at no cost, which amounts to an uncounted EMF of size J at
the node.

Compare §3. The O(1/a) corner content is the point test against the point
charges, and check 1 pinned its cancellation. The O(ln a) content is the point
test against the LINE charges, and check 1 never evaluated it.

### §8.2 Size and sign, and why the radii look exchanged [derived]

On a short buried rise (L ≪ λ_m) the line charge is nearly uniform, q ≈ Q/L.
The rise's potential is then dominated by its own log term,
(Q/2πεL)·ln(L/a_B), which is the radius dependence NEC-5 and D1 show. Under
observer-side the above side sees Φ_A(0) = Φ_B(0) + J, with
J = (Q/2πεL)·ln(a_B/a_A). That replaces the rise's ln(1/a_B) by ln(1/a_A).

The shift is therefore ∝ −ln(a_A/a_B)/L, carries exactly the physics' weight,
and makes the rise behave as if it had the above wire's radius. This is
observation 1 in form, sign and size.

### §8.3 What the same-medium junctions already do [cited from src]

At a same-medium junction, `_build_basis_polynomials` keeps each wire's value-1
end basis as a directional basis and adds a Lagrange-multiplier KCL row,
`kcl_A` (±1 per outflow), which `_solve_with_kcl` solves. The multiplier enters
every node row as one unknown, a single node potential, and it is not evaluated
at any radius. Φ is single-valued at the node, so J is counted.

D1 and D2i put buried radius steps through that path, and both match NEC-5 to
about 1 %. The crossing node has no such row. The fill evaluates point tests
there instead, which momwire#524 phase 2 measured as split ≡ merged ≡
V-constrained at one radius.

### §8.4 The candidate rule [derived; not implemented]

**Every evaluation takes the radius of its observation point, and the crossing
node is ONE observation point.**

- Line tests keep the observer wire's radius, as §5 and the same-medium blocks
  do. That covers the main sandwich, the source-end terms seen from a line and
  the self column terms.
- Every point test at the node, in both node rows, uses one radius a_n.

Consequences, derived:

1. **a_n is a gauge for Z.** The merged row no longer depends on a_n, because
   its point terms cancel under continuity, as the node's point charges do.
2. **The corner still cancels per row.** All four corners sit at a_n, so their
   O(1/a) content cancels in each row, and check 1 and check 2's row condition
   both still hold.
3. **Continuity is small but not exact.** In the split form it is held by the
   corner's 1/a_n stiffness. A split row alone is no longer a pointwise test on
   its own surface, so the KCL deficit stays small but does not reach
   observer-side's 1e-8. That is observation 4.
4. **The uniform spellings are this rule.** `single_A` and `single_B` are this
   rule with a_n at that radius, plus O(a ln a) changes to the line terms. That
   accounts for observations 2 and 3: `single_B` tracking NEC-5 is a
   consequence, not a fit.
5. **A multiplier twin, VC, needs no radius at the node.** Drop the node point
   tests from both node rows and add a KCL row, exactly as §8.3's junctions do.
   The column terms from the node's point charges stay at each line observer's
   radius, and they cancel under the constraint. VC is the reference for
   consequences 1 to 3.
6. **Equal radii.** Every spelling reduces to the shipped fill.

**The slope reference changes too.** 1/ε̃ is the one-radius AGARD condition:
potential continuity at the node with the same log weight on both sides. With
one node potential and two surface log weights, the node's charge ratio moves
by O(ln(a_A/a_B)/ln(h/a)). At mixed radii the slope therefore cannot be checked
against 1/ε̃. Step (b)'s Pb.3 and check 2's slope column were read against a
reference that does not apply, and §7's slope gate needs a derived target.

### §8.5 Check 3, registered before the run

`check3_node_potential.py`, harness-side (no src change). The spellings
(line-test radii / node point-test radii):

| spelling | line tests | node point tests |
|---|---|---|
| `obs` | own wire | row na at a_A, row nb at a_B |
| `node_A` | own wire | both rows at a_A |
| `node_B` | own wire | both rows at a_B |
| `single_B` | a_B | a_B |

Each spelling is assembled piecewise. VC is built from each run's captured Z
by removing that run's node point tests from rows na and nb, then adding the
crossing junction's KCL row.

| id | test | prediction | verdict |
|---|---|---|---|
| C3.0a | equal radii (1 mm), check 2's rod, refine 1, all four spellings | reproduce the unpatched Z to ≤ 1e-9 relative (asserted; the run stops otherwise) | pending |
| C3.0b | equal radii: VC against the shipped split | **derived** (split ≡ V-constrained, momwire#524 phase 2): \|ΔZ\|/\|Z\| ≤ 1e-4, and VC's slope ratio within 1e-3 of 1/ε̃ | pending |
| C3.0c | VC built from `obs`, `node_A` and `node_B`, every row | agree to ≤ 1e-9 relative (the same terms remain) | pending |
| P3.1 | a_n is a gauge: check 2's rod, a_A = 1 mm, ratios 2 and 4, refine 1 and 2 | **derived**: \|Z(node_A) − Z(node_B)\| ≤ 0.1 Ω | pending |
| P3.2 | one node potential | **derived**: \|Z(node_X) − Z(VC)\| ≤ 0.1 Ω, X = A and B, same rows | pending |
| P3.3 | the jump is the shift | **derived**, with the ln scaling informed by check 2's observer − single rows: Re[Z(obs) − Z(VC)] at ratio 4 over ratio 2 lies in [1.85, 2.15], at both refines | pending |
| P3.4 | the line rule barely matters | **derived**: \|Z(VC from `single_B`) − Z(VC from `obs`)\| ≤ 0.1 Ω | pending |
| P3.5 | continuity | **informed** by check 2's `single_A` / `single_B` KCL rows: `node_A` and `node_B` deficits ≤ 1e-3 | pending |
| P3.6 | slope at mixed radii | **derived, direction only**: VC's slope ratio differs from 1/ε̃ by more than 0.05 (relative) at ratios 2 and 4 | pending |
| P3.7 | step (b)'s rod, L = 0.30 m, r = 2, ratios 2, 4 and 0.5, against the NEC-5 values in `b_*.json` at the same rung | **informed** by observation 2: \|ΔZ(VC) − ΔZ(control)\| ≤ 0.5 Ω, with ΔZ = NEC-5 − momwire. Also reports P3.1 and P3.2 on this rod | pending |

How the outcomes will be read, fixed now:

- **C3.0b misses** → the VC construction (which terms count as node point tests)
  is wrong. Fix it before reading anything else.
- **P3.1, P3.2 and P3.3 hit** → the ln(a_A/a_B) term is identified as the node's
  point tests evaluated on two surfaces, and the timebox's question is answered.
  The src candidates are VC, which is preferred (no radius choice at the node,
  the same construction as the same-medium junctions), or the point tests at
  one a_n.
- **P3.1 hits, P3.2 misses** → a_n is a gauge, but the split is not the
  one-potential form. The missing piece is elsewhere in the node rows.
- **P3.1 misses** → the point terms do not cancel in the merged row, and §8.1 is
  wrong.
