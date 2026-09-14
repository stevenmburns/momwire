# U9 — more than one crossing node per deck: registration

- **Plan of record:** antennaknobs `docs/plan-buried-scope-closure.md`, section
  "U9 — more than one crossing node".
- **Base:** momwire main 1ca872511 (0.55.0). Branch `u9-multi-crossing`.
- **Registered** 2026-09-14, before any source change and before any Z run.
  Amendments go in before the runs they amend. A miss is reported as a miss.

## Measured before this registration (geometry and plan only, no fill, no Z)

Three preflights, each in this directory with its log. None of them computes an
impedance.

**pre1, the rod ladder's grazing angle** (`pre1_theta_rod_ladder.py`).
- **The deck.** Two copies of `test_crossing_serve_524.crossing_deck(1)`, the
  second shifted by d along x, one feed per monopole. Soil A at 7 MHz, where
  λ_m = 10.019 m and the 4 λ_m cap sits at 40.1 m.
- **The quantity.** The shallowest below/below pair angle over the fill's own
  quadrature nodes, from `_pair_extents_below`, the function the serve plan
  uses.
- **Why it is so shallow.** The shallowest below node is 1.688 mm deep, on the
  rise's graded node segment.

| d | θ_min | R1_max |
|---|---|---|
| one node | 90° | 0.40 λ_m |
| 5 m | **0.0387°** | 0.64 λ_m |
| 12 m | **0.0161°** | 1.26 λ_m |
| 50 m | **0.0039°** | 5.01 λ_m |
| 200 m | **0.0010°** | 19.97 λ_m |

The floor is `_SOMM_BELOW_TH_MIN_DEG` = 0.05°. **Every two-node rung is under
it.**
- **Where the shallow pair is.** It is always rise to rise across the two nodes,
  at θ ≈ atan(2·h_node / d).
- **Consequence.** The floor refuses any two-node crossing deck whose nodes are
  further apart than 2·h_node / tan(0.05°), which is 3.9 m on this mesh.
- **Why this is not a mesh accident.** Grading a node finer lowers h_node, and
  with it the separation at which the floor starts to refuse.

**pre2, whether the plan refuses at ε̃ = 1** (`pre2_eps1_plan.py`).
- **The call.** `_buried_serve_plan(..., crossing=True)` on the same deck.
- **The result.** It refuses at ε̃ = 1 as well: θ = 0.01612° at 12 m and
  0.0009673° at 200 m. It refuses at soil A at 12 m too.
- **Why that refusal is conservative only.** At ε̃ = 1 the below/below
  integrals are identically zero: both `_six_integrals_below` and its batch
  form return zeros when `eps_t == 1.0`, so there is no table to be wrong.

**pre3, the corpus decks' walls** (`pre3_corpus_walls.py`, on the kwargs
antennaknobs' momwire engine builds at refine 1).
- **Sources.** Deck paths are from AK `scratch/956-census/popb-members.json`;
  hashes are in `decks.sha256`.
- **Setup.** antennaknobs' vertex preflight was stubbed so the solver could be
  constructed; the solver itself ran unpatched.

| deck | crossing nodes | radii above / below | θ_min | R1_max |
|---|---|---|---|---|
| `Phased-Arrays/nec/1r5-bc3elendfire-burrad.nec` | 3, 64.2 m apart | 0.0762 (×3) and 0.019254 (source wire) / 0.0762 (×3 rises) and 0.01 (×38) | **0.0058°** | 21.82 λ_m |
| `Phased-Arrays/nec/1r8-4el-endfire-burrad.nec` | 4, 54.6 m apart | 0.0127 (×4) and 0.016382 (source wire) / 0.0127 (×4 rises) and 0.001 (×49) | **0.0039°** | 25.89 λ_m |
| `LPDAs/nec/lpma3r5-4-6el86ft75o-buriedradials.nec` | 8, 4.72 → 2.89 m apart | 0.00127 (×8) / 0.00127 (×192) | **0.0239°** | 4.68 λ_m |

Each node has exactly one above member. On this main all three refuse on the
second crossing node (#1054).

## (a) The read, and the cross-node corner

### The complete charge

Each basis on the crossing axes carries two kinds of charge:
- the line charge −f′;
- a point charge σ·f(E) at each wire end E where the basis is nonzero. σ is −1
  at a wire's start and +1 at its end.

### The four products

The charge × charge part of the cross block (above test × below source) is
therefore four products:

| product | where it lives |
|---|---|
| line × line | `s_phi` in the main sandwich |
| line × point | SQ |
| point × line | BT |
| point × point | the corner |

W's two by-parts remnants, SW and TW, pair a vertical current with a point
charge. Both are line × point. There is no point × point W product, which is
why the shipped fill has no W corner at a node either.

### The cross-node form

The point × point product for above end E and below end E′ is

  −σ_E·σ_E′·c1·V(ρ_eff, z = 0⁺, z′ = 0⁻)·f_m(E)·f_n(E′),  with ρ_eff = √(ρ_EE′² + a²),

- **ρ_EE′** is the horizontal distance between the two ends.
- **a** is the wire radius, folded in by the thin-wire reduced distance.
- **At a shared node** ρ = 0, so this is exactly the shipped −σσ′·c1·V(a).
- **At two nodes** it is the same expression at √(ρ² + a²).

### The same-medium partners

bspline's self completions already carry them:
- `_bnd_and_corner` evaluates its point × point product over every end pair
  (`Gee`) at √(‖Δr‖² + a²).
- So the above × above and below × below cross-node pairs are already in the Z
  of a two-node deck.
- At ε̃ = 1, all four pairs across two nodes (above–above, below–below and the
  two cross pairs) enter Z as +σ_i·σ_j·c1·G(d). Their sum is c1·G(d)·q₁·q₂,
  where q_k is node k's net point charge, which continuity drives to about 0.
- **Without the cross-block pair**, what is left is 2·c1·G(d) times a node
  current at each node. Nothing is left to cancel it.

SG takes no self completions: its base fill's per-shape charges already include
the end deltas at every end. So SG gets the corner change and nothing else.

### Where every term is evaluated

`_crossing_fill` at 1ca872511:

| term | pair it evaluates | node-local? |
|---|---|---|
| main sandwich, `_main_sandwich` / `_main_split` | quadrature node × quadrature node; admissibility on segment-tree box distance | no |
| BT, TW, SW, SQ end loops | end point × the other axis's quadrature nodes, any ρ | no |
| `axis_data` end table | every wire end on the axis whose basis values are nonzero | no |
| self completions `_bnd_and_corner` (both families, image and direct, one- and two-radius) | E × E at √(‖Δr‖² + a²) | no |
| **corner loops in `_ends_and_corner` and `_ends_and_corner_reversed`** | **one `six_point(a, 0, 0)`, under an assert that every in-plane end pair shares a point** | **yes** |
| two-radius blocks | route through `_ends_and_corner` (the corner at `a_below`), so they inherit the change | via the corner |

**Outside the fill:**
- `crossing_node_members`, SG's `_crossing_wing_view` and
  `_refuse_coincident_crossing_members` all loop over every crossing junction.
- The one-node gate is `crossing_junctions`' `len(crossing) > 1` refusal.
- No code reads a single node's position.

**Result of (a): one term, as hypothesised. No second term turned up.** The
grazing floor above is a serve-plan wall, not a fill term. It is reported
alongside (a) because it decides what (c) and (d) can measure.

### The source change this registers (not yet made)

- **Both corner loops.** Replace the assert with ρ = hypot(Δx, Δy).
  - **Same-node pairs** (ρ < 1e-9) keep the shipped call and float expression
    unchanged: `six_point(eps_t, k_p, a_wire, 0.0, 0.0, rtol=_CORNER_RTOL)`.
  - **Cross-node pairs** call `six_point(eps_t, k_p, hypot(ρ, a_wire), 0.0,
    0.0, rtol=_CORNER_RTOL)`, evaluated once per distinct ρ.
- **Scope.** Narrow the #1054 refusal to what stays unserved. Proposed:
  - two-node decks on the TWO-RADIUS spelling stay refused, because nothing
    here measures them;
  - nodes closer than the smallest separation (b) gates stay refused;
  - both by name.
  The final text is settled after (b).

## (b) The go/no-go gate: the ε̃ = 1 collapse

### Decks

**Pair 1** is `crossing_deck(1)` at x = 0:
- the rod runs −2 → 0 and ends at the node (σ = +1);
- the monopole runs 0 → 10 and starts at the node (σ = −1);
- the feed is on the monopole, 4.3333 m along.

**Pair 2** sits at x = d, in one of two spellings:
- **Spelling A.** The same as pair 1. Cross-node σσ′ is **−1** on both cross
  pairs.
- **Spelling B.** Both wires of pair 2 reversed:
  - the monopole runs 10 → 0 (σ = +1);
  - the rod runs 0 → −2 (σ = −1);
  - the feed sits 5.6667 m along, at the same physical point.
  Cross-node σσ′ is **+1** on both cross pairs.

**Common settings.** Ground `(1.0, 0.0)`, sommerfeld, and `n_qp_pair =
_COLLAPSE_N_QP` on both sides.

**Truth.** A free-space `BSplineSolver` with each pair as ONE polyline:
- the same vertices and edge counts;
- spelled in its crossing spelling's above-wire direction (A: −2 → 10; B's pair
  2: 10 → −2), so the port polarities agree;
- feeds at the same physical points.

**Quantity.** The 2×2 Z = inv(Y), where Y comes from `compute_port_solution().y`.
All four entries are compared.

### Probe patches (scratch only; no src edit before (b) reads)

- **P1, the scope.**
  - **What it does.** Removes `crossing_junctions`' second-node refusal: a
    source-level patch of that one `if len(crossing) > 1:`, asserted to match
    exactly once.
- **P2, the floor, at ε̃ = 1 only.**
  - **What it does.** `bspline._pair_extents_below` clamps θ_min up to the floor
    for the plan, and `_sommerfeld_below.remainder_field_proj_below` returns
    zeros.
  - **Why that is allowed.** At ε̃ = 1 the below/below integrals are
    identically zero (pre2).
- **P3, the corner, three modes.** It is a wrapper on
  `_crossing_fill._ends_and_corner`: call it with `corner=False`, then add the
  corner over every in-plane end pair, in the shipped float expression.

| mode | same-node pairs | cross-node pairs |
|---|---|---|
| `same` (control) | as shipped | skipped |
| `cross` (the hypothesis) | as shipped | −σσ′·c1·V(√(ρ² + a²)) |
| `blind` (control) | as shipped | +c1·V(√(ρ² + a²)), the class sign with orientation ignored |

### Guards, read before any Z

- **G-b0.** On the single-node ε̃ = 1 deck (θ = 90°, served unpatched), the
  real `remainder_field_proj_below` returns an all-zero array of the shape P2
  returns.
- **G-b1.** The single-node ε̃ = 1 deck's Z under P3 `cross` is `array_equal`
  to the unpatched Z. The patch adds nothing where there is no cross-node pair.
- **G-b2.** At ε̃ = 1, `six_point(1, k, √(d² + a²), 0, 0)`'s V equals
  e^{−jkR}/R to 1e-8 relative at d = 1 m and 12 m.
- **G-b3.** The P3 wrapper was actually called with cross-node pairs present,
  counted per solve. That count is non-zero in `cross` and `blind`, and is the
  number of cross-node in-plane end pairs.

### Predictions (blind)

| id | prediction |
|---|---|
| **PB1** (the go criterion) | `cross`, d = 12 m, spellings A and B: every Z entry within **0.05 Ω** of the truth, the class bar of g524_5 and g524_6. Point estimate: Z11 and Z22 within 0.03 Ω (g524_5 measured 0.0124 Ω on this mesh class), Z12 within 0.01 Ω. |
| **PB2** (the term matters) | `same`, d = 12 m, both spellings: \|Z12 − Z12_truth\| in **[5, 100] Ω**. |
| **PB3** (orientation is visible) | `blind` is equal to `cross` to 1e-12 relative on spelling A. On spelling B, \|Z12 − Z12_truth\| is in **[10, 200] Ω** (twice the omission). |
| **PB4** (a close pair) | `cross`, d = 1 m, spellings A and B: every Z entry within **0.05 Ω**. |
| **PB5** (reciprocity) | `cross`, d = 12 m: \|Z12 − Z21\| ≤ 1e-9·\|Z12\| on both the patched solve and the truth. |

**Basis for PB2.** The omitted product is 2·c1·G(d)·(node current)².
- The shipped same-node corner is \|c1·V(a)\| ≈ 2.04e5 at a = 1 mm (g524_5's
  docstring), so \|c1\| ≈ 204.
- That gives \|c1·G(12 m)\| ≈ 17 Ω.
- Under the unit drive at its own port, a node's current is O(1). So δZ12 is of
  order 34 Ω.
- δZ11 is smaller, because port 2's node current under port 1's drive is only
  the induced current.

### Reading rules, fixed now

- **G-b0 to G-b3 miss.** The corresponding Z is not read. Fix the probe, and
  record what was fixed.
- **PB1 misses on either spelling: NO-GO.** Stop, report to Laptop-builder,
  and run nothing else. This is the 10-days-or-more branch, and Steve decides.
- **PB1 hits but PB2 does not miss.** The gate cannot see the corner at this
  separation: report it as vacuous, and stop.
- **PB1 and PB2 both hold: GO.** Report to Laptop-builder with the numbers.
- **PB3 on B does not miss.** Orientation is not visible at this separation.
  This is reported and does not block GO.
- **PB4 misses.** The narrowed refusal's minimum separation is set to the
  smallest separation that hits (12 m). This is reported.
- **PB5 misses.** It is a finding about the fill's symmetry. Report it before
  any source work.

## (c) The separation ladder at 5 / 12 / 50 / 200 m on soil A

- **PC0.** The production path refuses all four rungs on the grazing floor, with
  or without the source change. On the src branch with the scope lifted,
  `buried_serve_refusal()` names the floor at θ = 0.0387°, 0.0161°, 0.0039° and
  0.0010°. (These are computed in pre1. The run confirms it on the production
  path.)
- **Rungs 5 m and 12 m.** The sub-floor pairs are inside the cap: R1 is at most
  12.6 m, below 40.1 m. Serving these rungs needs the below/below table below
  its floor.
- **Rungs 50 m and 200 m.** Every sub-floor pair is past the cap, so each would
  be served as zero. But the floor reads zeroed pairs too, which is U4's
  registered design.
- **What is not decided here.** The spelling that measures Z11 → the
  single-node print, Z12 → 0 and Z12 = Z21 is Steve's decision, and no (c) Z
  run happens before an amendment registers it with its predictions. The
  options go to Laptop-builder with the (a) report.

## (d) The three corpus decks

**Walls predicted.** They are listed in the order `plan_buried` asks them, once
the second-node refusal is lifted. Each from pre3.

| deck | first wall | behind it |
|---|---|---|
| **1r5** | U5's radius spread WITHIN a side: below, 0.0762 rises against 0.01 radials; above, 0.0762 elements against the 0.019254 source wire | the grazing floor at 0.0058°; R1 at 21.82 λ_m is past the cap and served as zero, which is not a refusal |
| **1r8** | the same: below, 0.0127 against 0.001; above, 0.0127 against 0.016382 | the floor at 0.0039°; R1 at 25.89 λ_m, zeroed |
| **LPDA** | one radius on both sides, so the grazing floor at 0.0239° comes first | R1 at 4.68 λ_m, past the cap by 0.68 λ_m, zeroed |

- **PD1.** On the src branch, each deck's `buried_serve_refusal()` names the
  first wall in the table above.
- **Correction to the brief.** All three decks hit the grazing floor, not only
  the LPDA. The phased arrays meet U5's spread first.
- **What waits.** U2's `ladder --refine` runs against NEC-5, the LPDA's
  fill-time and peak-memory measurement, and the Z12 / change-from-single-node
  gates all need a served spelling. They wait for the same decision as (c),
  and get an amendment first.
- **What that amendment must also spell.**
  - **Element ports.** The phased arrays are current-fed through `NT` networks
    from a remote wire (wires 70 and 93), so their Z12 needs ports at the
    elements.
  - **The NEC-5 binary.**
  - **The single-node reference decks.**

## PR gates

| gate | prediction |
|---|---|
| **G1.** Single-node decks bit-identical: the crossing rod (g524_4, g524_5), the rise deck (g524_6), the fan (g524_7, g674_1, g674_2), the buried hub, the U5 rod (`test_crossing_two_radius_u5`) and SG (`test_sg_crossing_980_d3`). Method: Z `array_equal` against main in one process per deck, and the named tests pass. | bit-identical |
| **G2.** Split-vs-dense parity on the two-node deck: block-relative, against `test_g688_4`'s 1e-6 gate, on the cross blocks directly. This needs no serve plan, so it runs at soil A. | ≤ 1e-7 |
| **G3.** SG against bspline on the two-node deck at the class tolerance, 1e-2 relative (`test_sg_crossing_980_d3`'s direct fan). This runs at ε̃ = 1 through P2 unless the floor decision gives a served soil. | Z11 and Z12 both ≤ 1e-2 relative |
| **G4.** The #1054 refusal narrowed to what stays unserved and still refused by name; `test_g524_2_*second_crossing_node*` and `test_refusals_are_declared`'s entry updated. | — |
| **G5.** momwire's slow lane and crossgate, run locally. | pass, apart from failures that reproduce on origin/main (U4's record: the 20 s ceiling breaches on this box) |

## Report points

To Laptop-builder:
1. After (a), with the grazing-floor wall and the options for (c) and (d).
   Sent 2026-09-14, after 191ef0b.
2. After (b), with GO or NO-GO.
3. When the PR is up.

## Amendment 1 (2026-09-14, before any (b) run)

- **G-b2 was mis-stated.** It compared `six_point`'s V with e^{−jkR}/R, but at
  ε̃ = 1 the identity is k²·V = e^{−jkR}/R
  (`test_g524_3_eps1_kernel_identity`). The guard therefore compares k²·V,
  at the same 1e-8 relative bar.
- **PB2's basis, re-worded; the band is unchanged.** "\|c1\| ≈ 204" and
  "\|c1·G(12 m)\|" mixed V and G. The estimate is a ratio: the cross-node
  corner at 12 m is \|c1·V(a)\|·a/d ≈ 2.04e5 × 1e-3 / 12 ≈ 17 Ω. The band
  [5, 100] Ω stands.
- **G-b1 is read two ways.** P3 `cross` alone, and P1 + P2 + P3 together, are
  each compared with the unpatched single-node Z. Both are predicted
  bit-identical.
- **G-b3's expected count.** There are two cross-node in-plane end pairs per
  `_ends_and_corner` call on the two-node deck: each pair's above end against
  the other pair's below end. They are counted before any mode skips them.
- **P2 records two non-vacuity facts per run.**
  - The unclamped θ_min is below the floor, which shows the clamp engaged.
  - The zeroed projection was called at least once, and always with
    k_m == k_p.
- **Tooling.** `b_collapse.py` and `run_b.sh`. The runner stops if the guards
  miss, so no (b) Z is written behind a failed guard.

## (b) Results [2026-09-14]

- **Amendment 1 and the tooling** landed at fcf4dca, before the run.
- **The run** was `u9-b-collapse.service` on fcf4dca, 15:50:24–15:50:32Z. The
  records are the `b_*.json` files, the `b_*.log` files and `run_b.out`.
- **Why it took 8 s.** The two-node ε̃ = 1 fill is about 0.2 s per solve, and
  the truth about 0.1 s.

### Guards

Every guard was read before any Z, and all of them hold.

| guard | result |
|---|---|
| G-b0 | The real projection on the single-node deck returns 1 call, of the shape P2 returns, with max \|value\| exactly 0.0. |
| G-b1 | The single-node Z, 17.557770975310156 − 758.501352375456j, is `array_equal` under P3 alone and under P1 + P2 + P3. P3 was called, and 0 cross-node pairs were seen on the single-node deck. |
| G-b2 | k²·V against e^{−jkR}/R is 1.1e-16 at 1 m and 4.9e-16 at 12 m. |
| G-b3 | Every run makes 1 call with 2 cross-node pairs. |
| P2 | At 12 m the clamp engaged, over an unclamped θ_min of 0.01612°, and the projection was called once with k_m == k_p. At 1 m θ_min is 0.1935°, so the clamp was not needed. |

### Predictions

| id | result |
|---|---|
| **PB1** | **HIT.** `cross` at 12 m, spellings A and B. Worst entry 2.024e-4 Ω (Z11 and Z22), 247× under the 0.05 Ω bar. Z12 is within 1.73e-6 Ω. A: Z11 = 17.5125 − 758.4989j against the truth's 17.5125 − 758.4987j, and Z12 = 8.1876 − 9.0050j on both. B reads the same with Z12's sign flipped, which is the reversed port polarity on both sides. |
| **PB2** | **MISS of the band, low.** `same` at 12 m: \|Z12 − Z12_truth\| = 4.6963 Ω on both spellings, against the registered [5, 100] Ω. The estimate assumed an O(1) node current under the port drive. 4.70 Ω is consistent with about 0.37 of the port current at the node, but that ratio is not measured. The omission still misses the truth by 94× the bar, and by 2.7e6× the `cross` residual on Z12. So the gate sees the corner, and the vacuity rule ("PB1 hits, PB2 does not miss") is not triggered. |
| **PB3** | **Half HIT.** Spelling A's `blind` Z is bitwise equal to `cross`, as predicted. Spelling B's band **misses low**: \|ΔZ12\| = 9.386 Ω against [10, 200] Ω. The structural half holds: 9.386 / 4.696 = 1.999, twice the omission, as the sign flip says it must be. So orientation is visible. |
| **PB4** | **HIT.** `cross` at 1 m, A and B: worst entry 2.031e-4 Ω. |
| **PB5** | **HIT.** \|Z12 − Z21\| / \|Z12\| ≤ 1.3e-15 on every patched solve and every truth. |

### Reading, as registered

- **PB1 holds on both spellings, and the gate is not vacuous.**
- **PB2's and PB3's magnitude bands both missed low, by the same 6 %.** Both
  misses come from the same node-current estimate, and they are reported as
  misses.
- **My GO rule was written as "PB1 and PB2 both hold".** Read literally, PB2's
  number did not hold. Read as intended, it held: PB2 exists to show the term
  matters, and the term moves Z12 by 4.70 Ω. I read this as **GO** and say so
  in the report, so Laptop-builder and Steve can read it the other way.
- **PB4 hits,** so the narrowed refusal's minimum node separation is 1 m, the
  smallest separation gated.

### Steve's ruling on (b) [2026-09-14, relayed by Laptop-builder]

**GO.** It is recorded exactly as ruled.
- **The registered rule reads NO-GO literally.** It said "GO when PB1 and PB2
  both hold", and PB2 did not hold.
- **PB2 stays a MISS of its band:** 4.696 Ω against [5, 100] Ω.
- **PB3's spelling B stays a MISS of its band:** 9.386 Ω against [10, 200] Ω.
- **The cause of both is stated:** the O(1) node-current assumption in the
  estimate.
- **The GO is Steve's post-hoc ruling (2026-09-14),** on these grounds:
  - PB1 is 247× under the bar;
  - the omission misses the truth by 94× the bar, so the gate demonstrably
    sees the term;
  - PB4 and PB5 hit.
- **PB2's band is not widened or re-registered after the fact.**

## Amendment 2 (2026-09-14): route 2, registered before the screen and before any (c) or (d) Z run

### Steve's decision (relayed by Laptop-builder)

- **Route 2 now.** Extend the below/below table under 0.05°.
- **Route 1 is deferred.** Serving past-cap pairs at any θ is a named
  follow-up, not work in this PR. It comes back only once the phased arrays'
  other walls are addressed, and only behind its own grazing Z gate.
- **(d) is the LPDA only.**
- **The phased arrays stay refused by name.** Their walls are recorded in (d)
  above: grazing past the table, U5's within-side spread on both sides, and
  NT-fed ports.

### The table change: one family of candidates, none of them needing C++

- **What changes.**
  - Lower `_SOMM_BELOW_TH_MIN_DEG`, which is the low band's first node, and
    keep the four bands.
  - Give the low band a Δθ that divides 0.05/3, so every shipped low-band
    node (0.05 + k·0.05/3) stays a node.
  - Raise `_MAX_TAIL_PANELS`.
- **Why no C++ change is needed.** Read at 1ca872511:
  - `proj_one_below` routes on the two band edges (0.1° and 1°), which do not
    move, and reads each region's th0 / dth from the region arrays;
  - the budget reaches `below_six_integrals_batch` as an argument at call
    time;
  - the transmitted family has its own `_MAX_TAIL_PANELS_T`;
  - `_tail_below` stops on convergence, so a larger budget cannot change a
    node that converges under 8000.

| lattice | floor | low-band Δθ | covers | 6.4/tan(floor) |
|---|---|---|---|---|
| **L2** | 0.016667° (0.05 − 2·0.05/3) | 0.05/3, unchanged | the LPDA (0.0239°); rungs 3/5/8/11 m (0.0645/0.0387/0.0242/0.0176°) | 22,002 |
| **L2p** | 0.016667° | 0.05/6 | the same | 22,002 |
| **L3** | 0.011111° (0.05 − 7·0.05/9) | 0.05/9 | also 12 m (0.0161°) | 33,002 |

**What moves in the shipped domain.**
- **Outside the low band [0.05°, 0.1°]:** nothing. That band is deferred and
  filled only by decks that reach under 0.1°.
- **Inside it:**
  - under L2, the stencil re-centres in [0.05°, 0.0667°), and the rest of the
    band moves by rounding only;
  - under L2p and L3, nodes are added across the whole band.

### The screen (S): no Z

Tooling is `s_table_screen.py` and `run_s.sh`, run on the accelerated path,
which is asserted.

- **S1, panels.**
  - **What is recorded.** The worst tail-panel count and the non-convergent
    count, under a 48,000-panel budget.
  - **Angles.** θ ∈ {0.04, 0.033333, 0.025, 0.02, 0.016667, 0.0125,
    0.011111}°.
  - **Media (8).** SPEC soils A/B/C × 7/21 MHz, plus soil A at the rod ladder's
    `F7`, plus the LPDA's own `GN`/`FR` medium.
  - **Ranges.** R1/λ_m ∈ {0, 0.02, 0.05, 0.2, 1, 2}.
- **S2, interpolation on the real grid, per lattice.**
  - **Queries.** Cell midpoints and thirds of every low-band cell from the
    floor up to 0.0667°, × R1/λ_m ∈ {0.2, 1.0, 1.9, 3.0, 3.9} (all three
    zones), × the 8 media, × the four surfaces.
  - **Reference.** `iv_surfaces_direct_below`, relative to each point's own
    scale. Every reference must converge.
- **S3, cost.** The wall time to fill the low band in all three zones, per
  lattice and medium.

| id | prediction (blind) |
|---|---|
| **PS1** | The worst count is within [1.00, 1.06] × the law at every θ, and non-convergent is 0 everywhere under 48,000. That is [22,002, 23,322] at 0.016667° and [33,002, 34,982] at 0.011111°. |
| **PS2** | L2's worst is ≤ 4.7e-4, with a point estimate in [1e-9, 1e-5]. #935 measured 6.8e-10 along θ with Δθ/θ ≤ 0.33; L2's bottom cell has Δθ/θ = 1, and #935's probe did not cover the far zone. L2p ≤ L2, and L3 ≤ L2p. |
| **PS3** | L2's low-band fill takes ≤ 120 s per medium on this box. L3 takes ≤ 5 × L2. |

### Decision rule, fixed now

1. **A lattice qualifies** if all three hold:
   - its floor is ≤ 0.0239°;
   - S1 converges at its floor;
   - S2's worst is ≤ 4.7e-4, the low band's own bar (#553 U2 and #935).
2. **Pick** L2 if it qualifies, else L2p, else L3. Cost comes first, so L3 is
   not picked merely to keep 12 m. If none qualifies, stop and report: route 2
   on a nested uniform band fails, and Steve decides.
3. **The budget** is S1's worst at the chosen floor over the 8 media, × 1.05,
   rounded up to the next 1,000.
4. **The ladder.**
   - L2 or L2p gives 3/5/8/11 m;
   - L3 gives 3/5/8/12 m;
   - θ_min is re-read with pre1's function on the branch before (c) runs.

### Gates on the table change (0.55.0 G8 style)

- **GT1, banked accuracy.** S2 at the chosen lattice is the banked number. It
  also becomes a slow-lane test: #935's real-grid test, extended to the new
  cells.
- **GT2, single-node decks unchanged.**
  - **The G1 decks** are predicted bit-identical, because none of them reads
    the low band. Each deck's solve also records whether the low band was
    filled.
  - **#935's 3 mm dipole** (0.0582°, inside the re-centred cell) is predicted
    to move by \|ΔZ\|/\|Z\| ≤ 1e-8.
- **GT3, re-pinned refusals.**
  - **What gets updated.** The tests that pin 0.05° or 8000 move to the new
    floor.
  - **`test_the_floor_still_refuses_what_it_cannot_reach`.** Its 1.0 mm case
    (0.0194°) becomes served under L2 or L2p; its 0.5 mm case (0.0097°) stays
    refused.
  - **Stop rule.** A failing test outside the files that name the floor, the
    budget, the low band or the second node is a miss, and a stop.

### (c), on the production path at soil A, after the source change

- **The deck.** Spelling A, two `crossing_deck(1)` pairs, at each rung of the
  chosen ladder; bspline with production defaults; Z = inv(Y).
- **The single-node print.** `crossing_deck(1)`, with its one feed, at the same
  settings.

| id | prediction (blind) |
|---|---|
| **PC1** | Every rung is served. |
| **PC2** | \|Z11(d) − Z_single\| is non-increasing over the four rungs, and ≤ 1 Ω at the widest. |
| **PC3** | \|Z12(d)\| is strictly decreasing over the four rungs, and at the widest rung lies in [3, 30] Ω. |
| **PC4** | \|Z12 − Z21\| ≤ 1e-9·\|Z12\| at every rung. |
| **PC5** | At the 8 m rung, Z under the chosen lattice and under the next finer candidate (L2 → L2p, L2p → L3, L3 → Δθ 0.05/18) differ by ≤ 1e-2 of that rung's own mesh ladder step (every edge count × 3). |

G2 (split against dense) and G3 (SG against bspline at 1e-2) run at soil A on
the 5 m rung, with the predictions in the PR-gates table.

### Screen results [2026-09-14]

- **The run.** `u9-s-screen.service` on 9012369, 16:01:14–16:45:19Z.
- **Records.** `s_panels.json`, `s_interp_L2.json`, `s_interp_L2p.json`,
  `s_interp_L3.json`, their logs, and `run_s.out`.
- **Search note.** The word "errors" in `s_panels.log` is the JSON key, which
  is an empty list on every row.

| id | result |
|---|---|
| **PS1** | **HIT.** Across 0.04 … 0.011111°, worst over law is 1.0321, 1.0277, 1.0207, 1.0152, 1.0108, 1.0038, 1.0009. Every worst case is at C/21 MHz, R1 = 0.05 λ_m. There are no non-convergent points and no errors, over 8 media × 6 R1 values under 48,000. At 0.016667° the worst is 22,239, inside [22,002, 23,322]; at 0.011111° it is 33,032, inside [33,002, 34,982]. |
| **PS2** | **L2's bar and its estimate: HIT.** The worst is 3.94e-9 (the LPDA medium, R1 = 3.0 λ_m, `IphiH`, at 0.02217°), 1.2e5× under the 4.7e-4 bar and inside the [1e-9, 1e-5] estimate. Every reference converged. **The ordering half: MISS.** L2p reads 5.05e-9, above L2, and L3 reads 5.12e-9, above L2p. Each lattice is queried at its own cell midpoints and thirds, and all three land at the same few-1e-9 level. I read that as the comparison's own floor rather than the lattice, but that reading is not measured. |
| **PS3** | **HIT.** L2's low-band fill takes 34.7–47.0 s per medium, against the ≤ 120 s prediction. L3 takes 95.3–143.5 s, 2.8–3.1× L2, against ≤ 5×. L2p, not predicted, takes 51.5–75.6 s. |

### Decision, by the rule fixed in Amendment 2

- **L2 qualifies,** so L2 is chosen:
  - its floor, 0.016667°, is ≤ 0.0239°;
  - S1 converges at 0.016667° on every medium;
  - its S2 worst, 3.94e-9, is ≤ 4.7e-4.
- **Budget:** 22,239 × 1.05 = 23,351, rounded up to **24,000**. The worst case
  uses 92.7 % of it, leaving 7.3 % headroom, against 4.9 % at the old floor.
- **Ladder:** **3 / 5 / 8 / 11 m.** PC5's finer candidate is **L2p**.
- **Tooling committed before the runs it serves:**
  - `g1_single_node.py` for G1 and GT2;
  - `c_ladder.py` for (c) and PC5.
- **PC5's metric,** fixed here before the run: the max over the four Z entries
  of \|Z_L2p − Z_L2\|, against the max over the four of \|Z_mesh×3 − Z_L2\|.
- **Tooling defect, found before any branch run and fixed.**
  - **What was wrong.** `g1_single_node.py` recorded `low_band_filled` from
    `SommerfeldGridBelow._band_lo_filled`, which is true only when all three R1
    zones' low-band regions are filled. So a deck that never reaches the far
    zone read False even where its low band was filled; #935's 3 mm dipole, at
    0.0582°, read False on main.
  - **The same flaw** was in `c_ladder.py`'s PC5 low-band guards.
  - **The fix.** Both now ask whether ANY low-band region is filled.
  - **What it changes.** Main's first G1 run solved every deck and its Z
    values stand. It was re-run so that the field is right.

### GT3: the branch's named-file run, and one re-measure it needs

**The run.** The default lane of the test files that name the floor, the
budget, the low band or the second node (plus the U5 and SG crossing files), on
`u9-multi-crossing-src` with the corner, the scope and L2 in place:
**357 passed, 7 failed.**
- **Where the failures are.** All seven are in the three files that pin
  refusals under the old floor or budget. There are no failures outside the
  named files, so GT3's stop rule is not triggered.
- **What each one is.** Every failure is a refusal at an angle the change now
  serves:
  - five in `test_grazing_band_838`, at 0.023° and 0.03° (they now converge
    under 24,000);
  - `test_below_fills_568`'s grazing row, an 8 mm pair at 0.0457°;
  - `test_grazing_band_lo_935`'s 1 mm dipole at 0.0194°, which GT3 predicted.

**G1 and GT2 [2026-09-14].** The single-node Z was compared bitwise between
main (`g1_main.json`, run on the u9 tree, whose src is main at 1ca8725) and the
src branch with the corner, the scope and L2 (`g1_branch.json`). The comparison
is `g1_compare.json`.
- **Bit-identical, as predicted (HIT).** Eight decks:
  - the crossing rod at soil A and at ε̃ = 1;
  - the rise deck at ε̃ = 1;
  - the fan and the buried hub at soil A;
  - U5's two-radius rod;
  - SG's direct fan (N = 4) and buried hub (N = 2).
  None of them fills the low band.
- **#935's 3 mm dipole (HIT).** It reads the low band on both trees and moves by
  1.1e-17 relative (1 ulp in X), against GT2's ≤ 1e-8.
- **Not predicted: a cost change.** The dipole's solve took 9.8 s on main and
  28.0 s on the branch.
  - **Why.** The low band is filled as one region per zone, so a deck that only
    reaches 0.058° now also pays for the two new nodes, at 0.0167° and 0.0333°,
    each costing up to ~22k panels.
  - **What it costs.** Nothing about the value; every deck that reaches under
    0.1° pays about 3× the low-band fill time.
  - **What would avoid it.** A separate fifth band. That is a C++ layout change,
    and it is reported rather than decided here.

**The re-measure.** #838's served/refused ladder on its own deck (soil A,
7 MHz, R1 = λ_m) is re-measured before it is re-pinned, as that test's comment
requires. Tooling is `t_cap_ladder_24000.py`.

| id | prediction (blind) |
|---|---|
| **PT1** | Served: every rung from 0.12° down to 0.016667°. Refused by name: 0.0125° and 0.01°. At 0.015° the law gives 24,446 against a budget of 24,000, and this deck reads about 0.95 of the law at 0.05° (6959 against 7330), so either outcome is possible and neither is predicted. |
| **PT2** | The numpy dispatch refuses at 0.0125° within 20 s, the fast lane's hard ceiling. |

**PT results [2026-09-14]** (`t_cap_ladder_24000.json`, run on the src branch,
registered at 8c0cdf9):
- **PT1: HIT.**
  - **Served:** every rung from 0.12° down to 0.016667°. Panels:
    0.04° → 8649, 0.03° → 11447, 0.023° → 14828, 0.02° → 16990,
    0.016667° → 20291.
  - **Refused by name:** 0.0125° and 0.01°, each at 24000 panels.
  - **0.015°, not predicted:** served, at 22483 panels.
- **PT2: HIT.** The numpy dispatch refuses at 0.0125° in 6.63 s.
- **What that re-pins.**
  - **#838's ladder:** served (0.12 … 0.02), refused (0.0125, 0.01).
  - **Its three refusal-message tests:** they ask at 0.0125° instead of 0.03°.
- **The other three re-pins, made from geometry.**
  - **#568's grazing row:** a 2 mm pair at 0.011°.
  - **#935's refused dipole depths:** 0.5 and 0.8 mm (0.0097° and 0.0155°).
  - **#838's BLE grid refusal:** at 0.0125°.

### (c) results [2026-09-14]

- **The run.** `u9-c-ladder.service` on src 96dd5be. The ladder ran
  17:05:06–17:07:05Z; PC5 ran 17:07:05–17:08:18Z and exited 1.
- **Records.** `c_ladder.json`, `c_ladder.log`, `c_pc5.log`, `run_c.out`.
- **Single-node print.** Z = 169.7756 − 82.2800j (0.95 s).

| d | θ_min | Z11 | Z12 | \|Z11 − Z_single\| | \|Z12\| | \|Z12 − Z21\|/\|Z12\| | solve |
|---|---|---|---|---|---|---|---|
| 3 m | 0.0645° | 169.5766 − 83.2830j | 67.0946 − 8.1693j | 1.0225 | 67.590 | 2.2e-16 | 27.5 s |
| 5 m | 0.0387° | 169.2952 − 82.8829j | 53.2161 − 17.8160j | 0.7709 | 56.119 | 1.5e-15 | 28.1 s |
| 8 m | 0.0242° | 169.2964 − 82.3938j | 34.8399 − 26.9724j | 0.4925 | 44.061 | 4.6e-16 | 30.3 s |
| 11 m | 0.0176° | 169.4824 − 82.1325j | 18.8171 − 30.5152j | 0.3283 | 35.851 | 8.9e-16 | 31.4 s |

| id | result |
|---|---|
| **PC1** | **HIT.** Every rung is served. |
| **PC2** | **HIT.** \|Z11 − Z_single\| is non-increasing, and 0.328 Ω at 11 m, ≤ 1 Ω. Z22 agrees with Z11 to 1e-11. |
| **PC3** | **The decreasing half HITS:** 67.59 → 56.12 → 44.06 → 35.85 Ω. **The widest-rung band MISSES:** 35.85 Ω against [3, 30] Ω. |
| **PC4** | **HIT.** ≤ 1.5e-15. |
| **PC5** | **NOT READ.** The registered mesh step, every edge count × 3, is itself refused by the grazing floor. Refining the node-adjacent edges puts the rises' shallowest nodes 0.56 mm deep, θ = 0.0081° at 8 m. The L2 and L2p Z were computed, but the refusal came before anything was recorded. |

**Findings.**
- **Refining the node pushes a served deck back under the floor.** A served
  two-node deck sits at the table's edge on the node-mesh axis: refining the
  node-adjacent segments lowers h_node, and θ_min falls under the floor again.
  This is reported, not decided here.
- **Cost.** A two-node solve takes 27.5–31.4 s against 0.95 s for the single
  node, with 110 MB peak RSS. Every rung reads the low band. A served two-node
  solve therefore belongs in the slow lane, not the PR lane.

### G2 and G3 results [2026-09-14]

**The run.** On src 96dd5be with probe 262cdce, spelling A at 5 m, soil A.
- **A record was lost the first time.** The first run went through
  `systemd-run --unit`, which expands `${g}` itself before bash runs. Both
  gates wrote `_d5.json`, and G3's run overwrote G2's.
- **Recovery.** The surviving file was checked to be G3 (`mode: g3`) and renamed
  `g3_d5.json`. G2 was re-run under `--scope` as `g2_d5.json`.
- **What that says about G2.** It is cheap (1.4 s), and nothing about it
  depends on run order.

| id | result |
|---|---|
| **G2** | **HIT.** Split against dense-direct on the 64 × 64 cross block: 1.3e-18 block-relative, against the 1e-7 prediction and the 1e-6 gate. **But this deck cannot see an ACA defect:** at `_FAR_Q == _NEAR_Q` the far blocks share the dense sampling, and no block here passes the ACA cost guard. `test_g688_5`'s dense-mesh deck is the gate that exercises ACA. |
| **G3** | **HIT.** SG against `BSplineSolver(degree=1)` on the 2 × 2 Z: Z11 2.88e-3 and Z12 8.42e-3 relative, against the 1e-2 gate. Z12's margin is 1.19×. SG takes 28.9 s against bspline's 0.2 s. |

## Amendment 4 (2026-09-14, before PC5 is re-run)

- **PC5's mesh step becomes the FAR mesh × 3:** every edge count × 3 except
  each wire's node-adjacent edge.
  - **Why that edge is kept.** It sets h_node, so keeping it leaves θ_min
    unchanged.
  - **Why this axis.** It is the far-mesh axis `crossing_deck`'s own comment
    names, where degree 2 moves 0.36 Ω at × 3.
- **The record adds** the far-×3 deck's preflight, with a guard that it is
  served.
- **Otherwise unchanged.** PC5's prediction and metric stand: max over the
  entries of \|Z_L2p − Z_L2\| ≤ 1e-2 × max over the entries of
  \|Z_far×3 − Z_L2\|.

### PC5 re-run under Amendment 4 [2026-09-14]: a guard missed, so it is not read

- **The run.** `u9-c-pc5.service` on src 96dd5be with records 3902752,
  17:10:36–17:12:20Z. Records: `c_pc5.json`, `c_pc5.log`, `run_c_pc5.out`.
- **Guards.**
  - **The far-×3 deck is served,** with θ_min = 0.0242°, unchanged.
  - **The low band was read on all three solves.**
  - **L2p's Δθ was applied:** the grid read 0.008333°.
  - **`lattice_delta_not_bit_zero` MISSED.** Z under L2p is bitwise equal to
    Z under L2.
- **So PC5 is not read.**
  - **What was measured anyway.** The far-mesh step is 0.318 Ω on Z11 and
    0.173 Ω on Z12.
  - **Timing.** The L2p solve took 42.4 s against 30.5 s for L2.
- **Why not read.** A lattice change that moves nothing bitwise is either
  unplumbed or unreachable from Z on this deck. D1, below, asks which, before
  anything is concluded.

**D1, a diagnostic registered before it runs** (`d1_low_band_reach.py`). It
solves the 8 m deck three times:
- with the tables as filled;
- with the low band's values × (1 + 1e-6);
- with the mid band's values × (1 + 1e-6), the control that the scaling
  reaches Z at all.

A spy counts the projection's pairs by band.

| id | prediction |
|---|---|
| **PD1a** | The mid-band control moves Z; it is not bitwise equal. |
| **PD1b** | The low-band scaling also moves Z, with a non-zero low-band pair count. |

**Reading D1.**
- **If PD1b misses,** the low band does not reach this deck's Z. Then PC5 is
  structurally unable to fail here, which is also what L2's fill cost buys
  this deck. That is reported.
- **If PD1b hits,** L2 and L2p interpolate these pairs bitwise alike. PC5 is
  then re-read as ratio 0 against the far-mesh step, with that cause stated.

**D1 results [2026-09-14]** (`d1_low_band_reach.json`, run on the src branch
after e761111 registered it). The spy sees one projection call per solve:
7,056 below/below pairs, 244 in the mid band, and **6 in the low band**, the
shallowest at 0.0242°.

| id | result |
|---|---|
| **PD1a** | **HIT.** Mid band × (1 + 1e-6) moves Z by 1.80e-10 Ω (4.1e-12 relative). |
| **PD1b** | **HIT.** Low band × (1 + 1e-6) moves Z by 1.81e-12 Ω (4.1e-14 relative). |

**Reading, as registered (PD1b hit).**
- **The low band does reach this deck's Z,** through 6 pairs.
- **How much.** Z moves by about 1.8e-6 Ω per unit relative change of those
  surfaces.
- **Why L2p read bitwise equal.** S2 puts the interpolation at ≤ 4e-9
  relative, so the L2 → L2p change should be about 1e-14 Ω, below the solve's
  resolution.
- **PC5 is therefore read as registered: ratio 0 against the 0.318 Ω far-mesh
  step, a HIT.**

**How weak that hit is.**
- **Almost unable to fail.** To exceed 1e-2 of the step, the low band would
  have to be wrong by about 3.2e-3 / 1.8e-6 ≈ 1.8e3 relative.
- **So the low band's accuracy rests elsewhere,** on S2's real-grid 3.9e-9, not
  on PC5.
- **Where the low band carries weight.** The LPDA (d): eight nodes 2.9–4.7 m
  apart over 192 radials.

### Banking (b) on the production path (registered before the test runs)

**The test.** `test_g524_8_two_node_eps1_collapse`, marked slow and crossgate,
in `tests/test_crossing_serve_524.py`:
- spellings A and B at d = 8 m, where θ_min is 0.0242°, which L2 serves with
  no patch;
- ε̃ = 1, against the free-space two-wire truth;
- every Z entry within 0.05 Ω, the class bar of g524_5 and g524_6.

| id | prediction (blind) |
|---|---|
| **PB6** | Both spellings pass, and the worst entry is ≤ 1e-3 Ω. The point estimate is 2.0e-4 Ω, what the (b) probe read at 1 m and at 12 m. |

**PB6 result [2026-09-14]: HIT.** On src 4d8241b the worst entry is 2.024e-4 Ω
for spelling A and for spelling B, as the probe read. Each solve takes about
0.3 s at ε̃ = 1.

The same commit carries two further changes:
- **`test_g524_2_a_two_node_deck_meets_the_grazing_floor_by_name`,** a PR-lane
  check with no fill. The serve plan serves the pair at 8 m and refuses it by
  name at 12 m.
- **The floor test's split.** `test_the_floor_itself_is_untouched_by_the_refusal`
  had run 24.1 s against the 20 s ceiling. Its worst SPEC point stays in the PR
  lane (4.2 s call), and the 18-point sweep moves to the slow lane as
  `test_the_floor_is_served_on_every_spec_soil`.

### G5, the slow lane [2026-09-14]: one MISS, and a STOP by GT3's rule

- **The run.** `u9-g5.service` on src 4d8241b, a clean tree, 17:16:21–17:42:00Z.
- **The result.** **346 passed, 1 failed** (`g5_slow.log`).
- **The failure.** `tests/test_buried_serve_553.py::test_gu5_6_a_grazing_buried_pair_refuses_with_the_lateral_wave`
  expects a grazing-floor refusal and got a served solve. The same test passes
  on main (the u9 tree, whose src is main): 1 passed in 0.63 s.
- **Cause.**
  - **Geometry.** The test's pair is a 5 m radial 1 mm deep, θ =
    atan(0.002/5) = 0.0229°. That was under the 0.05° floor, and is above
    L2's 0.016667°.
  - **Why the search missed it.** The file pins the angle through geometry and
    never names `_SOMM_BELOW_TH_MIN_DEG`, `_MAX_TAIL_PANELS` or the low band.
    GT3's search for re-pin sites was by those names, so it did not find this
    test.
- **By GT3's rule this is a MISS and a STOP.** A failing test outside the files
  that name the floor, the budget, the low band or the second node.
  - **What stops.** No src edit, re-pin included, until this has been reported
    and acknowledged.
  - **What continues.** Crossgate, which was already running, and the
    fill-cost timing. Neither touches this test, and a test-only re-pin would
    not change any timed code path.
  - **The re-pin that will be proposed.** Depth 0.5 mm, θ = 0.0115°, under L2's
    floor. The docstring's "1 deg floor" is stale and gets corrected with it.

**Acknowledged by Laptop-builder, with a condition (2026-09-14).** It stays
recorded as a GT3 miss: the search went by constant names and missed a
geometry pin. It is not re-registered.
- **The condition.** The proposed 0.5 mm depth is wrong: with the test's 1 mm
  radius it would put the wire's surface through z = 0. Depth must stay ≥
  radius.
- **What was done instead.** The radial is lengthened from 5 m to 10 m, at the
  same 1 mm depth, radius and 0.5 m segments. θ = atan(0.002/10) = 0.0115°.
- **Checked before committing.**
  - **Old geometry:** main refuses at 0.02307° under 0.05; the branch serves it.
  - **New geometry:** main refuses at 0.01150° under 0.05, and the branch
    refuses at 0.01150° under 0.016667.
  - **The refusal is the right one.** On both trees it contains
    "grazing floor" and the lateral-wave sentence, not a stand-off or
    crossing refusal.
  - **The test passes on both trees:** 0.48 s and 0.43 s.
- **Where it lives.** The src branch, as its third commit.
- **The whole-tree sweep before the timing is read** has two halves:
  - **Text.** A grep for grazing-refusal wording found no other below/below
    grazing pin outside files that already ran on the branch.
  - **Behaviour.** All five of momwire's pytest lanes run on the branch, and
    this second half is the one that decides. After G5's slow and crossgate,
    `run_lanes.sh` runs the default, integration and memgate lanes alone on
    the box. The timing is read only after all five are in.

### G5, crossgate [2026-09-14]: HIT

- **The run.** `u9-g5.service` on src 4d8241b, 17:42:00–17:48:29Z.
- **The result.** **38 passed, 0 failed** (`g5_crossgate.log`, `run_g5.out`).
- **G5 as registered** ("pass, apart from failures that reproduce on
  origin/main"):
  - **Slow lane: MISS.** It had one failure that does not reproduce on main,
    gu5_6, now re-pinned at 3331ef1.
  - **Crossgate: HIT.**

The default, integration and memgate lanes started on src 3331ef1 at
17:48:45Z (`run_lanes.sh`, `u9-lanes.service`).

## Amendment 5 (2026-09-14): fill-cost timing, registered before any timing run

**Steve's instruction (relayed by Laptop-builder).**
- Time main 1ca8725 against the src branch on the same box, warm and cold.
- The decks:
  - the shipped buried designs that reach the low band;
  - #935's 3 mm dipole;
  - one 48-radial screen from the #983 harness.
- Report the table. Steve decides whether a fifth band, so that only decks under
  0.05° pay, becomes a merge requirement.
- The src PR is not opened until this is in.

**Step 0, geometry only** (`f0_timing_preflight.py`, `f1_knob_sweep.py`), on
the kwargs antennaknobs' momwire engine builds at soil A. antennaknobs'
`below_reach_refusal` is stubbed, for construction only.

**At defaults, no shipped catalog design reaches the low band:**

| deck | θ_min |
|---|---|
| `specialty.buried_dipole` | 2.909° |
| `verticals.buried_radial_vertical` (default and bundle) | 1.358° |
| `verticals.elevated_buried_counterpoise` | 1.358° |
| the 48-radial screen (`buried_radial_vertical`, `n_radials = 48`) | 1.358° |

- **The `detached` variant** is refused (contact plus buried).
- **#935's 3 mm dipole** reaches the low band, at 0.0583°.

**Within the app's knob ranges, nothing reaches it either.** Depth bottoms out at
0.05 m, radial_factor tops out at 1.5, and length_factor at 1.2. The corner that
minimizes θ, depth 0.05 with radial 1.5 and length 1.2, reads 0.1508°.

**Off the knobs:** depth 0.02 with radial_factor 1.5 reads **0.0724°**, which
both trees serve, and depth 0.02 alone reads 0.181°.

**The deck list** (`t2_fill_cost.py`, run by `run_t2.sh`):

| deck | why it is on the list |
|---|---|
| `buried_dipole`, `brv_default`, `ebc_default` | the shipped buried designs at defaults |
| `brv48` | the #983 harness's 48-radial screen |
| `dipole935` | the one deck Steve named that reaches the low band |
| `brv_corner` | depth 0.02, radial_factor 1.5: the smallest catalog-built change that reaches the low band on both trees, standing in for imported and API decks |

**Protocol.**
- **One fresh process per (tree, deck, repeat).**
  - *Cold* is the first solve in that process.
  - *Warm* is a second solve with the engine's result cache cleared, reusing
    the grids.
  - The catalog decks go through the #983 harness's `MomwireEngine`
    BSplineSolver(degree=2) spelling.
- **Order.** The two trees alternate within every deck, the order flips each
  repeat, and there are 3 repeats.
- **Reported:** per-tree medians, the branch/main ratio, max RSS, and whether
  the low band was filled.
- **The box.** It runs alone: after G5, with nothing else running.

| deck | cold ratio (branch/main) | warm ratio |
|---|---|---|
| `buried_dipole`, `brv_default`, `ebc_default`, `brv48` | **1.00 ± 0.05.** The low band is never filled, and nothing else on their path changed. | 1.00 ± 0.05 |
| `dipole935` | **2.9 ± 0.3. NOT blind:** G1 already timed this deck at 9.8 → 28.0 s. | 1.00 ± 0.05 |
| `brv_corner` | **[1.5, 3.5], blind.** The low band fills in three R1 zones out to 3.19 λ_m, but its share of the cold solve is unmeasured. | 1.00 ± 0.05 |

## Amendment 3 (2026-09-14): (d) on the LPDA, registered before any (d) run

### Steve's answers (relayed by Laptop-builder)

- **The comparison is (i).**
  - Feed Z through the refine ladder, on the full LPDA and on a one-node
    spelling of it, in momwire and in NEC-5.
  - Each engine's CHANGE is gated, not absolute Z.
  - Fill time and peak memory come first.
- **(ii), element ports,** runs only if (i)'s change comes out below the
  ladder's own resolution.
- **A refused rung** is recorded by name, and the rungs that serve are used.
- **AK#1464's preflight** is stubbed in the probe only, and named here.

### Named stub

`momwire.bspline.below_reach_refusal = lambda *a, **k: None` is set before
antennaknobs is imported, in `d0_lpda_decks.py` and `d2_lpda_solve.py`.
- **What it stands in for.** antennaknobs' vertex preflight, which pairs z = 0
  vertices on different crossing nodes at θ = 0 and would refuse the deck
  outright.
- **What still judges the deck.** momwire's own scope and serve plan, which run
  unstubbed.
- **The real fix,** where the preflight reads the fill's own nodes, is the
  named antennaknobs follow-up after momwire lands.

### The decks, written by `d0_lpda_decks.py`

- **Full.** The corpus deck as translated: 976 GW, 8 crossing nodes, one EX on
  GW 48, 7 crossed 75 Ω TL cards, 3.5 MHz, GN 2 over soil A, GE −1.
- **One-node** (`lpda_one_node.nec`).
  - **What stays.** The fed element's screen, GW 861–976: its rise and 24
    radials. Screens are told apart by tag block, because a radial runs 24 m
    and crosses the neighbouring nodes' x.
  - **What goes.** The other 812 below wires (GW 49–860) are removed, together
    with their LD cards.
  - **What is lifted.** The other seven elements' node-adjacent wires (GW 6,
    12, …, 42) now end at z = +0.0762 instead of 0, so no element ends on the
    plane.
  - **What is unchanged.** The TL network, the EX and every element.
  - **Size.** 164 GW, 417 segments, 1 crossing node.
- **Far × 3** (`lpda_full_far3.nec`, `lpda_one_node_far3.nec`).
  - **What is multiplied.** Every GW count × 3, except each element's
    node-adjacent wire (GW 6 … 48, which also carries the TL network and the EX
    on segment 1) and each screen's rise (GW 49, 165, …, 861).
  - **Why those two are kept.** They set the shallowest below quadrature nodes.

### Rung preflight (geometry only, `d0_lpda_decks.json`)

| deck | rung | segments | crossing nodes | θ_min | serve plan |
|---|---|---|---|---|---|
| full | refine 1 | 2690 | 8 | 0.0239° | served |
| full | refine 3 | 8070 | 8 | 0.0080° | **REFUSED: grazing floor** |
| full | refine 9 | 24210 | 8 | 0.0027° | **REFUSED: grazing floor** |
| full | far × 3 | 8014 | 8 | 0.0239° | served |
| one-node | refine 1 / 3 / 9 | 417 / 1251 / 3753 | 1 | 0.360° / 0.359° / 0.359° | served |
| one-node | far × 3 | 1233 | 1 | 0.359° | served |

**U2's `--refine` ladder serves momwire one rung on the full deck, and
refine 3 and refine 9 are refused by name.** So by the rung rule, (d)'s ladder
is **{refine 1, far × 3}**, on both decks and both engines. Both engines read
the same bytes, and the full and one-node decks are refined the same way.

### Order, alone on the box, after G5 and the fill-cost timing

Tooling is `d2_lpda_solve.py`: one fresh process per (deck, rung, engine),
recording feed Z, wall time and max RSS.

1. **Cost first.**
   - **The runs.** momwire and NEC-5, each on the one-node deck at refine 1
     and on the full deck at refine 1.
   - **The rule for momwire's full far × 3** (8014 segments). Project its cost
     from the full refine-1 run: peak RSS × (8014/2690)², and time ×
     (8014/2690)³ as the upper bound.
   - **Run it only if** the projected peak is ≤ 20 GB (under the 24 GB cap)
     and the projected time is ≤ 6 h.
   - **Otherwise** report the projection. (d) then has no momwire ladder on
     the full deck, and that is reported rather than served another way.
2. **The remaining rungs.** NEC-5 on both far-× 3 decks, and momwire's
   one-node far × 3.
3. **momwire's full far × 3,** if rule 1 allows it.

### What is gated

- **Per engine e and rung r:**
  - the change Δ_e(r) = Z_full,e(r) − Z_one,e(r);
  - the resolution s_e = \|Z_full,e(far × 3) − Z_full,e(refine 1)\|, the full
    deck's own ladder step.
- **(i)'s gate:** \|Δ_momwire(far × 3) − Δ_nec5(far × 3)\| ≤
  0.1·\|Δ_nec5(far × 3)\| + s_momwire + s_nec5.
- **(ii)'s trigger, fixed now:** run the element-port comparison only if
  \|Δ_e(far × 3)\| ≤ 2·s_e for either engine, that is, if the change is not
  resolved by its own ladder.

| id | prediction (blind) |
|---|---|
| **PDd1** | momwire's full refine-1 solve fits under 24 GB (predicted peak ≤ 4 GB) and takes ≤ 30 min. Its far-× 3 projection passes rule 1. |
| **PDd2** | \|Δ_nec5(far × 3)\| is in [2, 60] Ω. Removing seven of eight radial screens should move this LPDA's feed Z by several ohms or more. |
| **PDd3** | (i)'s gate HITS. |
| **PDd4** | (ii) is not triggered: \|Δ\| > 2·s on both engines. |

### Not predicted, and reported whatever it reads

- **Absolute agreement.** How close momwire's absolute feed Z is to NEC-5's.
  The census had momwire refused on this deck, so there is no prior.
- **The phased arrays.** They stay refused by name, and are not U9's to serve.
  Their three walls are recorded above: grazing past the table, U5's
  within-side spread on both sides, and NT-fed ports.
