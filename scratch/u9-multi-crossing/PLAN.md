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
