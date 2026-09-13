# U5 src plan: serve a two-radius crossing node (registered before any src change)

Registered 2026-09-13, before any `src/` edit. The src branch is
`u5-two-radius-crossing`, from momwire `origin/main` at `42112b9`. The PR goes up
unmerged, for Steve.

The physics and the measurements are in DERIVATION-MIXED-RADIUS.md §8–§9 and
MEASUREMENTS.md, step (b′). This file fixes what the src change does, what it
leaves alone, the tests that pin it, and the checks that must pass before the
PR opens.

## 1. Scope

**Served (BSplineSolver only).** A crossing deck, within today's topology scope
(one above member per crossing junction, N ≥ 1 below members, other junctions
only wholly below and off the plane). Every ABOVE wire has one radius a_A,
every BELOW wire has one radius a_B, and a_A ≠ a_B.

**Still refused, by name:**

- radii that differ WITHIN the above wires or within the below wires, since only
  a two-radius node is measured;
- any mixed-radius crossing deck in `RazorSolver` and
  `SinusoidalGalerkinSolver`, whose fills were not measured under the rule. The
  shared `crossing_junctions` keeps its current refusal unless the caller opts in;
- everything the topology scope already refuses.

**Unchanged:** every deck with one radius, and every mixed-radius deck without a
crossing junction.

## 2. The rule, as implemented

This is the spelling measured by check 3′ and step (b′): `node_B`, plus the
crossing junction's KCL multiplier (VC_keep).

1. **Line tests at the observer wire's radius.** Above-observer rows of the cross
   block use a_A; below-observer rows use a_B. The self completions' column terms
   use the family's own radius.
2. **One node potential.** Every point test at the crossing node — in both
   families' node rows — uses a_n = a_B, the buried member's radius. That covers
   the self completion's row term and node corner, the test-side end terms, and
   the cross corner. The choice of a_n is a gauge worth ≤ 7 mΩ (P3′.2).
3. **In code, the cross pair splits into two blocks:**
   - `t_below = cross_complete_block_split(ctx at a_B)`, unchanged: every term is
     at a_B, which is exactly the below rows' spelling.
   - `t_above = main(a_A) + source-end terms(a_A) + test-end terms(a_B) +
     corner(a_B)`.
   - They compose as `Z −= t_above; Z −= t_belowᵀ`.
4. **Self completions.** The below family uses a_B throughout. The above family
   takes its column terms at a_A, and its row terms and corner at a_B.
5. **Grading** (`axis_data`) is from min(a_A, a_B).
6. **KCL.** The crossing junction keeps its KCL multiplier row, as a same-medium
   junction does. The value-1 bases are the ones already kept as directional.
   `_grounded_junctions()` is unchanged, so node gaps and junction ports on a
   crossing junction stay refused. The set of junctions that carry a KCL row
   comes from one helper, which `_build_basis_polynomials` and
   `_split_kcl_ports` both read, so their row counts cannot drift apart.

## 3. Where the code changes

| file | change |
|---|---|
| `_below_interface.crossing_junctions` | new keyword `two_radius=False`. With `False`, behaviour and message are unchanged. With `True`, a two-radius deck passes, and a spread within one side is refused with a message naming the per-side rule |
| `_below_interface.compute_Z_operator_buried` | at the crossing branch, when the context carries two radii, compose `t_above`, `t_below` and the two-radius self completions. Otherwise the existing three lines run untouched |
| `_crossing_fill.CrossingContext` | two trailing fields, `a_above` and `a_below`, defaulting to `None`. Razor's and SG's constructors are unaffected |
| `_crossing_fill._ends_and_corner` | keywords `test_ends=True, source_ends=True`. The defaults run today's loops in today's order |
| `_crossing_fill` | new `cross_complete_blocks_two_radius` and `self_completions_two_radius`. `self_completions` and `cross_complete_block_split` are not edited |
| `bspline.BSplineSolver` | `_crossing_junctions` passes `two_radius=True`. A new `_two_radius_crossing()` returns `(a_above, a_below)` or `None`, and is `None` without calling anything when every radius is equal. `_crossing_context` fills the two fields and sets `a_wire = min` only in the two-radius case; the equal-radius expression is kept as written. A new `_kcl_row_junctions()` is read by `_build_basis_polynomials` and `_split_kcl_ports`. The basis cache key gains the two-radius flag |

## 4. Equal radii stay bit-identical (§7 item 1)

- **The fill.** Every change is gated on `_two_radius_crossing()` returning
  radii. At equal radii that helper returns `None` before touching the crossing
  scope, the crossing branch runs its existing three lines, `_ends_and_corner`
  runs with default keywords, the context's `a_wire` is the same expression, and
  the KCL row set is the same.
- **The basis cache.** Its key changes shape but not its values.

## 5. Tests: `tests/test_crossing_two_radius_u5.py`, plus two edits

The deck is check 2's momwire-native rod: 2 m below, 10 m above, fed at 4.33 m,
soil A with Sommerfeld ground, 7 MHz, graded toward the node, refine 1. Radii
are in mm.

| id | test | bar | basis |
|---|---|---|---|
| T1 | equal radii take the shipped path | with the two-radius fill functions monkeypatched to raise, the equal-radius rod solves. `kcl_A` has no row for the crossing junction, and `_two_radius_crossing()` is `None` | §4 |
| T2 | a two-radius node is served with a KCL row | (0.25, 0.125) solves. `kcl_A` has a row for the crossing junction with ±1 on its directional bases. The node's KCL deficit \|I(0⁺) − I(0⁻)\|/\|I(0⁺)\| is ≤ 1e-9 | exact by construction |
| T3 | the Z guard reads the physics | with ΔR_X = Re[Z(X thinned) − Z(control)]: ΔR_rise(ratio 2) ≥ 3 Ω; \|ΔR_top(ratio 0.5)\| ≤ 0.25·ΔR_rise(ratio 2); ΔR_rise(ratio 4)/ΔR_rise(ratio 2) ∈ [1.7, 2.3]. These are ordering and ratios with tolerances, not literals, so they survive re-pins | §8.2 and P4.5. **Informed** by check 4's r = 1 and 2 rows (+7.86 / +15.62 Ω at r = 1). The two-surface rule fails the first bar (−0.13 Ω at r = 2) |
| T4 | a fan's KCL row covers every member | a node fan (1 above, 2 below at one radius a_B ≠ a_A) builds a crossing-junction row with 3 nonzeros and the right signs. Basis only, no solve | §2 item 6 |
| T5 | refusals kept | (a) two below wires of different radii at a crossing deck → `NotImplementedError` naming the per-side rule; (b) `crossing_junctions` with the default flag → the existing "per-wire radii" refusal (`test_below_interface_980` already pins this and stays untouched); (c) a `RazorSolver` mixed-radius crossing deck is still refused | §1 |
| T6 | ports and the Y matrix stay consistent | on the T2 deck, `compute_y_matrix()` returns 1/Z to ≤ 1e-12 relative, so the multiplier reaches every solve path | §2 item 6 |

**Edit:** `test_crossing_serve_524.py::test_g524_2_mixed_radii_refused_by_name`
pins the refusal this PR lifts. It becomes two tests: the two-radius deck is
served (its crossing junctions are returned and the context carries both
radii), and a spread within one side is refused by name.

**Marks:** whatever the tests' measured runtime puts them in (slow if the four
solves in T3 exceed the lane's budget). Whichever lane that is, `make slow` and
`make crossgate` are run locally before the PR, because PR lanes skip both.

## 6. Checks before the PR opens (bars fixed now)

| id | check | bar |
|---|---|---|
| V1 | **src ≡ the measured spelling.** The src branch against check 3′'s VC_keep(`node_B`) rows (`check3k_check2.json`: ratios 2 and 4, r = 1 and 2) and step (b′)'s rows (`bq_*.json`, through antennaknobs' engine with the src tree imported) | \|Z_src − Z_harness\| ≤ 1e-9 relative on every row |
| V2 | **§7 item 1.** The catalog `buried_radial_vertical` through antennaknobs' momwire engine, at its defaults and at two further frequencies, on momwire main (`42112b9`, via the antennaknobs submodule at `1dbd384`, src-identical) against the src branch. Also check 2's rod at equal radii, momwire-native | Z equal bit for bit (`repr`) |
| V3 | **Lanes, locally on the src branch:** `make lint`, `make test`, `make slow`, `make crossgate` | all green. The PR says which ran, with counts |
| V4 | **The existing crossing tests** (`tests/test_crossing_*.py`, `test_buried_*`, `test_sg_crossing_*`, `test_razor_crossing_*`) | pass inside V3; any change to a banked number is a stop |

If V1 misses, the src spelling is not the one measured, so fix it before
anything else. If V2 misses, the equal-radius path was touched, which is a stop.

## 7. Not in this PR

- A slope test: §9.6 keeps the slope out of §7, and check 5 is deferred.
- (c)'s corpus decks, blocked on U3's GE −1 scope.
- Razor and SG two-radius serving.
- A fan whose buried members differ in radius.

## 8. Checks measured before the PR [2026-09-13]

The src branch `u5-two-radius-crossing` is at `adc009c`, local and not yet
pushed.

| id | verdict |
|---|---|
| V1 | **HIT**. Check 3′'s rows: worst 2.9e-13 over 4 rows (`v1_check2.*`). Step (b′)'s mixed rungs, through antennaknobs' engine: worst 1.8e-12 over 27 rows (`v1_b.*`). Both runs asserted that momwire was imported from the src worktree |
| V2 | **HIT**. On the catalog `buried_radial_vertical` at 7.1 / 6.8 / 7.4 MHz, soil A, and check 2's rod at equal radii (r = 1 and 2), main and the src branch agree bit for bit (`repr`). The determinism control also held: main reproduced itself exactly. The crossing fill ran 3 times on the catalog decks and 5 in all, so the comparison is not vacuous. The catalog design at 7.1 MHz reads 78.13206040784942+46.33767699982075j on both (`v2_main.*`, `v2_main_repeat.*`, `v2_src.*`) |
| V3 | lanes running |
| V4 | see V3 |

**T3's measured values** (refine 1): R₀ = 186.6216 Ω. Halving the rise radius
adds +7.861 Ω; halving the top radius adds +0.492 Ω, 0.063 of the rise's
response. The rise's response at ×¼ over its response at ×½ is 1.988. Every
bar holds with margin.

**Runtime.** The two test files (7 new tests plus the edited `test_g524_2`
pair, 44 tests in all) pass in 3.3 s serially. T3 takes 0.82 s, so the tests
are unmarked and run in the PR lane.
