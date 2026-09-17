# momwire#1029 phase 1: where this branch stands

Written 2026-09-17 on Steve's wind-down instruction. Supersedes the 2026-09-16
stopping point, which is in history at 9b1e051 and e3bb8aa.

**The `rows=` parameter has landed and is gated by S-0, S-A, S-B and — as of
2026-09-17 — S-D.** No route code exists; no G gate has run.

## 2026-09-17 amendment: S-D is GREEN, the red was this box

Re-run on the laptop from a FRESH worktree at 0d90313 with the current
accelerators linked in (the branch changes no C++, so main's `.so` files are
the right ones):

- `python -c "import momwire._below_interface"` — imports; the constant
  `MIN_CROSSING_NODE_SEPARATION_M` reads as 1.0.
- `make test`'s default lane (`pytest tests/`, xdist, `not slow and not memgate
  and not integration`): **5255 passed, 5 skipped, 4 xfailed in 423 s**. The
  files the red run named first (`test_crossing_serve_524`,
  `test_plan_extents_914`) pass.

So the 119 import-time `AttributeError`s were diagnostic 4 below — a stale
`.so` / `egg-info` in this box's shared build lineage — not the patch. The
S-D row in the gate table reads PASS from here on; the "What is left" list
starts at its step 2.

## The branch

`feat/1029-rotational-symmetry`, based on momwire 227491d, with main a6a67f93a
merged in at cf3aaab.

| commit | what |
|---|---|
| 97e0c13 | phase 0's registration |
| 8d42ea7 | phase 0's records |
| 9b1e051, e3bb8aa | the earlier stopping point and the baseline notes |
| cf3aaab | merge of momwire main a6a67f93a; accelerators rebuilt |
| c38790b | **phase 1's registration**, before any measurement |
| d85ee24 | **gate S-0 passes** |
| dcdd8a7 | Amendment 1, the three `rows=` contract details |
| 9d51d53 | **the `rows=` patch, with S-A and S-B records and Amendment 2** |

The antennaknobs submodule pointer has not been moved.

## What landed

`rows=None` is exactly today's path; a subset computes only those rows of Z. The
surface is what §1 registered, and nothing else:

- one optional argument on `_below_interface.compute_Z_operator_buried`;
- one optional observer subset on each of `bspline._build_J_blocks_subset` and
  `bspline._accumulate_Z_subset_chunked`;
- **no change** to `_assemble_Z` or `_image_Z_weighted` (zero observer rows in the
  moment tensor give zero Z rows), and none to `_field_galerkin_block`, which
  already took `obs_idx`;
- the plan is resolved before `rows` is read, so grids and quadrature orders stay
  computed from the full observer sets.

## Gate table

| gate | result |
|---|---|
| **S-0** | **PASS** at 4 / 12 / 48 radials. No shared dofs (255 / 695 / 2675 = N×55 + 35); the sector map is a bijection on the dof set; every dof matches its image's kind, local index, end position and junction; the 35 axial dofs are fixed points; the KCL row is invariant with max change exactly 0.0; geometry rotates onto itself by 3.9e-16 / 4.4e-15 / 6.4e-15 m against a 1.06e-8 m tolerance |
| **S-A** | **PASS, bit-identical** against a pre-change build at dcdd8a7. `sha_Z` fbba2249… and a818422f…, `sha_coeffs` 87520f4e… and c82a075d…, Z_in equal to the hex at 4 and 12 radials. The dumps differ only in tree path and timing |
| **S-B** | **PASS on four assertions together** (the registered wording alone would pass vacuously): requested rows equal the full fill (0.0); non-requested rows equal a `rows=[]` control exactly (0.0), carrying only the crossing block and self completions; the fills really narrowed — below-class 222→60 observers at 4 radials, 654→60 at 12, removing 1245.0 from the non-requested rows against 123007.5 of pair-class work on the requested ones; `q_factor` and every plan field equal |
| **S-C** | **half done.** Static half satisfied: only `bspline.py` calls `compute_Z_operator_buried`, nothing anywhere passes `rows=`, and sinusoidal-Galerkin uses `_below_interface` helpers (`field_nodes`, `serve_plan`, `crossing_junctions`) rather than the patched routing. The numeric bit-identity half is **not run** |
| **S-D** | **PASS** (2026-09-17, fresh worktree on the laptop: 5255 passed) — the earlier RED was this box's stale build, see the amendment above |
| **G1–G4** | **not run.** The route does not exist |

## S-D: what is actually known

- `make test` on the patched tree: **23 failed, 4833 passed, 7 skipped, 4 xfailed,
  119 errors**, in 320 s.
- Every error is an import-time `AttributeError: module 'momwire._below_interface'
  has no attribute …`. The first touch is
  `tests/test_crossing_serve_524.py:371` reading
  `_below_interface.MIN_CROSSING_NODE_SEPARATION_M`.
- **That constant EXISTS**, at `_below_interface.py:190`. So this is a module that
  fails to initialise and then reports as missing whatever attribute a test
  touches first — not a genuinely absent name. The 119 errors are one failure
  cascading.
- The same symptom appears on the **pre-change** worktree at dcdd8a7. **That is
  not conclusive and does not clear this patch**: both worktrees share one venv
  and one build lineage, and a stale `.so` or a stale `src/momwire.egg-info`
  would produce exactly this symptom in both.
- Ignoring the first five failing files does not help: the breakage reaches
  `test_plan_extents_914`, `test_pair_order_ladder_906`,
  `test_near_interface_columns_accel_899`, `test_razor_crossing_axis_813` and
  more, which is the shape of one module-level failure, not of five unrelated
  files.

### The diagnostics that would settle it, none of them run

1. `python -c "import momwire._below_interface"` under each tree, for the **real**
   traceback rather than pytest's second-order `AttributeError`.
2. Print `momwire.__file__` as the test run resolves it — `make test` may be
   importing the installed editable package rather than this worktree's `src`.
3. A pristine worktree at momwire main a6a67f93a with its own build: the only
   control that does not share this patch's lineage.
4. Check for a stale `src/momwire.egg-info` (the known trap in this repo) and
   rebuild.

**Until 1–4 are done, treat S-D as red and this patch as unproven against the
suite.** S-A's bit-identity and S-B's four assertions stand on their own
measurements and are unaffected by whatever S-D turns out to be.

## Registered amendments carried forward

- **Amendment 1** (dcdd8a7): the crossing block is filled in full by contract,
  because the routing uses it as `Z -= t; Z -= t.T`; narrowing the field-form
  observers is a positional gather at the plan's own order; S-B compares
  `q_factor` and every plan field.
- **Amendment 2** (9d51d53): the crossing family does **not** shrink under the
  restriction, so the predicted band at 150 radials moves from 5.54–21.0 s to
  about **14–30 s** against G3b's 30.14 s bar. **G3b is marginal at the high
  bound**, and that family is where to look first if the bar is missed.
  Compaction would not help it: the cost is the transpose's columns.

## What is left, in order

1. ~~Settle S-D~~ — done 2026-09-17, green; nothing to fix on 9d51d53.
2. S-C's numeric half: an SG buried deck, bit-identical against dcdd8a7.
3. The route: opt-in `rotational_symmetry=True`, the symmetry check with the six
   by-name refusals (ports and lumped loads included), the sector fill, the
   harmonic-0 solve.
4. G1 (4/12/48, Z, currents, far field, plus the entry-by-entry check at 48), G2's
   six refusals, G3's ladder with RSS, G4 bit-identity plus `make test`.
5. PR to momwire. Steve merges. No issue comments.

**Afterwards, not started:** momwire#570 phase 3's contract derivation, registered
in `scratch/570-far-field/PLAN.md` on a branch, no issue comment.
