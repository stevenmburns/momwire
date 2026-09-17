# momwire#1029 phase 1: the axis-drive route — registration

Registered 2026-09-16, **before any phase-1 measurement**. Phase 0 is complete
(`PLAN.md`, `README.md`); this is the build.

- **Branch** `feat/1029-rotational-symmetry`, at cf3aaab (momwire main a6a67f93a
  merged in, accelerators rebuilt). momwire main has since moved to 03e483d
  (#1081, test-only); rebasing is optional and not done.
- **Dev mode.** No antennaknobs submodule pointer moves.
- **Scope, as Steve narrowed it:** the axis-symmetric drive only — the harmonic-0
  block K₀ — on bs2, opt-in, refusing by name, with the default path
  bit-identical. razor-2p and sinusoidal are out of scope.
- A literal miss goes to Steve for a ruling; it is never re-read silently.
- **Nothing in this arc pins bits** across rotated geometry: rotated copies round
  differently, so every cross-sector comparison is to the assembled tolerance.

## 1. Placement

| piece | where | why |
|---|---|---|
| `rows=None` | `_below_interface.compute_Z_operator_buried` | the shared buried routing, the only place the three pair classes are ordered |
| an optional observer subset | `bspline._build_J_blocks_subset`, `bspline._accumulate_Z_subset_chunked` | the mixed-potential direct and image blocks |
| nothing | `_assemble_Z`, `_image_Z_weighted` | zero observer rows in the moment tensor give zero Z rows |
| nothing | `_field_galerkin_block` | it already takes `obs_idx`; the routing passes a narrowed one |
| the route | a new bspline-side method plus the opt-in kwarg | basis-shaped, so it belongs with the basis |

## 2. The `rows` contract

- **`rows=None` is exactly today's path.** Byte for byte, no new branch taken.
- **`rows=<segment index array>`** computes only those observer rows. **Z keeps
  its full size and every other row stays zero.** It is not a compacted block.
- **Plan invariance.** `rows` must not reach `plan_buried`'s decisions: the
  Sommerfeld grids, the quadrature orders, `pair_extents`, `near_q_factor` — all
  stay computed from the **full** observer sets. An adaptive choice taken from a
  global maximum over observer pairs is exactly where a restricted plan would
  drift, so S-B asserts the plan's whole output is identical, not merely the Z
  rows.
- **The caller owns the consequences.** A matrix filled with `rows` is not a
  solvable operator on its own; anything downstream that assumes a full matrix
  (a dense solve, a norm, a preconditioner) is the caller's responsibility.
- **A registered consequence (Laptop-builder's point, recorded rather than
  discovered):** because Z stays full size, the route's **RSS is the dense RSS**.
  Phase 1 buys time, not memory. G3's RSS column will show no win, and the
  150-radial rung is as memory-bound as dense. Compaction is a separate, later,
  gated step.

### Amendment 1 (2026-09-16, before any `rows` code): three contract details

Read off the fill at cf3aaab, while planning the patch and before writing it.

1. **The crossing block is not restricted, and that is part of the contract.**
   `_crossing_fill.cross_complete_block_split` returns a matrix the routing uses
   as `Z -= t_ab; Z -= t_ab.T`, and `self_completions` builds `(n_basis,
   n_basis)` directly. A row restriction there would also have to drop the
   columns the transpose reads, so the crossing term stays **full-size and
   exact on every row**. It is about 2.8 % of warm self time at 150 radials, so
   the bounded cost is worth the untouched correctness.
   **S-B must therefore expect rows OUTSIDE the requested set to be non-zero
   wherever the crossing block writes them**, and compares only the requested
   rows. "Extra rows filled" is the contract, not a failure.
2. **Narrowing the field-form observers is a positional gather at the plan's own
   order.** `field_nodes` returns nodes as `(n_seg·q, 3)` and tangents as
   `np.repeat(tangents, q)`, with `W[p, i, q]` indexed by segment position. A
   narrowed call passes the kept positions' `obs[i*q:(i+1)*q]`, the same rows of
   the tangents, and `W[:, positions, :]` — never a re-derived `q`.
   `_field_galerkin_block` infers `q` from the arrays' shapes, so the order must
   stay the one the full plan chose.
3. **What S-B compares to prove plan invariance.** `plan_buried` derives
   `q_factor` from `near_q_factor(cross_pair_separation(seg_l, seg_r, a_idx,
   b_idx))` and the extents from `serve_plan_fn` over the FULL index sets and the
   full node stacks. S-B asserts equality of `q_factor` and of every plan field —
   `r1_below`, `r1_above`, `r_cross_max`, `r_cross_min`, `zp_min`, `zp_max` —
   between `rows=None` and `rows=subset`, in addition to the row-by-row Z
   comparison.

### Amendment 2 (2026-09-16, after S-A and S-B, before G3): what the crossing family costs

**S-B as actually gated.** The registered wording ("the restricted fill equals the
corresponding rows of the full fill") passes trivially if the restriction never
narrows anything, so the gate asserts four things together, and all four hold at
4 and 12 radials:

1. the requested rows equal the full fill entry by entry — measured 0.0;
2. the non-requested rows equal a `rows=[]` control **exactly** — measured 0.0,
   i.e. they carry only the crossing block and the self completions, which
   Amendment 1 registers as filled in full;
3. the fills really narrowed — the call log shows the below-class direct and
   image fills at 60 observers against 222 sources (654 at 12 radials), with
   1245.0 of content removed from the non-requested rows and 123007.5 of
   pair-class work on the requested ones;
4. `q_factor` and every plan field are equal between `rows=None` and the subset.

**The consequence for G3, registered before the ladder runs.** The crossing block
and the self completions do **not** shrink with the restriction — by contract,
since the routing reads their transpose. The cost model in §7 of `PLAN.md`
classified that family as pair-scaling, which was wrong. Re-reading the fea1ad0
profile at 150 radials, the terms that stay full-cost are about **8.7 s**:

| term | s at 150 radials |
|---|---:|
| `_sandwich_dense` | 4.38 |
| `_row_weights` | 2.23 |
| `_real_matvec_c` | 0.73 |
| `_g_of_r` | 0.54 |
| `axis_data` | 0.18 |
| `_rank1_add_cols` | 0.15 |
| `_bnd_and_corner` | 0.23 |
| `cross_complete_block_split` | 0.12 |
| `self_completions` | 0.10 |
| `_main_split` | 0.06 |

So the predicted sector-route band at 150 radials moves from **5.54 – 21.0 s** to
about **14 – 30 s**, against G3b's bar of 30.14 s. **G3b is therefore marginal at
the high bound rather than comfortable**, and the prediction stays "hit" with that
stated. If the bar is missed, this family is where to look first — and compaction
(the deferred step from Amendment 1's item 1) does nothing for it either, because
the cost is the transpose's columns and not the rows.

## 3. The symmetry rule, as the check will implement it

Every condition is a geometry, material or drive fact known at construction.

1. **An axis:** a vertical line through which every wire of the axial group
   passes, within `tol` = 1e-9 × the deck's largest extent.
2. **N ≥ 2 sectors** whose wires map onto the next sector under rotation by
   2π/N about that axis, vertex for vertex, in the same order, within `tol`.
3. **Identical discretisation and material per sector:** the same per-edge
   segment counts and grading, radius, conductivity and jacket.
4. **Every port on the axis** — not only the feed. A multiport solve drives each
   port in turn, and an off-axis port breaks the symmetry for that drive.
5. **Every lumped load** either on the axis or identical in every sector.
6. **A ground invariant under rotation about the axis:** free space, PEC,
   reflection-coefficient, or a Sommerfeld half-space with the axis normal to the
   interface. Terrain, two media, and a tilted axis do not qualify.

### The six refusal sentences (G2), by name

Each names the first failing condition and the way out. Drafts, to be pinned by
the tests:

1. `rotational symmetry: sector 2's wire is 1.00 % longer than sector 0's (6.397 m against 6.334 m). Every sector must map onto the next under rotation by 2*pi/N. Drop rotational_symmetry=True to solve this deck densely.`
2. `rotational symmetry: sector 3 sits at 272.500 deg, where 4 sectors require 270.000 deg (tolerance 1e-09 x 10.6 m). ...`
3. `rotational symmetry: port 'feed' sits at (6.334, 0.000, 0.000), off the symmetry axis (0.000, 0.000). An off-axis drive excites every harmonic, and this route serves the axis-symmetric drive only. ...`
4. `rotational symmetry: sector 1's conductor radius is 0.800 mm against sector 0's 0.500 mm. Every sector must carry the same radius, conductivity and jacket. ...`
5. `rotational symmetry: the ground model 'terrain' is not invariant under rotation about the axis. Free space, PEC, a reflection-coefficient ground and a Sommerfeld half-space qualify. ...`
6. `rotational symmetry: the axial group's wires are not parallel to z (wire 5 runs 2.300 deg off). The symmetry axis must be normal to the interface. ...`

## 4. The route

1. Check §3; refuse by name on the first failure.
2. Group the unknowns: sector s, and the axial group M.
3. Fill once with `rows` = sector 0's segments + the axial segments.
4. Build K₀ = [[Λ₀, √N·B], [√N·C_M, Z_MM]] with Λ₀ = Σ_d C_d from sector 0's
   rows, B = Z[sector 0, M], C_M = Z[M, sectors] summed over sectors, Z_MM from
   the axial rows.
   - **Use the sum of the N axis-against-sector copies,** and assert they agree
     to the assembled tolerance — a free consistency check on the sector
     assignment (they will not be bit-identical).
5. Solve K₀ with the KCL rows closed by the same Schur step `_solve_with_kcl`
   uses; reassemble the full-ordering coefficient vector.
6. **The far field needs no new code.** `BSplineSolver` inherits
   `_ElementCurrents.element_currents(coeffs, subdiv=1)`
   (`_element_currents.py:33`, mixed in at `bspline.py:677`), which wants a 1-D
   solution vector in the family's full basis ordering. Both the route's
   reassembled coefficients and the dense solve's go through that call and then
   `_far_readout._far_moments(mid, moment, k, theta, phi, ground, ground_z,
   freq_hz)` (`_far_readout.py:431`), which since U8 splits elements at the
   interface itself.

## 5. Gates on the parameter, run BEFORE the route is wired up

| id | what | bar |
|---|---|---|
| **S-0** | structural, before any fill: rotating the mesh by 2π/N maps the dof set onto itself dof by dof, including the hub's junction and jump dofs (#138), and every dof belongs to exactly one sector | an exact bijection; no dof's support straddles two sectors |
| **S-A** | `rows=None` is bit-identical on non-symmetric buried decks: the default `buried_radial_vertical` convention and a multi-node crossing deck | Z, Z_in and currents identical to the pre-change tree, bit for bit |
| **S-B** | the restricted fill equals the corresponding rows of the full fill, **and** the plan is untouched | rows agree to the assembled tolerance; `plan_buried`'s entire output is identical between `rows=None` and the subset |
| **S-C** | sinusoidal-Galerkin's buried path is bit-identical and never passes `rows` | bit-identical |
| **S-D** | `make test` green plus momwire CI | green |

## 6. Gates on the route, with predictions

At 4 / 12 / 48 radials on `verticals.buried_radial_vertical`, soil 13/0.005, bs2
degree 2, against the dense solve on the same deck.

| id | what | bar | prediction |
|---|---|---|---|
| **G1a** | Z_in through the route | \|ΔZ_in\|/\|Z_in\| ≤ 1e-9 | hit |
| **G1b** | currents | max \|Δc\| / max \|c\| ≤ 1e-8 | hit |
| **G1c** | far field, through `element_currents` → `_far_moments` on one fixed grid | max relative difference on \|m_θ\| and \|m_φ\| ≤ 1e-8 over the grid | hit |
| **G1d** | the entry-by-entry check at 48 radials (phase 0 measured 1.7e-9 worst at 12) | worst per-entry non-invariance ≤ 1e-7, graded against each entry's own magnitude above a 1e-9 × max floor | hit |
| **G2** | the six refusals, each by name | each raises at construction, naming its condition | hit |
| **G3a** | warm seconds at 48 radials | ≤ 3.0 s | hit |
| **G3b** | warm seconds at 150 radials | ≤ 30.14 s (3b75639's column) and ≥ 5× faster than dense | hit |
| **G3c** | **RSS is the dense RSS** — registered, not discovered | peak RSS within 5 % of dense at every rung; no memory win in phase 1 | hit |
| **G4** | bit-identity on non-symmetric decks and on the default path, plus `make test` | bit-identical, green | hit |

**Why G1a's bar is 1e-9 when phase 0 measured 2e-13 and 5e-13:** phase 0 permuted
a dense fill; the route fills only some rows and sums rotated copies, so its
roundoff path differs. The bar allows four orders of margin over phase 0 and is
still far below anything physical.

**G3's ladder** runs 12 / 24 / 48 / 96 / 150 radials, warm seconds and peak RSS,
beside dense and beside #1067's NEC-5 columns, under the 24 GB cap.

### Amendment 3 (2026-09-17, after G1 ran): what G1c's bar can measure

**Registered:** G1c, "the far field, through `element_currents` →
`_far_moments` on one fixed grid | max relative difference on \|m_θ\| and
\|m_φ\| ≤ 1e-8 over the grid | hit".

**Measured** on the 19 × 12 upper-hemisphere grid, the route against the dense
solve on the same deck (`g1.jsonl`):

| radials | max \|m_θ\| | max \|m_φ\| dense | max \|m_φ\| route | \|m_φ\|/\|m_θ\| | Δ\|m_θ\| / pattern | Δ\|m_φ\| / pattern | Δ\|m_φ\| / \|m_φ\| itself |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.08548 | 1.126e-5 | 1.126e-5 | 1.3e-4 | 8.7e-14 | 6.3e-15 | **4.8e-11** |
| 12 | 0.11854 | 2.07e-15 | 3.88e-18 | 1.7e-14 | 4.5e-13 | 1.7e-14 | **0.99924** |
| 48 | 0.14478 | 2.31e-15 | 7.37e-18 | 1.6e-14 | 3.4e-13 | 1.6e-14 | **0.99887** |

**The two readings, and which one is a measurement.** Read as "each component
against its OWN largest entry", G1c **hits at 4 radials (4.8e-11) and misses
at 12 and 48**. Read as "over the grid", i.e. both components against the
pattern's own scale, it hits everywhere by four orders. The miss is REPORTED
here rather than re-read silently, and this is why the second reading is the
one that measures anything:

- \|m_φ\| is not small by accident. An N-fold screen's far field carries the
  azimuthal harmonics 0, ±N, ±2N… and the φ component is the ±N one. It comes
  from the DISCRETE radial positions, not from the currents — which are
  identical on every radial on both routes. At N = 4 that harmonic is a real
  quantity, 1.3e-4 of the pattern, and the route reproduces it to 4.8e-11. At
  N = 12 and 48 it is 1.7e-14 of the pattern: the roundoff floor.
- So at 12 and 48 the own-scale ratio divides roundoff by roundoff. It reads
  0.999 because the ROUTE's \|m_φ\| is three orders *smaller* than dense's
  (3.9e-18 against 2.1e-15) — the route makes every sector's current exactly
  equal by construction, where the dense solve leaves N independent roundoff
  paths that do not cancel. The route's pattern is the cleaner of the two, and
  a bar that calls that a failure is measuring its own floor.

**Amended:** G1c grades Δ\|m_θ\| and Δ\|m_φ\| against max \|m_θ\| over the
grid — one scale, the pattern's — and `g1.jsonl` records both readings and the
magnitudes behind them at every rung, so the choice is auditable rather than
asserted. Under the amended reading G1c is a hit at 4, 12 and 48 radials.

### Amendment 4 (2026-09-17, while writing the check): refusal 5 has no deck

§3's sixth condition and its sentence 5 are about a ground that is not
invariant under rotation about the axis — "terrain, two media, a tilted
axis". **`BSplineSolver` cannot express one.** Its whole ground surface is
`ground_z` + `ground_eps` + `ground_model ∈ {refl-coef, sommerfeld}`, and a
horizontal half-space is axisymmetric about any vertical line; the tilted-axis
case is condition 1's, with its own sentence. The far-field cliff (`RP 2`/`RP
3`, momwire#570) is a READOUT choice, not a ground the fill sees, and it
arrives long after construction.

So sentence 5 is reachable only through a seam, and the check has one on
purpose: `BSplineSolver._rotational_ground_kind()` is the single place the
ground's name is decided, the whitelist
(`_rotational_symmetry.AXISYMMETRIC_GROUNDS`) is what it is checked against,
and a family that grows a terrain ground refuses the route by answering with a
name the whitelist does not carry. G2's fifth deck is a one-line subclass that
answers `"terrain"`, and it **does** raise at construction like the other
five. A companion test asserts the other half — that every ground this family
CAN express returns a qualifying name — so the whitelist cannot quietly start
refusing served grounds.

Recorded because the registration read as though six decks existed; five do,
and the sixth is a solver rather than a deck.

### Amendment 5 (2026-09-17, after G3 ran): G3a misses on this box, and G3b has no box

The ladder as measured, on the laptop (xps13, 4c/8t), `OMP_NUM_THREADS=4`,
each rung in its own process under `prlimit --as=8G` (`g3.jsonl`):

| radials | unknowns | dense warm s | route warm s | speedup | dense fill / solve | route fill / solve | dense RSS MB | route RSS MB | RSS ratio | ΔZ_in |
|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| 12 | 695 | 1.552 | 0.702 | 2.21× | 1.531 / 0.021 | 0.700 / 0.0014 | 261.6 | 256.9 | 0.982 | 4.8e-13 |
| 24 | 1355 | 4.744 | 1.357 | 3.50× | 4.605 / 0.139 | 1.355 / 0.0022 | 482.0 | 474.9 | 0.985 | 6.2e-13 |
| 48 | 2675 | 18.566 | **3.327** | 5.58× | 18.006 / 0.560 | 3.323 / 0.0043 | 1051.2 | 1047.6 | 0.997 | 9.1e-13 |
| 96 | 5315 | 77.112 | 10.776 | 7.16× | 73.696 / 3.416 | 10.765 / 0.0115 | 3597.1 | 3564.7 | 0.991 | 1.9e-12 |
| 150 | 8285 | — | — | — | — | — | — | — | — | — |

**G3c: HIT.** Peak RSS is the dense RSS at every rung — 0.982, 0.985, 0.997,
0.991 of it, all inside the registered 5 %. Registered, not discovered, and
now measured.

**G3a: MISS.** 3.327 s at 48 radials against a 3.0 s bar. Reported, not
re-read. The context, which is not a rescue: the bar came from #1067's
Skylake column, and this box is **1.215× slower** on the same deck (dense
18.566 s here against #1067's momwire warm 15.28 s). Divided by that factor
the route reads 2.74 s, inside the bar and inside the cost model's own
1.18–2.54 s prediction at its high end. **The bar is missed on the box it ran
on**, and whether it is missed on #1067's box is unmeasured.

**G3b: NOT MEASURABLE HERE.** The 150-radial rung raises on **both** modes,
at the same line:

    _crossing_fill._row_weights  <- _sandwich_dense <- _main_split
                                 <- cross_complete_block_split
    ArrayMemoryError: Unable to allocate 124. MiB, shape (2035, 7992) float64

at about 5.3 GB of RSS, under the 8 GB address-space cap this box is held to.
Two attempts: the registered cold-then-warm pair (the cold pass completed in
about 195 s dense, the warm one raised), and a single-pass escape added for
it (raised in that one pass). No third attempt. The registered ladder ran
under a **24 GB** cap in phase 0, which is what it needs.

**That failure is Amendment 2's finding, sharpened.** The crossing block is
filled in FULL by contract — the routing reads its transpose — so `rows=`
narrows nothing there, and the route hits the wall at the identical
allocation. The crossing family is not only the part of the time that does
not shrink; it is the memory FLOOR, and it is the same reason G3c reads 0.99.
Compaction does not move it either, for the reason Amendment 2 gives.

**What is therefore unmeasured:** whether the route's warm 150-radial second
is under 30.14 s. The route's fill is linear in N once the sector is fixed
(one sector's rows against N sources), and 10.776 s at 96 extrapolates to
about **16.8 s at 150** — inside Amendment 2's registered 14–30 s band. That
is an EXTRAPOLATION from four measured rungs, not a measurement, and G3b
stays open until a box with the registered cap runs it.

**What IS measured about the speedup claim:** "≥ 5× faster than dense" holds
from 48 radials up (5.58× and 7.16×), and the ratio is still climbing.

## Order

1. Commit this registration; push.
2. S-0, then the `rows` parameter, then S-A to S-D. **Stop and report if the
   parameter needs more than the two bspline-side subsets and the one shared
   argument.**
3. The route, then G1, G2, G3, G4 and `make test`.
4. PR to momwire. Steve merges. No issue comments.

## Afterwards (not this arc)

momwire#570 phase 3's first unit, the direct+image readout contract for the
below/below case, is Steve's next assignment for this box. It is a derivation
with an independent oracle and **no reader code**, registered in momwire
`scratch/570-far-field/PLAN.md` on a branch — not on the issue (Laptop-builder
corrected that; whether anything reaches the issue is Steve's call).
