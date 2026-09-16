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
