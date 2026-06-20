# Plan: element-aware block-low-rank solver for antenna arrays

Status: **planned** (not started). This document captures the design and the
validated measurements behind it so a future session can pick it up without
re-deriving them.

## Motivation

`HMatrixPySim` (the generic H-matrix / ACA solver, see `hmatrix.md`) compresses
the impedance matrix with a *geometry-blind* binary space-partition cluster
tree. For **arrays of identical elements** that leaves a lot on the table,
because the tree does not know where the element boundaries are. The driving
example `bowtiearray2x4` is the case in point: at its native mesh (~1.5k
unknowns) the generic H-matrix is ~6× *slower* than the dense C++ path, and it
only compresses the matrix to 41%.

An array has obvious block structure the generic tree ignores: an 8-element
array is an 8×8 grid of blocks — strong dense **self-blocks** on the diagonal,
weak **coupling blocks** off it. Exploiting that structure directly should make
the work scale like `8·N²` (eight self-blocks) `+ O(N)` (low-rank coupling)
instead of `(8N)²`.

## Validated findings (measured, free space, 28.57 MHz)

Measured on `bowtiearray2x4` (n=1488, 8 elements ≈ 186 bases each), grouping
bases into 8 elements by spatial k-means and inspecting the dense Z:

- **Coupling is weak:** mean `‖Z_ab‖_F / ‖Z_aa‖_F = 0.02` (max 0.08). Off-
  diagonal element blocks are a few percent of the self-blocks.
- **Coupling is low-rank:** each ~186×186 coupling block has numerical rank
  **~4** at a 1% threshold (max 5).
- **Storage:** 8 dense self-blocks = **13%** of n²; 56 coupling blocks at
  rank 4 ≈ **3%**. So an element-aware decomposition reaches **~16%** vs the
  generic H-matrix's 41%.

Crossover context (from `hmatrix.md`): the generic H-matrix only beats the
dense C++ path above ~2.5–3k unknowns; the driving examples run below that, so
neither the generic H-matrix nor tolerance-loosening helps them at native size.
For animation of these designs the current best lever is mesh coarsening on the
dense path (rhombic ~10×, bowtie ~4× at <2% error). The array-block solver
below is the structural way to actually beat dense for arrays.

## Target scaling

For an array of `P` identical elements with `N` unknowns each (total `PN`):

| quantity | dense | element-block |
|---|---|---|
| storage / matvec | `(PN)² = P²N²` | `P·N²` (self) `+ O(N)` (coupling) |
| factorization | `(PN)³ = P³N³` | `N³` (one shared self-block) |
| solve (per RHS) | `O((PN)²)` back-sub | few × `O(P·N²)` block-Jacobi iters |

`P²N²` → `P·N²` storage (P× less); `P³N³` → `N³` factor (P³× less, via
identical-element reuse). For `P=8` that is ~8× storage and a large solve win.

## Design: `ArrayBlockPySim`

A **new** solver (sibling of `HMatrixPySim`, reusing the same B-spline
geometry/basis/kernels/KCL and the C++ `bspline_assemble_offedge_block`), not a
mode of the generic one — the partition is structural, not geometric.

### 1. Element grouping (the key enabler)

Need a map `basis → element id`. Options, in order of preference:

1. **From the array builder.** `bowtiearray2x4` arrays a single element `P`
   times; the wire/feed lists are `P` contiguous copies. Surface the per-element
   basis ranges from the build (cleanest, exact, gives identical ordering).
2. **Connected components** of the wire graph, if elements are electrically
   separate (no shared junctions across elements).
3. **Spatial k-means** on basis centroids (fallback; boundary bases can be
   mis-assigned — k-means gave element sizes 171–197 for what should be equal
   elements, i.e. ~±7% noise — so prefer 1 or 2).

The grouping must give **consistent intra-element basis ordering** for the
identical-self-block reuse (step 2) to work.

### 2. Self-blocks (with identical-element reuse)

- Self-block `Z_aa` = the element's own dense MoM matrix (has the singular
  near-field; keep dense, or H-compress later if a single element is itself
  large). Assemble via the existing dense bspline path restricted to the
  element's bases.
- **Identical elements ⇒ one self-block for all P** (free-space self-impedance
  is translation-invariant). **Verify this first** (compare two self-blocks in
  consistent ordering; should match to ~1e-12 free space). With a ground plane
  the images differ per element — fall back to per-element self-blocks, or treat
  the image as additional coupling.
- Factor the shared self-block once (LU) for the block-Jacobi preconditioner.

### 3. Coupling blocks

- Off-diagonal `Z_ab` (a≠b): well-separated elements ⇒ smooth kernel ⇒ low rank.
  Compress with the existing `aca_partial` + the C++ off-edge assembler
  (`_offedge_block_evaluators`) — same machinery as the generic far blocks, just
  with element-aligned index sets.
- **Block-Toeplitz reuse (regular grids):** on the 2×4 grid the coupling depends
  only on the element displacement (Δrow, Δcol). Compute only the unique
  displacement blocks (a handful) instead of all `P(P-1)` ordered pairs. Needs
  the element index → grid coordinate map (from the builder).

### 4. Solve: block-Jacobi-preconditioned GMRES

- Matvec: 8 self-block products (dense) + low-rank coupling products — `O(P·N²)`.
- Preconditioner: block-diagonal = the factored self-block(s). Because coupling
  is ~2%, expect **~2–4 GMRES iterations**.
- KCL/junctions: handle as in `HMatrixPySim._solve_hmatrix` (augmented saddle
  system) if elements have internal junctions; cross-element constraints (a
  shared feed network) go in the constraint rows.
- Multi-RHS (the array has P feeds): factor once, reuse across RHS columns (as
  the current solver already does).

## Animation payoff (the original goal)

Arrays are usually animated by **steering/phasing or spacing**, and the
structure makes those cheap by *reusing the factorization across frames* — which
a dense LU of the full `PN` matrix cannot:

- **Phase / excitation sweep:** geometry fixed ⇒ Z fixed ⇒ only the RHS changes.
  Re-solve = cached block-Jacobi back-subs, ~instant per frame.
- **Spacing slider:** self-blocks unchanged; only the low-rank coupling blocks
  recompute (cheap). Factorization reused.
- **Element-shape slider:** refactor one shared self-block (`N³`), recompute
  coupling.

This is a better animation story than mesh-coarsening for arrays, and it
composes with coarsening (coarse mesh *and* block structure).

## Implementation phases

- **P0 — Verify assumptions.** Exact element grouping from the builder;
  confirm identical self-blocks to ~1e-12 (free space); confirm coupling rank
  ~4 and weakness ~2% on `bowtiearray2x4` and at least one other array
  (`invveearray`). Gate the whole effort on these holding.
- **P1 — Block partition + matvec.** Build self + coupling blocks (ACA), an
  `ArrayBlock` container with a fast matvec, validate matvec vs dense `Z@x`.
- **P2 — Block-Jacobi GMRES solve.** Shared self-block factor + GMRES;
  validate impedance/Y vs dense bspline within ~1e-4; measure iterations.
- **P3 — Identical-element + block-Toeplitz reuse.** One self-block, unique
  coupling blocks only; measure fill/factor savings.
- **P4 — Engine integration + animation path.** Register as a selectable
  solver; expose factorization reuse across solves for phase/spacing sweeps;
  scaling study vs dense and vs `HMatrixPySim` on the array designs.
- **P5 (optional) — CBFM / macro-basis.** Reduce each element to K
  characteristic modes → tiny `(P·K)²` reduced system; the right tool for
  large arrays (hundreds of elements).

## Open questions / risks

- **Grouping source.** Does the array builder cleanly expose per-element basis
  ranges, or do we need to thread metadata through `PysimEngine`? (P0.)
- **Ground plane.** Breaks identical self-blocks (images differ per element).
  Decide: per-element self-blocks, or fold the image into coupling.
- **Strong-coupling robustness.** 2% coupling ⇒ fast block-Jacobi here, but
  tighter-spaced arrays could need block-Gauss-Seidel or a Schur-complement
  solve. Keep the solver pluggable.
- **Non-identical / non-regular arrays.** Identical-element and Toeplitz reuse
  degrade gracefully to per-element self-blocks and all-pairs coupling — still
  `P·N²`, just without the P³ factor win.
- **Worthwhile size.** Like the generic H-matrix, the constant factors mean the
  win shows above some size; measure where it beats dense for `P=8` (likely at
  or just above native, since storage is already 16% vs 41%).

## Validation criteria

- Impedance / Y-matrix within ~1e-4 of dense `BSplinePySim` (matched mesh) on
  `bowtiearray2x4` and a second array design.
- Storage ≈ self-fraction + small (target ~16% on bowtie at native).
- GMRES iterations small and ~flat in P and N.
- Wall-clock beats dense on the array designs at native size (the bar the
  generic H-matrix missed), and the per-frame phase-sweep re-solve is
  near-instant with a cached factorization.

## Relationship to existing code

- Reuses: `_build_geometry`, `_build_basis_polynomials`, the kernels,
  `_offedge_block_evaluators` (C++ `bspline_assemble_offedge_block`),
  `aca_partial`, and the augmented-saddle solve pattern.
- Does **not** reuse the geometric cluster tree (`build_cluster_tree`) — the
  whole point is a structural partition instead.
- Lives alongside `HMatrixPySim`; the generic H-matrix remains the tool for
  single large structures, the array-block solver for arrays.
