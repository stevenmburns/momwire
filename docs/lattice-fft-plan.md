# Lattice-FFT array solver — BTTB/FFT matvec for very large regular arrays

Design plan resolving the FMM scoping issue (#168). Driving example (set in
the scoping discussion): a **very large array, 10,000 elements in a 100×100
square lattice** — ~1.4–4×10⁵ basis functions at practical element meshes,
electrical aperture ~70–100λ at half-to-0.7λ element spacing.

## Decision: FFT convolution over the element lattice, not FMM

Issue #168 listed three candidates — kernel-independent FMM (bbFMM/KIFMM), an
H² upgrade of the existing cluster tree, and AIM/precorrected-FFT. The driving
example discriminates sharply between them, and it does so *against* the two
generic candidates:

- **KIFMM / bbFMM — wrong regime.** A 100×100 lattice at 0.7λ spacing is a
  ~70λ aperture. Kernel-independent FMM (Chebyshev/equivalent-source
  translation) has translation ranks that grow like `(k·D)²` once a tree box's
  diameter `D` exceeds ~1λ — and the upper levels of any octree over a 70λ
  aperture are tens of λ across. The interpolation-based operators explode
  exactly where the tree is supposed to be doing its work. The fix is
  directional/diagonal translation — MLFMA — which #168 already ruled out as a
  multi-month, wrong-regime lift. Capping the tree at ~1λ boxes instead leaves
  ~10⁴ top-level boxes interacting all-pairs: the O(P²) wall again.
- **H² upgrade — same disease.** ACA/H² far-block rank grows with the
  *electrical diameter of the clusters*, and the upper-level admissible blocks
  of a 70λ structure are electrically huge. The existing `hmatrix.md` already
  flags this ("electrically very large structures") as the boundary of the
  H-matrix's regime. H² fixes the O(N log N)→O(N) storage constant, not the
  oscillatory rank growth.
- **Lattice FFT (AIM/pFFT-kin) — exact at any frequency.** On a regular
  lattice of identical elements, free-space translation invariance makes the
  element-pair coupling block depend *only on the displacement*:
  `Z_ab = T(r_b − r_a)`. That is not an approximation that degrades with
  aperture — it is exact at any electrical size. The coupling operator is
  block-Toeplitz in each lattice dimension (BTTB), and a BTTB matvec is a 2-D
  FFT convolution over the element grid: **O(N log P) per matvec, O(P·n_e²)
  storage**, no rank growth, no tree, no new kernel math.

`ArrayBlockSolver` (docs/array_block_solver_plan.md) already exploits exactly
this structure for *storage and assembly* — its P3 dedup keys coupling blocks
by `(shape_a, shape_b, displacement)` and runs ACA once per unique key. What
it does **not** do is exploit the structure in the *matvec* or in the *pair
enumeration*, and those are precisely the two O(P²) walls at P = 10⁴:

1. `build_array_blocks` classifies pairs with a `for a: for b:` Python double
   loop — 10⁸ iterations — and materialises a `coupling` list with one entry
   per ordered pair — ~10⁸ tuples, ~10 GB of list overhead, before any math.
2. `ArrayBlock.matvec` loops that list — 10⁸ small `U@(V@x)` products per
   Krylov iteration.

Everything else already scales: unique-displacement ACA/dense fills are
O(P) on a lattice (~4×10⁴ at 100×100, ~2×10⁴ with the `Z_ba = Z_abᵀ`
symmetry), the block-Jacobi preconditioner factors **once per shape** and
applies as one wide BLAS-3 solve, and the augmented-GMRES/KCL/Sommerfeld
machinery is shape-agnostic. So this is not a new solver: it is a third
representation of the coupling operator inside `ArrayBlockSolver`, chosen
automatically when the geometry is a lattice.

### Measured walls (per-pair `ArrayBlock` baseline, vertical-dipole lattices)

15-seg degree-1 half-wave dipoles, 0.7λ spacing, free space, centre feed
(block-Jacobi preconditioner, GMRES restart 50):

| grid | P | n | ACA fills | rank | build | matvec | solve | iters |
|---|---|---|---|---|---|---|---|---|
| 4×4 | 16 | 224 | 24 | ~5 | 0.03 s | 0.001 s | 0.07 s | 12 |
| 8×8 | 64 | 896 | 112 | ~4 | 0.12 s | 0.014 s | 2.4 s | 46 |
| 16×16 | 256 | 3,584 | 480 | ~4 | 0.77 s | 0.23 s | 80 s | 178 |
| 24×24 | 576 | 8,064 | 1,104 | ~4 | 2.6 s | 1.19 s | 584 s | 266 |

Two separate walls, both visible already:

- the per-pair matvec grows ~O(P²) (0.001 → 1.19 s) and the pair list with
  it — extrapolated ~2 min *per matvec* and ~10 GB of list overhead at
  100×100;
- **block-Jacobi iteration counts explode** (12 → 46 → 178 → 266): a
  resonant half-wave lattice at 0.7λ is strongly coupled, and a
  preconditioner that ignores all coupling degrades with array size. This
  wall is independent of how fast the matvec is.

### Measured: lattice-FFT path (same ladders)

FFT coupling operator + Floquet (periodic-array) preconditioner:

| grid | n | kernel fills | build | matvec | solve | iters | store % |
|---|---|---|---|---|---|---|---|
| 8×8 | 896 | 112 | 0.40 s | <1 ms | 0.48 s | 39 | 5.5 |
| 16×16 | 3,584 | 480 | 1.7 s | 1 ms | 1.9 s | 82 | 1.6 |

Exactness: `to_dense` matches dense Z to ~1e-17 (the FFT kernel stores
displacement blocks exactly — *better* than the ACA'd pair path); impedance
matches the dense solver to ~1e-10 on small grids.

The 16×16 end-to-end: 80 s → 1.9 s (~40×). Iterations still grow (39 → 82):
the Floquet preconditioner is exact for the *infinite* lattice, so what
remains is genuinely the edge effect, and restart-50 truncation compounds it
(restart 200 gives 64). Scaling of the iteration count to 100×100 is the
open risk being measured; per-bin conditioning of the Floquet blocks is
elevated but bounded (worst ~3×10⁴ near the lattice's surface-wave
resonances).

## Design

### Operator: `LatticeArrayBlock`

A sibling representation to `ArrayBlock` with the same solve-facing surface
(`near`, `groups`, `shape_of_elem`, `shape_blocks`, `matvec`, `matmat`,
`storage`, `stats`, `to_dense`), consumed unchanged by
`HMatrixSolver._solve_hmatrix` and `_BlockJacobiAugPrecond`. Differences:

- **Coupling kernel tensor** instead of a pair list:
  `K[di, dj] = T(di·a₁ + dj·a₂)`, a dense `(n_e, n_e)` block per lattice
  displacement, `di ∈ [−(Pₓ−1), Pₓ−1]`, `dj ∈ [−(P_y−1), P_y−1]`, with
  `K[0, 0] = 0` (self-blocks stay separate, exactly as today). Assembled with
  the existing `_offedge_aca_evaluators(...)[2]` **dense** closure — which
  already folds in PEC-image / refl-coef / Sommerfeld-C2 ground terms — one
  fill per unique displacement, halved by `T(−Δ) = T(Δ)ᵀ` (same
  uniform-radius gate the ACA dedup uses).
- **Matvec = 2-D FFT convolution.** Reshape `x` to `(Pₓ, P_y, n_e)` by
  lattice coordinate; zero-pad the lattice axes to FFT-friendly sizes
  `≥ 2P−1`; precompute `K̂ = FFT₂(K)` once at build; per apply:
  `ŷ = K̂ · x̂` as a batched `(n_e × n_e) @ (n_e,)` matmul per frequency bin;
  inverse FFT, crop, add the self-block products and `extra_lowrank`
  (Sommerfeld remainder) exactly as today. `matmat` batches the RHS axis
  through the same FFTs.

### Lattice detection

`build_array_blocks` gains a structural gate before its pair loop. The FFT
path fires iff:

- exactly **one block-shape class** (which, per the existing key logic,
  already implies same height under ground — so ground correctness rides the
  existing keys for free), and
- element centroids fit an integer lattice: take two shortest linearly
  independent centroid differences as basis `(a₁, a₂)` (or one, for a line
  array — the 1-D case degenerates naturally), solve for integer coordinates,
  and verify the fit to the existing `disp_tol`. Any misfit → fall back to
  the current per-pair path unchanged.

The lattice may be any regular Bravais grid in any plane orientation —
nothing assumes axis alignment or square cells.

### Budget at the driving example (100×100, n_e ≈ 14–40)

| item | 15-seg elements (n_e=14) | 41-seg (n_e=40) |
|---|---|---|
| N | 1.4×10⁵ | 4×10⁵ |
| kernel `(199², n_e²)` | 124 MB | 1.0 GB |
| spectrum (dropped kernel) | ~130 MB | ~1.0 GB |
| unique dense fills | ~2×10⁴ | ~2×10⁴ |
| matvec | ~0.05 s | ~0.3 s |
| ACA per-pair path (today) | 10⁸ tuples, ~10 GB, minutes/matvec | worse |

Solve = block-Jacobi GMRES; iteration count at 0.7λ spacing measured 12 at
4×4 (ladder will show the P-trend; coupling decays as 1/R so growth should
be mild). A nearest-neighbour-ring strengthened preconditioner is the known
lever if iterations grow — the blocks are already in `K`.

## Phases

- **P0 — probe + doc (this).** Ladder measurements; design review.
- **P1 — lattice detection + kernel assembly.** `_detect_lattice(cen)`;
  dense displacement fills with transpose reuse; exactness gate: kernel
  blocks vs existing ACA blocks `U@V` on 4×4 (≤ aca_tol) and vs
  `zblock` directly.
- **P2 — FFT matvec + solve.** `LatticeArrayBlock` with FFT `matvec`/
  `matmat`; validated to roundoff against the per-pair `ArrayBlock.matvec`
  on the same kernel (small grids), then end-to-end impedance vs dense
  `BSplineSolver` on 4×4/6×6 (~1e-5, the ArrayBlock bar); ground + Sommerfeld
  variants on a small grid.
- **P3 — scale + integrate.** Ladder to 100×100 (build, matvec, solve, RSS);
  registered behaviour is automatic (same `arrayblock` solver key — the FFT
  representation is an internal choice); scaling table into this doc;
  close-out comment on #168.

## Out of scope (recorded for #168)

- **True FMM (KIFMM) for irregular geometry** — the honest storage-ceiling
  case from #168 (wire-grid vehicles, non-lattice meshes) remains open; it
  needs sub-λ tree caps or directional operators at these apertures. Revisit
  if a driving irregular workload materialises.
- **MLFMA** — confirmed out of scope (multi-month, and the lattice case is
  served exactly by the FFT path).
- **Full 10⁴-port Y-matrices** — a 100×100 array's port matrix is 10⁸
  entries; the solve path stays multi-RHS-capable but nothing chunks a
  10⁴-column RHS yet. Phased-excitation solves (one RHS) and few-port Y are
  the supported driving usages.
