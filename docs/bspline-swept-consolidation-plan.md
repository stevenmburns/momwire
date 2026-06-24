# BSpline swept-solve consolidation — plan & status

**Goal:** make `BSplineSolver` fast enough on frequency sweeps to retire
`TriangularSolver`, collapsing the two Galerkin solvers (same linear basis at
`degree=1`) into one. This tracks NEXT_STEPS item 19 and the work on PR #101.

This doc is the working memory for the effort: what's done, what the
measurements say, what's left, and how to do it. Pick it up from here.

---

## Why

- `TriangularSolver` and `BSplineSolver(degree=1)` are the **same linear (tent)
  basis**. Carrying two independent assembly pipelines for one basis is the
  maintenance cost we want to remove.
- `BSplineSolver(degree=2)` has genuinely better accuracy and is the desired
  default; we only kept triangular because bspline's swept path was slow.
- Retiring triangular nets **≈ −1400 lines** and one fewer solver:

  | removed | lines |
  |---|--:|
  | `triangular.py` | 1078 |
  | `_triangular_kernels.py` | 272 |
  | triangular-only C++ (`seg_seg_quad_batch_3d`, `seg_seg_reg_quad_batch_1d`, `assemble_Z`, `assemble_Z_general`) | ~500–600 |
  | **added** (bspline swept, PR #101) | ~500 |
  | **net** | **≈ −1400** |

  ...but *only once bspline covers everything triangular does* (see Remaining
  work). Until then it's a net add.

---

## Status

### Merged (`main`)
- **Streaming reg-moment kernel** (`seg_seg_reg_moments_bspline_swept`): the
  same-edge smooth-kernel moments, batched over k, streaming (no `n_qp²` phase
  intermediate). Wired into both swept and single-k paths. This was the real,
  memory-safe win: d=2 swept ~3× single-thread / ~4.5× threaded; single solves
  2.0–2.24×. **bspline d=1/d=2 single solves now beat triangular.**

### PR #101 (draft, `feat/bspline-swept-batched-assemble`) — the common case
Triangular-style fully batched swept path for the **simple case only**:
- `seg_seg_full_moments_bspline_swept` (C++): off-edge moments batched over k
  (analog of `seg_seg_quad_batch_3d`).
- `assemble_Z_bspline_swept` (C++): batched assemble over a leading k axis
  (analog of triangular's batched `assemble_Z`).
- `_compute_impedance_swept_batched` (Python): chunked fast path in
  `compute_impedance_swept`; per-k loop / numpy remain as fallback.

Covers: single + multi-feed `compute_impedance_swept`, **no junctions, no
enrichment**, free-space or PEC ground. Bit-exact (`|Δz|~1e-10`), 158 tests
pass, OpenMP to physical cores.

---

## Measurements (single-thread / 4 threads, 41-freq sweep)

`bspline d=1 / triangular` (same basis) with the batched path active:

| case | 1 thread | 4 threads |
|---|--:|--:|
| dipole101 free | 1.29× | 1.36× |
| dipole101 ground | 1.37× | 1.47× |
| yagi3 free | 1.32× | **1.04×** |
| yagi3 ground | 1.37× | **1.09×** |

**Read:** parity on **multi-wire + threaded** (the real interactive case);
still 1.3–1.5× slower on a **single large wire** and single-threaded.

### Why the residual gap (single large wire)
At `d=1` the moment *count* equals triangular's (4 components), so it isn't the
basis (clamped B-spline degree 1 *is* the tent function). The gap is in the
**same-edge moment scheme**, which the two solvers implement differently —
bspline via the general-degree polynomial-moment path (`_seg_seg_static_moments`
+ the reg streaming kernel), triangular via tent-specific closed forms
(`_triangular_kernels`: `_H1`/`_H2`/`_Sigma`/`_J_static_all` + its reg
quadrature). On one big 100×100 same-edge block (a single wire) bspline's more
general path dominates; on multi-wire the same-edge blocks are small and most
work is cross-wire off-edge (shared kernel) → parity.

> ⚠️ **These are genuinely different quadratures, not equivalent ones** — see
> the dedicated section below. The `_bspline_kernels.py` module docstring claims
> d=1 is "bit-for-bit equivalent" to triangular's same-edge kernels; that does
> **not** hold end-to-end (the impedances differ by up to ~5% at coarse N). So
> the performance gap and a real **accuracy/result-shift** question share the
> same root: the two solvers are different numerical schemes.

### Why batching helps multi-wire but not single big wires
The win tracks the **per-k Python/overhead fraction**, not compute. Small
multi-wire matrices are overhead-heavy → batching removes it → win. Large
single edges are compute-bound on genuine off-edge moment work that batching
can't reduce → wash. (Confirmed: 19a off-edge batching *alone* was a wash;
only 19a+19b together, removing the per-k assemble+solve, helped — and only
where overhead dominated.)

### Memory / chunking
The batched moment tensor is `(chunk, NM², N², )` complex — `NM²=9` at d=2.
`_compute_impedance_swept_batched` chunks k to keep it ~32 MB/chunk; the UI
sweep chunker can shrink further for large N. No memory wall in practice.

### OpenMP
Both new kernels scale to physical cores (~3.3× on 4) then plateau on
hyperthreading (normal for compute-bound SIMD). No tuning needed; server pins
OMP to physical cores.

### SIMD / libmvec (correction to an earlier wrong conclusion)
An earlier ".so-swap" experiment was a **no-op** (imports resolved to a
site-packages copy, not the swapped `src/` build). Corrected result: libmvec
vector sincos gives **~3.3× on the hot kernels at AVX2 width**; a NEON port
would recover **~1.5×** at 128-bit width (arm64 has no 4-wide doubles). The
kernels are **sincos-bound, not memory-bound**. macOS currently runs scalar
sincos (the `__APPLE__` guard disables the libmvec block), so a SLEEF/Accelerate
vForce NEON port is a real (if modest) lever — separate from this effort.

---

## ⚠️ Triangular and bspline d=1 are genuinely different quadratures

**This must be understood before retiring triangular — it's a correctness
question, not just performance.**

Despite being the same linear basis (and a `_bspline_kernels.py` docstring
claiming d=1 is "bit-for-bit equivalent" to triangular's same-edge kernels),
the two solvers **do not produce the same impedance**. Driver-point Z, same
geometry, same quadrature order (`n_qp=4` both):

| case | triangular | bspline d=1 | rel diff |
|---|---|---|--:|
| dipole N=21 | 11.61 − 963.8j | 12.84 − 1013.2j | **5.1%** |
| dipole N=81 | 11.57 − 958.9j | 11.97 − 975.4j | **1.7%** |
| yagi2 N=21 | 34.86 − 11.65j | 34.93 − 11.55j | 0.34% |
| yagi2 N=81 | 35.55 − 9.85j | 35.57 − 9.82j | 0.11% |

The difference is largest at coarse N and **shrinks with refinement**
(dipole 5.1% → 1.7% as N: 21 → 81) — the signature of two **genuinely different
finite-N quadrature schemes** that converge to the same continuum limit but
disagree at the segment counts users actually run (the UI default is ~21
segs/wire, i.e. the 5%-disagreement regime, worst on a single long wire).

**Implications (both matter):**

1. **Correctness / result-shift.** Retiring triangular would shift every
   user's computed Z — by up to ~5% on a coarse single-wire reactance. Before
   doing that we must understand *which scheme is more accurate* (compare both
   against NEC / the convergence study in NEXT_STEPS item 13, and against a
   refined-N reference) and decide whether the shift is acceptable. The
   "bit-for-bit equivalent" docstring is **wrong end-to-end** and should be
   corrected once we know why (candidates: the same-edge static-moment split,
   the delta-gap source projection, or the off-edge full-kernel evaluation —
   not yet pinned down).

2. **Performance.** This is *why bspline can't match triangular even in the
   limited cases*: it isn't running the same (cheaper, tent-specific) numerical
   scheme — it runs a more general polynomial-moment scheme that is both costlier
   and numerically distinct. So "make bspline match triangular's speed" and
   "make bspline match triangular's answer" are the same underlying question.

**Action before retirement:** pin down the mechanism of the disagreement
(diff the same-edge, off-edge, and source contributions between the two at a
fixed geometry), establish which is more accurate, and document the expected
result-shift. Only then is the speed comparison even the right comparison.

---

## Remaining work (the general case) — priority order

The batched fast path falls back to the per-k loop for these. Triangular
*batches* the first two, so they are the real blockers to retirement.

### 1. Wire junctions — batched KCL Schur solve  *(hard half; do first)*
Fan dipoles and any K≥3 junction. Triangular already does this:
- `triangular._assemble_Z_general_batch` — batched assemble for
  junction/directional bases (per-basis support arrays, not the dense
  left/right-seg layout).
- `triangular._solve_with_kcl_batch` / `_solve_with_kcl_swept_ports` — batched
  Schur-complement constrained solve `[Z Aᵀ; A 0][I;λ]=[v;0]`.

Plan:
- Add **`assemble_Z_bspline_general_swept`** (C++): batched version of
  `assemble_Z_bspline` but taking the general per-basis `(support_seg,
  support_L, support_R)` layout (mirror the single-k `assemble_Z_general`).
  The junction path uses the general assembler, not the dense one this PR's
  `assemble_Z_bspline_swept` targets.
- In `_compute_impedance_swept_batched`: drop the `not self.junctions` guard;
  when `kcl_A` is non-empty, replace the plain stacked `np.linalg.solve` with a
  **batched Schur solve** over the `(n_k, N, N)` Z stack + the (k-independent)
  `kcl_A`. Mirror `bspline._solve_with_kcl` but vectorized over k (the
  per-frequency Schur reduction is independent, so it batches as stacked
  matmuls + two batched solves).
- Reference test: `test_bspline_fandipole_swept_matches_per_freq` (already
  exists; currently exercises the per-k path).

### 2. `compute_y_matrix_swept` — multi-port Y matrix
Transmission-line networks, multi-band traps. Triangular generalizes the same
batched path to a **matrix RHS** (`_solve_with_kcl_swept_ports`,
`compute_y_matrix_swept`). Plan: add a batched fast path to bspline's
`compute_y_matrix_swept` reusing the batched assemble + a batched multi-RHS
solve (RHS is the per-port source matrix instead of a single vector).

### 3. Singular enrichment — batched  *(lowest priority)*
bspline-only (no triangular equivalent, so not a retirement blocker). The
`assemble_Z_enrich` kernel is single-k; a swept variant would keep the
enriched-junction sweep on the batched path. The auto variant's
pass-1-needs-no-kernel optimization mitigates the worst case. Defer.

---

## Decision criteria before actually retiring triangular

Retire only when **all** hold:
0. **(gate) The quadrature disagreement is understood.** Per the section above,
   bspline d=1 and triangular give different Z (up to ~5% at coarse N). Pin down
   the mechanism, establish which is more accurate vs an external reference, and
   accept/document the result-shift retirement would impose on users. *This is
   the first gate — the speed comparison is only meaningful once we know the two
   are computing acceptably-equivalent answers.*
1. Items 1 & 2 above are batched in bspline (junction + multi-port sweeps no
   longer regress vs triangular).
2. `MomwireEngine` default switched from `TriangularSolver` to a bspline degree
   (and antennaknobs `geometry.py` updated — it currently says it feeds
   `TriangularSolver`).
3. Triangular's tests migrated or dropped; its C++ kernels confirmed unused by
   the H-matrix / array-block subclasses (those are `BSplineSolver` subclasses,
   so expected clean — verify).
4. Accept the residual: **single large-wire single-threaded sweeps stay
   ~1.3–1.5× slower** than triangular. If that case matters, close the
   same-edge static-moment gap first (a cheaper same-edge swept path), or keep
   triangular purely for it.

If the junction/multi-port batching proves not worth it, the fallback is:
**keep triangular as the swept engine, use bspline d=2 only as the accuracy
option** — the merged reg kernel already makes d=2 interactively fine.

---

## Pointers
- Branch / PR: `feat/bspline-swept-batched-assemble` → PR #101 (draft).
- Key files: `src/momwire/_accelerators.cpp` (kernels),
  `src/momwire/_bspline_kernels.py` (Python wrappers + capability flags),
  `src/momwire/bspline.py` (`_compute_impedance_swept_batched`,
  `_build_J_blocks`, `compute_impedance`).
- Triangular reference: `src/momwire/triangular.py`
  (`compute_impedance_swept`, `compute_y_matrix_swept`,
  `_assemble_Z_general_batch`, `_solve_with_kcl_batch`).
