# Sommerfeld everywhere — Sinusoidal + fast-solver support

Goal: every shipping solver handles the full ground menu — free space,
PEC image, refl-coef ("finite fast"), and Sommerfeld ("finite") — so the
antennaknobs solver/ground matrix loses its per-solver special cases.
TriangularSolver is exempt: it is retirement-bound (frontend retirement
landed separately; backend removal is a follow-up once its one unique
capability is transferred to BSpline).

Builds directly on docs/sommerfeld-ground-plan.md (BSpline dense
support, v0.6.0) and docs/sommerfeld-perf-plan.md (module-level grid
cache + C++ fill, v0.7.0). Everything below reuses that machinery — the
NEC decomposition (theory manual eqs 136–147)

    Z_ground = C₂·(exact image) + (smooth remainder F),
    C₂ = (ε̃−1)/(ε̃+1)

is the load-bearing asset: the singular part rides the existing image
code paths in every solver, and only the SMOOTH remainder needs new
plumbing. Smoothness is what makes each phase cheap:

- Sinusoidal is field-based, so the interpolated F dyad plugs into its
  image machinery even more directly than bspline's (which needed a
  special field-form Galerkin block precisely because its normal
  assembly is mixed-potential).
- For the fast solvers the remainder is smooth EVERYWHERE (near-diagonal
  included — the image term absorbed the singularity), i.e. globally
  low-rank: it can be carried as one global ACA term without touching
  the block partition at all.

## Phase 1 — shared grid-cache home

- [x] Move `_SOMM_GRID_CACHE`, `_SOMM_GRID_CACHE_MAX`, `_somm_r1_bucket`
      and the lookup/build logic from `bspline.py` into `_sommerfeld.py`
      as `get_grid(eps_t, k2, r1_max, omega, mu, cancel_flag)` (module
      FIFO, same `(ε̃, k, r1_bucket, ω, μ)` key, same 128 bound).
      `BSplineSolver._somm_grid` delegates. Pure refactor — existing
      sommerfeld tests must pass unchanged.

## Phase 2 — SinusoidalSolver

The refl-coef path already does "compute image-source field components,
weight by a per-pair dyad, project on the observer tangent"
(`_field_tensor_image_refl`). Sommerfeld swaps the Fresnel dyad for
(a) a CONSTANT C₂ weight on the exact image — a plain scalar on the
already-projected PEC-image tensor, so the C++ `sinusoidal_field_tensor`
kernel keeps serving it — plus (b) a remainder tensor from the grid.

- [x] Constructor: `ground_model="refl-coef"` (default) / `"sommerfeld"`
      + `n_qp_sommerfeld=3`, validation mirroring bspline (model value
      set; sommerfeld requires `ground_eps`; wires strictly above
      `ground_z` checked at assembly). Default preserves every existing
      result bit-exactly. De-stale the module-docstring scope list.
- [x] Remainder tensor S[3, M, N]: for source segment n, GL nodes z'_q
      along the segment (± half-length, `n_qp_sommerfeld` points);
      source shapes {1, sin(k z'), cos(k z')} at the nodes; F dyad per
      (obs-center m, node q) from `grid.eval(R₁, θ)` with the SAME
      azimuth/projection algebra as `BSplineSolver._Z_sommerfeld_
      remainder` (eqs 143–147 factors, ρ→0 d̂ fallback, obs-tangent
      projection). Point-matched at observer centers — single (source-
      side) quadrature, no Galerkin double integral. Chunked over
      observers like bspline to bound the working set.
- [x] Assembly seam (`_assemble_Z` ground branch):
      `Phi = Phi_free − C₂·Phi_img + S` — sign analysis says the
      remainder ADDS in the field-form tensor convention (bspline's
      subtracted `+Q` is a −⟨f,E⟩ convention artifact); pinned
      empirically against bspline-sommerfeld on a low dipole (see
      tests) exactly the way the ground-plan pinned bspline's signs,
      NOT trusted from this paragraph.
- [x] Grid extent from segment endpoints (obs-to-image distance is
      convex ⇒ endpoint-pair max, same as bspline), `_sommerfeld.
      get_grid` shared cache; swept loops already update `self.omega`
      per k, so per-k ε̃/grids come free.
- [x] Tests (`tests/test_sommerfeld_ground_sinusoidal.py`, 19 tests): constructor validation; default-model bit-exactness;
      free-space limit ε̃=1; PEC-limit collapse to the PEC image at
      ε̃=1e16; swept-vs-single-k; cross-solver pin — sinusoidal-somm vs
      bspline-somm on a dipole at 0.05 λ (remainder ~20 Ω there: a sign
      error is ~2× the effect, unmissable) within the cross-solver
      floor; gn 2 golden gates at the heights the ground-plan used.
      Payoff CONFIRMED (measured 2026-07-07, full 39-case gn 2
      matrix): dipole max 0.10 Ω over 0.02-0.35 λ (0.91 at 0.5 λ, where
      the image ray runs past nec2c's own 1 λ grid edge), inverted_l
      max 0.14, yagi max 0.21 — vs bspline's 2.36/2.74/0.98. The
      sinusoidal suite is now the sharpest validation of the Sommerfeld
      engine itself. Remainder sign pinned by the 0.05 λ cross-solver
      test (sin-somm vs bspl-somm 1.45 Ω apart; refl-coef 22 Ω away).

## Phase 3 — HMatrix/ArrayBlock: one global low-rank remainder term

Architecture: do NOT thread the remainder into per-block fills. The
remainder is globally smooth ⇒ globally low-rank ⇒ carry it as ONE
extra term

    Z ≈ H(free − C₂·image) − (−1)·U_S V_Sᵀ

where H is today's refl-coef-shaped machinery with constant-C₂ weight
tables (the weighted-image fills already exist for Fresnel weights;
C₂ is the degenerate constant case) and U_S V_Sᵀ is a single global ACA
factorization of the remainder block Q over ALL basis functions.

- [ ] Remainder row/col samplers over the full basis set: row i = the
      Galerkin Q[i, :] against every basis (batch `grid.eval` over the
      (q·supp) × (N·q) node pairs — vectorized Lagrange stencil gather,
      same math as the dense `_Z_sommerfeld_remainder` restricted to one
      row/column). Reuses the Phase-1 shared grid.
- [ ] `aca_partial(get_row, get_col, m, n)` once, global; tolerance tied
      to the existing far-block ACA tol. GMRES operator gains one
      `U_S (V_Sᵀ x)` term in both solvers; the near-field/block-Jacobi
      preconditioners IGNORE the term (smooth perturbation — measure
      iteration-count impact, expected small).
- [ ] Block-reuse audit (ArrayBlock): the remainder rides OUTSIDE the
      block cache, so the `(shape_a, shape_b, displacement)` dedup and
      the sweep-frame self-block factorization reuse are untouched.
      Height dependence (z+z′) is the ACA term's problem, and ACA
      doesn't care.
- [ ] Flip the gates: `ground_model=="sommerfeld"` leaves
      `_hmatrix_unsupported` (hmatrix.py) / `skip_block_cache`
      (array_block.py). Constant-C₂ weight tables through the
      weighted-image block fills (near dense + far ACA image blocks).
- [ ] Tests: replace `test_fast_solvers_fall_back_to_dense` with
      fast-vs-dense agreement at ACA tolerance on the dipole + an array
      case; PEC/free-space limits through the fast path; measure and
      LOG the global remainder rank on bowtiearray2x4-class geometry
      (expect low tens; if it grows past ~50 at native sizes, fall back
      to per-element-pair remainder blocks — decision point, not a
      silent regression).

## Phase 4 — perf smoke, release

- [ ] Latency smoke vs the dense sommerfeld path and vs refl-coef on:
      dipole N=45, yagi N=255, bowtiearray2x4 native N. Grid fill
      dominates and is shared/cached — the marginal cost of the fast
      path must be the samplers + ACA only. 41-freq sweep number
      recorded (grid refill per k is the known dominant cost, tracked in
      sommerfeld-perf-plan; not made worse here).
- [ ] Docstrings (sinusoidal scope list, hmatrix/array_block "correct at
      dense cost" comments), this doc's checkboxes.
- [ ] momwire minor release v0.8.0 (new public kwargs on
      SinusoidalSolver; fast-solver behavior change is perf-only —
      results identical to the dense path within ACA tol).

## Phase 5 — antennaknobs wiring (separate PR, after the release)

Per the default-cost-audit discipline: trace the unqualified request
path, latency-smoke before the wiring PR merges, sommerfeld stays
OPT-IN (UI default "fast").

- [ ] `MomwireEngine`: drop the `__name__ == "BSplineSolver"` gate —
      `("finite", εr, σ)` → `ground_model="sommerfeld"` for ALL momwire
      solvers with ground_eps support (now: bspline, sinusoidal,
      hmatrix, arrayblock). `("finite-fast", ...)` stays refl-coef
      everywhere. Exact-pin momwire==0.8.0 + floor in the same PR.
- [ ] Frontend: `backendHasGroundMethod` allowlist → derived from "is a
      momwire solver with finite ground" (or deleted outright); remove
      the per-backend "solved as refl-coef" apology strings. Triangular
      is already retired from the UI (separate branch), so the solver ×
      ground matrix is uniform.
- [ ] `ground_model_applied` in the web adapter keeps reporting the
      engine's actual mapping — the response contract doesn't change,
      the values just stop surprising people.
