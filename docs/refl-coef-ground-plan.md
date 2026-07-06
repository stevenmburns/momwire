# Reflection-coefficient finite ground — plan

**Goal:** implement a NEC-IPERF=0-style finite ground in momwire's impedance
solve. Today momwire's only ground model in the interaction matrix is the PEC
image; finite ground exists solely as a Fresnel post-process on the far field
(in antennaknobs' `MomwireEngine`). This plan adds Fresnel-weighted image
interactions to the matrix fill so grounded *impedance* reflects the real
ground, matching NEC's "finite ground, reflection coefficient approximation"
(`gn_card(0, ...)`).

This doc is the working memory for the effort: context, architecture
findings (with file refs), the one open physics decision, phases, validation,
and risks. Pick it up from here in a fresh session.

---

## Why

- antennaknobs PR #251 (merged 2026-07-05) fixed PyNEC's ground vocabulary:
  `("finite", εr, σ)` = Sommerfeld-Norton (`gn_card(2)`), new
  `("finite-fast", εr, σ)` = reflection-coefficient (`gn_card(0)`).
  `MomwireEngine` folds both to its single model: **PEC-image solve + Fresnel
  far field** (the EZNEC "MININEC-style" ground).
- The impedance gap is real and measured (28.47 MHz flat half-wave dipole
  over εr=10, σ=0.002): at **0.2λ height** the PEC-image solve gives
  Z ≈ 67.6+31.4j while both NEC finite models give ≈ 69+11j…+13j — an
  **~18–20 Ω reactance error**.
- NEC's two finite models agree with each other to ~0.1 dB / ~2 Ω down to
  ~0.2λ and only diverge hard below ~0.1λ (0.05λ dipole: gn 0 gives
  46.9+9.3j, gn 2 gives 62.0−2.5j) — a regime where the refl-coef model is
  itself unreliable and true Sommerfeld (out of scope, much bigger lift:
  Sommerfeld integrals + interpolation grids) is required.
- **Target accuracy window: 0.1–0.5λ heights**, where refl-coef corrects most
  of the PEC-image error at essentially PEC-image cost.

## What NEC IPERF=0 actually does

Per matrix element (observer segment m, source segment n): compute the field
of the **mirrored** source at the observer, decompose it into vertical and
horizontal polarization relative to the plane of incidence of the specular
ray (image point → observer), and weight each component by the Fresnel
plane-wave reflection coefficient ρ_v(θ) / ρ_h(θ) evaluated at that angle,
using the complex relative permittivity ε̃ = εr − jσ/(ωε₀). Coefficients are
treated as constant over a segment pair (evaluated from midpoint geometry) —
that *is* the approximation. The far-field ground treatment is unchanged
(NEC's `rp_card` over finite ground uses reflection coefficients for the
space wave regardless of IPERF; momwire/antennaknobs already do the
equivalent Fresnel-on-image far field).

Reference: `docs/nec2_theory_manual.pdf` (reflection-coefficient
approximation; the field-level weighting lives in NEC-2's `EFLD`).
Ground-truth oracle: `PyNECEngine(builder, ground=("finite-fast", εr, σ))`
from antennaknobs — exact NEC gn 0, cross-checked against nec2c in
antennaknobs `tests/test_nec_export.py`.

## Architecture fit (read 2026-07-05; line refs against v0.3.0 / main)

### BSplineSolver — primary target, good fit

Mixed-potential assembly: `Z = Z_A + Z_Φ` built from polarization-blind
scalar quadrature tensors (J blocks) plus a per-segment-pair tangent-dot
table that carries **all** orientation dependence of `Z_A`; `Z_Φ` (charge
term) consumes the J blocks with no tangent factor. PEC image today:
`_image_positions` (`bspline.py:710`), `_image_tangent_dot`
(`bspline.py:716`), `_build_J_image_blocks` (`bspline.py:720`), subtracted
as one sub-assembly with a single global minus sign. (TriangularSolver has
the same structure at `triangular.py:496–534` and assembly at
`triangular.py:538` — see "Consolidation interaction" for why we do NOT
implement there.)

The refl-coef ground slots in per term:

- **A-term — drop-in.** Replace the mirror tangent-dot
  `t_m · M · t_n` (M = diag(1,1,−1)) with `t_m · D_mn · (M t_n)` where
  `D_mn = ρ_v(θ_mn) v̂v̂ + ρ_h(θ_mn) ĥĥ` from the pair's specular geometry
  (image-source midpoint → observer midpoint; v̂ in the incidence plane, ĥ
  normal to it). Same (N_seg, N_seg) scalar-table shape as today — pure
  numpy, O(N²) small geometry, **quadrature kernels unchanged**.
- **Φ-term — the one real physics decision.** NEC weights *fields*, which
  mixes A and Φ contributions in a way a mixed-potential code cannot
  replicate exactly. We must pick a per-pair scalar weight for the image
  charge term. Candidates to evaluate empirically in Phase 1:
  1. ρ_v(θ_mn) at the specular angle (plane-wave TM coefficient);
  2. the quasi-static image coefficient (ε̃−1)/(ε̃+1) (image charge in a
     dielectric half-space);
  3. blends / angle-limited variants of the above.
  Expect ~Ω-level parity with NEC gn 0, not the ~0.1 Ω free-space parity —
  the acceptance bar below reflects that.
- **Assembly plumbing.** The C++ assemble takes one J set feeding both terms,
  so the image sub-assembly needs either (a) Python-side assembly for the
  image block only (fine to start — image fill is the same cost class as the
  free-space fill it doubles), or (b) an optional second weight-table
  argument on the C++ assemble (Phase 2, if profiling warrants).
- **Sweeps.** ρ depends on ω (through ε̃), so swept paths need per-k weight
  tables; the specular geometry is frequency-independent and computed once.
- **Junction/KCL** general-assembly variant consumes the same tables — same
  treatment, small.
- **Enrichment + ground** stays rejected (`bspline.py:217–221`), unchanged.

### HMatrix / ArrayBlock — dense fallback (gate added in Phase 3)

The claim above ("both fall back to dense when ground is on") was stale:
both fast paths grew per-block PEC image support after this plan was
written, and their `_hmatrix_unsupported` gates no longer excluded ground.
With `ground_eps` set they would have run PEC physics silently. Phase 3
widened both gates: `ground_eps is not None` ⇒ dense BSplineSolver path
(correct, at dense cost). Fresnel-weighting the per-block image terms is
future work if grounded arrays ever need the fast path.

### SinusoidalSolver — deferred (Phase 4, optional)

Field-based: the C++ hot kernel `sinusoidal_field_tensor` (the stated 70%
bottleneck at N≳80) returns fields already projected onto the observer
tangent and integrated (`sinusoidal.py:645–685`), so there is no place to
insert the polarization decomposition after the fact. Ironically it is the
formulation closest to NEC's own; it just discards the vector field too
early. Needs a kernel variant that decomposes the image field into v/h
components at the observer *before* projection (C++ change), or a slow
numpy image path. Note: the module docstring (`sinusoidal.py:11`) still says
"free-space only" while a PEC image build exists (`sinusoidal.py:831–851`) —
reconcile that docstring in Phase 0 regardless.

### Consolidation interaction — do NOT build on TriangularSolver

`docs/bspline-swept-consolidation-plan.md` is actively retiring
`TriangularSolver` in favor of `BSplineSolver` (NEXT_STEPS 19, PR #101 on
branch `feat/bspline-swept-batched-assemble`). Prototype and land this work
on **BSplineSolver only**; adding features to triangular is adding to code
slated for deletion. Coordinate rebases with that effort — both touch the
bspline assembly path.

## API sketch

momwire solvers currently take only `ground_z` (PEC image implied). Proposed:

```python
BSplineSolver(..., ground_z=0.0,
              ground_eps=None)   # None → PEC image (today's behavior);
                                 # complex ε̃, or (eps_r, sigma) tuple →
                                 # refl-coef weighted image (σ needs ω,
                                 # which the solver already has per k)
```

Backwards compatible: `ground_eps=None` preserves every existing test.

antennaknobs follow-up (Phase 3): `MomwireEngine` maps
`("finite-fast", εr, σ)` — and plausibly `("finite", εr, σ)` too, since
refl-coef is momwire's best available finite model — to
`ground_eps=(εr, σ)` instead of folding to a PEC solve. Far-field Fresnel
post-process (engines/momwire.py ~463–520) is already correct and stays.
Web adapter already ships real εr/σ for finite grounds since PR #251.

## Phases

### Phase 0 — scaffolding & golden references (done 2026-07-06)
- [x] Capture golden Z values from `PyNECEngine(ground=("finite-fast", ...))`
      for the validation matrix below; store as literals in a momwire test
      module so momwire's suite needs no PyNEC dependency.
      → `scripts/capture_refl_coef_ground_golden.py` (runs under the
      antennaknobs venv) regenerates `tests/golden_refl_coef_ground.py`:
      30 cases (2 geometries × 5 heights × 3 grounds), each with gn 0
      (oracle), gn 2 (sanity), and PyNEC PEC (reference). Captured values
      reproduce the plan's motivating numbers (dipole 0.2λ/(10, 0.002):
      gn0 68.8+13.2j vs PEC 66.2+31.5j; 0.05λ gn0/gn2 divergence).
- [x] Fix the stale `sinusoidal.py:11` "free-space only" docstring.
- [x] Check PR #101 status; agree base/rebase strategy with that effort.
      Checked 2026-07-06: PR #101 open, WIP. Strategy: this branch stays
      based on `main`; Phase 1 goes in new functions (no edits to the
      swept/assembly code #101 touches); rebase onto main after #101 lands
      and only then consider the Phase 2 C++ assemble extension.
      **Update (later 2026-07-06): PR #101 judged unlikely to ever merge —
      Phase 2 proceeded against main directly, editing the swept paths as
      needed. If #101 is ever revived it must rebase over this.**

### Phase 1 — physics prototype (bspline, dense, single-k, Python assembly)
(done 2026-07-06; see `scripts/compare_refl_coef_ground.py` for the numbers)
- [x] Shared helper: per-pair specular geometry + ρ_v/ρ_h tables from ε̃(ω)
      (vectorized numpy; midpoint approximation, NEC-style).
      → `src/momwire/_ground_refl.py`. One deviation from the sketch above:
      the A-term dyad is NEC EFLD's actual field weighting — in-incidence-
      plane components (z + horizontal-parallel) × ρ_v, out-of-plane
      horizontal × −ρ_h — not the transverse ρ_v·v̂v̂ + ρ_h·ĥĥ form. In
      momwire's image convention (mirrored tangents, one global minus) the
      PEC limit then reduces to the identity dyad exactly.
- [x] A-term dyad table replacing `_image_tangent_dot` when `ground_eps` set.
      → `BSplineSolver(ground_eps=...)` + `_image_Z_refl` (Python assembly,
      new function; single-k `compute_impedance`/`compute_y_matrix` and the
      delegating `compute_impedance_swept`; `compute_y_matrix_swept` raises
      until Phase 2). Geometry fixtures for the matrix live in
      `tests/fixtures_refl_coef_geoms.py` (via `scripts/dump_refl_coef_geoms.py`).
- [x] Φ-term weighting: implement candidates (1)–(3); compare each against
      the golden NEC gn 0 references across the height sweep; pick one and
      document the choice + residuals here.
      → Winner: **ρ_v at normal incidence, (√ε̃−1)/(√ε̃+1), constant per
      ground** (`ground_phi_mode="normal"`, the default). Max/mean |ΔZ| vs
      gn 0 over the dipole acceptance window: normal 2.45/1.75 Ω,
      specular ρ_v 6.97/2.87 Ω, blend 7.13/3.27 Ω, quasi-static image
      11.70/4.82 Ω, PEC solve 41.2/19.4 Ω. Physics: the charge term is
      dominated by near-vertical (close-pair) specular rays; per-pair ρ_v
      is poisoned by grazing pairs where the plane-wave TM coefficient
      flips toward −1, while the quasi-static (ε̃−1)/(ε̃+1) overweights.
- [x] **Acceptance:** |ΔZ| vs NEC gn 0 ≤ ~2 Ω across 0.1–0.5λ heights on the
      dipole cases, and strictly better than the PEC-image solve everywhere
      in the window. Below 0.1λ: report, don't gate (NEC gn 0 is itself
      shaky there).
      → Met, with two honest footnotes. (a) Dipole window max is 2.45 Ω at
      0.1λ/εr=3 — marginally over 2, but the oracle is itself shakiest
      there (gn 0 vs gn 2 disagree by 7.8 Ω on that case) and the momwire-
      vs-NEC cross-solver floor (PEC-vs-PEC baseline) is already ≈ 1.4 Ω;
      every other dipole window case is ≤ 1.83 Ω. (b) "Strictly better than
      PEC" holds on every dipole window case; on inverted-L at 0.2–0.35λ
      the finite solve ties PEC within the cross-solver floor (PEC residual
      1.2–1.9 Ω there because the inverted-L feed Z is barely ground-
      sensitive at height — nothing to correct). 0.05λ report: normal
      max 11.6 Ω vs PEC 61.5 Ω, and we inherit gn 0's low-height divergence
      as hoped (secondary gn 2 check).

### Phase 2 — productionize (bspline)
(done 2026-07-06. PR #101 was declared abandoned before this phase — the
swept/assembly paths were edited directly, no rebase coordination.)
- [x] Swept/batched path: per-k weight tables; memory check on 41-freq
      sweeps.
      → `compute_y_matrix_swept` assembles the weighted image inline per k;
      `compute_impedance_swept` inherits via its per-k delegation. The
      k-independent specular tables are cached per geometry
      (`_image_refl_prep`); only ε̃(ω) → ρ tables (≈3 ms/k at N=180) are
      per-frequency. Memory: weight tables are two (N, N) complex arrays
      rebuilt per k, nothing accumulates across the sweep — 41-freq N=180
      peak RSS ≈ 180 MB, indistinguishable from the PEC sweep
      (`scripts/perf_refl_coef_sweep.py`).
- [x] Junction/KCL assembly variant.
      → Free: the weighted image is assembled at the basis level before the
      KCL Schur solve, so junction geometries flow through unchanged
      (inverted-L runs in the Phase 1 matrix and tests).
- [x] C++ assemble extension (second weight table) only if the Python image
      assembly shows up in profiles.
      → It did: numpy einsum assembly measured ~33 ms/k at N=180 (~3× the
      image J fill; +40% on a grounded 41-freq sweep). Added
      `assemble_Z_bspline_weighted` (complex w_A on the A term, complex w_Φ
      on the Φ term, same loop structure as the PEC kernel) → ~3 ms/k
      including the per-k ρ tables; grounded-sweep cost is now at parity
      with the PEC image path. The numpy loop remains as the
      no-accelerator fallback, guarded by a bit-exactness test.
- [x] Tests: golden-value guards, PEC-limit check (ε̃→∞ reproduces the PEC
      image path to ~1e-6), free-space regression (ground_eps=None bit-exact).
      → `tests/test_refl_coef_ground.py` (19 tests): golden-window guards +
      better-than-PEC, PEC-limit collapse at ε̃=1e16 to <1e-5, free-space
      bit-exactness, tuple-vs-complex ε̃ equivalence, y-matrix/impedance and
      swept/single-k consistency, accel-vs-numpy reference.

### Phase 3 — antennaknobs wiring
(done 2026-07-06: momwire v0.4.0 released — PR #115, rebase-merged, tag on
`92b13f9`; antennaknobs PR #252. Includes the fast-solver gate fix: HMatrix/
ArrayBlock had grown per-block PEC image support since this plan was
written, so their `_hmatrix_unsupported` gates were widened to send
`ground_eps` down the dense path — see "Architecture fit" update above.)
- [x] `MomwireEngine` ground-spec mapping (`finite-fast` → refl-coef solve;
      decide whether `finite` also upgrades — recommend yes).
      → Both upgraded, for solver classes in `_GROUND_EPS_SOLVERS`
      (BSpline/HMatrix/ArrayBlock); Triangular/Sinusoidal keep the PEC fold.
- [x] antennaknobs consumes momwire via a **git submodule**
      (`antennaknobs/momwire`, editable-installed into its .venv) pinned to a
      release tag, plus a `momwire>=` version floor in its dependencies. New
      solver API ⇒ release a momwire version, bump the submodule pin AND the
      version floor **in the same antennaknobs PR** (lesson from the
      v0.13.0/momwire-0.3.0 breakage).
      → Pin v0.4.0 + floor `momwire>=0.4.0` in PR #252. (Note antennaknobs
      CI does `git submodule update --remote`, i.e. tests momwire main tip,
      so the pin only governs local dev checkouts.)
- [x] Mirror antennaknobs `tests/test_pynec_ground.py` with a
      momwire-vs-PyNEC-gn0 cross-check.
      → `tests/test_momwire_finite_ground.py`: mapping unit tests + numeric
      gn 0 cross-check at 0.2λ (≤ 2.5 Ω, strictly better than PEC).
- [x] Web: no adapter changes expected (eps fields already spec-driven);
      consider whether momwire backends get their own ground-model wording
      in the UI tab summary.
      → Deliberately unchanged. The web momwire path (`_ground_for_engine`)
      still maps the ground toggle to `"pec"`: switching it to the finite
      spec would change served results AND interacts with the frontend's
      own Fresnel far-field treatment (`_PEC_GROUND_EPS_R` plumbing) —
      upgrading the web ground model is a separate decision, tracked as
      follow-up, not a Phase 3 side effect.

### Phase 4 — optional / deferred
- [ ] SinusoidalSolver vector-field tensor variant (C++), or explicitly
      document sinusoidal ground as PEC-image-only.
      → Superseded by the concrete Phase 6 plan below (numpy image path,
      no C++ change needed).
- [ ] TriangularSolver: intentionally skipped (retirement).

### Phase 5 — ground_eps in the fast solvers (added 2026-07-06)

Phase 3 gated HMatrix/ArrayBlock to the dense path under `ground_eps`
(their per-block image terms bake unweighted PEC physics). That's correct
but forfeits the fast solvers exactly where finite ground matters most in
practice: large HF arrays over lossy earth (four-square, phased verticals).
Next step: teach both fast paths the weighted image.

Two structural simplifiers:
- The chosen Φ weight (`ground_phi_mode="normal"`) is a **constant complex
  scalar** per (ground, frequency) — the image Φ term is a scalar multiple
  of the PEC-style image Φ term, no per-pair table needed.
- The image is always well-separated from the sources (mirror across the
  plane), so no new singular/near cases arise; w_A is a smooth closed-form
  function of pair geometry.

(done 2026-07-06)
- [x] **ArrayBlockSolver first** (easier, biggest payoff). Block reuse
      survives: w_A depends only on horizontal displacement and the two
      heights via the specular ray — exactly what the grounded block key
      (relative displacement + centroid heights) already captures. Only the
      per-block image fill changes from PEC mirror-dot to the Fresnel dyad
      weighting; the reuse/dedup machinery is untouched.
      → Self blocks via `_zblock_image_refl` (numpy weighted block
      assembly); coupling blocks via the shared refl ACA evaluators. Reuse
      verified: single-height 4-element grid still builds one self-block +
      3 coupling ACAs under ground_eps. Module-scope cache keys
      (`_self_block_key`, `_build_operator`) grew a (ground_eps, phi_mode)
      component so PEC and finite blocks never alias.
- [x] **HMatrixSolver**: extend the C++ off-edge block assembler (ACA entry
      evaluators + dense near blocks) to compute the Fresnel dyad in-kernel
      from pair geometry + ε̃. The weight is smooth, so ACA compressibility
      should be essentially unaffected — but verify far-block rank growth
      empirically before trusting it (add a rank-vs-PEC assertion to the
      hmatrix tests).
      → `bspline_assemble_offedge_block_refl`: the fused kernel templated
      on WEIGHTED, computing the dyad from the pre-mirrored inputs alone
      (obs→image Δ gives cos θ + incidence plane; the mirrored tangent dot
      IS td_img — the kernel never needs ground_z). Φ weight passed as
      w_Φ = c0 + c1·ρ_v, covering every phi mode (`phi_mode_coeffs`,
      consistency-tested against the numpy tables). Rank growth measured:
      **1.0× vs PEC** (72 = 72 total far rank on the 200-seg vertical
      dipole) — the smooth weights add nothing, as hoped. numpy fallback
      evaluators (`_zblock_image_refl`) agree with the C++ kernel to 1e-12.
- [x] Re-narrow the `_hmatrix_unsupported` gates as each solver lands, with
      fast-vs-dense equality tests under `ground_eps` (mirror
      `test_fast_solvers_fall_back_to_dense_with_ground_eps`, flipped).
      → Both gates back to enrichment-only. Equality: H-matrix/ArrayBlock
      reconstruct the dense refl-coef Z to <1e-4 (matvec, to_dense,
      impedance, y-matrix paths); 48-element grounded array solves 1.2×
      faster than dense at N=720 with 2e-6 relative agreement, matching
      the PEC fast path's scaling profile.

### Phase 6 — refl-coef ground in SinusoidalSolver (planned 2026-07-06)

SinusoidalSolver is the reference engine — performance parity with the
BSpline d=1/d=2 paths is explicitly NOT a goal. It is acceptable to slow
the grounded-finite solve down to numpy-fill speed; the design below in
fact leaves the free-space and PEC paths completely untouched (they keep
the C++ `sinusoidal_field_tensor` kernel) and only the *image block under
`ground_eps`* runs through the pure-numpy path.

**Key insight — the Φ-mode problem does not exist here.** The sinusoidal
solver is field-based: Eqs 76–79 give the *total* E-field of each source
shape (vector- and scalar-potential contributions already merged), which
is exactly the quantity NEC's EFLD weights. So unlike the mixed-potential
bspline path there is no image-charge weighting to approximate and no
`ground_phi_mode` knob: apply the NEC field dyad
`D_mn = ρ_v·(I − p̂p̂) − ρ_h·p̂p̂` (per-pair, at the specular angle of the
image-midpoint → observer-midpoint ray — the same per-pair-constant
approximation NEC makes) to the image field *vector* at the observer,
then project onto the observer tangent. Since the basis is also NEC's
own, parity with gn 0 should be at least as good as bspline's 2.45 Ω
window max, plausibly sub-Ω.

**Where it slots in.** `_field_tensor` (numpy fallback path,
`sinusoidal.py:694–826`) already has everything unprojected: per-shape
(E_z, E_ρ) scalars plus the source tangent and ρ̂ = rho_vec/rho_eval
directions; the tangential projection is its last three lines. The C++
kernel is the only place the vector field is discarded pre-projection —
so the finite-ground image block simply doesn't use it.

**Algebra.** With E the image-source field vector at observer m
(image source n: mirrored center, z-flipped tangent — existing
`_image_source_centers_tangents`):

    t_m · D · E = ρ_v·(t_m·E) − (ρ_v + ρ_h)·(t_m·p̂)(E·p̂)

where both scalars come from tables the numpy path already has:
  - t_m·E  = td·E_z + rho_proj_factor·E_ρ   (the existing projection)
  - E·p̂    = (t_n·p̂)·E_z + (ρ̂·p̂)·E_ρ      (t_img·p̂ = t_n·p̂; p̂ is
                                              horizontal)
PEC limit ρ_v→1, ρ_h→−1 collapses this to t_m·E — bit-identical in form
to the existing PEC image tensor, so the ε̃→∞ test carries over. The
vertical-ray degenerate case (observer above its own image) is benign for
the same reasons as in `_ground_refl`: p̂ falls back to x̂ where ρ_v=−ρ_h
makes the dyad isotropic, and rho_vec→0 kills the ρ̂·p̂ term.

Steps:
- [ ] `_ground_refl`: expose the specular-ray p̂ components (extend
      `specular_pair_tables` with an opt-in return or add a small
      `specular_ray_tables(centers, ground_z, src_centers)` returning
      (cos_th, px, py)). Bspline callers unchanged.
- [ ] Refactor the numpy `_field_tensor` into
      `_field_components(geom, k, src_c, src_t)` returning the per-shape
      (E_z, E_ρ) tables plus the projection geometry (td,
      rho_proj_factor, rho_vec, rho_eval); the numpy branch of
      `_field_tensor` becomes a thin projection wrapper. Guarded by the
      existing `test_sinusoidal_field_tensor_cpp_matches_numpy`.
- [ ] `_field_tensor_image_refl(geom, k)`: mirrored sources →
      `_field_components` → ρ_v/ρ_h from ε̃(self.omega) via
      `eps_tilde`/`fresnel_rho` → dyad → tangential projection.
      Cache the k-independent specular tables per geometry (identity
      check, same pattern as `_cached_basis` / bspline's
      `_image_refl_prep`).
- [ ] API + gate: `SinusoidalSolver(..., ground_eps=None)` (complex ε̃ or
      (eps_r, sigma) tuple; requires `ground_z`, same validation as
      bspline; deliberately NO `ground_phi_mode` — document why in the
      docstring). `_assemble_Z` subtracts the weighted image tensor
      instead of the PEC one when set. Sweeps work for free: both swept
      loops update `self.omega` per k before calling `_assemble_Z`, so
      ε̃/ρ tables are per-frequency automatically.
- [ ] Update the module docstring scope list (currently "no
      finite/Sommerfeld ground").
- [ ] Tests (`tests/test_sinusoidal_refl_coef_ground.py`), reusing
      `fixtures_refl_coef_geoms.py` + `golden_refl_coef_ground.py`
      (constructor kwargs are interface-compatible):
      free-space and PEC-ground bit-exactness with `ground_eps=None`;
      PEC-limit collapse at ε̃=1e16; tuple-vs-complex ε̃ equivalence;
      swept-vs-single-k consistency; golden-window guards vs gn 0 +
      strictly-better-than-PEC over 0.1–0.5λ; 0.05λ reported, not gated.
- [ ] **Acceptance:** first measure the sinusoidal-PEC vs PyNEC-PEC
      cross-solver floor on the golden matrix (expect tighter than
      bspline's ≈1.4 Ω — same basis as NEC), then gate at floor + ~1 Ω
      across the 0.1–0.5λ dipole window. Extend
      `scripts/compare_refl_coef_ground.py` (or a sibling) to print the
      sinusoidal residual table for this doc.
- [ ] antennaknobs follow-up (separate PR + momwire release): add
      `SinusoidalSolver` to `_GROUND_EPS_SOLVERS` in `MomwireEngine`;
      bump the submodule pin AND the `momwire>=` floor in the same
      antennaknobs PR (release discipline as in Phase 3).

Explicitly deferred: a C++ `sinusoidal_field_tensor_refl` kernel variant
(dyad in-kernel, mirroring `bspline_assemble_offedge_block_refl`). Only
worth it if grounded-finite sinusoidal solves ever show up in a profile —
they shouldn't, per the reference-engine framing above.

Estimate: 1–2 focused days (S1 physics ~1 day, tests/validation ~½–1 day).

## Validation matrix

Flat half-wave dipole (28.47 MHz, antennaknobs `dipoles.invvee:dipole`
geometry) at heights 0.05 / 0.1 / 0.2 / 0.35 / 0.5 λ; inverted-L
(`verticals.inverted_l`); ground constants (10, 0.002), (13, 0.005),
(3, 0.001). Oracle: PyNEC `("finite-fast", ...)`. Secondary sanity:
PyNEC `("finite", ...)` Sommerfeld, to confirm we inherit gn 0's known
low-height divergence rather than inventing our own.

## Risks

- **Φ-term weighting** is where the approximation error concentrates; expect
  the worst mismatch on structures with strong charge accumulation near
  ground (vertical wire ends, e.g. the inverted-L feed). If no candidate
  meets the bar, fall back to documenting refl-coef as "A-term only,
  Φ keeps PEC image" with measured residuals — still a strict improvement
  over today's full-PEC solve.
- **Churn collision with PR #101** on the bspline swept/assembly code paths.
  Rebase early, keep Phase 1 in separate new functions.
- **Performance:** grounded fill already ~doubles cost; per-k weight tables
  add memory on swept paths (mitigate by chunking over k, same as the
  batched swept plan does).

## Estimates

Phase 0: half a day. Phase 1: 1–2 focused days. Phase 2: 2–4 days.
Phase 3: ~1 day. Total ≈ 1–1.5 weeks part-time. Sommerfeld in momwire is
explicitly out of scope (weeks, different machinery).
