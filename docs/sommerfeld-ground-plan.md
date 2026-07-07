# Sommerfeld/Norton ground in BSplineSolver — plan

**Goal:** implement NEC's Sommerfeld/Norton finite ground (`gn_card(2)`,
antennaknobs `("finite", εr, σ)`) in momwire's `BSplineSolver` impedance
solve. Today's `ground_eps` refl-coef model (docs/refl-coef-ground-plan.md,
complete) matches NEC gn 0 to the cross-solver floor — but gn 0 is itself an
approximation that fails close to ground: at 0.05λ the dipole goldens show
gn 0 and gn 2 disagreeing by **~22 Ω** (εr=10: 46.9+9.3j vs 68.0+1.6j), and
at 0.02λ by **>130 Ω** (70+163j vs 95+32j — gn 0 isn't even in the right
regime). Sommerfeld is the exact half-space solution; landing it removes
the "unreliable below 0.1λ" caveat and covers the antennas where ground
matters most: low wires, verticals, radial-less end-fed types.

This doc is the working memory for the effort, in the same format as the
refl-coef plan. Everything below the "What NEC gn 2 actually does" section
is grounded in a fresh read of `docs/nec2_theory_manual.pdf` §IV.1–IV.2
(pp. 38–55) — equation numbers refer to that manual.

---

## Why

- The refl-coef program's own validation motivated this: the 0.05λ rows of
  `tests/golden_refl_coef_ground.py` were "report, don't gate" precisely
  because gn 0 diverges from the exact model there. We already carry the
  gn 2 oracle values (`"finite"` key) for all 30 cases — Phase 0 is mostly
  done by accident.
- antennaknobs maps `("finite", εr, σ)` — the spec that *means* Sommerfeld
  in NEC vocabulary — to the refl-coef solve today, documented as "momwire's
  best available finite model". This plan makes that mapping honest.
- The refl-coef plan closed with "Sommerfeld in momwire is explicitly out of
  scope (weeks, different machinery)". The machinery estimate shrank on
  reading the manual: NEC's own decomposition routes the singular part
  through the image machinery momwire already has (see below), leaving one
  genuinely new component — the Sommerfeld-integral engine + interpolation
  grid.

## What NEC gn 2 actually does (theory manual §IV.1–IV.2)

The field over ground of an infinitesimal current element (eqs 118–122) is
split (eqs 136–147) into three parts:

1. **Free-space term** (the G₂₂ = e^{−jk₂R₂}/R₂ terms) — the existing
   free-space machinery, untouched.
2. **Exact image term scaled by a constant**: the G₂₁ = e^{−jk₂R₁}/R₁
   image terms carry the coefficient

       C₂ = (k₁² − k₂²)/(k₁² + k₂²) = (ε̃ − 1)/(ε̃ + 1),
       k₁² = k₂²·ε̃,   ε̃ = εr − jσ/(ωε₀)

   This is the full image field of the element multiplied by one complex
   constant per (ground, frequency). It absorbs **all** the singular
   (1/R₁³ … 1/R₁) behavior of the ground field. NEC evaluates it with its
   free-space routines on the image; momwire evaluates it with
   `assemble_Z_bspline_weighted` and *constant* weight tables (below).
3. **Smooth remainder** F (eqs 143–147): four independent field functions
   of the source element — F_ρ^V, F_z^V, F_ρ^H, F_φ^H, with
   F_z^H = −cosφ·F_ρ^V (eq 147) — built from two Sommerfeld integrals
   U′₂₂, V′₂₂ and their ρ/z derivatives (eqs 148–155). The integrands
   contain D₁, D₂ (eqs 154–155) which vanish identically in **both** the
   PEC limit (ε̃→∞) and the free-space limit (ε̃→1) — two exact
   engine-level test oracles.

For interpolation NEC removes the residual 1/R₁ behavior and free-space
phase by tabulating I = C₁·R₁·e^{+jk₂R₁}·F with the sinφ/cosφ factors
stripped (eqs 156–159), on grids in **(R₁, θ)** where R₁ = |obs − image
point| = √(ρ² + (z+z′)²) and θ = tan⁻¹((z+z′)/ρ). R₁→0 limits are analytic
functions of θ (eqs 169–172). Grid layout (fig 12; bivariate cubic on 4×4
neighborhoods, measured rel. error 1e−3…1e−4):

| Grid | region              | ΔR₁    | Δθ  |
|------|---------------------|--------|-----|
| 1    | R₁ ≤ 0.2λ           | 0.02λ  | 10° |
| 2    | 0.2–1.0λ, θ ≤ 20°   | 0.05λ  | 5°  |
| 3    | 0.2–1.0λ, θ > 20°   | 0.1λ   | 10° |

Caveat from the manual: low-loss high-εr ground puts an evanescent
interface wave into the surfaces (fig 11, εr=16 σ=0) whose oscillation
scale is ~λ/√εr — grid 2 needs ΔR₁ keyed to |k₁|, not k₂.

The λ-integrals are evaluated on deformed contours in the complex plane
(figs 13–15): Bessel-function form (J₀, contour break at p+jp,
p = min(1/ρ, 1/(z+z′))) when ρ < (z+z′)/2, else Hankel form (H₀⁽²⁾,
descending contour with slope tan⁻¹(ρ/(z+z′)), detours around k₂ and k₁);
adaptive Romberg with Shanks acceleration on the tails. Branch points at
±k₁, ±k₂ with vertical cuts; no poles on the primary sheet, but a near
singularity at k₂ when ε̃→1 forces the contour off the real axis there.

Beyond R₁ = 1λ NEC switches to Norton's asymptotic approximations
(subroutine GWAVE) and treats each source segment as a lumped moment.
NEC's measured fill cost with interpolation: ~4× free space.

## Architecture fit

### The image part is already built

`_image_Z_refl` (`bspline.py:790`) assembles a weighted image from per-pair
tables (w_A, w_Φ) through the C++ `assemble_Z_bspline_weighted` kernel. The
Sommerfeld image part is that same call with **constant** tables:

    w_A = C₂ · td_img      (td_img = existing PEC mirror tangent-dot)
    w_Φ = C₂

Note the irony: C₂ = (ε̃−1)/(ε̃+1) is exactly `phi_mode_coeffs("image")`
(`_ground_refl.py:199`) — the quasi-static image coefficient that *lost*
the refl-coef Φ-mode bake-off returns here as the exact coefficient of the
singular part. No approximation this time: the remainder block supplies
what the scaled image lacks.

### The remainder block is new — field-form Galerkin quadrature

The F remainder is a *field* dyad, so it is tested directly
(⟨f_m t_m, E⟩), not via the mixed-potential split — adding a field-form
block to Z is legitimate because the operator is additive and the
integration-by-parts in Z_Φ only ever applied to the free-space/image
kernels. Since F is the field of an infinitesimal current element
(endpoint charges included), integrating it over the source basis current
gives the complete remainder — no separate charge term exists:

    Z_S[m,n] = Σ_{wings a,b} ∫∫ f_m(u) f_n(u′) · t_a · F(r_a(u), r_b(u′)) · t_b du′ du

by Gauss quadrature (q ≈ 2–4 points/segment/side; the kernel is smooth on
the segment scale because R₁ ≥ z+z′ ≥ 2·min-height and the 1/R₁ factor is
mild — same class as the image J fill). Per node pair: decompose t_b into
vertical + horizontal parts, get (R₁, θ, φ) from geometry, interpolate the
four I surfaces, multiply back e^{−jk₂R₁}/R₁ and the sinφ/cosφ factors,
combine into the Cartesian dyad, project on t_a. All vectorizable over the
(N_seg·q)² node-pair set (chunk over observer segments at large N).

Basis values at the quadrature nodes come from `polys` (the same
per-wing coefficient tensor `_assemble_Z` consumes), so junction/KCL
geometries flow through unchanged — the block is assembled at basis level
before the Schur solve, exactly like `_image_Z_refl`.

### Insertion points

The four `ground_eps` seams route through one more branch,
`ground_model="sommerfeld"`:

- `compute_impedance` (`bspline.py:1505–1518`)
- `compute_y_matrix` (`bspline.py:1651–1656`)
- `compute_y_matrix_swept` per-k loop (`bspline.py:1734–1742`) —
  `compute_impedance_swept` inherits via delegation
- assembly: `Z = Z_free − C₂·Z_img − (−1)·Z_S` — the exact signs pinned by
  the PEC-limit test (ε̃→∞: C₂→1, Z_S→0 must reproduce the PEC image to
  rounding) rather than derived on paper here.

k-independent per-geometry precompute (node positions, ρ, z+z′, R₁, θ,
tangent decompositions, φ factors) cached like `_image_refl_prep`
(`bspline.py:760`); per-k work = grid fill + interpolation + combine. Both
swept loops update `self.omega` per k before assembling, so per-k ε̃/grids
happen automatically — same free ride the refl-coef path got.

### Fast solvers / enrichment / triangular

- HMatrix/ArrayBlock: widen `_hmatrix_unsupported` to send
  `ground_model="sommerfeld"` down the dense path (the Phase-3 refl-coef
  pattern; correct at dense cost). Fast-path support is future work — the
  smooth remainder should ACA-compress fine, but it's not this plan.
- `use_singular_enrichment + ground_z` stays rejected (`bspline.py:223`).
- TriangularSolver: skipped (retirement-bound). SinusoidalSolver: natural
  follow-on (its field-based numpy image path would consume the same
  interpolated F dyad and its 0.1 Ω floor would show Sommerfeld parity
  much sharper than bspline's ~1.5 Ω floor) — deferred, own phase, only
  after bspline lands.

## API sketch

```python
BSplineSolver(..., ground_z=0.0,
              ground_eps=(13, 0.005),
              ground_model="refl-coef")  # default; "sommerfeld" opts in
```

- `ground_model` is keyword-only, validated against
  `("refl-coef", "sommerfeld")`, meaningful only with `ground_eps` set
  (error otherwise). Default preserves every existing test bit-exactly.
- `ground_phi_mode` applies to refl-coef only (documented; sommerfeld has
  no Φ knob — the image coefficient is exact).
- Same `ground_eps` value forms (complex ε̃ or `(eps_r, sigma)`), via the
  existing `eps_tilde`.
- Restriction, same as NEC: all wire z strictly above `ground_z`
  (z + z′ > 0; a wire touching ground needs a ground-stake model neither
  code has).

## Phases

### Phase 0 — scaffolding & goldens (done 2026-07-06)
- [x] Goldens: extended `scripts/capture_refl_coef_ground_golden.py` +
      `scripts/dump_refl_coef_geoms.py` with 0.02λ heights (both
      geometries) and a 6-element flat yagi (beams.yagi, n_directors=4,
      >1.2λ boom) at 0.2λ — 39 cases / 13 geometries regenerated; all
      pre-existing gn 0 and PEC values reproduced byte-identically; both
      refl-coef test suites still green (57 passed).
- [x] **Finding — the gn 2 oracle is nec2c, NOT PyNEC.** Regenerating
      goldens exposed two independent PyNEC (nec2++) Sommerfeld defects,
      confirmed by controlled experiments (details in the capture-script
      docstring):
      1. *Cross-solve state*: the same gn 2 case solved twice in one
         process returns different impedances (62.791−2.173j then
         62.308−2.532j); a preceding 0.02λ solve shifts the next answer
         by 17 Ω. The two pre-existing golden gn 2 values that were each
         geometry's first-in-process solve changed when run order
         changed — that's what unmasked this.
      2. *Low-height breakage*: even first-in-process, PyNEC gn 2 at
         0.02λ is erratic across the three similar matrix grounds
         (572−227j / 216−783j / 78+43j) where nec2c varies smoothly
         (95+32j / 91+30j / 93+31j) and is segment-refinement stable
         (0.8 Ω under 2× segments).
      Controls: identical exported decks agree nec2c-vs-PyNEC to 0.02 Ω
      on gn 0/PEC at all heights (deck translation faithful) and on gn 2
      at 0.1–0.5λ (both Sommerfelds healthy there); the split is only
      below 0.1λ — precisely the regime this plan targets. Capture now
      routes every gn 2 value through `antennaknobs.nec_export` + the
      `nec2c` CLI (fresh process per case as a bonus). Payoff visible in
      the new goldens: dipole 0.02λ/(10, 0.002) gn 0 = 70+163j vs
      gn 2 = 95+32j.
- [x] Pointwise oracles for the integral engine, independent of any MoM
      solve: manual figs 7–11 Max/Min literals →
      `tests/oracle_sommerfeld_figs.py` (with normalization caveats in
      its docstring). D₁ = D₂ = 0 identities at ε̃→∞ / ε̃→1 verified
      analytically (γ-limits) — encode as engine tests in Phase 1.
- [x] License note: implement from the theory-manual equations + the
      public-domain NEC-2 Fortran listing (SOMNEC/GWAVE, Part II) if
      needed for tie-breaking. Do NOT read GPL derivatives (nec2c, PyNEC
      sources); nec2c/PyNEC stay dev-time oracles behind capture
      scripts, as established (invoking the nec2c CLI at capture time
      links nothing).

### Phase 1 — Sommerfeld integral engine (done 2026-07-06)
`src/momwire/_sommerfeld.py`, pure numpy/scipy, no C++.
- [x] Integrand kit: γ via √(−j(λ−k))·√(j(λ+k)) (principal sqrts —
      NEC's vertical cuts down from +k / up from −k in one closed form;
      radiation branch on the real axis pinned by the identity test
      below), D₁/D₂ (eqs 154–155), the six λ-integrals (eqs 148–153) in
      a J₁/x formulation that is ρ→0-safe and satisfies the Laplacian
      identity f₀ + f₃ + λ²f₄ = 0 pointwise (unit-tested).
- [x] Contour evaluation: fig 13 Bessel + fig 14 Hankel with adaptive
      bisection Gauss (24-pt) on the waypoint sections and geometric-ramp
      panel tails. Three deviations from the manual's recipe, each
      earned empirically: (a) the Bessel domain is widened to ρ < 2h
      (its tail converges there at ~2× cost and has no λρ→0 pole),
      shrinking the delicate Hankel region; (b) tail panels ramp
      geometrically from the k-scale up to the 0.2π/max(ρ,h) asymptotic
      length — a single Gauss panel leaping |λ| ~ k → 1/R₁ was a 270×
      error at R₁ = 1e−4λ; (c) adaptive tolerance is RELATIVE (the
      small-|λρ| Hankel sections legitimately reach ~1/(k₂ρ)² before
      cancellation). No Shanks needed (exponential tail decay ≥ e^0.63
      per panel by construction); fig-15 contour variant not needed so
      far (the lossless εr=16 stress case validates on fig 14 — revisit
      only if a Phase-2 grid case fails).
- [x] F components → I surfaces (eqs 156–159) + analytic R₁→0 limits
      (eqs 169–172), continuous against direct evaluation to ≤ 1.4e−3
      at R₁ = 1e−4λ.
- [x] `iv_surfaces_direct(eps_t, k2, R1, theta)` point evaluator
      (~4 ms/point at rtol 1e−7 — a ~300-node Phase-2 grid fills in
      ~1.2 s single-threaded before any vectorization).
- [x] Tests → `tests/test_sommerfeld_engine.py` (27 tests, ~40 s):
      Sommerfeld identity on BOTH contour machines vs e^{−jkR}/R
      (≤1e−8, incl. h=0 and R₁=1e−4λ); Laplacian integrand identity;
      same-point J-vs-H cross-form agreement (strongest contour check —
      the paths share nothing); ε̃=1 → I ≡ 0 (identically, D₁=D₂=0);
      PEC ~1/√ε̃ scaling; R₁→0 continuity; ε̃=1+δ near-singularity
      stability; manual figs 7–11 extrema.
      **Figure-oracle outcome:** with the single pinned normalization
      (manual plots Iℓ = 1 A·wavelength → ×λ(10 MHz) = ×29.979), the
      extrema of all five published surfaces reproduce to ~4 significant
      digits where the mesh samples cleanly — e.g. I_z^V computed
      −16.31 / 219.9 / −98.16 vs manual −16.31 / 219.9 / −98.16;
      I_ρ^V −80.68/−137.88 vs −80.65/−137.9; fig 11's −166.2 exact —
      residual 6–15% gates cover mesh-sampling of the small-θ ridge
      only. The engine reproduces NEC's published Sommerfeld surfaces.

### Phase 2 — grid + bivariate interpolation
- [ ] NEC's three-region layout as the baseline, two modernizations:
      (a) grid R₁ extent sized to the geometry's actual max R₁ (computed
      from cached geometry) instead of hard 1λ + Norton — memory is not
      1979's constraint; (b) ΔR₁ in the R₁ > 0.2λ regions keyed to
      2π/|k₁| (the manual's own caveat for low-loss high-εr).
- [ ] Bivariate cubic interpolation, vectorized over query batches
      (Re/Im via two real splines per surface, or a hand-rolled 4×4
      NEC-style stencil — pick whichever benchmarks better at ~1e6
      queries).
- [ ] Accuracy harness: interp vs `eval_I_direct` at random (R₁, θ) per
      grid region; target ≤ 1e−3 rel (NEC's own bar).
- [ ] Grid-fill cost target: ≤ ~0.5 s per (ε̃, k) at default extent
      (~300 nodes × 6 integrals, vectorizable across nodes sharing a
      contour family). This is the per-k sweep cost — measure on the
      41-freq sweep before accepting.
- [ ] Norton/GWAVE asymptotic region: **deferred** unless the
      geometry-sized grid proves inadequate (huge arrays). Record measured
      interp error at the largest golden-case R₁ to justify.

### Phase 3 — BSplineSolver wiring
- [ ] `ground_model` kwarg + validation (API sketch above).
- [ ] Image part: constant-table variant of `_image_Z_refl` (w_A = C₂·td,
      w_Φ = C₂) — reuses `assemble_Z_bspline_weighted` and the numpy
      fallback unchanged.
- [ ] Remainder block `_image_Z_sommerfeld`: k-independent quadrature
      geometry cached per geometry; per-k grid fill + interpolated
      field-form Galerkin assembly (numpy). Chunk over observer segments;
      exploit Z_S symmetry (reciprocity) for a 2× if it matters.
- [ ] Wire the four seams; swept paths inherit per-k ε̃ for free.
- [ ] Tests (`tests/test_sommerfeld_ground.py`), reusing
      `fixtures_refl_coef_geoms.py` + goldens: free-space/PEC/refl-coef
      bit-exactness when not selected; PEC-limit collapse at ε̃=1e16;
      free-space-limit collapse at ε̃=1+1e−12 (Z → Z_free + 0·image);
      tuple-vs-complex ε̃; swept-vs-single-k (3-k middle entry);
      quadrature-order convergence (q vs q+2 stability); Z symmetry.

### Phase 4 — validation vs gn 2
- [ ] Extend `scripts/compare_refl_coef_ground.py` with a sommerfeld
      section: residual tables vs the gn 2 goldens, alongside the
      refl-coef-vs-gn 0 and PEC-floor sections.
- [ ] **Acceptance:** |ΔZ| vs NEC gn 2 within the bspline cross-solver
      floor + ~1 Ω across **all** heights *including 0.05λ and the new
      0.02λ rows* — the low-height cases are the entire point (gn 0 is
      ~22 Ω wrong at 0.05λ and >130 Ω at 0.02λ; we must land on gn 2's
      side of that gap, which
      also cleanly proves we implemented Sommerfeld and not refl-coef
      again). Sanity: agree with the refl-coef solve within ~2.5 Ω over
      0.2–0.5λ (where gn 0 ≈ gn 2). Record residuals here.
- [ ] Fill-cost measurement vs free space (NEC parity is ~4×; flag if
      grossly worse) and 41-freq sweep wall time.

### Phase 5 — fast solvers, docs, release
- [ ] Widen `_hmatrix_unsupported` in hmatrix.py/array_block.py:
      `ground_model == "sommerfeld"` ⇒ dense path, with a
      falls-back-to-dense test (the refl-coef Phase 3 pattern). Per-block
      fast support is explicitly out of scope.
- [ ] Module docstrings, this doc's checkboxes, README solver table if
      touched.
- [ ] momwire minor release (new public constructor parameter — same
      call as v0.5.0).

### Phase 6 — antennaknobs wiring (separate PR, after the release)
- [ ] `MomwireEngine`: map `("finite", εr, σ)` → `ground_model="sommerfeld"`
      for solvers that support it (BSpline initially; HMatrix/ArrayBlock
      inherit via the dense gate — decide whether silent dense fallback or
      refl-coef is the better default for them and document the choice).
      `("finite-fast", εr, σ)` stays refl-coef everywhere. Pin + floor in
      the same PR; wheel-smoke race discipline per memory.
- [ ] Web adapter: `ground_model_applied` gains a "sommerfeld" value for
      the bspline backend; site docs (reference/solver.md, web.md) update
      the ground-model story ("finite is now true Sommerfeld on the
      B-spline solver, matching NEC gn 2"); home-page what's-new box as
      part of the release ritual.
- [ ] Mirror test: momwire-vs-gn 2 cross-check at 0.05λ (the height
      where the mapping visibly changes results). Careful: PyNEC gn 2 is
      untrustworthy below 0.1λ and order-dependent in-process (Phase 0
      finding) — gate against captured nec2c literals, not live PyNEC.
      Also consider whether antennaknobs' own PyNEC `("finite", ...)`
      path needs a warning for sub-0.1λ geometries — the hosted
      simulator serves those solves from PyNEC today.

## Validation matrix

The refl-coef matrix extended to 2 geometries × 6 heights (0.02–0.5λ)
× 3 grounds plus the 6-element yagi at 0.2λ (span > 1λ for the large-R₁
grid region) — 39 cases, captured. Oracle: **nec2c** gn 2 via deck
export (NOT PyNEC — see Phase 0 finding). Secondary sanity: gn 0
agreement above 0.2λ, divergence below 0.1λ matching the known pattern.

## Risks

- **Contour numerics** are the concentrated risk: branch-point handling,
  the ε̃→1 near-singularity at k₂, and the low-loss evanescent tail
  (fig 11) are exactly where a hand-rolled integrator fails quietly. The
  figure-value oracles + limit identities + Bessel/Hankel cross-checks in
  Phase 1 exist to make failures loud before any MoM solve depends on it.
- **Interpolation error at large R₁** for low-loss high-εr grounds
  (oscillation ~2π/|k₁|). Mitigated by keying ΔR₁ to |k₁|; measured, not
  assumed.
- **Quadrature order at very low heights** (R₁ ~ 2h comparable to segment
  length): guarded by the q-convergence test; fallback is bumping q for
  near pairs only (per-pair distance already known from the cached
  geometry).
- **Sign/convention slips** between NEC's image convention and momwire's
  subtract-with-mirrored-tangents: pinned mechanically by the PEC- and
  free-space-limit tests, not by derivation.
- **Sweep cost**: a per-k grid fill is new per-k work with no PEC-path
  analogue. Budget ~0.5 s/k; if the 41-freq sweep hurts, vectorize the
  node loop before reaching for C++.

## Estimates

Phase 1 is the lift: 2–4 focused days. Phase 2: 1–2 days. Phase 3: 1–2
days (the image half is nearly free). Phases 4–5: ~1 day. Phase 6: ~1 day.
Total ≈ 1.5–2 weeks part-time — down from the refl-coef plan's "weeks,
different machinery" estimate, because the manual's C₂-image decomposition
lets the singular half of the problem reuse `assemble_Z_bspline_weighted`
as-is.
