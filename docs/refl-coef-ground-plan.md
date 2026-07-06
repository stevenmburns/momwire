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

### HMatrix / ArrayBlock — free

Both already fall back to the dense bspline path when ground is on; they
inherit the bspline work with zero changes.

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

### Phase 0 — scaffolding & golden references
- [ ] Capture golden Z values from `PyNECEngine(ground=("finite-fast", ...))`
      for the validation matrix below; store as literals in a momwire test
      module so momwire's suite needs no PyNEC dependency.
- [ ] Fix the stale `sinusoidal.py:11` "free-space only" docstring.
- [ ] Check PR #101 status; agree base/rebase strategy with that effort.

### Phase 1 — physics prototype (bspline, dense, single-k, Python assembly)
- [ ] Shared helper: per-pair specular geometry + ρ_v/ρ_h tables from ε̃(ω)
      (vectorized numpy; midpoint approximation, NEC-style).
- [ ] A-term dyad table replacing `_image_tangent_dot` when `ground_eps` set.
- [ ] Φ-term weighting: implement candidates (1)–(3); compare each against
      the golden NEC gn 0 references across the height sweep; pick one and
      document the choice + residuals here.
- [ ] **Acceptance:** |ΔZ| vs NEC gn 0 ≤ ~2 Ω across 0.1–0.5λ heights on the
      dipole cases, and strictly better than the PEC-image solve everywhere
      in the window. Below 0.1λ: report, don't gate (NEC gn 0 is itself
      shaky there).

### Phase 2 — productionize (bspline)
- [ ] Swept/batched path: per-k weight tables; memory check on 41-freq
      sweeps.
- [ ] Junction/KCL assembly variant.
- [ ] C++ assemble extension (second weight table) only if the Python image
      assembly shows up in profiles.
- [ ] Tests: golden-value guards, PEC-limit check (ε̃→∞ reproduces the PEC
      image path to ~1e-6), free-space regression (ground_eps=None bit-exact).

### Phase 3 — antennaknobs wiring
- [ ] `MomwireEngine` ground-spec mapping (`finite-fast` → refl-coef solve;
      decide whether `finite` also upgrades — recommend yes).
- [ ] antennaknobs consumes momwire via a **git submodule**
      (`antennaknobs/momwire`, editable-installed into its .venv) pinned to a
      release tag, plus a `momwire>=` version floor in its dependencies. New
      solver API ⇒ release a momwire version, bump the submodule pin AND the
      version floor **in the same antennaknobs PR** (lesson from the
      v0.13.0/momwire-0.3.0 breakage).
- [ ] Mirror antennaknobs `tests/test_pynec_ground.py` with a
      momwire-vs-PyNEC-gn0 cross-check.
- [ ] Web: no adapter changes expected (eps fields already spec-driven);
      consider whether momwire backends get their own ground-model wording
      in the UI tab summary.

### Phase 4 — optional / deferred
- [ ] SinusoidalSolver vector-field tensor variant (C++), or explicitly
      document sinusoidal ground as PEC-image-only.
- [ ] TriangularSolver: intentionally skipped (retirement).

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
