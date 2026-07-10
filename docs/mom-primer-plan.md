# MoM primer site plan — "A voltage in a gap"

A tutorial/primer website on how the method of moments works, keyed off the
actual momwire Python implementation, served at **momwire.antennaknobs.dev**.

## The framing

The whole site answers one question: *you put 1 volt across a tiny gap in a
wire; what current flows, and what does it radiate?* That single question is
literally what `BSplineSolver` computes, and it is strong enough to carry a
reader from Maxwell to H-matrices without ever feeling like a textbook.

**The signature device, on every page: "the idea in 20 lines, the real thing
in momwire."** Each chapter first builds a naive, runnable version of that
chapter's machinery (small enough to read whole), then shows the production
code and explains exactly what the engineering added — with GitHub permalinks
pinned to a release tag so line references never rot. momwire is unusually
good raw material for this: pure Python with the C++ as an optional mirror,
docstrings that already explain design decisions, and a real narrative history
to draw on (the triangular retirement, the Sommerfeld perf war, the ground
benchmark).

**Second through-line: one antenna, all the way down.** A half-wave dipole is
the specimen for Acts I–III; it graduates to a yagi and then `bowtiearray2x4`
when the story needs scale. Every chapter ends with a `pip install momwire`
snippet that reproduces its headline figure, and a "turn the knob yourself"
link into the live simulator at app.antennaknobs.dev.

**Audience**: hams and EE-adjacent readers who are comfortable with Python and
numpy. Maxwell stays light; the spine is "integral equation → linear algebra".
Every numeric claim in the prose comes from an actual run — same discipline
as the benchmark docs.

## The arc — four acts

### Act I — From a wire to a matrix (the core idea)

1. **The question.** Fields of a current filament, the thin-wire
   approximation, and why the answer is an integral equation: the current
   everywhere determines the field everywhere, but the field on the wire
   surface is constrained (E_tan = 0 on a conductor, except in the gap).
   The `a²` regularization in the kernel is introduced here as "how a thin
   wire avoids dividing by zero" — keyed to the kernel evaluation in
   `bspline.py` / `sinusoidal.py`.
   *Figures*: a dipole with the gap zoomed; |E| of a short current filament;
   the thin-wire kernel R(s, s′) sketch showing the a² floor.

2. **You can't solve for a function, so solve for coefficients.** Basis
   functions, testing → `Z I = V`. The ~40-line toy: pulse basis on the
   **2n+1-point grid** (n+1 endpoints carry charge, n midpoints carry
   current), mixed-potential form with the analytic log self-term — Harrington's
   classic straight-wire example. The pulse basis *works* on the real thin
   specimen (staircase current sits on momwire's smooth one); the crack — slow
   convergence, and *why* — is chapter 3's cliffhanger, not a strawman. (The
   naive point-matched-Pocklington variant that famously *doesn't* converge is
   named and left to the literature.)
   *Figures*: pulse basis on the 2n+1 grid (midpoints=current, endpoints=charge)
   + staircase vs smooth; toy current on momwire's on the thin specimen; the Z
   matrix heatmap (log |Z_mn|, diagonal dominance → Act IV).

3. **The feed and the answer.** Delta-gap source, what V actually is, drive-
   point impedance Z_in = V/I(feed), and why that number is what hams care
   about (SWR, resonance). Keyed to `compute_impedance` and the delta-gap RHS.
   Then the honest convergence story: the pulse *converges* but slowly and
   unevenly — R first-order (Richardson-extrapolates dead on), X on a
   log-corrected trend `X∞ + (b + c·lnN)/N` that extrapolates to ~−17 and lands
   ~1 Ω short (the crude delta-gap feed, an X-only offset); brute force runs
   into the kernel's own `dz > 8a` limit. The deep "why" (discontinuous current
   → concentrated endpoint charge → the scalar potential that carries the
   reactance) is *teased* here and *paid off* in Act II — continuity is the fix.
   *Figures*: toy R and X vs N with the fitted X-trend + extrapolated asymptotes
   vs momwire and the `dz=8a` wall; R+jX across a frequency sweep through
   resonance for the specimen dipole.

### Act II — Bases and accuracy (where the craft lives)

4. **Sinusoids, NEC's bet** — `sinusoidal.py`. Why a three-term sinusoidal
   basis fits wire currents so naturally, and why sharing NEC's basis makes
   it momwire's sharpest cross-validator (the ~0.1 Ω Sommerfeld agreement).
   *Figures*: the three-term basis shape; current on the dipole with 5
   segments — sinusoidal nearly exact, splines needing more.

5. **Splines and junctions** — `bspline.py`. Tents (d=1), quadratics (d=2),
   KCL at junctions, multi-wire polylines. The triangular-retirement story
   told straight: "d=1 *is* the tent basis, to roundoff" (pins in
   `tests/test_tent_parity.py`) — a real refactor as a lesson about basis
   equivalence.
   *Figures*: tent vs quadratic B-spline shapes on a knot grid; a junction
   (three wires meeting) with the KCL constraint drawn; d=1-vs-triangular
   parity plot.

6. **Integrals done honestly** — `_quadrature.py` (all 30 lines of it),
   `n_qp_pair` / `n_qp_source`, singularity handling via precomputed static
   moments (`_bspline_static_moments.py` and
   `scripts/derive_bspline_static_moments.py`). "MoM is 90% quadrature
   engineering."
   *Figures*: integrand of a same-segment pair (nasty) vs a far pair
   (smooth); error vs quadrature order for both, showing why the smooth/
   singular split exists.

7. **How do you know it's right?** Convergence sweeps, the knee, cross-engine
   validation against NEC-2 — pulling from `docs/convergence_analysis.md`
   and the antennaknobs benchmark methodology. This chapter is the site's
   credibility anchor.
   *Figures*: R+jX vs N with the knee annotated; momwire-vs-NEC residuals
   across the 10 benchmark designs.

### Act III — The ground (the physics deepens)

8. **Mirror worlds** — PEC ground and the method of images: one extra kernel
   term, nearly free. Keyed to the image handling in the solvers
   (`ground_z`).
   *Figures*: dipole + image; impedance vs height over PEC (the classic
   oscillation).

9. **Real dirt, cheap** — `_ground_refl.py`: Fresnel reflection coefficients
   as an approximation, where it's good, and where it's tens of ohms wrong
   (the sub-0.1λ story).
   *Figures*: Γ(θ) for real ground; refl-coef vs Sommerfeld impedance vs
   height showing the low-height divergence.

10. **Sommerfeld, or paying full price** — `_sommerfeld.py`: exact-image-plus-
    remainder decomposition, the interpolation grid, and the performance
    narrative (bowtie·somm·Bs2 500 s → 34 s) as a case study in making
    rigorous physics affordable.
    *Figures*: the remainder-field grid; timing bars before/after the fused
    kernel (from the ground-model benchmark).

### Act IV — Scale (from physics to numerics)

11. **N² is the enemy** — dense fill/factor cost, frequency sweeps,
    k-batching under a memory budget (`compute_impedance_swept`,
    `swept_mem_mb`, `_swept_batched_z_chunks`).
    *Figures*: measured fill+factor time vs N on the specimen; sweep time
    per point vs single-solve time.

12. **Matrices that are secretly small** — `_aca.py` and `hmatrix.py`: why
    far-apart wire chunks interact through a low-rank block, ACA as "peeling
    the matrix," the rhombic as the demo.
    *Figures*: singular-value decay of a far block; the H-matrix block
    partition drawn over the Z heatmap from chapter 2; rhombic time vs N,
    dense vs ACA.

13. **Arrays know their own symmetry** — `array_block.py`: identical elements
    → repeated blocks, the LPDA beating the NEC reference by 12×.
    *Figures*: the block structure of an array Z; lpda/bowtiearray timing
    bars from the benchmark.

14. **Epilogue: the same math, twice** — `_accel.py` and the C++ kernels as a
    compiled mirror of the Python spec; cooperative cancellation
    (`_cancel.py`) as what "production solver" means. Ends by pointing back
    at the simulator: everything you just read runs when you drag a knob.

Emotional shape: Act I gives the reader a working solver they wrote
themselves; Act II earns their trust; Act III raises the physics stakes;
Act IV turns the story from physics into engineering — which is momwire's
actual differentiator.

## Pedagogical / production rules

- **Idea-then-real-thing**: every chapter pairs a minimal runnable
  implementation with permalinks into the pinned momwire source.
- **Pinned permalinks**: all source links use a release tag
  (`blob/vX.Y.Z/...#L…`). A build-time check greps the site for pinned line
  refs and verifies the referenced lines still match the tag, so a refactor
  can't silently strand the prose.
- **Generated figures**: every figure comes from a checked-in script under
  `site/figures/*.py` (running against the repo venv), committed alongside
  its output, so figures regenerate on engine changes instead of rotting.
- **Verified numbers**: any timing/impedance number in the prose is produced
  by a figure script or a quoted benchmark doc, never typed from memory.
- **Runnable endings**: each chapter ends with a copy-paste
  `pip install momwire` snippet reproducing the headline figure.

## Infrastructure

- **Location**: `site/` in this repo, Astro + Starlight — the same stack as
  the antennaknobs docs site (`@astrojs/starlight` ^0.34, `astro` ^5.6), so
  the two sites share conventions and maintenance knowledge.
- **Deploy**: a third Fly app, `momwire-docs`, static-served like
  `antennaknobs-docs`; cert for `momwire.antennaknobs.dev` (one CNAME at the
  registrar — the domain is already ours). A `deploy-docs.yml` workflow in
  this repo fires on `v*` tags (and `workflow_dispatch`), so the tutorial
  versions with the code it documents.
- **Cross-linking**: the antennaknobs docs get one LinkCard ("How the solver
  actually works →" → momwire.antennaknobs.dev); the primer's header links
  back to the simulator and the antennaknobs docs.

### Bootstrap checklist

- [x] `site/` scaffold (Astro + Starlight, dark theme to match), landing
      page with the four-act table of contents.
- [x] Sidebar nav: Acts as groups, chapters as pages; unwritten chapters
      omitted until they land (no stub pages).
- [x] `site/figures/` runner convention + matplotlib style (one shared
      `style.py`).
- [x] Permalink-check script (`scripts/check_site_permalinks.py`) with a
      `site/permalinks.json` manifest (expected substring per line-anchored
      link); runs in the deploy workflow (not `npm run build` — the Docker
      build stage has no Python or git history).
- [x] Fly app config: `site/fly.toml` + `deploy-docs.yml` (token-gated
      skip like antennaknobs, so CI is green before the secret exists).
      Still manual: `fly apps create momwire-docs` + first `fly deploy` +
      the FLY_API_TOKEN_DOCS repo secret.
- [ ] `fly certs add momwire.antennaknobs.dev` + registrar CNAME (manual,
      after first deploy).
- [ ] LinkCard from the antennaknobs docs (separate antennaknobs PR, after
      the primer has an Act I to link to).

### Build order

1. Plan (this doc) + site scaffold + Act I (chapters 1–3) — proves the
   format end to end, including figure scripts and the permalink check. **DONE.**
2. First deploy + domain + antennaknobs cross-link. *Still pending (manual:
   `fly apps create momwire-docs`, cert, registrar CNAME, FLY_API_TOKEN_DOCS).*
3. Act II (4–7), Act III (8–10), Act IV (11–14). **DONE** — all 14 chapters
   written with verified figures from real momwire runs; every source link
   pinned + permalink-checked. Note: Act I ch2–3 were rewritten mid-stream when
   the original "pulse basis fails" framing turned out to be wrong (the toy was
   a broken bare-Pocklington point-match); the corrected pulse toy is the
   Harrington 2n+1 mixed-potential scheme, which converges. The triangular-
   retirement story (planned for ch5) was dropped as momwire-internal.
