# Sommerfeld ground performance plan

Working-memory doc for making `ground_model="sommerfeld"` fast enough for
interactive use and sweeps. Successor to `sommerfeld-ground-plan.md`
(functionality shipped in v0.6.0; this plan is about cost). Check items
off as they land and record measured results inline.

## Where the time goes (profiled 2026-07-07, 8-core linux)

Per-solve wall time, `BSplineSolver(..., ground_model="sommerfeld")`,
ground (10.0, 0.002):

| case              | total  | grid fill | remainder assembly        |
|-------------------|--------|-----------|---------------------------|
| dipole @ 0.05 λ   | 1.29 s | 1.25 s    | ~0.04 s                   |
| yagi @ 0.2 λ      | 3.51 s | 2.21 s    | 1.3 s (eval 1.0, einsum 0.5) |
| isolated 2 λ grid | 4.11 s | 4.11 s    | —                         |

Grid fill = adaptive Gauss quadrature at 460–800 (R₁, θ) nodes →
21k–39k 24-point segments → ~1M integrand points. Within it:

- ~25 % irreducible scipy special functions: AMOS complex Bessel/Hankel
  costs 0.8–1.6 µs/point *even fully batched* (measured on 1M-element
  arrays) — the algorithm is expensive, not the ufunc dispatch.
- ~75 % Python/numpy per-call overhead on 24-element arrays:
  `_integrand_six` is 41 µs/call of which the FLOPs are ~1 µs
  (`np.stack`, ~15 tiny-ufunc dispatches, `_gamma` at 6.5 µs for two
  sqrts, ...).

The remainder assembly (`SommerfeldGrid.eval` 4×4 Lagrange + einsum) is
already properly vectorized; secondary target only.

Compiled reference (same algorithm, black-box CLI/process timing only —
no GPL source consulted): nec2c's gn 2 increment over gn 0 on identical
exported decks is **24 ms** (dipole @ 0.05 λ) / **48 ms** (yagi @ 0.2 λ),
single-threaded; PyNEC in-process agrees (25/73 ms total). So ~50–70×
faster today. Attribution: compiled scalar loops (no interpreter tax),
own order-0/1 complex Bessel/Hankel (~10× under AMOS generality), and —
per the public-domain theory manual — a grid capped at R₁ ≈ 1 λ with an
asymptotic form beyond, plus looser (~4-significant-digit) interpolation
targets. nec2c is the existence proof for the Phase 3 target.

A compounding structural cost: the grid cache is per solver instance
(`bspline.py` `_cached_somm_grid`) and antennaknobs' web adapter builds a
fresh engine per request, so **every interactive knob-turn re-pays the
full 2–4 s fill** even at unchanged frequency and ground. Sweeps refill
per k regardless (ε̃ depends on k through σ/ωε₀).

## Phase 1 — module-level bucketed grid cache (pure Python)

- [x] Replace the per-instance cache with a module-level FIFO cache
      (same pattern as `_cached_geometry` / `_evict_fifo`) keyed
      `(eps_t, k, r1_max_bucket, omega, mu)`; bound 128 (one entry per
      sweep k × a few grounds; a grid is a few tens of kB).
- [x] Bucket `r1_max` upward in geometric steps (~25 %, floor ~0.1 λ) so
      small geometry changes (knob turns) land in the same bucket. A
      grid built for a larger `r1_max` is valid (and marginally finer in
      θ) for any smaller radius — oversizing only costs fill time once.
      (`_somm_r1_bucket`)
- [x] Tests: two solver instances share one grid object; nearby r1_max
      values hit one bucket; FIFO eviction bounded; full sommerfeld
      suite green (golden gn 2 gates have 1.3× headroom over the
      measured cross-solver floor — bucketing shifts values only within
      interpolation error). 53/53 pass with bucketed grids.

Outcome: repeat solves at fixed (ε̃, k) — the interactive web case —
skip the fill entirely. Measured (fresh solver instances): dipole
@ 0.05 λ 1.17 s cold → **33 ms warm**; yagi @ 0.2 λ 3.43 s cold →
1.34 s warm (the residue is the uncached eval+einsum assembly — see
non-goals).

## Phase 2 — cap tabulation at ~1 λ, extrapolate beyond — **REJECTED**

Geometrical-optics argument: the four normalized surfaces (free-space
factor e^{−jkR₁}/R₁ divided out) should tend to θ-only functions
(Fresnel-minus-C₂ shapes) as R₁ → ∞, with O(1/R₁) correction.

- [x] Numerically characterized (2026-07-07): fit `A(θ) + B(θ)/R₁` from
      rings at 0.8/1.0 λ, predict at 1.5–10 λ, compare against
      `iv_surfaces_direct`, three grounds, θ = 0.5°–89.5°.
- [x] **Result: the hypothesis fails.** Relative errors *grow* with R₁
      instead of shrinking — IrhoH at grazing reaches 3–15× the local
      value by 5–10 λ (poor ground worst), IzV near vertical degrades to
      ~2× by 10 λ, and even mid-θ errors sit at 5–25 %. The normalized
      surfaces do not settle to θ-only limits: the lateral-wave content
      oscillates at the k₁−k₂ beat (the same beat the grid's ΔR₁ keying
      resolves), which no smooth function of 1/R₁ can represent. NEC's
      1 λ cap evidently rests on an explicit asymptotic decomposition
      (GO + lateral-wave terms), and deriving one cleanly is a research
      project, not a perf patch.

Decision: keep the geometry-sized grid — it is precisely what buys the
0.98 Ω yagi agreement beyond NEC's 1 λ grid edge
(`test_yagi_tracks_gn2_large_r1`) — and let Phase 3 make its fill cheap.
Node count grows ~linearly with r1_max (beat-keyed ΔR₁), so even a 20 λ
grid is a few-thousand-node fill: trivial in C++, and any geometry that
large pays far more in the dense O(N²) solve anyway.

## Phase 3 — C++ fill kernel in `_accelerators.cpp`

The extension infrastructure (pybind11, OpenMP, cibuildwheel,
pure-Python fallback + wheel import smoke test) already exists.

- [ ] Clean-room complex-argument J₀/J₁/H₀⁽²⁾/H₁⁽²⁾ from Abramowitz &
      Stegun (power series small |x|, asymptotic expansions large |x|;
      **no nec2c/nec2++/somnec source consulted**). Validate pointwise
      against scipy over the actual contour-point domain (sampled from
      real fills) to ~1e−12.
- [ ] Port `_integrand_six` / `_gauss_segment` / `_adaptive_segment` /
      `_tail` / `_six_integrals`; expose a batch entry point
      `somm_six_integrals_batch(eps_t, k2, rho[], h[], rtol, form[])`
      with OpenMP across nodes (nodes are independent).
- [ ] Python `SommerfeldGrid` uses the accelerator when importable,
      falls back to the existing pure-Python path otherwise (same
      pattern as the other kernels). Cross-check test: C++ vs Python
      six-integrals over sampled nodes to quadrature tolerance.
- [ ] Target: ≤ 100 ms per (ε̃, k) fill single-threaded, tens of ms with
      OpenMP (nec2c does the whole gn 2 increment in 24–48 ms
      single-threaded at looser tolerance). Re-run the profile table
      and record results here.

Outcome: sweeps stop being dominated by grid fills (21-point sweep
sommerfeld overhead ~60 s → ~2 s or less).

## Non-goals / notes

- `SommerfeldGrid.eval` + einsum assembly (~1.3 s on the yagi) stays
  numpy for now; revisit only if it dominates after Phases 1–3.
- Accuracy is not traded away: grid rtol stays 1e−6 (the measured
  2.4 Ω agreement floor vs nec2c may partly be *their* looser
  tolerance).
- Related but separate (antennaknobs repo): pin momwire to an exact
  release so momwire upgrades are always deliberate antennaknobs PRs,
  never silent behavior changes on deploy.
