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

- [x] Clean-room complex-argument J₀/J₁/H₀⁽²⁾/H₁⁽²⁾ from Abramowitz &
      Stegun (ascending series |x| ≤ 12, asymptotic expansions with
      optimal truncation beyond; **no nec2c/nec2++/somnec source
      consulted**). Measured domain from real fills: |x| ≤ ~110, arg(x)
      ∈ [−100°, +45°] — never near the negative-real-axis cut.
- [x] Ported `_integrand_six` / `_gauss_segment` / `_adaptive_segment` /
      `_tail` / `_six_integrals` verbatim (same 24-pt rule, same
      contours); batch entry `somm_six_integrals_batch(eps_t, k2,
      rho[], h[], rtol, form)` with `omp parallel for schedule(dynamic)`
      across nodes.
- [x] `_six_integrals_batch` in `_sommerfeld.py` routes through the
      accelerator when loaded, per-node Python loop otherwise;
      `iv_surfaces_direct` (and therefore every grid fill) uses it.
      Cross-checks in tests/test_sommerfeld_accel.py: C++ vs Python
      agree to 3e−12 per node for physical grounds (3e−9 at the
      ε̃ = 1e16 PEC-limit stress case) — which doubles as validation of
      the A&S Bessel port against scipy's AMOS, since the Python path
      evaluates the same λ points through jv/hankel2. Fallback and grid
      equivalence tests included; golden gn 2 gates run the accelerated
      path automatically.
- [x] Measured: isolated 2 λ grid fill 4.11 s → **0.34 s** (12×; 1.05 s
      single-thread, so ~30× scalar + ~4.6× from 8-core OpenMP on
      uneven node costs). Dipole @ 0.05 λ cold solve 1.29 s → 0.12 s;
      yagi @ 0.2 λ cold 3.51 s → 1.45 s (the rest is eval+einsum
      assembly).
      Sommerfeld test suites 36 s → 7 s. The remaining gap to nec2c's
      24–48 ms is tolerance policy — we hold 1e−11 per contour segment
      where NEC holds ~4 digits — and accuracy is a non-goal to trade
      (see below).

- [x] Cooperative cancellation: `somm_six_integrals_batch` takes the
      standard trailing `cancel_flag`, polls per node with the drain
      pattern, and is registered in `_CANCELLABLE_KERNELS`; the Python
      fallback loop polls the same flag address, and the new Python-side
      seams checkpoint too (`_Z_sommerfeld_remainder` per observer
      chunk). A cancelled fill raises before the module cache insert, so
      no partial grid is ever cached. Tests: kernel-level tripped-flag,
      fallback tripped-flag, solve-level abort with Python checkpoints
      neutralized, untripped-token bit-identity.

Outcome: a 21-point sweep's fill overhead drops from ~25–75 s to
~1.5–3 s, and it composes with the Phase 1 cache (repeat sweeps at
unchanged ground/frequency pay nothing).

## Phase 4 — compiled remainder assembly (the eval + einsum non-goal, now the bottleneck)

Phases 1–3 drove the *grid fill* to near-zero (cached + C++). The
benchmark that motivates this phase lives in the antennaknobs repo
(`docs/status/2026-07-08-ground-model-benchmark.md`): across 10 designs,
momwire's Sommerfeld path trails PyNEC's native NEC gn 2 by **10–100×**,
and the cost scales cleanly as O(N²) per band on the dense solvers
(moxon somm 388→1412→5261 ms for N=21→41→81 — ×4 per doubling). That is
not the fill (cached); it is the per-pair **remainder assembly** the
non-goal below deferred.

### Where the time goes now (profiled 2026-07-08, 4-core linux)

`BSplineSolver(degree=2, ground_model="sommerfeld")`, yagi @ N=81,
**warm** (grid already cached, so this isolates assembly only) —
7.54 s/solve, by `tottime`:

cProfile's flat view lumps all `c_einsum` calls together; attributing
each einsum to its caller (measured by wrapping `np.einsum`) gives the
real picture — the hotspot is the grid **interpolation**, not the
Galerkin projection:

| stage                                                | time   | share |
|------------------------------------------------------|--------|-------|
| `SommerfeldGrid.eval` — bicubic interp of 4 surfaces (gather `(4,n,4,4)` + `snij,ni,nj->sn` einsum) | 5.4 s | 73 % |
| `remainder_field_proj` — azimuth combine + tangent projection (own) | 0.89 s | 12 % |
| Galerkin projection einsums (`piq,iqjr,Pjr` + `mp,pPmn,nP`) | 0.29 s | 4 %  |
| `_lagrange4` + `np.stack` etc.                       | 0.35 s | 5 %   |
| EFIE fill + dense solve (everything else)            | <0.1 s | <1 %  |

So ~90 % of a warm Sommerfeld solve is the per-pair remainder assembly,
dominated by the O(N²q²) bicubic interpolation of the four grid surfaces
(the `eval` gather materializes a `(4, n, 4, 4)` complex block —
hundreds of MB — and the contraction is memory-bandwidth-bound). PyNEC
does the analogous per-pair work in compiled scalar loops with no
intermediates. This is the same interpreter/bandwidth-tax gap Phase 3
closed for the fill (~30× scalar), now on the assembly.

All three dense/fast Sommerfeld paths funnel through the same two
Python routines, so one compiled kernel helps every solver:
- `bspline._Z_sommerfeld_remainder` — dense O(N²) Galerkin block.
- `sinusoidal._field_tensor_sommerfeld_remainder` — field-tensor form.
- `hmatrix/array_block._zblock_sommerfeld_remainder` — rectangular
  sampler the fast solvers call O(N·rank) times through ACA (why Arr/ACA
  also trail PyNEC on the benchmark, not just the dense bases).

### 4a — pure-Python einsum/interpolation tuning — **REJECTED (measured 2026-07-08)**

Initial reading of cProfile's flat `c_einsum` line suggested the Galerkin
contraction was the target. Wrapping `np.einsum` to attribute per-caller
showed that was wrong: the Galerkin einsums are only ~4 % of the solve;
the cost is the bicubic **interpolation** inside `SommerfeldGrid.eval`.
Both were then tuned on representative shapes (n≈4×10⁵ points):

- The interpolation contraction `snij,ni,nj->sn` is *already* the fastest
  form. `optimize=True` (297 ms), a two-step einsum (344 ms), a
  broadcast-sum (611 ms), and a `matmul`-style split (339 ms) are all
  **slower** than the current 3-operand call (163 ms). numpy's `c_einsum`
  handles this small bilinear contraction near-optimally.
- Avoiding the big `(4,n,4,4)` gather by accumulating over the 16 stencil
  offsets, or a separable two-pass gather, is also slower (720 / 714 ms
  vs 559 ms) — 16 advanced-index gathers cost more than one large one.

Conclusion: the assembly is memory-bandwidth-bound numpy with no
reassociation left to exploit. There is no meaningful pure-Python win;
the only lever is compiled code with no materialized intermediates
(register-level per-point contraction). Skip straight to 4b.

### 4b — fused C++ remainder-field-projection kernel — **LANDED (2026-07-08)**

First increment shipped: `remainder_field_proj_batch` in `_accelerators.cpp`
ports the whole of `_sommerfeld.remainder_field_proj` (which internally
calls `SommerfeldGrid.eval`) to C++ — per (observer, source) pair it does
the 4×4 Lagrange interpolation of the four surfaces and the eqs 143–147
azimuth/tangent projection inline, with **no materialized `(4,n,4,4)`
intermediate** and one `omp parallel for` over observer rows. The Python
`remainder_field_proj` routes to it when the accelerator is loaded and
keeps the vectorized numpy body as fallback (mirrors the Phase-3
`_six_integrals_batch` dispatch).

Because all three solver families call `remainder_field_proj`, this one
kernel covers 4c for free.

Measured (warm — grid cached — so this isolates the assembly, the honest
attribution of this change; 4-core, yagi @ N=81):

| solver | warm somm before | after | speedup |
|--------|------------------|-------|---------|
| BSpline deg 2 | 7443 ms | 965 ms | 7.7× |
| Sinusoidal    | (assembly-bound) | 236 ms | ~9× |
| ArrayBlock / HMatrix | — | inherit via shared fn | — |

- Kernel vs numpy fallback agree to 4e-16 (round-off); the full
  sommerfeld suite is green (184 passed).
- Remaining warm cost on the dense path (~960 ms) is now the **Galerkin
  projection**: the `Jf` handling + the two einsums (`piq,iqjr,Pjr` and
  `mp,pPmn,nP`) that the kernel does not yet absorb — 0.35 s numpy +
  0.28 s einsum. The C++ projection itself is 0.14 s (was 5.4 s).

Follow-ups (next increments on this branch):

- [ ] Fold the basis-weighted Galerkin quadrature (`W`, `polys`) into the
      kernel so it returns the projected J-block / Q directly, removing
      the `Jf` intermediate and both Galerkin einsums (the residual
      ~0.6 s on the dense path).
- [ ] Hoist the grid marshalling out of `remainder_field_proj`: the ACA
      sampler calls it O(rank) times on small rectangles, re-building the
      `reg_vals` arrays each call — pass a pre-marshalled grid handle so
      HMatrix/ArrayBlock don't re-pay it per sample.
- [ ] Re-run `scripts/profile_ground_models.py` somm column and refresh
      the antennaknobs status-doc ratios (cold-grid numbers, the
      user-visible figure).

### 4b (original scope) — fused C++ remainder-Galerkin kernel

Mirror Phase 3 exactly (pybind11 + OpenMP + pure-Python fallback +
`cancel_flag` + wheel smoke test all already exist in
`_accelerators.cpp`). Add one kernel that, per (observer node, source
node) pair, does the whole assembly inline and accumulates straight into
the moment-weighted block — fusing eval + `remainder_field_proj` + the
inner einsum (≈88 % of the time) into a single `omp parallel for` over
observer segments:

- [ ] Expose the tabulated `SommerfeldGrid` to C++: pass the four surface
      arrays + axis params (`r1`/`θ` node grids, `r1_max`, region splits)
      into the kernel. The grid is *built* in C++ already
      (`somm_six_integrals_batch`); this hands the finished table back
      down for interpolation. This grid-marshalling is the only genuinely
      new plumbing vs Phase 3.
- [ ] Port the 4×4 Lagrange interpolation (`_lagrange4` + the region
      dispatch in `SommerfeldGrid.eval`) and the eqs 143–147 azimuth /
      tangent projection (`remainder_field_proj`) to C++ scalar code.
- [ ] Fold the basis-weighted quadrature (`W`, `polys`) into the same
      loop so the kernel returns the projected J-block (or Q directly),
      removing the Python `einsum` entirely for the dense path.
- [ ] Cross-check vs the Python fallback to ~1e-11/node (same bar as the
      Phase 3 six-integrals cross-check) and against the golden gn 2
      gates. Register in `_CANCELLABLE_KERNELS`; poll `cancel_flag` per
      observer chunk.

Target: the Phase 3 scalar factor (~30×) applied to a 7.5 s warm solve →
~0.25 s, i.e. parity with PyNEC's 0.18 s on the yagi somm N=81 cell, and
proportional wins across the matrix. OpenMP across cores on top of that.

### 4c — route the fast solvers through the kernel

- [ ] Point `_zblock_sommerfeld_remainder` (the rectangular ACA sampler
      shared by HMatrix/ArrayBlock) at the same C++ kernel so the
      low-rank paths inherit the speedup — they already win structurally
      on large single wires / arrays (rhombic, bowtiearray2x4); a
      compiled per-pair kernel is what closes their residual gap to
      PyNEC too.

### Validation

- [ ] Re-run `scripts/profile_ground_models.py` (antennaknobs) somm
      column before/after and update the status doc with the new ratios;
      the win is real only if the O(N²) constant drops, not the scaling.
- [ ] Full sommerfeld suite green; golden gn 2 gates unchanged within
      their 1.3× headroom.

## Non-goals / notes

- Accuracy is still not traded away: grid rtol stays 1e-6 and the 4-point
  Lagrange interpolation order is preserved through the C++ port (4b is a
  language port of the *same* arithmetic, not a coarser scheme). The
  residual few-Ω agreement floor vs nec2c may partly be *their* looser
  tolerance, and is out of scope here.
- Accuracy is not traded away: grid rtol stays 1e−6 (the measured
  2.4 Ω agreement floor vs nec2c may partly be *their* looser
  tolerance).
- Related but separate (antennaknobs repo): pin momwire to an exact
  release so momwire upgrades are always deliberate antennaknobs PRs,
  never silent behavior changes on deploy.
