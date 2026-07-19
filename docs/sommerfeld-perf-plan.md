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

Stage-2 fusion — **LANDED (2026-07-09)**: `sommerfeld_remainder_bspline_Q`
now folds the moment quadrature *and* the basis assembly into the kernel,
returning the `(n_basis, n_basis)` block Q directly — no `Jf` tensor, no
`piq,iqjr,Pjr` / `mp,pPmn,nP` einsums, no large `J_blk` fancy-index gather.
The dense `_Z_sommerfeld_remainder` calls it (numpy path kept as fallback).

Measured warm (yagi N=81, 4-core), cumulative down the increments:

| path | warm somm | note |
|------|-----------|------|
| numpy assembly (pre-4b) | 7443 ms | the deferred non-goal |
| + proj kernel (stage 1) | 965 ms | 7.7× |
| + fused Q kernel (stage 2) | **415 ms** | 18× total; **2.3× off PyNEC (~179 ms)** |

- Fused Q vs the proj-kernel + numpy-Galerkin path: **0.0** (bit-identical
  final Z), parametrized over degree 1 & 2; golden gn 2 gates unchanged;
  full suite green (218 + new parity tests).
- At 415 ms the Sommerfeld remainder is one 0.192 s kernel; the other
  ~0.18 s is the base EFIE fill + dense solve + image assembly, shared
  with every ground model (not Sommerfeld-specific). The remainder is now
  compute-bound (per-pair interpolation + projection arithmetic × OMP),
  near the algorithmic floor at this accuracy.
- Grid marshalling hoisted into `_sommerfeld.grid_cpp_args`, shared by
  `remainder_field_proj` and the fused kernel.

Fast-solver ACA path — **LANDED (2026-07-09)**: the fused kernel was
generalized to a rectangular obs/src form (the dense block is its
symmetric `obs==src`, `loc==supp_seg` case), and the ACA sampler
`_zblock_sommerfeld_remainder` (shared by HMatrix + ArrayBlock) now calls
it — one C++ call per sampled rectangle instead of proj-kernel + numpy
einsum + `J_blk` gather. `_sommerfeld_global_lowrank` marshals the grid
once and reuses it across all O(rank) samples.

Measured (fused ON vs OFF, isolating this change; warm grid, cold matrix;
4-core, N=81), numerics identical to <4e-12:

| design | fused | numpy | speedup |
|--------|-------|-------|---------|
| rhombic (HMatrix)          | 8939 ms | 10431 ms | 1.2× |
| bowtiearray2x4 (ArrayBlock)| 7794 ms | 10367 ms | 1.3× |
| delta_looparray_1x4 (Arr)  | 683 ms  | 1096 ms  | 1.6× |

Modest by design: in the fast solvers the Sommerfeld ACA sampling is only
a fraction of the total (the free-space H-matrix / array-block assembly +
iterative solve dominate), unlike the dense path where the remainder was
~90 %. Golden gn 2 gates for both fast solvers run the fused path and are
green; a fused-vs-numpy parity test covers each.

Remaining follow-ups:

- [ ] Re-run `scripts/profile_ground_models.py` somm column and refresh
      the antennaknobs status-doc ratios (cold-grid numbers, the
      user-visible figure).
- [ ] Optional: SIMD / lower quadrature-node count on the per-pair proj to
      chase the last ~2× to PyNEC on the dense path, only if the remainder
      still dominates after the base-solve cost is accounted for.
- [ ] Optional: SIMD / lower quadrature-node count on the per-pair proj to
      chase the last ~2× to PyNEC, only if the remainder still dominates
      after the base-solve cost is accounted for.

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

## Phase 5 — far-pair grid-extent cap (issue #157) — **LANDED**

Phase 2/3's cost model assumed the geometry-sized grid is always cheap
("even a 20 λ grid is a few-thousand-node fill: trivial"). That holds to a
couple tens of λ, but the region-2 θ spacing is keyed to the grid extent
(`_sommerfeld.py`, `dth2_target = deg(0.07 λ / r1_max)`), so the node count
grows ~**quadratically**, not linearly, once r1_max is large: a wire parked
hundreds of λ away — the NEC TL-anchor idiom, or a rhombic/long-wire array
over real ground — drives r1_max to 100–500 λ and the fill to millions of
oscillatory Sommerfeld integrals, i.e. an effective hang (a 52-seg problem
that otherwise solves in 1.3 s runs > 300 s).

Fix: cap `r1_max` at `_SOMM_R1_CAP_LAMBDA` (15 λ) in `SommerfeldGrid` and
`get_grid`. Beyond the cap, `proj_one`'s existing r1 → r1_max clamp stands
in (the free-space factor g = e^{−jkR₁}/R₁ keeps the *true* distance, only
the slowly-varying surface amplitude freezes at the cap); the Python
`grid.eval` guard now clamps to match instead of raising. No C++ change.

Why this is **not** the rejected Phase 2 (cap at ~1 λ + extrapolate):

- Phase 2 tried to *extrapolate* the surfaces past the grid with a smooth
  A(θ)+B(θ)/R₁ model and failed — the lateral-wave content oscillates at
  the k₁−k₂ beat, so relative surface error *grows* with R₁. Phase 5 does
  not model the far surfaces at all; it asserts their *contribution* is
  negligible and freezes it.
- The cap is 15 λ, not 1 λ, so every real structure keeps its full
  geometry-sized grid — the yagi's >1.2 λ boom (`test_yagi_tracks_gn2_
  large_r1`, 0.98 Ω) is untouched, and Phase 2's reason for keeping the
  geometry-sized grid is preserved intact.

Calibration (finite ground, sinusoidal vs nec2c): a 6 λ long wire, two
dipoles 8 λ apart, and a 171 λ TL anchor all give **bit-identical**
impedance for every cap ≥ ~8 λ vs a 25 λ grid; the remainder is
empirically negligible beyond ~3–4 λ. 15 λ leaves ~4× margin and bounds
the pathological fill to a few seconds (cached). The residual vs nec2c is
the pre-existing ground-model floor (cap-independent, matches the issue's
own control values). Overridable via `MOMWIRE_SOMM_R1_CAP_LAMBDA` for
benchmarking. Regression: `test_grid_r1_max_is_capped`,
`test_remote_wire_stays_bounded_and_irrelevant`.

## Phase 6 — near/far tabulation split (issue #159) — **LANDED**

Phase 5 bounded the pathological (100–500 λ) case but left every large
*real* structure under the cap paying the quadratic keying: an 11.6 λ
terminated long-wire built a 15,218-node grid (region 2 alone 230×60) and
antennaknobs' full-catalog benchmark measured the grid fill as 85–96 % of
every cold Sommerfeld solve — momwire ~13× slower than PyNEC's gn 2 with
the remainder assembly (Phase 4b) already near-free.

The keying was calibrated at the wrong scale. Dense (R₁, θ) scans of the
four surfaces (three grounds incl. the lossless εr=16 stress case) show the
fine structure — the lateral-wave interference near grazing — lives at
**moderate R₁ (~0.5–3 λ) and decays beyond**: at R₁ = 5–10 λ a
2.5° θ / 0.2 λ R₁ lattice interpolates to ≤ 7e−4 of surface scale, vs the
≤ 2e−3 near-zone bar. Keying Δθ to the full extent refines exactly where
the surfaces are smoothest.

Fix: stop the fine tabulation at `r_near = min(r1_max,
_SOMM_R1_NEAR_LAMBDA · λ)` (default 4 λ — where Phase 5's calibration
found the remainder itself going negligible), key region 2's Δθ to
`r_near`, and cover [r_near, r1_max] with two coarse far regions
(ΔR₁ = 0.2 λ; Δθ = 2.5° / 10°). Grids with r1_max ≤ 4 λ build
bit-identically to the pre-split layout. `GridView`/`proj_one` in the C++
kernels take 3 or 5 region tables with a two-level region select; the
numpy `eval` mirrors it (cross-checked to 1e−15 by the accel parity
tests).

Not the rejected Phase 2, again: nothing is extrapolated or modeled — the
far zone is still *tabulated* from `iv_surfaces_direct`, just on the
lattice the measured smoothness supports.

Measured (default ground): 11.64 λ grid 15,218 → 2,915 nodes, fill
3.8 s → 1.3 s; at the 15 λ cap 24,342 → 3,187 (7.6×); node count now
grows linearly with extent (~85 nodes/λ). Catalog: terminated_longwire
cold solve 5.2 s → 2.8 s, rhombic 2.5 s → 2.3 s, impedance shifts ≤ 1e−6
relative; random-point interpolation error vs direct evaluation unchanged
to the printed digit on every surface and ground tested. Sub-4 λ
geometries (most of the catalog) are untouched — their cold cost is the
near-zone fill itself; the next lever there is frequency-axis grid reuse
(#159 proposal 2). Overridable via `MOMWIRE_SOMM_R1_NEAR_LAMBDA`.
Regression: `test_grid_far_zone_*`, `test_grid_small_extent_keeps_pre_
split_layout`, `test_remainder_field_proj_accel_matches_python_far_zone`.

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
