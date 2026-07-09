# Triangular retirement — transfer the batched sweep to BSpline, then delete

**STATUS: EXECUTED (2026-07-08, branch `feat/retire-triangular`)** — all
phases below are done except the release (Phase 4). PR #125 landed first
as planned. Measured results are inlined per phase.

Goal: remove `TriangularSolver` from momwire. antennaknobs already
retired it end to end (v0.19.0: frontend, adapter, CLI, engine default →
BSpline d=2), so momwire carries it for nothing — except for ONE
capability BSpline doesn't have yet: the k-batched sweep fast path.

## Prior art — this was attempted before. Build on it, don't restart.

**PR #101 (`feat/bspline-swept-batched-assemble`, draft) + its
`docs/bspline-swept-consolidation-plan.md` already did the easy half**
(NEXT_STEPS item 19a+19b):

- `seg_seg_full_moments_bspline_swept` (C++) — off-edge moments batched
  over k, analog of triangular's `seg_seg_quad_batch_3d`.
- `assemble_Z_bspline_swept` (C++) — batched assemble, analog of
  triangular's batched `assemble_Z`.
- `_compute_impedance_swept_batched` — chunked (~32 MB/chunk) fast path;
  per-k loop retained as fallback. Bit-exact vs single-k (|Δz|~1e-10),
  158 tests passed at the time.

Covered: single/multi-feed impedance sweeps, free-space + PEC, **no
junctions, no multi-port Y, no enrichment**. Measured: parity with
triangular on multi-wire threaded sweeps (yagi3 1.04–1.09×); single
large wire still 1.3–1.5× slower (same-edge moment scheme, not
batching — see the gate below). The branch predates v0.6.0+; expect a
substantial rebase (refl-coef/sommerfeld ground landed in the swept
loop since).

## Gate 0 — the "different solvers" scare: RESOLVED (2026-07-08)

The consolidation plan on the PR #101 branch reported up to **5.1%**
impedance disagreement between triangular and bspline d=1 at coarse N
and concluded they are "genuinely different quadrature schemes" needing
an accuracy arbitration before retirement. **That conclusion is wrong —
the entire difference is the FEED CONVENTION, measured on odd nsegs.**

Reproduction on current main (short 0.24 λ dipole, the doc's worst
case):

- **Even nsegs** (feed knot exactly at midpoint): |ΔZ| ≤ 8e-11 Ω at
  N = 10/20/40/80, resonant AND short — **numerically identical**.
- **Odd nsegs**, both fed "at the midpoint": 5.1% at N=21 — but
  triangular SNAPS the delta-gap to the nearest interior knot (half a
  segment off-center) while bspline evaluates the source at the exact
  arclength, splitting the delta-gap between two tents.
- **Odd nsegs with bspline's `feed_arclength` set to the knot
  triangular snapped to: |ΔZ| ≤ 4e-11 Ω.** Assembly, same-edge
  moments, off-edge kernel — all bit-equivalent to roundoff; the
  `_bspline_kernels.py` "bit-for-bit" docstring is RIGHT about the
  kernels. Only the source placement differs.

Notes that fall out of this:

- The parity is ENFORCED by the caller, not just conventional:
  antennaknobs `_parity_for_solver` (engines/momwire.py) coerces
  nsegs — bumping by one when needed — to even for d=1 (and its
  HMatrix/ArrayBlock subclasses) and odd for d=2/sinusoidal, before
  the mesh is built. So through antennaknobs the odd-N feed-split
  case is unreachable and d=1 ≡ triangular to roundoff. The exposure
  is limited to direct momwire API users choosing odd nsegs with a
  midpoint feed.
- Of the two odd-N conventions, triangular's snap gives the better
  finite-N answer (a between-knots delta-gap is poorly represented by
  the d=1 basis: N=21 short dipole, snap = 11.99−944.75j vs mid-split
  13.24−992.76j vs converged ≈ 11.95−939.8j). d=2 doesn't care (its
  basis can put current extrema between knots).

Remaining Gate 0 work (small now):

- [x] Test migration rule: ported triangular's unique tests at EVEN
      nsegs / knot-fed arclengths; `tests/test_tent_parity.py` carries
      the pinned d=1 values (recorded at roundoff agreement with
      triangular before deletion) plus the odd-N feed-convention pin.
- [x] d=1 odd-N feed guidance documented in the BSplineSolver `degree`
      docstring and the tent-parity module docstring; no snap-to-knot
      option needed (no migrated caller wanted one; antennaknobs
      coerces d=1 to even N anyway).
- [x] The "different quadratures" scare is corrected here and in the
      `_bspline_kernels.py` docstring (whose bit-for-bit claim was in
      fact RIGHT); `docs/bspline-swept-consolidation-plan.md` only ever
      existed on the PR #101 branch, which this branch supersedes —
      close #101 unmerged when this lands.

## Phase 0 — baselines and honest sizing

- [x] Gate 0 parity pins: `tests/test_tent_parity.py` (11 tests) —
      dipole/yagi/V/PEC-ground/K=3-junction/multi-feed-Y/swept all
      match to ≤1e-8 relative on knot-fed meshes; odd-N feed-snap
      convention pinned explicitly.
- [x] Production-shape swept baselines (41-point sweep, this machine,
      post-#125 main — the Phase 2 acceptance reference):

      | shape | tri | d=1 | d=2 |
      |---|--:|--:|--:|
      | dipole N=400 | 1.04 s | 7.88 s | 8.58 s |
      | yagi 6×40 | 0.24 s | 1.92 s | 1.69 s |
      | fan dipole K=3, 4×(50+50) | 0.69 s | 7.41 s | 8.04 s |
      | skyloop 4×100, K=2 closure | 0.82 s | 7.57 s | 8.14 s |

      7–11× everywhere at production scale (interactive small cases
      N=21/41 stay fine at 19/33 ms). Junction shapes (fan dipole,
      skyloop) confirm the batched-KCL half matters, not just the
      dense case.

## Phase 1 — rebase + revive PR #101 (after #125 lands) — DONE

- [x] Cherry-picked #101's feature commit onto post-#125 main; resolved
      `_accelerators.cpp` / `bspline.py` / `_bspline_kernels.py`;
      batched == per-k to ≤3e-11 (regression tests added — #101 carried
      none). Dropped the unused `offedge_full`/`image_full` pass-through
      and the fallback's unchunked whole-sweep off-edge prebuild
      (~1 GB at production scale).
- [x] Ground: batched path serves free space + PEC; finite grounds
      (refl-coef, sommerfeld) fall back to the per-k loop — they carry
      per-k ε̃(ω) weight tables / (ε̃, k)-keyed grids, were never fast
      under triangular (which had no finite ground at all), and #125
      just compiled the sommerfeld remainder. Batched refl-coef is an
      optional later enhancement, NOT a retirement blocker.

## Phase 2 — the hard half: junctions + multi-port — DONE (no new C++!)

The consolidation plan assumed a `assemble_Z_bspline_general_swept`
kernel was needed. It isn't: bspline's assembly is already general —
junction directional bases live inside `supp_seg`/`polys`, which
`assemble_Z_bspline_swept` consumes as-is. Only the KCL constraint
needed batching, in pure Python:

- [x] `_solve_with_kcl_batch` / `_solve_with_kcl_swept_ports` ported
      verbatim from triangular.py (basis-agnostic Schur algebra).
- [x] Shared `_swept_batched_z_chunks` generator; junction-capable
      `compute_impedance_swept` + batched `compute_y_matrix_swept`.
- [x] Chunking fix (the real perf unlock): hoist same-edge reg moments
      for the FULL sweep (the streaming kernel amortizes its R hoist
      across the whole k axis; per-chunk calls were 52% of the
      fandipole sweep) and raise the J budget 32→256 MB (24 MB/k at
      N=400 d=2 degenerated to chunk=1, killing the off-edge kernel's
      R reuse).
- [x] Enrichment stays gated per-k (existing NotImplementedError).
- [x] Regression: batched == per-k to ≤1e-11 across the junction ×
      ground × multi-port matrix (existing + new tests).
- [x] Acceptance measured (41-pt sweep vs triangular; was 7–11×):
      fandipole d1 1.4× / d2 2.6×; dipole400 d1 1.1× / d2 2.2×;
      skyloop d1 1.3× / d2 2.4×. d2's arithmetic floor is 2.25×
      (9 vs 4 moment channels), so d2 is at its floor and d1 is at
      parity — accepted.

## Phase 3 — delete Triangular — DONE

- [x] Deleted `triangular.py`, `_triangular_kernels.py`, the
      `TriangularSolver` export, and the four dead C++ kernels
      (`seg_seg_quad_batch_3d`, `seg_seg_reg_quad_batch_1d`,
      `assemble_Z`, `assemble_Z_general`) + their m.defs and the
      `_accel.py` cancellable-kernel registry entries. Verified
      hmatrix/array_block clean (their `assemble_Z*` hits are the
      bspline names).
- [x] Tests migrated: 19 redundant triangular tests deleted (bspline
      equivalents existed); geometry smokes (yagi/V/collinear/moxon/
      hexbeam/hentenna/bowtie/K=2-junction) converted to d=1; the
      hentenna arbitration + ground cross-checks now reference d=1
      with the same pinned constants; `test_tent_parity.py` rewritten
      as pinned-value regressions; test_cancel's abort-mapping test
      re-targeted at `assemble_Z_bspline`. Note learned in migration:
      K=2-junction ≡ single-polyline is EXACT only for d=1 (d=2's
      split changes the basis space slightly, ~4e-6 Ω — documented in
      the test).
- [x] `validation/momwire_backend.py` default → bspline with
      retired-name fallback; example comments de-staled; deleted the
      triangular-purposed scripts (`profile_triangular`,
      `compare_vdipole_nec`, `compare_yagi_nec`,
      `compare_fandipole_solvers`, `vtune_hentenna_width_sweep` — all
      already dead: they still imported the pre-rename `pysim`
      package).
- [x] Docstrings de-staled (sinusoidal.py "the default", bspline.py
      TriangularSolver cross-references, `_bspline_kernels.py`
      bit-for-bit claim now stated correctly); NEXT_STEPS.md carries a
      retirement banner; the consolidation-plan doc stays on the
      superseded PR #101 branch (close #101 unmerged).
- [x] Net: −2900 lines (triangular.py 1091, kernels 272, ~630 C++,
      scripts + tests), one Galerkin solver family.

## Phase 4 — release

- [ ] momwire v0.9.0 via the /release skill (public-class removal
      called out in the PR title — titles are the release notes).
- [ ] antennaknobs follow-up PR: exact pin bump in the THREE places
      (pyproject, Dockerfile, submodule pointer), latency-smoke the
      sweep path on the Phase 0 shapes — it should get FASTER, verify.
      antennaknobs has no TriangularSolver references left, so the pin
      bump is the whole PR.

## Risks / notes

- **PR #125 contention** (the reason for the hold): same
  `_accelerators.cpp`, and its 4b/4c compiled remainder kernel changes
  the sommerfeld code the swept loop calls per-k. Mostly-mechanical
  merge conflicts, but don't run them concurrently.
- Batched (d+1)² moment tensors: PR #101's ~32 MB k-chunking already
  handles it; keep it.
- The old plan's fallback remains available: if the hard half proves
  not worth it, keep triangular purely as the swept engine and stop —
  but post-v0.19.0 nothing ships on triangular, so the fallback now
  costs 1400 lines of dead-to-users code, which is the thing this plan
  exists to remove.
