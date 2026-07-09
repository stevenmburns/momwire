# Triangular retirement — transfer the batched sweep to BSpline, then delete

**STATUS: ON HOLD (2026-07-08)** — sequenced behind PR #125 (Sommerfeld
perf Phase 4), which rewrites the remainder-assembly code the swept
drivers call and adds kernels to the same `_accelerators.cpp` this plan
touches. Land #125 first, then rebase this work. Phase 0 below is
disjoint from #125 and safe to run any time.

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

- The recorded parity convention already runs both solvers at even
  nsegs, where they are identical — which is why this never bit in
  practice.
- Of the two odd-N conventions, triangular's snap gives the better
  finite-N answer (a between-knots delta-gap is poorly represented by
  the d=1 basis: N=21 short dipole, snap = 11.99−944.75j vs mid-split
  13.24−992.76j vs converged ≈ 11.95−939.8j). d=2 doesn't care (its
  basis can put current extrema between knots).

Remaining Gate 0 work (small now):

- [ ] Test migration rule: port triangular's unique tests at EVEN
      nsegs (values carry over exactly), or set `feed_arclength` to a
      knot. Never port an odd-N midpoint-fed triangular value verbatim.
- [ ] Decide the documented d=1 odd-N feed guidance (use even N /
      explicit knot feed; optionally add a snap-to-knot opt-in to
      BSplineSolver if any migrated caller needs it — likely none do).
- [ ] Correct `docs/bspline-swept-consolidation-plan.md`'s "different
      quadratures" section (and any docstring that echoes it) so the
      scare isn't rediscovered a third time.

## Phase 0 — baselines and honest sizing (safe during the hold)

- [ ] Gate 0 items above.
- [ ] Measure swept latency at the actual worst production shapes
      (skyloop n_per_wire=100 ⇒ ~400 segs, hexbeam, 5-band fan dipole
      with junctions) for triangular vs bspline d=1/d=2, sweep sizes
      the frontend actually sends. Current main (per-k loop): dipole
      41-pt sweep at N=21/41/101 = 19/33/1721 ms for d=2 vs
      3/14/149 ms triangular — invisible at interactive sizes, opens
      ~13× at N=101 and grows N² per k-point.

## Phase 1 — rebase + revive PR #101 (after #125 lands)

- [ ] Rebase `feat/bspline-swept-batched-assemble` onto main; resolve
      `_accelerators.cpp` against #125's kernels; re-verify bit-exact
      + the 1e-10 swept-vs-per-freq gate.
- [ ] Extend the batched path's ground coverage to refl-coef (batched
      weighted assemble + per-k w_A/w_Phi weights); sommerfeld remainder
      stays per-k inside the chunk loop (grid keyed on (ε̃, k); it's the
      opt-in expensive path and #125 is making it fast).

## Phase 2 — the hard half: junctions + multi-port (the real blockers)

Per the consolidation plan's priority order:

- [ ] `assemble_Z_bspline_general_swept` (C++): batched general
      per-basis `(support_seg, support_L, support_R)` assemble —
      mirror of triangular's batched `assemble_Z_general`.
- [ ] Batched KCL Schur solve over the `(n_k, n_b, n_b)` stack: port
      `_solve_with_kcl_batch` / `_solve_with_kcl_swept_ports` from
      triangular.py — pure Schur algebra on (Z, kcl_A), basis-agnostic,
      moves verbatim.
- [ ] `compute_y_matrix_swept` batched fast path (matrix RHS through
      the same solve).
- [ ] Enrichment stays gated per-k (bspline-only feature, no
      retirement blocker; existing NotImplementedError stands).
- [ ] Regression: swept == per-k to ~1e-10 on dipole/V/yagi/K=3
      fan-dipole × {free, PEC, refl-coef} × {single-feed, multi-port}.
- [ ] Acceptance: bspline d=2 swept within ~2× of old triangular on
      the Phase 0 production shapes; accept the documented residual
      (single large-wire single-thread ~1.3–1.5×, same-edge scheme).

## Phase 3 — delete Triangular

- [ ] Delete `triangular.py`, `_triangular_kernels.py`, the
      `TriangularSolver` export in `__init__.py`.
- [ ] Delete the now-dead C++ kernels: `seg_seg_quad_batch_3d`,
      `seg_seg_reg_quad_batch_1d`, `assemble_Z`, `assemble_Z_general`
      (grep first — bspline/hmatrix `assemble_Z*` hits are the
      `assemble_Z_bspline*` names; hmatrix/array_block are
      BSplineSolver subclasses, expected clean — verify).
- [ ] Tests: migrate per Gate 0's pinning rule; update
      `test_cancel.py` / `test_gil_release.py` entries.
- [ ] Migrate `validation/momwire_backend.py` + examples (hentenna,
      hexbeam, bowtie, yagi, feedline) and `scripts/compare_*` /
      `probe_*` to bspline; delete `profile_triangular.py` and
      triangular-specific probes.
- [ ] De-stale docstrings naming TriangularSolver as "the default"
      (sinusoidal.py, bspline.py); retirement note in NEXT_STEPS.md
      (it's a log — keep history intact). Fold the still-relevant
      parts of `docs/bspline-swept-consolidation-plan.md` into this
      doc and retire that one.

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
