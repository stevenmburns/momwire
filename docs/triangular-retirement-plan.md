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

## ⚠️ Gate 0 — Triangular and BSpline d=1 are NOT the same solver

The consolidation plan documents this (it's why the last attempt
stalled): despite being the same tent basis, the two are **genuinely
different finite-N quadrature schemes**. Same geometry, same n_qp=4:

| case | triangular | bspline d=1 | rel diff |
|---|---|---|--:|
| dipole N=21 | 11.61 − 963.8j | 12.84 − 1013.2j | **5.1%** |
| dipole N=81 | 11.57 − 958.9j | 11.97 − 975.4j | 1.7% |
| yagi2 N=21 | 34.86 − 11.65j | 34.93 − 11.55j | 0.34% |

They converge to the same limit (agreement ~0.1 Ω by N=101 on the
resonant dipole) but disagree up to ~5% at coarse N on off-resonance
reactance. Suspected mechanisms (not yet pinned): same-edge
static-moment split vs tent-specific closed forms, delta-gap source
projection, off-edge full-kernel evaluation. The `_bspline_kernels.py`
"bit-for-bit equivalent" docstring is **wrong end-to-end** — fix it as
part of this work.

**What changed since that gate was written:** antennaknobs v0.19.0
already switched every user to bspline d=2 — the user-facing result
shift the old gate guarded against **has already happened and been
recalibrated** (zepp/skyloop/PyNEC-sanity test updates). Triangular is
now reachable only via direct momwire API use. So the gate softens
from "arbitrate before any user sees a shift" to:

- [ ] Diff the same-edge, off-edge, and source-vector contributions
      between the two solvers at fixed geometry; name the mechanism(s).
- [ ] Establish which scheme is more accurate at coarse N against a
      refined-N reference + nec2c (extends the NEXT_STEPS item 13
      convergence study).
- [ ] Port triangular's unique tests to d=1 pinning **bspline's own
      converged values** (not triangular's finite-N values — they WILL
      differ at coarse N; a naive "port the numbers" migration fails,
      which is the trap the last attempt hit).
- [ ] Correct the false docstring; document the scheme difference in
      the module docstrings so it isn't rediscovered a third time.

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
