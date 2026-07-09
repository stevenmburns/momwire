# Triangular retirement — transfer the batched sweep to BSpline, then delete

Goal: remove `TriangularSolver` from momwire. antennaknobs already
retired it end to end (v0.19.0: frontend, adapter, CLI, engine default →
BSpline d=2), so momwire carries it for nothing — except for ONE
capability BSpline doesn't have yet. This plan names that capability,
transfers it, and deletes the solver.

## The one unique capability: the k-batched sweep fast path

Feature audit (2026-07-08), Triangular vs BSpline:

| capability | Triangular | BSpline |
|---|---|---|
| multi-wire polylines, kinks | yes | yes |
| K-wire junctions + KCL (Lagrange/Schur) | yes | yes |
| multi-feed / Y-matrix (single + swept) | yes | yes |
| `currents_at_knots(coeffs, s_array)` | yes | yes |
| ground | PEC only | PEC + refl-coef + Sommerfeld (superset) |
| singular enrichment, feed smoothing | no | yes (superset) |
| cancel checkpoints | yes | yes |
| **k-batched sweep** | **yes** | **no — per-k Python loop** |

Triangular's swept path is batched end to end and BSpline's is not:

- `_build_J_blocks_batch` → C++ `seg_seg_quad_batch_3d` /
  `seg_seg_reg_quad_batch_1d`: per-pair R tables built ONCE, only
  `exp(-jkR)` carries the k axis; returns `(n_k, N, N)` J tensors.
  Image blocks batch the same way.
- C++ `assemble_Z` / `assemble_Z_general` (junction support arrays)
  assemble `(n_k, n_basis, n_basis)` Z in one call.
- One batched `np.linalg.solve` over all k; junctions via the batched
  Schur solves `_solve_with_kcl_batch` / `_solve_with_kcl_swept_ports`.

BSpline's `compute_impedance_swept` says it itself: "no batched assembly
here yet". Only the same-edge reg moments are hoisted
(`_same_edge_prep` + `_seg_seg_reg_moments_from_geometry_swept`); the
all-pairs off-edge quadrature, assembly, ground image, and LU all re-run
per k in Python.

Measured (straight dipole, 41-point sweep, this machine):

| nsegs | triangular | bspline d=1 | bspline d=2 |
|---|---|---|---|
| 21 | 3.3 ms | 14.2 ms | 19.3 ms |
| 41 | 13.6 ms | 25.4 ms | 33.3 ms |
| 101 | 149 ms | 1916 ms | 1721 ms |

At interactive segment counts the gap is invisible; it opens ~13× at
N=101 and grows with N² per k-point. The exposed user path is
antennaknobs `impedance_sweep` / Y-matrix sweeps on LARGE geometries —
skyloop now requires n_per_wire=100 (4 wires ⇒ ~400 segments), hexbeam
and fan-dipole land in the hundreds too. That's the capability to
transfer before deletion. (BSpline d=1 already reproduces the tent basis
itself — same Z to ~0.1 Ω on the dipole — so nothing about the BASIS
needs transferring, only the sweep machinery.)

## Phase 0 — baselines and honest sizing

- [ ] Measure swept latency at the actual worst production shapes
      (skyloop n_per_wire=100, hexbeam, 5-band fan dipole with
      junctions) for triangular vs bspline d=1/d=2, sweep sizes the
      frontend actually sends. This sets the acceptance number for
      Phase 3 and confirms the port is worth C++ work (default-cost
      audit discipline — measure the request path first).
- [ ] Parity pin tests while both solvers still exist: bspline d=1 vs
      triangular on dipole, V-dipole, yagi, K=3 fan-dipole junction,
      PEC-ground dipole, multi-feed Y-matrix — asserting the ~0.1 Ω
      cross-basis floor. These are the transfer-of-trust artifacts;
      after deletion they survive as d=1-vs-pinned-value tests (the
      hentenna arbitration test already does this for d=2).

## Phase 1 — k-batched off-edge moments kernel

- [ ] C++ `seg_seg_full_moments_bspline_batch`: same structure as
      `seg_seg_quad_batch_3d` (per-pair R[q,r] table built once, k loop
      innermost) but producing the `(d+1, d+1, N, N)` moment tensors per
      k. Output memory is the constraint — d=2 has 9 tensors vs
      triangular's 4 (41 k × 300² × 16 B × 9 ≈ 530 MB) — so the driver
      chunks the k axis (reuse the `max_chunk_bytes=256 MB` convention
      from `_seg_seg_reg_moments_from_geometry_swept`).
- [ ] Batched image variant = same kernel with mirrored source
      endpoints (exactly how `_build_J_image_blocks` reuses the
      single-k kernel today).
- [ ] Pure-numpy reference path + bit-exact regression test vs the
      existing single-k `_seg_seg_full_moments_offedge` per k.

## Phase 2 — k-batched Z assembly

- [ ] C++ `assemble_Z_bspline_batch`: leading k axis + `omega_array`,
      mirroring what `assemble_Z` (triangular batch) does relative to
      the single-k form. The general support-array formulation already
      handles junction directionals, so ONE kernel serves both cases —
      no separate `assemble_Z_general` split needed on the bspline side.
- [ ] Batched weighted variant for refl-coef ground
      (`assemble_Z_bspline_weighted` + per-k w_A/w_Phi weight vectors —
      the weights are cheap ω-dependent scalars already computed per k
      by `_image_refl_weights`).
- [ ] Numpy fallback + bit-exact test vs per-k `_assemble_Z`.

## Phase 3 — swept drivers

- [ ] Rewrite `compute_impedance_swept` / `compute_y_matrix_swept`:
      batched J (Phase 1) + batched assembly (Phase 2) + ONE batched
      LU (`np.linalg.solve` on `(n_k, n_b, n_b)`), k-chunked to the
      memory cap. Junctions: port `_solve_with_kcl_batch` and
      `_solve_with_kcl_swept_ports` from triangular.py — they're
      basis-agnostic (pure Schur algebra on Z + kcl_A) and move
      verbatim.
- [ ] Sommerfeld ground stays per-k INSIDE the chunk loop (the grid is
      keyed on (ε̃, k) and the remainder fill is the dominant cost of
      that opt-in path anyway); enrichment stays gated with the
      existing NotImplementedError.
- [ ] Regression: swept == per-k loop to ~1e-10 on the Phase 0
      geometry set (mirror of `test_triangular_fandipole_swept_matches_
      per_freq`), including PEC + refl-coef ground and junctions.
- [ ] Acceptance: bspline d=2 swept latency within ~2× of the old
      triangular sweep on the Phase 0 production shapes.

## Phase 4 — delete Triangular

- [ ] Delete `triangular.py`, `_triangular_kernels.py`, the
      `TriangularSolver` export in `__init__.py`.
- [ ] Delete the now-dead C++ kernels: `seg_seg_quad_batch_3d`,
      `seg_seg_reg_quad_batch_1d`, `assemble_Z`, `assemble_Z_general`
      (grep first — the bspline/hmatrix `assemble_Z*` hits are the
      `assemble_Z_bspline*` names, not these).
- [ ] Tests: port the triangular-only coverage in `test_momwire.py`
      (junction K=2/K=3 equivalence, KCL, Y-matrix, swept-vs-per-freq,
      bent-wire, ground) to bspline d=1; drop pure duplicates bspline
      tests already cover; update `test_cancel.py` / `test_gil_release`
      entries.
- [ ] Migrate `validation/momwire_backend.py` + examples (hentenna,
      hexbeam, bowtie, yagi, feedline) and the `scripts/compare_*` /
      `probe_*` scripts to bspline; delete `profile_triangular.py` and
      probes that only existed to chase triangular-specific questions.
- [ ] De-stale docstrings that name TriangularSolver as "the default"
      (sinusoidal.py module docstring, bspline.py references) and add a
      retirement note to NEXT_STEPS.md (keep its history intact — it's
      a log, not living docs).

## Phase 5 — release

- [ ] momwire v0.9.0 via the /release skill (removal of a public class
      = minor bump under 0.x, called out in the PR title since PR
      titles are the release notes).
- [ ] antennaknobs follow-up PR: exact pin bump in the THREE places
      (pyproject, Dockerfile, submodule pointer), latency-smoke the
      sweep path on the Phase 0 shapes — it should get FASTER, verify
      it does. antennaknobs has no remaining TriangularSolver
      references (removed in v0.19.0), so the pin bump is the whole PR.

## Risks / notes

- Memory of batched (d+1)² moment tensors is the main new constraint —
  k-chunking bounds it, at the cost of rebuilding R tables per chunk
  (acceptable: the R build amortizes over the chunk's k count; measure
  in Phase 1).
- `compute_y_matrix_swept`'s per-k `B.T @ X` readout becomes a batched
  einsum — cheap, but keep the port-batched Schur path covered by a
  junctioned multi-feed test.
- Feed-convention difference between tent and d=1 bases (delta-gap
  projection) is already documented in the bspline docstring; the
  Phase 0 parity pins make it visible if it ever matters more than
  ~0.1 Ω.
