# momwire#1029 phase 0: a block-circulant solve for radial screens — plan and feasibility

Registered 2026-09-15, **before any feasibility measurement**. Two things have
run so far, and neither solves or assembles Z:

- the structure-only probe (`structure.jsonl`: sector grouping and sizes);
- the cost model, over existing profiles (`cost_model.json`).

- **Branch:** `feat/1029-rotational-symmetry` off momwire origin/main 227491d.
- **Dev mode:** `make build` in this worktree. No pointer moves.
- **antennaknobs** is untouched and only read through `PYTHONPATH`; its `src`
  is at 97ca2b6b0.
- **Phase 0 ends with a report.** No solver-route code is written before Steve
  decides on phase 1.

A literal miss goes to Steve for a ruling; it is never re-read silently.

## Why

- **The speed gap.** On the buried radial screen (`verticals.buried_radial_vertical`,
  soil 13 / 0.005, bs2 degree 2), momwire's warm solve is 5.43× slower than the
  clean-room NEC-5 build 3b75639 at 48 radials, 4.53× at 120 and 5.38× at 150
  (#1067, Skylake).
- **The dominant term.** In the warm profile (fea1ad0) the below-interface
  remainder projection is 56 % of self time at 150 radials, and ACA on it is
  closed negative (#982). By class, pair-proportional fill terms are 83 % of the
  warm solve at every rung (`cost_model.json`).
- **#1029's argument.** N identical radials make Z block-circulant in the sector
  index. A DFT over sectors turns it into N blocks the size of one sector, so
  pairs drop from (N·m)² to about N·m² and the solve from (N·m)³ to N·m³.

## 1. The symmetry rule

A deck qualifies for the sector route only if all of the following hold. They are
checked from geometry and specs before any fill.

1. **An axis.** A vertical line exists through which every wire of the axial group
   passes: every vertex within `tol` of the line. On this design that group is the
   rise, the feed gap and the radiator.
2. **N ≥ 2 sectors.** The remaining wires split into N groups, and rotation by
   2π/N about the axis maps group s onto group s + 1: every vertex within `tol`, in
   the same vertex order. `tol` = 1e-9 × the deck's largest extent.
   The structure probe measures the worst mismatch at 3.9e-16 m (4 radials) and
   4.4e-15 m (12 radials) on 6.33 m radials.
3. **Identical discretisation and material.** The same per-edge segment counts and
   grading, the same radius, conductivity and jacket per sector wire. Junction
   membership must be rotation-covariant: a junction maps to a junction, and a
   junction on the axis maps to itself.
4. **Loads and ports** sit on the axis, or come as N identical rotated copies. See
   §2 for which harmonics each case needs.
5. **The ground.** It qualifies when it is invariant under rotation about a
   vertical axis:
   - free space;
   - a PEC ground, a reflection-coefficient ground, or a Sommerfeld half-space,
     each with the axis normal to the interface.

   It does not qualify for faceted terrain, a two-media or wedge ground, a tilted
   axis, or anything else that is not axisymmetric.
6. **The fill must be rotation-invariant too,** not only the geometry. Every term
   has to depend on relative geometry (ρ, z, z′) alone:
   - the below-interface remainder columns follow the per-(ρ, z′) column rule
     (#893), which is invariant by construction;
   - the crossing fill sits on the axis.

   **At risk:** anything that selects quadrature orders, chunks, or ACA / H-matrix
   clusters from Cartesian coordinates. `_aca.py:build_cluster_tree` appears in the
   150-radial profile. F1 measures this directly.

## 2. Excitation

- **An on-axis feed** (the base-fed vertical; the gap is on the mast) has a
  right-hand side that is zero on every radial. It lives on the axial group, so
  under the DFT it excites **harmonic 0 only**. The hub KCL row's radial entries
  are equal across sectors, so its columns transform to harmonic 0 only as well.
  - For Z_in, currents and the far field, only the harmonic-0 block
    K₀ = [[Λ₀, √N B], [√N C_M, Z_MM]] needs factoring: (m + p)³.
  - The fill still needs sector 0's rows against every sector, for
    Λ₀ = Σ_d C_d, and the mast rows.
- **A general drive** needs all N harmonics: an off-axis port, a multi-port Y
  matrix, a plane wave, or anything whose right-hand side is not rotation-invariant.
  - The cost is N blocks of m³ plus one of (m + p)³.
  - Rotated copies of a port are served. A load that is not N-fold symmetric breaks
    the route: refused on the opt-in route, dense on the automatic one.

## 3. Hub, crossing node and mast: how they decompose

- **bs2's junction treatment**, as read at 227491d:
  - each wire keeps a value-1 directional basis at a junction end;
  - KCL is a Lagrange row (+1 / −1 per directional basis) closed by the Schur step
    in `_solve_with_kcl`;
  - grounded junctions drop their row, and a two-radius crossing junction keeps it
    (`_kcl_row_junctions`);
  - no basis function is shared between wires.
- **On this design** (the connected convention), the structure probe finds:
  - **one KCL row,** the hub: the N radials' directional start bases plus the rise's;
  - **no row at the z = 0 crossing node,** where the rise meets the feed gap: #980
    D3's two-k node recipe, entirely on the axis;
  - m = 55 unknowns per radial, and p = 35 on the axis (the rise 8, the gap and
    radiator 27);
  - 4 radials give 255 unknowns and 12 give 695.
- **The decomposition:**
  - sector s is radial s's 55 bases, in wire order; M is the axial 35;
  - Z⁻¹ is applied per harmonic, with harmonic 0 carrying M;
  - the KCL rows stay dense (one row) and are closed by the same Schur step;
  - the crossing node's recipe lives entirely inside Z_MM, B and C_M, and is not
    touched.
- **What fails the check:** a junction joining radials to each other away from the
  axis, or radials graded differently.

## 4. Which solver goes first

bs2 (`BSplineSolver`, degree 2). It is the default basis, and the one #1067 and
fea1ad0 time. razor-2p and sinusoidal follow only after G1–G4 pass on bs2.

## 5. Where the route is selected

- **Opt-in first.** A solver keyword (working name `rotational_symmetry=True`) runs
  the §1 check and **refuses by name** when it fails, naming the first failing
  condition (for example "radial 7 is 1.0 % longer than radial 0"). This is G2's
  opt-in branch.
- **Automatic later,** only when the check passes, falling back to the dense path
  with no refusal otherwise (G2's automatic branch).
- **The default path is bit-identical (G4).** Without the keyword, no new code runs.

## 6. Feasibility (this phase; no new fill code)

`feasibility.py`, on `verticals.buried_radial_vertical` at 4 and 12 radials, with
the design's defaults, soil `("finite", 13.0, 0.005)`, bs2 degree 2, and
OMP / OpenBLAS = 4.

- **What it takes from the solver:** the dense Z, the feed vector, the port vector
  and the KCL rows, through `_build_geometry` → `_build_basis_polynomials` →
  `_compute_Z_operator` → `_feed_drive_and_readout`, the calls
  `compute_impedance` makes.
- **The reference:** `_solve_with_kcl` on that same Z.
- **What it measures:**
  - ε_rr = max over s, d of \|Z[s, s+d] − Z[s+1, s+1+d]\|, over max \|Z\| of the
    radial blocks;
  - the same shift test on the mast rows and columns (ε_rM, ε_Mr), the feed and
    port vectors, and the KCL columns;
  - the per-harmonic solve on the sector-averaged operator plus the Schur step,
    against the dense solve, on Z_in and on the currents;
  - the harmonic content of the solution.

| id | what | bar | prediction |
|---|---|---|---|
| **F1a** | the radial blocks are block-circulant to machine precision | ε_rr ≤ 1e-9 at 4 and at 12 radials | hit |
| **F1b** | and at least to a loose tolerance (the premise) | ε_rr ≤ 1e-5 at 4 and at 12 radials | hit |
| **F2** | the axial coupling is sector-invariant, and the drive sits on the axis | ε_rM ≤ 1e-9 and ε_Mr ≤ 1e-9; the feed and port vectors are exactly 0 on every radial; the KCL columns are identical across sectors (ε = 0) | hit |
| **F3** | the per-harmonic solve reproduces the dense one | \|Z_in,sector − Z_in,dense\| / \|Z_in,dense\| ≤ 1e-7, and max \|c_sector − c_dense\| / max \|c_dense\| ≤ 1e-6, at 4 and at 12 radials | hit |
| **F4** | an on-axis feed excites harmonic 0 only | max over h ≠ 0 of ‖ĉ_h‖ / ‖ĉ₀‖ ≤ 1e-9 | hit |
| **F5** | the extraction is the engine's own path | \|Z_in,dense − Z_in,engine\| / \|Z_in,engine\| ≤ 1e-10 (enrichment is off, "raw") | hit |

- **The reasoning behind F1a.** The geometry rotates to about 4e-15 m, and every
  fill term is meant to depend on relative geometry only. A Cartesian-keyed choice
  (an ACA cluster tree, chunk boundaries that change a reduction order) would show
  up as about 1e-6 if it approximated, or about 1e-15 if it only reordered sums.
- **The reasoning behind F3.** A backward-stable dense solve and a per-harmonic one
  should differ by about cond · ε_rr, and MoM conditioning is 1e4–1e6.
- **The stop rule.**
  - **F1b misses:** the premise fails. Stop and report.
  - **F1a misses but F1b hits:** report where the non-invariance lives (radial
    blocks or axial rows) before any build is proposed.

## 7. The cost model

This is a model, not a measurement. `cost_model.py` reads the fea1ad0 warm
cProfiles and puts each function's self time in a class:

- **pairs:** the remainder projection, seg-seg moments, field-Galerkin and B-spline
  assembly, the crossing cross block and its glue. Scaled by (m + p) / (N m + p).
- **solve:** the dense LU. Scaled by ((m + p)³ + (N − 1) m³) / (N m + p)³.
- **threads** (lock waits) **and uncertain** (generic NumPy): bounded between the
  pair factor and unchanged.
- **fixed:** unchanged.

The totals are rescaled to #1067's momwire warm column. The model uses m = 54 and
p = 32 from #1067's segment column. The basis counts from the structure probe
(55 and 35) move the pair factors by under 3 %.

| radials | profile warm s | pairs / solve / threads / uncertain / fixed % | sector warm, predicted s | #1067 momwire warm s | #1067 3b75639 s | predicted / 3b75639 |
|---:|---:|---|---|---:|---:|---|
| 48 | 13.65 | 83.6 / 2.6 / 7.1 / 2.1 / 4.6 | **1.18 – 2.54** | 15.28 | 2.82 | 0.42 – 0.90 |
| 120 | 79.05 | 83.1 / 4.9 / 8.0 / 1.5 / 2.5 | **3.13 – 11.12** | 84.54 | 18.67 | 0.17 – 0.60 |
| 150 | 157.59 | 83.3 / 4.6 / 8.3 / 1.4 / 2.4 | **5.54 – 21.0** | 162.02 | 30.14 | 0.18 – 0.70 |

- **What shrinks by N:**
  - the remainder projection (88.7 s at 150);
  - seg-seg moments (13.4 s);
  - field-Galerkin assembly (9.1 s);
  - B-spline assembly (8.1 s);
  - the crossing cross block and its row weights (6.6 s);
  - the Sommerfeld interpolation glue.
- **What shrinks by about N²,** or to (m + p)³ for an on-axis drive: the solve,
  7.3 s at 150.
- **What does not shrink:**
  - the buried-serve plan;
  - basis polynomials;
  - the Sommerfeld tables;
  - geometry;
  - 2.4 % of other Python.
- **Bounded:** thread waits (8.3 %). They set the top of the range: 13.4 of the
  high estimate's 21.0 s at 150.
- **Not claimed:** that the 120 → 150 super-quadratic behaviour (local exponent
  3.46 in the brief) goes away, although a fill 95× smaller at 150 would plausibly
  leave the cache regime it comes from.
- **Not included:** the symmetry check's own cost, O(N·m) geometry; and the route's
  transforms and small solves, ≤ 0.03 s by flop count.

**Registered for G3** (the later build, not this phase): the sector route's warm
solve at 150 radials is ≤ 30.14 s (3b75639's column) and ≥ 5× faster than dense;
at 48 radials it is ≤ 3.0 s.

## 8. Gates for the later build (from #1029 and the brief)

- **G1.** Equal to the dense solve to the assembled tolerance on Z, currents and far
  field, at 4, 12 and 48 radials.
- **G2.** A non-symmetric perturbation (one radial 1 % longer) is refused by name on
  the opt-in route, or falls back to dense on the automatic route.
- **G3.** The cost ladder at 12, 24, 48, 96 and 150 radials, beside dense and beside
  #1067's NEC-5 columns, with RSS.
- **G4.** Bit-identical on non-symmetric decks and on the default path.

## Order

1. Commit this plan with `feasibility.py`, `cost_model.py`, `cost_model.json` and
   `structure.jsonl`, and push.
2. Run `feasibility.py --radials 4 12` under `systemd-run --user -p MemoryMax=24G`
   with `ulimit -v`.
3. `README.md` with the feasibility table. Report to Laptop-builder: this plan's
   SHA, the table and the cost model. Stop.
