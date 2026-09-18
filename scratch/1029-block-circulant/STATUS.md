# momwire#1029 phase 2 / #1109: where this branch stands

Rewritten 2026-09-18 on the day the gates ran. Phase 1's status is kept below
under its own heading, unedited; `PLAN-phase2.md` is the registration and
carries Amendment 1 (unit C landed half, and why).

**Everything registered is built and gated, and every bar is hit.** Four units
on `fix/1109-crossing-columns`, off momwire main 5bfc731 (rebased onto the
0.59.0 bump before the PR):

| commit | unit |
|---|---|
| `ca59663` | the registration, the attribution probe and its three records |
| `83417ab` | **A** — the crossing axis carries its basis samples as a CSR |
| `d2a4262` | **B** — `rows=` reaches the crossing family; the contract hardens |
| `f233aaa` | **C** — the ends' and completions' full-size transients go (the field block's stays: Amendment 1) |
| `c32a534` | **D** — the route's algebra moves beside its check |
| `8d3ef0b` | the laptop records: P2-6, P2-7, P2-8 and the probe |
| (this commit) | the Skylake and laptop ladder records, and this rewrite |

## What the memory was, and what it is

Attributed phase by phase on the Skylake box at 150 radials (`p2mem_150_*`),
the 8.55 GB peak was: `axis_data`'s DENSE basis samples 4.13 GB, `_main_split`
2.60 GB (t_main + the far-direct blocks' dense row weights, 528 MB × 4),
`self_completions` 1.99 GB, the ends' own block 1.06 GB, each field block
1.05 GB, Z 1.1 GB. The `_row_weights` line momwire#1109 named was the straw,
not the floor. After phase 2 the crossing family builds no `(n, n)` array on
either path; what is left at the peak is Z, `t_main`, the field block's `Q`
(Amendment 1) and the batched designed-tables window.

## Gate table

| gate | bar | measured | |
|---|---|---|---|
| **P2-1** route peak RSS @150, Skylake | ≤ 2.5 GB (from 8394 MB) | **2311.5 MB** | HIT (band 1.3–2.0 missed high; Amendment 1 explains the 1.05 GB) |
| **P2-2** dense peak RSS @150, Skylake | ≤ 4.0 GB (from 8424 MB) | **3366.4 MB** | HIT |
| **P2-3** route warm s @150 / @48, Skylake | ≤ 18.15 / ≤ 2.63 | **6.998 / 1.661** | HIT, 2.6× / 1.6× under phase 1 |
| **P2-4** dense warm s @150, Skylake | ≤ 131 | **116.2** | HIT |
| **P2-5** the 150 rung on the LAPTOP under `prlimit --as=8G` | runs (phase 1: REFUSED both modes) | **route 9.53 s / 2308 MB; dense 208.8 s / 3390 MB** | HIT |
| **P2-6** route = dense, `g1_route.py` 4 / 12 / 48, Skylake | G1a ≤ 1e-9, G1b ≤ 1e-8, G1c ≤ 1e-8 | 4.2e-13 / 4.4e-13 / 2.4e-13; 6.8e-13 / 6.7e-13 / 3.9e-13; 6.1e-13 / 6.1e-13 / 2.0e-13 | HIT (laptop rows in `p2_g1_xps13.jsonl` agree) |
| **P2-7** default path vs main, 4 and 12 radials | ≤ 1e-12 on Z, coefficients, Z_in | Z **7.8e-18**; coefficients 3.0e-14 / 5.9e-14; Z_in 3.0e-14 / 5.9e-14 | HIT; the sha moved ONCE, at unit A (`4a15d55c…`→`58124baf…`, `a28d9ef6…`→`af44d749…`); B, C and D each bit-identical to the unit before |
| **P2-8** the `rows=` contract, `s_ab_gates.py --sb` | non-requested rows exactly 0; requested rows ≤ 1e-12; plan equal | requested rows **0.0** from the full fill; non-requested **0.0**; `rows=[]` fill **0.0**; plan and `q_factor` equal | HIT |
| **P2-9** lint; default lane; the crossing suites; `make crossgate` on Skylake | green | ruff check + format clean; default lane **5521 passed / 5 skipped / 4 xfailed** (383 s, laptop); 17 crossing/route files green after every unit; crossgate **38 passed** (274 s, Skylake) | HIT |

Pinned constants byte-identical to `ca59663`: `_NEAR_Q`, `_FAR_Q`, `_ADM_ETA`,
`_CLUSTER_LEAF_SEGS`, `_ACA_COST_GUARD`, `_ACA_TOL`, `_CROSS_RTOL`,
`_CORNER_RTOL`, `_FAR_GROWTH`, `_NEAR_GROWTH`, `_PLANE_TOL`.

## The ladder, phase 2 against phase 1, on the same box

**smburns-Z170-WS, i7-6700K, 8 threads, 40 GB cap, OMP 8, AVX2 accelerator**
(`p2_g3_skylake.jsonl` against `g3_skylake.jsonl`), warm seconds and peak RSS:

| radials | dense s ph1 → ph2 | route s ph1 → ph2 | speedup ph1 → ph2 | dense MB ph1 → ph2 | route MB ph1 → ph2 | ΔZ_in |
|---:|---|---|---|---|---|---:|
| 12 | 1.260 → 1.248 | 0.676 → **0.610** | 1.86× → 2.05× | 263 → 236 | 259 → 224 | 6.8e-13 |
| 24 | 3.171 → 3.004 | 1.083 → **0.915** | 2.93× → 3.28× | 487 → 454 | 476 → 346 | 5.1e-13 |
| 48 | 10.983 → 9.941 | 2.627 → **1.661** | 4.18× → 5.98× | 1065 → 641 | 1044 → 626 | 6.1e-13 |
| 96 | 48.282 → 42.512 | 8.187 → **3.636** | 5.90× → 11.69× | 3582 → 1550 | 3575 → 1231 | 8.5e-13 |
| 150 | 130.933 → **116.179** | 18.148 → **6.998** | 7.21× → **16.60×** | 8424 → **3366** | 8394 → **2312** | 5.9e-13 |

Fill / solve at 150: dense 108.579 / 7.600 s, route 6.982 / 0.0157 s. The
route's fill is now linear in N from 12 to 150 (0.61 → 7.0 s over 12.5× the
unknowns) and the crossing family no longer scales it: what remains is the
narrowed pair fills and the designed tables.

**And on the laptop (xps13, i7-8550U, 8 GB address-space cap, OMP 4),
`p2_g3_xps13_150.jsonl`:** the rung phase 1 could not run at all: dense
208.8 s / 3390 MB, route 9.53 s / 2308 MB, 21.9×, ΔZ_in 2.7e-12. **A 150-radial
buried screen solves on a 16 GB laptop in ten seconds.**

## What the route is now

Unchanged in what it computes (P2-6, identical digits to phase 1 at 4 and 12
radials); changed in where and how. The algebra lives in
`_rotational_symmetry.{dof_groups, observer_rows, harmonic_zero_block, solve,
compute_impedance}` beside the check, with `BSplineSolver`'s five methods as
one-line delegations (unit D). The buried fill under `rows=` now leaves every
row outside the request EXACTLY zero: the crossing family answers
`(t[R, :], t[:, R])` from one evaluation of the kernel tables and one ACA
factorisation per far block (a spy gate asserts the call counts are equal
with and without `rows=`), and the routing composes
`Z[R] -= t_r; Z[R] -= t_c.T; Z[R] += s_r`. `R` is derived from the segment
request through the polynomial mask, all-or-nothing per basis, refused by
name otherwise.

## Follow-ups (filed or to file; none in this PR)

1. **The field-block accumulation** (Amendment 1): a `scale` argument on
   `assemble_field_galerkin`, a strided or refused target instead of the
   silent C-contiguous copy, and then `_field_galerkin_block(out=Z)` under the
   1e-12 gate. Removes the last `(n, n)` transient beyond Z on the route
   (≈ 1.05 GB at 150) and one of two on the dense path.
2. **The compact Z** for the route, `(|R|, n)`: the four narrowed fills and the
   chunked accumulator write into a full Z by observer index. With 1 above,
   the route's floor would be the tables window.
3. **The batched designed-tables window** in `_main_split` (≈ 4 × the largest
   `_tables` output) is the route's next peak term; chunk the direct batch
   over blocks.
4. `_ends_and_corner_reversed` (razor / SG only) still allocates its own
   `(n, n)`; the same `out=` treatment if razor is ever run at scale.
5. Six historical scratch probes (`813-reversed-block/probe{1,3,4,5}*`,
   `813-node-derivations/probe1_sw_by_parts.py`, `936-study/probe10_fd_correction.py`)
   read the removed `ax["F"]` keys. Left as the record of their own commits;
   they need `.toarray()` to re-run.
6. The unburied deck and the general drive (phase 3), as `PLAN-phase2.md` §1
   lists them.

---

# Phase 1 (2026-09-17), kept as written

# momwire#1029 phase 1: where this branch stands

Rewritten 2026-09-17 on the day the route landed. It supersedes the earlier
stopping points, which are in history at 9b1e051, e3bb8aa and 0ede3de; the
amendment trail below is what changed and when.

**The route is built and every registered gate has run but one.** `rows=`
landed earlier and is gated by S-0, S-A, S-B, S-C and S-D; the opt-in
`rotational_symmetry=True` route is gated by G1, G2, G3a/G3c and G4.

Three registered predictions moved, and all three are written down in
`PLAN-phase1.md` as amendments rather than re-read here:

- **G1c's bar** is not a measurement under one reading of it (Amendment 3);
- **refusal 5 has no deck** — this solver cannot express a non-axisymmetric
  ground, so its G2 case is a seam (Amendment 4);
- **G3a and G3b** read as a MISS and as unmeasurable ON THE LAPTOP (Amendment
  5), and both are **HITS re-measured on the Skylake box** that #1067's bars
  came from (Amendment 6): 2.627 s at 48 radials against 3.0 s, and 18.148 s
  at 150 against 30.14 s at **7.21× faster than dense**.

## The branch

`feat/1029-rotational-symmetry`, based on momwire 227491d, with main merged in
twice: a6a67f93a at cf3aaab, and **37e8257c0 (v0.58.0) at 91da54f**.

| commit | what |
|---|---|
| 97e0c13, 8d42ea7 | phase 0's registration and records |
| 9b1e051, e3bb8aa | the first stopping point, and why a resumed branch re-baselines |
| cf3aaab | merge of momwire main a6a67f93a |
| c38790b | **phase 1's registration**, before any measurement |
| d85ee24 | **gate S-0 passes** |
| dcdd8a7 | Amendment 1, the three `rows=` contract details |
| 9d51d53 | **the `rows=` patch**, with S-A and S-B and Amendment 2 |
| 0d90313, 0ede3de | the S-D scare and its resolution (a stale build on this box) |
| 91da54f | **merge of momwire main 37e8257c0 (v0.58.0)** |
| c54778c | **the re-baseline on 0.58.0, and S-C's numeric half** |
| cf99adf | **the route**: the opt-in flag, the check, the six refusals, the harmonic-0 solve, and the tests |
| (this commit) | the G1–G4 records, Amendments 3, 4 and 5, and this rewrite |

The antennaknobs submodule pointer has not been moved.

## What the route is

`rotational_symmetry=True` on `BSplineSolver`. Default off, and off changes
nothing: `_rotational_map` is None, no attribute the dense path reads moves,
and no route code runs (G4).

- **The check** is `src/momwire/_rotational_symmetry.py` and runs **at
  construction**. Every condition is a geometry, material or drive fact frozen
  there, so no fill could rescue a deck that fails one.
- **The grouping.** Wires on the axis are the axial group; everything else
  falls into orbits of size N under rotation by 2π/N. That reading is tried
  first; only when what is left is not a screen does the axial group GROW
  across degree-2 junctions, and the grown group is then held to condition 1.
  The growth is what makes a tilted mast fail condition 1 by its own name
  instead of reading as a spare sector; ordering it as a fallback is what
  keeps a one-radial deck refused for having fewer than two sectors.
- **The fill** is one call to the landed `rows=` restriction with sector 0's
  wires plus every axial wire. No basis straddles two wires (S-0's S0a), so a
  whole-wire segment set is exactly a whole-dof row set.
- **The solve** is the harmonic-0 block
  `K_0 = [[Λ_0, √N B], [√N C, Z_MM]]`, size (m + p) whatever N is, closed by
  `_solve_with_kcl`'s own Schur step. Λ_0 is the row sum of sector 0's rows
  over every sector's columns; the axis-against-sector block is the SUM of the
  N copies over √N, which averages their roundoff and makes their spread a
  free check on the sector assignment.
- **The sector labelling is free**, and the route leans on that: Λ_0 sums a
  row over a source dof's whole orbit, so any transversal of the orbits is a
  fundamental domain. What is not free is a dof's POSITION inside its sector,
  which is what S-0 pinned dof by dof.
- **The readout needs no new code.** The coefficients come back in the dense
  ordering, so `element_currents` → `_far_moments` is the same call.
- **Every entry point but `compute_impedance` refuses** rather than quietly
  handing back the dense answer — `require_lattice_fft`'s lesson on the array
  solver.

## Gate table

| gate | result |
|---|---|
| **S-0** | **PASS** at 4 / 12 / 48 radials (`s0.jsonl`). No shared dofs (255 / 695 / 2675 = N x 55 + 35); the sector map is a bijection on the dof set; every dof matches its image's kind, local index, end position and junction; the 35 axial dofs are fixed points; the KCL row is invariant with max change exactly 0.0; geometry rotates onto itself by 3.9e-16 / 4.4e-15 / 6.4e-15 m against a 1.06e-8 m tolerance |
| **S-A** | **PASS, bit-identical, RE-BASELINED on momwire 0.58.0** — see amendment B |
| **S-B** | **PASS on four assertions together** (`sb.json`): the requested rows equal the full fill (0.0); the non-requested rows equal a `rows=[]` control exactly (0.0); the fills really narrowed (below-class 222 -> 60 observers at 4 radials, 654 -> 60 at 12, removing 1245.0 from the non-requested rows against 123007.5 of pair-class work on the requested ones); `q_factor` and every plan field equal |
| **S-C** | **PASS, both halves.** Static: only `bspline.py` calls `compute_Z_operator_buried` and nothing passes `rows=`. Numeric (`sc.json` against `sc_pre.json`): three SG decks covering every buried pair class, bit-identical |
| **S-D** | **PASS on the laptop** — 5473 passed, 5 skipped, 4 xfailed in 450 s. On the Skylake box: 5469 passed, 8 skipped, 4 xfailed, **1 failed in 236 s** — `test_same_edge_window_968`, a PRE-EXISTING toolchain pin unrelated to this branch; see the box section below |
| **G1** | **PASS at 4 / 12 / 48 radials on BOTH boxes** (`g1_xps13.jsonl`, `g1_skylake.jsonl`), with Amendment 3 on G1c's bar |
| **G2** | **PASS** — the six registered sentences, one deck each, each at construction (`g2.json`, and `tests/test_rotational_symmetry_1029.py` is the gate) |
| **G3** | **ALL THREE HIT on the Skylake box** — G3a 2.627 s ≤ 3.0 s, G3b 18.148 s ≤ 30.14 s at 7.21×, G3c 0.978–0.998 of dense (`g3_skylake.jsonl`). On the laptop G3a missed and G3b could not run at all (`g3_xps13.jsonl`, Amendment 5). Both ladders are below |
| **G4** | **PASS** — the default path is bit-identical to merged main (`g4_default_path.json` against `sa_pre_0580.json`), and the lane is green |

## G1 — the route against the dense solve

The same deck built twice, once with the flag and once without, both answers
through `compute_impedance`'s own return (`g1_route.py`, `g1.jsonl`).

| radials | unknowns | G1a Z_in | G1b currents | G1c m_theta | G1c m_phi | copy spread | dense s | route s | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 255 | 2.06e-13 | 2.06e-13 | 8.7e-14 | 6.3e-15 | 3.4e-19 | 1.40 | 0.31 | 4.5x |
| 12 | 695 | 4.79e-13 | 5.07e-13 | 4.5e-13 | 1.7e-14 | 7.8e-18 | 1.47 | 0.77 | 1.9x |
| 48 | 2675 | 9.10e-13 | 9.07e-13 | 3.4e-13 | 1.6e-14 | 1.2e-16 | 17.12 | 3.42 | 5.0x |

Bars: G1a 1e-9, G1b 1e-8, G1c 1e-8. Every cell is two to four orders inside
its bar. **G1c's bar needed Amendment 3** — |m_phi| is the N-fold azimuthal
harmonic and is at the roundoff floor above N = 4, so it is graded against the
pattern's own scale; both readings are in the record.

**G1d, the entry-by-entry census at 48 radials**, in the route's own sector
grouping, each entry graded against its own magnitude above a 1e-9 x max floor:
worst 1.88e-9 against a 1e-7 bar, on an entry of 2.08e-5 ohm where the block's
largest is 2.06e4. Phase 0 measured 1.7e-9 at 12 radials, so the
non-invariance has NOT grown with N — no size-dependent fill path switches on
between 12 and 48.

## G2 — the six refusals, as they read

One deck each, every one raising at construction. The text is
`g2.json`; the GATE is `tests/test_rotational_symmetry_1029.py`, which pins
each sentence and runs on every PR. All six begin "rotational symmetry: " and
end "Drop rotational_symmetry=True to solve this deck densely."

1. `sector 2's wire is +1.00 % longer than sector 0's (6.397 m against 6.334 m). Every sector must map onto the next under rotation by 2*pi/N.`
2. `sector 3 sits at 272.500 deg, where 4 sectors require 270.000 deg (tolerance 1e-09 x 10.6 m). Every sector must map onto the next under rotation by 2*pi/N.`
3. `port 'feed 0' sits at (3.000, 0.000, -0.150), off the symmetry axis (0.000, 0.000). An off-axis drive excites every harmonic, and this route serves the axis-symmetric drive only.`
4. `sector 1's conductor radius is 0.800 mm against sector 0's 1.000 mm. Every sector must carry the same radius, conductivity and jacket.`
5. `the ground model 'terrain' is not invariant under rotation about the axis. Free space, PEC, a reflection-coefficient ground and a Sommerfeld half-space qualify.`
6. `the axial group's wires are not parallel to z (wire 5 runs 0.217 deg off). The symmetry axis must be normal to the interface.`

Three more are recorded beside them because they are what a user will
actually hit: a one-radial screen (`N >= 2`), singular enrichment, and an
unburied deck. Plus the route's one RUNTIME guard, which no deck can reach at
construction: `the right-hand side differs between sector 0 and sector 1 by
1.000e+00 of its largest entry, so the drive is not rotation-invariant and
needs every harmonic.`

**Amendment 4 records that refusal 5 has no deck** — `BSplineSolver` cannot
express a non-axisymmetric ground — so its G2 case is a one-line subclass
answering a name the whitelist does not carry, and a companion test asserts
that every ground the family CAN express qualifies.

## G3 — the cost ladder

Each rung in its own process (peak RSS is a high-water mark and cannot be
reset), one at a time, `OMP_NUM_THREADS=4`, under `prlimit --as=8G`
(`g3_ladder.py`, `g3.jsonl`). A row is a COLD solve to fill the module caches
followed by the timed WARM one, with the fill and the solve timed apart.

| radials | unknowns | dense warm s | route warm s | speedup | dense fill / solve | route fill / solve | dense RSS MB | route RSS MB | RSS ratio |
|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| 12 | 695 | 1.552 | 0.702 | 2.21x | 1.531 / 0.021 | 0.700 / 0.0014 | 261.6 | 256.9 | 0.982 |
| 24 | 1355 | 4.744 | 1.357 | 3.50x | 4.605 / 0.139 | 1.355 / 0.0022 | 482.0 | 474.9 | 0.985 |
| 48 | 2675 | 18.566 | **3.327** | 5.58x | 18.006 / 0.560 | 3.323 / 0.0043 | 1051.2 | 1047.6 | 0.997 |
| 96 | 5315 | 77.112 | 10.776 | 7.16x | 73.696 / 3.416 | 10.765 / 0.0115 | 3597.1 | 3564.7 | 0.991 |
| 150 | 8285 | REFUSED | REFUSED | — | — | — | — | — | — |

**And the same ladder on the Skylake box** — #1067's own, 40 GB cap, OMP 8
(`g3_skylake.jsonl`). This is the one to read against the bars:

| radials | dense warm s | route warm s | speedup | dense fill / solve | route fill / solve | dense RSS MB | route RSS MB | RSS ratio |
|---:|---:|---:|---:|---|---|---:|---:|---:|
| 12 | 1.260 | 0.676 | 1.86x | 1.190 / 0.070 | 0.674 / 0.0016 | 263.0 | 259.0 | 0.985 |
| 24 | 3.171 | 1.083 | 2.93x | 3.070 / 0.101 | 1.081 / 0.0027 | 487.1 | 476.4 | 0.978 |
| 48 | 10.983 | **2.627** | 4.18x | 10.630 / 0.353 | 2.623 / 0.0046 | 1064.6 | 1043.6 | 0.980 |
| 96 | 48.282 | 8.187 | 5.90x | 46.093 / 2.189 | 8.181 / 0.0063 | 3581.9 | 3575.1 | 0.998 |
| 150 | 130.933 | **18.148** | **7.21x** | 123.078 / 7.856 | 18.136 / 0.0117 | 8424.0 | 8394.3 | 0.996 |

**G3a HIT** (2.627 s ≤ 3.0 s), **G3b HIT on both halves** (18.148 s ≤ 30.14 s,
7.21x ≥ 5x), **G3c HIT** (0.978–0.998). 18.148 s lands inside Amendment 2's
registered 14–30 s band. Z_in agrees between the routes at 6.4e-13 / 5.0e-13 /
4.1e-13 / 1.1e-12 / 6.6e-13.

**Why the laptop could not do the 150 rung, confirmed:** dense peak RSS there
is **8424 MB**, above the laptop's whole 8 GB address-space cap. The route's
8394 MB is 0.996 of it — the crossing family is the memory floor and the route
cannot lower it, exactly as Amendments 1, 2 and 5 said.

Z_in agrees between the two routes at 4.8e-13 / 6.2e-13 / 9.1e-13 / 1.9e-12
across the four rungs the laptop measured.

- **G3c HIT.** Peak RSS is the dense RSS everywhere — 0.982 to 0.997 of it,
  inside the registered 5 %. Registered rather than discovered, and now
  measured: the restricted fill keeps Z full size, so phase 1 buys time and
  not memory.
- **G3a MISS.** 3.327 s at 48 radials against the 3.0 s bar. See Amendment 5:
  the bar is #1067's Skylake column and this box is 1.215x slower on the same
  deck, which puts the route at 2.74 s Skylake-equivalent — context, not a
  rescue. The bar is missed on the box it ran on.
- **G3b NOT MEASURABLE HERE.** The 150-radial rung raises on BOTH modes at the
  same line — `_crossing_fill._row_weights`, 124 MiB for a (2035, 7992)
  float64, at about 5.3 GB RSS — under the 8 GB cap this box is held to. Two
  attempts (the registered cold+warm pair, then a single-pass escape), no
  third. The registered ladder ran under 24 GB.
- **The 150-radial failure is Amendment 2's finding sharpened.** The crossing
  block is filled in FULL by contract, so `rows=` narrows nothing there and
  the route hits the identical allocation. That family is the memory FLOOR,
  which is also why G3c reads 0.99. Extrapolating the route's linear-in-N fill
  from 96 gives about 16.8 s at 150, inside Amendment 2's 14-30 s band — an
  extrapolation from four rungs, not a measurement.
- **The 5x claim IS measured** from 48 radials up: 5.58x and 7.16x, still
  climbing.
- **The solve is gone.** Dense spends 3.42 s in the solve at 96 radials; the
  route spends 0.0115 s, because the harmonic-0 block is (m + p) = 90 whatever
  N is. Everything the route still pays is fill.

## Which box ran what, and the one red test

Execution moved to the Skylake box mid-build; the worktree, the editing and
every commit stayed on the laptop. `box.py` reads the machine out of the
process — hostname, CPU, cap, thread policy, python, numpy, BLAS and which
accelerator variant resolved — and every G1/G3 record carries it, because a
ladder row that does not say which box it came from cannot be compared with
#1067's bars.

| gate | laptop (xps13, i7-8550U, 8 GB cap, OMP 4) | Skylake (smburns-Z170-WS, i7-6700K @ 4.0 GHz, 40 GB cap, OMP 8) |
|---|---|---|
| S-0, S-A, S-B, S-C | ran, PASS | not re-run (bit-identity is per-toolchain; see below) |
| G2, G4 bit-identity | ran, PASS | not re-run |
| **default lane** | 5473 passed / 5 skipped / 4 xfailed, 450 s | 5469 / 8 / 4 **+ 1 failed**, 236 s |
| **G1 (4/12/48 + census)** | PASS | **PASS** |
| **G3 ladder (12…150)** | 150 rung REFUSED by the 8 GB cap | **complete, all bars hit** |

**The one red test is not this branch.**
`test_same_edge_window_968::test_the_square_entry_points_are_bit_identical_to_the_pre_change_build`
fails on the Skylake box and passes (by skipping) on the laptop. Chain of
evidence:

- this branch changes **four Python files and zero C++** — no `.cpp`, no
  header, no `setup.py`, nothing in `extern/` — so the accelerator built on
  the box is bit-for-bit what origin/main would produce there;
- the test compares banked `.npz` oracles against pure `_acc.*` accelerator
  entry points, and the failing key is `stat/3/1`, a closed-form static
  moment;
- measured: **all 6 static keys differ, worst 6.227e-15 relative** — the
  "5.3e-15 codegen drift" the test's own docstring says it exists to catch;
- the test guards itself with `_fingerprint()` = OS, machine, python, numpy.
  The bank reads `['Linux', 'x86_64', '3.14.5', '2.5.2']` and this box reads
  **exactly the same four values**, so the guard did not skip — **but the
  fingerprint does not include the COMPILER**, which is the one thing the
  docstring names as what moves these bits.

So a new Linux/x86_64 box with the same python and numpy but a different GCC
fails a test designed to skip in precisely that situation. That is a
momwire#968 issue, not a #1029 one; the follow-up is to put the compiler
(`platform.python_compiler()`, or the `CC`/`CXX` version) into `_fingerprint`.

**One other box lesson, measured:** the default lane took **910 s with 14
tests over the 20 s hard ceiling** when `OMP_NUM_THREADS=8` was exported, and
**236 s with none** when it was not. That is the trap `pyproject.toml`'s
`filterwarnings` comment documents — momwire is imported while pytest parses
the filters, before `conftest` can set `OMP_NUM_THREADS=1` per xdist worker,
so an exported width becomes a full OpenMP pool per worker. Export a thread
width for the single-process G1/G3 runs; never for the xdist lane.

## What is left

1. PR to momwire. Steve merges. No issue comments.
2. **razor-2p and sinusoidal are out of scope** and stay so: the route is
   bspline-side because the fill restriction is.
3. **Compaction** — the deferred step from Amendment 1's item 1 — is what
   would buy the memory phase 1 explicitly does not (G3c). Amendment 2 already
   records that it does nothing for the crossing family, whose cost is the
   transpose's columns and not the rows.
4. **The general drive** (all N harmonics, an off-axis port, a Y matrix) is
   phase 2's, and every entry point that would need it refuses by name today
   rather than answering densely.

**Afterwards, not started:** momwire#570 phase 3's contract derivation,
registered in `scratch/570-far-field/PLAN.md` on a branch, no issue comment.

## The amendment trail

## 2026-09-17 amendment B: momwire main 0.58.0 merged, and the baseline moved

`git merge origin/main` at **37e8257c0 (v0.58.0)** into the branch, merge commit
**91da54f**. No conflict: main's src work is deck/eznec/`_wire_loading` and
`bspline.py`'s new `distributed_rlc` kwarg, none of it in the two hunks this
branch owns. No C++ moved, so the linked `.so` files stay valid.

- **The default lane on the merged tree: 5453 passed, 5 skipped, 4 xfailed in
  431 s.** Green.
- **S-A RE-BASELINED and PASS, bit-identical** against a fresh worktree at
  37e8257c0 (`sa_pre_0580.json` / `sa_post_0580.json`; the dumps differ only in
  tree path and timing):

  | radials | sha_Z | sha_coeffs | Z_in hex |
  |---:|---|---|---|
  | 4 | `4a15d55c1f99aa128cc96d61fc147c05` | `df90967dd3a20c961b4777a768f560cf` | `0x1.38873ad7f3509p+6`, `0x1.72b38fffb6bc8p+5` |
  | 12 | `a28d9ef602d7e21dab8f05223507439d` | `5aabe80610819e7fa04de7ac3ad748ec` | `0x1.a4d80e230d17bp+5`, `0x1.329805532abc0p+5` |

- **The baseline hashes are NOT the ones the 9d51d53 record carries**
  (`fbba2249…` / `a818422f…`), and e3bb8aa's warning is why this was re-measured
  rather than re-read. **The mover is antennaknobs, not momwire.** Dumping at
  momwire a6a67f93a — the branch's own base — against today's antennaknobs
  (e4efc2bc2) reproduces the NEW hash `4a15d55c…` exactly, so nothing between
  a6a67f93a and 37e8257c0 touches these bits; the deck does. The size is
  roundoff: Z_in moved 78.13206040784942 → 78.13206040785384, 5.7e-14 relative.
- **S-C's numeric half: PASS, bit-identical** (`s_c_gate.py`, `sc.json` against
  `sc_pre.json`) on three sinusoidal-Galerkin decks covering every buried pair
  class — D1 fully-buried vertical and horizontal dipoles, and a D2 mixed deck.
  The operator's bytes, the coefficients' bytes and Z_in's hex all agree. SG's
  own fixtures are the decks, so this half needs no antennaknobs.

## 2026-09-17 amendment: S-D is GREEN, the red was this box

Re-run on the laptop from a FRESH worktree at 0d90313 with the current
accelerators linked in (the branch changes no C++, so main's `.so` files are
the right ones):

- `python -c "import momwire._below_interface"` — imports; the constant
  `MIN_CROSSING_NODE_SEPARATION_M` reads as 1.0.
- `make test`'s default lane (`pytest tests/`, xdist, `not slow and not memgate
  and not integration`): **5255 passed, 5 skipped, 4 xfailed in 423 s**. The
  files the red run named first (`test_crossing_serve_524`,
  `test_plan_extents_914`) pass.

So the 119 import-time `AttributeError`s were diagnostic 4 below — a stale
`.so` / `egg-info` in this box's shared build lineage — not the patch. The
S-D row in the gate table reads PASS from here on; the "What is left" list
starts at its step 2.
