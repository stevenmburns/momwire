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
- **G3a MISSES** at 3.327 s against a 3.0 s bar, and **G3b could not be run at
  all**: the 150-radial rung raises on both modes under the 8 GB cap this box
  is held to, at the crossing block Amendment 1 registered as un-restrictable
  (Amendment 5).

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
| **S-D** | **PASS** — 5473 passed, 5 skipped, 4 xfailed in 429 s on the final tree |
| **G1** | **PASS at 4 / 12 / 48 radials** (`g1.jsonl`), with Amendment 3 on G1c's bar |
| **G2** | **PASS** — the six registered sentences, one deck each, each at construction (`g2.json`, and `tests/test_rotational_symmetry_1029.py` is the gate) |
| **G3** | **G3c HIT** (RSS 0.982–0.997 of dense), **G3a MISS** (3.327 s against 3.0 s at 48 radials), **G3b NOT MEASURABLE** on this box (`g3.jsonl`). The ladder is below |
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

Z_in agrees between the two routes at 4.8e-13 / 6.2e-13 / 9.1e-13 / 1.9e-12
across the four measured rungs.

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
