# momwire#1064: confine U9's low-band cold-fill cost to decks under 0.05°

Registered 2026-09-15, **before any run**. Branch `perf/1064-fifth-theta-band`,
off main 54caf86 (src d6204e6). The records and the source change share one PR.

Every gate below has its bar and a blind prediction. A gate failure is a stop
and a report, not a re-spell. A missed prediction is reported as a miss and is
not re-registered.

## The decision, and the sizing it rests on

**The issue's spec:**
- a fifth band over [0.016667°, 0.05°];
- the low band back to [0.05°, 0.1°];
- 15 regions, a third edge argument, and `_N_BANDS` 4 → 5.

**What the spec leaves open, and the code forces.** A region needs four θ nodes
for its 4 × 4 Lagrange stencil.
- **Where the code needs it.** `_interp` and `proj_one_below` both clamp j0 into
  [0, n_th − 4].
- **The problem.** [0.016667°, 0.05°], at the low band's Δθ = 0.05/3, holds only
  three nodes.

**Two splits were sized.** The model is ≈ 6.4/tan θ tail panels per node,
counted per R₁ row, for the band fill only. It is arithmetic; nothing was run.

| layout | nodes under 0.1° | a deck in [0.05°, 0.1°] | a deck under 0.05° |
|---|---|---|---|
| main today (U9's single band) | 0.016667, 0.033333, 0.05, 0.066667, 0.083333, 0.1 | 53,904 | 53,904 |
| 0.55.0 | 0.05 … 0.1 | 20,901 | refused |
| split A: own Δθ = 0.011111°, 4 nodes ending on 0.05° | 0.016667, 0.027778, 0.038889, 0.05 + the low band | 20,901 (1.00× 0.55.0) | 72,867 (1.35× main), or 65,533 (1.22×) with the 0.05° node filled once |
| **split B: U9's cells, 4 nodes, one past the 0.05° edge** | **0.016667, 0.033333, 0.05, 0.066667 + the low band** | **20,901 (1.00× 0.55.0)** | 66,738 (1.24×) unshared; **53,904 (1.00× main) with both shared nodes filled once** |

**The decision was made upstream of this record, on 2026-09-15.** Build B if
it is cheap; otherwise build A, with the 0.05° node filled once. "Cheap" had
three conditions, and each is answered here from the code:

1. **Node reuse stays inside the fill layer.** **Yes.** It touches only
   `SommerfeldGridBelow._fill_region`.
   - Filling the floor band fills the low band first, as the owner, and copies
     its two columns.
   - The kernels read per-region tables and never see the sharing.
2. **A region's lattice may run past its routing edge, with no interface
   beyond the fifth band's own.** **Yes.**
   - **Routing and interpolation are separate.** `_interp` and `proj_one_below`
     route on the edges passed to them, and interpolate on each region's own
     th0, Δθ and n_th. Nothing derives an edge from a lattice.
   - **The only new interface** is the third edge (`th_band_floor_hi`), which
     the fifth band needs anyway.
3. **About a day of extra work at most.** **Yes.** It is about 30 lines in
   `_fill_region` and the constructor, plus tests.

**So B is built.**

## What B is

- **Bands, θ fastest:** `_BAND_FLOOR` 0, `_BAND_LO` 1, `_BAND_MID` 2,
  `_BAND_GRAZE` 3, `_BAND_STEEP` 4.
  - **Regions** are zone·5 + band: 0–4 inner, 5–9 near, 10–14 far.
- **Routing edges:** 0.05° (new), 0.1°, 1°, then the split. Every edge is a
  strict `<`, so θ = 0.05° routes to the low band.
- **The floor band's lattice.** th0 = 0.016667°, Δθ = 0.05/3, four nodes.
  - **What a query reads.** Every query routed here (θ < 0.05°) lands with
    j0 = 0.
  - **So its stencil is U9's.** It reads nodes 0–3 with the same weights as
    U9's single band gave it.
- **The low band's lattice.** th0 = 0.05°, Δθ = 0.05/3, four nodes. These are
  0.55.0's.
- **Shared nodes.** Floor nodes 2 and 3 are low-band nodes 0 and 1.
  - **The floats.** 0.05° is the same float in both bands. 0.066667° differs
    by one ulp in degrees: 0.06666666666666668 against 0.06666666666666667
    (arithmetic).
  - **Ownership.** The low band owns both nodes, so its values are the same
    bits whichever band fills first.
- **Deferral.** The floor band is deferred in every zone, like the other
  grazing bands.
  - **When it fills.** `_ensure_for` fills it when a query's θ interval reaches
    under 0.05°.
  - **The side effect.** Filling it also fills the low band. So a deck that
    reached under 0.05° and never 0.05°–0.1° would pay 8,067 panels more than
    main does. No deck on the timing list is such a deck.
- **Unchanged:** the 0.016667° floor, `_MAX_TAIL_PANELS` (24,000), the numpy
  contour, and the refusals.
- **C++.**
  - `proj_one_below` and `remainder_field_proj_batch_below` take
    `th_band_floor_hi` before `th_band_lo_hi`, as a pybind arg, with a
    five-way band select and a zone stride of 5.
  - The batch checks for 15 regions.
  - `build_grid_view` accepts 15 regions and carries `r_near` for them.
  - `MAX_REGIONS` is 15.

## Trees

| name | commit | worktree |
|---|---|---|
| branch | this PR | `momwire-wt-1064` |
| main | 54caf86 | `momwire-wt-1064-main` |
| 0.55.0 | 1ca8725 | `momwire-wt-1064-v055` |

- **Builds.** Each tree is built with `make build` (`MOMWIRE_REQUIRE_ACCEL=1`),
  and rebuilt after every checkout.
- **antennaknobs.** The catalog decks and the census use antennaknobs main, at
  a commit recorded with each run.

## Gates

| id | what | bar | prediction |
|---|---|---|---|
| **G1a** | **θ ≥ 0.05°, grid level, against 0.55.0.**<br>• `SommerfeldGridBelow.eval` (numpy) on soils A/B/C × 7/21 MHz, and soil A at 3.5 MHz.<br>• R₁/λ_m ∈ {0.02, 0.2, 1, 1.9, 3, 3.9}.<br>• θ at every low- and mid-band node, those bands' cell midpoints and thirds, and θ ∈ {1, 2, 5, 30, 60, 89.9}°.<br>• The C++ kernel on a pair set spanning the same (R₁, θ). | every surface value bit-identical to 0.55.0 | passes |
| **G1b** | **θ ≥ 0.05°, deck level.** Z from the G5 rows. | bit-identical:<br>• the four catalog decks, against main;<br>• `dipole935` (0.0582°) and `brv_corner` (0.0724°), against 0.55.0. | passes |
| **G2a** | **θ < 0.05°, grid level, against main.** G1a's queries over the floor band's nodes, midpoints and thirds, in every zone. Reports max \|Δ\|/\|main\|. | ≤ 4.7e-4, the band's interpolation bar | passes, max ≤ 1e-14. Same four nodes and same weights; only the copied 0.066667° column can differ, by its one-ulp spelling. |
| **G2b** | **The floor band against the direct surfaces** (the issue's re-measure).<br>• Cell midpoints and thirds of [0.016667°, 0.05°].<br>• Soils A/B/C × 7/21 MHz, and soil A at 3.5 MHz.<br>• R₁/λ_m ∈ {0.2, 1, 1.9, 3, 3.9}, all four surfaces. | ≤ 4.7e-4 | ≤ 1e-8. U9 read 3.9e-9 on these cells with this stencil. |
| **G2c** | **The low band against the direct surfaces,** by the same method over [0.05°, 0.1°]. | ≤ 4.7e-4 | ≤ 1e-8 |
| **G3** | **Dispatches across the seams.** `remainder_field_proj_below`:<br>• θ ∈ {0.016667, 0.02, 0.0333, 0.049999999, 0.05, 0.050000001, 0.06, 0.0666, 0.07, 0.099999999, 0.1, 0.100000001, 0.5}°;<br>• R₁ in each zone;<br>• soil A at 7 MHz and soil C at 21 MHz;<br>• C++ avx2, C++ sse2 (`MOMWIRE_FORCE_VARIANT`), and numpy. | pairwise ≤ 1e-12 relative | passes, ≤ 1e-13 |
| **G4** | **Each θ node filled once.** A counter wraps `iv_surfaces_direct_below` on real grids (soil A, 7 MHz, R₁ to the cap), filled floor-first and low-first on fresh grids. | every (zone, θ node) of the floor and low bands evaluated exactly once, in both orders; the floor band's own fill evaluates 2 columns | passes |
| **G5** | **Re-timing**, below. | • [0.05°, 0.1°] decks: cold, branch/0.55.0 in [0.92, 1.08].<br>• Decks under 0.05°: cold, branch/main in [0.92, 1.08].<br>• Catalog decks: cold, branch/main and branch/0.55.0 in [0.95, 1.05].<br>• Warm: [0.95, 1.05] everywhere. | passes |
| **G6** | **The standing buried census** (antennaknobs `scratch/956-census`), below. | a wholly buried design that moves, branch against main, is a stop | no Population A momwire cell moves (bit-identical at both rungs): no catalog design reaches under 0.1° |
| **G7** | **PR lanes.**<br>• `make test`.<br>• The slow and crossgate lanes on the touched test files, locally.<br>• `ci.yml` dispatched on the branch (macOS, slow, crossgate, memgate, integration). | green | green |

### G5: the re-timing

**The harness** is `t5_fill_cost.py`, run by `run_t5.sh`. It is U9 amendment
5's harness, with two decks added.
- **Processes.** One fresh process per (tree, deck, repeat). Cold is the first
  solve; warm is a second solve with the engine's cache cleared.
- **Order.** The three trees rotate order every repeat, over 3 repeats, and
  medians are reported.
- **Conditions.** Alone on the box, under `MemoryMax=24G`.
- **Each row records** Z, both timings, max RSS, which grazing bands were
  filled, and the tree's commit.

| deck | θ_min | trees | cold, predicted | warm |
|---|---|---|---|---|
| `buried_dipole`, `brv_default`, `ebc_default`, `brv48` | ≥ 1.358° | all three | branch/main 1.00 ± 0.05, branch/0.55.0 1.00 ± 0.05 | 1.00 ± 0.05 |
| `dipole935` | 0.0582° | all three | branch/0.55.0 1.00 ± 0.08; branch/main 0.35 ± 0.06 (amendment 5's 2.86×, inverted) | 1.00 ± 0.05 |
| `brv_corner` | 0.0724° | all three | branch/0.55.0 1.00 ± 0.08; branch/main 0.44 ± 0.08 (2.26×, inverted) | 1.00 ± 0.05 |
| `dipole1mm`, added: #935's dipole at 1 mm | 0.0194° | main, branch (0.55.0 refuses it) | branch/main 1.00 ± 0.08 | 1.00 ± 0.05 |
| `twonode11`, added: U9's two-node soil-A deck at 11 m | 0.0176° | main, branch (0.55.0 refuses it) | branch/main 1.00 ± 0.08 | 1.00 ± 0.05 |

**Step 0, geometry only, before G5.** Read each added deck's θ_min on the
kwargs, and confirm that main and the branch serve it and 0.55.0 refuses it.

### G6: the census

- **Population A.** antennaknobs `scratch/956-census/popa_run.py`, run from an
  antennaknobs worktree at main.
  - **Columns.** `MOMWIRE_MID` is main's src and `MOMWIRE_MAIN` the branch's;
    the "shipped" column is whatever the antennaknobs venv imports, and is
    recorded.
  - **NEC-5.** `NEC5_EXE` is the NEC-5 binary on this box, and its sha256 is
    recorded. It is not the census's `2068ae67…` binary, so the NEC-5 column is
    context and is not compared with the published one.
  - **Output.** Written into this records directory.
- **Population B.** The census's eight corpus decks, through momwire on main
  and on the branch, with the census's own corpus runner
  (`scratch/896-census/census_momwire.py`).
  - **Pinned later.** The exact invocation goes in an addendum before this
    step runs.
  - **Prediction:** each deck's outcome is the same on both trees, a refusal
    with the same sentence or a served Z. A served Z moves only on decks whose
    pairs reach under 0.1°, and then by ≤ 4.7e-4 relative.

## Order

1. Step 0 (geometry only).
2. G4.
3. G1a, G2a, G2b, G2c, G3.
4. G7's local lanes.
5. G5, alone on the box; G1b is read from its rows.
6. G6.
7. Push, dispatch `ci.yml`, and open the PR with the gate table. The PR is not
   merged from here.
