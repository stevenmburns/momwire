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

## Step 0 result [2026-09-15]: as registered

- **The run.** `run_gates.sh s0`, geometry only. Records: `s0_geometry.jsonl`,
  `s0_geometry.log`.

| deck | θ_min | 0.55.0 | main | branch |
|---|---|---|---|---|
| `dipole1mm` | 0.019417° | refused: "…below the 0.05 deg grazing floor…" | served | served |
| `twonode11` | 0.017587° | refused: "…the crossing serve completes ONE crossing node per deck…" | served | served |

- **A tooling fix, made before the reading counted.**
  - **What happened.** The first pass built `twonode11` through the tree's own
    `two_node_deck(11.0)`, and 0.55.0's `two_node_deck` takes no separation. So
    its 0.55.0 row was a TypeError, not a refusal.
  - **The fix.** `s0_geometry.py` now spells the deck from `crossing_deck()`,
    and records that the result equals `two_node_deck(11.0)` on main and on the
    branch: `same_as_two_node_deck: true` on both.
- **What it means for G5.** Both added decks sit under 0.05°; main and the
  branch serve them, and 0.55.0 refuses them. So G5 times them on main and the
  branch only, as registered.

## G6 addendum [2026-09-15]: Population B's procedure, pinned before G6 runs

- **Why an addendum.** This box has no copy of the census's translated corpus
  tree (`~/nec5-timing/nec5`).
  - **So Population B is rebuilt** for its six decks, from the public corpus on
    this box: `~/antennas/nec-wild/community/cebik-w4rnl/`.
  - **The census's other two members,** the `2lsloper` pair, were never buried,
    and its re-render dropped them.
- **Identity, geometry only (no solve).** Each local raw deck is read through
  `antennaknobs.nec_import.parse_nec` and classified by the census's own
  `popb_select.classify`.
  - **What matches.** All six match their `popb-members.json` row on class,
    zmin, zmax, crossing nodes and wire count.
  - **What is checked later, and why.** Two fields differ for known reasons, so
    they are checked after translation instead:
    - `gn`: the census read the translated tree, where `translate` rewrites
      GN 2 to GN 0;
    - `segs`: the census counted the translated tree's remesh.
- **The steps:**
  1. **Link.** The six raw decks are linked under their census paths into
     `~/nec5-timing/popb-1064/raw/`.
  2. **Translate.** Run `scripts/nec5_corpus/nec5_corpus.py translate --src
     raw --out nec5`, from antennaknobs a9c49416f.
     - **Checked before any momwire run:** every translated deck carries GN 0
       and the census's segment count (62, 62, 532, 2,690, 1,029, 1,372).
     - **A mismatch is a stop.**
  3. **Solve.** Run `scratch/896-census/census_momwire.py --src nec5 --report
     popb_momwire_<tree>.jsonl` twice.
     - **PYTHONPATH:** main's src the first time and the branch's the second,
       with antennaknobs src after it each time.
  4. **Compare.** `g6_popb_compare.py` reports, per deck, the status and the
     error sentence on both trees, and the relative Z change for a served deck.
- **The prediction,** unchanged from the registration:
  - each deck's outcome is the same on both trees;
  - a served Z moves only on a deck whose pairs reach under 0.1°, and then by
    ≤ 4.7e-4 relative.

## G4 result [2026-09-15]: PASS

- **The run.** `run_gates.sh g4`, 01:00:11–01:01:45Z, on real grids (soil A,
  7 MHz, R₁ to the cap). Records: `g4_fill_counter.json`, `g4_fill_counter.log`.

| zone | order | calls (θ columns) | seconds | shared columns |
|---|---|---|---|---|
| inner | floor first | 2, 4 | 24.6 | bit-identical |
| inner | low first | 2, 4 | 24.6 | bit-identical |
| near | floor first | 2, 4 | 9.2 | bit-identical |
| near | low first | 2, 4 | 9.4 | bit-identical |
| far | floor first | 2, 4 | 9.8 | bit-identical |
| far | low first | 2, 4 | 9.7 | bit-identical |

- **Evaluations.** In every zone and both orders the fills made exactly two
  evaluation calls. The low band's call carried its four θ nodes; the floor
  band's carried its own two. No θ node appears in both, and every call
  evaluated n_θ × n_R₁ points.
- **Sharing.** The floor band's top two columns are the low band's first two,
  bit for bit.
- **Order independence.** The low band's values are the same bits in both fill
  orders.
- **Prediction** (passes): hit.

## G6 Population B [2026-09-15]: STOP at the addendum's translation check, before any momwire run

- **The run.** `nec5_corpus.py translate`, antennaknobs a9c49416f, on the six
  decks linked under `~/nec5-timing/popb-1064/raw/`. All six translated.
- **The check, as pinned:** every translated deck carries GN 0 and the census's
  segment count. **It fails on all six.**

| deck | census `segs` | translated GW sum | translated GN | raw deck parsed (`parse_nec`, `n_seg`) |
|---|---|---|---|---|
| `1-3.nec` | 62 | 63 | 2 | 62 |
| `3-2.nec` | 62 | 63 | 2 | 62 |
| `11-4a-nec4.nec` | 532 | 178 | 2 | 532 (its raw GW column sum is 119) |
| `lpma3r5-4-6el86ft75o-buriedradials.nec` | 2,690 | 2,698 | 2 | 2,690 |
| `1r5-bc3elendfire-burrad.nec` | 1,029 | 1,035 | 2 | 1,029 |
| `1r8-4el-endfire-burrad.nec` | 1,372 | 1,380 | 2 | 1,372 |

- **By the pinned rule this is a stop.** No Population B momwire run is made,
  and nothing is re-pinned here.
- **Diagnosis: the check was mis-specified,** and in two ways. Neither is a
  change in the decks.
  - **GN.** The addendum expected GN 0, from `popb_select.py`'s account of the
    pre-U1 translator. U1 (antennaknobs#1453) made `translate` pass GN 2
    through, and today's tool does.
  - **Segments.** The addendum called `segs` the translated tree's count. It is
    the raw deck's count through antennaknobs' importer, and all six decks
    match it exactly. The translated sums are today's translator's own
    rewrites, and its report notes them ("GW tag N: N -> N segments so the
    referenced feed/load/port sits on a knot", 20 notes).
- **What stands.** With `segs` read correctly, the six local decks match the
  census's members on class, zmin, zmax, crossing nodes, wire count and
  segment count. The identity the addendum wanted is shown, by a check other
  than the one pinned.
- **Reported for a decision,** with a proposed re-pin: identity on the raw
  parse (as shown above), today's translation as the momwire input, and the
  rest of the procedure unchanged. Population A, G1–G5 and G7 do not depend on
  it and continue.

## G6 Population B re-pin [2026-09-15]: approved, and committed before any Population B momwire run

- **Approved upstream of this record, on 2026-09-15.** It can still be
  overruled.
- **The pinned translation check is a MISS.** It was mis-specified in two ways.
  1. It expected GN 0, but U1 (antennaknobs#1453) made `translate` pass GN 2
     through.
  2. It read the census's `segs` as translated counts. They are raw-parse
     counts through `parse_nec`.
  - **The diagnosis was geometry only.** No momwire run was made on these
    decks.
- **Identity, re-pinned.** Each raw deck is read through `parse_nec` and
  classified by `popb_select.classify`. It matches `popb-members.json` on
  class, zmin, zmax, crossing nodes, wires and segs, for all six decks, as the
  table in the stop section shows.
- **momwire's input, on BOTH arms (main 54caf86 and the branch):** today's
  translation, `~/nec5-timing/popb-1064/nec5`, made by antennaknobs a9c49416f.
  - **Provenance.** The translate report is `popb_translate_report.jsonl`, and
    each deck's sha256 is in `popb_translated_sha256.txt`.
- **The verdict is main against the branch, on identical inputs.**
  - **Not a reference:** the census's recorded momwire numbers. They came from
    a different translation, and nothing is compared against them.
- **The standing rule holds.** A wholly buried design that moves between main
  and the branch is a stop. None of these six decks is wholly buried: all are
  crossing decks.
- **Unchanged:** the prediction, and procedure steps 3 and 4 (`run_g6b.sh`,
  then `g6_popb_compare.py`). They run after G5, beside Population A.

## G3 result [2026-09-15]: FAIL by its bar, so a STOP; the prediction MISSES

- **The run.** `run_gates.sh g3 avx2` and `g3 sse2`, 01:02:54–01:11:39Z, on the
  branch. Records: `g3_avx2.json`, `g3_sse2.json`, `g3_compare.json`, and the
  logs.
- **The loaded variants** were `_accelerators_avx2` and `_accelerators_sse2`.
- **The bar was pairwise ≤ 1e-12, and it fails.** C++ avx2 against C++ sse2
  reads **7.8e-10** at the floor node (θ = 0.016667°, inner zone, soil A at
  7 MHz). Every pair across the two processes carries that same number.
- **The prediction (≤ 1e-13) MISSES.**
- **What the breakdown shows,** from the same records:
  - **Same-process C++ against numpy agrees at every point:** 5.7e-16 at worst,
    in both variants, at all 78 points. That includes 0.049999999, 0.05 and
    0.050000001°, and 0.099999999, 0.1 and 0.100000001°. The five-way selector
    and the stride of 5 agree between the two dispatches.
  - **The cross-variant spread follows the node's grazing angle, not the
    seams.** On soil A at 7 MHz, inner zone, it reads 7.8e-10 at the floor,
    1.1e-10 at 0.0333°, 1.00e-10 on both sides of 0.05°, 3.9e-11 on both sides
    of 0.1°, and 1.9e-12 at 0.5°. It is smaller in the near and far zones and
    on soil C at 21 MHz.
  - **So the spread is continuous across both seams,** which is the signature
    of the grid FILL rather than of routing. Each process fills its own grid
    with its own variant's contour.
- **Hypothesis, not yet measured.** The bar was mis-specified: it compared two
  independently filled grids, so it measured the avx2 and sse2 contours'
  difference at the most grazing nodes (their tolerance is rtol 1e-9), not the
  dispatch. That spread predates momwire#1064.

## D1, a diagnostic of G3, registered before it runs

- **What.** `g3_dispatch.py`, unchanged, on MAIN 54caf86. There are two
  processes, `MOMWIRE_FORCE_VARIANT=avx2` and `=sse2`, writing
  `d1_main_g3_avx2.json` and `d1_main_g3_sse2.json`. `d1_compare.py` reads them
  beside G3's branch captures.
- **Why this can tell.** momwire#1064 does not touch the fill's contour. At
  these points main's single low band reads the same nodes the branch's floor
  and low bands read, or neighbours at the same angles. So if the spread
  predates this change, main shows it too.
- **Predictions:**
  - main's cross-variant spread is within 2× of the branch's at every point;
  - main's same-process C++ against numpy is ≤ 1e-15.
- **What it can and cannot decide.** D1 does not re-open G3's bar and does not
  pass G3. A hit supports the hypothesis above, and G3's verdict waits for a
  decision on the gate's spelling.

## G1a, G2a, G2b, G2c results [2026-09-15]: G1a FAILS, so a STOP; G2a, G2b and G2c pass

- **The runs.** `run_gates.sh g12 v055|main|branch`, 01:02:54–01:19:31Z,
  concurrently with G3's two processes.
  - **Records:** `g12_*.json`, `g12_*.log`, `g12_compare.json`.
  - **A reading fix.** The comparator's first read crashed on the θ key strings
    (`repr(np.float64(...))`). It was fixed at 8c6661a. The captures were not
    re-run.

**G1a: FAIL.** 446 of 1,134 old-set points are not bit-identical to 0.55.0, in
the surfaces or the kernel. The differences are last-bit: at most 6.7e-14
relative in the surfaces and 1.9e-14 in the kernel.

| comparison | low [0.05°, 0.1°) (surface / kernel not bit-identical) | mid [0.1°, 1°) | grazing and steep ≥ 1° |
|---|---|---|---|
| branch vs 0.55.0 (378 / 504 / 252 points) | 157 / 137 | 193 / 167 | 95 / 87 |
| main vs 0.55.0 | 293 / 292 (U9's band, up to 1.5e-9, expected) | 0 / 0 | 0 / 0 |
| branch vs main | 327 / 322 | 193 / 167 | 95 / 87 |

- **The differences are confined to three media:** soil B at 7 and at 21 MHz,
  and soil A at 3.5 MHz. There they touch nearly every θ at every R₁. Soils A
  and C, at 7 and at 21 MHz, are bit-identical at every point.
- **They reach bands momwire#1064 does not change.** The mid, grazing and steep
  bands' code and lattices are untouched. Yet the branch's process produced
  different last bits there than main's and 0.55.0's, which agree with each
  other bit for bit at θ ≥ 0.1°.
- **The prediction (passes) MISSES.**
- **One process difference is known, and its effect is not yet measured.** The
  branch capture ran with `--direct`, which calls `iv_surfaces_direct_below`
  after each medium's capture. The main and 0.55.0 captures did not.

**The other three gates pass:**
- **G2a: PASS.** The max is 6.7e-14 (soil B, 7 MHz, R₁ = 3.9 λ_m, θ =
  0.049999999°), and 84 of 294 points are bit-identical; the bar is 4.7e-4.
  The prediction (≤ 1e-14) MISSES, at the same last-bit scale as G1a.
- **G2b: PASS.** The floor band reads 3.94e-9 against the direct surfaces (soil
  A, 3.5 MHz, R₁ = 3 λ_m, 0.022167°, IphiH). The prediction (≤ 1e-8) hits.
- **G2c: PASS.** The low band reads 4.32e-9. The prediction (≤ 1e-8) hits.

## D2, diagnostics of G1a, registered before they run

- **D2a.** `g12_grid.py` on the branch WITHOUT `--direct`, written to
  `d2_branch_nodirect.json`. `d2_compare.py` compares it with 0.55.0's and
  main's captures.
- **D2b.** `g12_grid.py` on 0.55.0 again, written to `d2_v055_repeat.json`,
  and compared with the first 0.55.0 capture. It asks whether a tree
  reproduces its own capture bit for bit across processes.
- **The predictions:**
  - D2b is bit-identical to the first capture;
  - D2a is bit-identical to 0.55.0 at θ ≥ 0.05° and to main at θ ≥ 0.1°, so
    the `--direct` history is the cause.
- **If D2a still differs,** the cause is on the branch's code path, and that is
  what gets reported.
- **Scope.** Neither diagnostic re-opens G1a's bar.

## D1 result [2026-09-15]: prediction HIT; G3's cross-variant spread predates momwire#1064

- **The run.** `g3_dispatch.py` on MAIN 54caf86, one process per variant,
  01:21:47–01:25Z. Records: `d1_main_g3_avx2.json`, `d1_main_g3_sse2.json`,
  `d1_compare.json`.
- **Main's cross-variant spread matches the branch's.** Its worst is 7.776e-10,
  at the same point (the floor node, inner zone, soil A at 7 MHz) and to every
  printed digit.
  - **72 of 78 points** agree within ×1.01.
  - **The six that do not** are all at θ = 0.06°, the one queried angle where
    the branch's low band and main's single band interpolate from different
    nodes. The worst of them is ×1.90, on a spread of 1.4e-13 (soil C, 21 MHz,
    far zone).
- **Main's same-process C++ against numpy** is 5.7e-16, the same as the
  branch's.
- **Both predictions (within 2×, and ≤ 1e-15) hit.**
- **Reading.** G3's failure measured the avx2 and sse2 builds' fill spread at
  the most grazing nodes, which main already carries. It did not measure the
  dispatch this change touches.
  - **What was tested and holds:** the five-way selector and the stride of 5
    agree between C++ and numpy to 5.7e-16 on both variants.
  - **G3 stays FAIL by its registered bar,** and waits for a decision on the
    gate's spelling.

## D2 results [2026-09-15]: D2b HITS; D2a MISSES, so the cause is on the branch's own path

- **The runs.** 01:24:19–01:33Z.
  - `g12_grid.py` on the branch without `--direct`, written to
    `d2_branch_nodirect.json`.
  - `g12_grid.py` on 0.55.0 again, written to `d2_v055_repeat.json`.
- **The comparisons,** by `d2_compare.py`, written to `d2_*__vs__*.json`.
  - **How it counts.** The comparator counts every captured row. The old set
    carries 0.1° and 1.0° twice (once as a band's last node, once as the next
    band's first), so it reads 1,218 rows where G1a's keyed count read 1,134
    points.

| comparison | not bit-identical (surface / kernel), low / mid / ≥ 1° | media |
|---|---|---|
| **D2b:** 0.55.0 repeat vs 0.55.0 | 0/0, 0/0, 0/0 | none |
| **D2a:** branch without `--direct` vs branch with `--direct` | 0/0, 0/0, 0/0 | none |
| **D2a:** branch without `--direct` vs 0.55.0 | 157/137, 208/179, 110/100; max 6.7e-14 / 1.9e-14 | B 7 MHz, B 21 MHz, A 3.5 MHz |

- **D2b hits.** A tree's capture reproduces itself bit for bit across
  processes.
- **D2a misses.** The branch without `--direct` is the branch with it, bit for
  bit, and it differs from 0.55.0 exactly as G1a found. So the `--direct`
  history is not the cause.
- **Reading.** The branch's last-bit differences are deterministic, and they
  come from its own code path, in bands whose code and lattice momwire#1064
  does not change, on three lossy media.
- **The leading hypothesis, not measured.** A grid fill's values depend on
  evaluation HISTORY within the process, for example per-thread contour state.
  The branch fills the new floor band on the same queries, which changes what
  was evaluated before each later fill.
- **G1a stays FAIL by its bar.**
