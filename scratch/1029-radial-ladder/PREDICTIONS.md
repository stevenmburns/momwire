# Registered before the first timed run — the buried radial-screen ladder

Written 2026-09-14, before any arm of this ladder was timed. Misses are reported,
not re-gated.

## Setup as it will run

- **Harness: momwire#983's, verbatim** — `scratch/914-study/ladder_today.py`
  (momwire arm, cold then warm after clearing `_solved_cache`) and
  `scratch/914-study/nec5_ladder.py` (NEC-5 arm), both at momwire `1ca8725`
  (v0.55.0), `make build` run (a no-op: the native diff from the last built
  commit is empty, so `build_ext` correctly relinked nothing).
- **Deck**: `verticals.buried_radial_vertical` from the antennaknobs catalog via
  `bench_converge.load_design`, with `n_radials` set on the builder — the
  design's other defaults untouched (`depth` 0.15 m, `length_factor` 1.0,
  `radial_factor` 0.6, 7.1 MHz, `convention` "connected").
- **Ground**: `("finite", 13.0, 0.005)` — soil A, Sommerfeld. **Basis**: BSpline
  `degree=2`. Both as #983 ran them.
- **Segment counts reproduce #983's exactly**, checked before timing:
  12 → **680**, 24 → **1,328**, 36 → **1,976**, 48 → **2,624**. Beyond its rungs:
  72 → 3,920, 96 → 5,216, 113 → 6,134, 120 → 6,512, 150 → 8,132.
- **Engines**: momwire (cold and warm), NEC-5 `x13` (sha256 `2068ae67…`) and
  NEC-5 `3b75639` (sha256 `82272474…`).
- **Threads**: `OMP`/`OPENBLAS`/`MKL` = 4 on every arm, one process at a time.
- **Arm order within each N**: momwire → x13 → 3b75639 → x13 → momwire, twice, so
  each build is bracketed by the other and the spread is measurable.
- **Memory**: peak **RSS**, from `getrusage(RUSAGE_CHILDREN).ru_maxrss` of each
  arm's own subprocess — NOT tracemalloc. The harness's tracemalloc figure is
  recorded beside it, because #983's column is tracemalloc and the two are not
  the same number.
- `RLIMIT_AS` = 35 GB per momwire arm, so a runaway dies cleanly.

## Predictions

| | claim |
|---|---|
| **P1** | at 48 radials, 3b75639 is **≥ 2.5× faster** than momwire warm |
| **P2** | momwire warm scales as N^a over 48 → 150 with **a ∈ [1.7, 2.3]** |
| **P3** | the momwire/3b75639 ratio at 150 is **no better** than at 48 |
| **P4** | momwire peak RSS at 150 is **under 30 GB** |
| **P5** | x13 at 48 is **within 15 %** of #983's NEC-5 column (21.59 s) |

Mine, added for the same treatment:

| | claim | why |
|---|---|---|
| **P6** | x13 cold and warm agree within **5 %** | NEC-5 re-runs the whole solve per call; there is no warm cache to hit, so the harness's "warm" number should just be a second cold run |
| **P7** | both NEC-5 builds complete **150 radials (8,132 segments)** without refusing | the corpus routinely runs decks past 4,000 segments and the 4,000 cap is the corpus tool's, not NEC-5's |
| **P8** | momwire peak RSS at 150 is **20–26 GB** | 2.3 GB at 48 scaling as N², i.e. 2.3 × (150/48)² ≈ 22.5 GB. Sharper than P4 and falsifiable in both directions |
| **P9** | the #1029 discrepancy resolves as a **radial-count** error, not a box error: 12.07 s will be near this box's **36**-radial warm figure, not its 48 | #983's own table puts 12.07 s at 36 radials, and this box has run within a few per cent of that table elsewhere |

## The discrepancy this ladder settles

momwire#1029's body quotes "48 radials warm 12.07 s on the Skylake box".
#983's table reads **12.07 s at 36 radials**, with 19.25 s at 48. The 36 and 48
rows measured here say which reading is right.
