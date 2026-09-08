# 2026-09-08 — the buried-screen ladder after #914 levers 1–3 (they are IN)

**#914's levers 1–3 are already built and on main.** This record replaces the
profile in #914's issue body, which predates them, so the next brief is written
from today's numbers rather than August's.

| lever | status | where |
|---|---|---|
| 1. `_pair_extents_below` in C++ | **done**, "#914 unit 1" | `_accel_mw568.cpp:1339`, `bspline.py:713` |
| 2. field Galerkin assembly in C++ | **done**, "#914 unit 2" | `_accel_mw568.cpp:1448`, `bspline.py:5345` |
| 3. the zero-padded `_build_J_blocks_subset` scatter | **done**, via **#915** | `bspline.py:5892` (chunked buried route) |

Runtime, not just source: `_HAVE_PLAN_EXTENTS_ACCEL` and
`_HAVE_FIELD_GALERKIN_ACCEL` are both `True`.

## The ladder today

haswell, bs2, Sommerfeld `('finite', 13.0, 0.005)`, `swept_mem_mb` raised, one
process per arm, `tracemalloc` for peak. **NEC-5 re-measured on this box** —
#914's NEC-5 column was the laptop's, and quoting it here would have been a
cross-box ratio.

| radials | segments | cold | warm | NEC-5 (here) | ratio | peak |
|--:|--:|--:|--:|--:|--:|--:|
| 12 | 680 | 4.33 s | 2.91 s | 1.66 s | 1.76× | 160 MB |
| 24 | 1,328 | 7.98 s | 6.59 s | 5.78 s | 1.14× | 369 MB |
| 36 | 1,976 | 13.94 s | 12.07 s | 12.24 s | **0.99×** | 738 MB |
| 48 | 2,624 | 20.91 s | 19.25 s | 21.59 s | **0.89×** | 1,307 MB |

**momwire crosses NEC-5 at 36 radials and is 11 % faster at 48**, against
#914's 2.2 / 1.9 / 1.6 / 1.6× slower. At 48 radials the warm solve has gone
**34.8 s → 19.25 s**.

The re-measured NEC-5 column (1.66 / 5.78 / 12.24 / 21.59) sits within ~15 % of
the laptop's (1.44 / 5.0 / 11.6 / 21.5), so the boxes are comparable and the
improvement is the levers, not the hardware.

Two instrument notes, so the table is not over-read: peak here is `tracemalloc`
and #914's was RSS (comparable arm-to-arm, **not** against #914's
0.35/0.73/1.41/2.33 GB); and the mesh is 680 segments at 12 radials against
#914's 654, so the rungs are not bit-identical decks.

## The 48-radial warm profile, today (18.29 s), self time

| term | s | C++ / numpy | was, in #914 |
|---|--:|---|---|
| `remainder_field_proj_batch_below` | **8.258** | **C++** | 7.2 |
| `seg_seg_full_moments_bspline_cplx_tiered` | 2.086 | C++ | 2.1 |
| `_sandwich_dense` | 0.963 | **numpy** | 1.5 (with `_row_weights`) |
| `seg_seg_static_moments_bspline_uniform` | 0.806 | C++ | 0.7 |
| `assemble_field_galerkin` | 0.789 | **C++** | **8.7 of numpy** |
| `_compute_Z_operator_buried` (self) | 0.673 | numpy glue | — |
| `near_interface_six_columns` | 0.601 | C++ | — |
| LU (`scipy _solve`) | 0.456 | LAPACK | 0.7 |
| `assemble_Z_bspline_weighted_windowed_cplx_eps` | 0.439 | C++ | — |
| `assemble_Z_bspline_windowed_cplx_eps` | 0.410 | C++ | — |
| `_row_weights` | 0.398 | **numpy** | (in the 1.5) |
| `_unique_rows` | 0.159 | **numpy** | — |
| `pair_extents_below` | **0.155** | **C++** | **2.6 of numpy** |
| `_g_of_r` | 0.153 | numpy | — |
| `_bnd_and_corner` (self) | 0.094 | numpy | 2.1 |
| `_main_split` (self) | 0.050 | numpy | 1.7 |

The three levers are visible and they delivered: extents **2.6 → 0.155 s**
(17×), the Galerkin assembly **8.7 → 0.789 s** (11×), and the
`_build_J_blocks_subset` scatter is gone from the top 28 entirely. `_main_split`
and `_bnd_and_corner` — #914's lever 4 — fell from 1.7 and 2.1 s to 0.050 and
0.094 s of self time as a side effect of the chunked route.

## What is left

**One term is 45 % of the warm solve and it is already C++.**
`remainder_field_proj_batch_below` is 8.258 s of 18.29 s — 243 M node pairs at
**34 ns each**. It is the dense below/below projection, and #982 measured that
neither a per-block ACA nor a square-caller restructure pays on it: the geometry
is low rank but the caller has already taken the compression, and the square
matrix is not low rank at all.

**All remaining numpy sums to ~2.3 s of 18.29 s — 13 %.** `_sandwich_dense`
0.963, `_compute_Z_operator_buried` self 0.673, `_row_weights` 0.398,
`_unique_rows` 0.159, `_g_of_r` 0.153. There is no lever-shaped work of the
#914 kind left: no single numpy term is worth more than 5 % of the wall.

So the honest statement for the next brief: **the buried screen is now
kernel-bound, not glue-bound.** Further gain has to come from the projection
kernel itself — 34 ns/pair, already OpenMP C++ — or from evaluating fewer pairs,
which is what ACA would have been and what #982 ruled out. Anything else on the
list is worth single-digit percentages.

## Reproducers

`scratch/914-study/ladder_today.py --radials N [--swept-mem-mb M]`,
`scratch/914-study/nec5_ladder.py --radials N`, `scratch/914-study/profile_48.py`.
