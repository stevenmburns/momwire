# Registered before the first timed run — profiling buried_radial_vertical at large N

Written 2026-09-14, before any profiled run. Misses are reported as misses.

Same deck and settings as #1067's momwire arm: `verticals.buried_radial_vertical`,
`BSplineSolver` degree 2, ground `("finite", 13.0, 0.005)`, default
`swept_mem_mb`, threads 4, momwire `d6204e6` (main's tip; `src`/`tests` unchanged
from the commit the brief names), avx2 accelerators confirmed loaded.

| | prediction |
|---|---|
| **PP1** | The below-remainder projection is still the largest single term at 48 radials, at **35–55 %** of the warm solve — #914's laptop table reads 45 % — and its share is **larger at 150 than at 48**. |
| **PP2** | The 120 → 150 super-quadratic behaviour is carried by **one** term, not spread evenly: exactly one of the phase terms shows a local exponent **above 2.5** over 120 → 150, and the chunked subset fill and the crossing cross block both stay **below 2.4**. |
| **PP3** | The 8.6 GB peak at 150 radials is **overlapping temporaries, not one huge array**: the largest single live allocation is **≤ 3 matrix-sized buffers** (3 × 1.06 GB = 3.2 GB), and the peak comes from several such buffers alive at once. |
| **PP4** | `py-spy record --native` at 100 Hz costs **under 5 %** of the warm solve's wall at 48 radials. |

**PP2 is the one that matters**, because a single super-quadratic term is a lever
and a spread-out one is not. It is falsifiable in both directions: two or more
terms above 2.5, or none, refutes it.
