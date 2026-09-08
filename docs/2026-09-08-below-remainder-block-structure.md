# 2026-09-08 — is the below-remainder projection worth an ACA? (#981 step 2)

**Verdict: no. The path is #914's plain C++ levers, not ACA.** The projection
*is* genuinely low rank — the physics intuition in #914 is right — but the
compression still available is ~3x on a term worth 20 % of the wall, against
~28 % of the wall available from bookkeeping with no numerics at all. And a
cluster partition beats a single ACA per call by only 1.4x, so the block
machinery is not what would earn it even later.

Measurement only. No kernel, no production change.

## Method

`verticals.buried_radial_vertical` at 12 / 24 / 36 / 48 radials, bs2,
Sommerfeld `('finite', 13.0, 0.005)`, `swept_mem_mb` raised, **one process per
arm**, `tracemalloc` for the memory figure rather than `ru_maxrss`. The
projection (`remainder_field_proj_below`) is intercepted to capture the node
clouds; the global table is **never densified** — at 48 radials it would be
~3.9 GB.

Instrument check: the 48-radial arm counts **242,985,744 node pairs =
15,588²**, which reproduces #914's "15.7k² pairs" exactly, so this is the term
#914 profiled.

## The geometry IS low rank

| radials | calls | node pairs | far fraction | SVD rank @1e-6 | ACA rank |
|--:|--:|--:|--:|--:|--:|
| 12 | 30 | 15,397,776 | 0.894 | 5–8 | 8–10 |
| 24 | 119 | 61,027,344 | 0.858 | 5–6 | 7–8 |
| 36 | 279 | 136,890,000 | 0.857 | 6 | 7–9 |
| 48 | 520 | 242,985,744 | 0.887 | 5–6 | 7–8 |

Far fraction 0.86–0.89 throughout, and **the rank does not grow with the
block**: block width runs from 666 to 15,588 columns across the ladder while
the numerical rank at 1e-6 stays at 5–6. That is genuine low rank, not a block
that happens to be small. ACA reaches the requested 1e-6 at rank 7–8, with no
sign of #973's stagnation — the sampled probe residual lands at 4e-7 to 8e-7,
i.e. where it was asked to.

## But the call structure has already taken most of the compression

The single global factorisation **per call**, for comparison:

| radials | n_obs | n_src | global ACA rank | probe residual |
|--:|--:|--:|--:|--:|
| 12 | 132 | 3,924 | 39 | 4.31e-07 |
| 24 | 66 | 7,812 | 22 | 6.50e-07 |
| 36 | 42 | 11,700 | 17 | 5.50e-07 |
| 48 | 30 | 15,588 | 14 | 8.08e-07 |

**The observer cloud per call is tiny and shrinking** — 132 nodes at 12
radials down to 30 at 48 — because the caller already blocks the observers.
Each call is a thin `30 x 15,588` matrix whose rank cannot exceed 30, and ACA
finds 14. So the headroom is bounded before any partition is built.

## What each route is worth

Cost model: dense `m·n` kernel evaluations against `rank·(m+n)`, far blocks
compressed at their measured rank and near blocks left dense.

| radials | global per call | per-block partition |
|--:|--:|--:|
| 12 | 3.27x | 3.26x |
| 24 | 2.97x | 2.82x |
| 36 | 2.46x | 3.20x |
| 48 | 2.14x | 3.08x |

Against #914's 48-radial profile, where the projection is 7.2 s of 34.8 s warm:

| route | speedup | projection | saved | share of wall |
|---|--:|--:|--:|--:|
| global ACA per call | 2.14x | 3.37 s | 3.83 s | **11 %** |
| per-block ACA | 3.08x | 2.33 s | 4.87 s | **14 %** |
| **#914 levers 1–3** (extents in C++, Galerkin assembly in C++, the tensor scatter) | — | — | ~9.6 s | **~28 %** |

## Why this is a "no"

1. **The prize is smaller than the bookkeeping.** ACA at its best takes 14 % of
   the wall; levers 1–3 take roughly twice that, and they are `min`/`max` in
   C++, an assembler that already exists in a complex-eps twin, and not
   zero-padding a tensor. No numerics, no accuracy gate, no stagnation risk.
2. **The partition is not what earns it.** Per-block beats one global ACA per
   call by 1.4x (3.08 vs 2.14). A cluster tree, an admissibility parameter and
   a per-block residual check is a large amount of machinery — and, after
   #973, a large amount of *gate* — for 1.4x.
3. **The ceiling is structural, not numerical.** The rank is 5–6 and the far
   fraction is 0.89; those are excellent. The limit is that each call is
   already narrow in the observer direction, so `m·n → rank·(m+n)` cannot win
   much when `m` is 30.

## What would change the answer

If the caller were restructured so the projection saw the whole node cloud at
once (15,588 x 15,588 rather than 520 calls of 30 x 15,588), the dense cost
would be unchanged but the ACA cost would become `rank·2n` on a square matrix,
and the ranks measured here say that would be a very large win. **That is a
change to the caller, not a kernel** — and it should be measured before it is
built, exactly as this was. Filed as the open question on #981; not attempted
here.

## Reproducer

`scratch/981-study/measure_below_blocks.py --radials N [--swept-mem-mb M]`,
one process per arm. Peak (tracemalloc, solve only): 160 / 369 / 736 / 1305 MB
at 12 / 24 / 36 / 48.
