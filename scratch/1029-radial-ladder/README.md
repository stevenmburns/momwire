# The buried radial-screen ladder, re-timed — momwire vs two NEC-5 builds

Skylake, 2026-09-14. momwire `1ca8725` (v0.55.0). Harness: momwire#983's,
**verbatim** — `scratch/914-study/ladder_today.py` and
`scratch/914-study/nec5_ladder.py`, same deck
(`verticals.buried_radial_vertical`, `n_radials` on the builder, other defaults
untouched), same soil `("finite", 13.0, 0.005)`, same BSpline `degree=2`, same
cold/warm definition. Segment counts reproduce #983's exactly — 680 / 1,328 /
1,976 / 2,624 at 12 / 24 / 36 / 48 — and so does its memory column: tracemalloc
peak at 12 radials reads **159.6 MB** against #983's published 160 MB.

---

## Summary (paste-ready)

Skylake, 4 physical cores / 8 logical; `OMP`/`OPENBLAS`/`MKL` = 4 on every arm,
one process at a time. Arm order within each N: momwire → x13 → 3b75639 → x13 →
momwire, twice, so each build is bracketed by the other. **Every arm spread is
≤ 3.9 %, and ≤ 1.0 % at 48 radials and above.**

| radials | segments | mw cold | mw warm | x13 | 3b75639 | mw/x13 | mw/3b | mw peak RSS | Z momwire | Z x13 = Z 3b75639 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 12 | 680 | 3.22 | 2.14 | 1.23 | 0.30 | 1.73 | **7.10** | 298 MB | 52.61+38.32j | 52.29+36.44j |
| 24 | 1,328 | 6.10 | 4.90 | 4.20 | 0.79 | 1.17 | 6.21 | 532 MB | 46.16+32.85j | 45.86+30.97j |
| 36 | 1,976 | 10.57 | 9.28 | 9.15 | 1.62 | 1.01 | 5.72 | 785 MB | 44.33+30.48j | 44.04+28.62j |
| 48 | 2,624 | 16.66 | 15.28 | 16.22 | 2.82 | 0.94 | **5.43** | 1,099 MB | 43.52+29.22j | 43.23+27.37j |
| 72 | 3,920 | 34.74 | 33.13 | 35.87 | 6.40 | 0.92 | 5.18 | 2,169 MB | 42.71+27.96j | 42.42+26.11j |
| 96 | 5,216 | 57.32 | 55.31 | 64.03 | 11.52 | 0.86 | 4.80 | 3,672 MB | 42.28+27.25j | 41.99+25.40j |
| 113 | 6,134 | 77.84 | 75.62 | 88.60 | 16.44 | 0.85 | 4.60 | 4,997 MB | 42.09+26.95j | 41.81+25.10j |
| 120 | 6,512 | 86.94 | 84.54 | 100.44 | 18.67 | **0.84** | **4.53** | 5,626 MB | 42.03+26.85j | 41.75+25.00j |
| 150 | 8,132 | 164.91 | 162.02 | 160.02 ᵈ | 30.14 | 1.01 | 5.38 | 8,568 MB | 42.10+26.70j | 41.81+24.85j |

Seconds. `mw/x13` and `mw/3b` are momwire **warm** over the NEC-5 arm; below 1.00
momwire is faster. ᵈ x13 at 150 radials **exceeded the harness's own 120 s
timeout**; the figure shown is a supplementary run with the timeout raised to
900 s (160.19 / 160.02 s over two runs). It is not a refusal and not a size limit —
see below.

**Scaling exponents in N:**

| | range | exponent |
|---|---|---|
| momwire warm | 48 → 150 | **N^1.99** |
| momwire warm | 12 → 150 | N^1.71 |
| momwire cold | 12 → 150 | N^1.56 |
| x13 | 12 → 120 | N^1.93 |
| 3b75639 | 12 → 150 | N^1.86 |
| momwire peak RSS | 12 → 150 | N^1.37 |
| momwire tracemalloc | 12 → 150 | N^1.60 |

**Predictions, registered before the first timed run (`PREDICTIONS.md`):**

| | claim | verdict |
|---|---|---|
| P1 | at 48 radials 3b75639 is ≥ 2.5× faster than momwire warm | **HIT** — 5.43× |
| P2 | momwire warm scales as N^a over 48 → 150, a ∈ [1.7, 2.3] | **HIT** — N^1.99 |
| P3 | the momwire/3b75639 ratio at 150 is no better than at 48 | **HIT at its two stated points** (5.38 against 5.43, inside the arm spread) — but **false as a general claim**: the gap closes steadily to **4.53 at 120** before reopening at 150 |
| P4 | momwire peak RSS at 150 is under 30 GB | **HIT** — 8.4 GB |
| P5 | x13 at 48 is within 15 % of #983's 21.59 s | **MISS** — 16.22 s, 25 % faster. A cross-box comparison: the momwire arm is 21 % faster here too (15.28 s against 19.25 s), so both arms moved together and the ratio survives |
| P6 (mine) | NEC-5 cold and warm agree within 5 % | **HIT** — 0.0–0.9 % on x13, 0.0–3.5 % on 3b75639. NEC-5 re-solves per call; there is no warm cache |
| P7 (mine) | both NEC-5 builds complete 150 radials without refusing | **HIT** — x13 needs ~160 s and finishes; nothing refused, no size limit |
| P8 (mine) | momwire peak RSS at 150 is 20–26 GB | **MISS, badly** — 8.4 GB. I projected N² from #983's "~2.3 GB at 48"; RSS here is 1.10 GB at 48 and scales N^1.37 |
| P9 (mine) | #1029's 12.07 s is a radial-count error, not a box error | **MISS** — it is both |

**Box, threads, order:** Skylake 4c/8t, `OMP`/`OPENBLAS`/`MKL` = 4, one process at a
time, arms ordered momwire → x13 → 3b75639 → x13 → momwire twice per rung; peak
memory is **RSS** from `/usr/bin/time -v` per arm, with the harness's tracemalloc
figure kept beside it because #983's column is tracemalloc and the two differ.

**#1029's 12.07 s:** it is #983's **36**-radial warm figure, not a 48-radial one,
and not a Skylake number — this box reads **9.28 s at 36** and **15.28 s at 48**.
The quote is wrong on both the radial count and the box.

---

## What the ladder says about #1029

- **Against x13 the old headline stands**: momwire crosses over at 36 radials
  (1.01) and is 6 % faster at 48, reproducing #983's 1.76 / 1.14 / 0.99 / 0.89 as
  1.73 / 1.17 / 1.01 / 0.94.
- **Against the clean-room build it does not.** 3b75639 is **5.4× faster than
  momwire warm at 48** and **4.5× at 120**, and momwire never crosses it at any
  rung measured.
- momwire's advantage over x13 **grows** to 120 radials (0.84) and then
  **vanishes at 150** (1.01).

## The 120 → 150 discontinuity, and a hypothesis killed

Between 120 and 150 radials momwire's warm time nearly doubles for a 1.25× rise
in N, while both NEC-5 builds stay on their quadratic trend:

| | 120 | 150 | local exponent |
|---|---:|---:|---|
| momwire warm | 84.54 | 162.02 | **N^2.92** |
| x13 | 100.44 | 160.02 | N^2.09 |
| 3b75639 | 18.67 | 30.14 | N^2.15 |

momwire's RSS jumps with it, 5.6 → 8.6 GB. The obvious suspect is the solver's
fixed `swept_mem_mb` budget (default **256 MB**), which chunks hoisted blocks and
so does more chunks as the problem grows. **Tested and refuted**: re-running 150
radials with `--swept-mem-mb 2048`, eight times the budget, gives warm **165.4 s**
against 162.0 s and the same RSS (8.58 GB). The budget is not the cause, and what
is remains open.

## Memory, and why P8 was so wrong

The brief's premise — "~1.3 GB tracemalloc / ~2.3 GB RSS at 48, growing ~N²" —
does not hold here. Measured: **1,029 MB tracemalloc and 1,099 MB RSS at 48**, and
the growth is **N^1.37** in RSS, not N². Projected to 150 that gave ~3.9 GB;
actual is 8.6 GB, because the growth steepens at the top of the range — so both
the brief's projection (~23 GB) and mine (~22.5 GB from N², and 3.9 GB from the
fitted exponent) were wrong, in opposite directions. The ~35 GB stop rule never
came close to firing, and the full ladder ran, 150 radials included.

## Z, as a same-deck check

momwire sits **0.28–0.32 Ω** from NEC-5 in R and **1.85–1.89 Ω** in X at every
rung — the known contact-class residual, flat in N, nothing larger. **x13 and
3b75639 return identical Z to every printed digit at all nine rungs**, which is
the correctness signal worth having beside a 5× speedup.

## Files

- `ladder-arms.csv` / `.jsonl` — every arm: radials, build, round, segments, cold,
  warm, RSS, tracemalloc, Z, and the timeout note on the four failed x13 arms.
- `x13-150-supplementary.jsonl` — x13 at 150 with a 900 s timeout, labelled.
- `mw-150-swept-budget-probe.jsonl` — the refuted budget hypothesis.
- `run_ladder.py` — the driver: arm order, `RLIMIT_AS` 35 GB, per-arm RSS.
- `x13_150_long_timeout.py` — the supplementary probe.
- `PREDICTIONS.md` — registered before the first timed run.

NEC-5 appears here as timings and impedances only; nothing about either build's
internals is recorded.
