# Status: partial — the phase runs are done, the analysis is not

The catalog razor/NEC-5/bs2 task took priority on 2026-09-15 before this was
written up. What is here is complete data and an unfinished reading of it.

**Done:** cProfile of the warm solve at 48, 120 and 150 radials (`warm-*.prof`),
RSS traces at 0.5 s (`rss-*.csv`), the per-rung results (`result-*.json`), and the
registered predictions (`PREDICTIONS.md`).

**Not done:** the per-term × rung table with local exponents, the peak-RSS
breakdown at 150, the three candidate levers, and the prediction verdicts.

## What the instrument choice cost, since it is the part that is settled

py-spy is unusable on this workload and cProfile is not. Measured at 48 radials
against a 13.35 s unprofiled warm solve:

| instrument | warm solve | overhead |
|---|---:|---|
| none | 13.35 s | — |
| **cProfile** | **13.65 s** | **2.2 %** |
| py-spy, no `--native` | 75.2 s | 5.6×, "2947 s behind in sampling" |
| py-spy `--native --rate 100` | — | 29 min for a 28 s solve, "810 s behind" |

py-spy walks every thread's stack and OpenBLAS runs four; cProfile charges per
Python call, and momwire's phases are a few Python calls each doing seconds of
work in C++. **2.2 % is the answer to the brief's "profiler overhead at 48"**, for
the profiler actually used.

`--pid` attach, which the brief suggested for isolating the warm solve, is
impossible here: `/proc/sys/kernel/yama/ptrace_scope` is **1**, so py-spy can only
profile a process it launched, and sudo wants a password on this box. The warm
solve is isolated instead by running it inside `_phase_warm()`, so the phase is a
frame rather than a clock offset.

## The 48-radial phase table, as far as it was read

Warm solve 13.65 s. The RSS sampler thread appears as `threading.py:651 wait` with
30.35 s cumulative — a concurrent thread, excluded from every share below.

| term | s | % |
|---|---:|---:|
| below-remainder projection (`proj_bb` → `remainder_field_proj_batch_below`) | 7.50 | 55 % |
| seg-seg moments (`_seg_seg_full_moments_offedge`) | 1.28 | 9 % |
| crossing cross block (`_sandwich_dense` + `_tables`) | ~1.4 | 10 % |
| chunked subset fill (`_accumulate_Z_subset_chunked`) | 0.94 | 7 % |
| field-Galerkin assembly | 0.67 | 5 % |

Against #914's laptop table (projection 7.35 s / 45 %, subset fill 3.64 s,
crossing 3.24 s, whole solve 16.49 s): the projection's **absolute** time is
almost unchanged while everything else shrank, so its share rose. That is the
shape a reader should expect from the levers landed since.

## The rungs, for whoever finishes this

| radials | segments | cold | warm | peak RSS |
|---:|---:|---:|---:|---:|
| 48 | 2,624 | 14.19 | 13.65 | 1,088 MB |
| 120 | 6,512 | — | — | — |
| 150 | 8,132 | 156.70 | 157.59 | 8,625 MB |

The 150 figures reproduce #1067's ladder (162.02 s warm, 8,568 MB) to about 3 %.
