---
title: "Speed against NEC-5"
description: Measured run times of momwire's standalone razor2p executable against the licensed NEC-5 console engine (NEC5CL_x13) on eight decks, with the machine, core placement, thread counts and method stated.
---

This page records one measured comparison: momwire's standalone
**nec5-razor2p** executable against the licensed NEC-5 console engine,
`NEC5CL_x13.exe`, the one EZNEC 7 ships. Both run the same decks on the
same Windows laptop. The results come from one machine and one set of
decks, so they are not a general speed claim. NEC-5 is faster on one of
the eight decks, and by a wide margin; the table says so.

razor2p is the like-for-like engine because it uses NEC-5's formulation.
On every deck here its input impedance matches x13's to a few thousandths
of an ohm in free space, and to about 0.1 % over Sommerfeld ground. The
default nec5-bs2 engine is a different basis, so its impedances differ
from NEC-5's by 1–9 % on these decks (see [Choosing an
engine](/reference/choosing-an-engine/)). It was timed as well but is left
out of the comparison.

## How it was measured

Measured 2026-11-17.

- **Machine:** Dell XPS 9320, Intel i5-1240P (4 performance cores with two
  threads each, and 8 efficiency cores), 16 GB, Windows 11 Pro. On AC
  power, with the Balanced plan and the Best-performance overlay.
- **Engines:**
  - `NEC5CL_x13.exe`, which runs on one thread;
  - momwire 0.93.0's `momwire-nec5-razor2p-avx2.exe`, from the release's
    `momwire-nec5-windows-x64.zip`.

  Each reads the deck and writes its printout, the way EZNEC or SimNEC
  calls an external engine.
- **Timing:** wall time for the whole process, from start to exit. Each
  configuration ran 7 times after a warm-up, interleaved across engines.
  The tables give the median.
- **Core placement.** This processor mixes fast performance cores with
  slower efficiency cores, and where Windows puts a single-threaded
  program changes its time:
  - x13 runs about 1.9× slower on an efficiency core than on a
    performance core;
  - left unpinned, x13 sometimes ran up to 8× slower even on performance
    cores, a state that came and went between runs and that we did not
    explain;
  - momwire's single-threaded runs read the same unpinned and pinned,
    within about 1 %.

  So every single-threaded run here, x13's and momwire's alike, is pinned
  to one performance core (`start /affinity 1`). Pinned, x13's times are
  steady to within a few per cent.
- **Threads:** momwire has two columns.
  - *1 thread* matches x13's single thread, on the same pinned core.
  - *Default* is what the executable picks with no thread variables set,
    unpinned. It uses more threads as the problem grows: 1 on the smallest
    deck, 4 on the mid-size ones, and 12 on the largest. [The standalone
    executables](/reference/standalone/) explains how to set the count.

## Results

Ratio = x13 time ÷ momwire time. Above 1 means momwire was faster.

| Deck | Segments × frequencies | x13 | razor2p, 1 thread | razor2p, default |
| --- | --- | --- | --- | --- |
| 2-band fan dipole, Sommerfeld ground | 106 × 21 | 0.51 s | 0.19 s (2.7×) | 0.21 s (2.4×) |
| 5-band fan dipole, Sommerfeld ground | 296 × 21 | 2.28 s | 1.14 s (2.0×) | 0.61 s (3.7×) |
| 5-band fan dipole, free space | 296 × 21 | 0.52 s | 0.30 s (1.7×) | 0.22 s (2.4×) |
| Long flat dipole, free space | 1097 × 13 | 4.28 s | 2.73 s (1.6×) | 1.87 s (2.3×) |
| Longer flat dipole, free space | 2071 × 13 | 17.6 s | 12.0 s (1.5×) | 5.87 s (3.0×) |
| Longer flat dipole, Sommerfeld ground | 2071 × 13 | 147 s | 54.5 s (2.7×) | 17.9 s (8.2×) |
| 32-radial buried hub | 640 × 5 | 2.98 s | 1.80 s (1.7×) | 0.86 s (3.5×) |
| Buried-radial vertical | 251 × 21 | 1.68 s | 11.5 s (0.15×) | 4.41 s (0.38×) |

- **NEC-5 wins on the buried-radial vertical,** by 6.9× at one thread and
  2.6× at default threads. Almost all of momwire's time on that deck goes
  to the vertical where it crosses the ground: an exact kernel through the
  interface, evaluated afresh at each of the 21 frequencies. The next two
  sections show the opt-in features that close most of that gap.
- **Sommerfeld ground** is ε<sub>r</sub> = 13, σ = 0.005 S/m in every
  deck that uses it.

## The crossing: opt-in features

Two opt-in features change how momwire evaluates the rows where a wire
crosses the ground. Both are off by default.

- **`crossing_sheet_retune`** (`MOMWIRE_X_CROSSING_SHEET_RETUNE=1`) serves
  the crossing from precomputed sheets. Its declared drift is 6e-11, and
  on this deck its impedances print identically to flag-off at every
  frequency. The executables carry it.
- **`crossing_table`** (`MOMWIRE_X_CROSSING_TABLE=1`) reads the crossing
  rows from one table per frequency. It is faster than the exact path but
  approximate: on this deck its impedance moves by up to 9.0e-4 relative,
  about 0.07 Ω. The executables carry it.

Buried-radial vertical, same machine and placement, a main-branch
executable after 0.93.0, warm runs:

| Configuration | 1 thread | Default |
| --- | --- | --- |
| x13 | 1.82 s | — |
| razor2p, both off | 11.2 s (0.16×) | 4.81 s (0.38×) |
| razor2p, `crossing_sheet_retune` | 2.94 s (0.62×) | 2.07 s (0.88×) |
| razor2p, `crossing_table` | 4.12 s (0.44×) | 2.56 s (0.71×) |

- **x13 still wins this deck,** by 1.6× at one thread and 1.1× at default
  threads, even with the sheets.
- **The sheets are built once and kept on disk.** The first run builds one
  sheet per frequency, which cost about 2.5 s extra at one thread here.
  Later runs reuse them.
- This x13 median is from the same run as these rows, so it differs a
  little from the 1.68 s in the main table.

## Fill reuse

The `razor_pair_reuse` feature fills each class of
repeated segment pairs once and copies the result. It is off by default and
turned on with `MOMWIRE_X_RAZOR_PAIR_REUSE=1`. The printed impedances are
identical with it on and off.

These runs used the same machine and a main-branch executable at default
threads:

| Deck | x13 | reuse off | reuse on |
| --- | --- | --- | --- |
| Long flat dipole, free space | 4.28 s | 1.84 s | 1.82 s (2.4×) |
| Longer flat dipole, free space | 17.6 s | 5.90 s | 4.90 s (3.6×) |
| Longer flat dipole, Sommerfeld ground | 147 s | 21.8 s | 10.7 s (13.8×) |

Reuse helps most where filling the matrix is expensive and many segment
pairs repeat: long uniform wires over Sommerfeld ground. On the 1097-segment
dipole it makes no measurable difference: the solve is too small for the
fill to dominate.

## The decks

The decks are synthetic: exports from antennaknobs' design catalogue and
momwire's own benchmark shapes. All are NEC-5-dialect decks.

- **Pattern cards:** the three flat-dipole decks use the plain radiation
  pattern card, XNDA 1000. Their original pattern card asked for a
  normalised-gain table (XNDA 1500), which momwire 0.93.0 does not print.
  Producing that table adds about 7–8 % to x13's time.
- **Not covered:** the comparison has no near fields, no NEC-5 networks,
  and no decks beyond 2071 segments. A deck of a different shape can come
  out differently, as the buried-radial vertical shows.
