---
title: "Speed against NEC-5"
description: Measured run times of momwire's standalone razor-2p executable against the licensed NEC-5 console engine (NEC5CL_x13) on eight decks, with the machine, thread counts and method stated.
---

This page records one measured comparison: momwire's standalone
**nec5-razor-2p** executable against the licensed NEC-5 console engine,
`NEC5CL_x13.exe`, the one EZNEC 7 ships. Both run the same decks on the
same Windows laptop. The results come from one machine and one set of
decks, so they are not a general speed claim. Where NEC-5 wins, the table
says so.

razor-2p is the like-for-like engine because it uses NEC-5's formulation.
On every deck here its input impedance matches x13's to a few thousandths
of an ohm in free space, and to about 0.1 % over Sommerfeld ground. The
default nec5-bspline engine is a different basis, so its impedances differ
from NEC-5's by 1–9 % on these decks (see [Choosing an
engine](/reference/choosing-an-engine/)). It was timed as well but is left
out of the comparison.

## How it was measured

Measured 2026-11-17.

- **Machine:** Dell XPS 9320, Intel i5-1240P (4 performance and 8
  efficiency cores, 16 threads), 16 GB, Windows 11 Pro. On AC power, with
  the Balanced plan and the Best-performance overlay.
- **Engines:**
  - `NEC5CL_x13.exe`, which runs on one thread;
  - momwire 0.93.0's `momwire-nec5-razor2p-avx2.exe`, from the release's
    `momwire-nec5-windows-x64.zip`.

  Each reads the deck and writes its printout, the way EZNEC or SimNEC
  calls an external engine.
- **Timing:** wall time for the whole process, from start to exit. Each
  configuration ran 7 times after a warm-up, interleaved across engines.
  The tables give the median, and the spread was within about 10 % on
  every momwire row.
- **Threads:** momwire has two columns.
  - *1 thread* matches x13's single thread.
  - *Default* is what the executable picks with no thread variables set.
    It uses more threads as the problem grows: 1 on the smallest deck, 4 on
    the mid-size ones, and 12 on the largest. [The standalone
    executables](/reference/standalone/) explains how to set the count.

## Results

Ratio = x13 time ÷ momwire time. Above 1 means momwire was faster.

| Deck | Segments × frequencies | x13 | razor-2p, 1 thread | razor-2p, default |
| --- | --- | --- | --- | --- |
| 5-band fan dipole, Sommerfeld ground | 296 × 21 | 2.54 s | 1.28 s (2.0×) | 0.79 s (3.2×) |
| 5-band fan dipole, free space | 296 × 21 | 0.49 s | 0.33 s (1.5×) | 0.21 s (2.4×) |
| Long flat dipole, free space | 1097 × 13 | 4.03 s | 2.74 s (1.5×) | 1.77 s (2.3×) |
| Longer flat dipole, free space | 2071 × 13 | 16.3 s | 11.9 s (1.4×) | 5.81 s (2.8×) |
| Longer flat dipole, Sommerfeld ground | 2071 × 13 | 142 s | 53.5 s (2.7×) | 21.8 s (6.5×) |
| 32-radial buried hub | 640 × 5 | 4.20 s | 1.94 s (2.2×) | 0.94 s (4.5×) |
| Buried-radial vertical | 251 × 21 | 4.61 s | 12.7 s (0.36×) | 6.17 s (0.75×) |

- **NEC-5 wins on the buried-radial vertical,** at every thread count.
- **Sommerfeld ground** is ε<sub>r</sub> = 13, σ = 0.005 S/m in every
  deck that uses it.
- **The eighth deck** is a small two-band fan dipole of 106 segments. It
  is left out because x13's times on it ranged from 0.54 s to 4.7 s,
  which is too wide to quote. razor-2p took 0.21 s.

## Fill reuse

The `razor_pair_reuse` feature follows 0.93.0. It fills each class of
repeated segment pairs once and copies the result. It is off by default and
turned on with `MOMWIRE_X_RAZOR_PAIR_REUSE=1`. The printed impedances are
identical with it on and off.

These runs used the same machine, default threads and the main-branch
executable:

| Deck | x13 | reuse off | reuse on |
| --- | --- | --- | --- |
| Long flat dipole, free space | 4.03 s | 1.85 s | 1.76 s (2.3×) |
| Longer flat dipole, free space | 16.3 s | 5.81 s | 4.81 s (3.4×) |
| Longer flat dipole, Sommerfeld ground | 142 s | 21.8 s | 10.7 s (13.3×) |

Reuse helps most where filling the matrix is expensive and many segment
pairs repeat: long uniform wires over Sommerfeld ground. On the small decks
above it makes no measurable difference.

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
