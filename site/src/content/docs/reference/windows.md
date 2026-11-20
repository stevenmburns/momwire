---
title: "Which momwire to run on Windows"
description: The standalone executable, the EZNEC drop-in, or the single-file preview — which to use on Windows, and what each costs, measured on momwire 0.95.0's release builds.
---

momwire comes for Windows in three forms. All three print the same input
impedance for a deck they both serve, so the choice is about speed and
convenience, not about the answer.

| You want | Use |
|---|---|
| The fastest run of a deck the executable accepts | The standalone executable, [`momwire-nec5-windows-x64.zip`](https://github.com/stevenmburns/momwire/releases/latest/download/momwire-nec5-windows-x64.zip) |
| A deck the executable refuses, or momwire inside EZNEC, SimNEC or 4nec2 | The drop-in, [`momwire-windows.zip`](https://github.com/stevenmburns/momwire/releases/latest/download/momwire-windows.zip) |
| One file per engine, nothing beside it | The single-file build (preview), `momwire-nec5-windows-x64-single.zip`, on releases that list it |

## The standalone executable first

When the standalone executable accepts a deck, it is the faster way to
solve it. When it doesn't, it says so: it names what it refuses, writes no
printout, and exits with status 2. That is the cue to run the same deck
through the drop-in, which serves everything the Python engines serve. The
[standalone page](/reference/standalone/) lists what the executables carry.

Leave the thread settings alone. Both forms size their thread pools from the
deck and from the cores you allow them, and since 0.95.0 they choose the same
counts, so neither needs a variable set to run at its best. The
[threads section](/reference/standalone/#threads) covers the cases where
setting `MOMWIRE_THREADS` helps.

## What the drop-in costs

Measured on one Windows laptop with 16 logical processors, using the signed
momwire 0.95.0 release zips. Each figure is the median of 7 timed runs after
a warm-up, the two forms interleaved, nothing pinned. The drop-in's times are
its warm runs, with its engine already started. The eight decks are listed
A to H in order of the executable's B-spline time.

**nec5-razor-2p**

| Deck | Executable | Drop-in | Drop-in / executable |
|---|---|---|---|
| A | 0.22 s | 0.34 s | 1.57 |
| B | 0.21 s | 0.38 s | 1.80 |
| C | 0.72 s | 1.05 s | 1.45 |
| D | 0.86 s | 1.54 s | 1.79 |
| E | 1.77 s | 2.13 s | 1.20 |
| F | 4.69 s | 7.22 s | 1.54 |
| G | 5.74 s | 6.79 s | 1.18 |
| H | 21.97 s | 26.39 s | 1.20 |

**nec5-bspline**

| Deck | Executable | Drop-in | Drop-in / executable |
|---|---|---|---|
| A | 0.51 s | 0.55 s | 1.07 |
| B | 0.88 s | 1.14 s | 1.29 |
| C | 1.94 s | 2.13 s | 1.09 |
| D | 2.48 s | 2.64 s | 1.06 |
| E | 4.70 s | 6.12 s | 1.30 |
| F | 6.06 s | 9.27 s | 1.53 |
| G | 15.73 s | 14.08 s | 0.90 |
| H | 33.50 s | 32.01 s | 0.96 |

On the three largest decks here (E, G and H) the drop-in takes about 20 %
longer than the executable with razor-2p, and 4–6 % longer overall with
B-spline, where it is faster on two of the three. On small decks it takes 40–80 % longer
with razor-2p. That is a fixed cost of a tenth to two-thirds of a second per
run, not a cost that grows with the deck. Pinned to four cores with
`start /affinity 55`, the ratios are the same within the noise of those
small runs.

These are one machine's numbers on eight decks: take them as typical, not as
a guarantee for yours. The gap on the drop-in's slowest decks is tracked in
[momwire#2040](https://github.com/stevenmburns/momwire/issues/2040).

## The single-file build

The single-file build carries both the avx2 and the sse2 build of an engine
in one program, with no DLLs beside it. On the same laptop and decks, with a
build that predates 0.95.0, it ran at about the speed of the standalone
executable: within about 5 % over all eight decks, with neither form ahead
across the board. The standalone executable was 3–16 % faster on the largest
decks, and the single-file build was faster on some razor-2p decks. Its
printouts matched the standalone executable's to the last printed digit.

It is a preview: use it when one file matters more to you than the last few
per cent, and keep the standalone executable as the default.

## What was not measured

Nothing here was measured on Linux or macOS. There the standalone
executables and the Python engines are the choices, and the
[standalone page](/reference/standalone/) describes them.
