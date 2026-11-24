---
title: "Which momwire to run on Windows"
description: The standalone executable, the EZNEC drop-in, the single-file preview, or pip — which to use on Windows, and what each costs, measured on momwire 0.95.0's and 0.97.1's release builds.
---

momwire comes for Windows in four forms: three downloads and a Python
install. Any two of them print the same input impedance for a deck both
serve, so the choice is about speed and convenience, not about the answer.

| You want | Use |
|---|---|
| A NEC-5 deck, run directly or from EZNEC's or SimNEC's NEC-5 engine setting | The standalone executable, [`momwire-nec5-windows-x64.zip`](https://github.com/stevenmburns/momwire/releases/latest/download/momwire-nec5-windows-x64.zip) |
| A deck the executable refuses (its printout says `not served by this build`), the NEC-4.2 engine slot, a NEC-2 engine, or another formulation | The drop-in, [`momwire-windows.zip`](https://github.com/stevenmburns/momwire/releases/latest/download/momwire-windows.zip) |
| One file per engine, nothing beside it | The single-file build (preview), `momwire-nec5-windows-x64-single.zip`: `momwire-nec5-bs2.exe` and `momwire-nec5-razor2p.exe`, NEC-5 only |
| momwire from your own Python, as a library or as commands | `pip install momwire` (Python 3.10 or later); see [From Python](#from-python) |

## The standalone executable first

When the standalone executable accepts a deck, it is the faster way to
solve it, whether you run it yourself or point EZNEC's or SimNEC's NEC-5
engine setting at it. When it doesn't, it says so: the printout it writes
names what it refuses (`not served by this build: ...`) instead of the
answer. That is the cue to run the same deck through the drop-in, which
runs the Python engine and serves more decks.

The Windows executables are nec5-bs2 and nec5-razor2p. For the NEC-4.2
engine slot in EZNEC, SimNEC or 4nec2, for a NEC-2 engine, or for another
formulation, use the drop-in. Its NEC-2 engines, `momwire-nec2c.exe` and
`momwire-nec2c-sinusoidal.exe`, are the same engines as the `pip install
momwire` commands of those names, and answer SimNEC's NEC-2 engine setting and
nec2c's command line the same way ([NEC-2 launchers](/reference/portal-usage/)).
4nec2's NEC-2 slot takes `momwire-nec2d.exe` and `momwire-nec2d-sinusoidal.exe`
instead: the same engines, writing the printout layout of 4nec2's own NEC-2
engines, which is the one 4nec2 reads
([In 4nec2](/reference/eznec-nec5/#in-4nec2)).

The drop-in ships two launchers per dialect. Any other formulation is one
copy away: copy a launcher in the unpacked folder and rename the copy
`momwire-nec5-<basis>.exe`, `momwire-nec42-<basis>.exe`,
`momwire-nec2c-<basis>.exe` or `momwire-nec2d-<basis>.exe`, and that basis
answers in that dialect
([Choosing the formulation](/reference/eznec-nec5/#choosing-the-formulation)).
A name carrying `nec2c` or `nec2d` is read first, so a copy named for NEC-2 is
a NEC-2 engine whichever launcher it was copied from, and a basis after
`nec2c-` that does not exist is refused by name rather than served as the
default. The
[standalone page](/reference/standalone/) lists what the executables carry.

Leave the thread settings alone. Both forms size their thread pools from the
deck and from the cores you allow them, and they choose the same counts, so
neither needs a variable set to run at its best. The
[threads section](/reference/standalone/#threads) covers the cases where
setting `MOMWIRE_THREADS` helps.

## What the drop-in costs

Measured on one Windows laptop with 16 logical processors, using the signed
momwire 0.95.0 release zips. Each figure is the median of 7 timed runs after
a warm-up, the two forms interleaved, nothing pinned. The drop-in's times are
its warm runs, with its engine already started. The eight decks are listed
A to H in order of the executable's B-spline time.

**nec5-razor2p**

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

**nec5-bs2**

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

On the four slowest decks (E to H), the drop-in takes
18–20 % longer than the executable with razor2p on three of them, and 54 %
longer on F. With B-spline it ranges from 10 % faster (G) to 53 % longer
(F). F is the outlier in both. The drop-in's overhead is tracked in
[momwire#2040](https://github.com/stevenmburns/momwire/issues/2040). On the small decks (A to D) it takes 45–80 % longer with razor2p. That
is a fixed cost of a tenth to two-thirds of a second per run, not a cost
that grows with the deck. Pinned to four cores with
`start /affinity 55`, the ratios are the same within the noise of those
small runs.

These are one machine's numbers on eight decks: take them as typical, not as
a guarantee for yours.

## One process per run, or a resident engine

The two forms start differently, and that matters most to EZNEC, which
starts its engine once for every frequency of a sweep.

- **The standalone executable is a new process per run.** It has no Python
  to start, so its first run is quick, but it keeps nothing in memory
  between runs. It does keep the Sommerfeld ground tables in a disk cache, so
  a run on a soil it has seen before reads them back instead of building
  them again.
- **The drop-in keeps a resident engine.** The first calculation starts it,
  later ones reuse it, and it retires after fifteen idle minutes. Besides
  skipping the start-up, it keeps some of what it computed in memory from
  one run to the next, such as its near-ground tables.
- **SimNEC's streaming call** (`- -`, a chain of decks on standard input)
  lets one standalone process answer the whole chain, so it keeps its
  ground tables in memory from one deck to the next.
- **The drop-in's NEC-2 engines start one engine per run.** SimNEC's NEC-2
  protocol streams every deck of a run into one engine process, so the
  engine's start-up is paid once per run, not once per deck, and there is no
  resident engine between runs. The pip install's `momwire-nec2c-shared`
  keeps one; the drop-in does not ship it.

The tables above time single runs, on 0.95.0. A sweep that starts the engine
at every frequency, the way EZNEC does, was re-measured on the signed
momwire 0.97.1 zips on the same laptop, with nec5-razor2p and 51
single-frequency decks launched one at a time (median of 3 interleaved
loops for warm):

| Deck | Standalone, warm | Drop-in, warm | Standalone, cold | Drop-in, cold |
|---|---|---|---|---|
| Fan dipole over Sommerfeld ground | 3.39 s | 3.66 s | 3.42 s | 4.70 s |
| The same fan dipole in free space | 1.96 s | 2.09 s | 2.35 s | 3.25 s |

Warm, with the drop-in's engine already running and the standalone
program's ground tables already cached, the drop-in takes 7–8 % longer:
about 71 ms a launch against 66 over Sommerfeld ground, 41 against 38 in
free space. Cold, most of the drop-in's extra time is its first launch,
about 1.2 s, which starts its engine; the standalone program's first launch
took 0.1 s. The two forms' printouts were byte-identical at all 51 points.

The drop-in writes advisories to its standard error only when
`MOMWIRE_ADVISORIES` asks for them ([advisories](/reference/eznec-nec5/#advisories-on-stderr-when-you-ask)).
EZNEC never reads that stream. On momwire 0.97.0's zips, which wrote them on
every launch, the drop-in took 36 % and 52 % longer than the standalone
program warm on these two sweeps; writing them cost about 15–23 ms a launch.

## From Python

`pip install momwire` puts the whole engine in your own Python (3.10 or
later). It installs the same commands the drop-in has, `momwire-nec5-<basis>`,
`momwire-nec42-<basis>` and the NEC-2 engines `momwire-nec2c-<basis>` and
`momwire-nec2d-<basis>` ([NEC-2 launchers](/reference/portal-usage/)), running the same Python
engine, one command per basis, plus two things neither zip has:

- `momwire-nec2c-shared`, a NEC-2 engine that stays resident between runs;
- the Python library itself, for scripts and for
  [antennaknobs](https://antennaknobs.dev).

It updates with `pip install -U momwire`. Point a host program's engine
setting at the command's full path, which `where momwire-nec5-razor2p`
prints in a Command Prompt.

## The single-file build

The single-file build carries both the avx2 and the sse2 build of an engine
in one program, with no DLLs beside it. On the same laptop and decks, with a
build that predates 0.95.0, it ran at about the speed of the standalone
executable: within about 5 % over all eight decks, with neither form ahead
across the board. The standalone executable was 3–16 % faster on the largest
decks, and the single-file build was faster on some razor2p decks. Its
printouts matched the standalone executable's to the last printed digit.

It is a preview: use it when one file matters more to you than the last few
per cent, and keep the standalone executable as the default.

## What was not measured

Nothing here was measured on Linux or macOS. There the standalone
executables and the Python engines are the choices, and the
[standalone page](/reference/standalone/) describes them.
