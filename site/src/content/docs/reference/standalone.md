---
title: "The standalone executables"
description: momwire's razor-2p and bs2 NEC-5 engines as native programs with no Python in them — downloads for Linux, Windows and macOS, which program to run, and how to set the thread count.
---

Each release since 0.79 carries two of momwire's NEC-5 engines as native
programs with no Python in them: **razor-2p** (what the Python command
`momwire-nec5-razor-2p` answers) and **bs2**, the default degree-2 B-spline
formulation (what `momwire-nec5-bspline` answers). They read a NEC-5 deck
and write the same printout the Python engines write. Nothing needs
installing: no Python, no pip, no momwire.

Each takes the forms the licensed console engine takes:

```bash
momwire-nec5-razor2p-linux deck.nec deck.out   # EZNEC's and SimNEC's form
momwire-nec5-razor2p-linux - - < chain.nec     # SimNEC's streaming form
momwire-nec5-razor2p-linux -version
```

To use one from SimNEC or EZNEC, point the NEC-5 engine setting at the
program, inside its unpacked folder. A deck an engine does not serve yet is
refused in the printout by name (`not served by this build: ...`), and the
Python command serves it. The other formulations stay Python.

## Downloads

All three are assets on the latest
[GitHub release](https://github.com/stevenmburns/momwire/releases/latest).

| platform | asset | status |
| --- | --- | --- |
| Linux, 64-bit x86 | [`momwire-nec5-linux-x86_64.tar.gz`](https://github.com/stevenmburns/momwire/releases/latest/download/momwire-nec5-linux-x86_64.tar.gz) | released |
| Windows 10/11 and 7 SP1, 64-bit x86 | [`momwire-nec5-windows-x64.zip`](https://github.com/stevenmburns/momwire/releases/latest/download/momwire-nec5-windows-x64.zip) | preview, signed |
| macOS 14 or later, Apple silicon | [`momwire-nec5-macos-arm64.tar.gz`](https://github.com/stevenmburns/momwire/releases/latest/download/momwire-nec5-macos-arm64.tar.gz) | preview, unsigned |

**Linux.** The tarball holds an avx2 and an sse2 build of each engine, and
a small `sh` picker for each, `momwire-nec5-razor2p-linux` and
`momwire-nec5-bs2-linux`, that runs the build this processor can run. Point
SimNEC at the picker and keep the folder together. They need glibc 2.28 or
newer (RHEL/Alma 8, Debian 10, Ubuntu 18.10 and later); everything else is
built in.

```bash
tar -xzf momwire-nec5-linux-x86_64.tar.gz
cd momwire-nec5-linux-x86_64
./momwire-nec5-razor2p-linux -version
./momwire-nec5-bs2-linux deck.nec deck.out
```

**NEC-4.2, Linux (preview).** From 0.83.0 the Linux tarball also holds
`momwire-nec42-sinusoidal-linux`, what the Python command
`momwire-nec42-sinusoidal` answers for EZNEC's and SimNEC's NEC-4.2 slot,
with its own avx2 and sse2 builds and picker. Point the NEC-4.2 engine
setting at it. Its printout is the Python command's byte for byte except
the two measured-time lines (`FILL= ... SEC.` and `RUN TIME =`). As a
preview it still refuses by name a wire below the ground plane (buried
wires), a GD far-field ground and an insulated wire (IS); the Python
command serves those decks.

**Windows (preview).** The zip holds `momwire-nec5-razor2p-avx2.exe` and
`momwire-nec5-bs2-avx2.exe`, the `-sse2` pair, and a `win7` folder with an
sse2 pair built to run on Windows 7 SP1. The programs are signed with
momwire's certificate. The DLLs they load ship beside them, so keep each
folder together. The zip's `README.txt` says which program to use.

**macOS (preview).** One build of each engine,
`momwire-nec5-razor2p-macos` and `momwire-nec5-bs2-macos`, using the
system's Accelerate library, with nothing beside them. They are ad-hoc
signed only: no Developer ID and no notarisation. Fetched with `curl` and
unpacked with `tar`, they carry no quarantine flag and run as they are:

```bash
curl -LO https://github.com/stevenmburns/momwire/releases/latest/download/momwire-nec5-macos-arm64.tar.gz
tar -xzf momwire-nec5-macos-arm64.tar.gz
cd momwire-nec5-macos-arm64
./momwire-nec5-razor2p-macos -version
```

A copy downloaded through a web browser is quarantined, and macOS refuses
to open it until the flag is cleared from the whole folder:

```bash
xattr -dr com.apple.quarantine momwire-nec5-macos-arm64
```

## Which program to run

- **avx2 or sse2.** The avx2 build needs a processor with AVX2 and FMA
  (Intel from Haswell on, apart from some Celeron and Pentium models; AMD
  from Excavator on, and every Ryzen). The sse2 build runs on any 64-bit x86
  processor, only more slowly. On Linux the picker chooses for you. On
  Windows, if you are not sure, use sse2: an avx2 program on a processor
  without AVX2 does not crash, it refuses each deck in the printout and
  says why.
- **Windows 10/11 or Windows 7.** On Windows 10 or 11 use the programs at
  the top of the zip. On Windows 7 SP1 use the `win7` folder; it carries
  Microsoft's Universal C Runtime for Windows 7, and its programs also run
  on Windows 10 and 11.
- **razor-2p or bs2.** The same choice as between the Python engines: see
  [Choosing the formulation](/reference/eznec-nec5/#choosing-the-formulation).

## How they differ from the EZNEC drop-in

The drop-in, [`momwire-windows.zip`](/reference/eznec-nec5/#setting-it-up)
(named `momwire-eznec-windows.zip` before 0.80.0), is still the supported way
to run momwire under EZNEC, SimNEC or 4nec2 on Windows. The two are built for different launch patterns:

- A standalone program is **a new process per run**, with no Python to
  start, so it answers the first run of a session quickly.
- The drop-in keeps **a resident engine** that the first run starts and
  later runs reuse (see [Launch economics](/reference/eznec-nec5/#launch-economics)).
  Its first run pays that start-up; its warm runs are close to the
  standalone program's.

SimNEC's streaming call (`- -`, decks on stdin separated by `NX`, printouts
on stdout) works with the standalone programs: one process answers the
whole chain.

## Threads

Unless told otherwise, each run picks its own thread count from the size
of the deck: one thread for a small deck, a few for a middling one, and one
per physical core for a large one. It never uses more threads than the
machine has physical cores. The environment variable `MOMWIRE_THREADS` fixes
the count for every deck.

- **Many independent runs** (a sweep split into separate decks, a batch of
  designs): set `MOMWIRE_THREADS=1` and run as many at once as the machine
  has physical cores. Never run more threads in total than physical cores.
- **One very large deck** (a free-space deck of roughly 4000 segments or
  more) wants every core for itself: leave `MOMWIRE_THREADS` unset and run
  it alone.
- **A chain of related decks**: stream it through one process (`- -`)
  rather than launching the program once per deck.

## Where the time goes

Set the environment variable `MOMWIRE_TIMES=1` to see how long each part of
a run takes. It is off unless you set it, and it changes no result. With it
set, two things change:

- The printout's timing lines carry measured seconds, in the places NEC-5
  prints them: `FILL=` and `FACTOR=` under `- - - MATRIX TIMING - - -` for
  each run that solves a matrix, and `RUN TIME = ... SEC. (WALL CLOCK)` at
  the end. Without the variable momwire prints zeros there, and none of the
  lines below.
- The printout also carries what a timed NEC-5 build prints. At the top is
  the environment in effect: `MOMWIRE_THREADS`, `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, the threads the fill and the factor actually ran
  on, the build (`avx2`, `sse2`, ...), `MOMWIRE_BELOW_ANCHORS` and the ground
  table cache's folder (or `OFF`). Under each MATRIX TIMING come the run's
  phases:

```text
                        FILL=    0.053 SEC.,  FACTOR=    0.001 SEC.
                        GROUND TABLES=    0.010 SEC.  (BUILT 4, FROM DISK CACHE 0, ANCHOR-BLENDED 1)
                        FILL: 53 msec (4 threads) FACTOR: 1 msec (4 threads)
                        SOLUTION (PORTS+CURRENTS+POWER)=    0.000 SEC.
                        RADIATED FIELD (RP)=    0.000 SEC.
                        NEAR FIELD (NE/NH)=    0.000 SEC.
```

  `GROUND TABLES` counts the Sommerfeld tables used since the previous run:
  how many were computed, how many came from the disk cache, and how many
  were interpolated from anchor tables.
- Each answer writes one `times:` line on standard error, with every stage
  in milliseconds:

```text
times: route=cxx threads=4 runs=1 start_ms=2.378 load_ms=0.000 seam_ms=0.478 ground_ms=9.752 fill_ms=50.326 factor_ms=1.209 rp_ms=0.022 near_ms=0.000 run_ms=0.062 printout_ms=0.371 write_ms=0.067 other_ms=0.195 total_ms=64.860 ground_built=4 ground_disk=0 ground_blended=1
```

| key | what it counts |
|---|---|
| `start_ms` | from the operating system starting the program to its first line of code (the standalone programs) |
| `load_ms` | loading Python and momwire (the Python engines, and a standalone program's fallback) |
| `seam_ms` | reading the deck and building the structure |
| `ground_ms` | the Sommerfeld ground tables; `ground_built`, `ground_disk` and `ground_blended` count tables computed, read from the disk cache, and interpolated from others |
| `fill_ms`, `factor_ms` | the matrix fill, and the LU factorisation and solve |
| `rp_ms`, `near_ms`, `run_ms` | the radiation pattern, the near fields, and the rest of each run (currents, power) |
| `printout_ms`, `write_ms` | writing the printout's text, and writing the file |
| `other_ms`, `total_ms` | what no stage claims, and the whole; the stages and `other_ms` add up to `total_ms` |

The keys are always present and always in this order, so a script can split
on spaces and `=`. A stage the run does not have reads 0. In SimNEC's
streaming form each structure writes its own line. On Linux `start_ms` is
only good to about 10 ms, which is how finely the system records when a
program started.

`FILL=` leaves out the ground tables, which `ground_ms` reports. They are
built the first time a ground is used and kept in a disk cache afterwards,
so the first run on a soil can be much slower than the next.

To time runs that SimNEC or EZNEC starts, set the variable on the command
line that starts that program, so that the engine inherits it. On a Mac:

```bash
MOMWIRE_TIMES=1 /Applications/SimNEC.app/Contents/MacOS/SimNEC
```

On Windows, type `set MOMWIRE_TIMES=1` in a Command Prompt and start the
program from that same window. The printout's timing lines then show in the
program's NEC output, and the `times:` lines go to the terminal you started
it from, if the program passes the engine's standard error through.

The Python commands (`momwire-nec5-razor-2p`, `momwire-nec5-bspline` and the
others) honour the same variable and write the same line. On those, warm
runs answered by the resident engine report `route=resident`, and
`other_ms` includes starting or reaching that engine.
