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

The EZNEC drop-in, [`momwire-eznec-windows.zip`](/reference/eznec-nec5/#setting-it-up),
is unchanged and is still the supported way to run momwire under EZNEC on
Windows. The two are built for different launch patterns:

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
