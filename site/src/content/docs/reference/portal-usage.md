---
title: "Running momwire as a NEC-2 engine"
description: Install the NEC-2 launcher (momwire-nec2c), point SimNEC at it, pick an engine by name (one command per solver), read the version probe and the refusals, and swap in the shared resident engine.
---

[SimNEC](https://ae6ty.com/smith_charts/) (AE6TY) does not link its NEC
solver — it shells out to a `nec2c` executable, starts one copy, and keeps
it. Decks go down that process's stdin framed by an `NX` card, printouts come
back on stdout, and SimNEC's own circuit solver reads two numbers per
feedpoint out of each printout to build the antenna's Y matrix.

momwire ships that process. `pip install momwire` puts **`momwire-nec2c`** on
your path: the **NEC-2 launcher** (`momwire-nec2c` and its siblings), a
drop-in for the engine binary with momwire's solver behind it. This page
covers SimNEC; 4nec2's NEC-2 engine takes the `momwire-nec2d` twins, the same
engine writing the NEC-2D printout layout 4nec2 reads
([In 4nec2](/reference/eznec-nec5/#in-4nec2)). The Smith chart, the
tuner and the sweeps stay SimNEC's; the electromagnetics become momwire's.

The deck language is momwire's own **`nec2` dialect** — wire structures and
the two-port cards attached to them, specified card by card in
[the deck grammar](/reference/deck-grammar-nec2/).
This page is about running the engine; that page is about what it accepts.

:::caution[Not a supported configuration]
No part of SimNEC knows momwire exists, and nothing here has been reviewed or
endorsed by SimNEC's author. Treat it as a cross-check you can run yourself.
:::

## Install

```bash
pip install momwire
which momwire-nec2c        # e.g. ~/.venvs/momwire/bin/momwire-nec2c
momwire-nec2c -version     # NEC2 momwire-nec2c [sb] <version>
```

`python -m momwire.dropin.nec2c` is the same program under its long name, for an
environment where the console script is awkward to reach.

**On Windows without Python**, the drop-in zip,
[`momwire-windows.zip`](https://github.com/stevenmburns/momwire/releases/latest/download/momwire-windows.zip),
ships `momwire-nec2c.exe` and `momwire-nec2c-sinusoidal.exe` (and the
`momwire-nec2d` pair, for 4nec2). Unzip it into a
folder of its own (`C:\momwire\`) and give SimNEC the full path of one of
them. They are the same engine as the pip commands of the same names and
answer the same way, the `-version` probe, the `NX` stream and
`--selftest` included, and a copy you rename `momwire-nec2c-<basis>.exe` in
that folder selects its basis by the rule below. They start the engine once
per SimNEC run and keep nothing resident between runs; the resident
`momwire-nec2c-shared` comes with pip only.

Before a live session, run the built-in smoke. It needs no checkout: it
spawns one resident copy of itself, sends embedded decks through it the way
SimNEC does, exercises every basis, and prints `PASS` or `FAIL`.

```bash
momwire-nec2c --selftest
```

## Pointing SimNEC at it

Open SimNEC's NEC portal dialog and give it the path the `which` above
printed. Two rules decide whether SimNEC accepts the command, and both fail
in ways that look nothing like their causes.

**The filename must contain `nec2c`.** SimNEC picks the engine — deck
dialect, daemon protocol, printout parse offsets, all of it — off the
command's file name, lowercased, before anything inside the file is
consulted. A name carrying none of `nec2c` / `nec5` / `nec42` is refused with
*NO NEC Command Available*, and a name carrying `nec5` or `nec42` selects a
different column layout that this engine does not emit. That is why the
script is called `momwire-nec2c` and not something tidier.

**The full path must not contain the substring `out`.** SimNEC refuses any
command whose path contains it — *Can't execute 'out' file:* — a guard
against pointing the engine field at a printout file. The shipped name is
clear of it; your install prefix may not be. A venv under
`~/checkouts/routing/` or `C:\Users\scout\` refuses this engine no matter
what it is called, and the fix is to install momwire somewhere else.

Both rules apply to wrappers and symlinks exactly as they do to the script.

## Choosing the physics: one command per engine

Every solver is its own command — `pip install momwire`
puts all of these on your path, and SimNEC's engine dialog takes any of them
directly, no arguments and no wrapper scripts:

```text
momwire-nec2c                                # the default — degree-2 B-spline (bs2)
momwire-nec2c-bs1                     # the same physics at degree 1 (tent basis)
momwire-nec2c-sinusoidal                     # NEC-2's own formulation
momwire-nec2c-sg            # the same basis, tested variationally
momwire-nec2c-hmatrix                        # same physics, hierarchical (ACA) solve
momwire-nec2c-arrayblock                     # same physics, element-block/FFT solve
momwire-nec2c-pulse                          # Harrington's pulse basis, point-matched
```

The mechanism is the *name*: everything after `nec2c-` in the executable's
basename selects the basis (a Windows `.exe` suffix is stripped first), so a
copy, symlink or hard link you make yourself works exactly like the shipped
commands. A name that matches no solver fails the `-version` probe loudly at
configure time, naming its source. The same choice has two other spellings
with a fixed precedence — an explicit `--basis <name>` flag beats the
filename, and the filename beats a `MOMWIRE_NEC2C_BASIS` environment
variable; the plain `momwire-nec2c` with neither is the default.

Four axes live in that list. `sinusoidal` and `sg` are
the **fidelity** ladder: the three-term basis point-matched with NEC's own
delta-gap source answers "does this reproduce NEC-2's behaviour, mesh walk
and all", and the Galerkin entry and the B-spline default answer "what
does a better-converged basis say". There is one Galerkin name, with one
feed model: a feed model is not a choice a portal dialog can put to you
meaningfully, so `sg` uses the better of the two measured (momwire#654). There is no `sinusoidal-converged`: a
zero-width gap has no collocation right-hand side, so the flag does not offer
a name the solver refuses. `bs1` is the **degree** axis — the same
B-spline physics at degree 1, so pairing it with the default is the cheapest
convergence check available. `hmatrix` and `arrayblock` are the **solve**
axis, not a physics choice: the same B-spline operator held compressed and
solved iteratively, so a large array answers without a dense fill.
`arrayblock` is the one to reach for on a repeated-element array — identical
elements on a regular lattice become an FFT convolution over the element grid
— and it degrades to the hierarchical solve on a deck with no repeated
structure, so it is safe to leave on. Its answers must agree with `bs2`'s;
if they do not, the deck is telling you about conditioning, not about basis.
**There is no razor command here.** The tent-basis, razor-blade formulation
NEC-5 identifies with (`razor2p`) is not a nec2 engine: a NEC-2 deck puts
its source at a segment *centre* and razor places its gap at a *knot*, so
it would answer for an antenna half a segment from the one you drew. The
NEC-2 drop-in refuses a razor name at configure time rather than serve that
(momwire#821): a copy named `momwire-nec2c-razor2p`, `momwire-nec2c-razor-2p`
or `momwire-nec2c-razor-nec5` fails the version probe with the reason. Razor is served where its grid is the deck's:
the NEC-5 dialect's `momwire-nec5-razor2p`, for EZNEC's NEC-5 slot and
SimNEC's own NEC-5 engine alike ([below](#simnecs-nec-5-engine)), which
writes a source at a node.

**On this launcher `sinusoidal` takes NEC-2's free-end cap by default**
(momwire#2033). NEC-2 lets the current at a free wire
end run onto the end cap, I = ±(X/k) dI/ds with X = J₁(ka)/J₀(ka) (NEC
Part I, Eqs. 23 and 54–64); momwire's sinusoidal basis everywhere else takes
it to zero there. The NEC-2 drop-in exists to answer like NEC-2, so this one
route (`momwire-nec2c-sinusoidal` and `momwire-nec2d-sinusoidal`, or any
`momwire-nec2c*` or `momwire-nec2d*` run with `--basis sinusoidal`) uses NEC-2's condition unless told otherwise. It is a
compatibility choice, not better physics: a flat cap is worth about 0.10a of
extra length per end and this condition acts like three to four times that
on coarse meshes, so it moves Zin most on fat wires (up to 58 % on a fat-wire
log-periodic Yagi against the no-cap answer). The other bases, the NEC-4.2
and NEC-5 launchers and the Python API keep the zero-current end. The cap
is certified for the drop-ins on Linux and Windows; on macOS it runs the
same code, not yet certified there.

| `MOMWIRE_X_NEC2_END_CAP` | `momwire-nec2c-sinusoidal` |
|---|---|
| unset (the default) | the cap; a deck it cannot act on runs without it |
| `0` | no cap: the zero-current end, as every other route takes it |
| `1` | the cap; a deck it cannot act on is refused with a NEC ERROR |

Set the variable in the environment SimNEC starts the engine in. The decks
the cap cannot act on are a buried or ground-crossing wire with a free end
(NEC-2 has no wires below the interface) and, under
`MOMWIRE_JUNCTION_RULE=nec42`, a junction the NEC-4 rule solves with a free
end anywhere on the deck. By default those run without the cap, and each
solve says so at column 0 between ANTENNA ENVIRONMENT and MATRIX TIMING:
`NEC-2 END CAP NOT APPLIED: a buried or ground-crossing wire has a free end`.
The banner does not name the cap: on this route it is the default, as the
banner never names a default, and `=0` gives the uncapped printout byte for
byte.

**On a tapered or stepped-radius wire, `sinusoidal` is NEC-2-faithful, not
converged.** At a junction where two wire radii meet it follows NEC-2, and
that includes NEC-2's well-known stepped-radius weakness: it mostly ignores
the step, and more so as the mesh is refined. On a 6 m dipole at 14 MHz with
a 0.5 mm half joined to a 1 mm half, the step's effect on Z, in ohms
(momwire#1632):

| | 20 seg/half | 40 seg/half | 80 seg/half |
|---|---|---|---|
| NEC-2 (PyNEC) | −0.59 − 8.90j | −0.77 − 1.21j | −1.01 + 9.43j |
| `sinusoidal` (this launcher's default, NEC-2's end cap) | −0.59 − 8.90j | −0.77 − 1.21j | −1.01 + 9.43j |
| `sinusoidal`, `MOMWIRE_X_NEC2_END_CAP=0` | −0.59 − 8.86j | −0.76 − 1.18j | −1.01 + 9.47j |
| `sg` | −0.02 − 33.44j | −0.00 − 34.04j | +0.01 − 34.42j |
| `bs2` | −0.07 − 32.55j | −0.03 − 33.42j | −0.01 − 33.95j |
| NEC-5 | +0.03 − 34.69j | +0.03 − 34.80j | |

With the end cap, `sinusoidal` reads PyNEC's row to the printed digit at
every mesh; without it, to 0.04 Ω. That is by design: this row exists to be
the NEC-2-closest rung of the
ladder, and the nec2 slot is meant to match nec2c, so inheriting NEC-2's own
behaviour on a taper is what "closest" means there. The solver says so: every
`sinusoidal` solve with two radii at one junction raises a
`RadiusStepJunction` advisory naming the wires. A NEC-4.2-style step for the
NEC-4 dialect is momwire#1644. For the converged answer use `bs2` or
`sg`; `bs1` with `EK` armed (the `EK` card, above) is
the row measured closest to NEC-5 on Ward Harriman AE6TY's flagship 20:1
tapered dipole. `sg` REFUSES `EK` outright on any deck
where two wires of different radii meet at a junction — measured divergent
rather than merely inaccurate there; see the `EK` card entry in
[the nec2 deck grammar
reference](/reference/deck-grammar-nec2/#ek--extended-thin-wire-kernel).

**Two portal entries differing only in `--basis` give you cross-basis
validation inside SimNEC itself.** Switch engines from the dialog and watch
whether the answer holds. The printout banner records which physics answered
(`VERSION:nec2c.ae6ty.momwire.9.1+sg`), and a mistyped basis fails the
version probe loudly at configure time rather than serving the default.

### When the dialog is a file picker

It always is — and the named commands above exist precisely so that the file
*is* the choice. Point the dialog at `momwire-nec2c-sg` and
that is the engine; two entries differing only in name are two engines. Both
filename rules above apply to every spelling, shipped or hand-made: keep
`nec2c` in the name, and keep `out` out of the whole path.

## SimNEC's NEC-5 engine

SimNEC also has a NEC-5 engine slot, and momwire fills it too.
SimNEC decides which deck **syntax** to write from the engine's path, not from
anything the engine says: a path containing `nec2c` gets NEC-2 decks, one
containing `nec5` gets NEC-5 decks. So the NEC-5 slot takes a different family
of commands, and the `momwire-nec2c` names above will not work there:

```bash
momwire-nec5                        # the default — degree-2 B-spline (nec5-bs2)
momwire-nec5-razor2p               # NEC-5's own formulation (tent basis, razor testing)
momwire-nec5-bs1             # degree-1 B-spline
momwire-nec5-sg    # sinusoidal basis, tested variationally
momwire-nec5-hmatrix                # nec5-bs2, hierarchical (ACA) solve
momwire-nec5-arrayblock             # nec5-bs2, element-block/FFT solve
```

`razor2p` is the one to pick for agreement with the licensed NEC-5: it runs
the same algorithm, and in SimNEC it reproduces the licensed engine's
impedances to about 0.01 Ω on the examples we have run. There is no
`sinusoidal` or `pulse` command in this family: a NEC-5 deck puts its source
at a **knot**, and those two bases can only feed a segment centre, so they
would refuse every deck SimNEC writes.

Setup is the same as for the NEC-2 slot: `which momwire-nec5-razor2p` and give
SimNEC that absolute path (one without spaces is safest). Its `-version` probe
answers `NEC5 momwire-nec5-razor2p [sb] <version>`. These commands run the same warm engine
as the EZNEC drop-in, so a run after the first answers in milliseconds, and
SimNEC's N-port measurement (one run block per port in a single deck) is served
as NEC-5 serves it. On Windows without Python, the drop-in zip,
[`momwire-windows.zip`](https://github.com/stevenmburns/momwire/releases/latest/download/momwire-windows.zip),
ships `momwire-nec5.exe` and `momwire-nec5-razor2p.exe`, the same names it
tells EZNEC's NEC-5 slot to use.

## The version probe

SimNEC runs `<command> -version` and reads the first line. Every momwire
launcher answers four fields separated by spaces (momwire#1606):

```text
NEC2 momwire-nec2c [sb] 0.82.0
```

1. The deck dialect the launcher reads: `NEC2`, `NEC42` or `NEC5`.
2. The launcher's name, which always starts with `momwire`. A no-Python
   executable from the [standalone downloads](/reference/standalone/) adds
   `-standalone` and the build it is (`avx2`, `sse2`, or `legacy` on macOS)
   to the name of the Python command it stands in for, e.g.
   `NEC5 momwire-nec5-razor2p-standalone-avx2 [sb] 0.87.1` beside the
   command's `NEC5 momwire-nec5-razor2p [sb] 0.87.1` (momwire#1831). That is
   the only difference: the printouts are the same.
3. The interface modes it serves beyond one deck per call, in square
   brackets: `s` for streaming, `b` for batch (decks chained by `NX`). Every
   launcher answers `[sb]`.
4. The full momwire version.

The dialect comes first because SimNEC takes the engine number from the first
digit of the line, and its NEC-2 and NEC-4.2 checks need 2 and 4 there.
SimNEC shows the string in the portal
dialog's `NECVersion` row, stamps it as a `CM version` comment card on every
deck it sends, and stores it in saved circuits — so a momwire session is
identifiable as one everywhere the version travels.

**The number is momwire's own version** — the engine's, which is the number
that changes solve results. An engine release moves it; nothing else does.

Every NEC-2 launcher answers this one form. A portal dialog line that carries
the retired `--legacy-probe` flag is refused at the probe with a line saying
so, and the fix is to delete the flag. The
probe line names the launcher, not
the engine: `--basis` never changes it, and only the printout banner records
which basis answered. A copy named for a basis (`momwire-nec2c-pulse`) gives
that name.

## Running a deck from the command line

The NEC-2 drop-in reads a deck file directly, the way you would test any NEC engine:

```bash
momwire-nec2c --basis sinusoidal < dipole.nec > dipole.printout
```

One framing rule makes this work: a deck is solved when its terminator card
arrives. Inside SimNEC that card is `NX`, appended automatically. Standalone,
a stock `.nec` file's final `EN` card does the same job and then ends the run.
End of input terminates a deck too, exactly as `EN` does — so a file with
neither, a deck that stops at `XQ`, solves and exits the same way.

The NEC-2 commands also take the call forms every momwire launcher answers
(momwire#1611):

```bash
momwire-nec2c dipole.nec dipole.out     # input file, output file
momwire-nec2c - - < chain.nec           # streaming: decks on stdin, printouts on stdout
momwire-nec2c < nec2d.tmp               # 4nec2's call form: the two file names on stdin
```

The two-file form writes the same printout the redirect above prints, with
Windows (CRLF) line endings, as every momwire launcher writes its printout
file. A deck it cannot read gets a printout that says so. `- -` is SimNEC's
own NEC-2 protocol spelled out, so it prints exactly what the bare command
prints. With no arguments, the first line on stdin decides: a NEC card means
decks, as SimNEC sends them, and anything else (or the name of a file that
exists) is the input file's name, with the output file's on the next line, as
4nec2 writes `nec2d.tmp`. 4nec2 itself reads only the NEC-2D layout, so its
NEC-2 engine is `momwire-nec2d`, which takes the same forms.

## Saving the decks SimNEC sends

A SimNEC circuit whose antenna is written in SimNEC's own script language
(`NECWire(...)`, `NECSource(...)`, `dcl` variables) has no cards in its `.ssn`
file. SimNEC runs the script and sends the resulting deck to the engine. To get
that model out as a `.nec`, have the engine keep a copy of every deck it
receives:

```bash
mkdir -p ~/simnec-decks
momwire-nec2c --save-decks ~/simnec-decks
```

Put that command in SimNEC's portal dialog, run the circuit once, and pick up
the newest file from the directory. Each deck is saved as
`<date>-<time>-<n>.nec`. The cards come through as SimNEC sent them, with its
`CM version` comment first, and the file ends in `EN` so it reads as one
standalone deck. antennaknobs opens it directly, as does any NEC-2 program.

The option is off unless you ask for it. `MOMWIRE_NEC2C_SAVE_DECKS=<dir>` does
the same for a host that passes environment instead of arguments, and the flag
wins if both are given. The directory must already exist; if it doesn't, the
`-version` probe fails at once. Saving never changes the answer, and a deck
that cannot be written (a full disk, say) is still solved.

## What refusals look like

A deck this engine cannot model is **reported and stepped over**, never
guessed at. The printout names the offending card and why, and still carries
the `NX` sentinel SimNEC blocks in `readLine()` waiting for — an engine that
dies or forgets the sentinel hangs SimNEC's UI with no timeout, which is
strictly worse than an error message. The daemon survives and runs the next
deck.

The refusal leads with a line whose first word is exactly `ERROR:`, which is
what trips SimNEC's own *NEC ERROR (1)* frame, so a refused deck surfaces as
a visible error dialog rather than an empty result with no explanation.

The complete list — cards refused by name, fields refused by value, and the
structural refusals — is the grammar's, with the exact message text:

* [cards refused by name](/reference/deck-grammar-nec2/#cards-refused-by-name)
  — the out-of-dialect geometry cards, `SP` / `SM`, and the rest of what is
  not a wire;
* [fields refused by value](/reference/deck-grammar-nec2/#fields-refused-by-value)
  — `EX` types other than 0, `RP` modes 1 and 4–6, spherical `NE` / `NH`
  grids, near fields over the `GN 0` reflection-coefficient ground, `GN`
  radial ground screens, the
  unsupported `LD` types, the four `IS` cases a lossless whole-wire jacket
  cannot express, and the three `TL` / `NT` cases NEC itself halts on or
  destroys in silence;
* [structural refusals](/reference/deck-grammar-nec2/#structural-refusals) —
  malformed cards, and addresses that name nothing.

Two behaviours upstream of any engine are worth knowing, because no engine
can report them:

* **SimNEC deletes `SP` cards from a script silently** before the deck
  reaches a NEC-2-class engine. A pasted patch model solves as bare wire with
  no warning from anyone. SimNEC's own *Surface elements* are properly
  wire-gridded instead, and those run here fine.
* **No apostrophes in `CM` / `CE` lines** pasted into the script editor —
  SimNEC's script parser reads `'` as a quote and errors.

## One fill per geometry

SimNEC probes an N-port antenna by sending N excitation groups in one deck,
and a stock `nec2c` refills and refactors the whole moment matrix for each —
N fills for one matrix. This engine takes the union of every group's ports,
fills and factors **once** per geometry and frequency, and answers each group
by back-substitution on the cached factors. A three-port deck costs one fill,
not three.

The deck need not be the boundary either. The protocol is stateless — a sweep
arrives as N separate decks, each re-sending the whole geometry — but the
engine may remember. With `--cache`, a solved structure is kept under a key
built from everything that decides its moment matrix (geometry after
transforms, ground, loading, port placement, kernel flag, basis), so a knob
dragged back to a value already probed, or a restarted sweep, is answered
without parsing, meshing or filling anything. The same structure at a new
frequency reuses the mesh and pays only the fill.

That is **off by default**, because what the cache exploits is how often a
real session re-sends a structure it has already sent — a property of your
sessions, not of the bench. Measure first:

```text
momwire-nec2c --cache-stats /tmp/nec-cache-stats.json
```

Put that in the portal dialog's engine command and run a normal session.
Every deck is solved exactly as it is without the flag — nothing is cached,
nothing retained, the answers are the stock answers — but the engine computes
each deck's key and counts how many decks a cache *would* have served. The
file is rewritten after every deck, because SimNEC ends a session by killing
the process and anything written at exit is never written:

```json
{"mode": "dry-run", "decks_rendered": 412, "hits": 273, "misses": 139,
 "fills": 412, "evictions": 0, "bytes": 0, "entries": 0}
```

`hits` against `decks_rendered` is the saving on offer. SimNEC runs a **crew**
of engine processes, and each counts only its own stream — give each portal
entry its own stats path (`--cache-stats /tmp/nec-stats.$$.json` in a
wrapper) or the crew members overwrite one file. If the numbers justify it:

```text
momwire-nec2c --cache
momwire-nec2c --cache --cache-stats /tmp/nec-cache-stats.json
```

The first serves; the second serves *and* keeps measuring, which is the one
to run for a while after switching over — `entries` and `bytes` then report
what the process is actually holding. Neither flag changes a byte of the
printout, in any mode.

### Pair reuse within one fill

A big deck repeats itself: many pairs of segments sit in the same pose
relative to each other. `MOMWIRE_X_PAIR_REUSE=1`, set in the environment
SimNEC starts the engine in, fills each class of repeated pairs once and
copies the result to the rest. It is one switch for every drop-in and every
standalone executable, and each turns on its own engine's reuse. It is
**off by default**.

- **`sinusoidal`** (`momwire-nec2c-sinusoidal`, `momwire-nec2d-sinusoidal`,
  `momwire-nec42-sinusoidal`, and the nec42-sinusoidal and nec2c-sinusoidal
  executables) finds the classes itself, in free space and over a perfect or
  finite ground, on decks large enough to pay for the search; a small deck
  runs the ordinary fill. It moves Z by at most about 5e-10 relative. The
  nec2c-sinusoidal executable reuses only under the extended kernel (an `EK`
  card); without one it runs the ordinary fill, where the Python drop-in
  reuses either way.
- **`razor2p`** (the NEC-5 drop-in and the nec5-razor2p executable) does the
  same, and moves Z by at most about 2e-9 relative.
- **`bs2`** (the drop-ins and the nec5-bs2 and nec42-bs2 executables) finds
  the classes itself on decks of 200 segments or more, a wire's own entries
  among them (one class per segment offset), and over a ground its image
  term and Sommerfeld remainder too. On any deck it also fills a uniform
  wire's own regularised moments once per segment offset. It moves Z by at
  most about 5e-10 relative.

`MOMWIRE_X_PAIR_REUSE=0` turns all of it off. Two older names are
deprecated. They still work, each for its own engine only, until two minor
releases after 0.99.0:

- `MOMWIRE_X_RAZOR_PAIR_REUSE` for `razor2p`;
- `MOMWIRE_X_OFFSET_REUSE` for `bs2`.

Each one now means `MOMWIRE_X_PAIR_REUSE` on that engine, and the
executables say so on stderr. When `MOMWIRE_X_PAIR_REUSE` is set too, it
wins.

Every one of these is far below the printed digits. It pays where one
geometry is solved many times; if it is slower on your deck, turn it off.

The NEC-4.2 and NEC-5 printouts name it on line 2 (`+pair_reuse`) wherever
it runs.
The NEC-2 printout keeps nec2c's layout and does not name it.

### Rotational symmetry for radial screens

A vertical over a radial screen is N identical sectors turned about one
vertical axis. `MOMWIRE_X_ROTATIONAL_SYMMETRY=1`, set in the environment
SimNEC starts the engine in, fills one sector's rows of the matrix instead of
all of them and solves one small block per harmonic instead of the whole
matrix. It is **off by default**, and only this variable turns it on: no deck
gets it by being symmetric.

A deck qualifies when the wires the rotation leaves in place lie on one
vertical line (the axis), every other wire belongs to one of N copies of the
same sector (N at least 2), and the sources, loads and ground are the same
in every sector, as a source on the axis is. Free space, a perfect ground
and a finite ground all qualify, with the screen above the ground, or below
it through the NEC-2 and NEC-4.2 drop-ins. A deck that does not qualify is
solved the ordinary way, with the ordinary answer to the bit, and the reason
is written to stderr as an advisory (on the NEC-5 and NEC-4.2 drop-ins when
`MOMWIRE_ADVISORIES` is on). On the decks it serves it moves Z by at most
about 1e-9 relative, far below the printed digits.

It runs on `bs2` and `bs1` (the NEC-5, NEC-4.2 and NEC-2 drop-ins) and on
`sinusoidal` through the NEC-2 drop-in (`momwire-nec2c-sinusoidal`). The
NEC-4.2 drop-in's `sinusoidal` engine solves every deck the ordinary way,
because this route does not serve the NEC-4.2 junction yet, and `razor2p`
ignores the variable. It is certified for the drop-ins on Linux and Windows;
on macOS it runs the same code, not yet certified there. The NEC-2 drop-in's
printout keeps nec2c's layout and does not name it; the NEC-4.2 and NEC-5
drop-ins name it on printout line 2 (`+rotational_symmetry`, followed by
`uncertified on this platform` on macOS). The standalone executables refuse
it.

### Turning a feature on in the deck

A variable set with `setx` turns a feature on for every deck the engine
answers, including the one-off solves where reuse costs more than it saves.
A deck can ask for one for itself instead, in a comment card:

```
CM momwire: pair_reuse
```

The card asks for that one deck exactly what the variable set to `1` would,
and nothing more. Two features take it, the two that cannot change a printed
digit: `pair_reuse` and `rotational_symmetry`. The rules:

- **One feature a line.** The text after `CM` starts with `momwire:` (any
  case), then the feature's name; the rest of the line is ignored. For both
  features, use two lines.
- **The variable wins.** `MOMWIRE_X_PAIR_REUSE=0` turns reuse off whatever
  the deck says: it is the switch for turning the pragma off everywhere. Set
  to `1`, the variable does what it does today, with or without the card.
- **It asks, it does not insist.** A deck the feature cannot serve runs the
  ordinary way, as it would under the variable; the card never refuses a
  deck.
- **A name it does not take is ignored**, and line 2 of the NEC-5, NEC-4.2
  and NEC-2D printouts says so: `(pragma ignored: <name>)` (the NEC-2
  printout has no line 2). Where the feature
  runs, line 2 names it as the variable would (`+pair_reuse`).

In EZNEC, type the line into the model's description: EZNEC copies the
description into the deck as `CM` cards, one card per line of text. Put the
pragma on a line of its own, at the start of the line. NEC itself reads the
card as a comment, so the same model runs unchanged in any other engine. In
4nec2 or SimNEC, add the card to the deck's comments.

The standalone executables read the card too, for `pair_reuse` only:
rotational symmetry is the drop-ins' alone, so there a
`CM momwire: rotational_symmetry` card is ignored and noted on line 2.

### Mixed-precision solve

`MOMWIRE_X_MIXED_LU=1` changes how the engine solves its matrix, not how it
fills it, on every dense engine: `sinusoidal` (`momwire-nec2c-sinusoidal`),
`bs2`, `bs1` and `sg`, and `razor2p` on the NEC-5 drop-in. The matrix is
factored in single precision, about twice as fast as the ordinary
double-precision factor, and the answer is then refined against the
double-precision matrix until it is as accurate as the ordinary solve. Each
refinement step must cut the error by a fixed factor; a matrix where it does
not (an ill-conditioned one), or one whose values single precision cannot
hold, is solved the ordinary way instead, with the ordinary answer to the
bit. It is **off by default**.

It pays only on large decks: below 1,024 unknowns, or with more right-hand
sides than the size makes worthwhile (3 at 1,024 unknowns, 64 at 4,392;
each port, and on `bs2` each junction constraint, is one), the ordinary
solve runs. On the W8IO ELT whip (4,392 segments, 13 frequencies) the
`sinusoidal` printout is identical to the stock one apart from its timing
lines, and the input impedance moves by at most 2.5e-13 relative, the same
size as the difference between two ordinary solves at different thread
counts. The whole run took 10.2 s instead of 12.2 s at 8 threads, and 46 s
instead of 63 s at one thread (an 8-core Intel i7-11700K, Linux). On
`razor2p` the same whip (6,372 unknowns, one frequency) took 2.6 s instead
of 3.0 s at 8 threads and 9.5 s instead of 13.3 s at one; over ground, where
the fill dominates, the gain is a few percent at most. A deck that falls
back pays for the single-precision factor it tried: 14 % slower on an
over-segmented 5,200-unknown pair of dipoles.

It costs memory: the matrix plus a half-size copy, where the ordinary solve
holds the matrix alone (737 MiB peak instead of 582 MiB on the
`sinusoidal` whip, 1,080 MiB instead of 806 MiB on `razor2p`).

It is certified for the hosted antennaknobs app only (Linux, avx2). On the
drop-ins it runs the same code, not yet certified there: the NEC-4.2 and
NEC-5 drop-ins name it on printout line 2 (`+mixed_lu uncertified on this
platform`), and the NEC-2 launcher's printout keeps nec2c's layout and does
not name it. The standalone executables refuse it.

## How many engines to run

A momwire process is not a 2 MB C binary: each carries NumPy, SciPy and
momwire — about 90 MB resident before it solves anything — plus the dense
complex matrix and its factors, which grow as the square of the segment
count. It is also much quicker per deck once warm, so a smaller crew keeps up
with a larger one of the C engine.

Without `--cache` that is the whole budget; a process keeps nothing between
decks. With it, each process's memory of past structures is capped at a few
hundred MB, and a full cache costs a re-solve rather than a failure.

**On a 16 GB machine, set the NEC crew size to 4.** Larger crews buy little —
the win here is the single fill per geometry, not parallelism across decks —
and they multiply the per-process floor by the segment count you are least
expecting.

## One resident engine: `momwire-nec2c-shared`

SimNEC does not keep one engine per session. It starts a crew, sends a burst
of decks, and destroys the processes — a live crew-16 session was measured
starting **51 engines**, each alive about a second. Every one of them paid the
full cold start (Python, NumPy, parse, mesh, first fill: ~650 ms measured
here) before it answered anything, and every one took its `--cache` with it
when it died, so the cache's main case — a value you retype a few minutes
later — could never hit.

No crew size fixes that. `momwire-nec2c-shared` changes the shape instead:

```bash
which momwire-nec2c-shared    # e.g. ~/.venvs/momwire/bin/momwire-nec2c-shared
```

Point the portal dialog at **that** path instead of `momwire-nec2c`, with the
same flags you were using. Nothing else changes — no preference, no new
setting, no restart discipline. What SimNEC now spawns is a few-hundred-line
forwarder that imports neither momwire nor NumPy; it connects to one
long-lived server holding the engine, and pumps your decks to it. On this
box:

| | stock `momwire-nec2c` | `momwire-nec2c-shared` |
|---|---|---|
| `-version` probe | 610 ms | **70 ms** |
| one cached deck, spawn to `NX` sentinel | 650 ms | **79 ms** |

The first client to arrive pays ~700 ms to start the server; every client
after it is the right-hand column. The printout is identical either way —
all 45 reference decks are compared byte for byte in the test suite,
refusals included.

### One server per engine, not per session

The server is chosen by a name hashed from the momwire version, the
interpreter, and the engine flags, under `$XDG_RUNTIME_DIR/momwire-portal/`
(or `/tmp/momwire-portal-<uid>/` where that is unset). That is the same *two
entries are two engines* rule the flags already carried: two portal-dialog
entries differing only in `--basis` get two servers, and a crew of sixteen
clients configured alike get **one**.

All three spellings of the basis count, not just the flag: a basis chosen by
the executable's name or by `MOMWIRE_NEC2C_BASIS` is resolved before the hash
is taken, so it picks out its own server exactly as `--basis` does. The
shared command reads its own name the way the stock one does, with its own
`shared` segment consumed rather than mistaken for a solver:

```bash
ln -s "$(which momwire-nec2c-shared)" ~/bin/momwire-nec2c-shared-bs1
momwire-nec2c-shared-bs1 -version    # a bs1 server, and only bs1 decks
```

`momwire-nec2c-bs1` works as a link name too — the spelling the named
commands above teach — and a bare `momwire-nec2c-shared` selects nothing, so
it stays the default engine. A name matching no solver fails the `-version`
probe loudly, naming its source, exactly as the stock command does. So the crew arithmetic above inverts —
one interpreter, one ~90 MB floor, one `--cache` warmed by every crew member
at once instead of sixteen cold ones, and one thread-pool budget, since
solves are serialised across connections.

The server exits when no client has been connected for 15 minutes
(`--idle-timeout SECONDS` on the same command line changes it) and removes
its socket on the way out. A server killed less politely leaves a socket file
behind; the next client notices it answers nothing, removes it, and starts a
fresh one, so there is nothing to clean up by hand. Its log sits next to the
socket — that is also where a deck's warnings go, since with several clients
on one process there is no honest stderr to return them on.

Upgrading momwire changes the hash, so the new version never gets served by
whatever is still resident from the old one. Servers from before are left to
their idle timeout.

:::caution[POSIX only, and opt-in]
The transport is a Unix-domain socket. Windows CPython gained `AF_UNIX` only
in 3.12 and none of this has been exercised there, so the client refuses on
one line and points you back at the stock engine. `momwire-nec2c` remains the
default and is unchanged — this is a second entry point beside it, not a new
version of it.
:::

Before pointing a session at it, run its own smoke — it starts a private
server, sends decks through two separate client invocations, and checks that
the second was answered by the process the first left behind:

```bash
momwire-nec2c-shared --selftest
```

## Licensing

SimNEC is proprietary freeware. This engine reproduces the printout *layout*
SimNEC's reader expects, worked out from observed output, and contains no
`nec2c` code — interoperability of the same kind as emitting a NEC deck or a
Touchstone file.
