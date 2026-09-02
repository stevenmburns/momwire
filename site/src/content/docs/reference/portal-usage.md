---
title: "Running momwire as SimNEC's engine"
description: Install the momwire-nec2c portal, point SimNEC at it, pick an engine by name (one command per solver since 0.36.1), read the version probe and the refusals, and swap in the shared resident engine.
---

[SimNEC](https://ae6ty.com/smith_charts/) (AE6TY) does not link its NEC
solver — it shells out to a `nec2c` executable, starts one copy, and keeps
it. Decks go down that process's stdin framed by an `NX` card, printouts come
back on stdout, and SimNEC's own circuit solver reads two numbers per
feedpoint out of each printout to build the antenna's Y matrix.

momwire ships that process. `pip install momwire` puts **`momwire-nec2c`** on
your path: a drop-in for the engine binary, with momwire's solver behind it.
The Smith chart, the tuner and the sweeps stay SimNEC's; the
electromagnetics become momwire's.

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
momwire-nec2c -version     # NEC2momwire.<major>.<minor>
```

`python -m momwire.portal` is the same program under its long name, for an
environment where the console script is awkward to reach.

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

Since momwire 0.36.1 every solver is its own command — `pip install momwire`
puts all of these on your path, and SimNEC's engine dialog takes any of them
directly, no arguments and no wrapper scripts:

```text
momwire-nec2c                                # the default — degree-2 B-spline (bs2)
momwire-nec2c-bspline-d1                     # the same physics at degree 1 (tent basis)
momwire-nec2c-sinusoidal                     # NEC-2's own formulation
momwire-nec2c-sinusoidal-galerkin            # the same basis, tested variationally
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

Four axes live in that list. `sinusoidal` and `sinusoidal-galerkin` are
the **fidelity** ladder: the three-term basis point-matched with NEC's own
delta-gap source answers "does this reproduce NEC-2's behaviour, mesh walk
and all", and the Galerkin entry and the B-spline default answer "what
does a better-converged basis say". There was a second Galerkin name until
momwire#654: it differed only in its feed model, which is not a choice a
portal dialog can put to you meaningfully, and the better of the two is now
the plain name's default. There is no `sinusoidal-converged`: a
zero-width gap has no collocation right-hand side, so the flag does not offer
a name the solver refuses. `bspline-d1` is the **degree** axis — the same
B-spline physics at degree 1, so pairing it with the default is the cheapest
convergence check available. `hmatrix` and `arrayblock` are the **solve**
axis, not a physics choice: the same B-spline operator held compressed and
solved iteratively, so a large array answers without a dense fill.
`arrayblock` is the one to reach for on a repeated-element array — identical
elements on a regular lattice become an FFT convolution over the element grid
— and it degrades to the hierarchical solve on a deck with no repeated
structure, so it is safe to leave on. Its answers must agree with `bspline`'s;
if they do not, the deck is telling you about conditioning, not about basis.
**There is no razor command here.** The tent-basis, razor-blade formulation
NEC-5 identifies with (`razor-2p`) is not a nec2 engine: a NEC-2 deck puts
its source at a segment *centre* and razor places its gap at a *knot*, so
it would answer for an antenna half a segment from the one you drew. Since
momwire#821 the portal refuses the name at configure time rather than serve
that — `momwire-nec2c-razor-nec5` and `momwire-nec2c-razor-2p`, which
shipped from 0.36.1 to 0.46.0, are gone, and a copy you kept fails the
version probe with the reason. Razor is served where its grid is the deck's:
the EZNEC drop-in (`momwire-eznec-razor-2p`), whose dialect writes a source
at a node.

**On a tapered or stepped-radius wire, `sinusoidal` and
`sinusoidal-galerkin` are NEC-2-identified, not NEC-5-accurate.** Both ride
NEC-2's per-segment end-condition convention, so on a radius step they
reproduce NEC-2's own stepped-radius error rather than resolving it —
measured converging onto nec2c's answer to 0.01-0.1 Ω while sitting tens of
ohms from the licensed NEC-5 reference on the same geometry
(momwire#398/#435). That is by design: this row exists to be the
NEC-2-closest rung of the ladder, and inheriting NEC-2's own defect on a
taper is what "closest" means there. Reach for `bspline-d1` with `EK` armed
(the `EK` card, above) for a tapered or fat-conductor deck instead — it is
the row measured closest to NEC-5 on Ward Harriman AE6TY's flagship 20:1
tapered dipole. `sinusoidal-galerkin` additionally REFUSES `EK` outright on
any deck where two wires of different radii meet at a junction — measured
divergent rather than merely inaccurate there; see the `EK` card entry in
[the nec2 deck grammar
reference](/reference/deck-grammar-nec2/#ek--extended-thin-wire-kernel).

**Two portal entries differing only in `--basis` give you cross-basis
validation inside SimNEC itself.** Switch engines from the dialog and watch
whether the answer holds. The printout banner records which physics answered
(`VERSION:nec2c.ae6ty.momwire.9.1+sg`), and a mistyped basis fails the
version probe loudly at configure time rather than serving the default.

### When the dialog is a file picker

It always is — and the named commands above exist precisely so that the file
*is* the choice. Point the dialog at `momwire-nec2c-sinusoidal-galerkin` and
that is the engine; two entries differing only in name are two engines. Both
filename rules above apply to every spelling, shipped or hand-made: keep
`nec2c` in the name, and keep `out` out of the whole path.

## The version probe

SimNEC runs `<command> -version` and reads the first line. The portal answers

```text
NEC2momwire.<major>.<minor>
```

which is SimNEC's own engine-family form: the `NEC2` prefix declares the deck
dialect, and the text after it is free. SimNEC shows the string in the portal
dialog's `NECVersion` row, stamps it as a `CM version` comment card on every
deck it sends, and stores it in saved circuits — so a momwire session is
identifiable as one everywhere the version travels.

**The number is momwire's own version** — the engine's, which is the number
that changes solve results. An engine release moves it; nothing else does.

For a SimNEC build old enough to predate that form, `--legacy-probe` on the
portal command line restores the older `nec2c.ae6ty.9.1` shape. Nothing else
about the session changes either way. The probe line never carries the
`--basis` suffix; only the printout banner does.

## Running a deck from the command line

The portal reads a deck file directly, the way you would test any NEC engine:

```bash
momwire-nec2c --basis sinusoidal < dipole.nec > dipole.printout
```

One framing rule makes this work: a deck is solved when its terminator card
arrives. Inside SimNEC that card is `NX`, appended automatically. Standalone,
a stock `.nec` file's final `EN` card does the same job and then ends the run.
End of input terminates a deck too, exactly as `EN` does — so a file with
neither, a deck that stops at `XQ`, solves and exits the same way.

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
  grids, near fields over finite ground, `GN` radial ground screens, the
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
ln -s "$(which momwire-nec2c-shared)" ~/bin/momwire-nec2c-shared-bspline-d1
momwire-nec2c-shared-bspline-d1 -version    # a bspline-d1 server, and only bspline-d1 decks
```

`momwire-nec2c-bspline-d1` works as a link name too — the spelling the named
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
