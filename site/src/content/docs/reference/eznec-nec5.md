---
title: "Running momwire as EZNEC's NEC-5 engine"
description: The momwire.dropin.nec5 drop-in engine serves EZNEC Pro+'s external-engine slot in the NEC-5 dialect — decks in, byte-compatible printouts out, and every deck either serves or refuses by name.
---

EZNEC Pro+ can drive an external NEC-5 console engine: it writes a deck
(`EZN5.NEC`), launches the engine with two positional arguments, and reads
the printout (`NEC5.OUT`) back for every display it draws. momwire ships
that process.

## Setting it up

1. Download
   [`momwire-windows.zip`](https://github.com/stevenmburns/momwire/releases/latest/download/momwire-windows.zip)
   from the latest release.
2. Before extracting, right-click the zip, choose Properties, tick
   **Unblock** and click OK. That clears the download mark on every file
   inside at once.
3. Extract it into **a folder of its own**, for example
   `C:\momwire\` (the zip already contains a `momwire` folder).
   Not into EZNEC's own folders, and not over a previous release.
   **Keep the folder together**: the exe needs the `_internal` runtime
   beside it, and a lone copied-out `.exe` is the one way a correct
   download still fails.
4. Point EZNEC's external-engine path at `momwire-nec5.exe` inside that
   folder. (For EZNEC's External NEC-4.2 engine option, point it at
   `momwire-nec42.exe` instead — see [below](#choosing-the-formulation).)

That is the whole installation. No Python, no environment, nothing on PATH.
EZNEC's interface, models and displays stay EZNEC's; the electromagnetics
become momwire's.

### In 4nec2

Steps 1–3 are the same: the unblocked zip extracted to a folder of its
own, `C:\momwire\` below. Tested with 4nec2 5.8.8.

4nec2 runs every engine through `4nec2.bat` in its `exe` folder
(`C:\4nec2\exe` in a default install): for each run it calls the batch
file with the engine's name, and the batch file runs that engine on
`nec2d.tmp`. **Settings → Nec engine** only chooses which name 4nec2
passes in; **Manual select** sets that name and the segment limit, and
does not name an outside program.

So momwire goes in by editing `4nec2.bat`. Because that file serves every
engine choice, do it in a **copy of the 4nec2 folder** (for example
`C:\4nec2MW\`), and keep your original 4nec2 for its own engines.

**NEC-4.2.** In the copy's `exe\4nec2.bat`, replace the engine line,
`%nec2d% < nec2d.tmp`, with one of

```text
C:\momwire\momwire-nec42.exe < nec2d.tmp              the default (nec42-bs2)
C:\momwire\momwire-nec42-sinusoidal.exe < nec2d.tmp   NEC-4.2's own formulation (the twin)
```

and leave the rest of the file as it is. In **Settings → Nec engine**,
choose 4nec2's NEC-4 engine (`NEC4D*.exe`; with **Manual select**, the
name `NEC42W32CL.exe` and a segment limit of 9999, as Dan AC6LA's NEC-5
setup does; 4nec2 only passes that name to the batch file, which ignores
it), and leave **Wait in DOS box** unticked: 4nec2 then tells the batch
file not to pause, and no key press is needed.

**NEC-2 is not supported in 4nec2.** 4nec2's NEC-2 engines read a NEC-2
deck and write the NEC-2D printout layout, which momwire does not write.
Don't point a NEC-2 engine choice at a NEC-4.2 launcher: it would read the
NEC-2 deck under NEC-4.2's rules. (momwire's `momwire-nec2c` launchers are
SimNEC's NEC-2 engine; 4nec2 does not read their printout.)

4nec2 reuses a model's last printout, without running the engine, when the
model has not changed since. After switching engines, edit the model or
delete its `.out` file in 4nec2's `out` folder to get a fresh run.

**NEC-5.** This goes through the NEC-5 package for 4nec2 that Dan, AC6LA,
describes in his QRZ post
[Using NEC-5 with 4nec2](https://forums.qrz.com/index.php?threads/using-nec-5-with-4nec2.817589/#post-7249839),
which sets 4nec2 up in a copy of its folder with `Edit_inp.exe`,
`Edit_out.exe` and its own `4nec2.bat`. Install it as his post says. Then,
in that copy's `4nec2.bat`, keep the `Edit_inp.exe` and `Edit_out.exe`
lines, since 4nec2 reads its results through them, and change only the
engine line between them:

```text
Edit_inp.exe
C:\momwire\momwire-nec5.exe < nec2d.tmp
Edit_out.exe
```

with `momwire-nec5-razor2p.exe` for the NEC-5 twin, or the standalone
program's `momwire-nec5-bs2.exe` from its own folder. NEC-5 puts a source
at a junction between segments, so give a centre-fed wire an even number
of segments.

For a new release, replace `C:\momwire\` as in
[Updating](#updating-to-a-new-release); the 4nec2 settings and the
`4nec2.bat` lines stay as they are. The same steps are in the zip's
`README.txt`.

### Names changed in 0.80.0

From 0.97.0 the basis names carry no dashes (momwire#2091): the zip's
`momwire-nec5-razor-2p.exe` is now `momwire-nec5-razor2p.exe`, and the pip
commands `momwire-nec5-bspline`, `-bspline-d1` and `-sinusoidal-galerkin`
(and their `nec42` and `nec2c` twins) are now `-bs2`, `-bs1` and `-sg`. A
copy of a launcher under its old name still answers, as the new name does,
and says so on line 2 of the printout; change the engine path before a later
release removes the old names.

Earlier releases used names that said "eznec". The same zip serves SimNEC
and 4nec2 too, so 0.80.0 dropped it. If an engine path points at an old
name, change it to the new one:

| Before 0.80.0 | From 0.80.0 |
| --- | --- |
| `momwire-eznec-windows.zip` | `momwire-windows.zip` |
| `momwire-eznec-windows-win7.zip` | `momwire-windows-win7.zip` |
| folder `momwire-eznec\` | folder `momwire\` |
| `momwire-eznec-engine.exe` | `momwire-engine.exe` |
| `momwire-eznec.exe` (removed) | `momwire-nec5.exe` |
| `momwire-eznec-razor-2p.exe` (removed) | `momwire-nec5-razor2p.exe` |
| `momwire-eznec-razor-nec5.exe` (removed) | `momwire-nec5-razor2p.exe` |

The `momwire-nec5` and `momwire-nec42` launcher names are unchanged.

### Updating to a new release

Delete the `momwire` folder entirely, then extract the new zip in its
place. The path stays the same, so EZNEC's setting does not change.

Never extract a new release over an old one. Extracting replaces the files
that share a name and leaves the rest, so the old release's files stay
behind in `_internal`. That includes its version record, and it can include
libraries.

- If Windows says a file is in use while you delete the folder, that is the
  resident engine described below. Wait fifteen minutes for it to retire, or
  end `momwire-engine.exe` in Task Manager.
- Line 2 of every `NEC5.OUT` names the engine that answered, for example
  ` momwire 0.69.0 bs2 avx2`. If it ends in `(stale: 0.66.0)`, the
  folder holds files from an older release: delete it and extract the
  current zip fresh.

### Launch economics

EZNEC launches its external engine once per frequency point, so the launch
cost *is* the sweep cost. What you point EZNEC at is a small native
**launcher** that keeps a warm momwire engine resident — started on the
first calculation, retired after fifteen idle minutes — so every launch
after the first costs milliseconds instead of a Python start-up. Measured
**inside real EZNEC**, driven by hand on one machine — a 51-point SWR sweep
of a back-yard dipole over real ground, every engine invocation timestamped:

| engine | 51-point SWR sweep | per point |
| --- | --- | --- |
| momwire launcher, warm | **3.0 s** | 58 ms |
| licensed NEC-5 console engine | 11.0 s | 215 ms |

3.7× on the sweep a user actually runs. Engine-side, the two are at parity —
22 vs 23 ms per launch when invoked the way EZNEC invokes them — and about
35 ms of every point is EZNEC's own per-point work, the same for either
engine. The resident engine wins the sweep because it is bounded where a
process-per-point engine is erratic (13–500 ms). For scale, the previous
packaging (a frozen one-shot paying the Python import on every launch) took
103 s on the same sweep, engine-side.

The first calculation after a cold start, or after the idle retirement, pays
the engine's start-up once (~5 s on that machine, with no progress shown —
EZNEC displays nothing during an external-engine run); if anything about the
resident path ever fails, the launcher runs the engine directly instead —
the same answer, at one-shot speed, never a broken engine.

### Threads

With no thread variable set, the engine sizes its threads by the deck, by the
rule the [standalone executables](/reference/standalone/#threads) use: one
thread for a deck under 128 segments, four up to 384, and one per physical
core above that. A buried deck whose fill is heavy for its size gets every
core whatever its segment count. The count never exceeds the physical cores
the process's affinity mask allows. OpenBLAS runs on the same count, except
that a one-thread deck keeps two OpenBLAS threads, because OpenBLAS's serial
path rounds differently in the last digits.

Each pool takes the first of these that is set:

| Pool | 1st | 2nd | 3rd | Otherwise |
| --- | --- | --- | --- | --- |
| OpenMP (the fill) | `OMP_NUM_THREADS` | `MOMWIRE_THREADS` | | by the deck |
| OpenBLAS (the factor) | `OPENBLAS_NUM_THREADS` | `MOMWIRE_THREADS` | `GOTO_NUM_THREADS`, then `OMP_NUM_THREADS` | by the deck |

The standalone executables differ in one way: there `MOMWIRE_THREADS` wins
over `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS`.

When several calculations reach a resident engine at once, its workers divide
the physical cores between them, and that share caps every count above. Under
`MOMWIRE_TIMES=1` the printout and the `times:` line report the counts the
answer ran on.

### The 4nec2 geometry file (`.GMO`)

Under 4nec2's NEC-5 package (the install steps are in the zip's
`README.txt`), a launcher writes the `.GMO` geometry file beside the
printout when the deck's `GE` card asks for it (`GE 0 0 1`, or any nonzero
third field). That drawing joins two wire ends only where the solver joins
them — ends within 1 micrometre. Ward AE6TY's port draws ends up to about
0.1 mm apart as joined; momwire draws them as the two separate ends the solve
treats them as. This is deliberate: the drawing shows the structure that was
solved.

## Choosing the formulation

The bundle carries six launchers and the one engine they run, and the
choice is the engine PATH you set. One family per EZNEC engine option, and
one for NEC-2:

```text
EZNEC's External NEC-5 slot, and SimNEC's NEC-5 engine:
momwire-nec5.exe              the default — degree-2 B-spline (nec5-bs2)
momwire-nec5-razor2p.exe     the tent basis with razor-blade path testing,
                              at NEC-5's two-point rule

EZNEC's External NEC-4.2 slot:
momwire-nec42.exe             the default — degree-2 B-spline (nec42-bs2)
momwire-nec42-sinusoidal.exe  the sinusoidal basis, point matched

SimNEC's NEC-2 engine, and nec2c's command line:
momwire-nec2c.exe             the default — degree-2 B-spline (nec2c-bs2)
momwire-nec2c-sinusoidal.exe  the sinusoidal basis, point matched

momwire-engine.exe            the compute engine the launchers run
```

The `momwire-nec2c` names are the [NEC-2 launchers](/reference/portal-usage/)
that `pip install momwire` installs under the same names, frozen: they speak
NEC-2's protocol (decks on standard input, printouts on standard output, the
`-version` probe), not EZNEC's, and EZNEC has no NEC-2 engine slot to put
them in. A name carrying `nec2c` is read before every other marker, so a
copy named for NEC-2 is a NEC-2 engine. They run the engine once per run
rather than keeping it resident.

The name picks the deck dialect as well as the basis, so a `momwire-nec42`
name belongs in EZNEC's NEC-4.2 slot and a `momwire-nec5` name in its NEC-5
slot. In the NEC-4.2 slot, `sinusoidal` is the basis family NEC-4.2 itself
belongs to; razor cannot serve there, because NEC-4.2 decks put their sources
at segment centres and razor places sources at knots, so a
`momwire-nec42-razor2p` copy refuses by name.

In the NEC-4.2 slot, momwire reads the cards EZNEC's NEC-4.2 writer emits:
- **Current sources** (`EX 6`), including several at once, so a phased array
  such as a cardioid or a four-square solves with its currents pinned.
  Current sources can also mix with voltage sources (`EX 0`).
- **Ground:**
  - perfect ground (`GN 1`);
  - a `GD` second medium for the far-field cliff;
  - the Sommerfeld ground (`GN 2` or `GN 3`);
  - wires below the surface (`GE -1`).
- **The Sommerfeld-table file name** that NEC-4.2 may write at the end of a
  `GN` card. NEC-4.2 uses that file only as a cache, so momwire reads the
  name and ignores it, and computes the ground from the card's own values.
- **Several result cards in one run.** Every `RP`, `NE` and `NH` card after
  the excitation prints its own table, in card order, so a deck can ask for a
  pattern and a near field together.
- **`RP` with a range**, the form SimNEC writes (`RP 0 19 73 1001 0 0 5 5
  1000`): the pattern's fields are printed in volts per metre at that range,
  as NEC-4.2 prints them. The gains are the same with or without a range.
- **`RP` gain forms** XNDA 1000 to 1003 and 1010 to 1013 (4nec2 writes 1003
  by default), and the NORMALIZED GAIN table an XNDA digit N of 1 to 5 asks
  for, as in `RP … 1500`: major axis, minor axis, vertical, horizontal or
  total gain, normalized to the table's peak or to the card's own factor,
  laid out as NEC-4.2 prints it. A pattern of one theta or one phi prints no
  average gain, as in NEC-4.2.
- **`PT -1`**, which 4nec2's ItsHF decks write beside `PQ -1`: the printout
  leaves out the CURRENTS AND LOCATION table from that card on, as NEC-4.2
  does, and prints everything else. Every other `PT` form is refused.
- **Geometry beyond `GW`:** `GM`, `GS`, the reflection `GX`, the wire arc
  `GA`, and the helix `GH` as NEC-4.2 reads it. NEC-4.2's `GH` is not NEC-2's:
  its real fields are the number of turns, the length along Z, the helix
  radius at each end and the wire radius at each end. A deck that writes
  NEC-2's `GH` fields (turn spacing, length, four radii, wire radius) means
  something else to NEC-4.2 and to momwire; the `momwire-nec2c` launchers
  read NEC-2's `GH`. A `GX` numbers its copies and their segments as NEC-4.2
  does, and the printout carries NEC-4.2's symmetric-cell lines. NEC-4.2
  stops on a `GA` of 360 degrees or more, and so does this slot. `GR`, `GC`
  and `GF` are refused.

`EX 4` is refused in the NEC-4.2 slot. NEC-4.2 reads it as a current element
at a point in space, not as NEC-5's current source on a segment. For a
current source on a segment, write `EX 6` with the same tag and segment.

A wire listed twice (two wire cards whose ends coincide, in either
direction) is refused by `momwire-nec42-sinusoidal`, `momwire-nec42-bs2`
and `momwire-nec42-bs1`, naming the `GW` tags of
each set of copies; NEC-4.2 stops on such a deck too. Delete the copy. A
`GM` that moves a wire without copying it is not a duplicate and serves.

Each source's impedance agrees with the licensed NEC-4.2 to within about 2 %.

**Both slots solve with the extended thin-wire kernel** (since momwire#1326).
NEC-5's own kernel behaves as the extended one, and NEC-4.2 has a single
thin-wire model with no kernel switch: an `EK` card in a NEC-4.2 deck is read,
echoed, and ignored, and the printout says so in NEC-4.2's own words, "THE EK
AND KH COMMANDS HAVE NO EFFECT IN NEC-4". On fat wire this lands closer to both
engines: on a 28 MHz dipole whose segments are 1.5 radii long it is 4 to 5 Ω
nearer NEC-4.2. On thin wire it moves the answer by a fraction of an ohm. A few
decks cannot take the extended kernel: a wire below the ground on
`sinusoidal`, or two wires of different radii meeting at a junction on
`sg`. Those decks are solved with the reduced kernel, and the
engine logs an `ExtendedKernelDefault` advisory instead of refusing. On
`bs2`, `razor2p` (since momwire#1614) and `sg` (since
momwire#1626), a deck with wires below the ground keeps the extended kernel on
its above-ground wires; the buried wires use the reduced kernel, and the
advisory says so.

**The `momwire-eznec` launchers are gone** (0.80.0). They were what earlier
releases said to point EZNEC at; point that path at the `momwire-nec5` name
in the table [above](#names-changed-in-0800) instead, which answers
identically and also works in SimNEC.

Point EZNEC at a **launcher**, never at `momwire-engine.exe` — the
engine answers correctly either way, but naming it directly gives up the
warm start above for nothing.

They accept the same models — measured, deck by deck, on the whole captured
corpus — and answer in different formulations. `razor2p` is the tent basis
with razor-blade path testing at NEC-5's two-point rule — the formulation
NEC-5 itself uses.

`razor-2p` and `razor-nec5` are **deprecated spellings** of the same basis
(momwire#2091, #785/#794). A launcher copied under either name runs the same
engine and answers identically, and line 2 of its printout ends in
`(renamed from <name>)`; they will stop working in a later release. Use
`momwire-nec5-razor2p.exe`.

### Reproduction is not accuracy

`razor2p` agrees with the licensed engine **because it runs the same
algorithm**, not because it is more correct. It inherits that engine's
discretization error along with its answers, and NEC-5's razor-blade testing
rule is known to walk its impedance slowly — O(1/N).

**A note on segment counts, because it is the opposite of the NEC-2 habit.**
NEC-5's basis is the tent, so its unknowns *and its sources* live at knots,
not at segment centres. An odd segment count leaves no knot at a dipole's
centre and therefore cannot feed it there — the source lands half a segment
off. Use **even** counts for a centre-fed dipole in this dialect; "odd
segments" is a NEC-2 convention, where sources sit at segment centres.

Measured on a 0.476 λ dipole in free space, even meshes, one deck per rung —
momwire through this seam, the licensed engine on the same deck text:

| segments | licensed NEC-5 | nec5-bs2 | nec5-razor2p | \|razor − NEC-5\| |
|---|---|---|---|---|
| 4 | 56.118 − 108.593j | 67.645 − 31.146j | 56.116 − 108.586j | **0.007** |
| 20 | 66.667 − 35.880j | 67.739 − 29.155j | 66.665 − 35.877j | **0.003** |
| 60 | 67.469 − 30.695j | 67.777 − 28.586j | 67.467 − 30.693j | **0.003** |
| 160 | 67.670 − 29.281j | 67.796 − 28.340j | 67.668 − 29.280j | **0.003** |

The twin tracks the reference to 0.003 Ω from 20 segments up and 0.007 Ω at
the coarsest rung — flat, not improving, which is what a twin looks like.

That says nothing about which is nearer the truth, so the second measurement
asks each basis about **itself**: the same decks through the same seam, each
basis scored against **its own N = 160 answer** — the finest rung of the
table above, so every number here can be read off it.

| segments | nec5-bs2 error | nec5-razor2p error |
|---|---|---|
| 4 | 2.81 Ω | 80.14 Ω |
| 20 | 0.82 Ω | 6.67 Ω |
| 60 | 0.25 Ω | 1.43 Ω |

Both converge. At matched mesh the B-spline basis is 5.8–28× nearer its own
limit, which is the O(1/N) walk of razor-blade testing priced in segments.

Neither is "converged at coarse mesh" — nec5-bs2 is still 2.8 Ω out at four
segments. The difference is how fast the error comes down, not whether it is
there.

**Where the last ohm actually lives** (measured 2026-08-26; `parity_limits()`
in the probe script is the receipt). Extrapolate both ladders and the two
formulations nearly meet: nec5-razor2p's limit lands 0.08–0.21 Ω from nec5-bs2's,
depending on the extrapolation model. Of the roughly one ohm between the
bases at N = 160 (0.95 Ω through this seam, 1.05 Ω on the probe's direct
ladder), almost all is the twin still descending the O(1/N) path it shares
with the licensed engine, and only the fraction-of-an-ohm remainder is
formulation. The same run settles a tooling question: with the feed
pinned at the exact centre, nec5-bs2's limit is the same whether that centre is a
knot (even N) or the middle of a span (odd N) — 0.009 Ω apart — so the
knot-feed machinery is not what separates the bases. One deck, free space;
but on it, the second table sharpens: the default is not merely nearer *its
own* limit, it is nearer the limit the two formulations share.

So: pick the **twin** when you want what NEC-5 *would have said* — checking a
published NEC-5 number, or matching a NEC-5 workflow. Pick the **default**
when you want momwire's own best answer. When they disagree at a practical
mesh, neither is broken: most of the gap is the twin's inherited
discretization, shrinking as O(1/N), and what remains — a fraction of an ohm
on the deck above — is formulation. A disagreement is information about the
mesh before it is information about either engine.

**Making another.** The basis rides on the *filename*: everything after
`nec5-` (or `nec42-`) selects it, a Windows `.exe` stripped first. So a copy
you make yourself works — copy a **launcher** (a couple of hundred kilobytes,
not the engine), rename the copy to `momwire-nec5-<basis>.exe` (or
`momwire-nec42-<basis>.exe`, or `momwire-nec2c-<basis>.exe` for NEC-2) in the
same folder, and that basis answers in that slot. This is the same rule the [NEC-2 launcher's](/reference/portal-usage/)
`momwire-nec2c-<basis>` commands use, with one owner behind both. The bundle
ships two per slot because that is what the parity work was about; the rest
are a copy away:

```text
NEC-5 slot:    bs2  bs1  hmatrix  arrayblock  razor2p
               sg
NEC-4.2 slot:  bs2  bs1  hmatrix  arrayblock  sinusoidal
               sg  pulse
```

The `momwire-nec5` names are also what
[SimNEC's NEC-5 engine setting](/reference/portal-usage/#simnecs-nec-5-engine)
needs, since SimNEC reads `nec5` in the path.

The NEC-4.2 launchers carry `nec42` for the same reason: SimNEC recognises an
engine by `nec42` in its name, as it does `nec5`, so `momwire-nec42.exe` is
what its NEC-4.2 engine setting takes, in each of SimNEC's processing modes:
single, batch (several runs in one deck, `NX` between them) and stream (`- -`)
since momwire#1592. (Releases before momwire#1443 shipped
them as `momwire-nec4[-<basis>].exe`.) A copy named with just `nec4` — no
`2` — still answers EZNEC's NEC-4.2 slot, its basis read past `nec4-`.

`sinusoidal` cannot answer the NEC-5 dialect — every deck in it drives a NODE, and
under point matching the match points are the segment centres, so a delta at a
node point-samples to nothing in every row and there is no excitation left to
solve. It says so by name in the printout rather than answering about a
different antenna. `sg` has no such trouble: its test
integral collapses the same delta to a well-defined drive, and it serves the
corpus alongside the B-spline and razor families. A filename matching
no basis does the same: it refuses, names itself and lists what exists, so a
typo can never be served as the default. The match is case-insensitive, as
Windows filenames are, so `Momwire-NEC5-Razor-2p.exe` is the twin too.

### Refining the mesh at a feed on a fat wire

A deck's source is a point gap: zero width, at a knot. A zero-width gap has no
limiting capacitance, so once the segments at the feed are shorter than the
wire's radius, each refinement adds a little more, and X keeps sliding
negative instead of settling. Every engine does this, the licensed NEC-5
included. They differ only in how fine a mesh it takes (momwire#1675). For a 38 mm
vertical over perfect ground at 14 MHz, base-fed, X in ohms:

| segs/λ | Δ/a at the feed | licensed NEC-5 | nec5-razor2p | nec5-bs2 |
|---|---|---|---|---|
| 200 | 2.8 | −0.52 | −0.53 | +0.01 |
| 400 | 1.4 | −0.08 | −0.09 | −0.05 |
| 800 | 0.70 | +0.03 | −0.06 | −0.25 |
| 1600 | 0.35 | −0.01 | −0.17 | −0.31 |
| 3200 | 0.18 | −0.14 | −0.26 | −0.34 |

`nec5-bs2` gets near its converged value at a far coarser mesh than the other two,
then starts sliding first, at about Δ/a = 1. On a convergence chart that looks
like a curve that never settles. It is the feed model, not the antenna, and on a
fat wire the tail is a few tenths of an ohm. Keep the segments at a feed no
shorter than the wire's radius. Below that, the engine raises a `GapMeshFloor`
advisory naming the source. Only a source of fixed width converges past it,
and no deck card asks for one. Through momwire's own API, nec5-bs2's end-port
feed does: two facing wire ends a real gap apart, under the exact kernel.

:::caution[Not a supported configuration]
No part of EZNEC knows momwire exists, and nothing here has been reviewed or
endorsed by EZNEC's author. The drop-in engine was built black-box, from the decks
and printouts of our own licensed EZNEC and NEC-5 installation and the
engine's public documentation — no NEC-5 source was consulted. Treat it as a
cross-check you can run yourself.
:::

## The contract: serve, or refuse by name

Every deck takes exactly one of two paths:

- **Serve**: the printout comes back in the engine's own layout — the same
  tables, the same headings, the same numeric formats, byte-compatible with
  the captured output of the licensed engine on our fixture corpus — so
  everything EZNEC parses out of it just works.
- **Refuse by name**: a deck asking for something momwire does not serve
  gets a `NEC ERROR` line in the printout that names the card, the wire, or
  the capability — and, where there is one, the remedy. Nothing is served
  silently wrong; a refusal sentence is the contract that it never will be.

As of 2026-08-26, 77 of the 80 EZNEC captures in our corpus serve; the three
refusals are one named sentence, about one observation point.

### The printout says which engine answered

**Line 2 of every printout — served and refused alike — names the engine.**
For example:

```
 momwire 0.55.0 bs2 avx2
```

Three facts after the name: the release, the formulation that answered (the
basis your launcher's filename selected), and which compiled accelerator
loaded — `avx2`, `sse2`, `legacy`, or `none` for the pure-Python fallback.

**Quote that line in any bug report.** The release and the formulation are
what move the numbers underneath it, and nothing else in the file says which
of each you had. The accelerator moves the speed, which is the first question
of every slowness report: `none` there means the solve was correct and many
times slower than it should have been.

Nothing else in the header changes. Line 2 is where the licensed engine prints
its own build tag, so it is the one line that already carried an engine
identity; the comment block EZNEC checks the printout against, the banner and
every table heading are untouched.

### Advisories: on stderr, when you ask

Besides refusals, momwire raises **advisories**: a warning about the mesh,
the kernel or a feed that does not stop the solve, such as `DuplicateWire`,
`CoarseCrossingNode` or `GapMeshFloor`. They never go in the printout, so it
stays in the engine's own layout. By default the engine writes none of them.
To see them, set `MOMWIRE_ADVISORIES` before your program starts:

| value | what the launcher writes on stderr |
|---|---|
| `off`, or not set | nothing (the default) |
| `deck` | each different advisory once per deck |
| `all` | every one, including a deck's repeats of the same advisory |

`stderr`, `on`, `1`, `yes` and `true` mean `deck`. Case does not matter, and a
value the engine does not recognise reads as `off`. Two advisories are
different when their text is: under `deck`, a deck that raises the same
advisory at each of its frequencies shows it once, and `all` shows each.

Every route honours the setting: a one-shot run, the stdin form, SimNEC's
stream mode, and a deck the warm engine answers, whose advisories travel back
to the launcher that sent it. Before the first advisory of each deck the
launcher writes a line naming the deck, so you can tell which deck raised it:

```
momwire: advisories for C:\EZNEC\Models\dipole.ez.nec
momwire: advisories for structure 2 of the stream
```

A deck that writes no advisory gets no such line.

One engine process writes at most 3072 bytes of advisories, then one line
saying the rest were not shown, then nothing more. So a host that never
reads the engine's stderr cannot stall it on a full pipe (momwire#1673): the
smallest pipe a host is known to hand over holds about 4 KB.

EZNEC does not show stderr, so under EZNEC no setting changes anything you
can see. Writing advisories to that unseen console cost each EZNEC launch
about 15-21 ms on Windows, which is why they are off unless asked for
(momwire#2119).

## What serves

- **Geometry**: `GW` wires — straight, junctioned, tapered via stepped
  radii, exactly as EZNEC emits them (EZNEC resolves its own transforms
  before writing the deck). Since momwire#1669, also the geometry cards
  4nec2's NEC-5 package writes, each read as the licensed NEC-5 reads it
  (measured as a black box):
  - `GA`, the wire arc, and `GH`, the helix, each as **one wire** of its
    segment count, so `EX` and `LD` address its nodes as they address a
    `GW`'s. `GH` is read as NEC-4.2 and NEC-5 read it: turns, the length
    along Z, the helix radius at each end and the wire radius at each end.
    A `GH` whose two wire radii differ is refused, and so is a `GA` or `GH`
    that reaches below a ground plane (write that part as `GW` cards). A
    `GA` of 360 degrees closes on itself.
  - `GM` moves or copies the wires from the tag `ITS` names to the last.
    With `NRPT` 0 it moves them in place and still adds `ITGI` to their
    tags, as NEC-5 does.
  - `GX` reflects in the planes its code names and numbers the copies as
    NEC-2 does, and the printout carries NEC-5's `Reflection symmetry`
    block. A reflection in Z = 0 over a ground is refused.
  - `GS` scales every coordinate and radius. NEC-5 reads no tag range on
    it, so neither does momwire.
  - A wire end within 1e-4 of an element length of an earlier wire's end
    joins it, as NEC-5 joins them, so a hand-typed `GW` meets an arc's end.

  On these cards razor2p matches the licensed engine to 0.01 % of the
  feed impedance (29 probe decks).
- **Grounds**: free space (`GN -1`), perfect ground (`GN 1`), the
  Sommerfeld finite ground (`GN 0` / `GN 2`) including ground contact, and
  EZNEC's MININEC-type ground (bare `GD`).
- **Excitation**: `EX 0` voltage sources and `EX 4` current sources,
  including phased multi-source drives; sources address nodes the way NEC-5
  does.
- **Loads and networks**: `LD 4` impedance loads, `LD 5` wire conductivity,
  `LD 2` / `LD 3` per-unit-length RLC (*added 2026-09-16, momwire#1088* —
  the shape antennaknobs' own NEC-5 writer emits for a jacketed wire, an
  `LD 2` inductance per metre beside an `LD 5`; the cards' capacitance
  field refuses, see the [nec2 grammar
  page](/reference/deck-grammar-nec2/#ld--loading) for the measurement),
  insulation (`IS`), transmission lines and non-radiating networks
  (`TL` / `NT`), including the mixed table layouts.
- **Multi-run decks** (*added in 0.66, momwire#1237*): a deck with several
  `EX … RP`/`XQ` blocks runs each block in turn, its `EX` cards replacing
  the source set, and prints the groups the way NEC-5 does. That is the
  shape SimNEC writes for an N-port measurement (one port at 1 V, the others
  at a vanishing probe voltage), so its multiport networks and traps work
  against momwire as they do against the licensed engine. A block that
  changes the geometry, ground, loads or frequency between runs refuses by
  name.
- **Requests**: impedance runs (`PQ` / `XQ`; `PT -1`, which 4nec2's ItsHF
  decks write, leaves out the Wire Currents table and nothing else),
  far-field patterns (`RP`, including the range form SimNEC writes,
  and the NORMALIZED GAIN table an XNDA digit N of 1 to 5 asks for, as in
  `RP … 1500`: major axis, minor axis, vertical, horizontal or total
  gain, normalized to the table's peak or to the card's own factor),
  and near fields (`NE` / `NH`) over **all four ground cards** — free space,
  perfect ground, the Sommerfeld finite ground, and the MININEC-type `GD`
  (whose near field the engine solves in the medium, and so does momwire).
  The finite-ground tables ride a Sommerfeld point evaluator and sit within
  a measured 2–6 % of the licensed engine's captured cells, the same
  envelope class as the feedpoint impedances.
  The grid can be rectangular (`NE 0`: X, Y, Z in metres) or spherical
  (`NE 1` / `NH 1`, *added momwire#1257*): R in metres, then θ measured from
  the zenith, then φ from +x toward +y, both in degrees, walked R fastest,
  then θ, then φ. That is NEC-5's order, which is not NEC-2's (NEC-2 puts φ
  second); an elevation angle is 90° − θ. The table prints X, Y, Z either
  way, as the licensed engine's does. Far out over a finite ground the near
  field stays right down to the horizon (*momwire#1258*): past momwire's
  15-wavelength Sommerfeld table the ground's contribution is continued
  asymptotically — the reflected wave, the surface wave and the lateral wave
  through the soil — rather than held at the table's edge. Against the
  licensed engine at 20, 50 and 1000 wavelengths, within 15° of the horizon
  and in both polarisations, the difference is under 0.6 % of the field at
  each point, where holding the edge had made the vertically polarised field
  at the horizon 58 times too large at 1000 m.
- **Near fields with buried wires, and in the soil** (*momwire#1653*): an
  `NE` / `NH` grid on a deck with buried wires, or a grid point below a
  Sommerfeld ground, is served on every knot engine. Each pair of source
  and observation point is composed by which side of the ground each is on:
  in air from air, the direct field, its image and the reflected
  remainder, as above ground; in the soil from the soil, the same three
  written in the soil; across the interface either way, the transmitted
  field alone, read from the same contour integrals the buried fill solves
  with. Tangential E, normal D and H are continuous across z = 0 to a few
  parts in 10⁴; on decks whose impedance matches the licensed NEC-5's, the
  tables match its near-field printout to a few parts in 10⁴ from 1 cm
  above the surface to 50 cm into it, out to 300 m. Where the two engines'
  currents differ (at a crossing node), the fields differ by that same
  factor.

## What refuses, and why

The refusals are part of the product, and the interesting ones are honest
capability statements rather than deck errors:

- **Buried wires now serve — with measured edges.** Wires strictly below
  the interface over the Sommerfeld ground get **impedance, currents,
  charges, the radiation pattern and the near field**: detached buried radials and screens,
  bonded radial screens and ground rods (the wire that crosses the
  interface, written as two `GW` cards meeting at z = 0 or as one card
  straight through the plane), buried fed elements, and elevated feeds
  over buried counterpoises. A buried element reaches the
  far zone through the **transmitted** Fresnel factors — the stationary-phase
  limit of the same below-to-above integrals its currents came out of — so
  `RP` reads the wave that crosses the interface rather than an image of a
  source that is not in the air. The soil's absorption shows in the pattern
  and in the average power gain; the power budget keeps NEC's own definition
  (input minus conductor loss), so a buried deck's printed efficiency is
  still about its wires. Validation below ground stands
  on exact identities (the lossless-limit collapse onto free space at
  4×10⁻¹⁵, the deep-burial limit onto the infinite-medium solve), on
  ladder-limit agreement at the half-percent class on the
  radial/counterpoise classes with the buried-coupling differential
  matching to ~1 mΩ, and on a licensed NEC-5 asked with its documented
  below-ground card: the two engines agree to 0.2 % in resistance on a
  wholly buried fed dipole across depths and to a few percent on a
  bonded-base vertical over buried radials, and both reproduce the shape
  of Brown, Lewis and Epstein's 1937 measured radial-count curve. Two honest notes: a
  deck's first buried solve builds its below-interface Sommerfeld tables,
  and the refusals below are the map of where the capability ends.
- **One card through the plane is split there.** A `GW` with one end
  below a finite ground and the other above it is cut at the exact point
  its line meets z = 0, into the same crossing junction the two-card
  spelling writes — and it solves as that hand-written deck to the last
  bit. Every node keeps its point, so every address still names what it
  named. When z = 0 falls inside a segment, that segment becomes two, one
  in each medium, so a wire of N segments solves as N + 1; the licensed
  engine keeps that segment whole instead (on Dan AC6LA's buried-radial
  vertical with z = 0 inside the first of 21 segments, 72.373 + j4.336 Ω,
  against 71.296 + j0.044 Ω for the same deck split by hand). That is a
  discretisation choice, not a disagreement about the antenna, and it
  vanishes when a segment boundary lands on z = 0. The printout keeps the
  deck's numbering and says which wire was split. When a node lands on
  z = 0, a source *at* that node is the crossing-node source below, and it
  answers as the same deck written as two cards fed at their junction.
- **What a buried deck still refuses, each by name with its measurement**:
  a **ground-contact wire combined with buried wires** (the contact
  model has no conductor for the current spreading in the soil — [the
  counterpoise chapter](/act-5/counterpoise/) is the measurement behind
  that; an elevated feed over a buried counterpoise serves, and so does
  a screen bonded to the vertical at z = 0);
  buried wires over the perfect ground or `GD`
  (no lower medium to be in); sources deeper or pairs farther than the
  tabulated domains (the sentence states the limit and its extension
  cost). A wire lying *in* the interface refuses as the degenerate case
  it is.
- **A source on the crossing node serves; two narrower cases refuse.** With
  the vertical ending at z = 0 and a buried wire running down from that node,
  the source can sit *on* the node (`EX 0,1,-1`), and naming the node through
  the buried wire instead (`EX 0,6,-1`) gives the same answer. That matches
  the licensed engine, which prints the same impedance for both. Every knot
  engine serves it (`bs2`, `bs1`, `razor2p`,
  `sg`). *Since 0.79 (momwire#1569)* the node may also be
  where several buried wires meet the one wire rising from it: radials
  starting at a vertical's base, say. With more than two wires at the node
  an address cannot say which branch the source sits in, so the source is
  the node port on the named rising wire, between that wire and the buried
  wires together. Checked on `bs2` and `razor2p`, the same source
  written one segment up the rising wire converges onto it under
  refinement, and the licensed engine reaches the same limit. A node with
  **more than one rising wire** still refuses by name; the sentence says to
  join the rising wires above the ground, or to put the source on the
  rising wire one segment up. The point-matched `sinusoidal` engine refuses
  a crossing-node source by design: it places a source only at a segment centre, and the only
  segment-centre spelling of this source is a segment straddling the
  interface, which is not a well-posed feed — NEC-4.2, whose sources are
  segment-centred too, refuses it or solves it wrong. A node in the plane where
  wires meet with nothing buried is still a grounded junction, and a source
  there still refuses.
- **A near-field point on a wire's ground contact** (an `NE`/`NH` grid
  point sitting exactly where a wire stands on a finite ground) refuses
  naming the point: the field there is genuinely singular — a residual
  charge sits exactly at the observer — and neither momwire's answer nor
  the engine's converges under refinement, so printing either would be
  publishing a sampling artifact. The sentence quotes the measured
  divergence and says to move the observation point; the same grid one
  step off the contact serves.
- **A near-field point below a `GD` ground** (an `NE`/`NH` point with
  z < 0, which a spherical grid reaches once θ passes 90°) refuses naming
  the point: a `GD` deck's currents are solved over a perfect image, and
  there is no measured reading of the soil under it. Over the Sommerfeld
  ground (`GN 0` / `GN 2`) the same point serves, and over the perfect
  ground it serves as it does in the licensed engine.
- **Cards outside the emitted dialect** — surface patches, geometry
  generators (arcs, helices, catenaries), incident-wave excitation, load
  types other than `LD 2` / `LD 3` / `LD 4` / `LD 5`, magnetic grounds —
  refuse naming the card.
  EZNEC never writes these; a hand-written deck that does gets a sentence,
  not a guess.

## What answers underneath

The solver behind the drop-in engine is whichever of the two bundled engines you
pointed EZNEC at, defaulting to momwire's degree-2 B-spline — the same
physics as [the NEC-2 launcher's](/reference/portal-usage/) default, where the
full eight-engine roster and the cross-basis validation workflow are
documented. On the models where we hold a licensed NEC-5 reference, the
drop-in's printouts agree with the engine's element for element at the
sub-percent level, and the in-house razor-blade formulation
(`razor2p`) rides the licensed engine's own convergence path at the
0.01 % level — the receipts behind the word "emulates".

## The standalone executables

Each release since 0.79 also carries the nec5-razor2p and nec5-bs2 engines as native
programs with no Python in them, for Linux and, as previews, for Windows and
macOS. They write the Python engines' printouts, one process per
run, and SimNEC's streaming call works with them.
[The standalone executables](/reference/standalone/) covers the downloads,
which program to run, and the thread count. Under EZNEC on Windows, this
drop-in is still the supported route.

## From a source checkout

The bundle's engine is the `momwire.dropin.nec5` module frozen with PyInstaller
(the launcher in front of it is a small C program speaking one socket
protocol), and CI gates the routes against each other: the same deck through
the launcher, the frozen engine, or the module produces byte-identical
printouts. So if you already have a Python environment, the module is there:

```bash
pip install momwire
python -m momwire.dropin.nec5 EZN5.NEC NEC5.OUT
```

The arguments are exactly EZNEC's own — positional, cwd-relative, no flags.
This spelling is for running decks by hand, scripting a corpus, or working
on momwire itself. **It cannot be selected from EZNEC's engine dialog**,
which is a file picker and wants a path to an executable; reaching it from
EZNEC means writing a `.bat` wrapper, and a wrapper between the host and the
engine is where host-side misconfiguration turns into broken-pipe symptoms
that read as crashes. Use the packaged exe for that job.
