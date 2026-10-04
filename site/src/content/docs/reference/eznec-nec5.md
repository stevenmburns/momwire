---
title: "Running momwire as EZNEC's NEC-5 engine"
description: The momwire.eznec portal serves EZNEC Pro+'s external-engine slot in the NEC-5 dialect — decks in, byte-compatible printouts out, and every deck either serves or refuses by name.
---

EZNEC Pro+ can drive an external NEC-5 console engine: it writes a deck
(`EZN5.NEC`), launches the engine with two positional arguments, and reads
the printout (`NEC5.OUT`) back for every display it draws. momwire ships
that process.

## Setting it up

1. Download
   [`momwire-eznec-windows.zip`](https://github.com/stevenmburns/momwire/releases/latest/download/momwire-eznec-windows.zip)
   from the latest release.
2. Before extracting, right-click the zip, choose Properties, tick
   **Unblock** and click OK. That clears the download mark on every file
   inside at once.
3. Extract it into **a folder of its own**, for example
   `C:\momwire-eznec\` (the zip already contains a `momwire-eznec` folder).
   Not into EZNEC's own folders, and not over a previous release.
   **Keep the folder together**: the exe needs the `_internal` runtime
   beside it, and a lone copied-out `.exe` is the one way a correct
   download still fails.
4. Point EZNEC's external-engine path at `momwire-nec5.exe` inside that
   folder. (For EZNEC's External NEC-4.2 engine option, point it at
   `momwire-nec4.exe` instead — see [below](#choosing-the-formulation).)

That is the whole installation. No Python, no environment, nothing on PATH.
EZNEC's interface, models and displays stay EZNEC's; the electromagnetics
become momwire's.

### Updating to a new release

Delete the `momwire-eznec` folder entirely, then extract the new zip in its
place. The path stays the same, so EZNEC's setting does not change.

Never extract a new release over an old one. Extracting replaces the files
that share a name and leaves the rest, so the old release's files stay
behind in `_internal`. That includes its version record, and it can include
libraries.

- If Windows says a file is in use while you delete the folder, that is the
  resident engine described below. Wait fifteen minutes for it to retire, or
  end `momwire-eznec-engine.exe` in Task Manager.
- Line 2 of every `NEC5.OUT` names the engine that answered, for example
  ` momwire 0.69.0 bspline avx2`. If it ends in `(stale: 0.66.0)`, the
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

## Choosing the formulation

The bundle carries seven launchers and the one engine they run, and the
choice is the engine PATH you set. One family per EZNEC engine option:

```text
EZNEC's External NEC-5 slot, and SimNEC's NEC-5 engine:
momwire-nec5.exe              the default — degree-2 B-spline (bs2)
momwire-nec5-razor-2p.exe     the tent basis with razor-blade path testing,
                              at NEC-5's two-point rule

EZNEC's External NEC-4.2 slot:
momwire-nec4.exe              the default — degree-2 B-spline (bs2)
momwire-nec4-sinusoidal.exe   the sinusoidal basis, point matched

The old names, deprecated — still shipped, still identical:
momwire-eznec.exe             = momwire-nec5.exe
momwire-eznec-razor-2p.exe    = momwire-nec5-razor-2p.exe
momwire-eznec-razor-nec5.exe  = momwire-nec5-razor-2p.exe

momwire-eznec-engine.exe      the compute engine the launchers run
```

The name picks the deck dialect as well as the basis, so a `momwire-nec4`
name belongs in EZNEC's NEC-4.2 slot and a `momwire-nec5` name in its NEC-5
slot. In the NEC-4.2 slot, `sinusoidal` is the basis family NEC-4.2 itself
belongs to; razor cannot serve there, because NEC-4.2 decks put their sources
at segment centres and razor places sources at knots, so a
`momwire-nec4-razor-2p` copy refuses by name.

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

Each source's impedance agrees with the licensed NEC-4.2 to within about 2 %.

**The `momwire-eznec` names are deprecated.** They are what earlier releases
said to point EZNEC at, and they still ship and answer exactly as before —
same engine, same warm server, same printout. An engine path typed once and
forgotten keeps working. Point new setups at the `momwire-nec5` names, which
also work in SimNEC.

Point EZNEC at a **launcher**, never at `momwire-eznec-engine.exe` — the
engine answers correctly either way, but naming it directly gives up the
warm start above for nothing.

They accept the same models — measured, deck by deck, on the whole captured
corpus — and answer in different formulations. `razor-2p` is the tent basis
with razor-blade path testing at NEC-5's two-point rule — the formulation
NEC-5 itself uses.

`razor-nec5` is the **deprecated spelling** of the same basis (#785/#794). It
runs the same engine and answers identically, and `momwire-eznec-razor-nec5.exe`
still ships because an EZNEC engine path is a string typed once and forgotten;
dropping the name would break those installs silently. Point new setups at
`momwire-nec5-razor-2p.exe`.

### Reproduction is not accuracy

`razor-2p` agrees with the licensed engine **because it runs the same
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

| segments | licensed NEC-5 | bs2 | razor-2p | \|razor − NEC-5\| |
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

| segments | bs2 error | razor-2p error |
|---|---|---|
| 4 | 2.81 Ω | 80.14 Ω |
| 20 | 0.82 Ω | 6.67 Ω |
| 60 | 0.25 Ω | 1.43 Ω |

Both converge. At matched mesh the B-spline basis is 5.8–28× nearer its own
limit, which is the O(1/N) walk of razor-blade testing priced in segments.

Neither is "converged at coarse mesh" — bs2 is still 2.8 Ω out at four
segments. The difference is how fast the error comes down, not whether it is
there.

**Where the last ohm actually lives** (measured 2026-08-26; `parity_limits()`
in the probe script is the receipt). Extrapolate both ladders and the two
formulations nearly meet: razor-2p's limit lands 0.08–0.21 Ω from bs2's,
depending on the extrapolation model. Of the roughly one ohm between the
bases at N = 160 (0.95 Ω through this seam, 1.05 Ω on the probe's direct
ladder), almost all is the twin still descending the O(1/N) path it shares
with the licensed engine, and only the fraction-of-an-ohm remainder is
formulation. The same run settles a tooling question: with the feed
pinned at the exact centre, bs2's limit is the same whether that centre is a
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
`nec5-` (or `nec4-`) selects it, a Windows `.exe` stripped first. So a copy
you make yourself works — copy a **launcher** (a couple of hundred kilobytes,
not the engine), rename the copy to `momwire-nec5-<basis>.exe` (or
`momwire-nec4-<basis>.exe`) in the same folder, and that basis answers in that
slot. This is the same rule the [SimNEC portal's](/reference/portal-usage/)
`momwire-nec2c-<basis>` commands use, with one owner behind both. The bundle
ships two per slot because that is what the parity work was about; the rest
are a copy away:

```text
NEC-5 slot:    bspline  bspline-d1  hmatrix  arrayblock  razor-2p
               sinusoidal-galerkin
NEC-4.2 slot:  bspline  bspline-d1  hmatrix  arrayblock  sinusoidal
               sinusoidal-galerkin  pulse
```

The `momwire-nec5` names are also what
[SimNEC's NEC-5 engine setting](/reference/portal-usage/#simnecs-nec-5-engine)
needs, since SimNEC reads `nec5` in the path. A deprecated
`momwire-eznec-<basis>.exe` copy still works in EZNEC, as it always did.

`sinusoidal` cannot answer the NEC-5 dialect — every deck in it drives a NODE, and
under point matching the match points are the segment centres, so a delta at a
node point-samples to nothing in every row and there is no excitation left to
solve. It says so by name in the printout rather than answering about a
different antenna. `sinusoidal-galerkin` has no such trouble: its test
integral collapses the same delta to a well-defined drive, and it serves the
corpus alongside the B-spline and razor families. A filename matching
no basis does the same: it refuses, names itself and lists what exists, so a
typo can never be served as the default. The match is case-insensitive, as
Windows filenames are, so `Momwire-NEC5-Razor-2p.exe` is the twin too.

:::caution[Not a supported configuration]
No part of EZNEC knows momwire exists, and nothing here has been reviewed or
endorsed by EZNEC's author. The portal was built black-box, from the decks
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
 momwire 0.55.0 bspline avx2
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

## What serves

- **Geometry**: `GW` wires — straight, junctioned, tapered via stepped
  radii, exactly as EZNEC emits them (EZNEC resolves its own transforms
  before writing the deck).
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
- **Requests**: impedance runs (`PQ` / `XQ`), far-field patterns (`RP`,
  including the range form SimNEC writes),
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

## What refuses, and why

The refusals are part of the product, and the interesting ones are honest
capability statements rather than deck errors:

- **Buried wires now serve — with measured edges.** Wires strictly below
  the interface over the Sommerfeld ground get **impedance, currents,
  charges and the radiation pattern**: detached buried radials and screens,
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
  **`NE`/`NH` on decks with buried wires** (#524 phase 3);
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
  engine serves it (`bspline`, `bspline-d1`, `razor-2p`,
  `sinusoidal-galerkin`). A node where three or more wires meet refuses by
  name, citing #1282. The point-matched `sinusoidal` engine refuses it by
  design: it places a source only at a segment centre, and the only
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
- **A near-field point below a finite ground** (an `NE`/`NH` point with
  z < 0 over `GN 0` / `GN 2` / `GD`, which a spherical grid reaches once θ
  passes 90°) refuses naming the point: the field in the soil is the
  transmitted field, and momwire's near-field readout composes the field
  above the interface only. Over the perfect ground such a point serves, as
  it does in the licensed engine.
- **Cards outside the emitted dialect** — surface patches, geometry
  generators (arcs, helices, catenaries), incident-wave excitation, load
  types other than `LD 2` / `LD 3` / `LD 4` / `LD 5`, magnetic grounds —
  refuse naming the card.
  EZNEC never writes these; a hand-written deck that does gets a sentence,
  not a guess.

## What answers underneath

The solver behind the portal is whichever of the two bundled engines you
pointed EZNEC at, defaulting to momwire's degree-2 B-spline — the same
physics as [the SimNEC portal's](/reference/portal-usage/) default, where the
full eight-engine roster and the cross-basis validation workflow are
documented. On the models where we hold a licensed NEC-5 reference, the
portal's printouts agree with the engine's element for element at the
sub-percent level, and the in-house razor-blade formulation
(`razor-2p`) rides the licensed engine's own convergence path at the
0.01 % level — the receipts behind the word "emulates".

## From a source checkout

The bundle's engine is the `momwire.eznec` module frozen with PyInstaller
(the launcher in front of it is a small C program speaking one socket
protocol), and CI gates the routes against each other: the same deck through
the launcher, the frozen engine, or the module produces byte-identical
printouts. So if you already have a Python environment, the module is there:

```bash
pip install momwire
python -m momwire.eznec EZN5.NEC NEC5.OUT
```

The arguments are exactly EZNEC's own — positional, cwd-relative, no flags.
This spelling is for running decks by hand, scripting a corpus, or working
on momwire itself. **It cannot be selected from EZNEC's engine dialog**,
which is a file picker and wants a path to an executable; reaching it from
EZNEC means writing a `.bat` wrapper, and a wrapper between the host and the
engine is where host-side misconfiguration turns into broken-pipe symptoms
that read as crashes. Use the packaged exe for that job.
