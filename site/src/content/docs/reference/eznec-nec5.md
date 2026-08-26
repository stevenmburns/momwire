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
2. Unzip it anywhere. **Keep the folder together** — the exe needs the
   `_internal` runtime beside it, and a lone copied-out `.exe` is the one way
   a correct download still fails.
3. Point EZNEC's external-engine path at `momwire-eznec.exe` inside that
   folder.

That is the whole installation. No Python, no environment, nothing on PATH.
EZNEC's interface, models and displays stay EZNEC's; the electromagnetics
become momwire's.

One launch costs on the order of a second — the frozen interpreter's import
cost. (The one-file packaging that re-extracts on every launch was measured
at 17 s and rejected.) A long SWR sweep is therefore slower through the
frozen exe than through the licensed engine's 18–37 ms launches; the
per-point economics, not the physics, are the current gap.

## Choosing the formulation

The bundle carries two engines, and the choice is the engine PATH you set:

```text
momwire-eznec.exe             the default — degree-2 B-spline (bs2)
momwire-eznec-razor-nec5.exe  the NEC-5 formulation twin
```

They accept the same models — measured, deck by deck, on the whole captured
corpus — and answer in different formulations. `razor-nec5` is the tent basis
with razor-blade path testing NEC-5 itself uses.

### Reproduction is not accuracy

The twin agrees with the licensed engine **because it runs the same
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

| segments | licensed NEC-5 | bs2 | razor-nec5 | \|razor − NEC-5\| |
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

| segments | bs2 error | razor-nec5 error |
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
formulations nearly meet: razor-nec5's limit lands 0.08–0.21 Ω from bs2's,
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
`eznec-` selects it, a Windows `.exe` stripped first. So a copy you make
yourself works — rename a copy to `momwire-eznec-<basis>.exe`, beside the
same `_internal`, and that basis answers. This is the same rule the
[SimNEC portal's](/reference/portal-usage/) `momwire-nec2c-<basis>` commands
use, with one owner behind both. The bundle ships two rather than all eight
only to keep the download small; the other six are a copy away:

```text
bspline  bspline-d1  hmatrix  arrayblock  razor  razor-nec5
sinusoidal  sinusoidal-galerkin
```

`sinusoidal` cannot answer this dialect — every deck in it drives a NODE, and
under point matching the match points are the segment centres, so a delta at a
node point-samples to nothing in every row and there is no excitation left to
solve. It says so by name in the printout rather than answering about a
different antenna. `sinusoidal-galerkin` has no such trouble: its test
integral collapses the same delta to a well-defined drive, and it serves the
corpus alongside the B-spline and razor families. A filename matching
no basis does the same: it refuses, names itself and lists what exists, so a
typo can never be served as the default. The match is case-insensitive, as
Windows filenames are, so `Momwire-EZNEC-Razor-Nec5.exe` is the twin too.

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
- **Loads and networks**: `LD 4` impedance loads, insulation (`IS`),
  transmission lines and non-radiating networks (`TL` / `NT`), including
  the mixed table layouts.
- **Requests**: impedance runs (`PQ` / `XQ`), far-field patterns (`RP`),
  and near fields (`NE` / `NH`) over **all four ground cards** — free space,
  perfect ground, the Sommerfeld finite ground, and the MININEC-type `GD`
  (whose near field the engine solves in the medium, and so does momwire).
  The finite-ground tables ride a Sommerfeld point evaluator and sit within
  a measured 2–6 % of the licensed engine's captured cells, the same
  envelope class as the feedpoint impedances.

## What refuses, and why

The refusals are part of the product, and the interesting ones are honest
capability statements rather than deck errors:

- **Buried wires now serve — with measured edges.** Wires strictly below
  the interface over the Sommerfeld ground get **impedance, currents and
  charges**: detached buried radials and screens, buried fed elements, and
  elevated feeds over buried counterpoises. Validation is deliberately
  engine-independent below ground — the NEC family's buried-conductor
  weakness is documented publicly (LLNL-TR-490316; corroborated in our
  licensed materials, details in private notes) — so the gates are exact
  identities (the lossless-limit collapse onto free space at 4×10⁻¹⁵, the
  deep-burial limit onto the infinite-medium solve) plus, on the
  radial/counterpoise classes where the reference engine's convergence
  ladders are clean, ladder-limit agreement at the half-percent class with
  the buried-coupling differential matching to ~1 mΩ. Two honest notes: a
  deck's first buried solve builds its below-interface Sommerfeld tables
  (about a minute or two today — the accelerated fills are momwire#568),
  and the refusals below are the map of where the capability ends.
- **What a buried deck still refuses, each by name with its measurement**:
  a wire **crossing** the interface (the crossing physics is #524 phase 2,
  and the refusal quotes the three banked reference anchors it must meet);
  a **ground-contact wire combined with buried wires** (the fill's
  boundary term is O(1) on a contact basis — measured 2.5 relative at the
  lossless-limit identity, against 10⁻⁵ with the feed lifted clear; an
  elevated feed over a buried counterpoise serves, and momwire#567 is the
  lift); **`NE`/`NH` on decks with buried wires** (#524 phase 3);
  **`RP` on decks with buried wires** (the far-zone transmitted
  asymptotics, momwire#570); buried wires over the perfect ground or `GD`
  (no lower medium to be in); sources deeper or pairs farther than the
  tabulated domains (the sentence states the limit and its extension
  cost). A wire lying *in* the interface refuses as the degenerate case
  it is.
- **A near-field point on a wire's ground contact** (an `NE`/`NH` grid
  point sitting exactly where a wire stands on a finite ground) refuses
  naming the point: the field there is genuinely singular — a residual
  charge sits exactly at the observer — and neither momwire's answer nor
  the engine's converges under refinement, so printing either would be
  publishing a sampling artifact. The sentence quotes the measured
  divergence and says to move the observation point; the same grid one
  step off the contact serves.
- **Cards outside the emitted dialect** — surface patches, geometry
  generators (arcs, helices, catenaries), incident-wave excitation, load
  types other than `LD 4`, magnetic grounds — refuse naming the card.
  EZNEC never writes these; a hand-written deck that does gets a sentence,
  not a guess.

## What answers underneath

The solver behind the portal is whichever of the two bundled engines you
pointed EZNEC at, defaulting to momwire's degree-2 B-spline — the same
physics as [the SimNEC portal's](/reference/portal-usage/) default, where the
full eight-engine roster and the cross-basis validation workflow are
documented. On the models where we hold a licensed NEC-5 reference, the
portal's printouts agree with the engine's element for element at the
sub-percent level, and the in-house NEC-5 formulation twin
(`razor-nec5`) rides the licensed engine's own convergence path at the
0.01 % level — the receipts behind the word "emulates".

## From a source checkout

The executable is the `momwire.eznec` module frozen with PyInstaller, and
CI gates the two against each other: the same deck through either route
produces byte-identical printouts. So if you already have a Python
environment, the module is there:

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
