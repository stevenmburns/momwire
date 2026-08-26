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

Measured on a 0.476 λ dipole in free space, ten mesh densities, against the
licensed engine:

| segments | licensed NEC-5 | B-spline (bs2) | razor-nec5 | \|razor − NEC-5\| |
|---|---|---|---|---|
| 5 | 64.22 − 91.78j | 68.14 − 28.81j | 64.23 − 91.74j | **0.041** |
| 21 | 67.06 − 35.66j | 67.92 − 28.59j | 67.06 − 35.62j | **0.037** |
| 61 | 67.51 − 30.68j | 67.86 − 28.37j | 67.52 − 30.64j | **0.037** |
| 161 | 67.68 − 29.28j | 67.84 − 28.23j | 67.68 − 29.24j | **0.037** |

The twin tracks the reference to 0.04 Ω at *every* density — flat, not
improving, which is what a twin looks like.

That says nothing about which is nearer the truth, so the second measurement
asks each basis about **itself**: even segment counts, the feed on the centre
knot, each basis scored against its own N = 320 answer. (The licensed engine
cannot join this one — on a plain `GW` it is segment-fed whichever card you
use, `EX 4` at a node returning the same impedance as `EX 0` at that index to
0.002 Ω — which is exactly why the table above had to be segment-fed.)

| segments | bs2 error | razor-nec5 error |
|---|---|---|
| 4 | 1.54 Ω | 80.63 Ω |
| 6 | 0.74 Ω | 41.36 Ω |
| 20 | 0.39 Ω | 7.16 Ω |
| 60 | 0.18 Ω | 1.91 Ω |
| 240 | 0.025 Ω | 0.17 Ω |

Both converge. At matched mesh the B-spline basis is 20–50× nearer its own
limit, which is the O(1/N) walk of razor-blade testing showing up as a cost
in segments: razor needs roughly 240 segments to reach the accuracy bs2 has
at 20.

Neither is "converged at coarse mesh" — bs2 is still 1.5 Ω out at four
segments. The difference is how fast the error comes down, not whether it is
there.

So: pick the **twin** when you want what NEC-5 *would have said* — checking a
published NEC-5 number, or matching a NEC-5 workflow. Pick the **default**
when you want momwire's own best answer. Neither is the accurate one in
general, and a disagreement between them is a difference of formulation, not
one of them being wrong.

**Making another.** The basis rides on the *filename*: everything after
`eznec-` selects it, a Windows `.exe` stripped first. So a copy you make
yourself works — rename a copy to `momwire-eznec-<basis>.exe`, beside the
same `_internal`, and that basis answers. This is the same rule the
[SimNEC portal's](/reference/portal-usage/) `momwire-nec2c-<basis>` commands
use, with one owner behind both. The bundle ships two rather than all nine
only to keep the download small; the other seven are a copy away:

```text
bspline  bspline-d1  hmatrix  arrayblock  razor  razor-nec5
sinusoidal  sinusoidal-galerkin  sinusoidal-galerkin-converged
```

The three sinusoidal families cannot answer this dialect — its printout
carries a `CHARGE DENSITY` table they have no basis to read it from — and say
so by name in the printout rather than failing quietly. A filename matching
no basis does the same: it refuses, names itself and lists what exists, so a
typo can never be served as the default.

One launch costs on the order of a second — the frozen interpreter's import
cost. (The one-file packaging that re-extracts on every launch was measured
at 17 s and rejected.) A long SWR sweep is therefore slower through the
frozen exe than through the licensed engine's 18–37 ms launches; the
per-point economics, not the physics, are the current gap.

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

As of 2026-08-22, 119 of the 122 EZNEC launches in our capture corpus serve;
the three refusals are one named sentence, about one observation point.

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
full nine-engine roster and the cross-basis validation workflow are
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
