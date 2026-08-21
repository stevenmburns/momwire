---
title: "Running momwire as EZNEC's NEC-5 engine"
description: The momwire.eznec portal serves EZNEC Pro+'s external-engine slot in the NEC-5 dialect — decks in, byte-compatible printouts out, and every deck either serves or refuses by name.
---

EZNEC Pro+ can drive an external NEC-5 console engine: it writes a deck
(`EZN5.NEC`), launches the engine with two positional arguments, and reads
the printout (`NEC5.OUT`) back for every display it draws. momwire ships
that process:

```bash
pip install momwire
python -m momwire.eznec EZN5.NEC NEC5.OUT
```

The invocation is exactly EZNEC's own — positional, cwd-relative, no flags —
so the engine slots into the external-engine mechanism as-is. EZNEC's
interface, models and displays stay EZNEC's; the electromagnetics become
momwire's.

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

As of 2026-08-21, 115 of the 122 EZNEC launches in our capture corpus serve;
every one of the seven refusals is a named sentence.

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
  and near fields (`NE` / `NH`) over free space and perfect ground.

## What refuses, and why

The refusals are part of the product, and the interesting ones are honest
capability statements rather than deck errors:

- **Wires below the ground interface** — buried radials and buried screens
  are real NEC-5 capabilities momwire does not have yet; such decks refuse
  naming the wire and the depth. A wire lying *in* the interface refuses as
  the degenerate case it is.
- **Near fields over the finite grounds** (`NE`/`NH` over `GN 0` or `GD`)
  refuse naming the ground: momwire has no Sommerfeld near-field evaluator
  yet, and the sentence says which card to change to get an answer.
- **Cards outside the emitted dialect** — surface patches, geometry
  generators (arcs, helices, catenaries), incident-wave excitation, load
  types other than `LD 4`, magnetic grounds — refuse naming the card.
  EZNEC never writes these; a hand-written deck that does gets a sentence,
  not a guess.

## What answers underneath

The solver behind the portal is momwire's default degree-2 B-spline engine —
the same physics as [the SimNEC portal's](/reference/portal-usage/) default,
where the full nine-engine roster and the cross-basis validation workflow
are documented. On the models where we hold a licensed NEC-5 reference, the
portal's printouts agree with the engine's element for element at the
sub-percent level, and the in-house NEC-5 formulation twin
(`razor-nec5`) rides the licensed engine's own convergence path at the
0.01 % level — the receipts behind the word "emulates".

A packaged Windows executable for pointing EZNEC's engine path at — no
Python environment required — is in preparation; today the `python -m
momwire.eznec` spelling above is the supported form.
