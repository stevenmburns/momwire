---
title: "16 · The counterpoise question: when the right answer is a refusal"
description: A vertical standing on the ground over buried radials is the most-requested buried deck there is. momwire refuses it. This chapter is the measurement campaign behind that refusal — what the deck fails to specify, what every candidate spelling missed, and which respellings are served instead.
---

Ask any antenna modeler what they want from real-soil support and the
answer is a picture: a quarter-wave vertical, base at ground level,
over a fan of radials buried a few inches down. momwire serves buried
wires. momwire serves ground-contact verticals. Ask it for the two
*together* — a contact vertical over **detached** buried radials — and
it refuses, by name, with a sentence that ends in suggestions.

That refusal is not a missing feature. It is the most-measured single
decision in this codebase, and this chapter is the story: three probe
campaigns, four candidate physics spellings, two convergence ladders,
one feed-addressing erratum that forced the whole thing to be re-derived
— and a residual that survived all of it because it is real physics the
deck cannot express.

## What the deck doesn't say

Follow the current. At the base of a contact vertical, I(0) is not
zero — charge arrives at the interface and must go *somewhere*. In a
deck with no buried wires, momwire's contact model (chapter 9's
reflection machinery, [momwire#151]) continues it as a scaled image: a
fiction, but a safe one, because every observer is above the ground and
the image reproduces what they see. A buried radial breaks the safety:
it is an observer *inside* the soil, parked 15 cm below the fiction.

What should it see? The real system puts the base current into the
soil as a **spreading conduction current** — the physics of a ground
electrode. And here the deck goes silent: a spreading current needs an
electrode geometry, and "a wire end touching z = 0" doesn't specify
one. A thin wire tip is a terrible electrode — the point-contact
spreading resistance is ~1/(4σa), which at σ = 0.005 S/m and a = 1 mm
is **tens of kilohms**. Real installations don't work that way and
real installers know it: the radials are *bonded* to the vertical
precisely so the current has a conductor to spread through. The
detached deck asks for the coupling of an electrode it never
describes. It underspecifies its own physics.

## What the licensed engine answers

NEC-5 answers this deck. Our licensed copy prints 92.130−70.141j Ω for
the one-radial anchor (15-segment mesh). The natural question — the
one this whole arc set out to answer — was: *can momwire spell
whatever physics that answer embodies, and match it?*

Two findings from the engine's own printed output changed the shape of
that question:

1. **Its printed junction currents violate current conservation at the
   contact point, divergently.** Reconstructing node currents from its
   printed tables across a refinement ladder, the current entering the
   interface point exceeds what leaves along the wires by 1.55 A
   growing to 2.23 A as the mesh refines (~√n). The engine's account
   of the base is a point-electrode sink — a stake fiction — and it is
   not converging to a consistent junction.
2. **Its coarse prints sit far from its own converged values.** Held
   at one drive node and refined ×1→×8, the engine's anchor print
   moves 6.8 Ω. Every honest comparison below is
   converged-vs-converged; single-mesh numbers on this class are
   discretization noise wearing a physics costume.

(All engine statements here are measurements of printed impedances and
printed current tables on these specific decks — nothing else.)

## The campaign: four spellings, two ladders, one erratum

momwire's candidate serve was the *continuation-consistent* spelling:
keep the contact fiction on the vertical's own account (chapter 9's
column, unchanged), couple the buried radial through the designed
cross-medium kernels, omit the end terms the fiction cannot support.
Measured against the engine's converged anchors at the matched drive
node, that spelling lands:

- **one radial: 3.02 Ω away** — and the split is remarkable: the
  converged *resistances* agree to **0.08 Ω**; the entire residual is
  reactive.
- **four radials: 6.26 Ω away** — now with a real part (ΔR 3.5,
  ΔX −5.2).

(An addressing erratum is buried in "matched drive node": the engine's
`EX` card drives a segment's far *node*, not its center. For a while
our comparisons fed momwire half a segment from the engine's actual
drive point, and on this reactive deck that half-segment was worth
~10 Ω of fictitious disagreement. Found, fixed, everything re-derived.
Feed conventions scale as Z² — the lesson is now load-bearing in the
test suite.)

Could any *added* physics close the remaining 3–6 Ω? Four families,
each measured, each excluded for its own reason:

1. **Solve the soil current exactly** — respell the contact as a real
   crossing junction. Convergent, served, and *a different antenna*:
   that is the bonded deck (below), not this one.
2. **A solved stub** under the continuation spelling — not a
   convergent object (its answer is set by the mesh, not the physics).
3. **A prescribed line ghost** — closable per-deck to 0.001 Ω, but the
   fitted strength **does not transfer between decks**
   (0.48+0.36j on one radial, 0.36−0.07j on four): whatever sink
   strength the engine's answer embodies, it is *solved per deck*, not
   a physical constant momwire could adopt.
4. **A derived Born electrode** (hemispherical spreading, first
   principles) — a measured null.

And the residual's own structure says why: purely reactive with one
collector, gaining a real part as collectors multiply — that is what a
*real spreading current* looks like as radials arrive to intercept it.
It is not what any fixed source constant looks like. The physics the
engine's answer embodies is exactly the physics the deck never
specified. There is nothing consistent to spell.

## The refusal, and what it hands you instead

So momwire refuses the detached contact+buried combination — not
because the answer is unknown, but because every honest answer is
"3–6 Ω from a stake fiction, and the stake isn't in your deck." The
refusal sentence itself hands you the three served respellings:

- **Bond the radials.** Respell the screen to rise to the surface and
  junction-join the vertical at z = 0. The crossing junction is served
  (one above wire over N below wires, buried-hub spelling included) —
  and it is what a real installation is.
- **Lift the feed.** An elevated vertical over detached buried radials
  is served, converged, and cross-validated against the licensed
  engine at the 0.2–2 Ω level — the full study is on the antennaknobs
  site: [Buried radials: two engines on the same
  dirt](https://antennaknobs.dev/advanced/buried-radials/).
- **Solve the buried structure alone.**

One last piece of engineering honesty: the two engine anchor numbers
are banked in the test suite as *self-arming gates*. They cost a
refusal today; if any future change lifts the refusal, they
automatically score the served answer against the engine's prints —
within the envelope this campaign measured — without anyone remembering
to switch them on. A refusal this load-bearing doesn't get to rot.

An engine that answers every deck is wrong somewhere and silent about
where. momwire's bet — the same one chapter 15's instrument made — is
that *attributable* accuracy beats universal answers: serve what
converges, refuse what the deck cannot specify, and leave a measurement
trail either way.

[momwire#151]: https://github.com/stevenmburns/momwire/issues/151
