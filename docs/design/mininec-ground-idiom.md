# The MININEC ground is a cliff: serving `GD` with `GN 1`

Design doc for momwire#456 workstream 2's remaining ground decision, measured
2026-08-20: **serve the `GD`-with-`GN 1` idiom letter-faithfully to NEC-2 —
perfect-ground physics under `RP 0`/request-less execution, the second medium
under `RP 2`/`RP 3` — and retire the loud refusal.** The census ladder's
biggest remaining rung (+42 decks, 78.3 % → 88.8 %) turns out to require no
new physics: everything the reference engines do here, this engine already
does, and the refusal itself is now the only divergence.

## What was measured (C0, oracle = nec2c ae6ty 1.17)

Probe decks and printouts: capture-corpus additions in this arc; findings
digest in the arc's FINDINGS. The facts that carry the decision:

1. **`GN 1` + `GD` + `RP 0` is a byte-level no-op in the oracle.** Impedance
   and pattern identical to bare `GN 1` to every printed digit, banner
   `PERFECT GROUND`; the only printout difference is the DATA CARD echo of
   the `GD` line. Holds for every media value and for nonzero CLT/CH.
2. **`GD` never touches impedance, in any combination.** Cliff modes modify
   the far field only; the current solution is untouched.
3. **The record is read only by `RP 2`/`RP 3`** — and *that* is the
   "PEC currents + real-ground pattern" folklore, verbatim: with the cliff
   at distance 0, the far field reflects off the second medium everywhere
   while Z stays the perfect-ground value. The folklore names a *request
   mode*, not a ground type.
4. **4nec2 documents exactly this split.** Its manual (captured 2026-08-19,
   `exe__4Nec2.rtf`): far-field data "takes into account the ground
   conductivity and dielectric constant at a distance", while currents and
   input impedance "consider the ground to be perfect"; `GN 3` "is converted
   to a combination of perfect ground (GN 1) and a second **circular cliff**
   ground-medium … distance zero … height of zero". Validity note, same
   entry: no horizontal wires below 0.2 λ.
5. **4nec2's own engine answers perfect ground for the idiom deck.** The one
   captured engine run of a `GN 3` model (GP80, `XQ`-only) prints `PERFECT
   GROUND` from NEC-2D; nec2c agrees to every printed digit. And its engine
   *cannot* honor `GD` under `RP 0`: in the nec2dxs source (the engine 4nec2
   ships; Arie Voors' changes merged), FFLD computes the second-medium
   reflection coefficients only when `IFAR ≥ 2`.
6. **Mechanics pinned:** `GD 2` ≡ `GD 0` (integer field ignored — the NEC-4
   slot and all four hand-written bundle decks write `GD 2`); a `GN` card
   *resets* the second medium (same /FPAT/ slots), so `GD`-before-`GN 1` is
   clean perfect ground — this parser already matches both; `GD` without any
   `GN` is free space with `GD` inert.

## The decision

**Serve, letter-faithfully; the refusal comes out.** The hygiene wave's
refusal (`_refuse_the_mininec_ground_idiom`, momwire#458) was built on the
assumption that answering perfect-ground physics betrays the host's intent —
"the most dangerous category" in the 4nec2 capture doc. The measurements
invert that: perfect-ground physics **is** the reference engines' answer to
these decks, documented by the host's own manual, and the host gets its
MININEC far field by asking for it explicitly (`RP 2`/`RP 3`) — a request
this engine has served, oracle-pinned, since the cliff fixtures landed. A
drop-in that refuses decks the original engine runs is the divergence, not
the safety margin.

Concretely:

* `GD` + `GN 1` + `RP 0` / request-less → **serve as perfect ground**,
  byte-gated against the oracle (new capture fixtures, including the GP80
  deck verbatim).
* `GD` + `GN 1` + `RP 2`/`RP 3` → already served (unchanged).
* The `RP 2`/`RP 3` carve-out in the refusal, and the refusal itself,
  dissolve — the narrowing that gate encoded (all-zero medium is no medium;
  cliff requests read the record) stays true in the serving path and stays
  tested.

### Honesty notes (docs, not printout — the printout is byte-gated)

* A `GN 3`/MININEC model run for impedance gets perfect-ground Z. That is
  what 4nec2 ships, what its manual says, and what MININEC heritage means;
  it is also why the idiom exists (avoids the unpredictable Z of
  ground-contacting wires under Sommerfeld). Deck-grammar gets a row saying
  so in one sentence, with the 0.2 λ horizontal-wire validity line.
* A `GN 3`/MININEC model run with its own `RP 0` card gets a perfect-ground
  *pattern* — from us, from nec2c, and from 4nec2's bundled engine alike.
  The finite-media pattern exists only behind `RP 2`/`RP 3`.
* Envelope, grounded vertical @30 MHz eps 13/sigma 0.005: MININEC
  (cliff-at-0) keeps Z = 41.1+24.1j where Sommerfeld answers 75.0−85.9j, and
  sits 2.3–2.8 dB hot vs Sommerfeld at low angles — consistent with the
  EZNEC capture's 34 %-in-R `GD` vs `GN 0` finding.

### The one open capture — CLOSED, observed (2026-08-20, same day)

The Windows capture session (antennaknobs#963, captures 0039/0040) observed
exactly what this record inferred: 4nec2's far-field runs of the `GN 3`
model emit **mode digit 3** — `RP 3 19 73 1503` for the pattern and
`RP 3 37 1 1500` for the sweep — over the manufactured `GN 1` + `GD`. The
finite far field is asked for as the circular cliff, the impedance runs are
the request-less form this engine serves as perfect ground, and no world
remains in which the served bytes and the host's meaning diverge. The same
session pinned EZNEC's ground menu: MININEC type emits a **bare `GD` with
no `GN` card at all** — confirming the workstream-3 inheritance below (the
NEC-5 dialect reads a bare `GD` as MININEC-type ground, where this dialect
measured it free-space-inert), with the `GD 0,0,0,0,ε,σ,1.,0.` payload for
ws3 to pin its field layout against.

## What workstream 3 (EZNEC / NEC-5 dialect) inherits

EZNEC's ground menu has a real MININEC type, and NEC-5 implements it as a
genuine ground mode: the capture measured `GD` vs `GN 0` **34 % apart in R**
under identical banners — so in the NEC-5 dialect, `GD ε σ` must map to
**PEC currents + second-medium far field on ordinary pattern requests**, not
to a cliff-request carve-out. That is precisely the physics this engine
already computes on the `RP 2`/`RP 3` path; ws3's build item is routing, not
physics: a ground mode that solves over the image and applies the
Fresnel-image pattern for plain requests. Validation targets exist: the
EZNEC `Vert1` MININEC capture, and the licensed-materials twin per the usual
courtesy rule. The NEC-5 `GD` field layout (EZNEC emits a trailing `1.,0.`)
is ws3's to pin from its own capture.

The 4nec2 NEC-4 slot (`GD 2` emitted there) needs nothing: the integer field
is ignored by every engine measured, including this one.

## Flip-backs when the arc lands

Discharged by the #487 arc (sub-PRs #488/#493 and the docs unit) except the
last, which is the antennaknobs-side follow-up:

* `_refuse_the_mininec_ground_idiom` (deck/_nec2.py) — removed; its tests
  became serving tests (#493).
* `dipole_gd_second_medium` — returned to byte comparison (#493).
* Portal selftest deck 4 — restored to the `GD` form, along with the
  comma-`GD` and knob-drag probes #458 had moved (#493).
* deck-grammar-nec2.md — refusal row replaced by the serving section with
  the honesty notes and the 0.2 λ validity line (docs unit).
* `scripts/census_4nec2_bundle.py` (antennaknobs) — the hand-modelled idiom
  gate in its scoring block is deleted; the by-name half already reads the
  live table. Census re-runs and re-baselines the matrix (expected: the
  +MININEC rung of the ladder, 88.8 % class). **Pending.**

The arc's own findings, filed rather than folded in: momwire#489 (`GE`'s
sign is unread by the solve; its printout half was closed in #493) and
momwire#490 (the vacuum cliff: `RP 2`/`RP 3` over `GN 1` with an all-zero
medium reflects off vacuum where FFLD reflects off medium 1). And the CHT
sign in every prose description of the cliff was backwards until this arc:
POSITIVE CHT is the far side lower — the code always agreed with FFLD; the
words did not.
