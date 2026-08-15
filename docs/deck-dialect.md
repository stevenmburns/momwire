# momwire's NEC dialect

**The normative grammar lives on the site:
[momwire.dev/reference/deck-grammar-nec2/](https://momwire.dev/reference/deck-grammar-nec2/)**
— source at `site/src/content/docs/reference/deck-grammar-nec2.md`.

That page is the spec `momwire.deck.parse(text, dialect="nec2")` implements
and the parser's tests cite by anchor (`#gw--straight-wire`,
`#ex--voltage-source`, …). It is card by card and field by field: what each
supported card means, which fields are read and which are ignored, and the
exact text of every refusal. Change the behaviour and the page changes in the
same commit, or a citing test goes red.

This file exists so a repo-only reader finds the spec. Everything below is
implementation context that does not belong in a normative document.

## Where the semantics came from

The dialect is a **descriptive** spec of proven behaviour, not an invention.
Ground truth, in priority order:

1. `antennaknobs/src/antennaknobs/nec_portal.py` — the SimNEC-facing contract.
   Card handling (`parse_deck`), the refusal voice (`_DEFERRED_CARDS`), the
   `RP` mode gate (`_validate_rp`), the `EX` type gate, `GN`/`GD` reading,
   excitation retention, arming.
2. `antennaknobs/src/antennaknobs/nec_import.py` — the geometry translator the
   portal delegates to: `GW`/`GM`/`GS`, `(tag, segment)` resolution, `LD`
   translation, and `_snap_nec_connections`.
3. `antennaknobs/tests/fixtures/nec_portal/*.deck` — 44 decks of real usage,
   most of them captured SimNEC traffic.

Where the two readers disagree the portal wins; the site page's appendix
tabulates every disagreement.

## Deliberate deviations from today's portal

The spec is stricter than the portal's current code in six places. Each is a
case where the portal accepts a card and then silently drops its physics —
which produces a wrong answer rather than a refusal — or where the portal's
own stated intent and its control flow disagree. The parser units implement
the spec; the portal converges on it in the rewire phase.

| # | site section | portal today | spec |
|---|---|---|---|
| 1 | [refusals](https://momwire.dev/reference/deck-grammar-nec2/#cards-refused-by-name) | `TL`/`NT` translate into circuit branches | refused by name, message points at antennaknobs |
| 2 | same | `GA`/`GH`/`GX`/`GR` pass through to the importer and build geometry | refused by name — the corpus never uses them |
| 3 | [`LD`](https://momwire.dev/reference/deck-grammar-nec2/#ld--loading) | types 2 and 3 print in the loading report and move nothing | refused |
| 4 | same | a ranged `LD 5` over part of a wire is dropped with a note | refused, matching the `IS` precedent |
| 5 | same | an `LD` range over 8 segments, or a second `LD` on one segment, is dropped with a note | refused |
| 6 | [arming](https://momwire.dev/reference/deck-grammar-nec2/#arming) | `GN` is listed in `_ARMING_CARDS`, but its branch `continue`s before the arming test, so a `GN` between two execute cards never re-arms | `GN` arms — a ground change moves the operator, and the second run must be real |

Item 6 is a defect in the portal, not a design choice: no fixture pins it (all
16 corpus `GN` cards sit in the preamble), and the intent is unambiguous in the
code. It wants an oracle measurement (`… XQ / GN 1 / XQ`) to confirm which side
NEC-2 is on before the portal changes.

## The plural-dialect seam

`parse()` returns a **dialect-neutral** `DeckModel` — wires, arclengths, feeds,
node gaps, loads, ground, frequencies, requests — and `build_solver(model,
basis=...)` maps it onto the solver families. A NEC-5 dialect is therefore a
second parser, not a second pipeline. Nothing NEC-2-specific may enter the
model's vocabulary; `node_gaps` is already there because it is how NEC-5's
segment-end sources will land.
