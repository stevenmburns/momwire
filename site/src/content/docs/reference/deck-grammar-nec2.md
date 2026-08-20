---
title: "The nec2 deck dialect"
description: The normative card-by-card grammar of momwire's NEC dialect — wire structures and the network cards attached to them: which cards run, which fields are read, which are ignored, and the exact text of every refusal.
---

This page is the contract. `momwire.deck.parse(text, dialect="nec2")` accepts
exactly the deck language defined here, and nothing else; the parser's tests
cite these sections by anchor, so the page cannot drift from the code without
a test going red.

Every heading below is a stable citation target. Card sections slug as
`#gw--straight-wire`, `#ex--voltage-source`, and so on — mnemonic first,
lowercased, with the em dash dropped and spaces hyphenated.

## Dialects

An engine's dialect is part of its identity. nec2c reads NEC-2's deck
language, NEC-5 reads its own, and momwire reads **`nec2`** — a restricted
NEC-2 that describes wire antennas, the two-port circuits attached to them,
and asks them questions. It is the first of a planned family: a NEC-5 flavour
is the probable second, and the seam is built for it from day one.

Every dialect front-end parses into one dialect-neutral [`DeckModel`](#the-deckmodel),
and `build_solver(model, basis=...)` maps that model onto momwire's solver
families. A second dialect is therefore a second parser, not a second
pipeline — and nothing NEC-2-specific may leak into the model's vocabulary.
The model speaks wires, arclengths, feeds, gaps and grounds; tags, segments
and card mnemonics stop at the parser.

`dialect="nec2"` is the only value this release ships.

### What the dialect describes, and what it excludes

The dialect describes **a structure of thin wires, driven by voltage sources,
optionally over a ground, with two-port circuits attached to it**: a
transmission line ([`TL`](#tl--transmission-line)) or an explicit admittance
matrix ([`NT`](#nt--two-port-network)) between two segments. That is NEC-2's
network vocabulary, in full, minus nothing.

What it does not describe is anything that is not a **wire**. Surface patches
(`SP`, `SM`, `SC`), arcs (`GA`), helices (`GH`) and tapered-wire continuations
(`GC`) are refused by name; so is the numerical Green's function (`GF`), and
so are the 4nec2 authoring extensions (`SY`) that never reach an engine
anyway. See [Refusals](#refusals) for each message.

The line is drawn at a measurement, not at a preference. Across the 44-deck
reference corpus the card histogram is `GW` 104, `EX` 55, `XQ` 52, `FR` 49,
`NX` 45, `GE` 45, `CE` 45, `CM` 21, `GN` 16, `RP` 7, `LD` 7, `GD` 5, `PT` 3,
`EK` 3, `MP` 2, two `TL`, two `NT`, and one each of `NE`, `NH`, `GS`, `GM`.
Zero patches, zero arcs, zero helices. The dialect covers what real decks
contain.

The network cards were themselves once outside it, refused by name as out of
an "antenna-only" grammar. That reading did not survive contact with the
seams: `NT` reaches the engine in 53 of the 457 bundled 4nec2 models and `TL`
in 45, and EZNEC's feed-system examples are network models outright. NEC-2
solves these cards natively, so serving them makes this dialect *more*
faithful to the engine it emulates, not less.

## Deck framing

A deck is a sequence of card lines. Cards are **free format**: the mnemonic is
the first two characters, fields are separated by spaces and/or commas, and a
mnemonic glued to its first field (`GE1`, `GD 2,0,0,0,...`) splits correctly.
Numeric fields accept Fortran `D` exponents (`1.0D-3`). Blank lines are
skipped.

A deck is framed like this:

```text
CM optional comment lines
CE last comment, or the end of the comment block
GW ...            geometry
GE 0              end of geometry
GN ...  LD ...  IS ...  EK          structure-level cards
EX ...  FR ...                      excitation and frequency
XQ / RP / NE / NH                   execute
NX                                  end of deck
```

Ordering rules the dialect enforces:

* **Comments come first.** `CM` and `CE` carry free text; `CE` conventionally
  ends the block. Neither is positional in the parser — a comment card
  anywhere is read as a comment — but a deck that scatters them is not
  idiomatic.
* **Geometry precedes `GE`.** `GE` terminates the geometry section and carries
  the ground-plane flag. A deck with no `GE` parses as though it wrote `GE 0`.
* **`IS` precedes the first execute card.** Insulation is a property of the
  structure, so it cannot change between runs. See [IS](#is--insulated-sheath).
* **`NX` ends the deck.** It is the frame terminator, and its echo is the
  sentinel a resident caller waits on. `EN` also terminates a deck and
  additionally ends the run. Everything after `NX` is a new deck; a body still
  open at end of input parses as though it ended with `EN`, matching NEC's own
  card reader, which synthesizes `EN` at EOF.

Two mnemonic-level errors are raised before any card is interpreted:

```text
CARD'S MNEMONIC CODE TOO SHORT OR MISSING: <line>
NON-NUMERICAL CHARACTER IN FIELD: <token> on <line>
```

### Field numbering

NEC names a card's fields `I1`–`I4` (integers) then `F1`–`F6` (reals), and
this page uses that naming. The parser reads fields **positionally** and
converts on demand, so an integer field written `1.` reads as `1` — except on
`MP`, where a fractional field is an error (see [MP](#mp--multiprocessing-hint)).
The geometry cards `GW` and `GM` carry a seventh real field, `F7`, in the
ninth position.

A field this page does not name is **read and discarded**. It never changes an
answer, and the parser never refuses a deck because of one.

## Addressing

NEC addresses a point on the structure as **(tag, segment)**. The dialect
resolves that to **(wire, arclength)**, which is what momwire's constructors
take.

* A **tag** is the integer label on a `GW` card. Several wires may share a
  tag; a tag of `0` on a reference means "absolute segment number".
* A **segment** is 1-based. With a nonzero tag it counts only segments of
  wires carrying that tag, in card order; with `tag = 0` it counts every
  segment of the structure, in card order. A segment number of `0` or below
  reads as `1`.
* Resolution walks the wire list accumulating segment counts, returns the
  first wire whose running total reaches the requested segment, and keeps the
  1-based **local** index within it. A request that runs off the end refuses:

  ```text
  segment <n> is out of range for tag <t>
  segment <n> is out of range for the structure
  ```

* **(wire, local segment) → arclength.** A wire's `NS` segments divide it into
  `NS` equal parts. Local segment `k` occupies the arclength interval
  `[(k−1)·L/NS, k·L/NS]` and its **centre** — arclength `(k − ½)·L/NS` metres
  from the wire's `p1` — is the point a feed, a load or a port lands on.

Addressing is shared: `EX`, `LD`, `IS` and `PT` all use it, and all of them
mean the same segment when they name the same pair.

## Connections

NEC connects **segments whose ends coincide**; the grouping of segments into
`GW` wires is irrelevant to it. The dialect reproduces that, in two passes over
the parsed geometry, before anything else is derived from it.

The tolerance is NEC's own: an L1 (Manhattan) separation no greater than
`1e-3` times the connecting segment's length. For a pair of segments of
different lengths the parser takes the **minimum** of the two — the
conservative intersection, which never connects what a NEC build would leave
open, at the cost of missing pairs in the narrow band between `1e-3·min` and
`1e-3·max`.

1. **Endpoint clustering.** Wire endpoints within tolerance are unioned and
   rewritten to one exact shared coordinate.
2. **Endpoint-onto-wire snapping.** A remaining endpoint within tolerance of
   another wire's *interior segment boundary* is snapped onto it, computed
   with the same `a + (b − a)·k/NS` formula the mesh uses. That is how NEC
   connects a wire end running into the middle of another wire, and it becomes
   a cut in the mesh so the crossing carries current.

Both passes move endpoints by at most 0.1 % of one segment — far below solver
noise. Gaps wider than the tolerance are left open, so deliberately gapped
geometry is unaffected.

Snapping is not cosmetic. Momwire junctions wire *ends* on near-exact
coordinate matches, so a deck whose corners sit a micrometre apart — the
routine outcome of `GM`-rotating six-significant-digit card coordinates —
would otherwise solve as a broken electrical graph, with current pinned to
zero at every unmatched end.

## The DeckModel

`parse()` returns a dialect-neutral model. Its vocabulary is momwire's, not
NEC's:

| field | meaning |
|---|---|
| `wires` | ordered polylines: vertices in metres, conductor radius, element count per edge, and an optional per-wire material (conductivity, insulation jacket) |
| `feeds` | `(wire, arclength, volts)` — a delta gap at a point on a wire, complex volts |
| `node_gaps` | gaps at a wire **knot** rather than mid-segment. The `nec2` dialect emits none; the seam exists because a NEC-5 dialect's edge sources are exactly this |
| `loads` | `(wire, arclength, impedance)` — series or parallel RLC, or a fixed complex `Z`, stamped at the same kind of point a feed occupies |
| `ground` | `None` (free space), `"pec"`, or `(model, eps_r, sigma)` with `model` one of `"finite-fast"` / `"finite"`; plus the ground plane's `z` and a cliff's second medium. The deck's **last** environment — see below |
| `environment` | the same three values as one record, per execute group: the ground, its plane and the second medium in force **at that group's execute card** |
| `frequencies` | the frequency list, in MHz, per execute group |
| `requests` | what each execute group asks for: nothing (a plain solve), a far-field pattern, a near-field grid, and the print controls that shape the readout |
| `networks` | the deck's [`TL`](#tl--transmission-line) / [`NT`](#nt--two-port-network) cards in card order: which card, both endpoints as `(wire, arclength)`, the six real fields **verbatim**, and the first execute group each is live for |
| `comments` | the deck's free text, in card order |

The model carries no tags, no segment numbers, no mnemonics and no card
ordinals. Anything a consumer needs in those terms — a printout that echoes
cards, a table addressed by segment — is the *dialect's* business and travels
alongside the model, not inside it.

`networks` is the one deliberate exception, and it is worth naming as one. A
network record keeps the card's own `(tag, segment)` pair **alongside** its
resolved endpoint, and its mnemonic alongside its fields, because the
`NETWORK DATA` block is addressed in exactly those terms and a deck whose tags
repeat cannot have them recovered from an arclength. Nothing in the physics
reads them. The endpoints are resolved the way every other address in this
model is, and the six real fields are recorded *uninterpreted*: what they mean
is card semantics, which belongs to whoever composes the network rather than
to the reader that found where it attaches.

### The environment is per execute group

`GN` **arms** (see [Arming](#arming)), so a deck may run once in free space
and once over ground, and each group is answered over whatever the cards had
reached by the time its execute card fired. The environment is therefore a
per-group quantity, carried on the execute group beside its frequency list and
its kernel flag.

The model's deck-level `ground`, `ground_z` and `second_medium` are the
environment the cards had reached at the **end** of the deck — the last
group's, and for the overwhelming majority of decks, which state their ground
once before the first execute card, every group's. A consumer that solves a
group reads the group's; a consumer that asks "what ground is this deck over"
reads the deck's, and gets the same answer whenever there is only one.

`build_solver(model, group=k)` builds over group `k`'s environment, and takes
an `environment=` override beside its `frequency_mhz=` and `extended_kernel=`
ones — the three operating-point choices a swept caller varies over one
geometry.

## Execution

A deck is a small state machine. Cards accumulate state; **execute cards**
(`XQ`, `RP`, `NE`, `NH`) run the structure with whatever is in force and
produce a group's worth of output.

### Arming

An execute card runs only if something has changed since the last one. The
**arming** cards are `EX`, `FR`, `LD`, `GN`, `EK`, `TL` and `NT` — the cards
that move the operator, the drive, or the network composed with them. A second
execute card with nothing new between it and the first produces no output at
all.

Explicitly **not** arming: `GD`, `MP` and `PT`. `GD` moves nothing outside the
far field's cliff modes; `MP` is advisory; `PT` changes what a run prints, not
what it computes. None of them can turn a bare `XQ` into a fresh run.

The first execute card of a deck always runs.

#### What a re-armed group rebuilds

Two of the seven arming cards move the **operator** — the matrix itself,
rather than the drive it is solved against. Those are `GN` and `EK`: a ground
card changes the half-space the fill runs over, a kernel card changes how the
fill integrates. The other five do not: `EX` moves the drive, `FR` moves the
frequency list, `LD` is stamped outside the fill, and `TL`/`NT` are composed
with the solved matrix rather than entering it.

So a re-armed group reports one of three shapes:

| between two execute cards | the group reports |
|---|---|
| a fresh `FR` (with or without anything else) | a whole refill — new frequency list, new operator |
| `GN` or `EK`, and no fresh `FR` | a **partial** refill: the operator was rebuilt, the frequency list was not |
| `EX`, `LD`, `TL` or `NT` only | neither — the operator is untouched |

The network row is oracle-verified: a deck that writes `FR`, `TL`, `XQ`, `NT`,
`XQ` prints two `ANTENNA INPUT PARAMETERS` blocks against exactly one each of
`MATRIX TIMING`, `FREQUENCY` and `STRUCTURE IMPEDANCE LOADING`. The second run
is real and the matrix behind it is the first run's.

The first execute card of a deck always reports a whole refill.

The test is the **card**, not the value it carries: NEC rebuilds because a
ground or kernel card arrived, so an `EK` naming the kernel already in force
refills exactly as a change does.

### Frequency groups

An execute card that follows a fresh `FR` runs **the whole frequency list**.
An execute card with no new `FR` since the previous one runs at the **last**
frequency of the list only. That is what makes a sweep a sweep and a re-run a
re-run.

### Excitation retention

NEC **retains** the excitation across an execute card. Concretely:

* the first `EX` after an execution **replaces** the source list;
* every further `EX` before the next execution **adds** to it;
* an execute card with no `EX` since the previous one re-drives the previous
  set unchanged.

So a deck may drive one segment, `XQ`, drive another, `XQ` — three groups,
three source sets — or `XQ` twice under one `EX` and get the same excitation
twice.

A deck with no `EX` card anywhere is refused:

```text
deck has no EX card — nothing drives the structure
```

### One geometry, one port set

Every `EX` segment across every group becomes a **port** in one union set, in
discovery order, and the whole deck is solved as one structure with one gap
per port. A group is then a voltage vector over a port set that never changes:
an undriven, unloaded port has `V = 0` and `Z = 0`, which collapses to a
shorted gap — present in the matrix, invisible to the physics.

`LD` segments join the same port set. A series load is NEC's semantics
exactly, an impedance in the segment's current path, which in port algebra is
`V_source = V_gap + Z·I`, hence `V_gap = (1 + Z·Y)⁻¹·V_source` and
`I = Y·V_gap`. The load is stamped outside the solver; the fill does not know
about it.

---

## Cards

The rest of this page is one section per card, in deck order: geometry, then
ground, then excitation and frequency, then the requests. Every field a card
carries is either named or declared ignored.

## GW — straight wire

One straight wire, divided into equal segments.

| field | meaning |
|---|---|
| `I1` | tag |
| `I2` | `NS`, segment count. Must be ≥ 1 |
| `F1 F2 F3` | end 1, metres |
| `F4 F5 F6` | end 2, metres |
| `F7` | conductor radius, metres. Must be > 0 |

Nothing is ignored. The wire becomes a polyline of two vertices with `NS`
elements between them; its radius is carried per wire, with no averaging
across the structure.

A non-positive radius is NEC's announcement of a tapered wire continued by a
`GC` card, which is out of dialect:

```text
segment count must be >= 1, got <n>
GW with a non-positive radius announces a tapered wire (GW + GC continuation), which is not part of this engine's nec2 dialect
```

## GM — move and replicate

Rotate, translate, and optionally replicate a block of wires.

| field | meaning |
|---|---|
| `I1` | tag increment applied to each replica |
| `I2` | `NRPT`, replica count. `0` transforms in place |
| `F1 F2 F3` | rotation about X, then Y, then Z, degrees |
| `F4 F5 F6` | translation, metres |
| `F7` | `ITS` — the tag whose first wire starts the affected block. `0` or below means the whole structure |

The block is every wire from the first one carrying tag `ITS` to the end of
the list. `ITS` is read as a real and rounded, which is NEC's own reading.

With `NRPT = 0` the transform is applied to the block in place. With
`NRPT > 0` the block is copied `NRPT` times and **each repetition transforms
the previous copy**, so rotations and translations compound — that is how one
loop side replicates into a polygon, or one bay into a stack. A wire with tag
`0` keeps tag `0` in every replica; every other tag gains `I1` per repetition.

The rotation is applied before the translation, in the order X, Y, Z.

```text
no wire has tag <t>
```

## GX — structure reflection

Reflect the whole structure built so far in one, two or three coordinate
planes.

| field | meaning |
|---|---|
| `I1` | tag increment applied to each image |
| `I2` | a three-digit plane code, `XYZ`: `ix = (I2/100) % 10`, `iy = (I2/10) % 10`, `iz = I2 % 10`. Any nonzero digit selects that plane |

So `GX 2 100` reflects in X=0, `GX 1 010` in Y=0, `GX 1 110` in both X=0 and
Y=0, and `GX 1 111` in all three.

**The planes fire in the order Z=0, then Y=0, then X=0** — NEC's own order,
not the order the digits are written in. Each reflection that fires copies
the **entire structure defined so far**, negating one coordinate; a wire
tagged `0` keeps tag `0`, and every other tag gains the increment.

**The tag increment doubles after each reflection that fires**, so every
image keeps a tag of its own. One wire tagged `1` under `GX 1 111` becomes
eight wires tagged `1` through `8`: the Z reflection adds 1, the Y reflection
adds 2 to both wires, and the X reflection adds 4 to all four.

A wire that **lies in** a firing plane (both ends on it) or **crosses** it
(the two ends on opposite sides) is a geometry error, not a silent no-op —
its image would land on top of it or inside it:

```text
a wire lies in or crosses the <X=0|Y=0|Z=0> symmetry plane
```

A wire **touching** a plane at one end is legal and common: that is how a
symmetric loop, a V or an inverted-L is built.

The test is read per **wire**. NEC reads the same expression per *segment*,
which differs on exactly one deck shape: a wire crossing the plane at a
segment boundary, which NEC accepts and then duplicates on top of itself. The
structure that produces is degenerate, so this dialect refuses it.

### The symmetric cell

`GX` does more than replicate geometry. It declares the structure symmetric,
so NEC fills and factors the matrix on **one cell** and reuses it — the
`SEGMENTS IN A SYMMETRIC CELL` line in its printout. The cell is the
structure exactly as it stood when the card fired, and its images follow it
contiguously; there is no hierarchy, and a `GX` selecting two planes still
reports one cell and four copies of it. A second `GX`/[`GR`](#gr--cylindrical-structure-rotation)
resets the cell to the whole prior structure rather than nesting inside the
first one's.

Only `GX` and `GR` create symmetry. What the other geometry cards do to it:

| card after `GX`/`GR` | symmetry |
|---|---|
| `GS`, any form | **kept** |
| `GM` over the whole structure (`ITS` unset, `NRPT = 0`) | **kept** |
| `GM` restricted to a tag, or with `NRPT > 0` | destroyed |
| `GW` | destroyed |
| `GX` selecting no plane at all (`I2 = 0`) | destroyed |
| `GR` again, any `nop` — including `nop = 1` | **reset**, not destroyed |

Symmetry survives a congruence of the *entire* structure and dies the moment
anything is added or transformed selectively. The typical real deck builds a
symmetric sub-assembly and then adds a feed wire or a mast, which collapses
it.

The geometry this engine builds is the fully expanded structure either way —
identical, wire for wire and segment for segment, to writing the images out
as `GW` cards by hand. The cell matters only for the cards that move the
**matrix**.

#### The cell rule

While symmetry is live the matrix is the **cell's**, so NEC reads a load
against the cell and stamps the result onto every copy afterwards. Two rules,
each verified on nec2c and on nec5cl to every printed digit (momwire#415):

| the load addresses | NEC | this dialect |
|---|---|---|
| a segment **inside** the cell | applies it to the corresponding segment of **every copy** | **served** — the load is expanded onto every copy |
| a segment **outside** the cell (a copy) | **silently drops it**, with no diagnostic, while still echoing the card in its `DATA CARD` list | **refused by name** |

Replication is served because it is the faithful answer and the one the
author meant. Putting the load where it was written instead is not an
approximation: on `k9ay_orig` the cell load also lands on the driven image,
and the deck reads 960.79 Ω rather than 490.79 Ω on the oracle — 53 % apart.
A deck written this way and the same deck with the reflection expanded into
`GW` cards and one `LD` per copy produce **the same model**: the same wires,
the same feed, the same load list, term for term.

The drop is refused because NEC's silence is a defect rather than a
behaviour worth matching — a card the user wrote is discarded and reported as
honoured. The message names what NEC does, so a reader cross-checking against
it is not surprised:

```text
LD addresses <n> segment(s) outside the GX/GR symmetric cell in force (the cell is segments 1-<c> of <t>), which this engine does not serve (momwire#415): while the symmetry is live the matrix is the cell's, so NEC silently drops such a card rather than loading the copy — address the cell instead, where a load applies to every copy of it, or write the GX/GR out as explicit GW cards
```

It costs one real deck. `1MHz_tower` in the xnec2c examples writes one `LD`
per tower leg — tags 3, 6, 9 and 12 under a `GR 3 4` whose cell is tags 1-3 —
so three of its four cards are dropped, and dropped onto exactly the segments
the surviving card is then replicated onto. nec2c prints the same impedance
for that deck as for the one-card form, to every digit; the drop is real and
invisible, and this engine refuses instead of choosing silently between two
readings the printout cannot tell apart. Both forms are committed as fixtures
(`tests/fixtures/nec2_symmetry/1MHz_tower*.deck`).

The rule reads the **segments** a card resolves to, not the way its tag field
spelled them. Three consequences:

- **Whole-structure addressing coincides with the cell.** Under a live
  symmetry the whole structure *is* cell plus copies, so `LD ... 0 0 0` and a
  card on the cell's own tag name the same set — which is exactly what the
  oracle prints (tag 0 and tag 1 are byte-identical there). It is served, and
  the replication does not stamp the copies a second time.
- **A global segment range is classified like a tag.** `LD 4 0 1 1 ...` and
  `LD 4 1 1 1 ...` are the same card when global segment 1 is the cell's
  first segment, and both replicate.
- **`LD 5` conductivity is cell-scoped too.** It is read under the `LD`
  mnemonic and lands in the same matrix diagonal. Its whole-structure form
  (`LD 5 0 0 0 <σ>`) already names every copy and is served directly.

The [≤ 8-segment expansion limit](#ld--loading) counts the range the deck
**typed**, not the total after replication: NEC's own reader never sees the
replicas as a range — they are stamped on one segment at a time, by a rule
that has no width. So `LD 4 1 1 8 ...` under a `GR 1 4` serves as thirty-two
loads, and it is the ninth *typed* segment that refuses. The
[doubled-load refusal](#ld--loading) is unaffected in the other direction:
replication widens the "already loaded" set rather than weakening it, so a
deck that genuinely loads one cell segment through two cards still refuses.

**`EX` is exempt.** Excitation is the right-hand side, not the operator, and
an asymmetric drive is exact under symmetry decomposition (even/odd for a
plane, Fourier modes for an n-fold rotation) — the oracle's cell-driven probe
matches an expanded twin carrying a single source to six figures.
`PT`/`PQ`/`GD`/`FR`/`GN`/`EK` address no tag into the structure and are
unaffected.

**`TL` and `NT` are exempt too**, for the same reason one step further out: a
network is composed with the *solved* matrix rather than being a term inside
it, so there is nothing for the cell to replicate. A network card attaches
**once, exactly as addressed**, and its endpoints resolve against the fully
generated structure — **image tags included**, which is precisely the address
an `LD` is refused for.

Measured, not argued. The eight probe decks
`tests/fixtures/nec2_symmetry/k9ay_{nt,tl}_*.deck` ask the question four ways
per card, on a k9ay-shaped structure with a live `GX 2 100` at `GE`:

| form | the card | nec2c (`NT` / `TL`) |
|---|---|---|
| *(no network card)* | — | 0.10161 + j514.86 |
| `..._cell` | one card, cell tags, `GX` live | 57.233 + j501.94 / 0.098596 + j577.58 |
| `..._naive` | the same one card over the hand-expanded structure | **identical**, to every printed digit |
| `..._expanded` | one card **per copy** — what a cell rule would mean | 109.75 + j492.48 / 0.094537 + j636.49 |
| `..._copy` | one card on the **image** tags, `GX` live | 55.036 + j497.78 / 0.096328 + j573.61 |

The cell form and the naive form are not merely close: their whole network
printout blocks are byte-identical. The replicated form is a measurably
different antenna, so replicating a network card would be wrong rather than
merely redundant. And the image-tag form resolves, attaches and answers — no
drop, no diagnostic, no error — which is where a network's addressing parts
company with a load's.

The practical consequence is that a symmetric deck with network cards needs no
special handling at all: the cross-boundary transmission lines real decks
write under a live `GX` are well defined exactly as typed.

**`IS` refuses under a live cell.** A sheath moves the matrix the way a load
does, so the cell rule must apply to it — but `IS` is this dialect's own card,
absent from NEC-2, so there is no oracle to measure that rule against the way
`LD`'s was measured. It refuses rather than guessing, which costs nothing: no
corpus deck pairs an `IS` with a live cell.

```text
IS while a GX/GR symmetric cell is in force is not supported by this engine (momwire#415): a sheath moves the matrix the way a load does, so the cell rule applies to it, but NEC-2 has no IS card to measure that rule against — write the GX/GR out as explicit GW cards
```

A deck whose symmetry is dead at `GE` — the great majority, the usual feed
wire or mast after the replication — never meets any of this: ordinary
per-tag addressing comes back untouched. Of the 36 xnec2c example decks that
write a `GX` or a `GR`, only four reach `GE` with a cell still live
(`40m-moxon`, `70cm_collinear`, `k9ay_orig`, `1MHz_tower`), and only the last
two of those also carry an `LD`. The rule is one check against
`symmetry`, so it reads identically whichever card, `GX` or `GR`, declared
the live cell.

## GR — cylindrical structure rotation

Rotate the whole structure built so far about the **Z axis**, forming a
cylindrical structure of `nop` copies.

| field | meaning |
|---|---|
| `I1` | tag increment applied to each copy |
| `I2` | `nop`, the total number of structures (the original plus `nop - 1` copies). Must be ≥ 1 |

The structure is copied `nop - 1` times, each copy rotated `2π / nop` radians
further about Z than the one before it, and **each copy is built from the
previous copy's coordinates** — not the original structure's — so the
rotations compound around the cylinder. A wire tagged `0` keeps tag `0` in
every copy.

**The tag increment does not double, the way `GX`'s does — it accumulates.**
Each copy adds `I1` to the *previous* copy's tag, so the `k`-th copy (the
original counts as copy 0) carries tag `original + k * I1`:

```text
GW 1 5  0. 0. 0.  1. 0. 0.  1.E-3
GR 1 4
```

lays out as one cell and three copies, contiguously — segments 1–5 the cell
(tag 1), 6–10 the first copy (tag 2), 11–15 the second (tag 3), 16–20 the
third (tag 4).

`I1 = 0` is legal and common: every copy then shares the original's tag, and
a tag-addressed card resolves across **all** of them, in structure order —
`GW 1 5 ...` followed by `GR 0 4` addresses tag 1 as all 20 segments, not
just the first 5.

`nop < 1` is a geometry error:

```text
structure count must be >= 1, got <n>
```

`nop = 1` is legal and produces no copies at all — the structure is
unchanged. It still **declares a symmetric cell**, though: NEC's own reader
sends every `GR` card through the same routine as `GX`, and that routine sets
the symmetry flag before it ever looks at `nop`, so even a `GR n 1` reports a
cell equal to the (unchanged) whole structure rather than leaving symmetry
untouched. This dialect follows that reading rather than treating `nop = 1`
as a no-op for the cell, the same way `GX n 0` deliberately *retires*
symmetry instead of falling through — see [the symmetric
cell](#the-symmetric-cell) above, which `GR` declares exactly as `GX` does,
including [the cell rule](#the-cell-rule) that governs `LD` while it is live.

## GS — scale

Multiply every coordinate and radius by a factor.

| field | meaning |
|---|---|
| `I1 I2` | tag range. The range applies only when `I1 > 0` and `I2 ≥ I1`, and then scales only wires whose tag falls in `[I1, I2]`; any other pair scales the whole structure |
| `F1` | scale factor. Must be > 0 |

The ranged form is an xnec2c extension, honoured here. Scaling touches
endpoints and radii; segment counts are unchanged.

```text
scale factor must be > 0, got <f>
```

## GE — end of geometry

Terminates the geometry section.

| field | meaning |
|---|---|
| `I1` | ground-plane flag: `0` = no ground plane, anything else = ground plane present |

`I2` and every real field are ignored.

**`GE` does not specify a ground.** The flag records that the structure sits
over a plane. The ground's *physics* comes from [`GN`](#gn--ground-parameters)
alone. A deck with `GE -1` and no `GN` solves in free space; a deck with
`GE 0` and `GN 1` solves over perfect ground. Both spellings occur in the
corpus, and both are honoured as written.

**The flag's sign is printed, not solved.** A positive `GE` is NEC's request
for the ground-contact current expansion, and the printout says so twice: the
`WHERE WIRE ENDS TOUCH GROUND…` banner under the structure annotation
(printed for every positive flag, contact or not — oracle-measured), and a
ground-touching segment's connection column, which carries the segment's own
number where a free end would carry `0`. This engine reproduces both lines
byte-for-byte. The *solve* consults neither: a wire end on the plane is
interpolated into its image under either sign, where NEC answers `GE -1`
contact differently (momwire#489 tracks the divergence).

## GN — ground parameters

The ground model and its constants.

| field | meaning |
|---|---|
| `I1` | ground type — see the table below |
| `I2` | `NRADL`, radial ground-screen wire count. Must be `0` for the reflection-coefficient grounds (`GN 0`, `GN 2`); `GN 1` and `GN -1` never read it (NEC's own reader checks it only on those two types — oracle-verified) |
| `F1` | `EPSR`, relative dielectric constant of medium 1 |
| `F2` | `SIG`, conductivity of medium 1, mhos/metre |
| `F3 F4 F5 F6` | a second medium, when `NRADL = 0` — see below |

| `I1` | ground | momwire |
|---|---|---|
| `-1` | free space; nullifies an earlier `GN` | none |
| `0` | finite ground, reflection-coefficient approximation | `("finite-fast", eps_r, sigma)` |
| `1` | perfectly conducting ground | `"pec"` |
| `2` | finite ground, Sommerfeld/Norton solution | `("finite", eps_r, sigma)` |

`GN -1` and `GN 1` take no parameters; `F1`/`F2` are read but move nothing.

The ground plane is at `z = 0`.

`GN` **arms** the next execute card, and a `GN` between two execute cards
rebuilds the operator without a new `FR`. So a deck may run once in free space
and once over ground: each execute group carries the environment in force at
its own execute card — see [The environment is per execute
group](#the-environment-is-per-execute-group).

**The second medium on a `GN` card.** When `NRADL` is zero, `F3`–`F6` carry a
whole second medium — the same four values a [`GD`](#gd--additional-ground-medium)
card sets, written into the same four slots by NEC's own card reader. A deck
can therefore state a complete cliff without ever sending a `GD`. A `GN` that
reaches the parser always rewrites those four slots, so a bare `GN 1` clears
an earlier cliff.

**Radial screens are refused.** NEC models a screen as a surface impedance
folded into the reflection coefficient; momwire has no screen model, so
ignoring the field would silently change the physics.

```text
GN <type> with a <n>-wire radial ground screen is not supported by this engine
GN type <type> is not supported by this engine
```

**`GN 0` decks whose geometry touches the plane are refused, by design.** A
`GN 0` ground plus a `GW` with an endpoint at `z = 0` — the ground-mounted
vertical, the most-imported deck of its class — builds a solver that raises

```text
wire <i> start lies in the ground plane: ground CONTACT under
ground_model='refl-coef' is refused (momwire#282 ...)
```

The reflection-coefficient model is a plane-wave construction evaluated on a
specular ray, and a contact node is its own mirror image, so at zero clearance
there is no ray to evaluate it on. This is not a momwire limitation to route
around: NEC-2 itself prints `175 − 779j Ω` on such a deck over average soil
and `155 − 1248j Ω` over poor, against `39 + 22j Ω` from the same binary over
`GN 1`. momwire served the same class of deck until momwire#282 stage 1 and
was ~27 Ω from its own Sommerfeld answer doing it.

Change the deck's `GN 0` to a [`GN 2`](#gn--ground-parameters) — NEC's
Sommerfeld/Norton solution, which momwire serves at contact and gates against
a reference engine there — or lift the geometry clear of the plane. `GN 1`
(perfect ground) at contact is unaffected and always has been.

## GD — additional ground medium

A second ground medium and the edge where medium 1 stops.

| field | meaning |
|---|---|
| `F1` | `EPSR2`, relative dielectric constant of medium 2 |
| `F2` | `SIG2`, conductivity of medium 2, mhos/metre |
| `F3` | `CLT`, distance from the origin to the edge where the media join |
| `F4` | `CHT`, the drop from medium 1's surface to medium 2's, signed — **positive means the far side is lower** (the extra image path `2·CHT·cosθ` enters as a phase lag; the sign is isolated by the `mininec_vertical_rp3_ch` fixture) |

`I1`–`I4`, `F5` and `F6` are ignored. A bare `GD` runs a deck exactly as a
fully populated one does.

**`GD` changes nothing outside the far field.** The moment method never sees
the second medium: every impedance and every segment current is the
flat-ground one. It reaches the answer only through
[`RP`](#rp--radiation-pattern) modes 2 and 3, where it selects the reflection
coefficient per direction. Under `RP 0` a deck with a `GD` and the same deck
without it produce identical patterns.

A later `GD` (or a later `GN` with `NRADL = 0`) overwrites an earlier one.
`GD` does not arm the next execute card. Like the ground, the second medium is
carried per execute group: the cliff a pattern reads is the one in force when
its own execute card fired.

**A second medium under a perfect ground is served as written — which is the
MININEC-type ground idiom.** `GD` alongside a [`GN 1`](#gn--ground-parameters)
is how both frontends spell MININEC-type ground: 4nec2 manufactures `GN 1` +
`GD 0 0 0 0 <eps> <sigma>` from its own `GN 3`, and EZNEC writes the `GD` with
the media payload. NEC-2 has no such ground *type* — it has the split, and
this engine reproduces it exactly as measured on the oracle (momwire#487;
decision record `docs/design/mininec-ground-idiom.md`):

* **Currents and impedance are perfect-ground, always.** That is what the
  idiom is *for* — 4nec2's own manual: with a wire on a real ground the
  reported impedance "is usually unpredictable", so MiniNec ground considers
  it perfect, "as could be the case using a nearly perfect buried radial
  system".
* **Under `RP 0` or a request-less execute, the pattern is perfect-ground
  too.** The record is carried and never consulted — byte-measured on the
  oracle and confirmed against 4nec2's own bundled NEC-2D engine: `PERFECT
  GROUND` banner, the `GD` echo the card's only trace (fixtures
  `mininec_gp80_seam` — 4nec2's emitted deck verbatim — and
  `mininec_vertical_rp0`/`mininec_vertical_gd2_rp0`).
* **The finite-media far field is behind `RP 2`/`RP 3`.** A cliff whose edge
  distance and height are zero puts the second medium under every reflected
  ray while the currents stay perfect-ground — which is precisely what
  4nec2's manual says `GN 3` becomes ("a second circular cliff ground-medium
  … distance zero … height of zero"), and what the `mininec_vertical_rp3_cliff_at_zero`
  fixture pins at 0.030 dB against the oracle.

The validity envelope is the idiom's own, stated by the host that invented
the manufacturing: no horizontal wires below 0.2 λ — which in practice
restricts it to verticals. And the idiom keeps the perfect-ground feed
impedance where a Sommerfeld ground (`GN 2`) answers a very different one
(41 + 24j vs 75 − 86j on a grounded quarter wave over average soil): the two
grounds disagree by design, not by defect.

## FR — frequency

| field | meaning |
|---|---|
| `I1` | sweep type: `0` = linear (additive), anything else = multiplicative |
| `I2` | `NFRQ`, number of frequencies. Values below 1 read as 1 |
| `F1` | starting frequency, MHz |
| `F2` | step: an increment in MHz for a linear sweep, a ratio for a multiplicative one |

`I3`, `I4` and `F3`–`F6` are ignored.

The list is `F1 + i·F2` for `i` in `0…NFRQ−1` (linear), or `F1 · F2^i`
(multiplicative). A multiplicative sweep whose ratio is zero or negative
degenerates to `NFRQ` copies of `F1`.

An `FR` arms the next execute card, and it also decides how much of the list
that card runs — see [frequency groups](#frequency-groups). Every `FR` in the
deck is read; a later one replaces the list entirely.

A deck that executes without ever sending an `FR` runs at NEC's own default,
**299.8 MHz** (`CVEL` — the frequency at which a wavelength is one metre;
oracle-verified). No corpus deck relies on this, but the rule is NEC's, not
an invention.

## EX — voltage source

An applied-field (delta-gap) voltage source on one segment.

| field | meaning |
|---|---|
| `I1` | excitation type. **Must be `0`** |
| `I2` | tag |
| `I3` | segment |
| `F1` | real part of the source voltage |
| `F2` | imaginary part of the source voltage |

`I4` (NEC's print-control flag) and `F3`–`F6` are ignored.

The source is placed at the centre of the addressed segment — see
[Addressing](#addressing) — as a delta gap in momwire's port model. Voltages
are complex and apply at readout: the port solution's columns are per volt, so
two decks alike but for their `EX` voltages share one fill.

Several `EX` cards may drive several segments in one group; the same segment
may be driven in different groups, at different voltages. Retention across
execute cards is described under [Excitation retention](#excitation-retention).

Every excitation type but `0` is refused. Types 1–3 are plane-wave
illumination — a scattering run, not a driven antenna. Type 4 is an elementary
current source and type 5 a current-slope-discontinuity source; type 6 is
4nec2's current source. None of them is a delta-gap voltage, and this dialect
drives voltage only.

```text
EX type <t> is not a voltage source; this engine drives EX 0 only
```

## LD — loading

Lumped and distributed loading on a segment range.

| field | meaning |
|---|---|
| `I1` | load type — see the table below |
| `I2` | tag |
| `I3` | first segment of the range |
| `I4` | last segment of the range |
| `F1 F2 F3` | type-dependent |

`F4`–`F6` are ignored.

`I3 = 0` means **every segment of the tag**, and `I4` is then ignored
entirely — it is not an upper bound (NEC's own reader takes this branch
before it looks at `I4`; oracle-verified). With `I2 = 0` too, the range is
the whole structure. The same rule serves [`IS`](#is--insulated-sheath).

| `I1` | meaning | `F1 F2 F3` | status |
|---|---|---|---|
| `-1` | nullify every load read so far | — | runs |
| `0` | series RLC | `R` Ω, `L` H, `C` F | runs |
| `1` | parallel RLC | `R` Ω, `L` H, `C` F | runs |
| `2` | series RLC per metre | — | refused |
| `3` | parallel RLC per metre | — | refused |
| `4` | fixed impedance `R + jX` | `R` Ω, `X` Ω | runs |
| `5` | wire conductivity | `σ` mhos/metre | runs |
| `6` | 4nec2 LC trap | — | refused |
| `7` | 4nec2 insulated wire | — | refused |

Zero-valued loads are no-ops and are dropped. `LD -1` clears the whole list —
and *only* loads: it does not touch [`IS`](#is--insulated-sheath) insulation,
which is a wire property here, not a load.

**Types 0, 1 and 4** expand over the segment range into one port per segment,
each carrying its own impedance. The load is stamped as a port impedance
outside the solver, so a loaded segment costs a gap in the fill and nothing
else. A range wider than **8 segments** is refused rather than silently
truncated.

**Type 5** is a material property, not a lumped element. The whole-structure
form (`I2 = 0`, `I3 = 0`) sets the conductivity of every wire; a ranged form
sets it per wire, and the range must cover each touched wire **in full** —
per-wire specs cover whole wires, and a partial range would otherwise silently
model a lossless wire where the deck asked for a lossy one.

**Types 2 and 3** are per-metre distributed loading, and **6 and 7** are 4nec2
extensions NEC-2 itself rejects. All four are refused rather than echoed and
dropped.

```text
LD type <t> is not supported by this engine
LD over <n> segments is not supported by this engine — at most 8 segments expand into per-segment loads
LD 5 conductivity on a partial-wire segment range is not supported by this engine — per-wire conductivity covers whole wires only
LD on a segment that already carries a load is not supported by this engine — a second load on one segment is not merged
```

While a [`GX`/`GR` symmetric cell](#the-symmetric-cell) is live, an `LD` is
read against the **cell** rather than against the segments it names: see
[the cell rule](#the-cell-rule), which governs the whole card, types 0, 1, 4
and 5 alike, and which the 8-segment limit and the doubled-load refusal above
both interact with.

## IS — insulated sheath

A dielectric jacket over a wire.

| field | meaning |
|---|---|
| `I2` | tag |
| `I3` | first segment of the range |
| `I4` | last segment of the range |
| `F1` | jacket relative permittivity |
| `F2` | jacket conductivity. **Must be `0`** |
| `F3` | jacket outer radius, metres |

`I1` and `F4`–`F6` are ignored.

The jacket is a **lossless dielectric** — a King quasi-static `L'` correction
on the wire — carried per wire, so the range must cover each touched wire in
full, exactly as a ranged `LD 5` must. A jacket with `F3 ≤ 0` or `F1 ≤ 1` is
electrically a vacuum and is dropped as a no-op; a jacket whose outer radius
does not clear the conductor is a deck error and refuses.

`IS` is **structural**: one geometry, one set of per-wire specs, every group.
It must therefore appear before the first execute card. It does not arm the
next execute card, because a deck that changes it between runs is refused
outright.

Where an unmodellable `IS` could plausibly be dropped with a warning, it is
refused instead: solving a bare wire where the deck asked for a jacket is a
wrong answer, not a translation compromise. An `IS` under a live
[`GX`/`GR` symmetric cell](#the-symmetric-cell) is refused for the same
reason — [the cell rule](#the-cell-rule) has no NEC-2 oracle for this card.

```text
IS after an execute request is not supported by this engine: wire insulation is part of the structure, so it cannot change between runs
IS with a conductive sheath (F2 != 0) is not modelled by this engine — the insulation jacket is a lossless dielectric; set the sheath conductivity to 0
IS: insulation on a partial-wire segment range — per-wire specs cover whole wires only
IS: insulation whose outer radius does not exceed the wire's conductor radius
```

## TL — transmission line

An ideal two-wire transmission line between two segments of the structure,
with an optional shunt admittance across each end.

| field | meaning |
|---|---|
| `I1` | tag of end one |
| `I2` | segment of end one |
| `I3` | tag of end two |
| `I4` | segment of end two |
| `F1` | characteristic impedance, ohms. **Must be nonzero**; a negative value selects a [crossed line](#crossed-lines) |
| `F2` | line length, metres. `0` means [the distance between the two segments](#zero-length-lines) |
| `F3`, `F4` | shunt admittance across end one, `G + jB` mhos |
| `F5`, `F6` | shunt admittance across end two, `G + jB` mhos |

The line is lossless and its velocity factor is 1: `F2` is an *electrical*
length in metres, and a deck modelling a velocity factor scales it before
writing the card. Both ends attach at the **centre** of the addressed segment,
the same point an [`EX`](#ex--voltage-source) drives — see
[Addressing](#addressing).

Both endpoints of one card may name the same segment, and several cards may
share an endpoint; nothing here is one-per-segment.

### Crossed lines

A negative `F1` is NEC's spelling for a line whose conductors are transposed
between the two ends — a half-turn of the pair, which inverts the sign of the
voltage carried across it. The magnitude is the impedance; the sign is a
topology flag and nothing else. nec2c prints the distinction in its network
block's `LINE TYPE` column as `STRAIGHT` or `CROSSED`.

This dialect keeps the two apart at the model level: the card's fields are
recorded as written, and the reading that matters — that this is a *positive*
impedance with a transposed polarity, never a negative impedance — is the
semantics the network solve applies. A negated impedance would be a different
and unphysical line.

### Zero-length lines

`F2 = 0` does not mean a zero-length line. It means "as long as the gap it
spans": NEC substitutes the **straight-line distance between the two segment
midpoints**, which is what makes a card written between two nearby feedpoints
model the stub that is physically there. A deck wanting a genuinely
zero-length connection writes a small nonzero length or an
[`NT`](#nt--two-port-network).

### End shunts

`F3`–`F6` are two admittances in mhos, one across each end, in parallel with
the line's own port. They are how NEC spells an open or shorted stub, a
loading coil at a feedpoint, or the resistor of a terminated line. Zero — the
usual case, and the value a short card supplies — is no shunt at all rather
than a short circuit.

A `TL` with `F1 = 0` is refused. NEC does not run it either: the oracle aborts
while *reading* the deck.

```text
TL with a zero characteristic impedance is not a transmission line — NEC aborts reading the deck on this card; Z0 must be nonzero, and its SIGN, not its magnitude, is what selects a crossed line
```

## NT — two-port network

An explicit two-port admittance matrix between two segments of the structure —
NEC's general network card, and the one every circuit that is not a
transmission line arrives as.

| field | meaning |
|---|---|
| `I1` | tag of end one |
| `I2` | segment of end one |
| `I3` | tag of end two |
| `I4` | segment of end two |
| `F1`, `F2` | `Y11`, real and imaginary, mhos |
| `F3`, `F4` | `Y12`, real and imaginary, mhos |
| `F5`, `F6` | `Y22`, real and imaginary, mhos |

**`Y21 = Y12` by construction.** The card carries three entries, not four:
NEC's network is reciprocal by definition and there is no field in which to
say otherwise. A deck needing a non-reciprocal two-port cannot say so in this
grammar.

Addressing is [`TL`](#tl--transmission-line)'s exactly — segment centres, the
same points `EX` drives.

### An all-zero `NT` is not a no-op

An `NT` whose six real fields are all zero is a **real card with a real
effect**, and this is worth stating because two other readings are tempting
and both are wrong. A zero-valued [`LD`](#ld--loading) *is* dropped as a no-op
by this dialect; antennaknobs' importer skips an all-zero `NT` as
unmodellable. NEC does neither. It attaches a network of zero admittance,
which **open-circuits both addressed segments**: their currents collapse to
numerical zero and the structure is cut at those two points.

Measured on a probe whose control answers 0.10161 + j514.86: with an all-zero
`NT` across two of its segments the same deck answers 0.68923 − j4651.8, with
both connection-point currents at ~1e-20. The card is read as written and the
zero admittance is honoured.

## Network addressing, retention and ordering

The rules below are `TL`'s and `NT`'s in common — they are one card as far as
NEC's reader is concerned, and a deck may mix them freely.

### Segment numbers must be positive

Unlike every other addressed card in this dialect, a network endpoint's
segment number may not be zero or negative. NEC does not clamp it, ignore it
or treat it as "the whole tag": it **halts the entire run**, printing

```text
CHECK DATA, PARAMETER SPECIFYING SEGMENT POSITION IN A GROUP OF EQUAL TAGS MUST NOT BE ZERO
```

The check is on the segment field alone and runs before the tag is looked at,
so `NT 0 0 ...` — where the segment number would be a global index rather than
a position within a group of equal tags — halts exactly as `NT 1 -3 ...` does.
This dialect refuses in kind rather than resolving something NEC would not:

```text
<M> addresses segment <s> of tag <t>, which is not a segment: NEC halts on this card with CHECK DATA, PARAMETER SPECIFYING SEGMENT POSITION IN A GROUP OF EQUAL TAGS MUST NOT BE ZERO — a network endpoint names one segment, so its segment number must be 1 or more
```

No corpus deck writes one: 0 of the 46 hand-written `TL`/`NT` decks surveyed
use a nonpositive endpoint.

### Network retention

Network cards are **retained across execute cards**, the way an
[`EX` set is](#excitation-retention) and unlike a frequency list. A deck that
states its networks once and then runs several groups is answered with the
same networks in every one — nec2c prints the identical network block in each.

Retention runs forward from the card and not backward: an execute card that
has already run is not retroactively given a network stated after it. A deck
that runs the bare antenna and then attaches a line gets two different
answers, in that order, which is what the oracle prints.

### Network contiguity

This is the one place where NEC's own behaviour is a defect this dialect
refuses to reproduce.

NEC keeps a single list of network cards and **resets it** whenever it reads a
network card whose predecessor was not one. The effect is that a card of the
wrong kind sitting between two network cards **silently destroys every network
card before it** — while still echoing those cards in the `DATA CARD` list as
though they had been read and honoured. Nothing in the printout says the line
is gone. The impedance is simply the impedance of a different antenna.

The class was measured card by card, each probe written as `TL`, *card*, `NT`
and read back off the network block:

| interposed card | the earlier `TL` |
|---|---|
| `PT`, `PQ`, `MP` | **survives** — the block is byte-identical to the contiguous control |
| `LD`, `EX`, `FR`, `GN`, `EK`, `GD` | destroyed, in silence |
| `XQ`, `RP`, `NE`, `NH` | destroyed, in silence |

The transparent cards are exactly the ones that change what a run *reports*
rather than what it computes, which is why serving them matters: SimNEC
appends an `MP` on structure size alone, without being asked.

This dialect serves contiguous network groups and refuses the destroy pattern
by name. It costs nothing measured — 0 of the 46 hand-written network decks
surveyed hit it, and the `EX 6` forms 4nec2 manufactures are contiguous by
construction — and the alternative is answering a question the deck did not
ask.

```text
<M> with an interposed <C> card between it and an earlier network card is not supported by this engine (momwire#456): NEC silently DESTROYS every TL/NT read before such a card, so this deck's earlier network cards would vanish with no diagnostic while still being echoed in its DATA CARD list as read — keep a deck's TL/NT cards contiguous (only PT, PQ and MP may sit between them)
```

`<C>` is the **first** interposed card, which is the deck's actual mistake.

A deck with a single network card never meets this rule: the pattern is about
a card destroyed by a later one, so what follows a lone `TL` is irrelevant.

### Networks under a symmetric cell

Exempt, measured, and set out under [the cell rule](#the-cell-rule): a network
card attaches once, exactly as addressed, and its endpoints resolve against
the fully generated structure including the image tags a `GX`/`GR` created.

### Printout ordering

A deck's networks reach the printout twice, in two blocks that order their
rows on different rules. Both were measured against the oracle rather than
assumed, because both look like card order and neither is.

**`NETWORK DATA` groups by row KIND.** The block carries one banner and then
one column header per kind of row beneath it, and the header re-emits when the
kind changes. The kind of the *first* card decides which group prints first;
within a group the cards keep their deck order. So a deck reading `TL`, `NT`,
`TL` prints both lines under one header and then the admittance matrix — the
`NT` last, though it was written second. A deck reading `NT`, `TL` prints the
matrix first. A `STRAIGHT` line followed by a `CROSSED` one is one table, not
two: `STRAIGHT`/`CROSSED` is a column, not a kind.

Addressing in this block is `(tag, global segment)`, so `TL 1 5 2 5` on a
structure whose first wire has nine segments prints as `1 5 2 14`. A `TL`'s
`LENGTH` column shows the **resolved** length, which is where a
[zero-length card](#zero-length-lines)'s substituted distance becomes visible.

**`STRUCTURE EXCITATION DATA AT NETWORK CONNECTION POINTS` partitions by
source.** Every distinct connection segment appears once — a card's end one
before its end two, cards in deck order, a segment named twice named once —
but the ones carrying no `EX` are printed *before* the ones that do. A deck
driving the first endpoint of its only card therefore prints that row second.

The rows report what the STRUCTURE took at each connection point: the voltage
the circuit put across the gap and the current the antenna drew through it.
At a driven connection point that current is **not** the one
[`ANTENNA INPUT PARAMETERS`](#ex--voltage-source) prints on the same segment —
that row is the source current, antenna plus network — and the difference
between the two rows is what the network carried.

## EK — extended thin-wire kernel

| field | meaning |
|---|---|
| `I1` | `-1` selects the standard kernel; **every other value, and a bare `EK`, selects the extended kernel** |

`I2`–`I4` and every real field are ignored.

The test is `I1 == −1` and nothing else. `EK`, `EK 0`, `EK 1`, `EK 2` and
`EK -2` all turn the extended kernel **on**; only `EK -1` turns it off. That
is NEC's own card reader, and matching it matters: refusing a card the
reference engine accepts turns a runnable deck into a fabricated readout.

The flag is honoured, not advisory — it selects the operator, the same `O(a²)`
on-axis tube expansion the card asks for. It is carried **per execute group**,
so one deck may answer two groups under two kernels with two fills.

`EK` arms the next execute card. An `EK` between two execute cards rebuilds
the operator without a new `FR` — a partial refill, the same shape a `GN`
there produces (see [What a re-armed group
rebuilds](#what-a-re-armed-group-rebuilds)).

On `--basis sinusoidal-galerkin`, `EK` is REFUSED at solver construction if
the deck has a junction where two `GW` sections carry different radii —
measured divergent there, not merely inaccurate (momwire#398 D2). Every
other basis serves `EK` on a stepped-radius deck; see [Choosing the
physics](/reference/portal-usage/#choosing-the-physics---basis) for the
row-by-row guidance on tapered and stepped-radius wire.

## XQ — execute

Run the structure with whatever is in force. `I1` and every other field are
ignored.

`XQ` produces a solve and its impedance/current readout. An `XQ` with nothing
new since the previous execute card produces no output at all — see
[Arming](#arming).

## RP — radiation pattern

A far-field pattern. `RP` is an **execute** card: it runs the pending group and
then reports.

| field | meaning |
|---|---|
| `I1` | mode. `0`, `2` and `3` run; `1`, `4`, `5` and `6` are refused |
| `I2` | `NTH`, number of theta values. Below 1 reads as 1 |
| `I3` | `NPH`, number of phi values. Below 1 reads as 1 |
| `F1` | `THETS`, initial theta, degrees |
| `F2` | `PHIS`, initial phi, degrees |
| `F3` | `DTH`, theta increment, degrees |
| `F4` | `DPH`, phi increment, degrees |
| `F5` | `RFLD`, range in metres. `0` selects the gain-only form |

`I4` (`XNDA`, output-format control) and `F6` (`GNOR`, gain normalisation) are
ignored.

| mode | asks for | here |
|---|---|---|
| `0` | space wave over the ground plane | runs |
| `1` | surface wave — a different banner, a different row shape, reached through a different field routine | refused |
| `2` | linear cliff: second medium beyond `x = CLT` | runs |
| `3` | circular cliff: second medium beyond `r = CLT` | runs |
| `4` | radial-wire ground screen | refused |
| `5` | screen inside, linear cliff beyond it | refused |
| `6` | screen inside, circular cliff beyond it | refused |

Modes 4–6 are refused for the reason `GN`'s `NRADL` is: the screen is a
surface impedance momwire has no model for, and running the deck as bare
ground would be a wrong answer rather than a refusal. Mode 1 is refused
because nothing here computes a surface wave.

Modes 2 and 3 read the [`GD`](#gd--additional-ground-medium) record — the only
path by which a second medium reaches any number.

`RFLD = 0` is the **gain-only** form: the field columns carry the far-field
amplitude itself rather than the field at a range. The gain columns never
depended on the range and do not move.

```text
RP mode <m> is not supported by this engine (modes 0, 2, 3 only)
```

## NE — near electric field

A rectangular grid of near-field E samples. `NE` is an **execute** card.

| field | meaning |
|---|---|
| `I1` | coordinate system. **Must be `0`** (rectangular) |
| `I2 I3 I4` | sample counts along X, Y, Z. Below 1 reads as 1 |
| `F1 F2 F3` | grid origin, metres |
| `F4 F5 F6` | step along X, Y, Z, metres |

Samples vary X fastest, then Y, then Z.

Over free space and over perfect ground the near field is computed from the
element currents and, for `"pec"`, its image. Over a **finite** ground it is
refused: the near field of a lossy half-space is not an image, and a
reflection coefficient is a far-field construction.

```text
NE coordinate system <c> (spherical) is not supported by this engine; rectangular (0) only
NE over a finite ground is not supported by this engine (the near field of a Sommerfeld half-space is not an image)
```

## NH — near magnetic field

Identical to [`NE`](#ne--near-electric-field) in every field and every
restriction; it reports H instead of E. Its refusals carry the mnemonic `NH`.

## PT — current print control

Which segment-current rows a run reports. `PT` changes what a run prints, not
what it computes, so it never arms an execute card.

| field | meaning |
|---|---|
| `I1` | flag — see below |
| `I2` | tag |
| `I3` | first segment of the range |
| `I4` | last segment of the range |

| `I1` | effect |
|---|---|
| `-1` | suppress the segment-current report entirely |
| `-2` | restore it |
| `0` | print only segments `I3…I4` of tag `I2`, addressed exactly as an `EX` addresses a segment. An all-zero range means "no restriction" |
| `1`, `2`, `3` | no restriction — the ordinary full report |

`PT` is a **toggle**, not a per-run flag: it holds across execute cards until
another `PT` moves it. One deck may therefore suppress the first run's report
and restore the second's.

## PQ — charge print control

Whether a run prints a charge-density report. `PQ` changes what a run
prints, not what it computes, so — like `PT` — it never arms an execute
card.

| field | meaning |
|---|---|
| `I1` | flag (`IPTFLQ`) — see below |
| `I2` | tag |
| `I3` | first segment of the range |
| `I4` | last segment of the range |

| `I1` | effect |
|---|---|
| negative (`-1` default) | suppress the charge-density report |
| `0` or greater | print a charge-density report, optionally restricted to `I2`–`I4` |

momwire's printout has no charge-density report to suppress, so the
suppression form is a no-op: `PQ -1` (and any other negative flag) parses and
changes nothing. A nonnegative flag is a *request* for the report this
engine does not produce, and refuses:

```text
PQ <n> requests a charge-density report this engine does not produce; PQ -1 (suppress) is the only form served
```

## MP — multiprocessing hint

A matrix-fill parallelism hint. Read, recorded, and then ignored.

| field | meaning |
|---|---|
| `I1` | processor count. Must be an integer |
| `I2` | block size. Must be an integer |

`MP` is emitted automatically by structure size, not by user request, so
refusing it would fail exactly the decks worth running. Honouring it is
neither possible nor correct: it describes how a *reference* engine fills and
factors its matrix, it moves no printed number, and momwire's parallelism is
decided once per process by the BLAS/OpenMP pools behind numpy and scipy —
which snapshot their environment at import time, long before a card arrives.

The card is advisory, and this dialect says so: it is recorded per execute
group (so a consumer can echo it in the right block) and changes no answer.
It does not arm.

A fractional field is an error, matching NEC's integer-field reader:

```text
MP field <k> must be an integer, not <v>
```

## NX — end of deck

The frame terminator. It has no fields, ends the current deck, and begins the
next. Its echo is the sentinel a resident caller blocks on, so it is emitted
on **every** path — including a refused deck.

## EN — end of run

Terminates a deck exactly as `NX` does, and additionally ends the run. A
standalone `.nec` file ending in `EN` therefore solves and exits. End of input
is the same terminator: a deck whose last card is an execute card, with no `EN`
written, solves and exits identically.

## CM / CE — comments

Free text. `CM` is a comment line; `CE` is conventionally the last one. The
text is carried into the model in card order.

Two directives are recognised inside the comment text, both consumed by the
parser rather than passed through:

* **`QQ n`** with `n > 0` — quiet: the caller may drop the bulkiest report
  section.
* **`FF n`** — reduced field: `n` is reported once, out of band.

Anything else in a comment is text.

---

## Refusals

Every refusal names the card or field responsible. Nothing in this dialect is
accepted and silently ignored: a card that would change the physics either
runs or refuses. `parse()` raises with the message; a caller frames it.

### Cards refused by name

| card | message |
|---|---|
| `GA` | `GA (wire arc) is not part of this engine's nec2 dialect, whose geometry is GW with GM / GS transforms` |
| `GH` | `GH (helix) is not part of this engine's nec2 dialect, whose geometry is GW with GM / GS transforms` |
| `GC` | `GC (tapered wire continuation) is not part of this engine's nec2 dialect` |
| `GF` | `GF (numerical Green's function) is not part of this engine's nec2 dialect` |
| `SY` | `SY (4nec2 symbolic variables) is not part of this engine's nec2 dialect` |
| `SP` | `SP (surface patch) is not supported by this engine yet` |
| `SM` | `SM (multiple-patch surface) is not supported by this engine yet` |
| `SC` | `SC (surface patch continuation) is not supported by this engine yet` |
| `KH` | `KH (interaction approximation limit) is not supported by this engine` |
| `CP` | `CP (coupling request) is not supported by this engine` |
| `PL` | `PL (plot request) is not supported by this engine` |
| `WG` | `WG (NGF write request) is not supported by this engine` |
| `ZO` | `ZO (impedance normalisation) is not supported by this engine` |

`GA`/`GH` are refused rather than translated because the corpus never uses
them and an untested geometry path is worse than an honest no.
([`GX`](#gx--structure-reflection) and [`GR`](#gr--cylindrical-structure-rotation) were
two of them and are now built.) The three "yet" messages mark the cards a
patch model would bring, not a dialect decision.

`TL` and `NT` were on this table and have left it: the dialect reads them, and
what they refuse is [by value](#fields-refused-by-value) — a nonpositive
endpoint segment, a zero characteristic impedance, and the non-contiguous
destroy pattern.

Anything else:

```text
unrecognised NEC card '<XX>'
```

### Fields refused by value

| card | condition | message |
|---|---|---|
| `EX` | `I1 ≠ 0` | `EX type <t> is not a voltage source; this engine drives EX 0 only` |
| `RP` | `I1 ∉ {0, 2, 3}` | `RP mode <m> is not supported by this engine (modes 0, 2, 3 only)` |
| `NE` / `NH` | `I1 ≠ 0` | `<M> coordinate system <c> (spherical) is not supported by this engine; rectangular (0) only` |
| `NE` / `NH` | finite ground | `<M> over a finite ground is not supported by this engine (the near field of a Sommerfeld half-space is not an image)` |
| `GN` | `NRADL ≠ 0` on `GN 0` / `GN 2` (the only types that read it) | `GN <type> with a <n>-wire radial ground screen is not supported by this engine` |
| `GN` | `I1 ∉ {-1, 0, 1, 2}` | `GN type <type> is not supported by this engine` |
| `LD` | `I1 ∈ {2, 3, 6, 7}` or unknown | `LD type <t> is not supported by this engine` |
| `LD` | range > 8 segments | `LD over <n> segments is not supported by this engine — at most 8 segments expand into per-segment loads` |
| `LD 5` | partial-wire range | `LD 5 conductivity on a partial-wire segment range is not supported by this engine — per-wire conductivity covers whole wires only` |
| `LD` | segment already loaded | `LD on a segment that already carries a load is not supported by this engine — a second load on one segment is not merged` |
| `IS` | after an execute card | `IS after an execute request is not supported by this engine: wire insulation is part of the structure, so it cannot change between runs` |
| `IS` | `F2 ≠ 0` | `IS with a conductive sheath (F2 != 0) is not modelled by this engine — the insulation jacket is a lossless dielectric; set the sheath conductivity to 0` |
| `IS` | partial-wire range | `IS: insulation on a partial-wire segment range — per-wire specs cover whole wires only` |
| `IS` | jacket inside the conductor | `IS: insulation whose outer radius does not exceed the wire's conductor radius` |
| `PQ` | `I1 >= 0` | `PQ <n> requests a charge-density report this engine does not produce; PQ -1 (suppress) is the only form served` |
| `MP` | fractional field | `MP field <k> must be an integer, not <v>` |
| `GW` | `NS < 1` | `segment count must be >= 1, got <n>` |
| `GW` | radius ≤ 0 | `GW with a non-positive radius announces a tapered wire (GW + GC continuation), which is not part of this engine's nec2 dialect` |
| `GS` | factor ≤ 0 | `scale factor must be > 0, got <f>` |
| `GX` | a wire lying in or crossing a firing plane | `a wire lies in or crosses the <p> symmetry plane` |
| `GR` | `nop < 1` | `structure count must be >= 1, got <n>` |
| `LD` | a segment addressed outside a live `GX`/`GR` [cell](#the-cell-rule) | `LD addresses <n> segment(s) outside the GX/GR symmetric cell in force (the cell is segments 1-<c> of <t>), which this engine does not serve (momwire#415): while the symmetry is live the matrix is the cell's, so NEC silently drops such a card rather than loading the copy — address the cell instead, where a load applies to every copy of it, or write the GX/GR out as explicit GW cards` |
| `IS` | a `GX`/`GR` symmetric [cell](#the-cell-rule) still live | `IS while a GX/GR symmetric cell is in force is not supported by this engine (momwire#415): a sheath moves the matrix the way a load does, so the cell rule applies to it, but NEC-2 has no IS card to measure that rule against — write the GX/GR out as explicit GW cards` |
| `TL` / `NT` | either endpoint's [segment number](#segment-numbers-must-be-positive) `< 1`, with any tag | `<M> addresses segment <s> of tag <t>, which is not a segment: NEC halts on this card with CHECK DATA, PARAMETER SPECIFYING SEGMENT POSITION IN A GROUP OF EQUAL TAGS MUST NOT BE ZERO — a network endpoint names one segment, so its segment number must be 1 or more` |
| `TL` | `F1 = 0`, including a card that omits it | `TL with a zero characteristic impedance is not a transmission line — NEC aborts reading the deck on this card; Z0 must be nonzero, and its SIGN, not its magnitude, is what selects a crossed line` |
| `TL` / `NT` | a card of the [destroy class](#network-contiguity) between this card and an earlier network card | `<M> with an interposed <C> card between it and an earlier network card is not supported by this engine (momwire#456): NEC silently DESTROYS every TL/NT read before such a card, so this deck's earlier network cards would vanish with no diagnostic while still being echoed in its DATA CARD list as read — keep a deck's TL/NT cards contiguous (only PT, PQ and MP may sit between them)` |

### Structural refusals

```text
CARD'S MNEMONIC CODE TOO SHORT OR MISSING: <line>
NON-NUMERICAL CHARACTER IN FIELD: <token> on <line>
deck has no EX card — nothing drives the structure
segment <n> is out of range for tag <t>
segment <n> is out of range for the structure
no wire has tag <t>
```

---

## Appendix: differences from antennaknobs' importer

antennaknobs reads NEC decks too, with `nec_import.parse_nec`, and keeps the
**full** grammar — networks, 4nec2 extensions, NEC-5 spellings and all — for
its `.nec` import feature. Where the two readers disagree, this dialect
follows the portal, which is the engine-facing contract. The differences, for
anyone moving a deck between them:

| topic | antennaknobs importer | this dialect |
|---|---|---|
| `TL` / `NT` | translated into circuit branches | read, resolved and solved — both readers serve them |
| `TL` / `NT` non-contiguity | the cards are collected wherever they appear; NEC's destroy rule is not reproduced | [refused](#network-contiguity) rather than answered as a different antenna |
| all-zero `NT` | skipped as unmodellable, with a note in an "ignored" list | [read as written](#an-all-zero-nt-is-not-a-no-op) — the oracle open-circuits both segments, so skipping it is a wrong answer |
| `TL` / `NT` nonpositive segment | resolved | [refused](#segment-numbers-must-be-positive), as NEC halts on it |
| `GA` / `GH` | built as geometry | refused by name |
| `LD` under a live `GX`/`GR` symmetry | the geometry expands; the symmetric-cell load rule is not applied, so the load stays where it was written | [the cell rule](#the-cell-rule): a cell load is expanded onto every copy, and a copy-addressed load is refused rather than silently dropped |
| `IS` under a live `GX`/`GR` symmetry | applied where written | refused (momwire#415) — no NEC-2 oracle for the cell rule on this card |
| `GE` ground flag | `GE I1 ≠ 0` alone marks the deck as grounded | records the flag; ground physics comes from `GN` only |
| `FR` | only the **first** `FR` is read, and it collapses to a `(min, max)` range | every `FR` is read; the list drives execute groups |
| `EX` type | accepts `0`, `5` (voltage) and `6` (4nec2 current source, network path) | accepts `0` only |
| `EX` edge form | recognises NEC-5's segment-end source (negative segment, or `I4 = 2`) and imports it as a vertex port | refused as a non-zero excitation type; the seam for it is the model's `node_gaps`, reserved for a NEC-5 dialect |
| `EX` duplicates | a segment driven twice is an error | the port set is a union over groups, so one segment may be driven in several groups |
| `LD 6` / `LD 7` | translated (LC trap, insulated wire) | refused |
| `LD` over long ranges, doubled loads, partial-wire `LD 5` | skipped, with a note in an "ignored" list | refused |
| unmodellable `IS` | skipped, with a note | refused |
| `SY` | 4nec2 symbolic variables are evaluated | refused |
| `'` comments | a leading apostrophe comments out a line; an inline one truncates it | not recognised — an apostrophe is a mnemonic error |
| unhandled cards | a documented set (`KH`, `CP`, `PL`, `WG`, `ZO`, …) is silently ignored | refused by name |
| `GC` / `GF` | refused | refused |

The two readers agree on everything load-bearing for geometry: `GW`, `GM`,
`GS`, `GX`, free-format tolerance, fused mnemonics, `(tag, segment)`
addressing, and
the connection tolerance and snapping described under
[Connections](#connections) — that code is shared, not merely equivalent.
