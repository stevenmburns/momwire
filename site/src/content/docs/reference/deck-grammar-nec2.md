---
title: "The nec2 deck dialect"
description: The normative card-by-card grammar of momwire's antenna-only NEC dialect — which cards run, which fields are read, which are ignored, and the exact text of every refusal.
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
NEC-2 that describes wire antennas and asks them questions. It is the first of
a planned family: a NEC-5 flavour is the probable second, and the seam is
built for it from day one.

Every dialect front-end parses into one dialect-neutral [`DeckModel`](#the-deckmodel),
and `build_solver(model, basis=...)` maps that model onto momwire's solver
families. A second dialect is therefore a second parser, not a second
pipeline — and nothing NEC-2-specific may leak into the model's vocabulary.
The model speaks wires, arclengths, feeds, gaps and grounds; tags, segments
and card mnemonics stop at the parser.

`dialect="nec2"` is the only value this release ships.

### What "antenna-only" excludes

The dialect describes **a structure of thin wires, driven by voltage sources,
optionally over a ground**. It does not describe circuits attached to that
structure. Transmission lines (`TL`) and two-port networks (`NT`) are refused
by name: solve the network outside the engine, or import the deck with
antennaknobs, which keeps NEC's full network grammar. Surface patches, arcs,
helices and structure reflections are refused too — see
[Refusals](#refusals) for each message.

The restriction is a measurement, not a preference. Across the 44-deck
reference corpus the card histogram is `GW` 104, `EX` 55, `XQ` 52, `FR` 49,
`NX` 45, `GE` 45, `CE` 45, `CM` 21, `GN` 16, `RP` 7, `LD` 7, `GD` 5, `PT` 3,
`EK` 3, `MP` 2, and one each of `NE`, `NH`, `GS`, `GM` — plus two `TL` and two
`NT`, confined to three hand-authored probe decks. The dialect covers what
real decks contain.

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
| `comments` | the deck's free text, in card order |

The model carries no tags, no segment numbers, no mnemonics and no card
ordinals. Anything a consumer needs in those terms — a printout that echoes
cards, a table addressed by segment — is the *dialect's* business and travels
alongside the model, not inside it.

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
**arming** cards are `EX`, `FR`, `LD`, `GN` and `EK` — the cards that move the
operator or the drive. A second execute card with nothing new between it and
the first produces no output at all.

Explicitly **not** arming: `GD`, `MP` and `PT`. `GD` moves nothing outside the
far field's cliff modes; `MP` is advisory; `PT` changes what a run prints, not
what it computes. None of them can turn a bare `XQ` into a fresh run.

The first execute card of a deck always runs.

#### What a re-armed group rebuilds

Two of the five arming cards move the **operator** — the matrix itself, rather
than the drive it is solved against. Those are `GN` and `EK`: a ground card
changes the half-space the fill runs over, a kernel card changes how the fill
integrates. The other three do not: `EX` moves the drive, `FR` moves the
frequency list, and `LD` is stamped outside the fill.

So a re-armed group reports one of three shapes:

| between two execute cards | the group reports |
|---|---|
| a fresh `FR` (with or without anything else) | a whole refill — new frequency list, new operator |
| `GN` or `EK`, and no fresh `FR` | a **partial** refill: the operator was rebuilt, the frequency list was not |
| `EX` or `LD` only | neither — the operator is untouched |

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
over a plane — it drives the annotation in the structure report, and nothing
else. The ground's *physics* comes from [`GN`](#gn--ground-parameters) alone.
A deck with `GE -1` and no `GN` solves in free space; a deck with `GE 0` and
`GN 1` solves over perfect ground. Both spellings occur in the corpus, and
both are honoured as written.

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
| `F4` | `CHT`, height of medium 2's surface relative to medium 1's, signed — negative means the far side is lower |

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
wrong answer, not a translation compromise.

```text
IS after an execute request is not supported by this engine: wire insulation is part of the structure, so it cannot change between runs
IS with a conductive sheath (F2 != 0) is not modelled by this engine — the insulation jacket is a lossless dielectric; set the sheath conductivity to 0
IS: insulation on a partial-wire segment range — per-wire specs cover whole wires only
IS: insulation whose outer radius does not exceed the wire's conductor radius
```

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
| `TL` | `TL (transmission line) is not part of this engine's nec2 dialect, which is antenna-only; antennaknobs imports decks with networks` |
| `NT` | `NT (two-port network) is not part of this engine's nec2 dialect, which is antenna-only; antennaknobs imports decks with networks` |
| `GA` | `GA (wire arc) is not part of this engine's nec2 dialect, whose geometry is GW with GM / GS transforms` |
| `GH` | `GH (helix) is not part of this engine's nec2 dialect, whose geometry is GW with GM / GS transforms` |
| `GX` | `GX (structure reflection) is not part of this engine's nec2 dialect, whose geometry is GW with GM / GS transforms` |
| `GR` | `GR (cylindrical structure rotation) is not part of this engine's nec2 dialect, whose geometry is GW with GM / GS transforms` |
| `GC` | `GC (tapered wire continuation) is not part of this engine's nec2 dialect` |
| `GF` | `GF (numerical Green's function) is not part of this engine's nec2 dialect` |
| `SY` | `SY (4nec2 symbolic variables) is not part of this engine's nec2 dialect` |
| `SP` | `SP (surface patch) is not supported by this engine yet` |
| `SM` | `SM (multiple-patch surface) is not supported by this engine yet` |
| `KH` | `KH (interaction approximation limit) is not supported by this engine` |
| `PQ` | `PQ (charge print control) is not supported by this engine` |
| `CP` | `CP (coupling request) is not supported by this engine` |
| `PL` | `PL (plot request) is not supported by this engine` |
| `WG` | `WG (NGF write request) is not supported by this engine` |
| `ZO` | `ZO (impedance normalisation) is not supported by this engine` |

`GA`/`GH`/`GX`/`GR` are refused rather than translated because the corpus
never uses them and an untested geometry path is worse than an honest no. The
two "yet" messages mark the cards a patch model would bring, not a dialect
decision.

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
| `MP` | fractional field | `MP field <k> must be an integer, not <v>` |
| `GW` | `NS < 1` | `segment count must be >= 1, got <n>` |
| `GW` | radius ≤ 0 | `GW with a non-positive radius announces a tapered wire (GW + GC continuation), which is not part of this engine's nec2 dialect` |
| `GS` | factor ≤ 0 | `scale factor must be > 0, got <f>` |

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
| `TL` / `NT` | translated into circuit branches | refused by name |
| `GA` / `GH` / `GX` / `GR` | built as geometry | refused by name |
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
| unhandled cards | a documented set (`KH`, `PQ`, `CP`, `PL`, `WG`, `ZO`, …) is silently ignored | refused by name |
| `GC` / `GF` | refused | refused |

The two readers agree on everything load-bearing for geometry: `GW`, `GM`,
`GS`, free-format tolerance, fused mnemonics, `(tag, segment)` addressing, and
the connection tolerance and snapping described under
[Connections](#connections) — that code is shared, not merely equivalent.
