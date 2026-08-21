# The seam rule: protocol fidelity, physics honesty

Design doc for a boundary that has been enforced since the first portal
shipped but never stated. Decision to write it down taken 2026-08-21, out of
a positioning discussion that deserved a durable answer: the campaign to
stand momwire in NEC-2's and NEC-5's external-engine slots had begun to read
— from the inside — as "match the incumbent's idiosyncrasies as closely as
we can," which is not what this project is for. The resolution is that the
tree already separates the two concerns cleanly; this document makes the
separation a rule with a name, so that every future card, quirk, and
divergence is decided by principle instead of by mood.

## The rule

> **The seam may reproduce any byte the host reads, and may never
> reproduce a wrong answer.**

One sentence, two halves, and every gate in the portal test suites enforces
one half or the other.

*Protocol fidelity*: a host application parses the emulated engine's output
with code we do not control. Every byte that host reads is interface, not
physics — layouts, headings, numeric spellings, banner text, line endings,
even the emulated engine's bookkeeping quirks. The seam reproduces those
bytes exactly, because a divergence there doesn't make anyone's antenna
model more truthful; it makes the host crash, misparse, or refuse the
engine.

*Physics honesty*: every number an operator reads as a statement about an
antenna — an impedance, a gain, a current — is momwire's own answer,
computed by momwire's own formulations. Where that answer differs from the
emulated engine's, the seam ships the difference as a named envelope, a
named gate, or a named refusal. It never narrows the difference by
imitation, and it never serves a number it believes is wrong because the
incumbent would have printed it.

## Why match anything at all

Distribution, not identity. The engines being emulated sit behind real
barriers — a license fee, a Windows-only binary, a host that speaks exactly
one dialect — and the hosts' external-engine slots are the door through
which their communities can meet a different engine without leaving the
tool they trust. The dialect is the doorway's shape. Matching it is the
price of admission; it is not the product. The product is the physics
underneath, and the moment the two conflict, the rule above says which one
bends: the bytes bend toward the host, the numbers never bend toward the
incumbent.

## The quarantine

The idiosyncrasy-matching is confined to the seam modules — the protocol
adapters:

- `momwire.eznec` (`_shell`, `_printout`, `_serve`): the NEC-5-dialect
  drop-in EZNEC Pro+ launches.
- `momwire.portal` + `momwire.deck`: the NEC-2-dialect resident engine
  SimNEC drives, and the deck grammars both seams parse.

Everything a host made us reproduce lives at that layer and cites a capture:
the exact banner and comment-box layout, CRLF line endings (LF printouts
made EZNEC refuse the engine outright — #512), the engine's own dielectric
spelling `εc = εr − j·σ·λ·59.96` measured black-box from its printouts, the
`200.00 PERCENT` and `161.38 PERCENT` budget quirks reproduced because the
host displays the field, and the `***** NEC ERROR` refusal frame the host's
error path expects. All of it derived
from captured input/output of our own licensed installations — conclusions
from observed bytes, never from anyone's source.

The rest of the tree — the solvers, the grounds, the network core, the
nine-engine roster — carries none of these fingerprints. A reader of
`momwire.bspline` cannot tell EZNEC exists.

Two corollaries the seam modules obey:

- **The seam leaves no droppings.** The real engine writes its Sommerfeld
  cache into the working directory; the seam never writes anything but the
  printout it was asked for. Reproducing the host-read bytes does not
  extend to reproducing side effects the host doesn't read.
- **The channel discipline is the host's, exactly.** Exit 0 always, the
  printout file is the only channel, stdin untouched, stderr unpinned
  (momwire#497 U1's three rules) — because that is what the host actually
  reads, and inventing a richer contract would be fidelity to nothing.

## The physics-honesty half, as practiced

The rule's second half has case law. Each entry is a moment where imitation
was available and refused:

- **Named divergence gates that fail if "fixed" silently.** momwire's
  Sommerfeld answer diverges from the licensed engine on radials at grazing
  height (the 0033/0034 finding, momwire#510: hundreds of ohms apart,
  X sign flipped, mesh-sensitive where the engine is not). The seam ships
  that as a test that *fails the day the numbers move* — forcing a
  re-measure and a decision — rather than a correction factor that would
  quietly track the incumbent.
- **Refuse-by-name instead of guessing.** Seven of the 122 corpus launches
  refuse today, every one with a sentence naming the card, the wire, or the
  missing capability, and where there is one, the remedy. A refusal is the
  contract that nothing is ever served silently wrong — and the refusal
  roster is itself gated, including a must-refuse fixture built from a
  once-accepted buggy geometry (#522).
- **When we disagree with the incumbent, we file against ourselves first —
  and adjudicate independently.** The coupled-loop study found our default
  B-spline lane apart from the licensed engine's limit; the response was a
  bug against our own default (#518) and an engine-free electrostatic
  referee to say which limit was right. The referee episode ended with the
  charge overturned — the gap was a geometry bug in the study's own ladder
  script, and all five formulations agree with the referee on the true
  geometry — but the *direction of suspicion* is the rule working: the
  incumbent is a reference to be adjudicated against, never an oracle to be
  matched.
- **The incumbent's defects are studied, not served.** momwire's
  formulation roster contains a point-matched sinusoidal solver that
  reproduces NEC-2's small-closed-loop pathology almost quantitatively —
  kept as a *study instrument*, because owning a twin of the defect is how
  you prove the mechanism. The served defaults are the clean formulations;
  a deck that would trip a 162-ampere-class defect in the emulated engine
  gets momwire's clean answer, loudly different, and that divergence is a
  feature with a writeup, not a compatibility bug.
- **Two lanes, both honest.** The everyday lane (`bspline`, the go-to
  engine) answers with momwire's best physics and a quantified envelope.
  The reference lanes (`razor-nec5` and siblings) exist to *measure*
  agreement with the incumbent's own convergence path — they are
  validation instruments in the roster, not a promise that the product
  converges the incumbent's way.

## The rubric for the next quirk

When a new card, field, or behavior arrives at a seam, ask one question:
**does the host read this byte?**

1. *Yes, the host parses or displays it* → reproduce it exactly, at the
   printout/protocol layer, gated by a capture of the real engine. Quirks
   included; the gate cites the capture, so the quirk is forever
   distinguishable from a belief.
2. *No — it is a number an operator will trust* → serve momwire's own
   physics. If it agrees with the reference, pin the envelope with a named
   tolerance. If it doesn't, ship a named divergence gate or a named
   refusal. There is no third option; "adjust toward the incumbent" is not
   in the table.
3. *It is a side effect the host never reads* (a cache file, an exit code,
   stderr) → don't reproduce it. Fidelity extends exactly as far as the
   host's parser.

The scoring that drives seam development follows the same line: coverage is
measured in *captures served or refused by name*, never in "how close did
we get to the incumbent's number" — closeness to the reference is evidence
about our physics, collected in envelopes, not a target to optimize.

## What this rule protects

The project's public claim is layered, and each layer depends on the rule:
*we measured everything; we agree with the best engine where it matters; we
say so out loud when we don't; and we built it to run inside the tool you
already use.* Break the first half of the rule and the hosts stop working.
Break the second half — once — and the third clause becomes false, which is
the clause the whole pitch stands on.
