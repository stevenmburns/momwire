synthetic_ex_endcode.nec / synthetic_ex_endcode.out -- momwire#1092

A SMALL deck of our own -- not a captured EZNEC session and not a
reporter's model -- written to exercise antennaknobs' own explicit-end
addressing (the fourth EX field) on an EX card, the same (segment, end)
pair momwire#1085 gave LD, run through our licensed NEC-5 materials to
get its printout. Kept in its own directory, separate from
tests/fixtures/eznec/, so it never touches that corpus's pinned 80-capture
count or its manifest.json (tests/test_eznec_reproducibility.py).

Three separate, non-interacting dipoles (spaced 10 m apart so each drives
its own column), one EX card each -- the "several EX cards" phased-array
shape the corpus already establishes, used here to gate three different
address spellings in one deck and one printout:

  EX 4,1,6,0,1.414214,0.   EZNEC's own spelling: node 6 of a 9-segment
                           wire. Baseline -- unaffected by momwire#1092,
                           included so the gate shows it stays byte-exact.
  EX 4,2,7,1,1.414214,0.   antennaknobs' explicit end code: segment 7,
                           end 1 -> node 6 (the same node as the baseline,
                           reached a different way).
  EX 4,3,5,2,1.414214,0.   antennaknobs' explicit end code: segment 5,
                           end 2 -> node 5.

Only the ANTENNA INPUT PARAMETERS table's TAG / SEG. / end-digit columns
(the first 12 characters of each row) are gated
(tests/test_eznec_ex_endcode_1092.py) -- the voltage/current/impedance
cells are a SOLVE, and momwire's own bspline solver reads those same three
decks to different numbers than the licensed engine's own basis by design
(a cross-solver difference, not a format one); the row ADDRESSING is what
this fixture measures. Verified against our licensed materials:
segment 7 end 1 and segment 5 end 2 both decode to a node one of momwire's
OWN feed cards could reach through EZNEC's sign-of-the-node-field
spelling, yet the licensed engine prints a DIFFERENT SEG. number and a
DIFFERENT trailing digit for each -- SEG. is the field as written (not the
decoded node, the same rule momwire#1085 measured for LD's loading table)
and the trailing digit is the INVERSE of the end code (end 1 prints 2,
end 2 prints 1).

invvee_apex.nec / invvee_apex.out -- the issue's own example (AK#898's
true-vertex inverted vee, antennaknobs.designs.dipoles.invvee_apex), at
the design's stock defaults, exported through antennaknobs'
NEC5Engine.deck() with no ground and no far-field request. Its one EX
card, `EX 0 1 1 1 1.000000E+00 0.000000E+00`, is exactly the shape the
issue names: a `p0` vertex feed, segment 1 end 1 of the fed arm, which
decodes to node 0 -- the arms' shared apex -- only once momwire#1092's
fix reads the fourth field; before it, this deck's EX card silently reads
as node 1, one knot in from the apex. Used by
tests/test_eznec_ex_endcode_1092.py's cross-engine gate: momwire's own
`razor-nec5` solve of this export (through `momwire.eznec._serve.serve`)
against the licensed engine's ANTENNA INPUT PARAMETERS row here, and
against the SAME geometry reconstructed BY HAND from this file's own two
`GW` cards and solved through momwire's own `node_gaps` port (#305's
series node gap -- the primitive `PortAtVertex` compiles to) instead of
an `EX` card. No antennaknobs import at test time either way
(momwire#988: a test gated behind `importorskip("antennaknobs")` runs in
no CI lane anywhere).

Citation, same convention as tests/fixtures/eznec/manifest.json and
tests/fixtures/eznec_ld01_1085/README.txt: both printouts were produced
by NEC-5, (c) LLNL, LLNL-CODE-746721, a user-licensed binary never
distributed with this repository. Only synthetic_ex_endcode.out's
ANTENNA INPUT PARAMETERS address columns and invvee_apex.out's single
ANTENNA INPUT PARAMETERS row (impedance included, as a physics gate to a
stated tolerance rather than a byte gate) are compared against
(tests/test_eznec_ex_endcode_1092.py); the rest of either file is kept
for context and is not otherwise compared.
