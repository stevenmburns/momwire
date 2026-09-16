synthetic_ld23.nec / synthetic_ld23.out -- momwire#1088

A SMALL deck of our own -- not a captured EZNEC session -- written to
exercise LD 2 (series RLC per unit length) and LD 3 (parallel RLC per unit
length) on the per-wire spelling, and run through our licensed NEC-5
materials to get its printout. Two wires so that both card types appear in
one loading table without either landing on the other's wire. Kept in its
own directory, separate from tests/fixtures/eznec/, so it never touches
that corpus's pinned 80-capture count or its manifest.json
(tests/test_eznec_reproducibility.py) -- the same rule
tests/fixtures/eznec_ld5_1082/ follows.

The CAPACITANCE field is 0. on both cards on purpose: momwire refuses a
nonzero one (see momwire.deck._nec2._Nec2Parser._ld23 for the measurement),
so a fixture carrying one could not be served and would not gate anything.

Citation, same convention as tests/fixtures/eznec/manifest.json: the
printout was produced by NEC-5, (c) LLNL, LLNL-CODE-746721, a user-licensed
binary never distributed with this repository. Only the STRUCTURE IMPEDANCE
LOADING table's two per-metre rows are gated against it
(tests/test_deck_ld23_per_metre_1088.py); the rest of synthetic_ld23.out is
kept for context and is not otherwise compared.
