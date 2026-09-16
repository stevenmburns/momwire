synthetic_ld01.nec / synthetic_ld01.out -- momwire#1085

A SMALL deck of our own -- not a captured EZNEC session and not a
reporter's model -- written to exercise the STRUCTURE IMPEDANCE LOADING
table's RESISTANCE / INDUCTANCE / CAPACITANCE columns (LD 0, LD 1) and
antennaknobs' own explicit-end-code spelling of an LD address (LDTAGT 1 or
2, the same (segment, end) pair its NEC-5 writer uses for an EX card), run
through our licensed NEC-5 materials to get its printout. Kept in its own
directory, separate from tests/fixtures/eznec/, so it never touches that
corpus's pinned 80-capture count or its manifest.json
(tests/test_eznec_reproducibility.py).

Card by card:

  LD 4,1,1,1,10.,20.        fixed impedance, segment 1 end 1 -> node 0
                            (antennaknobs' explicit-end spelling of the
                            SAME node EZNEC would write as `LD 4,1,-1,0`)
  LD 0,1,5,1,50.,1.E-6,1.E-10   series RLC, segment 5 end 1 -> node 4
  LD 1,1,9,2,75.,2.E-6,5.E-11   parallel RLC, segment 9 end 2 -> node 9
  LD 5,0,1,11,5.8E+7,1.     whole-structure conductivity (momwire#1082's
                            own shape, included here to gate DECK ORDER:
                            the loading table's four rows print in the
                            order the cards were written, not grouped by
                            type -- measured on this exact deck, a fact
                            momwire#1082's own fixture never exercised
                            since it carries only one load card)

Citation, same convention as tests/fixtures/eznec/manifest.json and
tests/fixtures/eznec_ld5_1082/README.txt: the printout was produced by
NEC-5, (c) LLNL, LLNL-CODE-746721, a user-licensed binary never distributed
with this repository. Only the STRUCTURE IMPEDANCE LOADING table (all four
rows, and their ORDER) is gated against it
(tests/test_eznec_ld01_1085.py); the rest of synthetic_ld01.out is kept for
context and is not otherwise compared.
