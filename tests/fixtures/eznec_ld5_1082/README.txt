synthetic_ld5.nec / synthetic_ld5.out -- momwire#1082

A SMALL deck of our own -- not a captured EZNEC session and not the
reporter's model -- written to exercise LD 5 (wire conductivity) on the
whole-structure spelling, and run through our licensed NEC-5 materials to
get its printout. Kept in its own directory, separate from
tests/fixtures/eznec/, so it never touches that corpus's pinned 80-capture
count or its manifest.json (tests/test_eznec_reproducibility.py).

Citation, same convention as tests/fixtures/eznec/manifest.json: the
printout was produced by NEC-5, (c) LLNL, LLNL-CODE-746721, a user-licensed
binary never distributed with this repository. Only the STRUCTURE IMPEDANCE
LOADING table's WIRE row is gated against it (tests/test_eznec_ld5_conductivity_1082.py);
the rest of synthetic_ld5.out is kept for context and is not otherwise
compared.
