synthetic_ld5_virtual.nec / synthetic_ld5_virtual.out -- momwire#1096

A SMALL deck of our own -- not a captured EZNEC session and not the
reporter's model -- in the shape EZNEC Pro/2+ 7.0.4 writes for an OCF
dipole with a transformer: three real wires, a far-away two-segment
VIRTUAL wire (#4) that the source and network hang on, pinned open by an
LD 4 1.E+10, and a whole-structure LD 5 whose range stops at the last REAL
segment (1..63 of 65). Run through our licensed NEC-5 materials for its
printout. Only the STRUCTURE IMPEDANCE LOADING table and the input
impedance are gated (tests/test_eznec_ld5_virtual_wires_1096.py); the rest
of the .out is context. Kept in its own directory so tests/fixtures/eznec/
and its manifest stay untouched.

Citation, same convention as tests/fixtures/eznec/manifest.json: the
printout was produced by NEC-5, (c) LLNL, LLNL-CODE-746721, a user-licensed
binary never distributed with this repository.
