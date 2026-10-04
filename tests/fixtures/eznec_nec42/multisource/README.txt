momwire#1295 - the phased drive: two decks EZNEC's External NEC-4.2 slot
writes with SEVERAL current sources, and the licensed NEC-4.2's printout of
each.

  m1_cardioid-2-current-sources.nec    EZNEC's stock Cardioid.ez, two EX 6
  m2_foursquare-4-current-sources.nec  a 40 m four-square, four EX 6 in
                                       quadrature over GN 1 + GD

Found 2026-10-04 at the 0.71.0 pre-tag Windows GUI check: both were written
by EZNEC Pro/2+ v7.0.4 with its External NEC-4.2 slot pointed straight at
the 0.71.0 release candidate's momwire-nec4.exe (not through the capture
shim, so there is no meta.tsv), and both came back refused - the reader then
admitted a current source only as a group's sole source.  The same
Cardioid.ez through the NEC-5 slot is capture 0032 in
tests/fixtures/eznec/decks/ (EX 4 at the wire end where this slot writes
EX 6 at segment 1's centre).  Recorded in antennaknobs PR #1883
(scratch/eznec-nec42-multisource/, commit 7276f890c).  Each deck is the
EZ.NEC the slot wrote, byte for byte, CRLF included (.gitattributes one
level up).

printouts/m1_*.out, printouts/m2_*.out (one level up, beside the captures')
  Licensed NEC-4.2 (LLNL-CODE-491368), the 2014 LLNL NEC42W64CL.exe on the
  Windows box, run 2026-10-04 on the identical decks, black box: deck in,
  printout out, byte for byte (CRLF, as that run wrote them).  Committed as
  test fixtures with the licensee's permission.  OUTPUT only; no NEC-4.2
  source, build tree or binary is in this repository, and none may be.
  The source currents come back pinned at the cards' +/-1.41421 A.  The
  four-square's source 1 reads a NEGATIVE resistance (-1.38797 - j20.3048):
  a driven element of a phased array taking power from its neighbours,
  correct physics, and the row a clamp would break.
