momwire#1295 - the decks EZNEC's External NEC-4.2 slot writes.

Seventeen decks captured 2026-10-03 from EZNEC Pro+ v7.0.4 with its engine
option set to External NEC-4.2 and the engine path pointed at a capture-only
shim named momwire-nec4-capture.exe (antennaknobs scratch/eznec-capture/,
captures 0223-0239, commit b21c9ddd1).  Each file is the EZ.NEC the slot
wrote, byte for byte, CRLF included (see .gitattributes); the file name is
the capture id plus what the capture varied.

Every capture recorded the same launch:

  "C:\EZNEC 7.0\Docs\momwire-nec4-capture.exe" "EZ.NEC" "NEC.OUT"
  cwd = the engine's own folder, stdin not redirected, exit 0 accepted

Pairs that are the same deck: 0223/0236 (re-captured), 0225/0226.
0223/0224 are one antenna under EX 6 (current) and EX 0 (voltage).

No printout is kept here, and none may be: NEC-4.2 is licensed for own use at
home.  The licensed engine's impedances for these decks were taken as numbers
only (antennaknobs scratch/eznec-capture/NEC42-SOLVE-NOTES-2026-10-03.md) and
are quoted, with that provenance, in tests/test_eznec_nec42_1295.py.
