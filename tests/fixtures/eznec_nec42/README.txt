momwire#1295 - the decks EZNEC's External NEC-4.2 slot writes, and the
licensed NEC-4.2's printout of each.

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

probes/<probe>.nec
  Nine decks written for momwire#1295 phase 2 (not EZNEC captures), each to
  show NEC-4.2's layout of a shape the captures do not carry:
    p1  three wires at one node (junction table, connection columns)
    p2  an FR sweep of three frequencies under one RP
    p3  series and parallel RLC loads (LD 0, LD 1)
    p4  an NE near-field table
    p5  GD over GN 1 with no pattern (XQ)
    p6  an EX 0 voltage source at a network point
    p7  GN 2 at 7.15 MHz (the complex dielectric constant off 299.79 MHz)
    p8  two execute runs in one deck
    p9  wires above and below a GN 2 interface (GE -1)

printouts/<capture or probe>.out
  Licensed NEC-4.2 (LLNL-CODE-491368), nec42cl-serial release 7768648, run
  2026-10-03 on the build machine, black box: each deck above in, the
  engine's printout out, byte for byte (LF line endings, as that run wrote
  them; the probes' printouts were run on the laptop the same day).
  Committed as test fixtures with the licensee's permission.  They
  are OUTPUT only; no NEC-4.2 source, build tree or binary is in this
  repository, and none may be.  Line 2 is blank in every one (the engine
  prints no build tag), and the Sommerfeld-table cache messages in 0231,
  0232, 0239, p7 and p9 are the engine's own file handling, which the gates
  normalise away (tests/test_eznec_nec42_printout_1295.py).
