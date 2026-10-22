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

The shim's name is recorded as it was at capture.  The shipped NEC-4.2
launchers have been named momwire-nec42[-<basis>] since momwire#1443
(SimNEC recognises an engine by nec42 in its name); a name carrying only
nec4, like the shim's, still selects the NEC-4.2 slot.

Pairs that are the same deck: 0223/0236 (re-captured), 0225/0226.
0223/0224 are one antenna under EX 6 (current) and EX 0 (voltage).

probes/<probe>.nec
  Nine decks written for momwire#1295 phase 2 (not EZNEC captures), each to
  show NEC-4.2's layout of a shape the captures do not carry, and twenty-three
  more:
    p1  three wires at one node (junction table, connection columns)
    p2  an FR sweep of three frequencies under one RP
    p3  series and parallel RLC loads (LD 0, LD 1)
    p4  an NE near-field table
    p5  GD over GN 1 with no pattern (XQ)
    p6  an EX 0 voltage source at a network point
    p7  GN 2 at 7.15 MHz (the complex dielectric constant off 299.79 MHz)
    p8  two execute runs in one deck
    p9  wires above and below a GN 2 interface (GE -1)
    p10 a fat dipole, Delta/a = 1.5, written for momwire#1326 (not a phase 2
        probe): the deck on which the extended kernel, the nec4 slot's
        default, lands nearest NEC-4.2.  Its printout was run 2026-10-04 on
        the laptop by the same black-box ritual (binary invoked, nothing
        else read), and the deck is committed as NEC-4.2 read it (LF).
  p11-p15, written for momwire#1352 (EZNEC's NE 1 near field reached the
  slot): the p4 dipole with a spherical grid, NE 1,2,4,3 (R 2 values,
  theta 0-90 by 30, phi 0-90 by 45), and one control.  Run 2026-10-05 on the
  laptop by the same black-box ritual, each in a fresh directory, decks LF.
    p11 NE 1 in free space
    p12 NH 1 in free space (the magnetic twin)
    p13 NE 1 over GN 3, the dipole 1 m up
    p14 NE 0 over GN 3, the rectangular control for p13
    p15 NE 1 over GN 2, the same dipole
  What the printouts show (black box, numbers only): an NE 1 table is the
  rectangular table, the same heading, column headings, units and row
  format, one row per point, each row the point's Cartesian X, Y, Z in
  metres and the field's three Cartesian components.  The card's three
  counts, origins and steps are (R, theta, phi), R in metres and the angles
  in degrees, theta from the zenith and phi from +x; R walks fastest, then
  theta, then phi.  A point on an axis angle carries ~2.6e-14 of R where
  the exact value is zero (theta = 90 prints Z = 2.6485E-14 at R = 1).  This
  is the NEC-5 oracle's NE 1, point for point (momwire#1257).

  p16-p32, written for momwire#1463 (SimNEC's RP card carries a range, and
  a run may carry several RP / NE / NH cards): one antenna, a 10 m vertical
  dipole of 21 segments with its lower end 1 m up, fed at its centre at
  7.1 MHz, over GN 3 and GN 2 with SimNEC's NOFILE (13, 0.005 S/m), over
  GN 1 with and without GD, and in free space.  Run 2026-10-13 on the laptop
  by the same black-box ritual, each in a fresh directory, decks LF.
    p16 RP at a range of 1000 m over GN 3      p17 the same RP at no range
    p18 RP at a range in free space            p19 the same RP at no range
    p20 RP then NE over GN 3                   p23 the same in free space
    p21 NE then RP at a range over GN 3        p24 the same in free space
    p22 XQ, then RP, then NE over GN 3         p25 the same in free space
    p26 SimNEC's own RP card (RP 0 19 73 1001 0 0 5 5 1000), then NE 1
    p27 RP, RP at a range, NH, then XQ over GN 3
    p28 GN 1 + GD: RP at a range, RP at none, then NE
    p29 RP at a range over GN 1                p30 RP at a range over GN 2
    p31 RP through the horizon over GN 3, at no range and then at a range
    p32 the same over GN 1
  What the printouts show (black box, layout and printed numbers only):
    - Every RP, NE and NH card after the run prints its own table, in card
      order.  The first prints at the end of the run's block; each further
      one echoes its card four blank lines below the previous table (six
      below a pattern) and prints its table three blank lines under the
      echo, with no input parameters, currents or power budget of its own.
      An XQ with nothing new prints nothing but its echo (p27), and an XQ
      that ran the deck leaves both tables to the cards after it (p22).
    - Under GD every pattern is preceded by its FAR FIELD GROUND PARAMETERS
      block, a further one included (p28).
    - RP with a range: the heading gains RANGE= and EXP(-JKR)/R= lines and
      the field columns read VOLTS/M.  The gains, the axial ratio, the tilt,
      the sense and the average-gain trailer are the range-free printout's
      to every digit (p16 against p17, p18 against p19); each E magnitude is
      the range-free one over the range, and each E phase the range-free
      one plus the heading's phase, not wrapped again (p18 prints 293.42).
      The heading's phase is that of exp(-jkR) at the wavelength the
      printout prints, 299.8/f: 114.32 degrees at 7.1 MHz and 1000 m.
    - Over GN 2 and GN 3 the heading carries a LOWER MEDIUM line as well,
      exp(-jkR)/R at k*sqrt(eps_c) with eps_c the complex constant the
      environment block prints: 2.23389-107 (a three-digit exponent, the E
      dropped) at -164.12 degrees.  GN 1 prints no such line, with GD or
      without (p28, p29).
    - A row below the horizon over a ground is all zeros, at a range too;
      its two phases at a range are the LOWER MEDIUM's over GN 3 (p31) and
      0.00 over GN 1 (p32).
    - A GN 3 that names its table file prints " GNDINO3: Unable to open
      file <name as A40>" and "Will compute Sommerfeld-ground tables" after
      the medium, where a GN 2 prints its own block above it (p16, p30).

  p33-p43, written for momwire#1592 (the slot's leftovers: the other XNDA
  forms, Dan AC6LA's FR/XQ/FR/RP program, and the GS and GM geometry
  cards).  Captured 2026-10-21, licensed NEC-4.2, clean-room Linux build
  (nec42cl-static on the Skylake box), black-box: one thread, each deck run
  in a fresh directory, printout out, nothing else read.  Decks LF.
    p33 RP XNDA 1002      p34 RP XNDA 1003
    p35 RP XNDA 1010      p36 RP XNDA 1011
    p37 FR 1 freq, EX, XQ, then FR 6 freqs and RP: Dan's sequence
    p38 GS scaling a GW written in millimetres
    p39 GM moving and copying a GW
    p40-p43, four more GM probes on a two-wire deck: ITS 2 with two
    copies, a rotation and move with no copy, a tag increment of 10, and
    a GM followed by a GS
  What the printouts show (black box, layout only):
    - 1002 prints the pattern heading, the column headings and the average
      gain trailer with no rows; 1003 prints 1001's printout; 1010 and 1011
      head the gains DIRECTIVE GAINS, the trailer unchanged.  Each is the
      NEC-5 printout's layout for the same XNDA.
    - p37 runs in the NEC-5 program order: the XQ's run at the first FR,
      then one whole block per frequency of the second FR, each with its
      input parameters, currents, power budget and pattern, in NEC-4.2's own
      sections.
    - GS and GM print their lines inside the wire table, in card order,
      after the rows of the GW cards before them; the rows stay as written
      and a GM's copies print no rows.  GM prints its two integers, six
      reals as written, its ITS, then 1 and 0 in every probe, then the
      segment count of the structure it moves.

  p44-p53, written for momwire#1592's last three items (XNDA 1012 and
  1013, the GA and GH geometry cards).  Captured 2026-10-22, same ritual
  and build as p33-p43.  Decks LF.
    p44 RP XNDA 1012      p45 RP XNDA 1013
    p46 GA: radius 12.5, from -45 to 225 degrees, 9 segments
    p47 GA closed by a GW into a loop, then a GM moving it up 5 m
    p48 GA whose angle falls, 90 to 0 degrees
    p49 GH: one turn, constant radius, a GW lead from its start
    p50 GH tapered (helix radius 0.15 to 0.10, wire radius 0.002 to
        0.0012) with a negative length
    p51 GH with negative turns
    p52 GH of zero length (a flat spiral), helix radius 0.05 to 0.3
    p53 GH written in millimetres, then a GS
  What the printouts show (black box, layout and geometry only):
    - 1012 prints 1002's layout and 1013 prints 1010's, both under the
      DIRECTIVE GAINS heading: NEC-5's layout for the same XNDA.
    - GA prints one row in the wire table, the arc radius and its two
      angles in place of the two ends; its segments are the chords of the
      arc in the XZ plane, angles from +X toward +Z (NEC-2's arc).
    - GH is NEC-4.2's own card, not NEC-2's: a LOG-SPIRAL OR HELIX whose
      reals are turns, length along Z, the helix radius at each end and
      the WIRE radius at each end (F7 moves nothing: one exploratory deck
      printed the same bytes with 0.002 and 0 there).  Three lines in the
      wire table: the wire number and segment columns, the six reals, and
      the total length of the chords.  The vertices step the angle evenly
      from +X (clockwise for negative turns); a taper scales the radius and
      the rise of each step by one ratio, so every segment keeps one pitch
      angle, and the wire radius steps geometrically from the first segment
      to the last.  The rule is deck/_nec4.py's helix_vertices docstring.
    - Exploratory decks (not kept) showed a GA or GH with a zero wire
      radius stopping with a segment data error; both refuse here.

refused/r1_gd_after_a_run.nec, .out
  One more deck and its licensed printout, same day and ritual, for a shape
  this engine REFUSES by name: an RP, then a GD card, then a second RP.  The
  licensed engine prints a FREQUENCY and an ANTENNA ENVIRONMENT block in
  front of the second pattern and no input parameters.  Kept as the record
  of why (tests/test_eznec_nec42_rp_ne_1463.py); not among the byte gates.

refused/r2_ex4_current_element.nec, .out
  Captured 2026-10-22 with p44-p53, for the EX 4 sentence (momwire#1592).
  EX 4 1 11 0 1.0 0 on a dipole: the licensed engine prints an EXCITATION
  table with a CURRENT SOURCE at the position (1, 0, 0), its first three
  reals, of dipole moment 0, and every current zero.  NEC-4.2's EX 4 is a
  current element at a point, not NEC-5's EX 4 on a segment, so this engine
  refuses it and names EX 6 (tests/test_eznec_nec42_printout_1295.py).

  The issue's own deck, a wire standing ON a GN 3 ground with no radials,
  is not among the probes: on it the licensed engine's GN 2 and GN 3 give
  65 + j134 and 190 + j1772 ohms, and this engine 45 - j20, so its numbers
  gate nothing.  Its layout is the probes'; the issue's decks themselves
  are served in tests/test_eznec_nec42_rp_ne_1463.py.

printouts/<capture or probe>.out
  Licensed NEC-4.2 (LLNL-CODE-491368), nec42cl-serial release 7768648, run
  2026-10-03 on the build machine, black box: each deck above in, the
  engine's printout out, byte for byte (LF line endings, as that run wrote
  them; the probes' printouts were run on the laptop the same day).
  Committed as test fixtures with the licensee's permission.  They
  are OUTPUT only; no NEC-4.2 source, build tree or binary is in this
  repository, and none may be.  Line 2 is blank in every one (the engine
  prints no build tag), and the Sommerfeld-table cache messages in 0231,
  0232, 0239, p7, p9, p13-p15 and the GN 2 / GN 3 probes of p16-p31 are the
  engine's own file handling, which the gates normalise away
  (tests/test_eznec_nec42_printout_1295.py; under a NAMED file the line
  saying it was not opened stays, and this engine prints it).  p16-p32 were
  run 2026-10-13 on the laptop, the same release.

  These printouts are a machine-portable oracle, NOT a byte-portable one.
  The Windows box's licensed NEC-4.2 (the 2014 LLNL NEC42W64CL.exe), run
  2026-10-03 on the same 15 unique captured decks, agrees with them on
  every line count, every heading sequence and every feed-point Z (to
  0.00e+00). About 33 lines per file still differ:
    - the timing lines (FILL, the Sommerfeld table time, RUN TIME);
    - rows at the floating-point floor, where a ~1e-24 charge magnitude
      carries an arbitrary phase;
    - rows that differ only in the last printed digit (worst relative
      difference 5.5e-6 away from the floor).
  None of these is a defect. The round-trip gate is immune, because the
  fixture is both sides of that comparison. A gate against a FRESH licensed
  run on another machine must mask the timing and floor rows and allow a
  last-digit tolerance; do not "fix" a byte mismatch there.
