momwire#1488 - 4nec2 as a front end for the NEC-5 launchers.

Dan AC6LA's 4nec2 Example 3 (an inverted V over ground), as 4nec2 hands it
to an engine, and Ward AE6TY's razor-2p port (76.7) run on it under 4nec2.
Sent by Dan by email, 2026-10-14.

  EXAMPLE3.inp   The deck exactly as 4nec2 writes it: CRLF, no EN card, and
                 `GE 0 0 1` -- the third GE field asking for a .GMO file.
                 4nec2 names it, and the printout, on the engine's stdin.
  EXAMPLE3.GMO   Ward's port's geometry file for that deck: 42 nodes, 41
                 elements. The byte oracle tests/test_eznec_4nec2_1488.py
                 holds momwire's .GMO to.
  EXAMPLE3.out   Ward's port's printout AFTER the 4nec2 NEC-5 package's
                 Edit_out.exe has edited it: the banner change and the
                 SEGMENTATION DATA table are the package's, not the engine's.
                 Not a NEC-5 printout; kept for its impedance only,
                 61.409 + j18.768 ohms at 3.68 MHz on razor-2p.

The .GMO layout is read from this one sample and nothing else (no NEC-5
source or manual): see src/momwire/eznec/_gmo.py.
