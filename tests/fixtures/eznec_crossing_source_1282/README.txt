momwire#1282 - a series source on a crossing junction in the ground plane.

Dan AC6LA's model (QRZ thread 1003328, post #182, 2026-10-01), rebuilt from
his Buried-radials.weq: a quarter-wave vertical (wire 1, 16.923 ft) over four
17 ft #12 copper radials buried 1 in deep, 14 MHz, average ground (GN 0,
13 / 0.005). The vertical ends at z = 0, and a one-segment connector (wire 6)
runs from there down to the radial hub, so the node at z = 0 is a CROSSING
junction: one wire rising from it, one buried wire ending on it.

  dan_end1.nec        EX 0,1,-1: the source on the junction, named through
                      the vertical (Dan's first try).
  dan_end1_via6.nec   EX 0,6,-1: the same node, named through the buried
                      connector.
  dan_seg1end2.nec    EX 0,1,1: the source one segment up the vertical (Dan's
                      workaround), served before momwire#1282.

Kept out of tests/fixtures/eznec/ so the capture corpus and its manifest stay
untouched.

The `.out` files are our licensed NEC-5 (x13 build) printouts for these decks
exactly as captured, run 2026-10-01, and are quoted as numbers only, never as
internals. ANTENNA INPUT PARAMETERS rows:

  deck              NEC-5 impedance
  dan_end1          70.787 - 0.69629j ohms
  dan_end1_via6     70.787 - 0.69629j ohms  (identical; currents negated,
                                             since the two cards drive the
                                             node in opposite directions)
  dan_seg1end2      71.415 - 0.0068143j ohms

So the source on the junction is the junction's single through-current, and
the tag that names it carries no physics. tests/test_crossing_junction_source_1282.py
gates every knot basis against both facts.
