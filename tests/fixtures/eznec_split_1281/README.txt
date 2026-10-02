momwire#1281 - one GW straight through z = 0 over a finite ground.

Dan AC6LA's buried-radial vertical (QRZ 1003328 #182): a 5.15813 m vertical
over four 5.1816 m radials 1 inch deep, #12 copper, 14 MHz, GN 0 13/.005,
with the vertical and its 1-inch connector down to the radial hub written as
ONE card from z = -0.0254 to z = 5.15813.

  dan_through.nec   21 segments, fed at node 2 (EX 0,1,2).
  thr_n21.nec       21 segments, fed at node 1.  z = 0 falls INSIDE
                    segment 1 (-0.0254 .. 0.221435).
  thr_n204.nec      204 segments, fed at node 1.  z = 0 falls inside
                    segment 1 too, 9.46 um below node 1: 204 x 25.41 mm is
                    the 5.18353 m card, not 204 x 25.4 mm, so its z = 0 cut
                    leaves a 9.46 um above half.  That is the deck's own
                    geometry; it was written meaning a boundary at z = 0.
  hs_n21.nec        thr_n21 hand-split at z = 0 into three cards (the cut
                    segment's two halves and the 20 above), coordinates
                    typed to six digits.
  hs_n204.nec       the 1-inch connector and a 203-segment vertical from
                    z = 0, fed AT the crossing node (EX 0,1,-1), which
                    momwire serves since momwire#1282.

The .out files are our licensed NEC-5 (x13) printouts for these decks exactly
as written, run 2026-10-01, quoted as numbers only, never as internals.
ANTENNA INPUT PARAMETERS rows:

  deck          NEC-5 impedance
  dan_through   73.715 + j4.7422
  thr_n21       72.373 + j4.3355   (keeps segment 1 whole, across z = 0)
  hs_n21        71.296 + j0.04432  (the same deck split at z = 0 by hand)
  thr_n204      71.111 + j1.2312
  hs_n204       71.108 + j1.2303

Every deck writes LD 5,0,1,141 - an absolute segment range sized for the
141-segment original.  On the 21-segment decks that is every wire.  On
thr_n204 / hs_n204 (324 segments) it coats the first 141 segments of the
vertical and leaves the radials bare, and momwire refuses a material range
that splits a wire, so the tests read every deck with LD 5,0,0,0 (copper on
every wire).  Copper is worth ~0.47 ohm of R on these decks (bspline thr_n204:
71.169 + j1.470 with it, 70.697 + j1.110 without), so the thr_n204 rows are a
comparison at that level and no closer.

momwire's split serve at e.g. bspline / razor-2p (copper everywhere):

  thr_n21       71.582 + j1.902  /  71.320 + j0.474
  thr_n204      71.169 + j1.470  /  71.187 + j1.415
  dan_through   72.846 + j2.493  /  72.618 + j0.810

thr_n21 lands next to NEC-5's HAND-SPLIT hs_n21 rather than its thr_n21,
which is the point: momwire splits segment 1 at z = 0 (momwire#1281's
decision), NEC-5 does not, and the ~1 + j4 ohm between NEC-5's two rows is
that discretisation choice.

Kept out of tests/fixtures/eznec/ so the 80-capture corpus and its manifest
stay untouched: these decks are hand-written, not EZNEC captures.
