momwire#1697 - CW, the catenary wire, on the nec5 and nec4 dialects.

CW ITG NS X1 Y1 Z1 X2 Y2 Z2 RAD ICAT RHM ZM hangs a wire of NS segments
between two points. Each probe here is a synthetic deck run once on each
licensed engine as a black box, on 2026-10-25, on Haswell:

  NEC-4.2  bench-nec42/exe/nec42                (sha256 02c6fc87...)
  NEC-5    nec5cl-simd-d0c1383-static           (sha256 4f27b568...)

`<probe>.nec42.out` and `<probe>.nec5.out` are their printouts as captured,
with the NUL padding and carriage returns stripped, and are quoted as numbers
only, never as internals. tests/test_cw_catenary_1697.py reads them.

  probe              cards                          what it settles
  c1_icat1_sag       CW, ICAT 1, a point mid-span   a sag: NS pieces of equal
                     below the ends                 arc length
  c2_icat2_arch      CW, ICAT 2, ZM -2              an arch: NS pieces of equal
                                                    horizontal step
  c3_icat3_length    CW, ICAT 3, ends at different  the wire's length fixes it;
                     heights and a skew span        the span need not lie on x
  c4_cw_between_gw   GW, CW, GW                     a CW joins wires at its ends
                                                    and numbers its segments in
                                                    card order
  c5_cw_gm_copy      CW then GM                     a GM copies a CW like any
                                                    wire
  c6_centre_fed      CW, ICAT 3, fed mid-wire       a centre-fed CW's impedance
  c7_centre_icat1    CW, ICAT 1, a point outside    the same, a point past the
                     the span, fed mid-wire         far end
  g6_straight        GW, the straight wire c6 and   the control: the two engines'
                     c7 hang from, fed mid-wire     own gap on a plain wire

c1-c5 feed the wire's end segment, where the two engines disagree with each
other by more than either does with momwire, so their impedances are no
check; c6, c7 and g6 are.

What both engines print beside the card:

  NEC-5    the card's echo, then "Catenary length = L    Total segment
           length = 0.00000E+00" (the second number is always zero).
  NEC-4.2  "THE FOLLOWING WIRE IS A CATENARY, ICAT RHM ZM", then "Catenary
           length = L    Total segment length = S" with S the chords' total,
           then the card's row in the wire table as a GW's.
