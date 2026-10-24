momwire#1669 - GA, GH, GM, GX, GS and PT on the nec5 dialect.

4nec2's NEC-5 package writes these cards and EZNEC does not, so the 08-20
dialect capture (built from EZNEC and SimNEC decks) refused every one of
them. Each probe here is a synthetic deck run once on our licensed NEC-5,
nec5cl-simd-d0c1383 (static, sha256 e8d2c60a...), single thread, as a black
box, on 2026-10-24. The `.out` files are its printouts as captured, with the
NUL padding and carriage returns stripped, and are quoted as numbers only,
never as internals. tests/test_nec5_4nec2_cards_1669.py reads them.

  probe   cards                        what it settles
  a1      GA 8 segs, -30 to 200 deg    GA is NEC-2's arc, one wire of NS elements
  a2      GA falling, 120 to -30       a falling sweep
  a3      GA 0 to 360 deg              a full sweep closes on itself (8 nodes,
                                       8 unknowns)
  a5      GA tag 3 + GW at its end     a GW typed 1.732 joins the arc's
                                       2 cos 30 end, 5e-5 m away
  h1      GH, F7 = 1e-3                F1 is TURNS (NEC-4.2's reading); F7 and
                                       later fields change nothing
  h3      GH, wire radii 2e-3          F5/F6 are the wire radii
  h4      GH tapered helix radius      the log-spiral, HR1 to HR2
  h5      GH negative turns            clockwise
  m2      GM ITGI 1, NRPT 1, YS 1      one translated copy, tag 2
  m3      GM NRPT 2, ROZ 30, XS 1      copies compound
  m5      GM ITS 5                     the block starts at tag 5
  m7      GM three rotations + move    the rotation order
  q6      GM ITGI 5, NRPT 0            a move IN PLACE still adds ITGI to the
                                       tags: tag 1 answers as tag 6
  x1      GX 1 010, EX on the copy     a copy keeps its node order
  y1      GX 1 110                     two planes: 4 sections
  y2      GX 1 111                     three planes, copy order Z, Y, X
  y6      GX with a wire on the plane  the plane node is non-symmetric
  y8      GX then a GW                 a later GW is non-symmetric
  y10     GX then a second GX          the last GX sets the count
  z3      GX then a GM copy            a GM copy is non-symmetric
  z6      GX then a feed wire across   4nec2's centre-fed shape
  z8      GX then a GA                 a later GA is non-symmetric
  g1      grounded verticals, GX 010   the image counts in a section
  s1/s2   GS 0 0 2 / GS 2 2 2          GS reads no tag range: identical
  s4      GS 0 0 0.5                   coordinates AND radii scale
  p0      no PT                        the baseline
  pt1     PT -1 0 0 0                  the Wire Currents table goes, and
                                       nothing else changes
  p5      PT -1 then PT 0              the last PT wins

The Reflection symmetry block a GX deck prints (sections, basis functions in
a section, non-symmetric basis functions) is reproduced, not computed from
the engine: sections = 2 ** (planes the last GX fires), the per-section count
is the unknowns of the wires that GX copied counted on their own, and the
non-symmetric count is the rest of the structure's unknowns.

Ends join as the engine joins them: an end within 1e-4 of the length of the
element an earlier node came from is that node (with 1 m elements 9e-5 m
joins and 1e-4 m does not; 0.1 m elements join 5e-6 and not 1e-5; 10 m join
5e-4 and not 1e-3; and the EARLIER wire's element sets the reach). Measured on
pairs of collinear GW cards with an offset end, outside this directory's
fixtures; tests/test_nec5_4nec2_cards_1669.py pins the two edges.
