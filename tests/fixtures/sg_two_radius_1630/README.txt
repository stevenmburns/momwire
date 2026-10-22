momwire#1630 - the two-radius crossing node on SinusoidalGalerkinSolver.

Synthetic decks (scratch/1630-two-radius/decks.py, written for this issue;
no user's deck): a 5 m vertical from z = 0 up, one 1-segment-per-r drop from
its base down to a buried hub at -0.0254 m, and ONE 4 m radial leaving the
hub horizontally, 14 MHz, soil 13 / 0.005 Sommerfeld. Refinement r = 2
(vertical 16 segments, drop 2, radial 16). The crossing node at z = 0 joins
the vertical (above) and the drop (below).

  stem          vertical radius   drop + radial radius
  hub1_all1mm   1 mm              1 mm      (one radius: the control)
  hub1_v19mm    19 mm             1 mm      (two-radius: fat vertical)
  hub1_b025mm   1 mm              0.25 mm   (two-radius: thin buried side)
  hub1_v1p1mm   1.1 mm            1 mm      (NEC-4.2 only: a 10 % step)

Each stem comes in two dialects:

  <stem>.nec42.nec   NEC-4.2 dialect (GE -1, GN 2 ... NOFILE, EX 0 on the
                     vertical's segment 5, centred 1.25 m + h/2 up). momwire
                     reads these (`parse(..., dialect="nec4")`).
  <stem>.nec5.nec    NEC-5 dialect (GE -1 0, GN 0 ... NOFILE, EX 0 at end 2
                     of the vertical's segment 4: the knot 1.25 m up).

The .out files are our licensed engines' printouts of exactly those decks,
run 2026-10-22 on Skylake, quoted as numbers only, never as internals:
NEC-5 (x13 nec5cl) for .nec5, NEC-4.2 (nec42cl-static) for .nec42.
ANTENNA INPUT PARAMETERS:

  deck                 impedance (ohms)
  hub1_all1mm.nec5     186.38 - 68.937j
  hub1_v19mm.nec5      179.88 - 60.511j
  hub1_b025mm.nec5     206.26 - 75.890j
  hub1_all1mm.nec42    188.94 - 68.070j
  hub1_v1p1mm.nec42    195.07 - 128.74j
  hub1_v19mm.nec42      66.58 - 486.52j
  hub1_b025mm.nec42    204.11 - 232.50j

What they are for: the NEC-5 three are the radius-RESPONSE differential (each
two-radius deck minus the control), where the node convention's own gap
cancels. The NEC-4.2 four record why NEC-4.2 cannot adjudicate a two-radius
crossing node: its answer is discontinuous in the radius ratio there (a 10 %
step moves X by -60 ohm here and by -340 ohm at r = 4), while in free space
the same 19 mm-on-1 mm step agrees with NEC-5 to ~2 ohm, and with one radius
it agrees with NEC-5 to ~3 ohm. The full ladders are in
scratch/1630-two-radius/.
