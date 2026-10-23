momwire#1651 - GA (wire arc) and GH (helix) on the nec2 dialect, against
nec2c, the public NEC-2 code.

Seventeen decks written for the issue, each run through nec2c 1.3.1 (the
public C translation of NEC-2, built from its own source) on 2026-10-23:
`nec2c -i<deck>.nec -o<deck>.out`, nothing edited. The printouts are nec2c's
own; the decks end in EN, not NX, because this nec2c stops on a trailing NX.

  n01 GA radius 2, 0 to 180 degrees, 9 segments, then GM +10 m in z (the
      issue's arc, the same deck as the NEC-4.2 probe p78)
  n02 GA radius 12.5, -45 to 225 degrees      n03 GA whose angle falls, 90 to 0
  n04 GH constant radius, five turns          n05 GH with B1 = B2 = 0 (circular)
  n06 GH with a negative length (left-handed) n07 GH tapered, 0.05 to 0.1
  n08 GH elliptical, constant radii           n09 GH tapered with B2 = 0
  n10 GH with a GW lead from its start        n11 the issue's GH in NEC-2's layout
  n12 GA of a full 360 degrees (nec2c reads it; NEC-4.2 stops on it)
  n13 GH tapered inward and left-handed
  n14-n16 n04, n09 and an elliptical left-handed GH at a thousand times the
      size, so the four printed decimals hold eight significant digits
  n17 n02 at a thousand times the size

What the printouts show, and what tests/test_nec2_ga_gh_1651.py holds:
  - the STRUCTURE SPECIFICATION block is nec2c's bytes: GA's one row, GH's
    two rows and its pitch-angle or cone-angle line;
  - every segment's centre, length, orientation, radius, connections and
    tag in SEGMENTATION DATA, to nec2c's four printed decimals. Two
    differences are not geometry: nec2c's PI is 3.141592654, so a value
    that rounds on a half in the fourth decimal may land one digit apart,
    and a coordinate a rounding away from zero may print -0.0000; and BETA
    has no value on a vertical segment (nec2c prints 0 or 180 there).
