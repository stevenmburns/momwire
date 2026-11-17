# Two bs2 refusals the NEC-5 seam printed as INTERNAL ERROR (census 11-17)

Synthetic decks, written for `tests/test_nec5_refusal_sentences_1117.py`
(its docstring has the story); no user deck.

- `crossing_radius_spread.nec`: a 5 m vertical rising from a crossing node
  at the interface, two buried radials sloping to 5 cm down from it at
  2 mm and 3 mm radius. The two-radius rule serves one radius per side of
  the interface, so every basis refuses the spread within the below wires;
  bs2 met it in the solve, past the constructor the seam refused at.
- `buried_grazing_inside_cap.nec`: a 40 m wire 3 mm deep in 13 / 0.005 S/m
  soil at 14 MHz. Pairs past the Sommerfeld range cap (4 in-medium
  wavelengths, about 22 m here) are served as zero; the pairs just inside it
  lie at theta ~0.0153 deg, under the 0.0166667 deg grazing floor, so the
  plan refuses in `_below_interface.BURIED_GRAZING_REFUSAL`'s words.

Both print the refusal sentence, never a decline or an INTERNAL ERROR, from
Python and from the C++ seam and run half (razor-2p and bs2), and from the
NEC-4.2 slot.
