# momwire#1295 phase 2: NEC-4.2 probe decks

Decks for the licensee to run through the licensed NEC-4.2 (black box: deck
in, printout out), so the NEC-4.2-slot printout writer
(`src/momwire/eznec/_nec4_printout.py`) can be measured on shapes the 17
EZNEC captures do not cover. None is an EZNEC capture. Until a printout comes
back, each shape below is either refused by name or printed in a layout
borrowed from NEC-5 / NEC-2, as noted.

| deck | settles | today |
|---|---|---|
| `p1_junction.nec` | the MULTIPLE WIRE JUNCTIONS row format; whether a junction end gets a charge `E` row | NEC-2 row format, no `E` row |
| `p2_three_frequencies.nec` | the layout of a run at several frequencies (which sections repeat) | refused |
| `p3_rlc_loads.nec` | the LD 0 / LD 1 loading rows | NEC-5's measured canvas |
| `p4_near_field.nec` | the NE table | refused |
| `p5_gd_xq.nec` | whether FAR FIELD GROUND PARAMETERS prints without a pattern | not printed |
| `p6_voltage_source_on_network.nec` | excitation-row order with an EX 0 (voltage) source at a network point | discovery order |
| `p7_gn2_7mhz.nec` | the COMPLEX DIELECTRIC CONSTANT at a second frequency (eps0 = 8.854E-12 reading) | that reading |
| `p8_two_runs.nec` | the layout of two execute runs in one deck | refused |
| `p9_above_and_buried_gn2.nec` | per-segment wavelength normalisation when a deck has wires on both sides | medium wavelength below, free above |
