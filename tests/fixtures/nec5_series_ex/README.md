# Several voltage EX cards on one gap

Hand-written decks with invented geometry: a 21-segment dipole at 14 MHz in free space, and a second one 3 m away where a deck needs two wires. Each `.out` is the licensed NEC-5's printout of the deck beside it (our Linux build, run as a black box, 2026-11-21). `tests/test_nec5_series_ex_one_gap.py` gates them.

| deck | what it pins |
|---|---|
| `one_card.nec` | the control: one 1 V card at the centre |
| `two_halves.nec` | two 0.5 V cards on that gap: one row each, each 0.5 V, the control's current |
| `three_cards.nec` | three cards whose complex volts sum to 1 V: three rows in card order, the control's current |
| `loaded_one.nec` / `loaded_two.nec` | the same identity with an `LD 4` on the driven node |
| `second_wire_pair.nec` | a 1 V card on wire 1 beside two 0.5 V cards on wire 2: the pair answers as wire 1's 1 V does |
| `probe_pair.nec` | an all-voltage deck with two 1e-10 V probes at one loaded node |
| `four_port.nec` | a SimNEC-shaped 4-port extraction: one `XQ` per port, 1 V on one card and 1e-10 V on the rest, ports 2 and 3 two series ports at one node, written with explicit end codes |
| `network_one_card.nec` / `network_two_halves.nec` | at a `TL` end the repeat is NOT a series sum: two 0.5 V cards print a current that is neither the 1 V card's nor half of it. Refused by name |
| `nt_two_halves.nec` | the same at an `NT` end. Refused by name |
| `voltage_and_current.nec` | an `EX 0` and an `EX 4` on one segment: NEC-5 stops with an error. Refused |
| `two_currents.nec` | two `EX 4` on one segment: NEC-5 prints two rows with the set current each and twice the one-card voltage. Refused here; the seam does not decide how two set currents share one path |
