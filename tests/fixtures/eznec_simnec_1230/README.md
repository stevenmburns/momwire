# SimNEC-shaped NEC-5 decks — momwire#1230

SimNEC 5.3 drives a NEC-5 engine with decks EZNEC never writes:

- **a five-field `EX`**: no imaginary drive;
- **a bare `GE 00`**;
- **an `RP` with a nonzero range field** (RFLD = 1000 m).

These three decks are hand-written in that shape, not copied from SimNEC. Each `.out` is the licensed NEC-5's printout of the deck beside it (`nec5cl`, Linux build, run 2026-09-27).

| deck | what it pins |
|---|---|
| `dipole-ex5.nec` | the five-field `EX` parses, and drives at 1 + j0 V |
| `dipole-rp-range.nec` | the range form of the pattern: the two extra header lines, `VOLTS/M`, E / R, and phase + ∠exp(−jkR) |
| `dipole-rp-norange.nec` | the same deck at RFLD = 0, the reference the range form is read against |

The five-field default was measured on the licensed engine before the seam was relaxed. The five-field deck's printout is identical to the same deck with an explicit `0.0` imaginary drive, timing lines aside. With `0.5` in that field it prints V = 1 + j0.5, so the field is read.
