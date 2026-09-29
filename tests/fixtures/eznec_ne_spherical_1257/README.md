# Spherical near-field grids (`NE 1`) — momwire#1257

EZNEC writes `NE 1` when its user enters a near-field grid in spherical form, and SimNEC's `NearField("p", …)` asks for the same grid. The two decks here are hand-written in that shape; the geometry is ours. Each `.out` is the licensed NEC-5's printout of the deck beside it (`nec5cl`, Linux x13 build, run 2026-09-28).

| deck | what it pins |
|---|---|
| `dipole-ne1-free.nec` | a half-wave dipole along Y in free space; R = 2 and 10 m, θ = 0 to 90° in 15° steps, φ = 0, 45 and 90° (42 points) |
| `dipole-ne1-ground.nec` | the same dipole 1 m over a Sommerfeld ground (εr 13, σ 0.005 S/m, `GN 2 … NOFILE`); R = 10 and 1000 m, θ = 0 to 90° in 5° steps, φ = 0 and 60° (76 points) |

Measured on the licensed printouts (and on single-point probe decks run the same day):

- The card is `NE 1,NR,Nθ,Nφ,R1,θ1,φ1,ΔR,Δθ,Δφ`: R in metres, θ from the zenith (+z) and φ from +x toward +y, in degrees. The point is R·(sinθ cosφ, sinθ sinφ, cosθ). This is NOT NEC-2's order, which puts φ second and θ third.
- The walk is R fastest, then θ, then φ.
- The table stays Cartesian. The heading, the three header lines (X, Y, Z in METERS) and the column layout are the `NE 0` block's, unchanged.
- A coordinate that is exactly zero prints as about 2.6e-14 of R (`2.6485E-13` at R = 10 m). momwire places those zeros exactly and prints `0.0000E+00`.
- Any other first field (2, −1) is not a near-field request at all.

On the ground deck, two kinds of rows are left out of the value gate, and the test says why:

- **The licensed engine's high-elevation defect.** At R = 1000 m, θ = 5°, its near field is about 290 times its own far field in that direction, and 290 times its own near field at the zenith one row away. Its far field and momwire's near field show no such jump. A sweep of Dan's test bed (AC6LA, QRZ 1003328 #172) puts it at 1° to 8° from the zenith (9 % at 1°, up to 1100 times at 7°).
- **momwire's grazing defect.** At R = 1000 m, off the broadside plane, within about 15° of the horizon, momwire's vertically polarised near field is wrong. At the horizon it is about 56 times too large. Its Sommerfeld ground tables stop at 15 wavelengths, and past that the ground remainder is frozen at the table's edge. This is a follow-up; the gate does not pin it.

`dipole-ne1-ground.out` carries the licensed engine's own ground-table cache chatter (`GMPINO: Unable to open file NOFILE`, `Will compute Sommerfeld-ground tables`), which momwire does not print. The layout test strips those lines and the blank before them.
