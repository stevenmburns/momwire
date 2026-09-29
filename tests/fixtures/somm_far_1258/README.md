# The near field past the ground table (momwire#1258)

Two hand-written decks, the geometry ours: the #1257 test bed's half-wave dipole along Y, 1 m over a Sommerfeld ground (εr 13, σ 0.005 S/m, `GN 2 … NOFILE`), 299.8 MHz, so one wavelength is 1 m. Each `.out` is the licensed NEC-5's printout of the deck beside it (`nec5cl`, Linux x13 build, run 2026-09-28).

| deck | grid |
|---|---|
| `grazing-near.nec` | `NE 1`: R = 20 and 50 m; θ = 0 to 90° in 1°; φ = 0 and 90° (364 points) |
| `grazing-far.nec` | `NE 1`: R = 1000 m; θ = 0 to 90° in 1°; φ = 0 and 90° (182 points) |

All three ranges are past momwire's 15-wavelength Sommerfeld table, where until #1258 the ground remainder was frozen at the table's edge. φ = 0 is the dipole's broadside plane (horizontal polarisation), φ = 90° its end-fire plane (vertical polarisation, the one the frozen edge got wrong near the horizon).

razor-2p against these printouts, vector difference over the licensed |E| at the point, worst per band:

| R | plane | elevation < 3° | 3–15° | before #1258, < 3° |
|---|---|---|---|---|
| 20 m | φ = 0 | 2.3e-3 | 2.4e-3 | 9.1e-3 |
| 20 m | φ = 90° | 8.8e-4 | 1.7e-3 | 3.1e-1 |
| 50 m | φ = 0 | 1.8e-3 | 2.2e-3 | 3.3e-2 |
| 50 m | φ = 90° | 4.5e-3 | 1.0e-3 | 2.1 |
| 1000 m | φ = 0 | 1.7e-3 | 2.3e-3 | 4.5e-1 |
| 1000 m | φ = 90° | 5.5e-3 | 8.5e-4 | 57.6 |

Most of what is left at 1000 m is a steady phase offset of about 0.3° between the two engines, the same at every elevation and present before #1258.

The licensed engine's near-zenith defect (see `../eznec_ne_spherical_1257/README.md`) shows here too, 1° to 8° from the zenith: 17 % at 20 m, 64 % at 50 m, 1100 times at 1000 m. The test leaves those rows out; the zenith itself is sound and stays in.
