# Stepping FR probes — momwire#1508

Each `.out` is the licensed NEC-5's printout of the deck beside it (`nec5cl`, Linux build `nec5cl-3b75639`, run on Skylake 2026-10-16), used strictly as a black box. The decks are hand-written. The stepping rule itself is NEC-2's public one: `FR I1 I2 . . F dF` asks for `I2` frequencies from `F`, each the previous plus `dF` (`I1` 0) or times `dF` (`I1` 1).

| deck | what NEC-5 does |
|---|---|
| `p01_lin1`, `p06_zero` | one frequency (`I2` 1 or 0) |
| `p02_lin2`, `p03_lin5` | 2 and 5 frequencies, linear |
| `p04_mul5` | 5 frequencies, multiplicative (26, 27.3, 28.665, ...) |
| `p05_neg3` | a negative step counts down (30, 29, 28) |
| `p15_fr_first`, `p17_ld5` | FR before EX; an LD 5 deck |
| `p08_rp`, `p14_xnda1003` | an RP runs the loop; each frequency prints its pattern and average gain |
| `p09_rp_xq` | RP then XQ: the loop runs at the RP; the XQ is echoed after the last frequency and runs nothing |
| `p10d_ne_xq`, `p10e_nh_rp` | NE/NH before the XQ or RP print a near field per frequency |
| `p10_ne`, `p10c_ne_fr_first`, `p16_no_xq` | nothing after FR runs the loop: NE alone does NOT start it, and NEC-5 prints no frequency at all |
| `p10b_ne_single` | contrast: with ONE frequency, NE alone does execute |
| `p11_ex_change`, `p12_ld_change` | a later XQ block after a stepped run is answered at the last frequency only |
| `p18_two_rp` | a second RP after the loop prints one pattern, at the last frequency only |
| `p07_two_fr`, `p13_fr_after_xq` | a second FR in a later block runs its own loop |
| `b01_buried_xq`, `b02_buried_rp` | the same over GN 2 with buried radials |

Layout, measured on all of them: the card echoes run through the card that executes the loop (the first RP or XQ). Then each frequency prints a whole single-run body: FREQUENCY, ANTENNA ENVIRONMENT, STRUCTURE IMPEDANCE LOADING, MATRIX TIMING, then the results. `ALLOCATE CM:` appears in the first frequency only, and one blank more separates each frequency block from the one before. The leftover echoes (an XQ after an RP), the EN echo and one RUN TIME close it.

momwire serves `p01`-`p06`, `p08`, `p09`, `p10b`, `p10d`, `p10e`, `p14`, `p15`, `p17`, `b01` and `b02` in this layout. It refuses by name the decks where nothing runs the loop (`p10`, `p10c`, `p16`) and the decks with a request or block after the loop (`p11`, `p18`), which NEC-5 answers at the last frequency only. `p07` and `p13` keep the multi-run refusal of a later block carrying FR, and `p12`'s LD card is not in this dialect's form. The buried decks' printouts carry momwire's own interface-split note (momwire#1281), which NEC-5 does not print.
