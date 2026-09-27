# Multi-run NEC-5 decks — momwire#1237

SimNEC measures an N-port by driving each port in turn at 1 V, with the others held at 1e-10 V, and writes one `XQ` block per port into a single deck. Each `.out` here is the licensed NEC-5's printout of the deck beside it (`nec5cl`, Linux build, run 2026-09-27). The decks are hand-written in that shape.

| deck | what it pins |
|---|---|
| `two-port.nec` | two runs, each `EX … RP … XQ`. An `RP` executes the run, so the `XQ` after it is echoed at the head of the NEXT run's group (and the last one before `EN`) |
| `two-port-xq.nec` | the same deck without `RP`: each `XQ` executes and is echoed with its own block (SimNEC's Cardioid shape) |
| `two-port-ground.nec` | over a Sommerfeld ground (`GN 2 … NOFILE`): the range form's `LOWER MEDIUM - EXP(-JKR)/R=` line, whose three-digit exponent prints Fortran's way (`4.87288-114`) |

Measured on the licensed printouts:
- Each block's `EX` cards REPLACE the source set; every run's input table carries that block's own sources, never an accumulation.
- The frequency, environment, loading and timing sections print once. Every later run prints only its card echoes, input parameters, currents, power budget and patterns.
- One `RUN TIME` at the end.

`two-port-ground.out` carries the licensed engine's own ground-table cache chatter (`GMPINO: Unable to open file SOMMPD.NEX`, `Will compute Sommerfeld-ground tables`), which momwire does not print. The layout test strips those lines and the blank before them.
