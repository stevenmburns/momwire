# momwire#1668: the NEC-4 junction-charge solve, licensed NEC-4.2 as the oracle

`decks/` holds the decks `gen_decks.py` writes (its docstring says what each
family is). `nec42/<deck>.out.gz` is licensed NEC-4.2's printout of each,
gzipped with `gzip -n -9`, run as a black box: deck in, printout out.

- Engine: licensed NEC-4.2, the release build `nec42-release` (sha256
  `5fc0d508e4cd0926b5e32d6e6701efcb61fff372967b31464c7f4f2eea64e680`), on
  Haswell (`~/stevenmburns/bench-nec42/exe/`). Every deck here is in free
  space.
- Protocol: each deck copied as `m.nec` into a fresh directory and run as
  `nec42-release m.nec m.out` with `OMP_NUM_THREADS=1`; `m.out` kept as
  `nec42/<deck>.out`.
- Run 2026-10-26 for momwire#1668.

Every deck is fed on a segment beside a node the junction rule solves, where
the members' charge ratio moves the input impedance most. The `H*` decks were
built after the solve's readings had been fitted (on the other families here
and on `../nec42_radius_step_1644/`), and were not used to fit them.

`tests/test_nec4_junction_1668.py` reads the input impedance off each
printout's ANTENNA INPUT PARAMETERS row and holds the NEC-4 dialect's
point-matched sinusoidal basis (`junction_rule="nec42"`) to it.

Regenerating a deck changes nothing here unless its text changes; a changed
deck needs a fresh NEC-4.2 run before its printout is replaced.
