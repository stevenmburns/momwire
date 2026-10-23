# momwire#1644: radius-step junctions, licensed NEC-4.2 as the oracle

`decks/` holds the decks `gen_decks.py` writes (its docstring says what each
family is). `nec42/<deck>.out.gz` is licensed NEC-4.2's printout of each,
gzipped with `gzip -n -9`, run as a black box: deck in, printout out.

- Engine: NEC-4.2 main-pre, `nec42` built on Haswell, sha256 `02c6fc87c8cb75cb…`.
- Protocol: each deck copied as `m.nec` into a fresh directory and run as
  `nec42 m.nec m.out` with `OMP_NUM_THREADS=1`; `m.out` kept as
  `nec42/<deck>.out`.
- Run 2026-10-23 for momwire#1644.

`tests/test_nec4_radius_step_1644.py` reads the input impedance off each
printout's ANTENNA INPUT PARAMETERS row and holds the NEC-4 dialect's
point-matched sinusoidal basis (`junction_rule="nec42"`) to it, within the
tolerance that module derives.

Regenerating a deck changes nothing here unless its text changes; a changed
deck needs a fresh NEC-4.2 run before its printout is replaced.
