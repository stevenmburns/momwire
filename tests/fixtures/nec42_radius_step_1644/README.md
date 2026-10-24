# momwire#1644: radius-step junctions, licensed NEC-4.2 as the oracle

`decks/` holds the decks `gen_decks.py` writes (its docstring says what each
family is). `nec42/<deck>.out.gz` is licensed NEC-4.2's printout of each,
gzipped with `gzip -n -9`, run as a black box: deck in, printout out.

- Engine: licensed NEC-4.2 on Haswell (`~/stevenmburns/bench-nec42/exe/`), two
  builds, the release build being the reference:
  - `nec42-release` (sha256 `5fc0d508e4cd0926b5e32d6e6701efcb61fff372967b31464c7f4f2eea64e680`)
    for every deck with a wire end on a GN 2 / GN 3 ground: `G_vert_somm_n*` and
    `G_vuni*_somm_n*`;
  - main-pre `nec42` (sha256 `02c6fc87c8cb75cbfc7444809d7e0656fcbeaf5b83689efb9e919956739647cd`)
    for the rest. It differs from the release build only where a wire end touches
    a Sommerfeld ground (it is wrong there), and is bit-identical to it elsewhere:
    18 decks were run through both, the 13 without such a contact (free space,
    GN 1 including a wire end on it, the dipole elevated over GN 2,
    `G_hdip_somm_n*`) printing the same
    impedance to the last digit. No other deck here has a wire end on a GN 2 /
    GN 3 ground, so those printouts stand.
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
