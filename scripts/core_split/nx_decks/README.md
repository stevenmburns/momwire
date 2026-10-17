# Buried NX chains (momwire#1512)

Each chain sweeps one buried design in frequency. With the below table's
anchors opted in (`MOMWIRE_BELOW_ANCHORS=auto`) that engages them, and their
state depends on the run's history (a fresh cache and a reused one can print
different current-table digits); they are off by default since momwire#1535,
so a plain run prints the same bytes whatever ran before it. `exe_compare.py`
and `printout_compare.py` take the directory explicitly;
`test_razor_core_standalone.py`'s buried test runs the first two.

- `ward_many_nx_99.nec` -- Ward's 99-structure chain (`aboveAndBelow.ssn`,
  above and buried wires over GN 2), whole; `ward_many_nx_cut5.nec` is its
  first five structures.
- `simnec_stream_*.nec` -- the stdin bytes of SimNEC's streaming calls in the
  A4 Ward session (`simcap.py` captures, the calls that carry `NX` chains), 2
  to 15 structures with segmentation and frequency changing between them.
  They end at the last `NX`, as the stream does: SimNEC closes stdin there.

`stream_compare.py` (momwire#1512) writes the `simnec_stream_*` decks to the
engines' stdin the way SimNEC does -- `- -`, one structure at a time, each
printout read to its `NX` echo before the next -- and holds the no-Python
executable to the launcher's streaming mode, byte for byte.
