# NX-chained decks (momwire#1512)

`nx_somm_free_network.nec` chains four structures with `NX`, so the walks over
`tests/fixtures` (and `exe_compare.py` / `printout_compare.py`) cover a chain:
a dipole over a Sommerfeld ground with a near field, the same dipole in free
space, two dipoles joined by a TL and an NT in free space, then a vertical over
the Sommerfeld ground at another frequency with a near field. Each structure
starts from scratch, so a ground or a network in one must not leak into the
next, and a lazily filled Sommerfeld table must serve the later structure's
nodes that the first one never asked for. (Synthetic: TL before NT, the order
the executable serves a mixed network table in.)

The buried chains (Ward's, SimNEC's streams) are in
`scripts/core_split/nx_decks/`: a chain of buried structures swept in
frequency engages the below table's anchors, whose state depends on what ran
before, so they are run with the anchors off (`README.md` there).
