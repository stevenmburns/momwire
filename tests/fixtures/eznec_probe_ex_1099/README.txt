momwire#1099 - EZNEC's 1e-10 V probe source at a lumped load.

EZNEC Pro/4+ writes one extra `EX 0,tag,node,0,1.E-10,0.` per lumped load
("1.E-10 volt sources for recording currents at load locations") and reads the
load current from that row of ANTENNA INPUT PARAMETERS. The seam refused any
deck with two EX 0 cards until this issue.

synthetic_probe_ex.nec       a 21-segment dipole, EX 0 at node 11, LD 4 (-j500)
                             at node 5 with EZNEC's probe EX 0 beside it
synthetic_probe_ex.out       our licensed NEC-5's printout for it (2026-09-17)
synthetic_probe_ex_noprobe.* the same deck without the probe, and its printout

What the oracle shows: one ANTENNA INPUT PARAMETERS row per EX card in deck
order; the drive row is identical to every printed digit with and without the
probe; the probe row carries its 1e-10 V, the current through the loaded node,
Z = V/I and P = 0.5 Re(V I*). The loading table is unchanged.

Kept out of tests/fixtures/eznec/ so the 80-capture corpus and its manifest
stay untouched. Decks of our own, not EZNEC captures; the oracle is the engine's
printed output, quoted as numbers only.
