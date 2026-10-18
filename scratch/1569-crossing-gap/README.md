# momwire#1569: a source at a crossing node, against the source one segment up

`check.py DECK M[,M...]` solves, at each refinement level (every wire's
segment count times M), the deck with a node gap at the start of wire 0 (the
rising wire, at the crossing node) and with a feed one segment up that wire,
on bspline (bs2) and razor-2p. GN 2 13 / 0.005 Sommerfeld ground, 14 MHz,
Rocketlake, default kernels (bs2's extended kernel is off on the solver API;
razor's buried fill takes neither the extended nor the exact kernel, and the
exact ring kernel is refused for buried wires on both).

## Synthetic slant, `slant16.nec` (K = 17)

A 5 m vertical (r = 3 mm) over 16 buried radials that start at its base and
slope to 0.1 m deep. Z in ohms; d = Z(one segment up) - Z(node).

| segs/wire | bs2 node | bs2 up | bs2 \|d\| | razor node | razor up | razor \|d\| |
|---|---|---|---|---|---|---|
| 5  | 38.84 - 18.85j | 43.82 - 19.72j | 5.06 | 37.74 - 22.07j | 42.57 - 23.76j | 5.12 |
| 10 | 39.06 - 17.62j | 40.67 - 17.41j | 1.62 | 38.53 - 19.25j | 40.04 - 19.29j | 1.51 |
| 20 | 39.13 - 16.89j | 39.77 - 16.59j | 0.71 | 38.86 - 17.85j | 39.42 - 17.67j | 0.59 |
| 40 | 39.16 - 16.47j | 39.47 - 16.24j | 0.39 | 39.01 - 17.07j | 39.26 - 16.92j | 0.30 |
| 80 | 39.18 - 16.22j | 39.35 - 16.06j | 0.24 | 39.09 - 16.61j | 39.22 - 16.51j | 0.17 |

|d| shrinks steadily on both engines, its level-to-level ratio settling at
1.6-1.7 (local order ~0.7), the rate the node impedance itself converges at
(bs2's node X steps by 1.23, 0.73, 0.42, 0.25: ~h^0.77). Richardson limits
at p = 0.77:

| | limit |
|---|---|
| bs2, node | 39.20 - 15.87j |
| bs2, one segment up | 39.18 - 15.80j |
| razor, node | 39.21 - 15.96j |
| razor, one segment up | 39.17 - 15.92j |
| NEC-5 at the node (black box, its own ladder to 160 segs/wire) | 39.22 - 15.88j |

All within ~0.1 ohm of each other: the node port is the limit of the source
one segment up, on both engines, and is NEC-5's.

## Thick verticals

On a real slant layout with a 19 mm-radius (1.5 in. diameter) vertical, bs2's
|d| shrank 4.7, 1.9, 1.2, 1.0 and then held at 1.0 ohm from 40 to 80
segments a wire (razor's kept shrinking: 4.8, 1.6, 0.8, 0.6, 0.4); with
bs2's extended kernel it kept shrinking (1.9, 1.2, 0.9, 0.8). There a
segment of the vertical is 3-7 radii long, and "one segment up" sits a few
radii from the node: the reduced-kernel thin-wire model's regime, not the
node port's. The same layout with the vertical at 3 mm converges like the
synthetic deck (|d| bs2 0.71, 0.40, 0.25; razor 0.59, 0.30, 0.18 at 20, 40,
80 segments a wire). Thick verticals near a crossing node are worth trying
under the different kernel choices.
