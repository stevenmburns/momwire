"""What does a coarse crossing node cost razor? crossing_deck(1) and
hub_deck(4): far mesh refined x8 with the node-adjacent edges held at x1
(node h = 50 mm / 75 mm) vs refined with them (6.25 mm / 9.4 mm), and bspline
at the fine rung for scale. Feed on a knot."""

import json

from common import BSplineSolver, crossing_deck, hub_deck, razor


def cd(m, node_m):
    d = crossing_deck(1)
    b, a = d["n_per_edge_per_wire"]
    d["n_per_edge_per_wire"] = [
        [n * m for n in b[:-1]] + [b[-1] * node_m],
        [a[0] * node_m] + [n * m for n in a[1:]],
    ]
    d["feeds"] = [(1, 4.5, 1 + 0j)]
    return d


def hd(m, node_m):
    d = hub_deck()
    d["n_per_edge_per_wire"] = [
        [n * m for n in e] for e in d["n_per_edge_per_wire"][:-2]
    ] + [
        [d["n_per_edge_per_wire"][-2][0] * node_m],
        [d["n_per_edge_per_wire"][-1][0] * m],
    ]
    d["feeds"] = [(len(d["wires"]) - 1, 4.0, 1 + 0j)]
    return d


for name, mk in (("crossing_deck", cd), ("hub_deck(4)", hd)):
    for m in (2, 4, 8):
        zc = complex(razor(mk(m, 1)).compute_impedance()[0])
        zf = complex(razor(mk(m, m)).compute_impedance()[0])
        zb = complex(BSplineSolver(**mk(m, m)).compute_impedance()[0])
        print(
            json.dumps(
                dict(
                    deck=name,
                    m=m,
                    razor_coarse_node=str(zc),
                    razor_fine_node=str(zf),
                    node_worth=abs(zc - zf),
                    bspline_fine=str(zb),
                    gap_coarse=abs(zc - zb),
                    gap_fine=abs(zf - zb),
                )
            ),
            flush=True,
        )
