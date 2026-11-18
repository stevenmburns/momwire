"""Synthetic NEC-4 decks for RP with XNDA N >= 1 on the NEC-4.2 printout.

Each deck is run through licensed NEC-4.2 as a black box (README.md); this
script only writes the decks.  Regenerate with

    python tests/fixtures/nec42_rp_normalized_gain/gen_decks.py \
        tests/fixtures/nec42_rp_normalized_gain/decks

The families mirror ../rp_normalized_gain/ (the NEC-5 set), in NEC-4.2's
dialect: no XQ after the RP (an RP executes), and the grounds this slot
serves (GN 1, GN 1 + GD, GN 2 and GN 3 with NOFILE) in place of NEC-5's GN 0.
"""

import os
import sys

TILTED = "GW 1 21 -0.8 0.3 -2.35 0.8 -0.3 2.35 0.001"
LOAD = "LD 0 1 6 0 40 0 0"
VERTICAL = "GW 1 21 0.1 0 1.0 -0.1 0 6.0 0.002"
FREQ = "FR 0 1 0 0 28.5 0"
FEED = "EX 0 1 11 0 1 0"


def rp(xnda, nth=7, nph=4, th0=0, ph0=0, dth=30, dph=45, rfld=0, gnor=0):
    return f"RP 0 {nth} {nph} {xnda} {th0} {ph0} {dth} {dph} {rfld} {gnor}"


def write(out, name, comment, geometry, ge, body):
    lines = [f"CM RP normalized-gain probe {name}", f"CM {comment}", "CE"]
    lines += geometry + [ge] + body + ["EN"]
    with open(os.path.join(out, f"{name}.nec"), "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def free(out, name, comment, *cards, load=True, freq=FREQ):
    body = ([LOAD] if load else []) + [FEED, freq, *cards]
    write(out, name, comment, [TILTED], "GE 0", body)


def main(out):
    os.makedirs(out, exist_ok=True)
    # N = 0 controls, and N = 1..5 at D = 0 and 1 (A = 1)
    for n in range(6):
        for d in (0, 1):
            x = 1000 + 100 * n + 10 * d + 1
            free(out, f"free{x}", "tilted dipole, free space, 40 ohm load", rp(x))
    # the averaging digit
    for x in (1500, 1502, 1503, 1510, 1512, 1513):
        free(out, f"free{x}", f"XNDA {x}", rp(x))
    # GNOR, RP's tenth field
    for n in (1, 5):
        for tag, g in (("gnor10", 10), ("gnor0p5", 0.5), ("gnorm3", -3)):
            x = 1000 + 100 * n + 1
            free(out, f"free{x}_{tag}", f"GNOR {g}", rp(x, gnor=g))
    free(out, "free1501_range", "RFLD 1000", rp(1501, rfld=1000))
    free(
        out,
        "free1500_sweep",
        "theta -180 to 178 in 2, one phi",
        rp(1500, nth=180, nph=1, th0=-180, dth=2, dph=0),
        load=False,
    )
    free(
        out,
        "free1500_big",
        "37 x 73 directions",
        rp(1500, nth=37, nph=73, dth=5, dph=5),
        load=False,
    )
    # two RP cards in one run
    for name, a, b in (
        ("free_tworp", 1501, 1301),
        ("free_tworp_mixed", 1501, 1001),
        ("free_tworp_mixed2", 1001, 1501),
    ):
        free(
            out,
            name,
            f"two RP cards, XNDA {a} then {b}",
            rp(a, nth=4, nph=2, dph=90),
            rp(b, nth=3, nph=3, th0=10, dth=20),
            load=False,
        )
    # the column split
    for n, card in (
        (1, rp(1501, nth=1, nph=1, th0=30)),
        (2, rp(1501, nth=2, nph=1, th0=30)),
        (4, rp(1501, nth=4, nph=1, th0=30, dph=0)),
        (5, rp(1501, nth=5, nph=1, th0=30, dth=10, dph=0)),
    ):
        free(out, f"small{n}", f"{n} directions", card, load=False)
    # three frequencies
    for x in (1101, 1501):
        free(
            out,
            f"mfreq{x}",
            "three frequencies",
            rp(x, nth=4, nph=3),
            load=False,
            freq="FR 0 3 0 0 27 1.5",
        )
    # a single cut (one phi, then one theta) at N = 0 and 5: whether the
    # averaging digit is read there
    for x in (1001, 1002, 1003, 1011, 1501, 1502):
        free(
            out,
            f"cut{x}",
            "one phi: a theta cut",
            rp(x, nth=4, nph=1, th0=30, dph=0),
            load=False,
        )
    for x in (1001, 1501, 1502):
        free(
            out,
            f"phicut{x}",
            "one theta: a phi cut",
            rp(x, nth=1, nph=4, th0=60, dth=0),
            load=False,
        )
    # a z dipole: every horizontal gain is the null
    write(
        out,
        "zdip1401",
        "z dipole, H gains null",
        ["GW 1 21 0 0 -2.5 0 0 2.5 0.001"],
        "GE 0",
        [FEED, FREQ, rp(1401)],
    )
    # crossed dipoles in quadrature: a real minor axis
    for x in (1101, 1201, 1211, 1501):
        write(
            out,
            f"xdip{x}",
            "crossed dipoles in quadrature",
            [
                "GW 1 21 -2.4 0 0 2.4 0 0 0.001",
                "GW 2 21 0 -2.4 0.3 0 2.4 0.3 0.001",
            ],
            "GE 0",
            ["EX 0 1 11 0 1 0", "EX 0 2 11 0 0 1", FREQ, rp(x)],
        )
    # grounds
    grounds = {
        "gn1": ["GN 1"],
        "gd": ["GN 1", "GD 2 0 0 0 13 0.005"],
        "gn2": ["GN 2 0 0 0 13 0.005 0 0 NOFILE"],
        "gn3": ["GN 3 0 0 0 13 0.005 0 0 NOFILE"],
    }
    for g, cards in grounds.items():
        for x, card in (
            (1501, rp(1501, nth=10, nph=3, dth=10)),
            (1511, rp(1511, nth=10, nph=3, dth=10)),
            (1301, rp(1301, nth=10, nph=3, dth=10)),
            ("1501_below", rp(1501, nth=7, nph=2, dth=30, dph=90)),
        ):
            write(
                out,
                f"{g}_{x}",
                f"near-vertical dipole over {' '.join(cards)}",
                [VERTICAL],
                "GE 1",
                [*cards, FEED, FREQ, card],
            )


if __name__ == "__main__":
    main(sys.argv[1])
