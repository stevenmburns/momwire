"""Capture the two momwire#567 ANCHOR decks against the NEC-5 binary, and
regenerate `tests/golden_buried_anchor_nec5.py`.

The anchors are the serve gates momwire#553 U5 banked for the
contact-plus-buried deck class it refuses (G-U5-12, and the deck-route twin
in `test_eznec_buried_refusal`): a 10 m ground-contact monopole fed at
segment 7 of 15 over (a) one detached 5 m radial 15 cm down and (b) a
four-radial fan, both over eps_r 13 / sigma 0.005 S/m soil at 7 MHz.

Until momwire#567 phase 0 the CARDS behind those numbers lived only in an
untracked scratch tree, and the phase-0 oracle run (issue #567, the phase-0
verdict comment) found the fan bank was a transcription 0.67 ohm off what
the engine actually prints — across eight deck spellings and GN0/GN2 alike.
This script is the durable home for both: the exact cards, the run recipe,
and the regeneration of the banked literals. The same run measured the
binary's buried-pair asymptotic workaround (`EZParam.txt`) at spread 0.0000
ohm on both decks, so the shipped configuration is captured with no
qualifier.

Must run under the antennaknobs venv (that is where the NEC-5 engine
wrapper lives), with the binary on `NEC5_EXE`:

    NEC5_EXE=~/antennas/NEC5-downloads/nec5-linux/nec5cl \
        /home/smburns/antennas/antennaknobs/.venv/bin/python \
        scripts/capture_buried_anchor_nec5.py

Only the binary's PRINTED impedances are recorded; nothing about NEC-5's
internals is read, quoted or inferred here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

GOLDEN = Path(__file__).resolve().parents[1] / "tests" / "golden_buried_anchor_nec5.py"

# The cards, verbatim. These are the spellings the phase-0 oracle validated
# against the deck-route templates in `test_eznec_buried_refusal` (same
# geometry, same GE/GN/EX/PQ requests), kept as literals so a future session
# can re-run the engine without reconstructing a deck from test parameters.
ANCHOR_DECKS = {
    "lone-radial": (
        "CM momwire#567 anchor lone-radial\nCE\n"
        "GW 1,15,0.,0.,10.,0.,0.,0.,.001\n"
        "GW 2,10,0.,0.,-0.15,5.,0.,-0.15,.001\n"
        "GE 1,-1\nFR 0,1,0,0,7.\nGN 0,0,0,0,13.,.005\n"
        "EX 4,1,7,0,1.,0.\nPQ 0\nXQ 0\nEN\n"
    ),
    "four-radial": (
        "CM momwire#567 anchor four-radial\nCE\n"
        "GW 1,15,0.,0.,10.,0.,0.,0.,.001\n"
        "GW 2,10,0.,0.,-0.15,5.,0.,-0.15,.001\n"
        "GW 3,10,0.,0.,-0.15,0.,5.,-0.15,.001\n"
        "GW 4,10,0.,0.,-0.15,-5.,0.,-0.15,.001\n"
        "GW 5,10,0.,0.,-0.15,0.,-5.,-0.15,.001\n"
        "GE 1,-1\nFR 0,1,0,0,7.\nGN 0,0,0,0,13.,.005\n"
        "EX 4,1,7,0,1.,0.\nPQ 0\nXQ 0\nEN\n"
    ),
}

# The momwire#524 phase-0/phase-2 CROSSING refinement ladder, verbatim (a
# 2 m buried vertical junction-joined at z = 0 to the 10 m monopole, fed at
# 4.333 m, soil A, ×1..×8 mesh rungs; EX lands on the segment whose center
# is nearest the feed height). These are captured FOR THE RECORD, not as
# gates: the 2026-08-26 adjudication measured that the engine's crossing
# junction is two contact ends plus a point-electrode sink (its printed
# junction currents violate its own AGARD condition divergently, ~√n, with
# a KCL deficit of 1.55 → 2.23 A into the interface point along exactly
# this ladder), so its crossing prints are a DIFFERENT EXPERIMENT from the
# exact-EM crossing serve momwire ships (soil-A answer 138.77 − 102.99j,
# mesh-stable, `test_crossing_serve_524`). Never gate the serve against
# these numbers; the ladder is kept because the divergence pattern itself
# is the adjudication evidence.
_CROSSING_RUNGS = {
    1: (4, 15, 7),
    2: (8, 30, 13),
    3: (12, 45, 20),
    4: (16, 60, 26),
    5: (20, 75, 33),
    8: (32, 120, 52),
}


def _crossing_deck(nb, na, feed_seg):
    return (
        "CM momwire#524 crossing rung\nCE\n"
        f"GW 1,{nb},0.,0.,-2.,0.,0.,0.,.001\n"
        f"GW 2,{na},0.,0.,0.,0.,0.,10.,.001\n"
        "GE 1,-1\nFR 0,1,0,0,7.\nGN 0,0,0,0,13.,.005\n"
        f"EX 4,2,{feed_seg},0,1.,0.\nPQ 0\nXQ 0\nEN\n"
    )


CROSSING_DECKS = {
    f"crossing-x{m}": _crossing_deck(*rung) for m, rung in _CROSSING_RUNGS.items()
}


def main() -> None:
    from antennaknobs.engines.nec5 import NEC5Engine

    sys.path.insert(0, str(Path.home() / "antennas/antennaknobs/scripts"))
    from bench_nec5_walk_why import make_dipole

    captures = Path(
        os.environ.get("BURIED_ANCHOR_CAPTURES", "/tmp/buried-anchor-captures")
    )
    captures.mkdir(parents=True, exist_ok=True)
    eng = NEC5Engine(make_dipole(20), ground=None, capture_dir=captures)

    anchors: dict[str, complex] = {}
    for name, deck in ANCHOR_DECKS.items():
        z = complex(eng.run_deck(deck)[0][0][2])
        anchors[name] = z
        print(f"{name:>12}: nec5 prints {z:.4f}")

    crossing: dict[str, complex] = {}
    for name, deck in CROSSING_DECKS.items():
        z = complex(eng.run_deck(deck)[0][0][2])
        crossing[name] = z
        print(f"{name:>12}: nec5 prints {z:.4f}  (convention record, not a gate)")

    _write_golden(anchors, crossing)


def _lit(z: complex) -> str:
    sign = "-" if z.imag < 0 else "+"
    return f"{z.real:.4f} {sign} {abs(z.imag):.4f}j"


def _write_golden(anchors: dict[str, complex], crossing: dict[str, complex]) -> None:
    lines = [
        '"""NEC-5 printed impedances for the two momwire#567 ANCHOR decks —',
        "the contact-plus-buried serve gates banked while the deck class",
        "refuses (momwire#553 U5, gated at G-U5-12 and the deck route).",
        "",
        "GENERATED by scripts/capture_buried_anchor_nec5.py — do not edit by",
        "hand. See that script for the cards, the run recipe and the fan",
        "re-bank story (the pre-phase-0 fan number was a transcription",
        "0.67 ohm off the engine's own printout). Citation: NEC-5",
        "(LLNL-CODE-746721), 10 m contact monopole over detached buried",
        "radials, eps_r 13 / sigma 0.005 S/m, 7 MHz, 2026-08-26.",
        "",
        "`ANCHOR_DECKS[name]` is the card text the binary ran, verbatim;",
        "`ANCHOR_LONE_RADIAL` / `ANCHOR_FOUR_RADIAL` are the impedances it",
        "PRINTED for them. The refusal prose in `_medium_spec` and",
        "`eznec/_serve` quotes these numbers; `test_buried_serve_553`'s",
        "G-U5-12 ties the spellings together and scores the decks the day",
        "the refusal lifts.",
        "",
        "No momwire float is recorded here on purpose: the momwire side is",
        "recomputed at test time. Pinning it would pin a cross-build float",
        "equality this tree does not have.",
        '"""',
        "",
        "ANCHOR_DECKS = {",
    ]
    for name, deck in ANCHOR_DECKS.items():
        lines.append(f'    "{name}": (')
        for card in deck.splitlines():
            lines.append(f'        "{card}\\n"')
        lines.append("    ),")
    lines.append("}")
    lines.append("")
    lines.append(f"ANCHOR_LONE_RADIAL = {_lit(anchors['lone-radial'])}")
    lines.append(f"ANCHOR_FOUR_RADIAL = {_lit(anchors['four-radial'])}")
    lines += [
        "",
        "# The momwire#524 crossing refinement ladder, FOR THE RECORD ONLY:",
        "# the engine's crossing junction was adjudicated 2026-08-26 as two",
        "# contact ends plus a point-electrode sink (its printed junction",
        "# currents violate its own AGARD condition divergently along this",
        "# ladder), so these prints are a different experiment from the",
        "# exact-EM crossing serve (soil-A answer 138.77 - 102.99j,",
        "# test_crossing_serve_524). NEVER gate the crossing serve on them.",
        "CROSSING_ENGINE_PRINTS = {",
    ]
    for name, z in crossing.items():
        lines.append(f'    "{name}": {_lit(z)},')
    lines.append("}")
    GOLDEN.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {GOLDEN}")


if __name__ == "__main__":
    main()
