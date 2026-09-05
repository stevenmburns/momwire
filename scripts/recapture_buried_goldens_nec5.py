"""Regenerate the buried NEC-5 goldens under the documented ground card (#929).

The 23 current decks and the 2 anchors were captured with `GE 1,-1` — ground
flag 1 with the segment-check field at -1. antennaknobs#1025 found the wrapper
that wrote those cards had the two fields transposed, and flag 1 is documented
as not usable when wires go below the surface. Every deck here is fully buried,
so every one of them was in that class.

Emits `tests/golden_buried_currents_nec5.py` with the documented-flag capture
as the primary record and the flag-1 capture kept alongside as a WITNESS, so a
revert of the card fails a test that names the number rather than quietly
republishing it.

The printouts are End-User Reports; NEC-5 is (c) LLNL, LLNL-CODE-746721. The
binary is user-licensed and never distributed. Executable only — nothing is
read from the licensed source tree, and nothing of the engine's internals is
transcribed here.

Usage:  NEC5_EXE=/path/to/nec5cl python scripts/recapture_buried_goldens_nec5.py
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from golden_buried_currents_nec5 import DECKS  # noqa: E402

EXE = os.environ.get("NEC5_EXE")
OUT = ROOT / "tests" / "golden_buried_currents_nec5.py"


def run(deck):
    with tempfile.TemporaryDirectory(prefix="nec5_929_") as td:
        (Path(td) / "m.nec").write_text(deck)
        subprocess.run(
            [EXE],
            input="m.nec\nm.out\n\n",
            text=True,
            capture_output=True,
            cwd=td,
            timeout=1800,
        )
        out = Path(td) / "m.out"
        if not out.is_file():
            raise RuntimeError("NEC-5 produced no printout")
        return out.read_text(errors="replace")


def parse_z(text):
    m = re.search(
        r"- - - ANTENNA INPUT PARAMETERS - - -(.*?)(?:\n\s*\n\s*\n|$)", text, re.S
    )
    for line in m.group(1).splitlines() if m else []:
        t = line.split()
        if len(t) >= 12 and re.fullmatch(r"\d+", t[0]):
            return complex(float(t[7]), float(t[8]))
    raise RuntimeError("no ANTENNA INPUT PARAMETERS row")


def parse_currents(text):
    """(element, z_norm, real_A, imag_A) per printed row.

    REAL and IMAG — columns 7 and 8 — not magnitude and phase. The previous
    file's docstring said magnitude/phase and its consumer believed it, which
    is momwire#929's third defect; the column choice is proved by the
    provenance mode of `scratch/929-study/recapture.py`, which reproduces
    every banked tuple bit for bit from the original card.
    """
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if "- - - Wire Currents - - -" in ln)
    rows = []
    for line in lines[start:]:
        if "Wire Charge Densities" in line or "POWER BUDGET" in line:
            break
        t = line.split()
        if len(t) >= 10 and re.fullmatch(r"\d+", t[0]) and re.fullmatch(r"\d+", t[1]):
            rows.append((int(t[0]), float(t[4]), float(t[6]), float(t[7])))
    return tuple(rows)


_GE = re.compile(r"^GE\s*[-\d]+\s*,\s*[-\d]+\s*$", re.M)

DOCUMENTED_CARD = "GE -1,0"
ORIGINAL_CARD = "GE 1,-1"


def with_card(deck, card):
    """The deck with its ground card replaced, whichever card it starts from.

    Written this way so the generator is IDEMPOTENT: after one run the golden
    file carries the documented card, and a version that could only rewrite
    `GE 1,-1` would fail on its own output. A generator that works exactly
    once is a trap, not a tool.
    """
    new, n = _GE.subn(card, deck, count=1)
    if n != 1:
        raise RuntimeError(f"no single ground card found in:\n{deck[:160]}")
    return new


HEADER = '''"""NEC-5 printed CURRENT tables for buried-dipole decks.

Captured from our licensed NEC-5 engine\'s printout (LLNL-CODE-746721) during
the momwire#524 phase-0 campaign, 2026-08-22, and RE-CAPTURED 2026-09-05 under
the documented ground card (momwire#929). Deck text, printed segment currents
and printed input impedance, nothing else — no engine internals, no run
configuration, no file or routine names.

WHY THE RE-CAPTURE. The original decks carried `GE 1,-1`: ground flag 1 with
the segment-check field at -1. antennaknobs#1025 found the wrapper that wrote
those cards had the two fields transposed, and flag 1 is the setting documented
as not usable when wires go below the surface. Every deck here is fully buried,
so every one was in that class. What the old captures showed — a printed
NEGATIVE resistance on three decks, and a current spiking ~10x at the driven
element and collapsing one element away — was our card, not the engine:

    negative printed R      flag 1: 3 of 23      documented: 0 of 23
    feed spike ratio        flag 1: 9.24-14.39   documented: 1.00-1.05

Verified against our licensed materials; conclusions only.

`DECKS[id]` carries:

* `deck` — the card set verbatim, so a consumer builds the momwire side from
  the geometry rather than from a transcription of it;
* `currents` — `(element, z_center_normalized, REAL_A, IMAG_A)` per printed
  row, in the printout\'s own order. **Real and imaginary parts**, columns 7
  and 8 of the printed row. The previous version of this docstring called them
  `magnitude_A, phase_deg`, and `test_gu5_8` believed it — which made every
  off-feed element read as exactly 180.0 deg, an artefact of a negative real
  taken for a magnitude. The coordinate column is normalized by the printout\'s
  own length unit, which for a buried segment is the IN-MEDIUM wavelength; it
  is carried for identification, never gated;
* `input_z` — the printed driving-point impedance;
* `witness_flag1` — the ORIGINAL flag-1 capture, `{"input_z", "currents"}`,
  kept so that a revert of the ground card fails a test naming the number
  rather than quietly republishing it.

Every deck is FULLY buried — no wire touches or crosses the interface — which
is why these are the arc\'s engine-referenced gate: they exercise the
below/below direct, image and remainder blocks and nothing else.

GENERATED by scripts/recapture_buried_goldens_nec5.py — do not edit by hand.
"""

DECKS = {
'''


def fmt_z(z):
    """`complex(re, im)` rather than the complex repr.

    repr gives `(417.89-420.55j)`; the formatter wants spaces around the
    operator, so a repr-emitted file is reformatted on every regeneration and
    the generator and the format gate fight forever. The call form has no
    operator to space, and it is what the original golden used.
    """
    return f"complex({z.real!r}, {z.imag!r})"


def fmt_rows(rows, indent):
    """Row lines at `indent` spaces. The witness rows nest one level deeper
    than the primary ones, and emitting both at the same depth is what made
    the generated file fail `ruff format --check` on every regeneration."""
    pad = " " * indent
    return "".join(f"{pad}({e}, {z!r}, {re_!r}, {im!r}),\n" for e, z, re_, im in rows)


def main():
    if not EXE:
        print("NEC5_EXE unset", file=sys.stderr)
        return 1
    body = []
    for name, d in DECKS.items():
        new_deck = with_card(d["deck"], DOCUMENTED_CARD)
        new_text = run(new_deck)
        old_text = run(with_card(d["deck"], ORIGINAL_CARD))
        old_z, old_cur = parse_z(old_text), parse_currents(old_text)
        want = d.get("witness_flag1", d)
        if old_z != want["input_z"] or old_cur != want["currents"]:
            print(
                f"PROVENANCE FAILED on {name}: the original card no longer "
                f"reproduces its banked capture",
                file=sys.stderr,
            )
            return 2
        # json.dumps rather than repr: repr emits SINGLE quotes, which the
        # repo's ruff config rejects (Q000), and a generated file that fails
        # lint on every regeneration is a generator bug, not a file to patch.
        body.append(
            f"    {json.dumps(name)}: {{\n"
            f'        "deck": {json.dumps(new_deck)},\n'
            f'        "input_z": {fmt_z(parse_z(new_text))},\n'
            f'        "currents": (\n{fmt_rows(parse_currents(new_text), 12)}        ),\n'
            f'        "witness_flag1": {{\n'
            f'            "input_z": {fmt_z(old_z)},\n'
            f'            "currents": (\n{fmt_rows(old_cur, 16)}            ),\n'
            f"        }},\n"
            f"    }},\n"
        )
    OUT.write_text(HEADER + "".join(body) + "}\n")
    print(f"wrote {OUT} — {len(body)} decks, each with its flag-1 witness")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
