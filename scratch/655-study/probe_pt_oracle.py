"""momwire#655: what the oracle prints for each PT form, measured here.

The issue records a table; this reproduces it on this box before anything is
changed, so the fix is gated on observed behaviour rather than on a
transcription. 5-segment dipole, one PT card per run, `CURRENTS AND LOCATION`
row numbers collected.
"""

import re
import subprocess
import tempfile
from pathlib import Path

ORACLE = "/home/smburns/.SimNEC/5/1/Examples/nec2c.ae6ty/bin/nec2c-ubuntu-x86"

DECK = (
    "CM pt probe\nCE\n"
    "GW 1 5 0. 0. -2.5 0. 0. 2.5 0.001\n"
    "GE 0\n"
    "EX 0 1 3 0 1. 0.\n"
    "FR 0 1 0 0 30. 0.\n"
    "{pt}"
    "XQ 0\n"
    "EN\n"
)


def rows(pt_card):
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "m.nec"
        d.write_text(DECK.format(pt=pt_card))
        out = Path(td) / "m.out"
        subprocess.run(
            [ORACLE, "-i", str(d), "-o", str(out)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        text = out.read_text(errors="replace") if out.is_file() else ""
    m = re.search(r"CURRENTS AND LOCATION(.*?)(?:\n\s*\n\s*\n|\Z)", text, re.S)
    if not m:
        return None  # the whole section is gone
    got = []
    for line in m.group(1).splitlines():
        t = line.split()
        if len(t) >= 9 and re.fullmatch(r"\d+", t[0]) and re.fullmatch(r"\d+", t[1]):
            got.append(t[1])
    return got


CARDS = [
    ("(no PT)", ""),
    ("PT 0 1 2 4", "PT 0 1 2 4\n"),
    ("PT 1 0 0 0", "PT 1 0 0 0\n"),
    ("PT 3 0 0 0", "PT 3 0 0 0\n"),
    ("PT 1 1 2 4", "PT 1 1 2 4\n"),
    ("PT 2 1 2 4", "PT 2 1 2 4\n"),
    ("PT 3 1 2 4", "PT 3 1 2 4\n"),
    ("PT 1 0 2 4", "PT 1 0 2 4\n"),
]

print(f"oracle: {ORACLE}")
for label, card in CARDS:
    r = rows(card)
    print(f"  {label:14s} -> {'(section absent)' if r is None else (r or '(none)')}")
