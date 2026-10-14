"""momwire's own NEC-5 printouts against the golden corpus (python-free arc, step 0).

`scripts/golden_printouts.py` froze, at one commit, the printouts the EZNEC
entry writes for a set of fixture decks on both bases. Each case here renders
its deck again, in the same regime (one BLAS/OpenMP thread, Sommerfeld disk
store off, the same deck order on one worker), and compares.

MEASUREMENT BUILD: strict byte equality, with every mismatching render
dropped under ``$MOMWIRE_403_DUMP/golden_printouts/`` so a CI run ships them.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

from golden_compare import moved_cells

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "golden_printouts", ROOT / "scripts" / "golden_printouts.py"
)
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

# One worker renders the whole corpus, in the generator's order.
pytestmark = pytest.mark.xdist_group("golden_printouts")


@pytest.fixture(scope="module", autouse=True)
def _elevated_sommerfeld_table():
    """The first render of an ELEVATED wire over a Sommerfeld ground builds a
    table that costs ~21 s on one core (the session runs with the disk store
    off); every later deck of that kind reuses it. Built here, once, so the
    cost is the module's setup rather than charged to whichever case runs
    first. Order does not move a printout (measured: the corpus rendered in
    reverse order, and single cases in fresh processes, are byte-identical to
    it), so this changes where the time is reported, not what is compared."""
    G.render("eznec/decks/0113_back-yard-dipole.nec", "bspline")


def _dump(deck: str, basis: str, text: str) -> None:
    root = os.environ.get("MOMWIRE_403_DUMP")
    if not root:
        return
    path = (
        Path(root)
        / "golden_printouts"
        / G.golden_path(deck, basis).relative_to(G.GOLDEN)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("latin-1"))


def test_the_manifest_is_the_corpus():
    """Every case has a golden file, the manifest names exactly those, and the
    decks are the decks it was rendered from."""
    manifest = json.loads(G.MANIFEST.read_text())
    named = [(e["deck"], e["basis"]) for e in manifest["entries"]]
    assert named == list(G.CASES)
    on_disk = sorted(
        p.relative_to(G.GOLDEN).as_posix() for p in G.GOLDEN.glob("*/*.out")
    )
    assert on_disk == sorted(e["golden"] for e in manifest["entries"])
    for e in manifest["entries"]:
        assert G._sha((G.FIXTURES / e["deck"]).read_bytes()) == e["deck_sha256"], e[
            "deck"
        ]
        golden = (G.GOLDEN / e["golden"]).read_bytes()
        assert G._sha(golden) == e["printout_sha256"], e["golden"]


@pytest.mark.parametrize(
    ("deck", "basis"), G.CASES, ids=[f"{d}[{b}]" for d, b in G.CASES]
)
def test_printout_matches_golden(deck, basis):
    golden = G.golden_path(deck, basis).read_bytes().decode("latin-1")
    text = G.render(deck, basis)
    if text == golden:
        return
    _dump(deck, basis, text)
    moves = moved_cells(golden, text)
    listing = "\n".join(str(m) for m in moves[:40])
    pytest.fail(f"{len(moves)} cells moved:\n{listing}")
