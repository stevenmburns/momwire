"""momwire's own NEC-5 printouts against the golden corpus (python-free arc, step 0).

`scripts/golden_printouts.py` froze, at one commit, the printouts the EZNEC
entry writes for a set of fixture decks on both bases. Each case here renders
its deck again in the same regime (one BLAS/OpenMP thread, Sommerfeld disk
store off) and compares. Two claims, and which one is asserted depends on
where the test runs:

**On the machine and stack that wrote the corpus** (CPU model, lane, numpy,
scipy and the BLAS kernel all as the manifest records them) the printout must
be BYTE-identical, every cell, the undefined ones included. That is the gate
a hardening step runs on its own box, base against change.

**Anywhere else** it must be byte-identical after `undefined_blanked` on both
sides: momwire#578's boundary (``eznec_reproducibility``) plus the network
V/I crumb rule in ``golden_compare``. Measured before choosing this, against
the corpus generated on Haswell (i7-4770K, OpenBLAS Haswell kernel): on a
GitHub Linux runner 23 of 108 printouts moved, on the macOS arm64 runner 24,
and on both every moved cell was one of those undefined cells (network-row
cancellation crumbs between 1e-27 and 1e-7 beside a 1e10 ohm pin, and the
gyrator decks' 1e-17 current crumbs). No printed reading moved on either. So
nothing here is a numeric tolerance: a cell is either compared exactly or is
one the arithmetic does not define. Windows (MSVC /fp:fast) has never been
measured and no CI lane runs it; it needs its own measurement before this
claim is extended there.

Every mismatching render is written under ``$MOMWIRE_403_DUMP/golden_printouts``
(CI ships that directory as an artifact on failure), and the failure lists
each moved cell in units of its last printed digit.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

from golden_compare import moved_cells, undefined_blanked

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "golden_printouts", ROOT / "scripts" / "golden_printouts.py"
)
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

MANIFEST = json.loads(G.MANIFEST.read_text())

# One worker renders the whole corpus, so the table the module fixture builds
# is built once.
pytestmark = pytest.mark.xdist_group("golden_printouts")

# What has to match for the strict, every-cell claim. The thread counts are
# not in it: `render` pins one thread whatever the process was started with.
_STACK = ("system", "machine", "cpu", "accelerator_variant", "numpy", "scipy")


def _stack(regime: dict) -> tuple:
    blas = sorted(
        (p.get("internal_api"), p.get("version"), p.get("architecture"))
        for p in regime["threadpools_during_render"]
        if p.get("user_api") == "blas"
    )
    return (*(regime.get(k) for k in _STACK), tuple(blas))


@pytest.fixture(scope="module")
def same_stack() -> bool:
    return _stack(G.regime()) == _stack(MANIFEST["regime"])


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
    named = [(e["deck"], e["basis"]) for e in MANIFEST["entries"]]
    assert named == list(G.CASES)
    on_disk = sorted(
        p.relative_to(G.GOLDEN).as_posix() for p in G.GOLDEN.glob("*/*.out")
    )
    assert on_disk == sorted(e["golden"] for e in MANIFEST["entries"])
    for e in MANIFEST["entries"]:
        deck = (G.FIXTURES / e["deck"]).read_bytes()
        assert G._sha(deck) == e["deck_sha256"], e["deck"]
        golden = (G.GOLDEN / e["golden"]).read_bytes()
        assert G._sha(golden) == e["printout_sha256"], e["golden"]


def test_the_blanking_touches_network_rows_only():
    """`undefined_blanked` narrows the cross-machine claim, so it must not
    reach past what it was measured on: over the whole corpus every line it
    changes is a row of a network-connection table (momwire#578's rule 4, the
    wire-loss crumb, and rule 3, wire-table dust, never fire on this corpus;
    if one starts to, this names the line so the claim is re-measured)."""
    for deck, basis in G.CASES:
        golden = G.golden_path(deck, basis).read_bytes().decode("latin-1")
        lines = golden.split("\n")
        blanked = undefined_blanked(golden).split("\n")
        assert len(blanked) == len(lines)
        table = False
        for before, after in zip(lines, blanked):
            if before.strip().startswith("- - -"):
                table = "NETWORK CONNECTION POINTS" in before
            if before != after:
                assert table, (deck, basis, before)


@pytest.mark.parametrize(
    ("deck", "basis"), G.CASES, ids=[f"{d}[{b}]" for d, b in G.CASES]
)
def test_printout_matches_golden(deck, basis, same_stack):
    golden = G.golden_path(deck, basis).read_bytes().decode("latin-1")
    text = G.render(deck, basis)
    if text == golden:
        return
    compared = text
    if not same_stack:
        golden, compared = undefined_blanked(golden), undefined_blanked(text)
        if compared == golden:
            return
    _dump(deck, basis, text)
    moves = moved_cells(golden, compared)
    listing = "\n".join(str(m) for m in moves[:40])
    claim = "every cell (same stack as the manifest)" if same_stack else "defined cells"
    pytest.fail(f"{len(moves)} cells moved, comparing {claim}:\n{listing}")
