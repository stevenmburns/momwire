"""Banked antennaknobs catalog geometries (momwire#988).

These decks gate momwire's OWN behaviour -- which solver path runs, whether a
fallback fires -- so they must not need antennaknobs installed. No momwire CI
lane installs it, so `importorskip("antennaknobs")` on such a test means it
runs nowhere.

THE BANKED FEEDS ARE NOT DRIVEN, and this has now cost two people a debugging
session in one day: antennaknobs hands the solver `(wire, s, 0j)` and resolves
the excitation through its OWN network layer (L-match, balanced line, ...)
after the solve, so a raw `compute_impedance()` on a captured cell returns NaN
with every current zero. `catalog_cell` drives the gap at 1 V for exactly that
reason -- see the note beside the feed rebuild below for what it does and does
not change.

The bank is `tests/fixtures/catalog_geometries.json`; regenerate it with
`scratch/988-study/bank_catalog_geometries.py`. Provenance (the antennaknobs
commit each cell was captured from) lives in the file.
"""

import json
import pathlib

import numpy as np

_PATH = pathlib.Path(__file__).parent / "fixtures" / "catalog_geometries.json"
_BLOB = None


def _blob():
    global _BLOB
    if _BLOB is None:
        _BLOB = json.loads(_PATH.read_text())
    return _BLOB


def provenance():
    return _blob()["_provenance"]


def available():
    """`{design}` present in the bank, either rung."""
    return sorted({k.split("|")[0] for k in _blob()["cells"]})


def _revive(v):
    if isinstance(v, dict) and "__c__" in v:
        return complex(*v["__c__"])
    if isinstance(v, list):
        return [_revive(x) for x in v]
    return v


def catalog_cell(design, rung="default", **overrides):
    """Solver kwargs for one banked design at one rung.

    `rung` is "default" (the catalog's nominal mesh x2) or "refined" (x4) --
    the two the ladders use. Overrides are merged last, so a test can vary
    `degree` or a tolerance without touching the geometry.
    """
    cells = _blob()["cells"]
    key = f"{design}|{rung}"
    if key not in cells:
        raise KeyError(
            f"{key} is not banked. Banked designs: {available()}. "
            "Add it with scratch/988-study/bank_catalog_geometries.py, or "
            "keep the test on the antennaknobs path and allowlist it in "
            "tests/test_no_hidden_antennaknobs_skips_988.py with a reason."
        )
    kw = {k: _revive(v) for k, v in cells[key].items()}
    kw["wires"] = [np.asarray(w, dtype=float) for w in kw["wires"]]
    # DRIVE THE GAP. antennaknobs hands the solver 0 V and resolves the feed
    # through its own network layer (L-match, balanced line, ...) after the
    # solve, so a banked feed is (wire, s, 0j) and `compute_impedance()` on it
    # returns NaN -- every current is zero. Driving at 1 V makes the fixture
    # self-contained: it measures the SOLVER, which is what these tests ask
    # about, rather than antennaknobs' network reduction. Both sides of any
    # accelerator-vs-dense comparison get the same excitation, so the
    # comparison is unaffected; the absolute Z is the raw solver's, not the
    # app's.
    kw["feeds"] = [
        (f[0], f[1], (1.0 + 0j) if len(f) > 2 and f[2] == 0 else f[2])
        for f in (tuple(x) for x in kw["feeds"])
    ]
    if kw.get("junctions"):
        kw["junctions"] = [[tuple(m) for m in j] for j in kw["junctions"]]
    if isinstance(kw.get("ground_eps"), list):
        kw["ground_eps"] = tuple(kw["ground_eps"])
    kw.update(overrides)
    return kw
