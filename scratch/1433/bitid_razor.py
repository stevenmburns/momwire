"""PR B gate (momwire#1433): razor-2p bit-identical where the exact kernel
does not engage.

Mode "base" runs a tree without the option (origin/main); "off" passes
exact_kernel=False and "auto" passes exact_kernel="auto" to every
RazorSolver the run builds (injected at construction, the way
antennaknobs' run_mw.py does). Each record carries the printout's / Z's
sha256, the resolved `exact_kernel`, and how many times
`_add_exact_kernel_correction` ran -- instrumented, so "did not engage" is
measured, not inferred. Diff base against off and against auto.

Usage: python -I bitid_razor.py MODE TESTS_DIR [EXTRA_DECK_DIR ...] > out.jsonl
"""

import hashlib
import json
import pathlib
import sys
import warnings

import numpy as np

from momwire.eznec import _shell
from momwire.razor import RazorSolver

MODE = sys.argv[1]
TESTS = pathlib.Path(sys.argv[2])
EXTRA = [pathlib.Path(p) for p in sys.argv[3:]]
sys.path.insert(0, str(TESTS))
import _catalog_fixture as cat  # noqa: E402

BUILT = []
CALLS = {"corr": 0}
_init = RazorSolver.__init__


def _wrapped(self, *a, **k):
    if MODE != "base":
        k["exact_kernel"] = False if MODE == "off" else "auto"
    _init(self, *a, **k)
    BUILT.append(self)


RazorSolver.__init__ = _wrapped
if hasattr(RazorSolver, "_add_exact_kernel_correction"):
    _corr = RazorSolver._add_exact_kernel_correction

    def _counted(self, *a, **k):
        CALLS["corr"] += 1
        return _corr(self, *a, **k)

    RazorSolver._add_exact_kernel_correction = _counted


def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


def resolved():
    return sorted({repr(getattr(s, "exact_kernel", None)) for s in BUILT})


def deck_records():
    decks = sorted((TESTS / "fixtures").rglob("*.nec"))
    for d in EXTRA:
        decks += sorted(d.glob("*.nec"))
    for path in decks:
        BUILT.clear()
        CALLS["corr"] = 0
        rec = {"deck": str(path.relative_to(path.parents[1]))}
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out = _shell.render(path.read_text(errors="replace"), basis="razor-2p")
            rec["printout"] = sha(out.encode())
            rec["warn"] = sorted({type(x.message).__name__ for x in w})
        except Exception as exc:  # noqa: BLE001 -- refusals are data
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        rec["solvers"] = len(BUILT)
        if MODE != "base":
            rec["resolved"] = resolved()
            rec["corr_calls"] = CALLS["corr"]
        yield rec


_RAZOR_KW = {
    "wires", "n_per_edge_per_wire", "junctions", "node_gaps", "nsegs",
    "wire_radius", "extended_kernel", "wire_conductivity", "insulation_radius",
    "insulation_eps_r", "distributed_rlc", "lumped_loads", "ground_z",
    "ground_eps", "ground_phi_mode", "ground_model", "wavelength",
    "halfdriver_factor", "feed_wire_index", "feed_arclength", "feeds",
}  # fmt: skip


def catalog_records():
    for d in cat.available():
        try:
            kw = cat.catalog_cell(d)
        except KeyError:
            continue
        kw = {k: v for k, v in kw.items() if k in _RAZOR_KW}
        BUILT.clear()
        CALLS["corr"] = 0
        rec = {"catalog": d}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = RazorSolver(nec5_quadrature=True, **kw)
                z, coeffs = s.compute_impedance()
            rec["Z"] = sha(
                np.concatenate([np.ravel(z), np.ravel(coeffs)])
                .astype(complex)
                .tobytes()
            )
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        if MODE != "base":
            rec["resolved"] = resolved()
            rec["corr_calls"] = CALLS["corr"]
        yield rec


for r in deck_records():
    print(json.dumps(r), flush=True)
for r in catalog_records():
    print(json.dumps(r), flush=True)
