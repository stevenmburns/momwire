"""The C++ NEC-5 seam for razor-2p (`_nec5_razor_seam_core.cpp`, unit 2b)
against the Python seam, deck by deck.

    python scripts/core_split/seam_compare.py OUT.jsonl DECK_OR_DIR...
    python scripts/core_split/seam_compare.py OUT.jsonl --texts HARVEST.jsonl

(`--texts` reads deck texts as JSON strings, one per line: what
`nec5_harvest_plugin.py` records from the test suite.)

Each deck (`.nec` / `.deck`, read latin-1 as the seam reads it) goes through

1. the PYTHON seam, as `momwire-nec5-razor-2p` serves it
   (`_shell.render(text, basis="razor-2p")`, the Python route forced), with
   `_serve._prepare` and `RazorSolver.__init__` / `compute_port_solution`
   recorded: every preparation's structure, mesh, medium and source sites,
   the keyword arguments the seam hands `RazorSolver`, the solved
   coefficients X and Y; and the refusal the printout carries
   (`_printout.render_refusal`'s reason) or the exception that escapes;
2. the C++ seam, deck text to solved coefficients with no Python
   (`_accelerators.nec5_razor_seam(..., solve=True)`).

and is counted by the C++ seam's route:

* ``served`` / ``refused`` / ``raised`` -- the C++ seam answered the whole
  deck; it must be the Python seam's answer exactly: the same refusal
  sentence or exception (type and message), the same number of
  preparations, and for each the same keyword arguments (bytes, dtypes,
  shapes and Python types, `front_compare.differences`), the same mesh,
  structure counts, medium and sites, and the same X and Y bytes;
* ``declined`` -- the C++ seam stopped at something it does not cover yet,
  by name; counted per reason, never as an agreement. What it prepared before
  the decline must still agree with the Python seam's first preparations.

One JSON line per deck; a summary on stdout; exit status 1 on any mismatch.
Threads are whatever OMP_NUM_THREADS / OPENBLAS_NUM_THREADS say.
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import Counter
from pathlib import Path


def _decks(args):
    out = []
    for arg in args:
        p = Path(arg)
        if p.is_dir():
            out += sorted(
                q for q in p.rglob("*") if q.is_file() and q.suffix in (".nec", ".deck")
            )
        else:
            out.append(p)
    return out


def python_seam(text):
    """The Python seam's answer, recorded: ``dict(outcome, message,
    error_kind, prepared=[...])``."""
    import numpy as np

    from momwire import RazorSolver, razor
    from momwire.eznec import _printout, _serve, _shell

    rec = {"prepared": [], "refusals": []}
    current = {}
    orig_prepare = _serve._prepare
    orig_init = RazorSolver.__init__
    orig_cps = RazorSolver.compute_port_solution
    orig_refusal = _printout.render_refusal

    def prepare(deck, *, basis=_serve.BASIS, shared=None):
        current.clear()
        p = orig_prepare(deck, basis=basis, shared=shared)
        rec["prepared"].append(dict(current, prepared=p))
        return p

    def init(self, **kw):
        current["kwargs"] = dict(kw)
        return orig_init(self, **kw)

    def cps(self, prepared=None):
        sol = orig_cps(self, prepared)
        current["X"] = np.array(sol.coeffs, copy=True)
        current["Y"] = np.array(sol.y, copy=True)
        return sol

    def refusal(text, reason, **kw):
        rec["refusals"].append(reason)
        return orig_refusal(text, reason, **kw)

    saved = razor._RAZOR_CXX
    razor._RAZOR_CXX = False
    _serve._prepare = prepare
    RazorSolver.__init__ = init
    RazorSolver.compute_port_solution = cps
    _printout.render_refusal = refusal
    try:
        _shell.render(text, basis="razor-2p")
        rec["outcome"] = "refused" if rec["refusals"] else "served"
        rec["message"] = rec["refusals"][-1] if rec["refusals"] else ""
        rec["error_kind"] = ""
    except Exception as exc:  # noqa: BLE001 -- the escape IS the answer here
        rec["outcome"] = "raised"
        rec["message"] = str(exc)
        rec["error_kind"] = type(exc).__name__
        # Raised while WRITING a printout (`eznec/_printout.py`), after the
        # solves it reports: the run half's (unit 3), not this seam's.
        tb = exc.__traceback__
        while tb is not None:
            if tb.tb_frame.f_code.co_filename.replace("\\", "/").endswith(
                "eznec/_printout.py"
            ):
                rec["outcome"] = "printout-raised"
                break
            tb = tb.tb_next
    finally:
        razor._RAZOR_CXX = saved
        _serve._prepare = orig_prepare
        RazorSolver.__init__ = orig_init
        RazorSolver.compute_port_solution = orig_cps
        _printout.render_refusal = orig_refusal
    return rec


def _phantom_tags_of(structure, mesh):
    tags, k = [], 0
    for wire in structure.wires:
        n = wire.segment_count
        if n and all(mesh.element_of[k + j][0] == -1 for j in range(n)):
            tags.append(wire.tag)
        k += n
    return sorted(tags)


def python_mesh(prepared):
    """`_prepare`'s structure, mesh, medium and sites in the C++ binding's
    layout (`seam_mesh`)."""
    mesh, structure = prepared.mesh, prepared.structure
    sites = []
    for s in mesh.sites:
        sites.append(
            {
                "at": (s.at.tag, s.at.node),
                "spelling": s.spelling,
                "piece": s.piece,
                "end": s.end,
                "sign": s.sign,
                "arclength": s.arclength,
                "contact": s.contact,
                "crossing": s.crossing,
                "index": s.index,
                "column": s.column,
                "weight": s.weight,
            }
        )
    medium = prepared.medium
    return {
        "pieces": [
            (
                p.tag,
                p.first_node,
                p.last_node,
                p.radius,
                p.cut_start,
                p.cut_end,
                # every node of a bent piece (momwire#1669), else its two ends
                *(tuple(float(v) for v in q) for q in p.points),
            )
            for p in mesh.pieces
        ],
        "junctions": [list(g) for g in mesh.junctions],
        "sites": sites,
        "feeds": [s.index for s in mesh.feeds],
        "gaps": [s.index for s in mesh.gaps],
        "element_of": [tuple(e) for e in mesh.element_of],
        "split_elements": {
            k: (v.lower, v.upper, v.centre_piece, v.centre_s)
            for k, v in mesh.split_elements.items()
        },
        "node_count": structure.node_count,
        "wire_element_count": structure.wire_element_count,
        "unknown_count": structure.unknown_count,
        "ground_plane": structure.ground_plane,
        "frequency": prepared.frequency,
        "wavelength": prepared.wavelength,
        "omega": prepared.omega,
        "medium": None
        if medium is None
        else (medium.eps_r, medium.sigma, complex(medium.eps_c)),
        "source_sites": sorted((n.tag, n.node) for n in prepared.source_sites),
        # `_cards`: kind, the six fields, the two sites, the NETWORK DATA
        # row's addresses
        "cards": [
            (
                c.card.kind,
                tuple(float(v) for v in c.card.payload),
                c.site_a,
                c.site_b,
                c.row.tag_from,
                c.row.segment_from,
                c.row.tag_to,
                c.row.segment_to,
            )
            for c in prepared.cards
        ],
        # a phantom GW has no piece: its elements map to piece -1
        "phantom_tags": _phantom_tags_of(structure, mesh),
    }


# The executable is Linux-only (momwire#1512), and its gates are bit for bit
# there. Elsewhere the core and the extension may round the same solve
# differently: on Windows MSVC builds /fp:fast (no bit-compatibility is
# asked there), and on macOS arm64 a Sommerfeld deck's solve differs in the
# last bits (numpy's complex scalar functions and BLAS order are not the
# libm and the order the core reproduces on x86 Linux). Off Linux a
# solution is the same within the derived lane tolerance
# (tests/_lane_gate.py, LANE_RTOL_WIN32, momwire#1371).
_OFF_LINUX_RTOL = 1e-11


def _rel_diff(a, b) -> float:
    import numpy as np

    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    if a.shape != b.shape:
        return float("inf")
    scale = float(np.max(np.abs(b))) if b.size else 0.0
    d = float(np.max(np.abs(a - b), initial=0.0))
    return 0.0 if d == 0.0 else (d / scale if scale > 0.0 else float("inf"))


def _same_solution(a, b) -> bool:
    import numpy as np

    if sys.platform.startswith("linux"):
        return np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()
    return _rel_diff(a, b) <= _OFF_LINUX_RTOL


def compare(text, acc, fc):
    """One deck: ``(row, mismatches)``."""
    py = python_seam(text)
    cx = acc.nec5_razor_seam(text.encode("latin-1"), solve=True)
    row = {"route": cx["route"]}
    bad = []
    if cx["route"] == "declined":
        row["reason"] = cx["message"]
    elif py["outcome"] == "printout-raised":
        # The Python seam answered every structure it reached and then raised
        # writing a printout (the run half, unit 3): counted on its own, and
        # the preparations it made before raising must still agree.
        row["route"] = f"{cx['route']}|python-printout-raised"
        row["python_raised"] = f"{py['error_kind']}: {py['message']}"[:200]
        if cx["route"] == "raised" or len(cx["prepared"]) < len(py["prepared"]):
            bad.append(f"outcome: C++ {cx['route']} vs Python {py['outcome']}")
    else:
        if cx["route"] != py["outcome"]:
            bad.append(f"outcome: C++ {cx['route']} vs Python {py['outcome']}")
        elif cx["message"] != py["message"] or cx["error_kind"] != py["error_kind"]:
            bad.append(
                f"message: C++ {cx['error_kind']}:{cx['message']!r} vs "
                f"Python {py['error_kind']}:{py['message']!r}"
            )
        if len(cx["prepared"]) != len(py["prepared"]):
            bad.append(
                f"preparations: C++ {len(cx['prepared'])} vs Python {len(py['prepared'])}"
            )
    row["prepared"] = len(cx["prepared"])
    for k, (c, p) in enumerate(zip(cx["prepared"], py["prepared"])):
        if "kwargs" not in p:
            bad.append(f"prepared[{k}]: Python constructed nothing")
            continue
        d = fc.differences(p["kwargs"], c["kwargs"], f"prepared[{k}].kwargs")
        mesh = python_mesh(p["prepared"])
        cmesh = dict(c["mesh"])
        cmesh["source_sites"] = sorted(cmesh["source_sites"])
        cmesh["phantom_tags"] = sorted(cmesh["phantom_tags"])
        d += fc.differences(mesh, cmesh, f"prepared[{k}].mesh")
        if "X" in p and "X" in c:
            y = c["cols"].T @ c["X"]
            x_same = _same_solution(p["X"], c["X"])
            y_same = _same_solution(p["Y"], y)
            if not x_same:
                d.append(f"prepared[{k}]: X differs ({_rel_diff(c['X'], p['X']):.2e})")
            if not y_same:
                d.append(f"prepared[{k}]: Y differs ({_rel_diff(y, p['Y']):.2e})")
            row.setdefault("solves", 0)
            row["solves"] += 1
        elif c.get("info") is not None and cx["route"] != "declined":
            d.append(f"prepared[{k}]: Python solved nothing")
        bad += d
    return row, bad


def main():
    out_path = sys.argv[1]
    if sys.argv[2:3] == ["--texts"]:
        lines = Path(sys.argv[3]).read_text().splitlines()
        items = [
            (f"{sys.argv[3]}:{k}", json.loads(line)) for k, line in enumerate(lines)
        ]
    else:
        items = [
            (str(p), p.read_bytes().decode("latin-1")) for p in _decks(sys.argv[2:])
        ]
    from momwire import _accel

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import front_compare as fc

    acc = _accel.acc
    if acc is None or not hasattr(acc, "nec5_razor_seam"):
        sys.exit("no accelerator with the C++ seam loaded")
    warnings.simplefilter("ignore")
    routes, reasons = Counter(), Counter()
    solves = mismatches = 0
    with open(out_path, "w") as out:
        for name, text in items:
            try:
                text.encode("latin-1")
            except UnicodeEncodeError:
                routes["not-latin-1"] += 1  # a deck file cannot carry it
                continue
            row, bad = compare(text, acc, fc)
            row["deck"] = name
            if bad:
                row["mismatch"] = bad[:8]
                mismatches += 1
            routes[row["route"]] += 1
            solves += row.get("solves", 0)
            if row["route"] == "declined":
                reasons[row["reason"]] += 1
            out.write(json.dumps(row) + "\n")
            out.flush()
    print(
        f"decks {len(items)}: "
        + ", ".join(f"{k} {v}" for k, v in sorted(routes.items()))
    )
    print(f"solves compared (X, Y): {solves}")
    print(
        "declined by reason: " + ", ".join(f"{k} {v}" for k, v in reasons.most_common())
    )
    print(f"mismatches {mismatches}")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
