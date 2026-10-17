"""momwire#1517 step 1: the C++ NEC-5 seam for bspline
(`_nec5_bspline_seam_core.cpp`) against the Python seam, deck by deck, up to
the solver's construction.

    python scripts/core_split/bspline_seam_compare.py OUT.jsonl DECK_OR_DIR...

The Python seam runs `_shell.render(text, basis="bspline")` with
`_serve._prepare` recorded and `_serve._port_solution` stubbed out (the fill
is not what this compares, and the front half is not ported), so the record
is each deck's first preparation: BSplineSolver's keyword arguments, the
mesh, the structure, medium and sites; or the refusal the printout carries.
The C++ seam's preparations decline "front:bspline-front" and are kept: the
first one must be the Python one, field for field (`front_compare.differences`:
bytes, dtypes, shapes and Python types). A deck the C++ seam refuses must be
refused with Python's sentence. Exit 1 on any mismatch.
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import front_compare as fc
from seam_compare import _decks, python_mesh


class _Stop(Exception):
    pass


def python_seam(text):
    from momwire import BSplineSolver
    from momwire.eznec import _printout, _serve, _shell

    rec = {"prepared": [], "refusals": []}
    current = {}
    orig_prepare = _serve._prepare
    orig_init = BSplineSolver.__init__
    orig_port = _serve._port_solution
    orig_refusal = _printout.render_refusal

    def prepare(deck, *, basis=_serve.BASIS, shared=None):
        current.clear()
        p = orig_prepare(deck, basis=basis, shared=shared)
        rec["prepared"].append(dict(current, prepared=p))
        raise _Stop

    def init(self, **kw):
        current["kwargs"] = dict(kw)
        return orig_init(self, **kw)

    def refusal(text, reason, **kw):
        rec["refusals"].append(reason)
        return orig_refusal(text, reason, **kw)

    _serve._prepare = prepare
    BSplineSolver.__init__ = init
    _serve._port_solution = lambda solver, shared: None
    _printout.render_refusal = refusal
    try:
        _shell.render(text, basis="bspline")
        rec["outcome"] = "refused" if rec["refusals"] else "served"
    except Exception as exc:  # noqa: BLE001 -- the escape IS the answer here
        rec["outcome"] = "raised"
        rec["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _serve._prepare = orig_prepare
        BSplineSolver.__init__ = orig_init
        _serve._port_solution = orig_port
        _printout.render_refusal = orig_refusal
    # `_shell.render` writes the INTERNAL ERROR frame for the stop: a
    # preparation recorded is the answer, whatever the frame said
    if rec["prepared"]:
        rec["outcome"] = "prepared"
    elif rec["refusals"]:
        rec["outcome"] = "refused"
        rec["message"] = rec["refusals"][-1]
    return rec


def compare(text, acc):
    py = python_seam(text)
    cx = acc.nec5_bspline_seam(text.encode("latin-1"))
    first = cx["structures"][0][0] if cx["structures"] and cx["structures"][0] else None
    row = {"route": cx["route"], "python": py["outcome"]}
    bad = []
    if py["outcome"] == "prepared":
        if not cx["prepared"]:
            row["reason"] = cx["message"]
            if cx["route"] != "declined":
                bad.append(
                    f"C++ {cx['route']} ({cx['message']!r}) where Python constructed"
                )
            return row, bad
        p, c = py["prepared"][0], cx["prepared"][0]
        if "kwargs" not in p:
            return row, ["Python constructed nothing"]
        d = fc.differences(p["kwargs"], c["kwargs"], "kwargs")
        mesh = python_mesh(p["prepared"])
        cmesh = dict(c["mesh"])
        cmesh["source_sites"] = sorted(cmesh["source_sites"])
        cmesh["phantom_tags"] = sorted(cmesh["phantom_tags"])
        d += fc.differences(mesh, cmesh, "mesh")
        bad += d
        row["compared"] = True
    elif py["outcome"] == "refused":
        if first is not None and first["kind"] == "refused":
            if first["message"] != py["message"]:
                bad.append(
                    f"refusal: C++ {first['message']!r} vs Python {py['message']!r}"
                )
        elif cx["route"] != "declined":
            bad.append(
                f"C++ {cx['route']} where Python refused: {py['message']!r}"[:300]
            )
        row["reason"] = cx["message"]
    else:
        row["reason"] = cx["message"]
        row["python_error"] = py.get("error", "")[:200]
    return row, bad


def main():
    out_path = sys.argv[1]
    items = [(str(p), p.read_bytes().decode("latin-1")) for p in _decks(sys.argv[2:])]
    from momwire import _accel

    acc = _accel.acc
    if acc is None or not hasattr(acc, "nec5_bspline_seam"):
        sys.exit("no accelerator with the bspline seam hook loaded")
    warnings.simplefilter("ignore")
    routes, reasons = Counter(), Counter()
    compared = mismatches = 0
    with open(out_path, "w") as out:
        for name, text in items:
            row, bad = compare(text, acc)
            row["deck"] = name
            routes[f"{row['route']}|py:{row['python']}"] += 1
            if "reason" in row and row["reason"]:
                reasons[row["reason"]] += 1
            compared += bool(row.get("compared"))
            if bad:
                mismatches += 1
                row["mismatch"] = bad[:20]
                print(f"MISMATCH {name}: {bad[:3]}")
            out.write(json.dumps(row) + "\n")
    print("routes:", dict(routes))
    print("reasons:", dict(reasons.most_common(12)))
    print(f"preparations compared: {compared}, mismatches {mismatches}")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
