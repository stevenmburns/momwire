"""U9 Amendment 8c: one deck set's 2x2 Y and Z from one engine, as a JSONL row.

  PYTHONPATH=<momwire src>:<antennaknobs src> python e4_knot_solve.py \\
      --stem two_node_d3_r1 --engine momwire [--corner same] --out F.jsonl
  PYTHONPATH=<momwire src>:<antennaknobs src> python e4_knot_solve.py \\
      --stem two_node_d3_r1 --engine nec5 --nec5-exe PATH --out F.jsonl

momwire: the kwargs antennaknobs' momwire engine builds from `<stem>_a.nec` (the
knot feeds; `e3_knot_decks.solver_kwargs`), then
`BSplineSolver(**kw).compute_port_solution().y`. Y does not depend on the card
voltages. `--corner same` returns a zero corner potential for every cross-node
pair (`_crossing_fill._corner_v` at rho >= `_SAME_NODE_RHO`). Every corner loop
calls it, forward and reversed alike. The patch counts the calls, and the pairs
it zeroed, per calling loop.

NEC-5: native runs of `<stem>_a.nec` and `<stem>_b.nec`. Each run drives both
ports, at card voltages (1, 0.5) and (0.5, 1). A port's current is the source
current NEC-5 prints for that port's EX card in the run's input-parameters
block, matched by tag: Y = [I_a I_b] [V_a V_b]^-1, with V taken from the cards.
`<stem>_a0.nec` (I4 = 0) runs too where it exists. Every input-parameters row is
kept with its printed tokens, for checks 2 and 3. Printout output only (the
licence courtesy rule).

The row carries Y, Z = inv(Y) and the reciprocity. Stdout gets a status line
only, so no Z is seen before the checks are read.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import resource
import time
import warnings
from pathlib import Path

import numpy as np

from e3_knot_decks import DECK_DIR, point_at, sha, solver_kwargs

AIP_HEADER = "ANTENNA INPUT PARAMETERS"
STEM = re.compile(r"^(two_node|ctrl)_d(\d+)_(r1|far3|all3)$")


def ex_cards(text):
    """[tag, seg, end, [Vre, Vim]] per EX card, in card order."""
    out = []
    for line in text.splitlines():
        f = line.split()
        if f and f[0] == "EX":
            out.append([int(f[2]), int(f[3]), int(f[4]), [float(f[5]), float(f[6])]])
    return out


def input_parameters(text):
    """The rows of the printout's one input-parameters block, with tokens kept
    as printed. The row layout is the one antennaknobs' parser pins:
    tag seg sub Vre Vim Ire Iim Zre Zim Yre Yim P."""
    chunks = text.split(AIP_HEADER)[1:]
    if len(chunks) != 1:
        raise RuntimeError(f"{len(chunks)} input-parameters blocks, expected 1")
    rows = []
    for line in chunks[0].splitlines():
        toks = line.split()
        if len(toks) != 12:
            if rows:
                break
            continue
        try:
            tag, seg = int(toks[0]), int(toks[1])
            vals = [float(t) for t in toks[3:12]]
        except ValueError:
            if rows:
                break
            continue
        rows.append(
            dict(
                tag=tag,
                seg=seg,
                sub=toks[2],
                tokens=toks[3:12],
                v=vals[0:2],
                i=vals[2:4],
                z=vals[4:6],
            )
        )
    if not rows:
        raise RuntimeError("no input-parameters rows")
    return rows


def momwire_y(stem, corner, info):
    import momwire
    from momwire import _crossing_fill
    from momwire.bspline import BSplineSolver

    path = DECK_DIR / f"{stem}_a.nec"
    info["deck_sha256"] = sha(path)
    _parsed, kw = solver_kwargs(path)
    feeds = list(kw.get("feeds") or [])
    info.update(
        momwire=str(Path(momwire.__file__).resolve().parent),
        feeds=[[int(f[0]), float(f[1])] for f in feeds],
        feed_points=[
            point_at(kw["wires"][int(f[0])], float(f[1])).tolist() for f in feeds
        ],
    )
    by_loop = {}
    if corner == "same":
        real_corner_v = _crossing_fill._corner_v

        def same_node_only(cache, eps_t, k_p, a_wire, rho):
            loop = inspect.currentframe().f_back.f_code.co_name
            n = by_loop.setdefault(loop, dict(calls=0, zeroed=0))
            n["calls"] += 1
            if rho >= _crossing_fill._SAME_NODE_RHO:
                n["zeroed"] += 1
                return 0j
            return real_corner_v(cache, eps_t, k_p, a_wire, rho)

        _crossing_fill._corner_v = same_node_only
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = np.asarray(BSplineSolver(**kw).compute_port_solution().y, dtype=complex)
    info["corner_calls_by_loop"] = by_loop
    info["cross_node_pairs_dropped"] = sum(n["zeroed"] for n in by_loop.values())
    return y


def nec5_y(stem, exe, info):
    from antennaknobs.engines.nec5 import run_deck

    runs = info.setdefault("runs", {})
    info["nec5_exe"] = exe
    for exc in ("a", "b", "a0"):
        path = DECK_DIR / f"{stem}_{exc}.nec"
        if not path.exists():
            continue
        text = path.read_text()
        runs[exc] = dict(deck_sha256=sha(path), ex=ex_cards(text))
        runs[exc]["input_parameters"] = input_parameters(
            run_deck(exe, text, timeout=3600)
        )
    tags = [c[0] for c in runs["a"]["ex"]]
    if len(tags) != 2 or [c[0] for c in runs["b"]["ex"]] != tags:
        raise RuntimeError(f"runs a and b do not drive the same two tags: {tags}")
    cur = np.zeros((2, 2), dtype=complex)
    volt = np.zeros((2, 2), dtype=complex)
    for j, exc in enumerate(("a", "b")):
        for p, (tag, _seg, _end, v) in enumerate(runs[exc]["ex"]):
            match = [r for r in runs[exc]["input_parameters"] if r["tag"] == tag]
            if len(match) != 1:
                raise RuntimeError(
                    f"run {exc}: {len(match)} input-parameters rows for tag {tag}"
                )
            cur[p, j] = complex(*match[0]["i"])
            volt[p, j] = complex(*v)
    info["port_tags"] = tags
    return cur @ np.linalg.inv(volt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--engine", choices=("momwire", "nec5"), required=True)
    ap.add_argument("--corner", choices=("cross", "same"), default="cross")
    ap.add_argument("--nec5-exe", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    m = STEM.match(args.stem)
    if m is None:
        ap.error(f"unknown stem {args.stem!r}")
    if args.engine == "nec5" and args.corner != "cross":
        ap.error("--corner same is a momwire patch")
    rec = dict(
        stem=args.stem,
        control=m[1] == "ctrl",
        d=float(m[2]),
        rung=m[3],
        engine=args.engine,
        corner=args.corner,
        t_start=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    info = {}
    t0 = time.perf_counter()
    try:
        if args.engine == "momwire":
            y = momwire_y(args.stem, args.corner, info)
        else:
            y = nec5_y(args.stem, args.nec5_exe, info)
        z = np.linalg.inv(y)
        rec["y"] = [[v.real, v.imag] for v in y.ravel()]
        rec["z"] = [[v.real, v.imag] for v in z.ravel()]
        rec["y_reciprocity"] = float(abs(y[0, 1] - y[1, 0]) / abs(y[0, 1]))
    except Exception as err:  # noqa: BLE001 - a record, not a handler
        rec["error"] = f"{type(err).__name__}: {err!s:.400}"
    rec.update(info)
    rec["seconds"] = round(time.perf_counter() - t0, 2)
    rec["maxrss_mb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
    )
    with args.out.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    status = rec.get("error", "ok")
    print(args.engine, args.stem, args.corner, f"{rec['seconds']} s", status)


if __name__ == "__main__":
    main()
